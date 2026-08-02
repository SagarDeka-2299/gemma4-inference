#!/usr/bin/env python
"""Dequantize the base checkpoint to uniform dense bf16 -- NO adapter folded in.

Why: unsloth/gemma-4-e4b-it-unsloth-bnb-4bit is mixed precision on disk (most
linear layers packed 4-bit, a deliberate skip-list kept dense for quality).
vLLM's Gemma4 bnb loader ignores that skip-list -- it allocates packed-4bit
shapes for every linear layer and crashes on the ones actually stored dense.
Loading normally (respecting the real skip-list) then saving as plain dense
bf16 sidesteps that loader bug entirely, and vLLM applies the LoRA adapter
itself at serve time (--enable-lora), so plain-gemma and autoqa-gemma both
still come from ONE resident base -- no need to bake the adapter in.

`model.save_pretrained()` hit a `NotImplementedError` in
`revert_weight_conversion`/`core_model_loading.py` on this checkpoint (a
transformers-5.5.0 bug in its own dequantized-save path). Bypassed here by
writing the state dict directly with safetensors instead of going through
that path.
"""
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("HF_HOME", "/opt/ml/hf-cache")

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
import unsloth  # noqa: F401,E402
from unsloth import FastModel  # noqa: E402

BASE_MODEL = "unsloth/gemma-4-e4b-it-unsloth-bnb-4bit"
TOWER_ATTRS = ("vision_tower", "audio_tower", "embed_vision",
               "embed_audio", "multi_modal_projector")
OUT = Path("/opt/ml/models/autoqa-base-dense")


def log(m):
    print(f"[dequant] {m}", flush=True)


def strip_towers(model):
    removed, seen, holders = [], set(), []
    cur = model
    for _ in range(3):
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        holders.append(cur)
        cur = getattr(cur, "model", None)
    for h in holders:
        for a in TOWER_ATTRS:
            if getattr(h, a, None) is not None:
                setattr(h, a, None)
                removed.append(a)
    return removed


def main():
    log(f"loading base (4-bit, respects the real skip-list): {BASE_MODEL}")
    model, tok = FastModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=8192,
        load_in_4bit=True, dtype=None, full_finetuning=False,
    )
    removed = strip_towers(model)
    log(f"stripped modality towers: {removed}")

    log("dequantizing via transformers' built-in model.dequantize() "
        "(hand-rolled bnb unpacking hit shape mismatches on grouped "
        "quantization internals -- this delegates to each Linear4bit "
        "module's own correct dequant/reshape logic instead)")
    model = model.dequantize()
    log("dequantize() complete")

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    # Host RAM on this box is only 15GB. The model itself (~16-18GB in dense
    # bf16) can be at or above that ceiling on its own -- a single
    # host-RAM copy of the whole state dict OOM-kills the process silently
    # (no traceback; that's what happened on the prior attempt). Tensors stay
    # on GPU until their shard's turn, so at most ~2GB of host RAM is used at
    # once -- this writes HF's standard sharded checkpoint format
    # (model-NNNNN-of-MMMMM.safetensors + model.safetensors.index.json),
    # which transformers loads natively, same as any large sharded model.
    log("writing sharded safetensors, ~2GB shards, tensors moved to CPU "
        "one shard at a time (avoids a full second copy in host RAM)")
    sd = model.state_dict()
    keys = [k for k in sd.keys()
            if sd[k] is not None and not any(t in k for t in TOWER_ATTRS)]

    SHARD_BYTES = 2 * 1024**3
    shards = []  # list of list[key]
    cur, cur_bytes = [], 0
    for k in keys:
        nbytes = sd[k].numel() * 2  # bf16 = 2 bytes/elem
        if cur and cur_bytes + nbytes > SHARD_BYTES:
            shards.append(cur)
            cur, cur_bytes = [], 0
        cur.append(k)
        cur_bytes += nbytes
    if cur:
        shards.append(cur)

    import gc
    n_shards = len(shards)
    weight_map = {}
    total_bytes = 0
    for i, shard_keys in enumerate(shards, start=1):
        fname = f"model-{i:05d}-of-{n_shards:05d}.safetensors"
        shard = {}
        for k in shard_keys:
            v = sd.pop(k)
            t = v.detach().to(torch.bfloat16).contiguous().cpu()
            shard[k] = t
            weight_map[k] = fname
            total_bytes += t.numel() * t.element_size()
            del v
        save_file(shard, str(OUT / fname), metadata={"format": "pt"})
        del shard
        gc.collect()
        torch.cuda.empty_cache()
        log(f"shard {i}/{n_shards} written ({fname})")
    del sd
    gc.collect()
    torch.cuda.empty_cache()

    if n_shards > 1:
        index = {"metadata": {"total_size": total_bytes}, "weight_map": weight_map}
        (OUT / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
        log(f"wrote model.safetensors.index.json ({n_shards} shards, "
            f"{total_bytes/1e9:.2f} GB)")
    else:
        # Single shard -- rename to the plain unsharded filename transformers expects.
        (OUT / f"model-00001-of-00001.safetensors").rename(OUT / "model.safetensors")
        log(f"single shard, saved as model.safetensors ({total_bytes/1e9:.2f} GB)")

    cfg = json.loads(json.dumps(model.config.to_dict()))
    cfg.pop("quantization_config", None)
    text_cfg = dict(cfg.get("text_config") or {})
    if text_cfg:
        derived = dict(text_cfg)
        derived["architectures"] = ["Gemma4ForCausalLM"]
        derived["model_type"] = "gemma4_text"
        derived["torch_dtype"] = "bfloat16"
        if "tie_word_embeddings" in cfg:
            derived["tie_word_embeddings"] = cfg["tie_word_embeddings"]
        cfg = derived
    else:
        cfg["torch_dtype"] = "bfloat16"
    (OUT / "config.json").write_text(json.dumps(cfg, indent=2))
    log("wrote text-only config.json (Gemma4ForCausalLM, no vision/audio config)")

    tok.save_pretrained(str(OUT))
    log("saved tokenizer")

    gen_cfg = {"bos_token_id": cfg.get("bos_token_id"),
               "eos_token_id": cfg.get("eos_token_id"),
               "pad_token_id": cfg.get("pad_token_id")}
    (OUT / "generation_config.json").write_text(json.dumps(gen_cfg, indent=2))

    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    log(f"done: {total/1e9:.2f} GB at {OUT}")


if __name__ == "__main__":
    main()

# gemma4-inference

OpenAI-compatible inference deployment for a LoRA-fine-tuned `gemma-4-e4b-it`
(AutoQA adapter), served with **vLLM** — real grammar-constrained JSON decoding,
real tool-calling, and continuous batching. Text-only — the vision/audio towers
are dropped before serving.

Two model names, one resident base model:

- `plain-gemma` — base model, no adapter
- `autoqa-gemma` — base + LoRA, applied by vLLM's own adapter hot-swap (`--enable-lora`)

## Why dequantize before serving

The trained checkpoint (`unsloth/gemma-4-e4b-it-unsloth-bnb-4bit`) is
**mixed precision on disk**: most linear layers are packed 4-bit, but a
deliberate skip-list of layers is kept dense bf16 for quality. vLLM's Gemma4
loader doesn't respect that skip-list — it allocates packed-4bit shapes for
every linear layer and crashes with a shape assertion the moment it hits a
layer that's actually stored dense. This isn't specific to this checkpoint:
it's a long-standing, still-open upstream issue affecting any of Unsloth's
dynamic-quantized checkpoints served on vLLM
([unslothai/unsloth#1886](https://github.com/unslothai/unsloth/issues/1886)).

The fix: `dequantize_base.py` loads the base checkpoint once (respecting the
real skip-list, via `transformers`' own `model.dequantize()`), and writes a
uniformly dense bf16 checkpoint with no mixed-precision metadata at all. vLLM
then loads that cleanly and applies its own **uniform** `bitsandbytes`
quantization at serve time (`--quantization bitsandbytes`) — smaller memory
footprint than dense bf16, more headroom for KV cache/concurrency, and no
skip-list for the loader to misinterpret.

The LoRA adapter is **not** folded into this step — it's applied separately by
vLLM (`--enable-lora`), so `plain-gemma` and `autoqa-gemma` still come from one
resident model instead of needing two copies in memory.

`dequantize_base.py` also writes the checkpoint in ~2GB shards, moving one
shard to host RAM at a time. The box this runs on has only 15GB host RAM —
building the whole ~16GB dense state dict in a single Python dict at once
silently OOM-kills the process (no traceback); sharding keeps peak host RAM
usage low regardless of total model size.

## Setup

Dequantize (one-time, run on the training/GPU box — needs a full 4-bit load of
the base model plus enough VRAM headroom to hold the dense conversion):

```bash
uv sync
uv run python dequantize_base.py
# writes /opt/ml/models/autoqa-base-dense (edit BASE_MODEL/OUT in the script to change)
```

Serving runs in a **separate** environment — vLLM pins its own `torch`/
`transformers` versions that conflict with the ones `unsloth` needs for the
dequantize step above, so this repo intentionally does not try to manage both
in one `pyproject.toml`:

```bash
pip install vllm==0.26.0 xgrammar==0.2.3
```

Drop the trained adapter at `/opt/ml/adapters/standard/` (standard PEFT
adapter layout: `adapter_config.json` + `adapter_model.safetensors`).

The server expects a bearer key at `/opt/ml/serve/api_key` — a plain text file
containing the token clients must send as `Authorization: Bearer <token>`.

## Run

```bash
API_KEY=$(cat /opt/ml/serve/api_key)
vllm serve /opt/ml/models/autoqa-base-dense \
  --served-model-name plain-gemma \
  --quantization bitsandbytes \
  --enable-lora --lora-modules autoqa-gemma=/opt/ml/adapters/standard --max-lora-rank 16 \
  --enable-auto-tool-choice --tool-call-parser functiongemma \
  --structured-outputs-config '{"backend": "xgrammar"}' \
  --max-model-len 8192 --dtype bfloat16 --gpu-memory-utilization 0.90 --enforce-eager \
  --port 8000 --api-key "$API_KEY"
```

`--enforce-eager` trades away CUDA-graph capture for lower baseline memory
usage — on a single L4 (24GB), the graph-capture memory reservation otherwise
leaves too little room for KV cache at `--max-model-len 8192`.

In production the flags live in `deploy/vllm_serve.sh` and the unit just runs
that script. That indirection is deliberate: `--structured-outputs-config`
takes a JSON value, and getting `{"backend": "xgrammar"}` intact through both
shell heredoc expansion *and* systemd's own quote parsing requires escaping
that silently collapses to `{backend: xgrammar}`, which then fails pydantic
validation at startup. Keeping the flags in a plain script avoids the problem;
paths reach it as environment variables from the unit.

```bash
sudo cp deploy/vllm_serve.sh /opt/ml/serve/ && sudo chmod +x /opt/ml/serve/vllm_serve.sh
sudo cp deploy/autoqa-vllm.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now autoqa-vllm.service
```

`deploy/bootstrap-7b-autoqa-vllm.sh.tftpl` is the terraform user-data section
that regenerates both files on every reprovision, so the server comes back on
its own after `terraform destroy` + `apply` — the model, adapter, serve env and
API key persist on the EBS volume; only the wrapper and unit need recreating.

## API

OpenAI-compatible `/v1/chat/completions` and `/v1/models`.

### Structured output / constrained decoding

Same guarantee as OpenAI structured outputs: the sampler is masked at every
generation step (via vLLM's `xgrammar` backend) so only tokens that keep the
output on a path to a schema-valid JSON document are allowed.

```bash
curl -s https://<host>:8000/v1/chat/completions \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "autoqa-gemma",
    "messages": [
      {"role":"system","content":"You are a conversation triage analyst."},
      {"role":"user","content":"...transcript..."}
    ],
    "temperature": 0,
    "response_format": {
      "type": "json_schema",
      "json_schema": {"name":"classify","schema": { "...": "..." }}
    }
  }'
```

### Tool calling

Native OpenAI-style `tools` / `tool_choice` is also supported, via vLLM's
Gemma-specific tool-call parser (`functiongemma`):

```bash
curl -s https://<host>:8000/v1/chat/completions \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "autoqa-gemma",
    "temperature": 0,
    "messages": [{"role":"user","content":"Budget is 50000, decision by end of quarter."}],
    "tools": [{"type":"function","function":{"name":"extract_deal","parameters":{"...":"..."}}}],
    "tool_choice": {"type":"function","function":{"name":"extract_deal"}}
  }'
```

**`tools` must be passed if the model was trained with them rendered into the
prompt** — the chat template puts the tool schema in the system turn, and
omitting it at inference puts the model off-distribution.

### Batching

Real continuous batching (paged KV cache, iteration-level scheduling) via
vLLM — not the time-windowed request-coalescing a hand-rolled server would
need. Concurrent requests share GPU execution properly; a slow request
doesn't block fast ones behind it in the same way a blocking `generate()` call
would. Verified: 8 concurrent requests complete in ~16s total on a single L4.

## Files

| File | Purpose |
|---|---|
| `dequantize_base.py` | One-time step: mixed-precision base checkpoint → uniform dense bf16 |
| `deploy/autoqa-vllm.service` | systemd unit for the live vLLM server (auto-start, auto-restart) |

## Known limitations

- Single GPU, single vLLM process — no horizontal scaling built in.
- `--enforce-eager` disables CUDA graphs to leave headroom for KV cache on a
  24GB card; a larger GPU could drop this flag for faster per-step decode.
- The dequantized checkpoint is larger on disk (~16GB dense bf16, sharded)
  than the original mixed-precision checkpoint (~10GB) — this is the tradeoff
  for sidestepping the vLLM loader bug; vLLM's own `--quantization
  bitsandbytes` flag re-quantizes it uniformly at load time, so resident GPU
  memory ends up smaller than the on-disk dense checkpoint, not larger.

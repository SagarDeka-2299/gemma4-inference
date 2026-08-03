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
then loads that cleanly and quantizes it **uniformly to FP8 W8A8 on load**
(`--quantization fp8`) — no skip-list left for the loader to misinterpret.

FP8 rather than bitsandbytes, and the difference is large. bitsandbytes saves
memory but dequantizes with slow kernels; the L4 is compute capability 8.9
(Ada), so FP8 tensor cores are native. Measured on identical 8-concurrent
payloads on the same GPU:

| | bitsandbytes | FP8 |
|---|---|---|
| Wall clock | 12 s | **3 s** |
| Aggregate throughput | 56.8 tok/s | **252.0 tok/s** |

FP8 weights are *larger* than bnb (10.65 vs 8.92 GiB — FP8 keeps embeddings and
`lm_head` in bf16) and CUDA graphs cost another 1.81 GiB, so the KV cache
shrinks (3.57 vs 10.55 GiB). `--kv-cache-dtype fp8` halves KV bytes/token and
buys most of that back — 429,904 cached tokens, ~61 concurrent requests at the
~7k a real AutoQA call uses. So the trade is 4.4x throughput for very little
concurrency headroom.

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
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --enable-lora --lora-modules autoqa-gemma=/opt/ml/adapters/standard --max-lora-rank 16 \
  --enable-auto-tool-choice --tool-call-parser functiongemma \
  --structured-outputs-config '{"backend": "xgrammar"}' \
  --max-model-len 131072 --dtype bfloat16 --gpu-memory-utilization 0.90 \
  --port 8000 --api-key "$API_KEY"
```

CUDA graphs are deliberately left on (no `--enforce-eager`) — that is where
decode throughput comes from, and FP8 frees the memory bitsandbytes needed to
make room for it. LoRA works fine alongside FP8; both routes were verified
serving under it.

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
would. Verified: 8 concurrent requests complete in ~3s total on a single L4
under FP8 (the same test took ~12s under bitsandbytes).

### Caching

Both are on by default in vLLM 0.26 — nothing to configure:

- **Paged KV cache** — 429,904 tokens under FP8 with `--kv-cache-dtype fp8`. At the
  ~7k tokens a real AutoQA request actually uses, that is ~61 concurrent requests.
  (vLLM's own startup line reports `3.28x` — that is the worst case where *every*
  request fills the whole 131k window, not this workload.)
- **Automatic prefix caching** (`enable_prefix_caching`) — reuses the shared
  system prompt and JSON schema across calls; a measured 42.4% hit rate on the
  AutoQA workload. Note it only saves *prefill*. Work that generates several KB
  of constrained JSON is decode-bound, so the quantizer choice moves latency far
  more than prefix caching does.
- **Chunked prefill** — on.

## Files

| File | Purpose |
|---|---|
| `dequantize_base.py` | One-time step: mixed-precision base checkpoint → uniform dense bf16 |
| `deploy/vllm_serve.sh` | Serve flags (kept out of ExecStart so the xgrammar JSON survives quoting) |
| `deploy/autoqa-vllm.service` | systemd unit for the live vLLM server (auto-start, auto-restart) |
| `deploy/bootstrap-7b-autoqa-vllm.sh.tftpl` | terraform user-data section that regenerates both on reprovision |

## Known limitations

- Single GPU, single vLLM process — no horizontal scaling built in.
- KV cache is the binding constraint under FP8 on a 24GB card: 3.57 GiB, ~61x
  concurrency at the ~7k a real request uses. Decode is memory-bandwidth-bound, so a card with
  more bandwidth *and* more VRAM (L40S: 864 GB/s, 48 GB) would raise both
  throughput and concurrency; the L4 is ~300 GB/s.
- The dequantized checkpoint is larger on disk (~16GB dense bf16, sharded) than
  the original mixed-precision checkpoint (~10GB) — the price of sidestepping
  the vLLM loader bug. It is only read at startup.
- First start on a volume restored from an EBS snapshot is slow: blocks are
  lazily hydrated from S3 on first read, so loading the 15.25 GiB checkpoint
  took ~10x longer than on a warm volume. One-time per volume; enable EBS fast
  snapshot restore, or pre-read the model files, to avoid it.

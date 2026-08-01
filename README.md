# gemma4-inference

OpenAI-compatible inference server for a LoRA-fine-tuned `gemma-4-e4b-it`
(AutoQA adapter), with real grammar-constrained JSON decoding and lightweight
dynamic batching. Text-only — the vision/audio towers are stripped at load.

Runs on `transformers` + `unsloth` (not vLLM). vLLM's Gemma4 loader does not
support this checkpoint's mixed 4-bit/dense layout (a deliberate per-layer
quality skip-list), so this serves directly off the same stack that trained
the adapter.

## Why not vLLM

The base checkpoint (`unsloth/gemma-4-e4b-it-unsloth-bnb-4bit`) is **mixed
precision**: most linear layers are packed 4-bit on disk, but a specific
skip-list of layers (e.g. `layers.1.mlp`) is kept dense bf16 by design, for
quality. vLLM 0.26.0's Gemma4 loader allocates packed-shape parameters for
every linear layer regardless of that skip-list, and crashes with a shape
assertion the moment it hits a deliberately-dense layer. `transformers` +
`unsloth` load this checkpoint correctly (it's the format they produced), so
that's what this serves on.

## Setup

```bash
uv sync
```

Requires an NVIDIA GPU with enough VRAM for a ~10 GB text-only 4-bit base
(tested on a single L4, 24 GB). `torch`/`torchvision`/`torchaudio` are pulled
from the PyTorch cu124 index (see `pyproject.toml`).

Drop the trained adapter at `/opt/ml/adapters/standard/` (or edit
`ADAPTER_DIR` in `serve_hf.py`) — `adapter_config.json` +
`adapter_model.safetensors`, the standard PEFT adapter layout.

The server also expects a bearer key at `/opt/ml/serve/api_key` (or edit
`KEYFILE`) — a plain text file containing the token clients must send as
`Authorization: Bearer <token>`.

## Run

```bash
uv run uvicorn serve_hf:app --host 0.0.0.0 --port 8000
```

## API

OpenAI-compatible `/v1/chat/completions` and `/v1/models`. Two model names:

- `plain-gemma` — base model, adapter disabled (`model.disable_adapter()`)
- `autoqa-gemma` — base + LoRA adapter

Both share one resident 10 GB weight copy; only the adapter toggles.

### Structured output / constrained decoding

Same guarantee as OpenAI structured outputs: the sampler is masked at every
generation step so only tokens that keep the output on a path to a
schema-valid JSON document are allowed. Invalid JSON is not just unlikely —
it's structurally impossible. Implemented with `lm-format-enforcer`
(`structured.py`), not a prompt hint.

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

`tools` + `tool_choice` (OpenAI function-calling shape) work the same way —
`structured.schema_from_request()` extracts the target schema from either
surface. **`tools` must be passed if the model was trained with them
rendered into the prompt** — the chat template puts the tool schema in the
system turn, and omitting it at inference puts the model off-distribution
(it starts inventing key names instead of following the trained schema).

### Dynamic batching

Not continuous/paged-attention batching (that's what vLLM or Triton give
you) — lightweight request coalescing. Requests arriving within a
`BATCH_WINDOW_SECONDS` (0.5s) window are padded together and run through one
`generate()` call, up to `MAX_BATCH_SIZE` (8) per batch. See `batching.py`.

Two correctness constraints this respects:

1. `plain-gemma` and `autoqa-gemma` never share a physical batch — the LoRA
   toggle is whole-model, not per-sequence. Requests are bucketed by
   `(model, temperature)`; one shared lock still serializes actual GPU
   execution across every bucket, since it's the same model object
   underneath regardless of route.
2. Different requests in the same batch can carry different JSON schemas —
   `lm-format-enforcer`'s masking function takes a `batch_id`, so each row
   gets its own grammar (`structured.batched_prefix_allowed_tokens_fn`).

## Files

| File | Purpose |
|---|---|
| `serve_hf.py` | FastAPI app: auth, request validation, OpenAI-shaped responses |
| `batching.py` | Request queues, coalescing window, padded batched `generate()` |
| `structured.py` | Grammar-constrained decoding via `lm-format-enforcer` |

## Known limitations

- One GPU, one process — no horizontal scaling built in.
- Batching is time-windowed, not continuous; a slow row in a batch holds up
  the fast ones sharing that `generate()` call.
- `completion_tokens` in the response `usage` can slightly overcount for
  rows that finish early within a batch (trailing pad tokens included in the
  count) — the returned completion *text* is always correct
  (`skip_special_tokens=True` strips them), only the token-count stat is a
  minor approximation.

#!/usr/bin/env bash
# Launch the AutoQA vLLM server.
#
# This lives in a script rather than inline in the unit's ExecStart on purpose:
# --structured-outputs-config takes a JSON value, and getting
# {"backend": "xgrammar"} through shell heredoc expansion *and* systemd's own
# quote parsing needs several layers of escaping that silently collapse it to
# {backend: xgrammar} (which fails pydantic validation at startup). Here the
# quoting is plain, and paths arrive as environment variables from the unit.
#
# FP8, not bitsandbytes: L4 is compute capability 8.9 (Ada), so FP8 tensor cores
# are native. bitsandbytes only saves memory -- it dequantizes with slow kernels.
# Measured on identical 8-concurrent payloads, same GPU: bnb 56.8 tok/s vs FP8
# 252.0 tok/s (4.4x). Worth the ~1.7 GiB extra weights (FP8 leaves embeddings and
# lm_head in bf16), and dropping --enforce-eager restores CUDA graphs (~1.8 GiB),
# which is where decode speed comes from. --kv-cache-dtype fp8 halves KV
# bytes/token and buys back some of the concurrency those two cost.
#
# Nothing is capped: --max-model-len is the base model's full
# max_position_embeddings (131072), there is no max-token limit, and structured
# output plus function calling are unchanged. 8192 was tried first (the value the
# adapter was trained at) and truncated real QA responses mid-JSON -- prompt plus
# a 24-sub-parameter constrained-JSON completion exceeded it, and the client got
# a SyntaxError instead of a score. Raising it did not cost KV cache; vLLM's
# memory profile came out slightly better (429,904 tokens vs 193,687).
#
# Expects, from the unit: MAMBA, AUTOQA_MODEL, AUTOQA_ADAPTER, AUTOQA_KEYFILE.
set -euo pipefail
exec "$MAMBA" run -n serve vllm serve "$AUTOQA_MODEL" \
  --served-model-name plain-gemma \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --enable-lora --lora-modules "autoqa-gemma=$AUTOQA_ADAPTER" --max-lora-rank 16 \
  --enable-auto-tool-choice --tool-call-parser functiongemma \
  --structured-outputs-config '{"backend": "xgrammar"}' \
  --max-model-len 131072 --dtype bfloat16 --gpu-memory-utilization 0.90 \
  --port 8000 --api-key "$(cat "$AUTOQA_KEYFILE")"

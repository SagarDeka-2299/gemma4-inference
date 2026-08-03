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
# Expects, from the unit: MAMBA, AUTOQA_MODEL, AUTOQA_ADAPTER, AUTOQA_KEYFILE.
set -euo pipefail
exec "$MAMBA" run -n serve vllm serve "$AUTOQA_MODEL" \
  --served-model-name plain-gemma \
  --quantization bitsandbytes \
  --enable-lora --lora-modules "autoqa-gemma=$AUTOQA_ADAPTER" --max-lora-rank 16 \
  --enable-auto-tool-choice --tool-call-parser functiongemma \
  --structured-outputs-config '{"backend": "xgrammar"}' \
  --max-model-len 8192 --dtype bfloat16 --gpu-memory-utilization 0.90 --enforce-eager \
  --port 8000 --api-key "$(cat "$AUTOQA_KEYFILE")"

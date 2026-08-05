#!/usr/bin/env bash
# Hands-off AutoQA vLLM deploy. Safe to re-run; safe to run from cloud-init.
#
# Nothing needs to be pre-staged on a disk. The base model comes from its HF
# repo, the LoRA adapter comes from S3, and the serving environment comes from
# the official vLLM image -- so a brand-new box reaches a serving state with no
# manual step and no dependency on a pinned EBS volume.
#
# Why the official image instead of the micromamba env we used before: the env
# took ~25 min to build and therefore had to live on a persistent volume, which
# is exactly the dependency we are removing. vllm/vllm-openai ships the same
# vLLM version already built, so "no volume" and "no 25-minute wait" are the
# same change.
#
# Scratch goes on the instance-store NVMe when present (g6 gives 232 GB, free,
# and faster than gp3). It vanishes on stop, which is fine -- everything on it
# is re-derivable from HF and S3. Falls back to the root disk otherwise.
#
# Two settings here were established by failure, not by preference:
#
#   --gpu-memory-utilization 0.80, not 0.90. The base repo is the *multimodal*
#   Gemma4 (vision + audio towers). Those towers add only ~0.2 GiB of weights,
#   but the model needs 172 CUDA graphs where the text-only checkpoint needed a
#   handful. At 0.90 vLLM sized the KV cache to fill the budget (10.87 weights +
#   8.25 KV) and then died capturing graphs -- OOM trying to allocate 128 MiB
#   with 31 MiB free. 0.80 leaves the ~1.8 GiB graphs need.
#
#   --limit-mm-per-prompt zero images and zero audio. This workload is text
#   only, so reserving multimodal capacity and capturing encoder graphs is pure
#   waste. Declaring zero gives that memory back to the KV cache.
#
# Net effect measured on an L4: KV cache 6.33 GiB / 762,516 tokens and 5.82x
# concurrency at 131k -- against 3.57 GiB / 429,904 / 3.28x on the old
# volume-pinned text-only setup, which was leaving memory unclaimed.
#
# Config comes from the environment; every value has a working default:
#   AUTOQA_BASE_REPO     HF repo id for the base model
#   AUTOQA_ADAPTER_S3    s3:// prefix holding the LoRA adapter
#   AUTOQA_ADAPTER_REGION region of that bucket
#   AUTOQA_IMAGE         vLLM image ref
#   AUTOQA_API_KEY       bearer token; EMPTY (default) means no auth at all
#   AUTOQA_MAX_MODEL_LEN, AUTOQA_GPU_UTIL
#   HF_TOKEN             only needed if the base repo is gated
set -euo pipefail

AUTOQA_BASE_REPO="${AUTOQA_BASE_REPO:-unsloth/gemma-4-e4b-it}"
AUTOQA_ADAPTER_S3="${AUTOQA_ADAPTER_S3:-s3://aurator-intellify/Autoqa/models/run1/best/}"
AUTOQA_ADAPTER_REGION="${AUTOQA_ADAPTER_REGION:-ap-south-1}"
AUTOQA_IMAGE="${AUTOQA_IMAGE:-vllm/vllm-openai:v0.26.0}"
AUTOQA_API_KEY="${AUTOQA_API_KEY:-}"
AUTOQA_MAX_MODEL_LEN="${AUTOQA_MAX_MODEL_LEN:-131072}"
AUTOQA_GPU_UTIL="${AUTOQA_GPU_UTIL:-0.80}"
HF_TOKEN="${HF_TOKEN:-}"

if [ -d /opt/dlami/nvme ] && mountpoint -q /opt/dlami/nvme; then
  SCRATCH=/opt/dlami/nvme/autoqa
else
  SCRATCH=/opt/autoqa
fi
ADAPTER_DIR="$SCRATCH/adapter"
HF_CACHE="$SCRATCH/hf-cache"
COMPILE_CACHE="$SCRATCH/vllm-compile-cache"
LOG_DIR="$SCRATCH/logs"
mkdir -p "$ADAPTER_DIR" "$HF_CACHE" "$COMPILE_CACHE" "$LOG_DIR" /opt/autoqa

echo "[autoqa] scratch=$SCRATCH base=$AUTOQA_BASE_REPO image=$AUTOQA_IMAGE"
if [ -n "$AUTOQA_API_KEY" ]; then
  echo "[autoqa] auth: bearer key supplied"
else
  echo "[autoqa] auth: NONE (keyless) -- port 8000 is open to whoever can reach it"
fi

# --- adapter: read-only pull, never a write back to the bucket ---------------
echo "[autoqa] syncing adapter from $AUTOQA_ADAPTER_S3"
aws s3 sync "$AUTOQA_ADAPTER_S3" "$ADAPTER_DIR/" \
  --region "$AUTOQA_ADAPTER_REGION" --only-show-errors
test -f "$ADAPTER_DIR/adapter_config.json" \
  || { echo "[autoqa] FATAL: adapter_config.json missing after sync"; exit 1; }
test -f "$ADAPTER_DIR/adapter_model.safetensors" \
  || { echo "[autoqa] FATAL: adapter_model.safetensors missing after sync"; exit 1; }
echo "[autoqa] adapter ok: $(du -sh "$ADAPTER_DIR" | cut -f1)"

# --- image ------------------------------------------------------------------
echo "[autoqa] pulling $AUTOQA_IMAGE"
docker pull --quiet "$AUTOQA_IMAGE"

# --- the serve command ------------------------------------------------------
# Written to a file rather than inlined in the unit for the same reason as
# before: --structured-outputs-config takes a JSON value, and getting
# {"backend": "xgrammar"} through systemd's quote parser needs escaping that
# silently collapses to {backend: xgrammar}, which fails pydantic validation at
# startup. A quoted heredoc expands nothing, so the JSON survives verbatim.
cat > /opt/autoqa/run.sh <<'WRAP'
#!/usr/bin/env bash
set -euo pipefail
ARGS=(
  --model "$AUTOQA_BASE_REPO"
  --served-model-name plain-gemma
  --quantization fp8
  --kv-cache-dtype fp8
  --enable-lora --lora-modules "autoqa-gemma=/adapter" --max-lora-rank 16
  --enable-auto-tool-choice --tool-call-parser functiongemma
  --structured-outputs-config '{"backend": "xgrammar"}'
  --max-model-len "$AUTOQA_MAX_MODEL_LEN"
  --dtype bfloat16
  --gpu-memory-utilization "$AUTOQA_GPU_UTIL"
  --limit-mm-per-prompt '{"image": 0, "audio": 0}'
  --port 8000
)
# Keyless unless a key was configured. Appending an empty --api-key would set
# the key to the empty string, which is not the same as disabling auth.
if [ -n "${AUTOQA_API_KEY:-}" ]; then
  ARGS+=(--api-key "$AUTOQA_API_KEY")
fi

ENVS=(-e "HF_HOME=/hf-cache")
# Deliberately an if, not `[ -n ... ] && ENVS+=(...)`. Under `set -e` a trailing
# && whose test fails makes the whole line return non-zero and kills the script,
# so the no-token case -- the default -- would abort before ever starting vLLM.
if [ -n "${HF_TOKEN:-}" ]; then
  ENVS+=(-e "HF_TOKEN=$HF_TOKEN")
fi

# The torch.compile cache is mounted out of the container on purpose. vLLM
# writes it to /root/.cache/vllm, which --rm discards, and rebuilding it costs
# ~130 s of Dynamo bytecode transform on every start. Mounting it to scratch
# means only the very first boot pays that; later restarts reuse it.
exec docker run --rm --name autoqa-vllm \
  --gpus all --ipc host -p 8000:8000 \
  -v "$HF_CACHE:/hf-cache" \
  -v "$COMPILE_CACHE:/root/.cache/vllm" \
  -v "$ADAPTER_DIR:/adapter:ro" \
  "${ENVS[@]}" \
  "$AUTOQA_IMAGE" "${ARGS[@]}"
WRAP
chmod +x /opt/autoqa/run.sh

# --- unit -------------------------------------------------------------------
# A stale container from a previous boot holds both the name and the GPU, and
# `docker run --name` fails on the collision rather than replacing it.
cat > /etc/systemd/system/autoqa-vllm.service <<UNIT
[Unit]
Description=AutoQA Gemma vLLM API (FP8 W8A8 + fp8 KV cache + LoRA + xgrammar, containerised)
After=network-online.target docker.service
Requires=docker.service

[Service]
Type=simple
Environment=AUTOQA_BASE_REPO=$AUTOQA_BASE_REPO
Environment=AUTOQA_IMAGE=$AUTOQA_IMAGE
Environment=AUTOQA_MAX_MODEL_LEN=$AUTOQA_MAX_MODEL_LEN
Environment=AUTOQA_GPU_UTIL=$AUTOQA_GPU_UTIL
Environment=AUTOQA_API_KEY=$AUTOQA_API_KEY
Environment=HF_TOKEN=$HF_TOKEN
Environment=HF_CACHE=$HF_CACHE
Environment=COMPILE_CACHE=$COMPILE_CACHE
Environment=ADAPTER_DIR=$ADAPTER_DIR
ExecStartPre=-/usr/bin/docker rm -f autoqa-vllm
ExecStart=/bin/bash /opt/autoqa/run.sh
ExecStop=/usr/bin/docker stop -t 30 autoqa-vllm
Restart=on-failure
RestartSec=15
TimeoutStartSec=infinity
LimitNOFILE=65536
StandardOutput=append:$LOG_DIR/autoqa-vllm.log
StandardError=append:$LOG_DIR/autoqa-vllm.log

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable --now autoqa-vllm.service
echo "[autoqa] service started; log: $LOG_DIR/autoqa-vllm.log"

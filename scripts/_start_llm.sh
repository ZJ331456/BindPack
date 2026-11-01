#!/usr/bin/env bash
# Internal model-service dependency of run_bindpack.py.
set -euo pipefail
vllm_python="${VLLM_PYTHON:-/root/anaconda3/envs/vllm/bin/python}"
if [[ ! -x "$vllm_python" ]]; then
  vllm_python="$(command -v python3)"
fi
models_dir="${MODELS_DIR:-/root/models}"
exec "$vllm_python" -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 --port 8035 \
  --model "$models_dir/Qwen3-8B" --served-model-name qwen3-8b \
  --trust-remote-code --dtype bfloat16 --max-model-len 8192 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.55}" --enforce-eager \
  --max-num-batched-tokens 2048 --max-num-seqs "${MAX_NUM_SEQS:-8}" \
  --generation-config vllm

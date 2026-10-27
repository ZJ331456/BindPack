#!/usr/bin/env bash
# Shared Qwen3-8B service. GPU headroom is reserved for Qwen3-Embedding-8B.
set -euo pipefail
VLLM_PYTHON="${VLLM_PYTHON:-/root/anaconda3/envs/vllm/bin/python}"
QWEN_RUNTIME="${QWEN_RUNTIME:-/root/.cache/pathcondrag/runtime_deps_qwen3_tf4513}"
export PYTHONPATH="${QWEN_RUNTIME}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.55}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
exec "${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 --port 8035 \
  --model /root/models/Qwen3-8B --served-model-name qwen3-8b \
  --trust-remote-code --dtype bfloat16 --max-model-len 8192 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" --enforce-eager \
  --max-num-batched-tokens 2048 --max-num-seqs "${MAX_NUM_SEQS}" --generation-config vllm

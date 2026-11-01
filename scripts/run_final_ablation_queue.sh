#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TZ=Asia/Shanghai
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export PYTHONPATH="/root/.cache/pathcondrag/runtime_deps_qwen3_tf4513${PYTHONPATH:+:$PYTHONPATH}"
exec /root/anaconda3/envs/rag/bin/python -B -u /root/BindPack/scripts/run_final_ablation_queue.py "$@"

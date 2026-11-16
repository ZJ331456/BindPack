#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TZ=Asia/Shanghai
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
exec /root/anaconda3/envs/bindpack/bin/python -B -u /root/BindPack/scripts/run_final_ablation_queue.py "$@"

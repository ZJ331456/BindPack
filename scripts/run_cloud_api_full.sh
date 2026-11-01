#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="/root/.cache/pathcondrag/runtime_deps_qwen3_tf4513${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
exec /root/anaconda3/envs/rag/bin/python -B -u /root/BindPack/scripts/run_cloud_api_full.py "$@"

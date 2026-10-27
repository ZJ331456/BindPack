#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="/root/.cache/pathcondrag/runtime_deps_qwen3_tf4513${PYTHONPATH:+:$PYTHONPATH}"
exec /root/anaconda3/envs/rag/bin/python -B -u "$project_dir/scripts/run_pack_150_comparison.py" "$@"

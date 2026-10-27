#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export PYTHONPATH="/root/.cache/pathcondrag/runtime_deps_qwen3_tf4513${PYTHONPATH:+:$PYTHONPATH}"
exec /root/anaconda3/envs/rag/bin/python -B -u "$project_dir/scripts/run_complete_mechanism_ablations.py" "$@"

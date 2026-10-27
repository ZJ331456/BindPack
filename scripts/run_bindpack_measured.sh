#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="/root/.cache/pathcondrag/runtime_deps_qwen3_tf4513${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
exec /root/anaconda3/envs/rag/bin/python -B -u "${SCRIPT_DIR}/run_bindpack_measured.py" "$@"

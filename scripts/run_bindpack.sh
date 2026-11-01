#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
bindpack_python="${BINDPACK_PYTHON:-/root/anaconda3/envs/rag/bin/python}"
if [[ ! -x "$bindpack_python" ]]; then
  bindpack_python="$(command -v python3)"
fi
qwen_runtime="${QWEN_RUNTIME:-/root/.cache/pathcondrag/runtime_deps_qwen3_tf4513}"
if [[ -d "$qwen_runtime" ]]; then
  export PYTHONPATH="$qwen_runtime${PYTHONPATH:+:$PYTHONPATH}"
fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
exec "$bindpack_python" -B -u "$project_dir/scripts/run_bindpack.py" "$@"

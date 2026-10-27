#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="/root/.cache/pathcondrag/runtime_deps_qwen3_tf4513${PYTHONPATH:+:$PYTHONPATH}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export TOKENIZERS_PARALLELISM=false
exec /root/anaconda3/envs/rag/bin/python -B -u /root/BindPack/scripts/run_component_diagnostic.py "$@"

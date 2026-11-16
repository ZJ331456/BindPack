#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=''
export BINDPACK_LLM_MAX_IN_FLIGHT=32
exec /root/anaconda3/envs/bindpack/bin/python -B -u "${SCRIPT_DIR}/run_closure_interaction.py" "$@"

#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TOKENIZERS_PARALLELISM=false
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
exec /root/anaconda3/envs/bindpack/bin/python -B -u "${SCRIPT_DIR}/run_bindpack_measured.py" "$@"

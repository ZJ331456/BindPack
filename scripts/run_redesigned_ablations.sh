#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TOKENIZERS_PARALLELISM=false
export TZ=Asia/Shanghai
export TQDM_DISABLE=1
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export BINDPACK_LLM_MAX_IN_FLIGHT=8
exec /root/anaconda3/envs/bindpack/bin/python -B -u "${SCRIPT_DIR}/run_redesigned_ablations.py" "$@"

#!/usr/bin/env bash
# Four whole-stage controls with one new Qwen3 embedding index per dataset.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAG_PYTHON="${RAG_PYTHON:-/root/anaconda3/envs/rag/bin/python}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=42
export TOKENIZERS_PARALLELISM=false OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
ARGS=()
if [[ -n "${OUT_ROOT:-}" ]]; then ARGS+=(--out-root "${OUT_ROOT}"); fi
exec "${RAG_PYTHON}" -B -u "${ROOT}/scripts/run_bindpack_ablations.py" \
  --python "${RAG_PYTHON}" "${ARGS[@]}" "$@"

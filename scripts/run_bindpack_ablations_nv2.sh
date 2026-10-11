#!/usr/bin/env bash
# Frozen NV2 indexes; retrieval ablations and fixed-input replay only.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${ROOT}/outputs/3multi_hop_datasets_results_with_nv2_10_8}"
RAG_PYTHON="${RAG_PYTHON:-/root/anaconda3/envs/rag/bin/python}"
RUN_LOG="${RUN_LOG:-${OUT_ROOT}/logs/run_nv2_ablations.log}"
mkdir -p "$(dirname "${RUN_LOG}")"
exec 9>"${ROOT}/outputs/.recall_top5_improvements.lock"
flock -n 9 || { echo 'Another retrieval experiment is running.' >&2; exit 1; }
exec > >(tee -a "${RUN_LOG}") 2>&1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=42
export TOKENIZERS_PARALLELISM=false OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export PATHCONDRAG_LLM_MAX_IN_FLIGHT=8 HIPPORAG_LLM_MAX_IN_FLIGHT=8
export HIPPORAG_KNN_DEVICE=cpu PATHCONDRAG_SHARED_KNN_DEVICE=cpu
export PATHCONDRAG_NVEMBED_OOM_SPLIT=false
"${RAG_PYTHON}" -B -u "${ROOT}/scripts/run_bindpack_ablations.py" \
  --out-root "${OUT_ROOT}" --python "${RAG_PYTHON}" "$@"

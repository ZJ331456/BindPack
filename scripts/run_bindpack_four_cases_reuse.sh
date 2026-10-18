#!/usr/bin/env bash
# Rerun Full / w/o Route / w/o Plan / w/o Pack, plus HippoRAG 2, on verified frozen indexes.
# Retrieval and reader QA are both recorded. HippoRAG results use cases/<dataset>/hippporag2_reproduce.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAG_PYTHON="${RAG_PYTHON:-/root/anaconda3/envs/rag/bin/python}"
SHARED_INDEX_ROOT="${SHARED_INDEX_ROOT:-${ROOT}/outputs/3multi_hop_datasets_results_10_5/shared_indexes}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=42
export TOKENIZERS_PARALLELISM=false OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

ARGS=(--reuse-shared-index-root "${SHARED_INDEX_ROOT}" --python "${RAG_PYTHON}" --include-hipporag --eval-mode rag_qa)
if [[ -n "${OUT_ROOT:-}" ]]; then ARGS+=(--out-root "${OUT_ROOT}"); fi
DRY_RUN=false
SMOKE=false
LOG_ROOT="${OUT_ROOT:-}"
LOG_SHARED_INDEX_ROOT="${SHARED_INDEX_ROOT}"
CLI=("$@")
for ((i=0; i<${#CLI[@]}; i++)); do
  case "${CLI[i]}" in
    --dry-run) DRY_RUN=true ;;
    --smoke) SMOKE=true ;;
    --out-root) LOG_ROOT="${CLI[i+1]:-}" ;;
    --out-root=*) LOG_ROOT="${CLI[i]#*=}" ;;
    --reuse-shared-index-root) LOG_SHARED_INDEX_ROOT="${CLI[i+1]:-}" ;;
    --reuse-shared-index-root=*) LOG_SHARED_INDEX_ROOT="${CLI[i]#*=}" ;;
  esac
done
if [[ "${DRY_RUN}" == true ]]; then
  exec "${RAG_PYTHON}" -B -u "${ROOT}/scripts/run_bindpack_ablations.py" "${ARGS[@]}" "$@"
fi
if [[ -z "${LOG_ROOT}" ]]; then
  if [[ "${SMOKE}" == true ]]; then
    LOG_ROOT="${ROOT}/outputs/stage_ablations_qwen3_reuse_smoke"
  else
    LOG_ROOT="$(dirname "${LOG_SHARED_INDEX_ROOT}")"
  fi
fi
mkdir -p "${LOG_ROOT}/logs"
MAIN_LOG="${LOG_ROOT}/logs/run_bindpack_four_cases_reuse.log"
{
  printf '\n[%s] Four-case shared-index run\n' "$(date -Iseconds)"
  printf 'Main log: %s\n' "${MAIN_LOG}"
  "${RAG_PYTHON}" -B -u "${ROOT}/scripts/run_bindpack_ablations.py" "${ARGS[@]}" "$@"
} 2>&1 | tee -a "${MAIN_LOG}"

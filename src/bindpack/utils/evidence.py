"""Deterministic text, response and local-rank helpers shared across stages."""
from __future__ import annotations

import json
import re
from typing import List, Optional, Tuple
import numpy as np

_REFERENCE = re.compile(r"\$\{([A-Za-z][\w-]*)\.answer\}")
_HASH = re.compile(r"\b(?:entity-)?[0-9a-f]{32}\b", re.I)


def normalized_text(value: str) -> str:
    return " ".join(re.findall(r"\w+", str(value).casefold()))


def parse_object(raw: str) -> Tuple[Optional[dict], Optional[str]]:
    raw = str(raw or "").strip()
    variants = [raw]
    left, right = raw.find("{"), raw.rfind("}")
    if 0 <= left < right:
        variants.append(raw[left:right + 1])
    for variant in variants:
        try:
            obj = json.loads(variant)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict):
            return obj, None
    return None, "invalid_json_object"


def branch_matches(requirements: dict, bindings: dict) -> bool:
    return all(k in bindings and normalized_text(v) == normalized_text(bindings[k])
               for k, v in requirements.items())


def local_rank_scores(scores: np.ndarray, constant: float) -> List[float]:
    """Local ranks always restart at one; later subquestions get equal priors."""
    values = np.asarray(scores, dtype=np.float64)
    if len(values) == 0:
        return []
    finite = np.where(np.isfinite(values), values, 0.0)
    low, high = float(finite.min()), float(finite.max())
    semantic = (finite - low) / (high - low) if high > low else np.ones(len(finite))
    constant = max(1.0, float(constant))
    return [float(0.75 * constant / (constant + rank) + 0.25 * semantic[rank - 1])
            for rank in range(1, len(finite) + 1)]

"""Canonicalize dependency metadata from existing literal references only."""
from __future__ import annotations
import re
from typing import List, Optional, Tuple

_PLAN_REFERENCE = re.compile(r"\$\{([A-Za-z][\w-]*)\.answer\}")
_ANSWER_REFERENCE_LIKE = re.compile(r"\b[A-Za-z][\w-]*\.answer\b")


def canonicalize_plan_dependencies(payload: dict) -> Tuple[dict, List[dict], Optional[str]]:
    """Derive dependency metadata from existing literal references only.

    The questions and node identifiers are never rewritten. Missing/redundant
    metadata can be corrected when the question already names its dependencies.
    A declared dependency without a placeholder has no such deterministic repair.
    Full ordering, cycle, capacity and depth checks remain with the validator.
    """
    changes = []
    if not isinstance(payload, dict) or not isinstance(payload.get("nodes"), list):
        return payload, changes, "invalid_node_count"
    nodes = payload["nodes"]
    ids = set()
    for item in nodes:
        if not isinstance(item, dict):
            return payload, changes, "invalid_node"
        nid = str(item.get("id", "")).strip()
        if not re.fullmatch(r"[A-Za-z][\w-]*", nid) or nid in ids:
            return payload, changes, "invalid_or_duplicate_node_id"
        ids.add(nid)
    rewritten = []
    for item in nodes:
        question = str(item.get("question", "")).strip()
        matches = list(_PLAN_REFERENCE.finditer(question))
        # Reject malformed, incomplete or merely similar reference text. It
        # must never become a root question by losing a syntactically bad ref.
        if (any(start.start() not in {match.start() for match in matches}
                for start in re.finditer(r"\$\{", question))
                or any(not any(match.start() <= token.start() < token.end() <= match.end()
                               for match in matches)
                       for token in _ANSWER_REFERENCE_LIKE.finditer(question))):
            return payload, changes, "invalid_dependency_placeholder"
        refs = list(dict.fromkeys(match.group(1) for match in matches))
        raw_deps = item.get("depends_on", [])
        if not isinstance(raw_deps, list) or any(not isinstance(dep, str) for dep in raw_deps):
            return payload, changes, "invalid_question_or_dependencies"
        deps = [dep.strip() for dep in raw_deps]
        if any(not re.fullmatch(r"[A-Za-z][\w-]*", dep) for dep in deps):
            return payload, changes, "invalid_dependency_id"
        nid = str(item["id"]).strip()
        if any(dep not in ids or dep == nid for dep in refs + deps):
            return payload, changes, "unknown_or_self_dependency"
        if deps and not refs:
            return payload, changes, "declared_dependency_without_placeholder"
        if raw_deps != refs:
            changes.append({"node_id": nid, "original_depends_on": list(raw_deps),
                            "canonical_depends_on": refs,
                            "added": [dep for dep in refs if dep not in deps],
                            "removed": [dep for dep in dict.fromkeys(deps) if dep not in refs]})
        rewritten.append(dict(item, depends_on=refs))
    return dict(payload, nodes=rewritten), changes, None

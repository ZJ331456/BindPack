"""Dependency reference validation and stable topological layering."""
from __future__ import annotations

import re
from typing import List, Optional, Tuple
from ...utils.evidence import _HASH, _REFERENCE

def validate_plan(payload: dict, max_nodes: int) -> Tuple[List[dict], Optional[str]]:
    """Validate references and return a stable topological ordering."""
    nodes = payload.get("nodes")
    if not isinstance(nodes, list) or not nodes or len(nodes) > max_nodes:
        return [], "invalid_node_count"
    clean: List[dict] = []
    seen = set()
    for item in nodes:
        if not isinstance(item, dict):
            return [], "invalid_node"
        nid, question = str(item.get("id", "")).strip(), str(item.get("question", "")).strip()
        if not re.fullmatch(r"[A-Za-z][\w-]*", nid) or nid in seen:
            return [], "invalid_or_duplicate_node_id"
        deps = item.get("depends_on", [])
        if not question or _HASH.search(question) or not isinstance(deps, list):
            return [], "invalid_question_or_dependencies"
        deps = list(dict.fromkeys(str(x) for x in deps))
        refs = set(_REFERENCE.findall(question))
        if refs != set(deps):
            return [], "dependency_reference_mismatch"
        clean.append({"id": nid, "question": question, "depends_on": deps,
                      "answer_type": str(item.get("answer_type", "unknown"))[:80]})
        seen.add(nid)
    if any(dep not in seen or dep == n["id"] for n in clean for dep in n["depends_on"]):
        return [], "unknown_or_self_dependency"
    ordered, done = [], set()
    while len(ordered) < len(clean):
        ready = [n for n in clean if n["id"] not in done and set(n["depends_on"]) <= done]
        if not ready:
            return [], "cyclic_dependencies"
        for node in ready:
            ordered.append(node)
            done.add(node["id"])
    return ordered, None


def plan_layers(nodes: List[dict]) -> List[List[dict]]:
    layers, finished = [], set()
    while len(finished) < len(nodes):
        ready = [n for n in nodes if n["id"] not in finished and set(n["depends_on"]) <= finished]
        if not ready:
            raise ValueError("Plan is cyclic or has missing dependencies")
        layers.append(ready)
        finished.update(n["id"] for n in ready)
    return layers

"""Dependency ancestor closure used by the final DAG selector and support guard."""

from __future__ import annotations


def _ancestors(plan):
    nodes = {}
    for node in plan:
        if not isinstance(node, dict) or not isinstance(node.get("id"), str):
            return {}, [], "invalid_plan_node"
        nid, deps = node["id"], node.get("depends_on", [])
        if nid in nodes or not isinstance(deps, list) or any(not isinstance(d, str) for d in deps):
            return {}, [], "invalid_plan_dependencies"
        nodes[nid] = dict(node, depends_on=deps)
    if any(dep not in nodes or dep == nid for nid, n in nodes.items() for dep in n["depends_on"]):
        return {}, [], "unknown_or_self_dependency"
    ancestors, ordered = {}, []
    while len(ordered) < len(nodes):
        ready = [nid for nid, n in nodes.items() if nid not in ancestors
                 and all(dep in ancestors for dep in n["depends_on"])]
        if not ready:
            return {}, [], "cyclic_dependencies"
        for nid in ready:
            ancestors[nid] = set(nodes[nid]["depends_on"])
            for dep in nodes[nid]["depends_on"]:
                ancestors[nid].update(ancestors[dep])
            ordered.append(nid)
    return ancestors, ordered, None

"""Bind only available intermediate answers to literal plan references."""
from __future__ import annotations
from typing import Optional
from ...utils.evidence import _HASH, _REFERENCE

def bind_question(question: str, bindings: dict) -> Optional[str]:
    refs = _REFERENCE.findall(question)
    if any(ref not in bindings for ref in refs):
        return None
    if any(_HASH.search(str(bindings[ref])) for ref in refs):
        return None
    return _REFERENCE.sub(lambda m: str(bindings[m.group(1)]), question)


def describe_parent_questions(question: str, plan: list) -> str:
    """Expand references with original subquestions, never intermediate answers.

    The dependency DAG and execution order remain intact. Recursive descriptions
    make the original unknown entity searchable without leaking an extracted
    parent answer into the child retrieval request or extraction prompt.
    """
    nodes = {node["id"]: node["question"] for node in plan}

    def expand(text, visiting):
        def replace(match):
            nid = match.group(1)
            if nid in visiting or nid not in nodes:
                return "the unknown entity"
            parent = expand(nodes[nid], visiting | {nid}).strip().rstrip("?")
            return f"(the answer to: {parent})"
        return _REFERENCE.sub(replace, text)

    return expand(question, set())

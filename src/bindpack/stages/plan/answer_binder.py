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

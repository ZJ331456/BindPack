"""Source-grounded citation and founder checks used by closed DAG packages.

These bounded relation helpers issue no retrieval, embedding or LLM requests.
"""

from __future__ import annotations

import re

from ..plan.answer_validation import _same_entity
from ..plan.relation_guard import _owned_clause, relation_spec, strict_relation_reason
from ..plan.answer_binder import bind_question
from ...utils.evidence import normalized_text
from .terminal_support import _literal, _relation_check, _same_subject, _subject_surface


def _title(document):
    lines = str(document).splitlines()
    return re.sub(r"\s*\([^()]+\)\s*$", "", lines[0]).strip() if len(lines) > 1 else ""


def _identity(entity, document):
    """An exact title or an explicitly supported full name, never surname alone."""
    title = _title(document)
    if not title:
        return False
    if _same_entity(entity, title, document):
        return True
    # Geographic titles sometimes append a region after the actual place name.
    if "," in title and normalized_text(title.split(",", 1)[0]) == normalized_text(entity):
        lead = " ".join(document.splitlines()[1:])[:450]
        return bool(re.match(re.escape(entity) + r"\s+(?:is|was)\b", lead, re.I))
    return False


def _clauses(document):
    body = " ".join(str(document).splitlines()[1:])
    return [c.strip() for c in re.split(r"(?<=[.!?;])\s+", body) if c.strip()]


def _proper_anchors(question):
    names = re.findall(r"\b[A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*)*", question)
    ignore = {"Who", "What", "Which", "Where", "When", "How", "Is", "Was", "The", "A"}
    return [n.rstrip("?.") for n in names if n.rstrip("?.") not in ignore]


def _year_conflict(question, clauses):
    years = set(re.findall(r"\b(?:18|19|20)\d{2}\b", question))
    # Only compare the actual relation-bearing sentences, not unrelated birth
    # dates or other works anywhere in the passage.
    mentioned = set(re.findall(r"\b(?:18|19|20)\d{2}\b", " ".join(clauses)))
    return bool(years and mentioned and years.isdisjoint(mentioned))


def _founder_witness(question, answer, quote, document):
    if not re.search(r"\b(?:co[- ]?)?founded\b|\bfounder\b", question, re.I):
        return None
    for anchor in _proper_anchors(question):
        if _literal(anchor, answer):
            continue
        for marker in re.finditer(r"\b(?:co[- ]?)?founded\s+by\b", quote, re.I):
            if (_owned_clause(anchor, quote[:marker.start()], document, _subject_surface, _same_subject)
                    and _literal(answer, quote[marker.end():])
                    and not re.search(r"\b(?:not|never)\b", quote, re.I)):
                return {"relation": "founder", "anchor": anchor, "quote": quote}
        # Active founder construction: the answer owns the predicate.
        for marker in re.finditer(r"\b(?:co[- ]?)?founded\b", quote, re.I):
            if (_owned_clause(answer, quote[:marker.start()], document, _subject_surface, _same_subject)
                    and _literal(anchor, quote[marker.end():])
                    and not re.search(r"\b(?:not|never)\b", quote, re.I)):
                return {"relation": "founder", "anchor": anchor, "quote": quote}
    return None


def _local_citation(node, proof, bindings, document):
    """Validate the promoted fact; predecessor citations stay literal anchors."""
    question = bind_question(node.get("question", ""), bindings)
    if question is None:
        return None, "unbound_node_question"
    check, reason = _relation_check(node, question, proof, document, bindings)
    if check:
        return {"check": check, "quote": proof["evidence"]}, None
    deps = node.get("depends_on", [])
    if len(deps) != 1 or deps[0] not in bindings:
        return None, reason
    entity, answer, quote = bindings[deps[0]], proof["answer"], proof["evidence"]
    if not _identity(entity, document):
        return None, "candidate_identity_mismatch"
    # An explicitly owned geographic/government membership fact.
    if re.search(r"\b(?:belong|government|region|located|situated|within)\b", question, re.I):
        for clause in _clauses(document):
            marker = re.search(r"\b(?:is|was)\b.*?\b(?:within|part of|in|located in|situated in)\b", clause, re.I)
            if (marker and _owned_clause(entity, clause[:marker.start()], document, _subject_surface, _same_subject)
                    and _literal(answer, clause[marker.start():])
                    and not re.search(r"\b(?:not|never)\b", clause, re.I)):
                return {"check": "owned_geographic_membership", "quote": clause}, None
    # A film based on a bound work may refer to itself as It in the actual
    # director sentence. Its same-document lead must establish the adaptation.
    creator = re.search(r"\b(?:directed|director)\b", question, re.I)
    if creator and re.search(r"\bfilm\b.*\bbased\b", question, re.I):
        lead = _clauses(document)[0] if _clauses(document) else ""
        if (_literal(entity, lead) and re.search(r"\bfilm\b.*\bbased\s+on\b", lead, re.I)
                and not _year_conflict(question, [lead])):
            direct_question = f"Who directed {entity}?"
            reason = strict_relation_reason(relation_spec(direct_question), direct_question, answer,
                                            node.get("answer_type", ""), entity, quote, document,
                                            _subject_surface, _same_subject)
            if reason is None:
                return {"check": "owned_adaptation_creator", "quote": quote, "identity_quote": lead}, None
    return None, reason or "unknown_local_relation"

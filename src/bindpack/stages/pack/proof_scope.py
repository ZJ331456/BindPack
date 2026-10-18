"""Scope and anchor predicates shared by bounded proposals and support protection.

These helpers restrict proposals and identify existing protected sources.
They neither retrieve passages nor perform standalone anchor reranking.
"""

from __future__ import annotations

import re
import unicodedata

from .dependency_closure import _ancestors


PREFIX_K = 5

_STOP = frozenset("a an the and or of in on at to for from by with as is are was were be "
                  "been being who what which where when why how do does did have has had "
                  "it its their his her he she they this that these those one name named "
                  "kind type called would could can show known former individual person".split())

_GENERIC = _STOP | frozenset("film movie song album book television city country county "
                            "town state river university company religion leadership "
                            "history music actor actress composer director president "
                            "birthday birthplace nationality football basketball".split())

_COMPARATIVE = re.compile(r"\b(?:older|younger|earlier|later|longer|shorter|taller|"
                          r"larger|smaller|compare|compared|versus)\b", re.I)


def _norm(value):
    # Accent folding is a spelling normalization, not a surname/name alias.
    text = unicodedata.normalize("NFKD", str(value)).casefold()
    return " ".join(re.findall(r"\w+", "".join(c for c in text if not unicodedata.combining(c))))


def _literal(phrase, text):
    phrase = _norm(phrase)
    return bool(phrase) and f" {phrase} " in f" {_norm(text)} "


def _title(document):
    lines = str(document).splitlines()
    return re.sub(r"\s*\([^()]*\)\s*$", "", lines[0]).strip() if len(lines) > 1 else ""


def _specific(title):
    tokens = _norm(title).split()
    content = [t for t in tokens if t not in _GENERIC]
    return bool(content) and (len(content) >= 2 or len(content[0]) >= 7) and len(tokens) <= 12


def _plan_reason(query, trace, state):
    if _COMPARATIVE.search(query):
        return "comparative_question_out_of_scope"
    plan = state.get("_evidence_plan", trace.get("plan", []))
    ancestors, order, error = _ancestors(plan) if isinstance(plan, list) else ({}, [], "invalid_plan")
    if error:
        return error
    if any(len(node.get("depends_on", [])) > 1 for node in plan):
        return "multi_parent_join_out_of_scope"
    if sum(not n.get("depends_on") for n in plan) > 1:
        return "parallel_roots_out_of_scope"
    if any(len(ancestors[nid]) > 1 for nid in order):
        return "deep_plan_out_of_scope"
    return None


def _protected(trace, original):
    diagnostic = trace.get("improvement_dag_package", {})
    eligible = diagnostic.get("eligible_proofs", {}) if isinstance(diagnostic, dict) else {}
    if not isinstance(eligible, dict):
        return set()
    protected = set()
    for rows in eligible.values():
        if isinstance(rows, (list, tuple)):
            for proof in rows:
                if isinstance(proof, dict) and proof.get("doc_id") in original[:PREFIX_K]:
                    protected.add(proof["doc_id"])
    return protected

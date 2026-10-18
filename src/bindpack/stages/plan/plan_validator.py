"""Plan capacity, relation faithfulness and terminal-computation validation."""
from __future__ import annotations
import re
from typing import List, Optional, Tuple
from ...utils.evidence import normalized_text
from .validation import plan_layers, validate_plan


_PLAN_REFERENCE = re.compile(r"\$\{([A-Za-z][\w-]*)\.answer\}")


_ANSWER_REFERENCE_LIKE = re.compile(r"\b[A-Za-z][\w-]*\.answer\b")


_DEATH_ATTRIBUTE = re.compile(r"\b(?:die|dies|died|death|dead|deceased|passed\s+away)\b", re.I)


_BIRTH_ATTRIBUTE = re.compile(r"\b(?:born|birth|birthday|birthdate)\b", re.I)


_END_ATTRIBUTE = re.compile(r"\b(?:end|ends|ended|ceased|terminated)\b", re.I)


_TIME_QUESTION = re.compile(r"\b(?:when|what\s+(?:date|year|time)|which\s+(?:date|year|time))\b", re.I)


_PURE_EQUALITY = re.compile(
    r"(?:are|is|were|was) VALUE(?: (?:and|or) VALUE)+ (?:the )?"
    r"(?:same|identical|different)(?: (?P<attribute>person|people|individual|individuals|entity|entities|"
    r"answer|answers|value|values|name|names|date|dates|country|countries|nationality|nationalities))?", re.I)


_PURE_ORDERING = re.compile(
    r"(?:which (?:date|year|number|value) is (?:earlier|later|higher|lower|larger|smaller|greater|less) "
    r"VALUE(?: (?:and|or) VALUE)+|"
    r"(?:is|was) VALUE (?:earlier|later|higher|lower|larger|smaller|greater|less) than VALUE)", re.I)


_COMPARABLE_TYPES = {"date", "year", "time", "number", "numeric", "integer", "quantity", "value"}


_IDENTITY_ATTRIBUTE_TYPES = {
    "person": {"person", "people", "individual"}, "people": {"person", "people", "individual"},
    "individual": {"person", "people", "individual"}, "individuals": {"person", "people", "individual"},
    "date": {"date", "year", "time"}, "dates": {"date", "year", "time"},
    "country": {"country"}, "countries": {"country"},
    "nationality": {"nationality"}, "nationalities": {"nationality"},
}


def relation_faithfulness_error(query: str, nodes: List[dict]) -> Optional[str]:
    """Reject explicit birth/death/end attribute swaps without inferring answers.

    This is a narrow lexical safeguard, not a claim of semantic entailment. An
    intermediate birth node is allowed if a requested death node is also present.
    """
    questions = "\n".join(node["question"] for node in nodes)
    requested_death = bool(_DEATH_ATTRIBUTE.search(query))
    requested_birth = bool(_BIRTH_ATTRIBUTE.search(query))
    planned_death = bool(_DEATH_ATTRIBUTE.search(questions))
    planned_birth = bool(_BIRTH_ATTRIBUTE.search(questions))
    if requested_death and planned_birth and not planned_death:
        return "relation_attribute_mismatch_death_to_birth"
    if requested_birth and planned_death and not planned_birth:
        return "relation_attribute_mismatch_birth_to_death"
    if (not requested_birth and _TIME_QUESTION.search(query) and _END_ATTRIBUTE.search(query)
            and planned_birth and not _END_ATTRIBUTE.search(questions)):
        return "relation_attribute_mismatch_end_to_birth"
    return None


def validate_retrieval_plan(payload: dict, node_budget: int,
                            depth_budget: int) -> Tuple[List[dict], Optional[str]]:
    """Keep capacity and dependency depth separate; reject derived comparisons."""
    nodes, reason = validate_plan(payload, node_budget)
    if reason:
        return nodes, reason
    if len(plan_layers(nodes)) > depth_budget:
        return [], "dependency_depth_exceeded"
    for node in nodes:
        # Comparing two retrieved answers is computation, not passage retrieval.
        answer_type = normalized_text(node["answer_type"])
        if len(node["depends_on"]) >= 2 and answer_type in {
            "comparison", "boolean comparison", "final answer", "aggregation"
        }:
            return [], "derived_comparison_is_not_retrieval"
    return nodes, None


def prune_derived_comparison_nodes(nodes: List[dict]) -> Tuple[List[dict], List[dict]]:
    """Remove only terminal computations over at least two explicit answers.

    This deliberately accepts a small grammar. A conjunction, boolean type or
    multi-parent node alone never establishes that no external fact is needed.
    All predecessor evidence nodes and their identifiers remain unchanged.
    """
    by_id = {node["id"]: node for node in nodes}
    used = {dep for node in nodes for dep in node["depends_on"]}
    removed, keep = [], []
    for node in nodes:
        deps = node["depends_on"]
        reason = None
        if node["id"] not in used and len(deps) >= 2:
            expression = _PLAN_REFERENCE.sub("VALUE", node["question"])
            expression = " ".join(re.findall(r"\w+", expression))
            equality = _PURE_EQUALITY.fullmatch(expression)
            if equality:
                attribute = (equality.group("attribute") or "").casefold()
                required_types = _IDENTITY_ATTRIBUTE_TYPES.get(attribute)
                # Two person names do not already establish their countries
                # or nationalities; those still require separate lookups.
                if (required_types is None or
                        all(normalized_text(by_id[dep]["answer_type"]) in required_types for dep in deps)):
                    reason = "literal_identity_comparison_of_bound_values"
            elif (_PURE_ORDERING.fullmatch(expression)
                  and all(normalized_text(by_id[dep]["answer_type"]) in _COMPARABLE_TYPES
                          for dep in deps)):
                reason = "ordering_of_bound_numeric_or_temporal_values"
        if reason:
            removed.append(dict(node, reason=reason))
        else:
            keep.append(node)
    return keep, removed

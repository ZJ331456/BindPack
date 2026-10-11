"""Replay the final support and semantic controls on one frozen proposal.

This experiment module never retrieves, calls a model, loads benchmark labels,
or changes production selection. Every condition retains the same typed scope,
literal-source validator and bounded swap proposal. ``none`` disables only the
*additional* support guard and semantic veto, not the base source checks.

Input is a result row containing ``question`` and ``retrieval_trace.evidence``.
A dict containing ``question`` and ``evidence`` is also accepted. Documents map
physical integer IDs (or their JSON string keys) to the exact source text.

An absent reviewer response is not an approval or a rejection: review-dependent
conditions return ``needs_review`` with no final ranking. The caller can obtain
the unchanged production review prompt with ``build_frozen_review_request`` and
attach a real response as ``trace['frozen_review']``. Context-budget abstentions
remain abstentions rather than requiring an out-of-budget counterfactual call.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any

from ..evidence.selection.evidence_typed_joint_selector import (
    apply_joint_selection,
    build_joint_prompt,
)
from ..evidence.verification.evidence_semantic_swap_gate import (
    build_verifier_prompt,
    validate_verification,
)
from ..evidence.verification.evidence_support_guard import support_guard


SCHEMA_VERSION = "bindpack-review-ablation-v1"
CONDITIONS = {
    "none": (False, False),
    "guard_only": (True, False),
    "veto_only": (False, True),
    "both": (True, True),
}


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _input(trace: dict, documents: dict) -> dict:
    if not isinstance(trace, dict) or not isinstance(documents, dict):
        raise TypeError("trace and documents must be dictionaries")
    query = trace.get("question")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("The frozen input must include its original question")
    if isinstance(trace.get("retrieval_trace"), dict):
        evidence = trace["retrieval_trace"].get("evidence")
    else:
        evidence = trace.get("evidence", trace)
    if not isinstance(evidence, dict):
        raise ValueError("Missing frozen evidence trace")
    diagnostic = evidence.get("improvement_support_semantic_veto")
    if not isinstance(diagnostic, dict):
        raise ValueError("Missing improvement_support_semantic_veto diagnostic")
    order = diagnostic.get("original_parent_doc_ids")
    if (not isinstance(order, list) or len(order) != len(set(order))
            or any(type(value) is not int or value < 0 for value in order)):
        raise ValueError("Frozen parent ranking must contain unique physical integer IDs")
    sources = {}
    for doc_id in order[:10]:
        value = documents.get(doc_id, documents.get(str(doc_id)))
        if not isinstance(value, str):
            raise ValueError(f"Missing original source text for physical ID {doc_id}")
        sources[doc_id] = value
    original_evidence = deepcopy(evidence)
    original_evidence.pop("improvement_support_semantic_veto", None)
    state = {
        "evidence_trace": original_evidence,
        "_evidence_plan": original_evidence.get("plan", []),
        "_evidence_winning_bindings": original_evidence.get("bindings", {}),
        "_evidence_beams": original_evidence.get("branch_scores", []),
    }
    stages = diagnostic.get("llm_stages")
    if not isinstance(stages, dict):
        raise ValueError("Missing frozen proposer/reviewer stages; cannot synthesize model responses")
    stage = stages.get("proposer") or {}
    output = stage.get("output")
    messages = build_joint_prompt(query, order, state, sources.__getitem__)
    _check_prompt(stage, messages, "proposer")
    preflight = stage.get("preflight") or {}
    known_abstention = messages is None or preflight.get("allowed") is False
    if output is None and not known_abstention:
        return {"status": "needs_proposal", "query": query, "order": list(order),
                "state": state, "sources": sources, "diagnostic": diagnostic,
                "stages": stages, "proposer_messages": messages}
    if output is not None:
        _validate_output(output, "proposer")
    proposed, proposal = apply_joint_selection(
        query, order, state, sources.__getitem__,
        output["response"] if output else '{"swap":null}',
        _finish(output) if output else "stop",
    )
    if preflight.get("allowed") is False:
        proposal = dict(proposal, abstain_reason="proposer_prompt_exceeds_context_limit")
    recorded = diagnostic.get("base_proposal")
    if isinstance(recorded, dict) and proposal != recorded:
        raise ValueError("The frozen proposer no longer reproduces its recorded base proposal")
    allowed, guard = support_guard(query, order, state, sources.__getitem__, proposal)
    return {"status": "ready", "query": query, "order": list(order),
            "proposed": proposed, "proposal": proposal, "guard_allowed": allowed,
            "guard": guard, "state": state, "sources": sources,
            "diagnostic": diagnostic, "stages": stages,
            "proposer_messages": messages, "frozen_review": trace.get("frozen_review")}


def _validate_output(output: dict, stage: str) -> None:
    if (not isinstance(output, dict) or not isinstance(output.get("response"), str)
            or not isinstance(output.get("metadata", {}), dict)):
        raise ValueError(f"Invalid real frozen {stage} response")


def _finish(output: dict | None) -> str:
    return (output.get("metadata") or {}).get("finish_reason", "unknown") if output else "stop"


def _check_prompt(stage: dict, messages: list | None, name: str) -> None:
    saved = (stage.get("preflight") or {}).get("messages_sha256")
    if saved and (messages is None or _hash(messages) != saved):
        raise ValueError(f"Frozen {name} prompt does not match its recorded SHA256")


def _review(context: dict) -> dict:
    if context["proposed"] == context["order"]:
        return {"status": "not_required", "messages": None, "output": None,
                "preflight": None, "origin": "no_valid_base_proposal"}
    messages = build_verifier_prompt(
        context["query"], context["order"], context["state"],
        context["sources"].__getitem__, context["proposal"],
    )
    if messages is None:
        raise ValueError("A source-validated base proposal has no production reviewer prompt")
    stage = context["stages"].get("reviewer") or {}
    _check_prompt(stage, messages, "reviewer")
    output = stage.get("output")
    preflight = stage.get("preflight")
    origin = "recorded_response"
    if output is None and context.get("frozen_review") is not None:
        supplement = context["frozen_review"]
        if not isinstance(supplement, dict) or supplement.get("messages_sha256") != _hash(messages):
            raise ValueError("Supplemental frozen review must identify the exact production prompt SHA256")
        output = supplement.get("output")
        preflight = supplement.get("preflight", preflight)
        origin = "supplemental_frozen_response"
    if output is not None:
        _validate_output(output, "reviewer")
        status = "available"
    elif isinstance(preflight, dict) and preflight.get("allowed") is False:
        status = "context_blocked"
    else:
        status = "needs_review"
    return {"status": status, "messages": messages, "output": output,
            "preflight": preflight, "origin": origin}


def build_frozen_review_request(trace: dict, documents: dict) -> dict:
    """Return the unchanged reviewer request and any already frozen response.

    On ``needs_review`` the caller should issue this request once, subject to the
    existing context/output budgets, and persist ``messages_sha256``, real output
    metadata, cache hit and token counts under ``trace['frozen_review']``. All
    four conditions then share that one response. This function issues no call.
    """
    context = _input(trace, documents)
    if context["status"] != "ready":
        return {"schema_version": SCHEMA_VERSION, "status": context["status"],
                "request": None, "recorded_output": None}
    review = _review(context)
    request = None
    if review["messages"] is not None:
        request = {"messages": review["messages"],
                   "response_format": {"type": "json_object"},
                   "messages_sha256": _hash(review["messages"])}
    return {"schema_version": SCHEMA_VERSION, "status": review["status"],
            "request": request, "recorded_output": deepcopy(review["output"]),
            "recorded_preflight": deepcopy(review["preflight"]),
            "origin": review["origin"], "proposal": deepcopy(context["proposal"]),
            "guard_allowed": context["guard_allowed"]}


def _cost(stage: dict | None) -> dict:
    stage = stage or {}
    output = stage.get("output")
    metadata = (output.get("metadata") or {}) if isinstance(output, dict) else {}
    preflight = stage.get("preflight") or {}
    return {"logical_requests": int(output is not None),
            "prompt_tokens": metadata.get("prompt_tokens", preflight.get("prompt_tokens")),
            "completion_tokens": metadata.get("completion_tokens"),
            "cache_hit": output.get("cache_hit") if isinstance(output, dict) else None,
            "preflight": deepcopy(preflight), "metadata": deepcopy(metadata)}


def run_review_ablations(trace: dict, documents: dict) -> dict:
    """Replay four controls without changing sources, proposals or model output.

    Unknown counterfactual rankings are ``None`` rather than an invented old
    ranking. A guard rejection already determines ``both`` without an unavailable
    review; ``veto_only`` still requires that review. No gold data enters choices.
    """
    context = _input(trace, documents)
    old = context["order"]
    if context["status"] != "ready":
        return {"schema_version": SCHEMA_VERSION, "status": context["status"],
                "gold_labels_used": False, "original_parent_doc_ids": old,
                "conditions": {name: {"status": "needs_proposal", "accepted": None,
                                      "final_doc_ids": None} for name in CONDITIONS}}
    review = _review(context)
    approved, semantic = None, None
    if review["status"] in ("available", "context_blocked", "not_required"):
        output = review["output"]
        approved, semantic = validate_verification(
            context["query"], old, context["state"], context["sources"].__getitem__,
            context["proposal"], output["response"] if output else '{}', _finish(output),
        )
        if review["status"] == "context_blocked":
            semantic = dict(semantic, rejection_reason="reviewer_prompt_exceeds_context_limit")
    has_proposal = context["proposed"] != old
    conditions = {}
    for name, (use_guard, use_veto) in CONDITIONS.items():
        requires_review = has_proposal and use_veto and (not use_guard or context["guard_allowed"])
        pending = requires_review and approved is None
        accepted = None if pending else bool(
            has_proposal and (not use_guard or context["guard_allowed"])
            and (not use_veto or approved)
        )
        final = None if pending else list(context["proposed"] if accepted else old)
        conditions[name] = {
            "status": "needs_review" if pending else "resolved",
            "support_guard_enabled": use_guard, "semantic_veto_enabled": use_veto,
            "accepted": accepted, "final_doc_ids": final,
            "guard_allowed": context["guard_allowed"] if use_guard else None,
            "semantic_approved": approved if use_veto else None,
            "top2_preserved": final[:2] == old[:2] if final is not None else None,
            "top10_set_preserved": set(final[:10]) == set(old[:10]) if final is not None else None,
            "suffix_preserved": final[10:] == old[10:] if final is not None else None,
            # These are deployment requirements, not the cost of common frozen acquisition.
            "deployment_proposer_requests": int(context["stages"].get("proposer", {}).get("output") is not None),
            "deployment_reviewer_requests": int(
                use_veto and has_proposal and review["status"] != "context_blocked"),
        }
    both = conditions["both"]
    recorded = context["diagnostic"]
    matches = None if both["status"] != "resolved" else (
        both["accepted"] == recorded.get("accepted")
        and both["final_doc_ids"] == recorded.get("final_doc_ids")
    )
    if matches is False and review["origin"] != "supplemental_frozen_response":
        raise ValueError("Full guard+veto replay differs from the original accepted decision or ranking")
    stages = {
        "proposer": _cost(context["stages"].get("proposer")),
        "reviewer": _cost({"output": review["output"], "preflight": review["preflight"]}),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "needs_review" if any(c["status"] == "needs_review" for c in conditions.values()) else "resolved",
        "gold_labels_used": False, "original_parent_doc_ids": list(old),
        "base_proposed_doc_ids": list(context["proposed"]),
        "base_proposal": deepcopy(context["proposal"]),
        "support_guard": deepcopy(context["guard"]), "independent_verification": semantic,
        "review_status": review["status"], "review_origin": review["origin"],
        "conditions": conditions, "both_matches_recorded": matches,
        "frozen_input_sha256": _hash({"query": context["query"], "parent": old,
            "plan": context["state"]["_evidence_plan"],
            "bindings": context["state"]["_evidence_winning_bindings"],
            "branches": context["state"]["_evidence_beams"],
            "dag": context["state"]["evidence_trace"].get("improvement_dag_package"),
            "sources": {str(k): v for k, v in context["sources"].items()},
            "proposer_output": context["stages"].get("proposer", {}).get("output"),
            "reviewer_output": review["output"]}),
        "frozen_acquisition_cost": stages,
        "frozen_logical_requests": sum(stage["logical_requests"] for stage in stages.values()),
        "http_calls_issued_by_replay": 0,
    }

"""Online source-admission counterfactuals for existing dependency execution.

These adapters replace only the answer-admission gate inside the existing
two-attempt verifier. Prompts, LLM client, output allowance and source text stay
unchanged. String admission is weaker than a literal proof; relation admission
is a bounded existing grammar, not a semantic judge or universal entailment.
The literal condition delegates to the original method without a wrapper.
"""

from __future__ import annotations

import json
import math
import re
from types import MethodType
from functools import wraps

from ..evidence.evidence_retrieval import normalized_text, verify_hypotheses
from ..evidence.planning.evidence_relation_guard import relation_spec
from ..evidence.selection.evidence_dag_package import _proof_relation


_HASH = re.compile(r"\b(?:entity-)?[0-9a-f]{32}\b", re.I)
_REFERENCE = re.compile(r"\$\{([A-Za-z][\w-]*)\.answer\}")


def _string_hypotheses(payload, documents):
    """Weaken exactly the source-span gates; keep original field safeguards."""
    accepted, rejected, seen = [], [], set()
    items = payload.get("hypotheses", [])
    if not isinstance(items, list):
        return [], [{"reason": "invalid_hypotheses_list"}]
    for item in items:
        if not isinstance(item, dict):
            rejected.append({"reason": "invalid_hypothesis"})
            continue
        answer = str(item.get("answer", "")).strip()
        evidence = str(item.get("evidence", "")).strip()
        match = re.fullmatch(r"D?(\d+)", str(item.get("doc_id", "")))
        doc_id = int(match.group(1)) if match else -1
        reason = None
        if doc_id not in documents:
            reason = "unknown_document_reference"
        elif not answer or len(answer) > 200 or _HASH.search(answer) or _REFERENCE.search(answer):
            reason = "invalid_answer_or_hash"
        elif len(evidence) < 8 or len(evidence) > 1500:
            reason = "invalid_evidence_length"
        elif normalized_text(answer) not in normalized_text(documents[doc_id]):
            reason = "answer_not_in_provided_source"
        elif normalized_text(answer) in {"unknown", "none", "not known", "not stated", "no answer"}:
            reason = "missing_answer"
        try:
            confidence = float(item.get("confidence", 0.5))
        except (ValueError, TypeError):
            confidence = 0.0
        if not math.isfinite(confidence):
            confidence = 0.0
        confidence = min(1.0, max(0.0, confidence))
        if confidence < .25 and reason is None:
            reason = "low_claimed_confidence"
        if reason:
            rejected.append({"answer": answer[:200], "doc_id": doc_id, "reason": reason})
            continue
        identity = (normalized_text(answer), doc_id, evidence)
        if identity in seen:
            continue
        seen.add(identity)
        accepted.append({"answer": answer, "doc_id": doc_id, "evidence": evidence,
                         "confidence": confidence})
    accepted.sort(key=lambda item: (-item["confidence"], item["doc_id"], normalized_text(item["answer"])))
    return accepted, rejected


def _relation_hypotheses(payload, documents, question, answer_type):
    accepted, rejected = verify_hypotheses(payload, documents)
    checked = []
    node = {"question": question, "depends_on": [], "answer_type": answer_type}
    for proof in accepted:
        check, reason = _proof_relation(node, question, proof, documents[proof["doc_id"]], {})
        if check:
            checked.append(proof)
        else:
            reason = reason or "unknown_relation_not_promoted"
            unknown = ("unknown" in reason or "unresolved" in reason or
                       "not_supported" in reason or "not_in_quote" in reason)
            rejected.append({"answer": proof["answer"], "doc_id": proof["doc_id"],
                             "reason": reason, "check_outcome": "unknown" if unknown else "rejected",
                             "relation": relation_spec(question)["relation"]})
    return checked, rejected


def _verify_control(self, question, answer_type, docs):
    # The initial and repair messages intentionally match EvidenceRetrieval._verify.
    # Only the admission result/rejection text can change the repair path.
    messages = [
        {"role": "system", "content": (
            "Verify possible answers to one retrieval subquestion using ONLY the "
            "provided passages. Output ONLY JSON {hypotheses:[{answer,evidence,"
            "doc_id,confidence}]}. evidence must be copied exactly from that passage "
            "and contain the literal answer. doc_id must match its D-number. "
            "Include the sentence that directly answers the subquestion. "
            "If a full name appears before the relevant sentence, include that preceding "
            "sentence in the same contiguous quote, or use the answer's literal surface in the quote. "
            "Check that the quoted sentence establishes the requested relationship; "
            "a mentioned name alone is insufficient. Return an empty hypotheses list "
            "when unsupported. Retain up to 3 genuinely different supported answers. "
            "Do not resolve unknown placeholders, invent entities, or output hashes.")},
        {"role": "user", "content": (
            f"Subquestion: {question}\nAnswer type: {answer_type}\n\n" +
            "\n\n".join(f"[D{k}]\n{text}" for k, text in docs.items()))},
    ]
    key = question + "::" + ",".join(map(str, docs))
    mode = self._binding_control_mode
    diagnostics = {"attempts": 0, "outputs": [], "admission_mode": mode,
                   "admission_results": [], "relation_scope": relation_spec(question) if mode == "relation" else None}
    self._verification_diagnostics[key] = diagnostics
    all_rejected = []
    reason = None
    for attempt in range(2):
        payload, reason = self._infer_object(messages)
        diagnostics["attempts"] += 1
        diagnostics["outputs"].append(payload)
        if reason:
            accepted, rejected = [], [{"reason": reason}]
        elif mode == "string":
            accepted, rejected = _string_hypotheses(payload, docs)
        else:
            accepted, rejected = _relation_hypotheses(payload, docs, question, answer_type)
        diagnostics["admission_results"].append({"accepted": len(accepted), "rejected": rejected})
        all_rejected.extend(rejected)
        if accepted or (not rejected and not reason):
            return accepted, all_rejected, reason
        if attempt == 0:
            messages = messages + [
                {"role": "assistant", "content": json.dumps(payload or {}, ensure_ascii=False)},
                {"role": "user", "content": (
                    "The proposed citations failed validation: " + json.dumps(rejected, ensure_ascii=False) +
                    ". Recheck the original subquestion and the provided passages. "
                    "Return a corrected hypothesis only if the exact original quote contains the literal answer "
                    "and establishes the requested relationship. Otherwise return {\"hypotheses\":[]}. "
                    "Do not change passage text or infer unsupported relationships.")},
            ]
    return [], all_rejected, reason


def install_binding_control(runtime, mode: str):
    """Install a controlled gate; ``literal`` restores the exact original verifier.

    Returned runtime can be used by the unchanged dependency-search loop.
    Installation should happen before worker requests are submitted.
    """
    if mode not in {"string", "literal", "relation"}:
        raise ValueError("binding mode must be string, literal or relation")
    if not hasattr(runtime, "_binding_control_original_verify"):
        runtime._binding_control_original_verify = runtime._verify
    runtime._binding_control_mode = mode
    if mode == "literal":
        runtime._verify = runtime._binding_control_original_verify
    else:
        @wraps(runtime._binding_control_original_verify)
        def controlled(instance, question, answer_type, docs):
            return _verify_control(instance, question, answer_type, docs)
        runtime._verify = MethodType(controlled, runtime)
    return runtime

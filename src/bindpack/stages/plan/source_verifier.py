"""Exact-source hypothesis validation and bounded LLM verification."""
from __future__ import annotations

import json
import re
from typing import Dict, List, Tuple
import numpy as np
from ...utils.evidence import _HASH, _REFERENCE, normalized_text
from ..mechanisms import mechanisms


def extract_unverified_hypotheses(payload, documents):
    """Schema/document-ID checks only; no source, quote, relation or type gate."""
    items = payload.get("hypotheses") if isinstance(payload,dict) else None
    if not isinstance(items,list): return [],[{"reason":"invalid_hypotheses_list"}]
    accepted,rejected,seen = [],[],set()
    for item in items:
        if not isinstance(item,dict):
            rejected.append({"reason":"invalid_hypothesis"}); continue
        value = item.get("answer")
        answer = str(value).strip() if isinstance(value,(str,int,float,bool)) else ''
        match = re.fullmatch(r"D?(\d+)",str(item.get("doc_id","")))
        doc_id = int(match.group(1)) if match else -1
        if doc_id not in documents or not answer or len(answer)>200:
            rejected.append({"reason":"invalid_answer_or_document_schema"}); continue
        try: confidence = float(item.get("confidence",.5))
        except (ValueError,TypeError,OverflowError): confidence = .5
        if not np.isfinite(confidence): confidence = .5
        key = (normalized_text(answer),doc_id)
        if key in seen: continue
        seen.add(key)
        accepted.append({"answer":answer.strip(),"doc_id":doc_id,
                         "evidence":str(item.get("evidence","") or ""),
                         "confidence":min(1.,max(0.,confidence)),"source_verified":False})
    accepted.sort(key=lambda p:(-p["confidence"],p["doc_id"],normalized_text(p["answer"])))
    return accepted[:3],rejected

def verify_hypotheses(payload: dict, documents: Dict[int, str]) -> Tuple[List[dict], List[dict]]:
    """Require an exact quoted span and an answer supported inside that span."""
    accepted, rejected, seen = [], [], set()
    items = payload.get("hypotheses", [])
    if not isinstance(items, list):
        return [], [{"reason": "invalid_hypotheses_list"}]
    for item in items:
        if not isinstance(item, dict):
            rejected.append({"reason": "invalid_hypothesis"})
            continue
        answer, span = str(item.get("answer", "")).strip(), str(item.get("evidence", "")).strip()
        raw_id = str(item.get("doc_id", ""))
        match = re.fullmatch(r"D?(\d+)", raw_id)
        doc_id = int(match.group(1)) if match else -1
        reason = None
        if doc_id not in documents:
            reason = "unknown_document_reference"
        elif not answer or len(answer) > 200 or _HASH.search(answer) or _REFERENCE.search(answer):
            reason = "invalid_answer_or_hash"
        elif len(span) < 8 or len(span) > 1500 or span not in documents[doc_id]:
            reason = "evidence_not_exact_substring"
        elif normalized_text(answer) not in normalized_text(span):
            reason = "answer_not_supported_in_quote"
        elif normalized_text(answer) in {"unknown", "none", "not known", "not stated", "no answer"}:
            reason = "missing_answer"
        try:
            confidence = float(item.get("confidence", 0.5))
        except (ValueError, TypeError):
            confidence = 0.0
        if not np.isfinite(confidence):
            confidence = 0.0
        confidence = min(1.0, max(0.0, confidence))
        if confidence < 0.25 and reason is None:
            reason = "low_claimed_confidence"
        if reason:
            rejected.append({"answer": answer[:200], "doc_id": doc_id, "reason": reason})
            continue
        unique = (normalized_text(answer), doc_id, span)
        if unique in seen:
            continue
        seen.add(unique)
        accepted.append({"answer": answer, "doc_id": doc_id, "evidence": span,
                         "confidence": confidence})
    accepted.sort(key=lambda x: (-x["confidence"], x["doc_id"], normalized_text(x["answer"])))
    return accepted, rejected


class SourceVerifier:
    """Verify answers using the original three bounded source passages."""

    def __init__(self, runtime):
        self.runtime = runtime

    def verify(self, question: str, answer_type: str, docs: Dict[int, str]):
        if not mechanisms(self.runtime.cfg).source_bound_eligibility:
            return self.extract_unverified(question, answer_type, docs)
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
                "Do not resolve unknown placeholders, invent entities, or output hashes." )},
            {"role": "user", "content": (
                f"Subquestion: {question}\nAnswer type: {answer_type}\n\n" +
                "\n\n".join(f"[D{k}]\n{text}" for k, text in docs.items()))},
        ]
        key = question + "::" + ",".join(map(str, docs))
        diagnostics = {"attempts": 0, "outputs": []}
        self.runtime._verification_diagnostics[key] = diagnostics
        all_rejected = []
        for attempt in range(2):
            payload, reason = self.runtime._infer_object(messages)
            diagnostics["attempts"] += 1
            diagnostics["outputs"].append(payload)
            if reason:
                accepted, rejected = [], [{"reason": reason}]
            else:
                accepted, rejected = verify_hypotheses(payload, docs)
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

    def extract_unverified(self, question, answer_type, docs):
        """Ordinary intermediate answer extraction, not source verification."""
        messages = [
            {"role":"system","content":(
                "Answer one retrieval subquestion using the supplied passages. Output ONLY JSON "
                "{hypotheses:[{answer,doc_id,confidence}]}. Give up to 3 different plausible "
                "answers and identify an existing D-number associated with each answer. "
                "Use the requested answer type and keep each answer within 200 characters. "
                "No exact quotation or source-verification "
                "certificate is required. Return an empty hypotheses list if no answer is available."
            )},
            {"role":"user","content":f"Subquestion: {question}\nAnswer type: {answer_type}\n\n"+
                "\n\n".join(f"[D{k}]\n{text}" for k,text in docs.items())},
        ]
        payload,reason = self.runtime._infer_object(messages)
        key = question+"::"+",".join(map(str,docs))
        self.runtime._verification_diagnostics[key] = {
            "attempts":1,"outputs":[payload],"source_bound_eligibility":False,
            "mode":"unverified_answer_extraction","format_retries":0}
        if reason: return [],[{"reason":reason}],reason
        accepted,rejected = extract_unverified_hypotheses(payload,docs)
        return accepted,rejected,None

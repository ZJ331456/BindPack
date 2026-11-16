"""Remove answer/source attribution while preserving the extraction candidates."""
from copy import deepcopy


def unbound_pack_state(state):
    """Return a Pack-only view with no precise source or parent-binding record.

    The original execution trace is deliberately left intact for reproducibility.
    Every extraction candidate for a node receives its same node-level quality;
    neither the chosen answer's document nor a source-level confidence is used.
    """
    trace = state.get("evidence_trace", {})
    if "extraction_inputs" not in trace and state.get("_evidence_plan"):
        raise ValueError("w/o Source Binding requires saved extraction_inputs; rerun the Full upstream")
    candidates = deepcopy(state.get("evidence_candidates", {}))
    for candidate in candidates.values():
        candidate["verified"] = []
        candidate["sources"] = [
            dict(source, requirements={}) for source in candidate.get("sources", [])
            if not str(source.get("goal", "")).startswith("dag:")]
    node_inputs = {}
    for item in trace.get("extraction_inputs", []):
        node = item["node"]
        entry = node_inputs.setdefault(node, {"doc_ids": [], "quality": 0.0})
        entry["doc_ids"] = list(dict.fromkeys(entry["doc_ids"] + list(item["doc_ids"])))
        entry["quality"] = max(entry["quality"], float(item["quality"]))
    records = []
    for node, item in node_inputs.items():
        for doc_id in item["doc_ids"]:
            if doc_id not in candidates:
                continue
            record = {"doc_id": doc_id, "goal": "dag:" + node,
                      "quality": item["quality"], "confidence": item["quality"],
                      "requirements": {}, "source_verified": False,
                      "unbound_extraction_candidate": True}
            candidates[doc_id]["verified"].append(dict(record))
            candidates[doc_id]["sources"].append({
                "source": "unbound:" + node, "goal": "dag:" + node,
                "score": item["quality"], "requirements": {}})
            records.append(record)
    clean_trace = {key: deepcopy(trace[key]) for key in (
        "stages", "candidate_count", "search_count", "llm_plan_calls", "llm_verification_calls",
        "plan", "bindings", "execution_mode", "parent_answer_propagation", "dependency_edges_executed",
        "extraction_inputs") if key in trace}
    clean_trace.update(source_binding=False, source_records=records, selected_prefix=[],
                       source_binding_mode="unbound_extraction_candidates",
                       source_binding_parent_context_exposed=False)
    bindings = deepcopy(state.get("_evidence_winning_bindings", {}))
    return {"evidence_candidates": candidates, "evidence_trace": clean_trace,
            "_evidence_plan": deepcopy(state.get("_evidence_plan", [])),
            "_evidence_winning_bindings": bindings,
            "_evidence_beams": [{"bindings": bindings, "proofs": {}}]}

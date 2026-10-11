"""CPU-only selector controls over an immutable pre-package snapshot.

No questions are replanned, proofs fabricated, embeddings recomputed, or gold
labels consulted. ``current_closed`` calls the actual production selector.
The factorial controls share its finite ancestor-closed package enumeration,
protected prefixes, source limits and rank tie break. The flat factor changes
coverage accounting only: a terminal's citation counts without its ancestors.
It is intentionally a controlled objective ablation, not an unrestricted flat
optimizer. Lexical MMR is explicitly named because no dense vectors are saved.
An additional ordinary greedy node-coverage reference permits independent
sources and changes both proposal generation and objective; it is deliberately
excluded from the factorial comparison.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import re
from typing import Callable

from ..evidence.evidence_retrieval import bind_question, branch_matches, normalized_text
from ..evidence.planning.evidence_binding import _same_entity
from ..evidence.selection import evidence_dag_package as dag
from ..evidence.selection.evidence_selection import _ancestors
from ..evidence.selection.evidence_terminal import _literal, _subject_surface


SCHEMA_VERSION = 1
SOURCE_MODES = ("weak", "literal", "current")
FINAL_SCORE_SOURCES = {"final", "final_output", "synthetic_order", "final_order"}


def _integer_id(value):
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("document IDs must be integers or integer strings")
    if isinstance(value, str) and not re.fullmatch(r"[0-9]+", value):
        raise ValueError("invalid document ID")
    return int(value)


def restore_snapshot(snapshot: dict) -> dict:
    """Validate score provenance and restore JSON object keys, without mutation."""
    if not isinstance(snapshot, dict) or snapshot.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("expected fixed-selection snapshot schema_version=1")
    restored = deepcopy(snapshot)
    ids = [_integer_id(d) for d in restored.get("candidate_ids", [])]
    if not ids or len(set(ids)) != len(ids) or len(ids) > 200:
        raise ValueError("candidate_ids must contain 1..200 unique IDs")
    scores = restored.get("candidate_scores")
    if not isinstance(scores, list) or len(scores) != len(ids):
        raise ValueError("one pre-selection relevance score is required per candidate")
    if any(isinstance(s, bool) or not isinstance(s, (int, float)) or not math.isfinite(s) for s in scores):
        raise ValueError("relevance scores must be finite numbers")
    source = restored.get("score_source")
    if not isinstance(source, str) or not source or source in FINAL_SCORE_SOURCES:
        raise ValueError("declare real pre-selection score_source; final synthetic scores are forbidden")
    state = restored.get("state")
    if not isinstance(state, dict):
        raise ValueError("snapshot state is required")
    candidates = state.get("evidence_candidates", {})
    if not isinstance(candidates, dict):
        raise ValueError("evidence_candidates must be an object")
    state["evidence_candidates"] = {_integer_id(d): item for d, item in candidates.items()}
    for record in state.get("evidence_trace", {}).get("selected_prefix", []):
        if isinstance(record, dict) and "doc_id" in record:
            record["doc_id"] = _integer_id(record["doc_id"])
    restored["candidate_ids"] = ids
    restored["candidate_scores"] = [float(s) for s in scores]
    if isinstance(restored.get("documents"), dict):
        restored["documents"] = {_integer_id(d): str(v) for d, v in restored["documents"].items()}
    return restored


def _diagnostic(source_mode, coverage_mode):
    return {"enabled": True, "policy": "fixed_input_controlled_package_selection",
            "source_mode": source_mode, "coverage_mode": coverage_mode,
            "gold_labels_used": False, "extra_requests": 0,
            "entailment_guaranteed": False, "proofs_fabricated": False,
            "eligible_proofs": {}, "rejected_proofs": [], "prefix_optimization": [],
            "promotions": [], "deferred": [], "packages": [], "plan_error": None,
            "max_sources_per_node": 3, "max_plan_nodes": 6, "max_plan_depth": 4,
            "package_enumeration": "same_ancestor_closed_node_subsets_as_production",
            "objective": "terminal_count_then_node_count_then_minimum_changes_then_original_rank"}


def _epistemic(reason):
    """Checker rejection is not a label that a source statement is false."""
    if reason in {"proof_conflicts_with_winning_binding", "proof_conflicts_with_winning_branch"}:
        return "inconsistent_with_frozen_binding"
    return "unknown"


def _unverified_choices(nodes, bindings, winning, candidates, rank, document, diagnostic, literal=False):
    choices = {nid: [] for nid in nodes}
    for nid, proof in winning.get("proofs", {}).items():
        if nid in choices and isinstance(proof, dict):
            choices[nid].append(proof)
    for doc_id, candidate in candidates.items():
        if not isinstance(candidate, dict):
            continue
        for proof in candidate.get("verified", []):
            if not isinstance(proof, dict):
                continue
            goal = str(proof.get("goal", ""))
            if goal.startswith("dag:") and goal[4:] in choices:
                choices[goal[4:]].append(dict(proof, doc_id=doc_id))
    valid = {}
    for nid, items in choices.items():
        seen, checked = set(), []
        for raw in items:
            reason = None
            try:
                doc_id = _integer_id(raw.get("doc_id"))
                value = raw.get("confidence", 0)
                confidence = float(value) if not isinstance(value, bool) else float("nan")
            except (ValueError, TypeError, OverflowError):
                diagnostic["rejected_proofs"].append({"node": nid, "reason": "invalid_proof_fields",
                                                       "epistemic_status": "unknown"})
                continue
            answer, quote = str(raw.get("answer", "")), str(raw.get("evidence", ""))
            if (doc_id, answer, quote) in seen:
                continue
            seen.add((doc_id, answer, quote))
            source = document(doc_id) if doc_id in rank else ""
            requirements = raw.get("requirements", {})
            if doc_id not in candidates or not source:
                reason = "proof_outside_existing_top200_pool"
            elif nid not in bindings or not _same_entity(answer, bindings[nid], source):
                reason = "proof_conflicts_with_winning_binding"
            elif not isinstance(requirements, dict) or not branch_matches(requirements, bindings):
                reason = "proof_conflicts_with_winning_branch"
            elif not math.isfinite(confidence) or confidence < .85:
                reason = "low_or_invalid_confidence"
            elif bind_question(nodes[nid].get("question", ""), bindings) is None:
                reason = "unbound_node_question"
            elif literal and (len(quote) < 8 or len(quote) > 1500 or quote not in source):
                reason = "proof_quote_not_literal_grounded"
            elif literal and not _subject_surface(answer, quote, source)[0] and not _literal(answer, quote):
                reason = "answer_not_source_grounded"
            if reason:
                diagnostic["rejected_proofs"].append({"node": nid, "doc_id": doc_id, "reason": reason,
                                                       "epistemic_status": _epistemic(reason)})
            else:
                checked.append({"doc_id": doc_id, "confidence": confidence,
                                "check": "literal_source_and_answer" if literal else "frozen_answer_string_binding"})
        by_doc = {p["doc_id"]: p for p in checked}
        if by_doc:
            valid[nid] = sorted(by_doc.values(), key=lambda p: (rank[p["doc_id"]], p["doc_id"]))[:3]
    return valid


def _flat_coverage(selected, choices, ancestors, node_order):
    return {nid for nid in node_order if any(p["doc_id"] in selected for p in choices.get(nid, []))}


def _optimize_flat(order, original, size, fixed_count, choices, ancestors, node_order, leaves, packages, diagnostic):
    """Same bounded proposals/guards/tie break as production, without closure credit."""
    if len(order) <= fixed_count:
        return order
    size = min(size, len(order))
    rank = {d: i + 1 for i, d in enumerate(original)}
    fixed = list(order[:fixed_count])
    supported = {p["doc_id"] for rows in choices.values() for p in rows}
    protected = set(order[:size]) & supported
    must_keep = set(fixed) | protected
    old_selected = set(order[:size])
    old_coverage = _flat_coverage(old_selected, choices, ancestors, node_order)
    baseline = (len(old_coverage & leaves), len(old_coverage))
    best_key, best_set, best_coverage = baseline + (0, 0), old_selected, old_coverage
    examined = blocked = 0
    for package in packages:
        required = must_keep | set(package)
        if len(required) > size:
            blocked += 1
            continue
        selected = set(required)
        for d in order:
            if len(selected) >= size:
                break
            selected.add(d)
        coverage = _flat_coverage(selected, choices, ancestors, node_order)
        additions = selected - old_selected
        key = (len(coverage & leaves), len(coverage), -len(additions), -sum(rank[d] for d in additions))
        examined += 1
        if key > best_key:
            best_key, best_set, best_coverage = key, selected, coverage
    closed_before = dag._closed_coverage(old_selected, choices, ancestors, node_order)
    closed_after = dag._closed_coverage(best_set, choices, ancestors, node_order)
    diagnostic["prefix_optimization"].append({"top_k": size, "packages_examined": examined,
        "capacity_deferred": blocked, "coverage_before": sorted(old_coverage),
        "coverage_after": sorted(best_coverage), "complete_terminals_before": sorted(closed_before & leaves),
        "complete_terminals_after": sorted(closed_after & leaves),
        "scored_terminal_nodes_before": sorted(old_coverage & leaves),
        "scored_terminal_nodes_after": sorted(best_coverage & leaves), "protected_doc_ids": sorted(protected),
        "fixed_doc_ids": fixed, "closure_required_for_reward": False})
    if best_key[:2] <= baseline:
        return order
    prefix = fixed + sorted(best_set - set(fixed), key=lambda d: (rank[d], d))
    final = prefix + [d for d in order if d not in best_set]
    for d in prefix:
        if d not in old_selected:
            diagnostic["promotions"].append({"doc_id": d, "top_k": size,
                "from_rank": order.index(d) + 1, "to_rank": final.index(d) + 1,
                "covered_nodes": sorted(best_coverage), "bundle_doc_ids": sorted(best_set & supported)})
    return final


def _control(order, state, document, source_mode, coverage_mode):
    diagnostic = _diagnostic(source_mode, coverage_mode)
    trace = state.get("evidence_trace", {})
    plan = state.get("_evidence_plan", trace.get("plan", []))
    bindings = state.get("_evidence_winning_bindings", trace.get("bindings", {}))
    if not isinstance(plan, list) or not isinstance(bindings, dict):
        diagnostic["plan_error"] = "invalid_plan_binding_or_ranking"
        return list(order), diagnostic
    ancestors, node_order, error = _ancestors(plan)
    depth, nodes = {}, {n["id"]: n for n in plan if isinstance(n, dict) and isinstance(n.get("id"), str)}
    if not error:
        for nid in node_order:
            depth[nid] = 1 + max((depth[d] for d in nodes[nid].get("depends_on", [])), default=0)
    if error or not node_order or len(node_order) > 6 or max(depth.values(), default=0) > 4:
        diagnostic["plan_error"] = error or "empty_or_out_of_budget_plan"
        return list(order), diagnostic
    branches = state.get("_evidence_beams") or trace.get("branch_scores", [])
    winning = next((b for b in branches if isinstance(b, dict) and isinstance(b.get("bindings"), dict)
                    and set(b["bindings"]) == set(bindings) and branch_matches(b["bindings"], bindings)), None)
    if not winning or not isinstance(winning.get("proofs"), dict):
        diagnostic["plan_error"] = "missing_exact_winning_branch"
        return list(order), diagnostic
    rank = {d: i + 1 for i, d in enumerate(order)}
    candidates = state.get("evidence_candidates", {})
    if source_mode == "current":
        choices = dag._collect_proofs(nodes, bindings, winning, candidates, rank, document, diagnostic)
        for rejected in diagnostic["rejected_proofs"]:
            rejected["epistemic_status"] = _epistemic(rejected["reason"])
    else:
        choices = _unverified_choices(nodes, bindings, winning, candidates, rank, document, diagnostic,
                                     literal=source_mode == "literal")
    diagnostic["eligible_proofs"] = choices
    packages = dag._package_sets(choices, ancestors, node_order)
    diagnostic.update(eligible_package_count=len(packages), packages=[sorted(p) for p in packages[:64]])
    children = {dep for n in plan for dep in n.get("depends_on", [])}
    leaves = set(node_order) - children
    for nid in node_order:
        missing = ({nid} | ancestors[nid]) - set(choices)
        if missing:
            diagnostic["deferred"].append({"node": nid, "reason": "missing_supported_ancestor_closure",
                                           "missing_nodes": sorted(missing)})
    optimize = dag._optimize_prefix if coverage_mode == "closed" else _optimize_flat
    final = optimize(list(order), order, 5, 2, choices, ancestors, node_order, leaves, packages, diagnostic)
    final = optimize(final, order, 10, 5, choices, ancestors, node_order, leaves, packages, diagnostic)
    return final, diagnostic


def _lexical_mmr(order, scores, document, weight):
    if not 0 <= weight <= 1 or not math.isfinite(weight):
        raise ValueError("mmr_lambda must lie in [0,1]")
    counts = {d: Counter(normalized_text(document(d)).split()) for d in order}
    df = Counter(token for count in counts.values() for token in count)
    vectors = {}
    for d, count in counts.items():
        values = {token: (1 + math.log(freq)) * (1 + math.log((1 + len(order)) / (1 + df[token])))
                  for token, freq in count.items()}
        norm = math.sqrt(sum(v * v for v in values.values()))
        vectors[d] = {token: value / norm for token, value in values.items()} if norm else {}
    lo, hi = min(scores.values()), max(scores.values())
    relevance = {d: (v - lo) / (hi - lo) if hi > lo else 1. for d, v in scores.items()}
    rank = {d: i for i, d in enumerate(order)}
    selected = list(order[:2])
    remaining = set(order[2:])

    def similarity(left, right):
        a, b = vectors[left], vectors[right]
        if len(a) > len(b):
            a, b = b, a
        return sum(value * b.get(token, 0.) for token, value in a.items())

    # Update the maximum redundancy once per new document, avoiding repeated
    # evaluation of all previously selected pairs at every selection position.
    redundancy = {d: max((similarity(d, p) for p in selected), default=0.) for d in remaining}
    while remaining:
        def key(d):
            return weight * relevance[d] - (1 - weight) * redundancy[d], -rank[d], -d
        chosen = max(remaining, key=key)
        selected.append(chosen)
        remaining.remove(chosen)
        for d in remaining:
            redundancy[d] = max(redundancy[d], similarity(d, chosen))
    return selected


def _ordinary_node_coverage(order, scores, choices):
    """Greedy set coverage reference, without package proposal constraints."""
    by_doc = {d: {nid for nid, rows in choices.items() if any(p["doc_id"] == d for p in rows)}
              for d in order}
    original = list(order)
    rank = {d: i for i, d in enumerate(original)}
    final = list(order)
    summaries = []
    for size, fixed_count in ((5, 2), (10, 5)):
        size = min(size, len(final))
        fixed = final[:min(size, fixed_count)]
        protected = {d for d in final[:size] if by_doc[d]}
        selected = list(fixed) + sorted(protected - set(fixed), key=lambda d: (rank[d], d))
        covered = set().union(*(by_doc[d] for d in selected)) if selected else set()
        remaining = set(final) - set(selected)
        while len(selected) < size and remaining:
            chosen = max(remaining, key=lambda d: (len(by_doc[d] - covered), scores[d], -rank[d], -d))
            selected.append(chosen)
            covered.update(by_doc[chosen])
            remaining.remove(chosen)
        old = set(final[:size])
        final = selected + [d for d in final if d not in set(selected)]
        summaries.append({"top_k": size, "covered_nodes": sorted(covered),
                          "protected_doc_ids": sorted(protected), "fixed_doc_ids": fixed,
                          "promoted_doc_ids": sorted(set(selected) - old)})
    return final, {"policy": "greedy_independent_node_coverage_with_true_relevance_tie_break",
        "gold_labels_used": False, "extra_requests": 0, "controlled_factorial_member": False,
        "production_selector_replayed": False, "source_mode": "current", "eligible_proofs": choices,
        "entailment_guaranteed": False, "package_feasibility_required": False,
        "closure_required_for_reward": False, "terminal_priority": False, "coverage_weight_tuning": False,
        "prefix_optimization": summaries, "top2_preserved": final[:2] == order[:2],
        "document_set_preserved": set(final) == set(order), "unique_documents": len(final) == len(set(final)),
        "top5_set_preserved": set(final[:5]) == set(order[:5]),
        "top10_set_preserved": set(final[:10]) == set(order[:10]),
        "comparison_limit": "changes proposal space and coverage objective; not a pure closure-factor ablation"}


def run_fixed_selection(snapshot: dict, document: Callable[[int], str] | None = None) -> dict:
    """Return six binding/selection controls and three reference rankings.

    Schema: ``schema_version=1``, ``query``, ``candidate_ids`` (the pre-package
    ranking, max 200), ``candidate_scores`` (actual merged/base relevance, never
    synthetic prefix scores), ``score_source``, ``state`` and optional
    ``documents`` (ID-to-text mapping). JSON string IDs are restored locally.
    Original model proposals remain in state; no new proposal is constructed.
    """
    frozen = restore_snapshot(snapshot)
    order, state = frozen["candidate_ids"], frozen["state"]
    docs = frozen.get("documents", {})
    lookup = document or (lambda d: docs.get(d, ""))
    cache = {}

    def source(d):
        if d not in cache:
            cache[d] = str(lookup(d) or "")
        return cache[d]

    if any(not source(d) for d in order):
        raise ValueError("the same source document text must be available for every candidate")
    def digest(value):
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    trace = state.get("evidence_trace", {})
    input_state = {k: state[k] for k in ("_evidence_plan", "_evidence_winning_bindings", "_evidence_beams",
                                       "evidence_candidates") if k in state}
    input_state["evidence_trace"] = {k: trace[k] for k in ("plan", "bindings", "branch_scores", "selected_prefix")
                                     if k in trace}
    hashes = {"candidate_pool_sha256": digest(order), "candidate_scores_sha256": digest(frozen["candidate_scores"]),
              "source_text_sha256": digest({str(d): source(d) for d in order}),
              "frozen_state_sha256": digest(input_state)}
    hashes["snapshot_sha256"] = digest({"query": str(frozen.get("query", "")),
        "score_source": frozen["score_source"], **hashes})
    scores = dict(zip(order, frozen["candidate_scores"]))
    rank = {d: i for i, d in enumerate(order)}
    relevance = order[:2] + sorted(order[2:], key=lambda d: (-scores[d], rank[d], d))
    variants = {}
    reference_diag = {"gold_labels_used": False, "extra_requests": 0, "top2_preserved": True,
                      "score_source": frozen["score_source"], "controlled_factorial_member": False}
    variants["relevance"] = {"ids": relevance, "diagnostics": dict(reference_diag, policy="raw_relevance_with_fixed_top2")}
    weight = float(frozen.get("mmr_lambda", .7))
    variants["mmr_lexical"] = {"ids": _lexical_mmr(order, scores, source, weight),
        "diagnostics": dict(reference_diag, policy="tfidf_cosine_mmr_with_fixed_top2", mmr_lambda=weight,
                            dense_mmr=False)}
    for source_mode in SOURCE_MODES:
        for coverage_mode in ("flat", "closed"):
            if source_mode == "current" and coverage_mode == "closed":
                selected, diagnostic = dag.rerank_dag_packages(order, state, source)
                diagnostic.update(source_mode=source_mode, coverage_mode=coverage_mode,
                                  production_selector_replayed=True)
                for rejected in diagnostic.get("rejected_proofs", []):
                    rejected["epistemic_status"] = _epistemic(rejected["reason"])
            else:
                selected, diagnostic = _control(order, state, source, source_mode, coverage_mode)
                diagnostic["production_selector_replayed"] = False
            diagnostic.update(controlled_factorial_member=source_mode != "literal",
                literal_intermediate_control=source_mode == "literal",
                top2_preserved=selected[:2] == order[:2],
                document_set_preserved=set(selected) == set(order), unique_documents=len(selected) == len(set(selected)),
                top5_set_preserved=set(selected[:5]) == set(order[:5]),
                top10_set_preserved=set(selected[:10]) == set(order[:10]))
            if not all(diagnostic[k] for k in ("top2_preserved", "document_set_preserved", "unique_documents")):
                raise AssertionError("fixed selection ranking invariant failed")
            variants[f"{source_mode}_{coverage_mode}"] = {"ids": selected, "diagnostics": diagnostic}
    ordinary, ordinary_diag = _ordinary_node_coverage(order, scores,
        variants["current_closed"]["diagnostics"].get("eligible_proofs", {}))
    variants["ordinary_node_coverage_greedy"] = {"ids": ordinary, "diagnostics": ordinary_diag}
    return {"schema_version": SCHEMA_VERSION, "candidate_count": len(order), "variants": variants,
            "frozen_hashes": hashes,
            "extra_requests": 0, "gold_labels_used": False,
            "factorial_variant_names": ["weak_flat", "weak_closed", "current_flat", "current_closed"],
            "limitations": ["one frozen winning binding branch", "max three sources per node",
                            "at most six nodes and four dependency levels", "fixed top2 and retained old eligible sources",
                            "flat and closed share finite closed package proposal enumeration",
                            "ordinary greedy coverage changes proposal space and objective; separate reference",
                            "lexical MMR is a reference, not dense MMR",
                            "rejected or unsupported proof does not establish factual falsehood"]}

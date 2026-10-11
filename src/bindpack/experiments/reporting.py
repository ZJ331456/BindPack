"""Offline exact-text retrieval metrics and paired uncertainty estimates.

Gold passages enter this reporting module only after the ranking is frozen.
Bootstrap intervals use query-level resampling and percentile endpoints. They
describe the supplied query population; they do not certify generalization or
turn a two-query smoke check into evidence of statistical significance.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import hashlib
import json
import math

import numpy as np


RECALL_K = (1, 2, 5, 10, 20, 200)
METRIC_NAMES = tuple(f"Recall@{k}" for k in RECALL_K) + (
    "PoolAllGold@200", "AllGold@5", "AllGold@10")


def _query_index(value):
    if type(value) is not int or value < 0:
        raise ValueError("query_index must be a nonnegative integer")
    return value


def score_ranking(query_index: int, ids: list[int], document: Callable[[int], str],
                  gold_docs: list[str]) -> dict:
    """Score physical IDs by exact full-passage text, deduplicating gold texts.

    Equal titles are never merged. Different physical IDs with identical text
    do not count twice toward recall of the same gold passage. No normalization,
    heading match, answer match or oracle reordering is applied.
    """
    index = _query_index(query_index)
    if (not isinstance(ids, list) or any(type(i) is not int or i < 0 for i in ids)
            or len(ids) != len(set(ids))):
        raise ValueError("Ranking IDs must be unique nonnegative physical integers")
    if not callable(document):
        raise TypeError("document must be a physical-ID-to-source-text callable")
    if (not isinstance(gold_docs, (list, tuple, set)) or not gold_docs
            or any(not isinstance(text, str) or not text.strip() for text in gold_docs)):
        raise ValueError("Gold passages must be a nonempty collection of nonblank full source texts")
    gold = set(gold_docs)
    ranked = []
    for doc_id in ids:
        text = document(doc_id)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Missing valid source text for physical ID {doc_id}")
        ranked.append(text)
    metrics = {f"Recall@{k}": len(gold.intersection(ranked[:k])) / len(gold)
               for k in RECALL_K}
    all5 = gold <= set(ranked[:5])
    all10 = gold <= set(ranked[:10])
    pool_all = gold <= set(ranked[:200])
    metrics["PoolAllGold@200"] = float(pool_all)
    first_rank = {}
    for rank, text in enumerate(ranked, 1):
        first_rank.setdefault(text, rank)
    return {"query_index": index, "metrics": metrics,
            "all_gold_top5": all5, "all_gold_top10": all10,
            "pool_all_gold_top200": pool_all,
            "n_gold_documents": len(gold), "n_candidate_documents": len(ids),
            "gold_document_ranks": [first_rank.get(text) for text in sorted(gold)]}


def _value(value, name):
    if not isinstance(value, (int, float, bool)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Invalid per-question metric {name}: {value!r}")
    return float(value)


def _row_values(row):
    if not isinstance(row, dict):
        raise ValueError("Per-question measurements must be dictionaries")
    _query_index(row.get("query_index"))
    metrics = row.get("metrics", row.get("retrieval_metrics"))
    if not isinstance(metrics, dict):
        raise ValueError("Per-question metrics are missing")
    values = [_value(metrics.get(f"Recall@{k}"), f"Recall@{k}") for k in RECALL_K]
    pool = row.get("pool_all_gold_top200", metrics.get("PoolAllGold@200"))
    if pool is None:
        # Compatibility with old reports: complete exact recall implies all-gold.
        pool = values[-1] == 1.
    values.append(_value(pool, "PoolAllGold@200"))
    values.append(_value(row.get("all_gold_top5"), "AllGold@5"))
    values.append(_value(row.get("all_gold_top10"), "AllGold@10"))
    return values


def _indexed(rows):
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise ValueError("At least one per-question measurement is required")
    indexed = {}
    for row in rows:
        values = _row_values(row)
        index = row["query_index"]
        if index in indexed:
            raise ValueError(f"Duplicate query_index in measurements: {index}")
        indexed[index] = values
    return indexed


def aggregate_rows(rows) -> dict:
    """Return macro means; each unique query receives equal weight."""
    indexed = _indexed(rows)
    means = np.asarray(list(indexed.values()), dtype=float).mean(axis=0)
    named = dict(zip(METRIC_NAMES, means.tolist()))
    return {"n_samples": len(indexed),
            "retrieval_metrics": {name: named[name] for name in METRIC_NAMES[:7]},
            "all_gold_top5": named["AllGold@5"],
            "all_gold_top10": named["AllGold@10"],
            "pool_all_gold_top200": named["PoolAllGold@200"],
            "per_question": list(rows)}


def _aligned(groups):
    indexed = [_indexed(group) for group in groups]
    indices = sorted(indexed[0])
    expected = set(indices)
    if any(set(group) != expected for group in indexed[1:]):
        raise ValueError("Paired comparisons require identical query_index sets; missing queries cannot be dropped")
    arrays = [np.asarray([group[index] for index in indices], dtype=float) for group in indexed]
    return indices, arrays


def _bootstrap(values, seed, n_bootstrap):
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(n_bootstrap) is not int or n_bootstrap < 2:
        raise ValueError("n_bootstrap must be an integer of at least two")
    rng = np.random.default_rng(seed)
    n = len(values)
    sampled = np.empty((n_bootstrap, values.shape[1]), dtype=float)
    # Avoid allocating a bootstrap-by-query-by-metric tensor for the full run.
    for start in range(0, n_bootstrap, 128):
        count = min(128, n_bootstrap - start)
        indices = rng.integers(0, n, size=(count, n))
        sampled[start:start + count] = values[indices].mean(axis=1)
    interval = np.quantile(sampled, [.025, .975], axis=0)
    return interval


def _method(indices, seed, n_bootstrap):
    return {"n_samples": len(indices), "query_indices": indices,
            "seed": seed, "n_bootstrap": n_bootstrap,
            "confidence_level": .95,
            "method": "paired_query_percentile_bootstrap",
            "small_sample_warning": len(indices) < 30,
            "claims_statistical_significance": False,
            "interpretation": (
                "Intervals describe paired changes on these queries, not independent validation. "
                "Small subsets and smoke tests do not establish stable gains or absence of regression.")}


def paired_bootstrap(left_rows, right_rows, seed=42, n_bootstrap=2000) -> dict:
    """Bootstrap right minus left using the same sampled query IDs in each arm."""
    indices, (left, right) = _aligned([left_rows, right_rows])
    difference = right - left
    intervals = _bootstrap(difference, seed, n_bootstrap)
    left_mean, right_mean, delta = left.mean(axis=0), right.mean(axis=0), difference.mean(axis=0)
    report = _method(indices, seed, n_bootstrap)
    report["direction"] = "right_minus_left"
    report["metrics"] = {
        name: {"left": float(left_mean[i]), "right": float(right_mean[i]),
               "delta": float(delta[i]), "delta_pp": float(delta[i] * 100),
               "ci95": intervals[:, i].tolist(),
               "ci95_pp": (intervals[:, i] * 100).tolist()}
        for i, name in enumerate(METRIC_NAMES)}
    return report


def interaction_bootstrap(A, B, C, D, seed=42, n_bootstrap=2000) -> dict:
    """Estimate the factorial interaction D-C-B+A on jointly paired queries."""
    indices, (a, b, c, d) = _aligned([A, B, C, D])
    interaction = d - c - b + a
    intervals = _bootstrap(interaction, seed, n_bootstrap)
    means = [arm.mean(axis=0) for arm in (a, b, c, d)]
    estimate = interaction.mean(axis=0)
    report = _method(indices, seed, n_bootstrap)
    report["contrast"] = "D-C-B+A"
    report["metrics"] = {
        name: {"A": float(means[0][i]), "B": float(means[1][i]),
               "C": float(means[2][i]), "D": float(means[3][i]),
               "interaction": float(estimate[i]), "interaction_pp": float(estimate[i] * 100),
               "ci95": intervals[:, i].tolist(),
               "ci95_pp": (intervals[:, i] * 100).tolist()}
        for i, name in enumerate(METRIC_NAMES)}
    return report


def annotation_items(snapshot: dict) -> list[dict]:
    """Export blank human labels for weak/current source-check disagreements.

    Accept a raw fixed-selection snapshot or its replay output. Checker reasons
    are diagnostic suggestions only; no reason is converted to factual falsehood
    and no benchmark gold label or model judge is consulted.
    """
    if not isinstance(snapshot, dict):
        raise TypeError("annotation snapshot must be a dictionary")
    if "variants" in snapshot:
        replay = snapshot
    else:
        from .fixed_selection import run_fixed_selection
        replay = run_fixed_selection(snapshot)
    variants = replay.get("variants") or {}
    weak = (variants.get("weak_closed") or {}).get("diagnostics") or {}
    current = (variants.get("current_closed") or {}).get("diagnostics") or {}
    state = snapshot.get("state") or {}
    trace = state.get("evidence_trace") or {}
    plan = state.get("_evidence_plan", trace.get("plan", []))
    nodes = {str(node.get("id")): node for node in plan if isinstance(node, dict)}
    documents = snapshot.get("documents") or {}

    def eligible(diagnostic):
        return {(str(node), proof.get("doc_id"))
                for node, proofs in (diagnostic.get("eligible_proofs") or {}).items()
                for proof in proofs if isinstance(proof, dict)}

    weak_eligible, current_eligible = eligible(weak), eligible(current)
    structural = {"invalid_proof_fields", "proof_outside_existing_top200_pool", "unbound_node_question",
                  "proof_conflicts_with_winning_binding", "proof_conflicts_with_winning_branch"}
    seen, exported = set(), []
    for mode, diagnostic in (("weak", weak), ("current", current)):
        for proof in diagnostic.get("rejected_proofs", []):
            if not isinstance(proof, dict):
                continue
            node, doc_id, reason = str(proof.get("node", "")), proof.get("doc_id"), str(proof.get("reason", ""))
            key = (node, doc_id, reason)
            if key in seen:
                continue
            seen.add(key)
            suggested = "structural_invalid" if reason in structural else "unknown" if reason else "unconfirmed"
            identity = json.dumps([snapshot.get("query_index"), node, doc_id, reason], ensure_ascii=False)
            exported.append({
                "annotation_id": hashlib.sha256(identity.encode()).hexdigest()[:20],
                "query_index": snapshot.get("query_index"),
                "question": snapshot.get("query", snapshot.get("question", "")),
                "node_id": node, "node_question": nodes.get(node, {}).get("question", ""),
                "doc_id": doc_id, "source_text": documents.get(doc_id, documents.get(str(doc_id), "")),
                "checker": mode, "rejection_reason": reason,
                "weak_eligible": (node, doc_id) in weak_eligible,
                "current_eligible": (node, doc_id) in current_eligible,
                "suggested_diagnostic_status": suggested,
                "human_source_support": "", "human_factual_correctness": "",
                "human_uncertainty_reason": "", "annotator": "", "annotation_notes": "",
                "gold_labels_used": False,
            })
    return exported

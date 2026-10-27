"""Offline document completeness and gold dependency-closed coverage metrics.

This module only reads completed result exports and independent dataset labels;
it never changes retrieval rankings, budgets, planner state, or model outputs.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

from .gold_graph import (DOCUMENT_MAPPING_VERSION, GOLD_GRAPH_VERSION,
                         build_gold_graph, gold_documents_from_sample,
                         validate_gold_graph)


DEFAULT_KS = (5, 10, 20, 200)
METRIC_VERSION = "gold-structural-metrics-v1"


def _validated_ks(ks: Sequence[int]) -> tuple[int, ...]:
    if not ks or any(not isinstance(k, int) or isinstance(k, bool) or k <= 0 for k in ks):
        raise ValueError("ks must contain positive integer ranking positions")
    return tuple(sorted(set(ks)))


def evaluate_graph_coverage(graph: Mapping[str, Any], hit_documents: Iterable[str]) -> dict[str, Any]:
    """Evaluate node support and all gold ancestors, with per-node diagnostics."""
    checked = validate_gold_graph(graph.get("nodes", []), source=graph.get("source", "independent_gold"))
    if not graph.get("valid") or not checked["valid"]:
        return {
            "NodeCov": None, "DCC": None, "ClosureGap": None,
            "reason": graph.get("reason") or checked["reason"],
            "covered_nodes": [], "closed_nodes": [], "missing_ancestors": {},
        }
    hits = set(hit_documents)
    covered = {
        node["id"] for node in checked["nodes"]
        if any(set(bundle).issubset(hits) for bundle in node["evidence_bundles"])
    }
    closed, diagnostics, missing_ancestors = [], [], {}
    for node in checked["nodes"]:
        missing = [ancestor for ancestor in node["ancestors"] if ancestor not in covered]
        is_covered = node["id"] in covered
        is_closed = is_covered and not missing
        if is_closed:
            closed.append(node["id"])
        missing_ancestors[node["id"]] = missing
        diagnostics.append({
            "id": node["id"], "covered": is_covered, "closed": is_closed,
            "ancestors": node["ancestors"], "missing_ancestors": missing,
            "missing_document_bundles": [
                [doc for doc in bundle if doc not in hits] for bundle in node["evidence_bundles"]
            ],
        })
    count = len(checked["nodes"])
    nodecov, dcc = len(covered) / count, len(closed) / count
    if not 0 <= dcc <= nodecov <= 1:
        raise AssertionError("DCC invariant violated")
    return {
        "NodeCov": nodecov, "DCC": dcc, "ClosureGap": nodecov - dcc,
        "reason": None, "node_count": count, "covered_node_count": len(covered),
        "closed_node_count": len(closed), "covered_nodes": sorted(covered),
        "closed_nodes": closed, "missing_ancestors": missing_ancestors,
        "nodes": diagnostics,
    }


def _ranking(row: Mapping[str, Any], *, ranking_complete: bool, ranking_length: int | None) -> dict[str, Any]:
    """Decode only explicitly exported information, never an imagined tail."""
    if "candidate_docs" in row:
        prefix, source = row["candidate_docs"], "candidate_docs"
    else:
        prefix, source = row.get("docs", []), "docs"
    error = None
    if not isinstance(prefix, list) or any(not isinstance(doc, str) for doc in prefix):
        prefix, error = [], "invalid_exported_ranking"
    ranks = {}
    if "gold_document_ranks" in row:
        raw_ranks = row["gold_document_ranks"]
        if not isinstance(raw_ranks, list):
            error = "invalid_gold_document_ranks"
        else:
            for item in raw_ranks:
                if not isinstance(item, Mapping) or not isinstance(item.get("doc"), str):
                    error = "invalid_gold_document_rank_entry"
                    break
                doc, rank = item["doc"], item.get("rank")
                if rank is not None and (not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0):
                    error = "invalid_gold_document_rank_value"
                    break
                if doc in ranks:
                    error = "duplicate_gold_document_rank_entry"
                    break
                ranks[doc] = rank
    first_ranks = {}
    for index, doc in enumerate(prefix, 1):
        first_ranks.setdefault(doc, index)
    for doc, rank in ranks.items():
        if doc in first_ranks and rank != first_ranks[doc]:
            error = "gold_rank_prefix_mismatch"
        elif doc not in first_ranks and rank is not None and rank <= len(prefix):
            error = "gold_rank_prefix_mismatch"
    present_ranks = [rank for rank in ranks.values() if rank is not None]
    if len(set(present_ranks)) != len(present_ranks):
        error = "duplicate_gold_document_rank_position"
    if ranking_length is not None and (
        not isinstance(ranking_length, int) or isinstance(ranking_length, bool)
        or ranking_length < len(prefix)
        or any(rank is not None and rank > ranking_length for rank in ranks.values())
    ):
        error = "invalid_actual_ranking_length"
    if ranking_complete and ranking_length is not None and ranking_length != len(prefix):
        error = "complete_ranking_length_mismatch"
    actual_length = len(prefix) if ranking_complete else ranking_length
    return {
        "prefix": prefix, "ranks": ranks, "first_ranks": first_ranks, "error": error,
        "basis": source + ("+gold_document_ranks" if ranks else ""),
        "exported_length": len(prefix), "actual_length": actual_length,
        "length_known": actual_length is not None,
        "duplicate_positions": len(prefix) - len(set(prefix)),
        "duplicate_rate": (len(prefix) - len(set(prefix))) / len(prefix) if prefix else 0.0,
        "ranking_complete": ranking_complete,
    }


def _membership(doc: str, k: int, ranking: Mapping[str, Any]) -> bool | None:
    if doc in ranking["ranks"]:
        rank = ranking["ranks"][doc]
        return rank is not None and rank <= k
    if doc in ranking["first_ranks"]:
        return ranking["first_ranks"][doc] <= k
    if len(ranking["prefix"]) >= k or ranking["ranking_complete"]:
        return False
    if ranking["actual_length"] is not None and len(ranking["prefix"]) >= ranking["actual_length"]:
        return False
    return None


def evaluate_query_metrics(row: Mapping[str, Any], sample: Mapping[str, Any], dataset_name: str,
                           ks: Sequence[int] = DEFAULT_KS, corpus_docs: Iterable[str] | None = None,
                           ranking_complete: bool = False,
                           ranking_length: int | None = None) -> dict[str, Any]:
    """Compute a query's metrics, using null and a reason when not evaluable."""
    ks = _validated_ks(ks)
    labels = gold_documents_from_sample(sample, dataset_name)
    if labels["valid"] and "gold_docs" in row:
        exported_gold = row["gold_docs"]
        if not isinstance(exported_gold, list) or any(not isinstance(doc, str) for doc in exported_gold):
            labels = dict(labels, valid=False, reason="invalid_exported_gold_documents")
        elif set(exported_gold) != set(labels["docs"]):
            labels = dict(labels, valid=False, reason="gold_documents_mismatch")
    graph = build_gold_graph(sample, dataset_name, corpus_docs=corpus_docs)
    ranking = _ranking(row, ranking_complete=ranking_complete, ranking_length=ranking_length)
    if "gold_document_ranks" in row and labels["valid"] and set(ranking["ranks"]) != set(labels["docs"]):
        ranking["error"] = "gold_rank_document_set_mismatch"
    if labels["reason"] in {"gold_documents_mismatch", "invalid_exported_gold_documents"}:
        ranking["error"] = labels["reason"]
    graph_docs = {
        doc for node in graph["nodes"] for bundle in node["evidence_bundles"] for doc in bundle
    } if graph["valid"] else set()
    at_k = {}
    for k in ks:
        doc_reason = labels["reason"] or ranking["error"]
        doc_membership = {doc: _membership(doc, k, ranking) for doc in labels["docs"]}
        if doc_reason is None and any(hit is None for hit in doc_membership.values()):
            doc_reason = "insufficient_exported_ranking_coverage"
        hits = [doc for doc, hit in doc_membership.items() if hit is True]
        metrics = {
            "Recall": len(hits) / labels["count"] if doc_reason is None else None,
            "FCR": float(len(hits) == labels["count"]) if doc_reason is None else None,
            "gold_document_count": labels["count"], "hit_gold_document_count": len(hits),
            "hit_gold_docs": hits,
            "missing_gold_docs": [doc for doc, hit in doc_membership.items() if hit is False],
            "unknown_gold_docs": [doc for doc, hit in doc_membership.items() if hit is None],
            "document_valid": doc_reason is None, "document_unavailable_reason": doc_reason,
            "NodeCov": None, "DCC": None, "ClosureGap": None,
            "graph_valid": False, "graph_unavailable_reason": graph["reason"] or ranking["error"],
            "covered_nodes": [], "closed_nodes": [], "missing_ancestors": {},
            "exported_prefix_below_k": len(ranking["prefix"]) < k,
            "actual_ranking_below_k": (ranking["actual_length"] < k if ranking["length_known"] else None),
        }
        if graph["valid"] and ranking["error"] is None:
            graph_membership = {doc: _membership(doc, k, ranking) for doc in graph_docs}
            if any(hit is None for hit in graph_membership.values()):
                metrics["graph_unavailable_reason"] = "insufficient_exported_ranking_coverage"
            else:
                graph_result = evaluate_graph_coverage(graph, [doc for doc, hit in graph_membership.items() if hit])
                graph_result.pop("reason", None)
                metrics.update(graph_result, graph_valid=True, graph_unavailable_reason=None)
        at_k[str(k)] = metrics
    valid_doc = [at_k[str(k)]["Recall"] for k in ks if at_k[str(k)]["document_valid"]]
    valid_dcc = [at_k[str(k)]["DCC"] for k in ks if at_k[str(k)]["graph_valid"]]
    if any(a > b for a, b in zip(valid_doc, valid_doc[1:])) or any(a > b for a, b in zip(valid_dcc, valid_dcc[1:])):
        raise AssertionError("fixed-ranking coverage must not decrease with K")
    ranking_metadata = {key: value for key, value in ranking.items() if key not in {"prefix", "ranks", "first_ranks"}}
    return {
        "sample_id": row.get("sample_id", sample.get("hover_id") if dataset_name == "hover"
                             else sample.get("id", sample.get("_id"))),
        "query_index": row.get("query_index"), "gold_documents": labels,
        "gold_graph": graph, "ranking": ranking_metadata, "at_k": at_k,
    }


def evaluate_structural_metrics(rows: Sequence[Mapping[str, Any]], samples: Sequence[Mapping[str, Any]],
                                dataset_name: str, ks: Sequence[int] = DEFAULT_KS,
                                corpus_docs: Iterable[str] | None = None) -> dict[str, Any]:
    """Return unrounded query macro means, label coverage, and full diagnostics.

    Callers must supply samples in exactly the exported row order. Explicit IDs
    and questions, when present in both sources, are checked before computing.
    """
    ks = _validated_ks(ks)
    if len(rows) != len(samples):
        raise ValueError("Result rows and dataset samples must have equal length")
    corpus = None if corpus_docs is None else set(corpus_docs)
    per_query = []
    for index, (row, sample) in enumerate(zip(rows, samples)):
        sample_id = (sample.get("hover_id") if dataset_name == "hover"
                     else sample.get("id", sample.get("_id")))
        if sample_id is not None and row.get("sample_id") is not None and str(sample_id) != str(row["sample_id"]):
            raise ValueError(f"Sample ID mismatch at row {index}")
        question = sample.get("claim") if dataset_name == "hover" else sample.get("question")
        if row.get("question") is not None and question is not None and row["question"] != question:
            raise ValueError(f"Question mismatch at row {index}")
        per_query.append(evaluate_query_metrics(row, sample, dataset_name, ks, corpus_docs=corpus))
    metrics, per_k, dag_subset = {}, {}, {}
    for k in ks:
        results = [query["at_k"][str(k)] for query in per_query]
        docs = [r for r in results if r["document_valid"]]
        graphs = [r for r in results if r["graph_valid"]]
        for name in ("Recall", "FCR", "NodeCov", "DCC", "ClosureGap"):
            valid = [r[name] for r in results if r[name] is not None]
            metrics[f"{name}@{k}"] = sum(valid) / len(valid) if valid else None
        same_subset = [r for r in graphs if r["document_valid"]]
        dag_subset[str(k)] = {
            "n": len(same_subset),
            "Recall": sum(r["Recall"] for r in same_subset) / len(same_subset) if same_subset else None,
            "FCR": sum(r["FCR"] for r in same_subset) / len(same_subset) if same_subset else None,
        }
        per_k[str(k)] = {
            "document_n": len(docs), "graph_n": len(graphs),
            "document_evaluable_fraction": len(docs) / len(results) if results else None,
            "graph_evaluable_fraction": len(graphs) / len(results) if results else None,
            "document_unavailable_reasons": dict(Counter(r["document_unavailable_reason"] for r in results if not r["document_valid"])),
            "graph_unavailable_reasons": dict(Counter(r["graph_unavailable_reason"] for r in results if not r["graph_valid"])),
            "exported_prefix_below_k_count": sum(r["exported_prefix_below_k"] for r in results),
            "exported_prefix_below_k_fraction": sum(r["exported_prefix_below_k"] for r in results) / len(results) if results else None,
            "actual_ranking_below_k_count": sum(r["actual_ranking_below_k"] is True for r in results),
            "actual_ranking_length_unknown_count": sum(r["actual_ranking_below_k"] is None for r in results),
        }
    return {
        "metrics": metrics,
        "coverage": {
            "sample_count": len(per_query),
            "document_labels_valid": sum(q["gold_documents"]["valid"] for q in per_query),
            "graph_labels_valid": sum(q["gold_graph"]["valid"] for q in per_query),
            "document_label_coverage_fraction": sum(q["gold_documents"]["valid"] for q in per_query) / len(per_query) if per_query else None,
            "graph_label_coverage_fraction": sum(q["gold_graph"]["valid"] for q in per_query) / len(per_query) if per_query else None,
            "duplicate_exported_positions": sum(q["ranking"]["duplicate_positions"] for q in per_query),
            "queries_with_duplicate_documents": sum(q["ranking"]["duplicate_positions"] > 0 for q in per_query),
            "per_k": per_k,
        },
        "dag_subset_document_metrics": dag_subset,
        "protocol": {
            "version": METRIC_VERSION, "ks": list(ks), "scale": "0_to_1",
            "aggregation": "query_macro_over_valid_labels_and_observable_rankings",
            "document_mapping_version": DOCUMENT_MAPPING_VERSION,
            "gold_graph_version": GOLD_GRAPH_VERSION,
            "ranking_budget": "original_rank_positions_before_set_conversion",
            "matching": "exact_canonical_passage_string",
            "empty_gold_policy": "not_evaluable",
            "unsupported_graph_policy": "null_with_reason",
            "rank_metadata": "earliest_rank_in_complete_final_ranking; null_means_absent",
            "retrieval_or_planner_inputs_modified": False,
        },
        "per_query": per_query,
    }

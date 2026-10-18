"""Fixed HippoRAG2-style candidate adapter used only by w/o Route.

Fact-frequency entity weights, dense passage restart weights and fixed damping
follow the local BaseRAG graph-search protocol. Safe division also handles graph
nodes which do not occur in selected facts. No enhanced Route method is called.
"""
from __future__ import annotations

from numbers import Integral

import numpy as np

from ...utils.misc_utils import compute_mdhash_id, min_max_normalize


def baseline_graph_search(rag, query, query_fact_scores, top_k_facts,
                          top_k_fact_indices, hop_override):
    if (isinstance(hop_override, bool) or not isinstance(hop_override, Integral)
            or not 1 <= hop_override <= 4):
        raise ValueError("hop_override must be an integer from 1 through 4")
    size = len(rag.graph.vs["name"])
    entity_weights = np.zeros(size, dtype=np.float64)
    occurrences = np.zeros(size, dtype=np.float64)
    entity_names = {}
    scores = np.asarray(query_fact_scores)
    for index, fact in zip(top_k_fact_indices, top_k_facts):
        score = float(scores[int(index)] if scores.ndim else scores)
        for phrase in (str(fact[0]).lower(), str(fact[2]).lower()):
            key = compute_mdhash_id(content=phrase, prefix="entity-")
            vertex = rag.node_name_to_vertex_idx.get(key)
            if vertex is None:
                continue
            frequency = len(rag.ent_node_to_chunk_ids.get(key, ()))
            entity_weights[vertex] += score / max(1, frequency)
            occurrences[vertex] += 1
            entity_names[key] = vertex
    np.divide(entity_weights, occurrences, out=entity_weights, where=occurrences != 0)
    limit = int(rag.config.linking_top_k)
    if limit:
        ranked = sorted(entity_names, key=lambda key: -entity_weights[entity_names[key]])
        for key in ranked[limit:]:
            entity_weights[entity_names[key]] = 0.0

    ids, dense_scores = rag.dense_passage_retrieval(query)
    passage_weights = np.zeros(size, dtype=np.float64)
    for doc_id, score in zip(ids, min_max_normalize(dense_scores)):
        key = rag.passage_node_keys[int(doc_id)]
        vertex = rag.node_name_to_vertex_idx.get(key)
        if vertex is not None:
            passage_weights[vertex] = float(score) * rag.config.passage_node_weight
    weights = entity_weights + passage_weights
    seed_weights = {key: float(entity_weights[vertex]) for key, vertex in entity_names.items()
                    if entity_weights[vertex] > 0}
    total = sum(seed_weights.values())
    ctx = {
        "route_adapter": "hipporag2_style_fixed",
        "hops": int(hop_override), "seed_entities": sorted(seed_weights),
        "seed_distribution": {key: value / total for key, value in seed_weights.items()},
        "fact_seed_distribution": seed_weights, "bridges_by_seed": {}, "bridge_count": 0,
        "iterative_used": False, "iterative_round2_seed_count": 0,
        "qd_used": False, "pcqd_used": False,
    }
    if np.isfinite(weights).all() and float(weights.sum()) > 0:
        ids, dense_scores = rag.run_ppr(weights, damping=rag.config.damping)
    return ids, dense_scores, ctx

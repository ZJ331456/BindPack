from __future__ import annotations

import math
from numbers import Integral
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...utils.misc_utils import min_max_normalize

from .query_seed_prior import QuerySeedPrior
from .bridge_expansion import BridgeExpansion
from .iterative_refinement import IterativeRefinement
from .route_refinement import RouteRefinement
from .path_set_consolidation import PathSetConsolidation
from .baseline import baseline_graph_search


class RouteStage(
    QuerySeedPrior, BridgeExpansion, IterativeRefinement,
    RouteRefinement, PathSetConsolidation,
):
    """RouteStage implementation for candidate discovery."""

    def _path_graph_search(
        self,
        query: str,
        query_fact_scores: np.ndarray,
        top_k_facts: List[Tuple],
        top_k_fact_indices: List[int],
        hop_override: Optional[int] = None,
        defer_qd: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        if getattr(self.config, "stage_ablation", "full") == "wo_route":
            return baseline_graph_search(
                self, query, query_fact_scores, top_k_facts,
                top_k_fact_indices, hop_override,
            )
        num_nodes = len(self.graph.vs["name"])
        phrase_weights = np.zeros(num_nodes, dtype=np.float32)
        passage_weights = np.zeros(num_nodes, dtype=np.float32)
        if (
            isinstance(hop_override, bool)
            or not isinstance(hop_override, Integral)
            or (not 1 <= hop_override <= 4)
        ):
            raise ValueError("hop_override must be an integer from 1 through 4")
        hops = int(hop_override)
        seed_entities = self._extract_seed_entities_from_facts(top_k_facts)
        seed_distribution = self._build_seed_distribution(query, seed_entities, hops)
        fact_seed_distribution = {}
        for fact_idx, fact in zip(top_k_fact_indices, top_k_facts):
            relevance = max(0.0, float(query_fact_scores[int(fact_idx)]))
            for entity in self._extract_seed_entities_from_facts([fact]):
                if entity in seed_distribution:
                    degree = max(1, len(self.ent_node_to_chunk_ids.get(entity, [])))
                    fact_seed_distribution[entity] = fact_seed_distribution.get(
                        entity, 0.0
                    ) + relevance / math.sqrt(degree)
        total = sum(fact_seed_distribution.values())
        if total > 0:
            fact_seed_distribution = {
                k: v / total for k, v in fact_seed_distribution.items()
            }
            seed_distribution = {
                k: 0.5 * v + 0.5 * fact_seed_distribution.get(k, 0.0)
                for k, v in seed_distribution.items()
            }
        for ent, prob in seed_distribution.items():
            v_idx = self.node_name_to_vertex_idx.get(ent)
            if v_idx is not None:
                phrase_weights[v_idx] += prob
        bridges_by_seed = self._detect_bridge_entities(query, seed_entities)
        bridge_count = 0
        for seed, bridge_entities in sorted(bridges_by_seed.items()):
            seed_w = seed_distribution.get(seed, 0.0)
            for b in bridge_entities:
                b_idx = self.node_name_to_vertex_idx.get(b)
                if b_idx is None:
                    continue
                else:
                    bridge_weight = self.config.route.bridge.weight * max(
                        seed_w, 1e-05
                    )
                phrase_weights[b_idx] += bridge_weight
                bridge_count += 1
        dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.dense_passage_retrieval(query)
        dpr_norm = min_max_normalize(dpr_sorted_doc_scores)
        for i, doc_id in enumerate(dpr_sorted_doc_ids.tolist()):
            if doc_id >= len(self.passage_node_keys):
                continue
            pkey = self.passage_node_keys[doc_id]
            pidx = self.node_name_to_vertex_idx.get(pkey)
            if pidx is None:
                continue
            base_w = float(dpr_norm[i]) * self.config.passage_node_weight
            passage_weights[pidx] = base_w
        node_weights = phrase_weights + passage_weights
        if float(np.sum(node_weights)) <= 0:
            return (
                dpr_sorted_doc_ids,
                dpr_sorted_doc_scores,
                {
                    "hops": hops,
                    "seed_entities": list(seed_entities),
                    "seed_distribution": seed_distribution,
                    "bridges_by_seed": bridges_by_seed,
                    "bridge_count": bridge_count,
                    "iterative_used": False,
                    "iterative_round2_seed_count": 0,
                },
            )
        damping, _ = self._qcappr_params(hops)
        ppr_sorted_doc_ids, ppr_sorted_doc_scores = self.run_ppr(
            node_weights, damping=damping
        )
        ctx = {
            "hops": hops,
            "seed_entities": list(seed_entities),
            "seed_distribution": seed_distribution,
            "fact_seed_distribution": fact_seed_distribution,
            "bridges_by_seed": bridges_by_seed,
            "bridge_count": bridge_count,
            "iterative_used": False,
            "iterative_round2_seed_count": 0,
            "qd_used": False,
            "qd_sub_questions": 0,
            "qd_sub_query_ppr_used": 0,
            "qd_sub_query_ppr_failed": 0,
            "qd_seq_rewrites": 0,
            "pcqd_used": False,
            "pcqd_sub_questions": 0,
            "pcqd_path_hints": 0,
            "pcqd_entity_replacement_num": 0.0,
            "pcqd_entity_replacement_den": 0.0,
            "pcqd_entity_replacement_rate": 0.0,
            "pcqd_fallback": False,
            "pcqd_conflict_rate": 0.0,
        }
        current_ids, current_scores = (ppr_sorted_doc_ids, ppr_sorted_doc_scores)
        current_ids, current_scores, iter_stats = self._iterative_round2_search(
            query=query,
            round1_sorted_doc_ids=current_ids,
            round1_sorted_doc_scores=current_scores,
            hops=hops,
            original_seed_entities=seed_entities,
        )
        ctx.update(iter_stats)
        if defer_qd:
            ctx["_qd_deferred"] = True
        else:
            (
                current_ids,
                current_scores,
                qd_stats,
            ) = self._query_decomposition_retrieval(
                query=query,
                hops=hops,
                base_sorted_doc_ids=current_ids,
                base_sorted_doc_scores=current_scores,
                seed_entities=seed_entities,
                seed_distribution=seed_distribution,
                bridges_by_seed=bridges_by_seed,
            )
            ctx.update(qd_stats)
        return (current_ids, current_scores, ctx)


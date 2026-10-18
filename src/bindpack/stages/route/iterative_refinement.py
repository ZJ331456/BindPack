from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

import numpy as np

from ...utils.misc_utils import min_max_normalize


class IterativeRefinement:
    """IterativeRefinement implementation for candidate discovery."""

    def _collect_entities_from_top_docs(self, doc_ids: List[int]) -> Set[str]:
        entities: Set[str] = set()
        for doc_id in doc_ids:
            if doc_id >= len(self.passage_node_keys):
                continue
            pkey = self.passage_node_keys[doc_id]
            entities |= self.chunk_to_entities.get(pkey, set())
        return entities

    @staticmethod
    def _merge_rankings(
        num_passages: int,
        base_sorted_ids: np.ndarray,
        base_sorted_scores: np.ndarray,
        extra_sorted_ids: np.ndarray,
        extra_sorted_scores: np.ndarray,
        alpha: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        alpha = min(1.0, max(0.0, float(alpha)))
        base_full = np.zeros(num_passages, dtype=np.float32)
        extra_full = np.zeros(num_passages, dtype=np.float32)

        base_norm = np.asarray(min_max_normalize(base_sorted_scores), dtype=np.float32)
        extra_norm = np.asarray(
            min_max_normalize(extra_sorted_scores), dtype=np.float32
        )
        base_full[base_sorted_ids] = base_norm
        extra_full[extra_sorted_ids] = extra_norm

        merged = (1.0 - alpha) * base_full + alpha * extra_full
        merged_sorted_ids = np.argsort(merged)[::-1]
        merged_sorted_scores = merged[merged_sorted_ids]
        return merged_sorted_ids, merged_sorted_scores

    def _select_iterative_bridge_seeds(
        self,
        query: str,
        candidate_entities: Set[str],
        original_seed_entities: Set[str],
    ) -> List[Tuple[str, float]]:
        """Rank novel bridge entities by IDF with a stable entity-key tie break."""
        scored: List[Tuple[str, float]] = []
        for ent_key in sorted(candidate_entities):
            if ent_key in original_seed_entities:
                continue
            local_idx = self.entity_key_to_local_idx.get(ent_key)
            if local_idx is None or local_idx >= len(self.entity_embeddings):
                continue
            idf = (
                float(self.entity_idf.get(ent_key, 1.0))
                if self.config.route.index.entity_idf
                else 1.0
            )
            scored.append((ent_key, idf))

        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[: self.config.route.iterative.seed_top_k]

    def _iterative_round2_search(
        self,
        query: str,
        round1_sorted_doc_ids: np.ndarray,
        round1_sorted_doc_scores: np.ndarray,
        hops: int,
        original_seed_entities: Set[str],
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Focused round-2 retrieval seeded by high-IDF novel bridge entities.

        Key change vs. original: instead of taking ALL entities from top-5 docs
        (avg ~40 seeds), we take only the top-k highest-IDF entities from the
        top-1 doc that are NOT already in the query's seed set.  This gives a
        tight, specific signal pointing at the second hop document.
        """
        top_doc_ids = round1_sorted_doc_ids[
            : self.config.route.iterative.round1_top_docs
        ].tolist()
        candidate_entities = self._collect_entities_from_top_docs(top_doc_ids)
        scored_seeds = self._select_iterative_bridge_seeds(
            query=query,
            candidate_entities=candidate_entities,
            original_seed_entities=original_seed_entities,
        )

        if len(scored_seeds) < self.config.route.iterative.min_seed_entities:
            return (
                round1_sorted_doc_ids,
                round1_sorted_doc_scores,
                {
                    "iterative_used": False,
                    "iterative_round2_seed_count": int(len(scored_seeds)),
                },
            )

        seed_distribution = self._softmax(dict(scored_seeds), temperature=0.7)

        num_nodes = len(self.graph.vs["name"])
        phrase_weights = np.zeros(num_nodes, dtype=np.float32)
        passage_weights = np.zeros(num_nodes, dtype=np.float32)
        for ent_key, prob in seed_distribution.items():
            v_idx = self.node_name_to_vertex_idx.get(ent_key)
            if v_idx is not None:
                phrase_weights[v_idx] += float(prob)

        round1_norm = min_max_normalize(round1_sorted_doc_scores)
        for i, doc_id in enumerate(round1_sorted_doc_ids.tolist()):
            if doc_id >= len(self.passage_node_keys):
                continue
            pkey = self.passage_node_keys[doc_id]
            pidx = self.node_name_to_vertex_idx.get(pkey)
            if pidx is None:
                continue
            passage_weights[pidx] = (
                float(round1_norm[i]) * self.config.passage_node_weight
            )

        node_weights = phrase_weights + passage_weights
        if float(np.sum(node_weights)) <= 0:
            return (
                round1_sorted_doc_ids,
                round1_sorted_doc_scores,
                {
                    "iterative_used": False,
                    "iterative_round2_seed_count": int(len(seed_distribution)),
                },
            )

        if self.config.route.iterative.damping > 0:
            damping = self.config.route.iterative.damping
        else:
            damping, _ = self._qcappr_params(hops)
        round2_sorted_ids, round2_sorted_scores = self.run_ppr(
            node_weights, damping=damping
        )

        merged_ids, merged_scores = self._merge_rankings(
            num_passages=len(self.passage_node_keys),
            base_sorted_ids=round1_sorted_doc_ids,
            base_sorted_scores=round1_sorted_doc_scores,
            extra_sorted_ids=round2_sorted_ids,
            extra_sorted_scores=round2_sorted_scores,
            alpha=self.config.route.iterative.merge_alpha,
        )

        return (
            merged_ids,
            merged_scores,
            {
                "iterative_used": True,
                "iterative_round2_seed_count": int(len(seed_distribution)),
            },
        )


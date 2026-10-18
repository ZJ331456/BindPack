from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np

from ...utils.misc_utils import compute_mdhash_id


class QuerySeedPrior:
    """QuerySeedPrior implementation for candidate discovery."""

    def _qcappr_params(self, hops: int) -> Tuple[float, float]:
        if hops <= 1:
            return (
                self.config.route.seed.single_hop_damping,
                self.config.route.seed.single_hop_temperature,
            )
        if hops == 2:
            return (
                self.config.route.seed.two_hop_damping,
                self.config.route.seed.two_hop_temperature,
            )
        return (
            self.config.route.seed.multi_hop_damping,
            self.config.route.seed.multi_hop_temperature,
        )

    @staticmethod
    def _softmax(scores: Dict[str, float], temperature: float) -> Dict[str, float]:
        if not scores:
            return {}
        t = max(1e-4, float(temperature))
        keys = list(scores.keys())
        arr = np.array([scores[k] for k in keys], dtype=np.float32) / t
        arr -= np.max(arr)
        exp = np.exp(arr)
        s = float(np.sum(exp)) + 1e-12
        return {k: float(v / s) for k, v in zip(keys, exp)}

    def _extract_seed_entities_from_facts(self, top_k_facts: List[Tuple]) -> Set[str]:
        seeds: Set[str] = set()
        for fact in top_k_facts:
            if len(fact) < 3:
                continue
            subj = compute_mdhash_id(content=str(fact[0]).lower(), prefix="entity-")
            obj = compute_mdhash_id(content=str(fact[2]).lower(), prefix="entity-")
            if subj in self.node_name_to_vertex_idx:
                seeds.add(subj)
            if obj in self.node_name_to_vertex_idx:
                seeds.add(obj)
        return seeds

    def _build_seed_distribution(
        self, query: str, seed_entities: Set[str], hops: int
    ) -> Dict[str, float]:
        if not seed_entities:
            return {}

        query_emb = self.query_to_embedding["triple"].get(query)
        if query_emb is None:
            return {}

        raw_scores: Dict[str, float] = {}
        for ent in sorted(seed_entities):
            local_idx = self.entity_key_to_local_idx.get(ent)
            if local_idx is None or local_idx >= len(self.entity_embeddings):
                continue

            ent_emb = self.entity_embeddings[local_idx]
            sim = float(np.dot(ent_emb, query_emb))

            # index-side IDF prior innovation
            if self.config.route.index.entity_idf:
                sim *= float(self.entity_idf.get(ent, 1.0))

            # query-conditioned hub penalty
            v_idx = self.node_name_to_vertex_idx.get(ent)
            if v_idx is not None:
                deg = float(self.graph.degree(v_idx))
                sim *= 1.0 / (
                    (1.0 + deg) ** self.config.route.seed.hub_penalty_gamma
                )

            raw_scores[ent] = sim

        _, temp = self._qcappr_params(hops)
        return self._softmax(raw_scores, temp)


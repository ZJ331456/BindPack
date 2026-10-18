from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np



class BridgeExpansion:
    """BridgeExpansion implementation for candidate discovery."""

    def _rank_bridge_neighbors(self, entity_key: str) -> List[str]:
        v_idx = self.node_name_to_vertex_idx.get(entity_key)
        if v_idx is None:
            return []

        neighbors = self.graph.neighbors(v_idx)
        cands: List[Tuple[str, float]] = []
        for nb_idx in neighbors:
            nb_name = self.graph.vs[nb_idx]["name"]
            if not isinstance(nb_name, str) or not nb_name.startswith("entity-"):
                continue
            deg = float(self.graph.degree(nb_idx))
            idf = float(self.entity_idf.get(nb_name, 1.0))
            score = idf / (1.0 + deg)
            cands.append((nb_name, score))

        cands.sort(key=lambda x: (-x[1], x[0]))
        return [x[0] for x in cands[: self.config.route.index.bridge_cache_top_k]]

    def _detect_bridge_entities(
        self, query: str, seed_entities: Set[str]
    ) -> Dict[str, List[str]]:
        """
        Enhanced bridge detection with semantic filtering.
        Only keep bridge entities that are:
        1. In first-hop DPR passages (structural signal)
        2. Semantically relevant to query (semantic signal)
        """
        if not seed_entities:
            return {}

        dpr_doc_ids, _ = self.dense_passage_retrieval(query)
        first_hop_doc_ids = dpr_doc_ids[: self.config.route.bridge.first_hop_k].tolist()
        first_hop_passage_keys = [
            self.passage_node_keys[i]
            for i in first_hop_doc_ids
            if i < len(self.passage_node_keys)
        ]

        first_hop_entities: Set[str] = set()
        for pkey in first_hop_passage_keys:
            first_hop_entities |= self.chunk_to_entities.get(pkey, set())

        # Get query embedding for semantic filtering
        query_emb = self.query_to_embedding["triple"].get(query)

        bridges: Dict[str, List[str]] = {}
        for seed in sorted(seed_entities):
            if seed in self.bridge_neighbor_cache:
                ranked_nb = self.bridge_neighbor_cache[seed]
            else:
                ranked_nb = self._rank_bridge_neighbors(seed)
                if self.config.route.index.bridge_cache:
                    self.bridge_neighbor_cache[seed] = ranked_nb

            filtered = []
            for nb in ranked_nb:
                if nb not in first_hop_entities:
                    continue
                if query_emb is not None:
                    nb_idx = self.entity_key_to_local_idx.get(nb)
                    if nb_idx is not None and nb_idx < len(self.entity_embeddings):
                        nb_emb = self.entity_embeddings[nb_idx]
                        sim = float(np.dot(nb_emb, query_emb))
                        if sim < self.config.route.bridge.semantic_threshold:
                            continue
                filtered.append(nb)
            keep = filtered[: self.config.route.bridge.top_k]

            if keep:
                bridges[seed] = keep

        return bridges


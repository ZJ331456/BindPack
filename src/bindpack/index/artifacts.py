"""Index side entity priors and bridge caches, shared with retrieval."""
from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from typing import Dict, List, Set

logger = logging.getLogger(__name__)


class IndexArtifactsMixin:
    """Build derived lookup tables without changing the persisted index format."""

    def index(self, docs: List[str], *, rebuild_graph: bool = False,
              retry_failed_openie: bool = True):
        super().index(docs, rebuild_graph=rebuild_graph,
                      retry_failed_openie=retry_failed_openie)

        if not (
            self.pcrag_config.use_entity_idf_index
            or self.pcrag_config.use_bridge_cache_index
        ):
            return

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        self._build_index_side_artifacts()


    def _build_index_side_artifacts(self):
        if self.ent_node_to_chunk_ids is None:
            logger.warning("ent_node_to_chunk_ids is empty; skipping pcrag index artifacts.")
            return

        self.entity_key_to_local_idx = {k: i for i, k in enumerate(self.entity_node_keys)}
        self.passage_key_to_local_idx = {k: i for i, k in enumerate(self.passage_node_keys)}

        # Build chunk -> entities reverse map (index innovation #1)
        chunk_to_entities: Dict[str, Set[str]] = defaultdict(set)
        for ent_key, chunk_ids in self.ent_node_to_chunk_ids.items():
            for cid in chunk_ids:
                chunk_to_entities[cid].add(ent_key)
        self.chunk_to_entities = chunk_to_entities

        # Build entity idf prior (index innovation #2)
        if self.pcrag_config.use_entity_idf_index:
            num_chunks = max(1, len(self.passage_node_keys))
            entity_idf: Dict[str, float] = {}
            for ent_key, chunk_ids in self.ent_node_to_chunk_ids.items():
                df = max(1, len(chunk_ids))
                entity_idf[ent_key] = float(math.log((num_chunks + 1) / (df + 1)) + 1.0)
            self.entity_idf = entity_idf

        # Precompute bridge neighbor cache (optional, can be expensive)
        self.bridge_neighbor_cache = {}
        if self.pcrag_config.use_bridge_cache_index:
            limit = min(self.pcrag_config.bridge_cache_entity_limit, len(self.entity_node_keys))
            for ent_key in self.entity_node_keys[:limit]:
                self.bridge_neighbor_cache[ent_key] = self._rank_bridge_neighbors(ent_key)


        payload = {
            "num_entities": len(self.entity_node_keys),
            "num_passages": len(self.passage_node_keys),
            "chunk_to_entities_size": len(self.chunk_to_entities),
            "entity_idf_size": len(self.entity_idf),
            "bridge_cache_size": len(self.bridge_neighbor_cache),
            "config": {
                "use_entity_idf_index": self.pcrag_config.use_entity_idf_index,
                "use_bridge_cache_index": self.pcrag_config.use_bridge_cache_index,
            },
        }
        with open(self._pcrag_artifacts_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        logger.info("BindPack index artifacts built: chunk_to_entities=%d, entity_idf=%d, bridge_cache=%d", len(self.chunk_to_entities), len(self.entity_idf), len(self.bridge_neighbor_cache))


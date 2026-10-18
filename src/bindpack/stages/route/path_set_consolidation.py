from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

import numpy as np

from ...utils.misc_utils import min_max_normalize

from .path_set_selection import (
    PathCandidate, aggregate_path_set_to_passage_scores,
    greedy_path_set_selection, normalize_candidate_scores,
)


class PathSetConsolidation:
    """PathSetConsolidation implementation for candidate discovery."""

    def _apply_path_set_optimization(
        self,
        query: str,
        base_sorted_doc_ids: np.ndarray,
        base_sorted_doc_scores: np.ndarray,
        ctx: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
        top_doc_ids = base_sorted_doc_ids[
            : self.config.route.paths.candidate_docs
        ].tolist()
        top_doc_scores = min_max_normalize(
            base_sorted_doc_scores[: self.config.route.paths.candidate_docs]
        )

        seed_entities: Set[str] = set(ctx.get("seed_entities", []))
        if not seed_entities:
            return (
                base_sorted_doc_ids,
                base_sorted_doc_scores,
                {
                    "path_candidates": 0,
                    "selected_paths": 0,
                    "selected_path_candidates": [],
                },
            )

        seed_distribution: Dict[str, float] = dict(ctx.get("seed_distribution", {}))
        bridges_by_seed: Dict[str, List[str]] = dict(ctx.get("bridges_by_seed", {}))

        candidates: List[PathCandidate] = []
        for rank, (doc_local_idx, doc_score) in enumerate(
            zip(top_doc_ids, top_doc_scores)
        ):
            if doc_local_idx >= len(self.passage_node_keys):
                continue
            pkey = self.passage_node_keys[doc_local_idx]
            doc_entities = self.chunk_to_entities.get(pkey, set())
            if not doc_entities:
                continue

            # Direct two-node path: seed -> passage
            for seed in sorted(seed_entities):
                if seed not in doc_entities:
                    continue

                rel = float(doc_score)
                conn = 1.0
                cons = float(seed_distribution.get(seed, 0.0))
                comp = len({seed} & seed_entities) / max(1, len(seed_entities))

                candidates.append(
                    PathCandidate(
                        nodes=[seed, pkey],
                        passage_key=pkey,
                        covered_entities={seed},
                        score_relevance=rel,
                        score_connectivity=conn,
                        score_consistency=cons,
                        score_completeness=comp,
                        score_total=0.0,
                        metadata={"type": "direct", "rank": rank},
                    )
                )

            # Three-node path: seed -> bridge -> passage
            for seed, bridge_entities in sorted(bridges_by_seed.items()):
                for b in bridge_entities:
                    if b not in doc_entities:
                        continue

                    rel = float(doc_score)
                    conn = 1.2
                    seed_prob = float(seed_distribution.get(seed, 0.0))
                    idf_b = float(self.entity_idf.get(b, 1.0))
                    cons = 0.5 * seed_prob + 0.5 * idf_b
                    covered = {seed, b} & seed_entities
                    comp = len(covered) / max(1, len(seed_entities))

                    candidates.append(
                        PathCandidate(
                            nodes=[seed, b, pkey],
                            passage_key=pkey,
                            covered_entities=covered if covered else {seed},
                            score_relevance=rel,
                            score_connectivity=conn,
                            score_consistency=cons,
                            score_completeness=comp,
                            score_total=0.0,
                            metadata={"type": "bridge", "rank": rank},
                        )
                    )

        if not candidates:
            return (
                base_sorted_doc_ids,
                base_sorted_doc_scores,
                {
                    "path_candidates": 0,
                    "selected_paths": 0,
                    "selected_path_candidates": [],
                },
            )

        # Normalize components and compute total score.
        candidates = normalize_candidate_scores(candidates)
        for c in candidates:
            c.score_total = (
                self.config.route.paths.relevance_weight * c.score_relevance
                + self.config.route.paths.connectivity_weight
                * c.score_connectivity
                + self.config.route.paths.consistency_weight * c.score_consistency
                + self.config.route.paths.completeness_weight
                * c.score_completeness
            )

        selected = greedy_path_set_selection(
            candidates=candidates,
            target_entities=seed_entities,
            set_size=self.config.route.paths.set_size,
            diversity_lambda=self.config.route.paths.diversity_lambda,
        )

        if not selected:
            return (
                base_sorted_doc_ids,
                base_sorted_doc_scores,
                {
                    "path_candidates": len(candidates),
                    "selected_paths": 0,
                    "selected_path_candidates": [],
                },
            )

        path_passage_scores = aggregate_path_set_to_passage_scores(selected)

        # Merge path-set score back to global passage ranking.
        full_ppr_scores = np.zeros(len(self.passage_node_keys), dtype=np.float32)
        full_ppr_scores[base_sorted_doc_ids] = base_sorted_doc_scores
        full_ppr_scores = np.asarray(
            min_max_normalize(full_ppr_scores), dtype=np.float32
        )

        for pkey, pscore in path_passage_scores.items():
            pidx = self.passage_key_to_local_idx.get(pkey)
            if pidx is None:
                continue
            full_ppr_scores[pidx] = 0.7 * full_ppr_scores[pidx] + 0.3 * float(pscore)

        final_sorted_ids = np.argsort(full_ppr_scores)[::-1]
        final_sorted_scores = full_ppr_scores[final_sorted_ids]

        selected_meta: List[Dict[str, Any]] = []
        for p in selected:
            pkey = str(p.passage_key)
            covered = sorted(self.chunk_to_entities.get(pkey, set()))
            selected_meta.append(
                {
                    "passage_key": pkey,
                    "covered_entities": covered,
                    "score_total": float(p.score_total),
                    "score_relevance": float(p.score_relevance),
                }
            )

        return (
            final_sorted_ids,
            final_sorted_scores,
            {
                "path_candidates": len(candidates),
                "selected_paths": len(selected),
                "selected_path_candidates": selected_meta,
            },
        )


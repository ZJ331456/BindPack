"""Coverage/redundancy greedy prefix and pointwise Pack ablation."""
from __future__ import annotations

from typing import Tuple
import numpy as np
from ...utils.evidence import normalized_text
from .redundancy_controller import max_token_redundancy


class CoverageSelector:
    """Preserve the evaluated greedy objective and its stable tie breaks."""

    def __init__(self, runtime, scorer):
        self.runtime = runtime
        self.scorer = scorer
        self.config = runtime.cfg.pack

    def select(self, query: str, ids: np.ndarray, scores: np.ndarray,
                 ctx: dict, state: dict) -> Tuple[np.ndarray, np.ndarray, dict]:
        trace = state.get("evidence_trace", self.runtime._trace())
        candidates = state.get("evidence_candidates", {})
        if not candidates:
            return ids, scores, trace
        ids, scores = np.asarray(ids, dtype=int), np.asarray(scores, dtype=float)
        finite = np.where(np.isfinite(scores), scores, 0.0)
        span = float(finite.max() - finite.min()) if len(finite) else 0.0
        norm = (finite - finite.min()) / span if span > 0 else np.ones(len(finite))
        base_by_id = dict(zip(ids.tolist(), norm.tolist()))
        base_weight = float(self.config.base_weight)
        merged = {doc_id: base_weight * value for doc_id, value in base_by_id.items()}
        bindings = state.get("_evidence_winning_bindings", {})
        goal_by_id = {}
        for doc_id, candidate in candidates.items():
            goals = self.scorer.goal_scores(candidate, bindings)
            goal_by_id[doc_id] = goals
            relation = max(goals.values(), default=0.0)
            merged[doc_id] = base_weight * base_by_id.get(doc_id, candidate["base_score"]) + (1.0 - base_weight) * relation
        ordered = sorted(merged, key=lambda d: (-merged[d], d))
        if self.runtime.cfg.stage_ablation == "wo_pack":
            # Freeze the Route + Plan pool and reuse the original base/goal
            # fusion exactly, before any coverage or redundancy selection.
            trace["pack_mode"] = "pointwise_base_goal"
            trace["selected_prefix"] = [
                {"doc_id": d, "selection_source": "pointwise", "score": float(merged[d]),
                 "goals": goal_by_id.get(d, {}),
                 "verified": candidates.get(d, {}).get("verified", [])}
                for d in ordered[:self.config.budget]
            ]
            trace["covered_goals"] = {}
            return (np.asarray(ordered, dtype=int),
                    np.asarray([merged[d] for d in ordered], dtype=float), trace)
        covered, selected, selected_tokens, details = {}, [], [], []
        remaining = set(candidates)
        goal_ids = set(goal for goals in goal_by_id.values() for goal in goals)
        goal_total = max(1, len(goal_ids))
        coverage_weight = float(self.config.coverage_weight)
        relation_weight = float(self.config.relation_weight)
        redundancy_weight = float(self.config.redundancy_weight)
        document_tokens = {d: set(normalized_text(self.runtime._document(d)).split()) for d in remaining}
        for position in range(min(self.config.budget, len(remaining))):
            ranked = []
            for doc_id in remaining:
                goals = goal_by_id[doc_id]
                gain = sum(max(0.0, score - covered.get(goal, 0.0)) for goal, score in goals.items()) / goal_total
                tokens = document_tokens[doc_id]
                redundancy = max_token_redundancy(tokens, selected_tokens)
                relation = max(goals.values(), default=0.0)
                utility = (base_weight * base_by_id.get(doc_id, candidates[doc_id]["base_score"])
                           + relation_weight * relation + coverage_weight * gain
                           - redundancy_weight * redundancy)
                ranked.append((utility, merged[doc_id], -doc_id, gain, redundancy, doc_id))
            utility, _score, _tie, gain, redundancy, chosen = max(ranked)
            selected.append(chosen)
            selected_tokens.append(document_tokens[chosen])
            remaining.remove(chosen)
            for goal, score in goal_by_id[chosen].items():
                covered[goal] = max(covered.get(goal, 0.0), score)
            details.append({"doc_id": chosen, "utility": float(utility),
                            "coverage_gain": float(gain), "redundancy": float(redundancy),
                            "goals": goal_by_id[chosen], "verified": candidates[chosen]["verified"]})
        # Keep greedy marginal-utility order as the actual retrieval prefix.
        # Tail scores encode the tail's existing order below the selected prefix.
        tail = [d for d in ordered if d not in set(selected)]
        final_ids = selected + tail
        top_score = max(merged.values(), default=1.0) + 1.0
        prefix_scores = [top_score + (len(selected) - position) / max(1, len(selected))
                         for position in range(len(selected))]
        final_scores = prefix_scores + [merged[d] for d in tail]
        trace["selected_prefix"] = details
        trace["covered_goals"] = covered
        return np.asarray(final_ids), np.asarray(final_scores, dtype=float), trace

"""Branch-compatible base and subgoal evidence scores."""
from __future__ import annotations

from typing import Dict
from ...utils.evidence import branch_matches


class EvidenceScorer:
    """Score only goals compatible with the winning answer bindings."""

    @staticmethod
    def goal_scores(candidate: dict, bindings: dict) -> Dict[str, float]:
        goals: Dict[str, float] = {}
        for source in candidate["sources"]:
            if not branch_matches(source["requirements"], bindings):
                continue
            goal = source["goal"]
            goals[goal] = max(goals.get(goal, 0.0), source["score"])
        for proof in candidate["verified"]:
            goals[proof["goal"]] = max(goals.get(proof["goal"], 0.0), proof["quality"])
        return goals

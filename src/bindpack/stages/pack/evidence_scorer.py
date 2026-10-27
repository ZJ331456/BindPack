"""Branch-compatible base and subgoal evidence scores."""
from __future__ import annotations

from typing import Dict
from functools import partial
from ...utils.evidence import branch_matches


class EvidenceScorer:
    """Score only goals compatible with the winning answer bindings."""

    def __init__(self, source_bound_eligibility=True):
        self.source_bound_eligibility = source_bound_eligibility
        if not source_bound_eligibility:
            self.goal_scores = partial(type(self).goal_scores, source_bound_eligibility=False)

    @staticmethod
    def goal_scores(candidate: dict, bindings: dict, *, source_bound_eligibility=True) -> Dict[str, float]:
        goals: Dict[str, float] = {}
        for source in candidate["sources"]:
            if source_bound_eligibility and not branch_matches(source["requirements"], bindings):
                continue
            goal = source["goal"]
            goals[goal] = max(goals.get(goal, 0.0), source["score"])
        for proof in candidate["verified"]:
            goals[proof["goal"]] = max(goals.get(proof["goal"], 0.0), proof["quality"])
        return goals

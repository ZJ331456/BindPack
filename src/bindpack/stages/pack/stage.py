"""Pack stage: score goals, select coverage, consolidate DAG packages and review."""
from __future__ import annotations

from .evidence_scorer import EvidenceScorer
from .coverage_selector import CoverageSelector
from .dag_package_selector import DAGPackageSelector
from .verification.support_semantic_veto import postprocess_support_semantic_veto
from ..mechanisms import mechanisms


class PackStage:
    """Compose the evaluated ranking objective and source-protected final review."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.mechanisms = mechanisms(runtime.cfg)
        self.scorer = EvidenceScorer(self.mechanisms.source_bound_eligibility)
        self.coverage = CoverageSelector(runtime, self.scorer)
        self.dag_selector = DAGPackageSelector(runtime._document,self.mechanisms.source_bound_eligibility)

    def finalize(self, query, ids, scores, context, state):
        result = self.coverage.select(query, ids, scores, context, state)
        if self.runtime.cfg.stage_ablation == "wo_pack":
            return result
        if self.runtime.cfg.stage_ablation != "wo_plan" and self.mechanisms.dependency_closure:
            result = self.dag_selector.select(result, state)
        if self.runtime.cfg.pack.selector_mode != "current" and self.mechanisms.joint_selection:
            from .budgeted_selector import BudgetedSelector
            result = BudgetedSelector(self.runtime, self.scorer).select(query, ids, scores, result, state)
        return result

    def postprocess(self, inputs):
        if self.runtime.cfg.stage_ablation == "wo_pack":
            return {}
        return postprocess_support_semantic_veto(self.runtime, inputs)

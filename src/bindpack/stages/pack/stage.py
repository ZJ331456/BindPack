"""Pack stage: score goals, select coverage, consolidate DAG packages and review."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import numpy as np

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

    def build_reference(self, query, ids, scores, context, state):
        """Build the shared S0, before any bounded-search intervention."""
        result = self.coverage.select(query, ids, scores, context, state)
        if self.runtime.cfg.stage_ablation == "wo_pack":
            return result
        if self.runtime.cfg.stage_ablation != "wo_plan" and self.mechanisms.dependency_closure:
            result = self.dag_selector.select(result, state)
        return result

    def finalize(self, query, ids, scores, context, state):
        # The reference may be frozen by the experiment runner. Its ranking is
        # public input to every Pack intervention, not a source-binding record.
        reference = getattr(self.runtime, "_pack_reference", None)
        if reference is None:
            reference = state.get("_pack_reference")
        result = (deepcopy(reference) if reference is not None else
                  self.build_reference(query, ids, scores, context, state))
        capture = getattr(self.runtime, "_capture_pack_reference", None)
        if capture is not None:
            capture(deepcopy(result))
        if self.runtime.cfg.stage_ablation == "wo_pack":
            return result
        if not self.mechanisms.source_binding:
            from .unbound_support import unbound_pack_state
            state = unbound_pack_state(state)
            clean_trace = deepcopy(state["evidence_trace"])
            clean_trace["reference_doc_ids"] = [int(d) for d in result[0][:self.runtime.cfg.pack.budget]]
            result = (np.asarray(result[0]), np.asarray(result[1]), clean_trace)
        if self.runtime.cfg.mechanism_ablation != "full" or self.runtime.cfg.pack.selector_mode != "current":
            result = (result[0], result[1], dict(result[2],
                mechanism_ablation=self.runtime.cfg.mechanism_ablation,
                pack_mechanism_ablation=self.runtime.cfg.pack_mechanism_ablation,
                mechanisms=asdict(self.mechanisms)))
        if not self.mechanisms.joint_selection:
            # Reference(S0) returns before search or reader-visible rescoring.
            result[2]["reference_doc_ids"] = [int(d) for d in result[0][:self.runtime.cfg.pack.budget]]
            result[2]["reference_only"] = True
            return result
        if self.runtime.cfg.pack.selector_mode != "current" and self.mechanisms.joint_selection:
            from .budgeted_selector import BudgetedSelector
            result = BudgetedSelector(self.runtime, self.scorer).select(query, ids, scores, result, state)
        return result

    def postprocess(self, inputs):
        if self.runtime.cfg.stage_ablation == "wo_pack":
            return {}
        return postprocess_support_semantic_veto(self.runtime, inputs)

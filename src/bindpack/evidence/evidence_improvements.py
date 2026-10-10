"""Compose the evaluated planner, dependency retrieval and closed DAG packages."""

from .planning.evidence_planning import PlannerImprovementMixin
from .selection.evidence_dag_package import DAGPackageMixin
from .evidence_retrieval import EvidenceRetrieval


class ImprovedEvidenceRetrieval(
    PlannerImprovementMixin,
    DAGPackageMixin,
    EvidenceRetrieval,
):
    """Final method; its four flags support disabling existing components."""

    def __init__(self, rag):
        self.improvements = frozenset(
            flag.strip()
            for flag in rag.pcrag_config.evidence_improvements.split(",")
            if flag.strip()
        )
        super().__init__(rag)

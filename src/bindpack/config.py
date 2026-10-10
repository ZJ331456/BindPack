from dataclasses import dataclass, field

from .utils.config_utils import BaseConfig


@dataclass
class BindPackConfig(BaseConfig):
    """The verified stage-4 BindPack method and its existing component switches.

    Each query must supply a benchmark hop count or an explicit dataset prior.
    Removed experimental algorithms are not accepted as configuration options.
    """

    # Evidence candidates and the verified component switches.
    improvement_stage: int = 4
    evidence_candidate_top_k: int = 20
    evidence_pool_size: int = 80
    evidence_budget: int = 5
    evidence_max_searches: int = 20
    evidence_local_rank_constant: float = 10.0
    evidence_base_weight: float = 0.4
    evidence_coverage_weight: float = 0.45
    evidence_relation_weight: float = 0.25
    evidence_redundancy_weight: float = 0.15
    evidence_improvements: str = "dag_package,plan_prune,planning,support_semantic_veto"
    evidence_plan_node_budget: int = 6
    evidence_plan_depth_budget: int = 4
    evidence_plan_validation: str = "canonical_refs"
    evidence_plan_routing: str = "question_structure"
    evidence_review_tokenizer: str = "/root/models/Qwen3-8B"
    evidence_review_context_length: int = 8192

    # Existing graph retrieval, bridge expansion and path selection.
    use_qcappr: bool = field(default=True)
    use_eba: bool = field(default=True)
    use_path_set_optimization: bool = field(default=True)

    # Query-conditioned propagation.
    qcappr_hub_penalty_gamma: float = field(default=0.35)
    qcappr_single_hop_damping: float = field(default=0.85)
    qcappr_two_hop_damping: float = field(default=0.50)
    qcappr_multi_hop_damping: float = field(default=0.45)
    qcappr_single_hop_temperature: float = field(default=0.35)
    qcappr_two_hop_temperature: float = field(default=0.7)
    qcappr_multi_hop_temperature: float = field(default=1.0)

    # Bridge entities from the first-hop passages.
    eba_first_hop_k: int = field(default=20)
    eba_bridge_top_k: int = field(default=8)
    eba_bridge_weight: float = field(default=0.25)
    eba_bridge_semantic_threshold: float = field(default=0.1)

    # Path-set optimization.
    path_candidate_docs: int = field(default=80)
    path_set_size: int = field(default=12)
    path_score_relevance_weight: float = field(default=0.55)
    path_score_connectivity_weight: float = field(default=0.15)
    path_score_consistency_weight: float = field(default=0.15)
    path_score_completeness_weight: float = field(default=0.15)
    path_set_diversity_lambda: float = field(default=0.35)

    # Persistent graph index artifacts.
    use_entity_idf_index: bool = field(default=True)
    use_bridge_cache_index: bool = field(default=True)
    bridge_cache_top_k_per_entity: int = field(default=32)
    bridge_cache_entity_limit: int = field(default=120000)

    # Focused second-round retrieval with novel IDF seeds.
    use_iterative_retrieval: bool = field(default=True)
    iterative_round1_top_docs: int = field(default=1)
    iterative_round2_seed_top_k: int = field(default=5)
    iterative_merge_alpha: float = field(default=0.45)
    iterative_min_seed_entities: int = field(default=1)
    iterative_damping: float = field(default=-1.0)

    # Static query decomposition and bounded LLM concurrency.
    use_query_decomposition: bool = field(default=True)
    qd_min_hops: int = field(default=2)
    qd_max_sub_questions: int = field(default=4)
    qd_sub_retrieval_top_k: int = field(default=3)
    qd_merge_alpha: float = field(default=0.30)
    qd_llm_temperature: float = field(default=0.0)
    qd_cache_decompositions: bool = field(default=True)
    llm_prefetch_workers: int = field(default=8)

    # Source-grounded path-conditioned decomposition with fixed fusion.
    use_path_conditioned_qd: bool = field(default=True)
    pcqd_include_static_qd: bool = field(default=True)
    pcqd_fallback_to_static: bool = field(default=True)
    pcqd_ground_top_docs: int = field(default=5)
    pcqd_entity_top_k: int = field(default=3)
    pcqd_path_score_threshold: float = field(default=0.6)
    pcqd_enable_bridge_voting: bool = field(default=True)
    pcqd_max_bridges_per_source: int = field(default=1)
    pcqd_weight_base: float = field(default=0.4)
    pcqd_weight_static: float = field(default=0.2)
    pcqd_weight_path: float = field(default=0.4)

    def __post_init__(self):
        super().__post_init__()
        if self.improvement_stage != 4:
            raise ValueError("BindPack only supports the verified improvement_stage=4")
        flags = {
            flag.strip()
            for flag in self.evidence_improvements.split(",")
            if flag.strip()
        }
        allowed = {"planning", "plan_prune", "dag_package", "support_semantic_veto"}
        if flags - allowed:
            raise ValueError(
                f"Unsupported evidence improvements: {sorted(flags - allowed)}"
            )
        self.evidence_improvements = ",".join(sorted(flags))
        if not 1 <= int(self.evidence_plan_node_budget) <= 8:
            raise ValueError("evidence_plan_node_budget must be 1 through 8")
        if not 1 <= int(self.evidence_plan_depth_budget) <= 4:
            raise ValueError("evidence_plan_depth_budget must be 1 through 4")
        if self.evidence_plan_validation != "canonical_refs":
            raise ValueError(
                "The verified method requires evidence_plan_validation='canonical_refs'"
            )
        if self.evidence_plan_routing != "question_structure":
            raise ValueError(
                "The verified method requires evidence_plan_routing='question_structure'"
            )
        self.evidence_candidate_top_k = max(1, int(self.evidence_candidate_top_k))
        self.evidence_pool_size = max(5, int(self.evidence_pool_size))
        self.evidence_budget = min(5, max(1, int(self.evidence_budget)))
        self.evidence_max_searches = max(1, int(self.evidence_max_searches))
        self.qcappr_hub_penalty_gamma = max(0.0, float(self.qcappr_hub_penalty_gamma))
        self.eba_bridge_weight = max(0.0, float(self.eba_bridge_weight))
        self.path_set_size = max(1, int(self.path_set_size))
        self.path_candidate_docs = max(
            self.path_set_size, int(self.path_candidate_docs)
        )
        self.eba_first_hop_k = max(1, int(self.eba_first_hop_k))
        self.eba_bridge_top_k = max(1, int(self.eba_bridge_top_k))
        self.iterative_round1_top_docs = max(1, int(self.iterative_round1_top_docs))
        self.iterative_round2_seed_top_k = max(1, int(self.iterative_round2_seed_top_k))
        self.iterative_min_seed_entities = max(1, int(self.iterative_min_seed_entities))
        self.iterative_merge_alpha = min(
            1.0, max(0.0, float(self.iterative_merge_alpha))
        )
        self.bridge_cache_top_k_per_entity = max(
            1, int(self.bridge_cache_top_k_per_entity)
        )
        self.bridge_cache_entity_limit = max(1, int(self.bridge_cache_entity_limit))
        self.qd_min_hops = max(1, int(self.qd_min_hops))
        self.qd_max_sub_questions = max(1, int(self.qd_max_sub_questions))
        self.llm_prefetch_workers = min(8, max(1, int(self.llm_prefetch_workers)))
        self.qd_sub_retrieval_top_k = max(1, int(self.qd_sub_retrieval_top_k))
        self.qd_merge_alpha = min(1.0, max(0.0, float(self.qd_merge_alpha)))
        self.pcqd_ground_top_docs = max(1, int(self.pcqd_ground_top_docs))
        self.pcqd_entity_top_k = max(1, int(self.pcqd_entity_top_k))
        self.pcqd_max_bridges_per_source = max(1, int(self.pcqd_max_bridges_per_source))
        self.pcqd_path_score_threshold = min(
            1.0, max(0.0, float(self.pcqd_path_score_threshold))
        )
        self.pcqd_weight_base = max(0.0, float(self.pcqd_weight_base))
        self.pcqd_weight_static = max(0.0, float(self.pcqd_weight_static))
        self.pcqd_weight_path = max(0.0, float(self.pcqd_weight_path))


# Public aliases retained for callers of the original API.
PathCondRAGConfig = BindPackConfig
PCRAGConfig = BindPackConfig

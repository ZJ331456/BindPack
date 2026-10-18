"""Typed route parameters; values are loaded from defaults.json."""
from dataclasses import dataclass


@dataclass
class SeedConfig:
    hub_penalty_gamma: float
    single_hop_damping: float
    two_hop_damping: float
    multi_hop_damping: float
    single_hop_temperature: float
    two_hop_temperature: float
    multi_hop_temperature: float


@dataclass
class BridgeConfig:
    first_hop_k: int
    top_k: int
    weight: float
    semantic_threshold: float


@dataclass
class PathSetConfig:
    candidate_docs: int
    set_size: int
    relevance_weight: float
    connectivity_weight: float
    consistency_weight: float
    completeness_weight: float
    diversity_lambda: float


@dataclass
class RouteIndexConfig:
    entity_idf: bool
    bridge_cache: bool
    bridge_cache_top_k: int
    entity_limit: int


@dataclass
class IterativeConfig:
    round1_top_docs: int
    seed_top_k: int
    merge_alpha: float
    min_seed_entities: int
    damping: float


@dataclass
class PCQDConfig:
    include_static: bool
    fallback_to_static: bool
    ground_top_docs: int
    entity_top_k: int
    path_score_threshold: float
    bridge_voting: bool
    max_bridges_per_source: int
    weight_base: float
    weight_static: float
    weight_path: float


@dataclass
class RefinementConfig:
    min_hops: int
    max_sub_questions: int
    sub_retrieval_top_k: int
    merge_alpha: float
    temperature: float
    cache_decompositions: bool
    pcqd: PCQDConfig


@dataclass
class RouteConfig:
    seed: SeedConfig
    bridge: BridgeConfig
    paths: PathSetConfig
    index: RouteIndexConfig
    iterative: IterativeConfig
    refinement: RefinementConfig

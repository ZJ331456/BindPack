"""Type and range checks for executable method configurations."""
from dataclasses import fields, is_dataclass
import math
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints


def _matches(value, kind):
    origin = get_origin(kind)
    if kind is Any:
        return True
    if origin is Union:
        return any(_matches(value, choice) for choice in get_args(kind))
    if origin is Literal:
        return value in get_args(kind)
    if kind is float:
        return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
    if kind is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if kind is type(None):
        return value is None
    return isinstance(value, origin or kind)


def validate_types(config, location="config"):
    annotations = get_type_hints(type(config))
    for field in fields(config):
        value = getattr(config, field.name)
        name = f"{location}.{field.name}"
        if not _matches(value, annotations[field.name]):
            raise ValueError(f"Invalid type or non-finite value for {name}: {value!r}")
        if is_dataclass(value):
            validate_types(value, name)


def positive(value, name):
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def unit(value, name):
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be within 0..1")


def nonnegative(value, name):
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def validate_runtime(config):
    validate_types(config)
    for name in ("embedding_batch_size", "embedding_max_seq_len", "num_gen_choices",
                 "max_retry_attempts", "synonymy_edge_topk", "synonymy_edge_query_batch_size",
                 "synonymy_edge_key_batch_size", "retrieval_top_k",
                 "qa_top_k", "max_qa_steps", "llm_prefetch_workers"):
        positive(getattr(config, name), name)
    if config.max_new_tokens is not None:
        positive(config.max_new_tokens, "max_new_tokens")
    if not 0 < config.damping < 1:
        raise ValueError("damping must be within 0..1, excluding endpoints")
    nonnegative(config.linking_top_k, "linking_top_k")
    nonnegative(config.passage_node_weight, "passage_node_weight")
    nonnegative(config.temperature, "temperature")


def validate_route(route):
    seed, bridge, paths = route.seed, route.bridge, route.paths
    nonnegative(seed.hub_penalty_gamma, "route.seed.hub_penalty_gamma")
    for prefix in ("single_hop", "two_hop", "multi_hop"):
        damping = getattr(seed, prefix + "_damping")
        if not 0 < damping < 1:
            raise ValueError(f"route.seed.{prefix}_damping must be within 0..1, excluding endpoints")
        positive(getattr(seed, prefix + "_temperature"), f"route.seed.{prefix}_temperature")
    for value, name in ((bridge.first_hop_k, "bridge.first_hop_k"), (bridge.top_k, "bridge.top_k"),
                        (paths.set_size, "paths.set_size"), (paths.candidate_docs, "paths.candidate_docs"),
                        (route.index.bridge_cache_top_k, "index.bridge_cache_top_k"),
                        (route.index.entity_limit, "index.entity_limit"),
                        (route.iterative.round1_top_docs, "iterative.round1_top_docs"),
                        (route.iterative.seed_top_k, "iterative.seed_top_k"),
                        (route.iterative.min_seed_entities, "iterative.min_seed_entities")):
        positive(value, "route." + name)
    if paths.candidate_docs < paths.set_size:
        raise ValueError("route.paths.candidate_docs must cover route.paths.set_size")
    for name in ("relevance_weight", "connectivity_weight", "consistency_weight", "completeness_weight"):
        nonnegative(getattr(paths, name), "route.paths." + name)
    unit(paths.diversity_lambda, "route.paths.diversity_lambda")
    nonnegative(bridge.weight, "route.bridge.weight")
    if not -1 <= bridge.semantic_threshold <= 1:
        raise ValueError("route.bridge.semantic_threshold must be within -1..1")
    unit(route.iterative.merge_alpha, "route.iterative.merge_alpha")
    if route.iterative.damping != -1 and not 0 < route.iterative.damping < 1:
        raise ValueError("route.iterative.damping must be -1 or within 0..1 excluding endpoints")
    qd = route.refinement
    for name in ("min_hops", "max_sub_questions", "sub_retrieval_top_k"):
        positive(getattr(qd, name), "route.refinement." + name)
    unit(qd.merge_alpha, "route.refinement.merge_alpha")
    nonnegative(qd.temperature, "route.refinement.temperature")
    for name in ("ground_top_docs", "entity_top_k", "max_bridges_per_source"):
        positive(getattr(qd.pcqd, name), "route.refinement.pcqd." + name)
    unit(qd.pcqd.path_score_threshold, "route.refinement.pcqd.path_score_threshold")
    for name in ("weight_base", "weight_static", "weight_path"):
        nonnegative(getattr(qd.pcqd, name), "route.refinement.pcqd." + name)

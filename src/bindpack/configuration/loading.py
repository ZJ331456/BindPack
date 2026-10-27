"""Strict loading of one method parameter source into typed stage schemas."""
from dataclasses import fields, is_dataclass
import json
import math
from pathlib import Path
from typing import get_type_hints

from .runtime import RuntimeConfig
from .validation import validate_route, nonnegative, unit, positive

DEFAULT_METHOD_CONFIG = Path(__file__).with_name("defaults.json")


def read_parameters(path=None):
    source = Path(path) if path is not None else DEFAULT_METHOD_CONFIG
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Method configuration must be an object: {source}")
    allowed = {"runtime", "candidates", "route", "plan", "pack"}
    if value.keys() - allowed:
        raise ValueError(f"Unknown method sections: {sorted(value.keys() - allowed)}")
    return value


def merge_parameters(base, overrides):
    result = dict(base)
    for name, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(name), dict):
            result[name] = merge_parameters(result[name], value)
        else:
            result[name] = value
    return result


def decode(schema, values, location="config"):
    if not isinstance(values, dict):
        raise ValueError(f"{location} must be an object")
    annotations = get_type_hints(schema)
    unknown = values.keys() - annotations.keys()
    if unknown:
        raise ValueError(f"Unknown parameters in {location}: {sorted(unknown)}")
    parsed = {}
    for name, kind in annotations.items():
        if name not in values:
            raise ValueError(f"Missing parameter: {location}.{name}")
        value = values[name]
        if is_dataclass(kind):
            value = decode(kind, value, f"{location}.{name}")
        elif kind is float:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{location}.{name} must be a finite number")
            value = float(value)
        elif not isinstance(value, kind) or kind is int and isinstance(value, bool):
            raise ValueError(f"{location}.{name} must be {kind.__name__}")
        parsed[name] = value
    return schema(**parsed)


def default_group(name, schema):
    return decode(schema, read_parameters()[name], name)


def construct_config(cls, path=None, **overrides):
    parameters = read_parameters()
    if path is not None:
        parameters = merge_parameters(parameters, read_parameters(path))
    groups = {name: overrides.pop(name) for name in ("route", "plan", "pack", "candidates") if name in overrides}
    for name, value in groups.items():
        if not isinstance(value, dict):
            raise TypeError(f"{name} overrides must be an object")
    parameters = merge_parameters(parameters, groups)
    runtime = dict(parameters.get("runtime", {}))
    runtime.update(overrides)
    allowed = {field.name for field in fields(RuntimeConfig)} | {"stage_ablation"}
    if runtime.keys() - allowed:
        raise ValueError(f"Unknown runtime parameters: {sorted(runtime.keys() - allowed)}")
    annotations = get_type_hints(cls)
    return cls(**runtime, **{name: decode(annotations[name], parameters[name], name)
                            for name in ("route", "plan", "pack", "candidates")})


def validate_groups(config):
    """Reject invalid budgets and incompatible selector/review combinations."""
    if not 1 <= config.plan.node_budget <= 8 or not 1 <= config.plan.depth_budget <= 4:
        raise ValueError("Plan node/depth budgets must be within 1..8 and 1..4")
    if config.plan.beam_width != 1:
        raise ValueError("The evaluated Plan retains exactly one binding branch")
    if min(config.plan.verification_top_k, config.plan.verification_passage_chars) < 1:
        raise ValueError("Plan verification limits must be positive")
    if not 1 <= config.pack.budget <= 5:
        raise ValueError("Pack prefix budget must be within 1..5")
    if min(config.candidates.top_k, config.candidates.max_searches) < 1 or config.candidates.pool_size < 5:
        raise ValueError("Candidate search budgets must be positive and the pool at least five")
    positive(config.candidates.rank_constant, "candidates.rank_constant")
    positive(config.pack.review_context_length, "pack.review_context_length")
    if config.pack.review_mode not in {"off", "standard", "suffix_stop"}:
        raise ValueError("pack.review_mode must be off, standard or suffix_stop")
    if config.pack.selector_mode not in {"current", "closed", "residual", "risk", "sources", "combined"}:
        raise ValueError("Unknown pack.selector_mode")
    if config.pack.selector_mode != "current" and config.pack.review_mode != "off":
        raise ValueError("Joint packing requires review_mode=off")
    for name in ("joint_beam_width", "joint_package_cap", "joint_search_seconds", "joint_state_cap"):
        positive(getattr(config.pack, name), "pack." + name)
    unit(config.pack.joint_soft_weight, "pack.joint_soft_weight")
    nonnegative(config.pack.joint_loss_weight, "pack.joint_loss_weight")
    if config.pack.joint_fixed_prefix not in {0, 2}:
        raise ValueError("pack.joint_fixed_prefix must be 0 or 2")
    unit(config.pack.base_weight, "pack.base_weight")
    for name in ("coverage_weight", "relation_weight", "redundancy_weight"):
        nonnegative(getattr(config.pack, name), "pack." + name)
    validate_route(config.route)
    for section in (config.candidates, config.route, config.plan, config.pack):
        validate_values(section)


def validate_values(section):
    for field in fields(section):
        value = getattr(section, field.name)
        if is_dataclass(value):
            validate_values(value)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{field.name} must be finite")

"""Typed pack parameters; values are loaded from defaults.json."""
from dataclasses import dataclass


@dataclass
class PackConfig:
    budget: int
    base_weight: float
    coverage_weight: float
    relation_weight: float
    redundancy_weight: float
    review_tokenizer: str
    review_context_length: int
    review_mode: str
    selector_mode: str
    joint_beam_width: int
    joint_package_cap: int
    joint_search_seconds: float
    joint_soft_weight: float
    joint_loss_weight: float
    joint_state_cap: int
    joint_fixed_prefix: int

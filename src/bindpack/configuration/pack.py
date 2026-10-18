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

"""Typed candidates parameters; values are loaded from defaults.json."""
from dataclasses import dataclass


@dataclass
class CandidateConfig:
    top_k: int
    pool_size: int
    max_searches: int
    rank_constant: float

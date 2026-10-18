"""Typed plan parameters; values are loaded from defaults.json."""
from dataclasses import dataclass


@dataclass
class PlanConfig:
    node_budget: int
    depth_budget: int
    beam_width: int
    verification_top_k: int
    verification_passage_chars: int

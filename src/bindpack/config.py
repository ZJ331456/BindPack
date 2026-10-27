"""Public configuration for the four Route/Plan/Pack experiments.

Edit configuration/defaults.json for method parameters; experiment resources
belong to configs/ablations_qwen3.json. No legacy fields or aliases are accepted.
"""
from dataclasses import dataclass, field
from typing import Literal

from .configuration import RuntimeConfig, CandidateConfig, RouteConfig, PlanConfig, PackConfig
from .configuration.loading import default_group, construct_config, validate_groups


@dataclass
class BindPackConfig(RuntimeConfig):
    stage_ablation: Literal["full", "wo_route", "wo_plan", "wo_pack"] = "full"
    candidates: CandidateConfig = field(default_factory=lambda: default_group("candidates", CandidateConfig))
    route: RouteConfig = field(default_factory=lambda: default_group("route", RouteConfig))
    plan: PlanConfig = field(default_factory=lambda: default_group("plan", PlanConfig))
    pack: PackConfig = field(default_factory=lambda: default_group("pack", PackConfig))

    def __post_init__(self):
        super().__post_init__()
        if self.stage_ablation not in {"full", "wo_route", "wo_plan", "wo_pack"}:
            raise ValueError("stage_ablation must be full, wo_route, wo_plan or wo_pack")
        if self.mechanism_ablation!='full' and self.stage_ablation!='full':
            raise ValueError('Run stage and mechanism ablations separately')
        if self.mechanism_ablation!='full' and self.pack.review_mode!='off':
            raise ValueError('Mechanism experiments require support_semantic_veto off')
        for name, schema in (("candidates", CandidateConfig), ("route", RouteConfig),
                             ("plan", PlanConfig), ("pack", PackConfig)):
            if not isinstance(getattr(self, name), schema):
                raise TypeError(f"{name} must be {schema.__name__}; use from_file for dict overrides")
        if not 1 <= self.llm_prefetch_workers <= 8:
            raise ValueError("llm_prefetch_workers must be within 1..8")
        validate_groups(self)

    @classmethod
    def from_file(cls, path=None, **overrides):
        """Load the default method or a partial parameter file, then apply overrides."""
        return construct_config(cls, path, **overrides)

    @property
    def route_enabled(self):
        return self.stage_ablation != "wo_route"

    @property
    def plan_enabled(self):
        return self.stage_ablation != "wo_plan"

    @property
    def pack_enabled(self):
        return self.stage_ablation != "wo_pack"

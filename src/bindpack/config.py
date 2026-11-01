"""Typed method configuration; Full evaluation uses configuration/defaults.json.

Dataset/model resources are frozen by scripts/run_bindpack.py in each run.
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
        if self.pack_mechanism_ablation!='none' and (self.mechanism_ablation!='full' or self.stage_ablation!='full'):
            raise ValueError('Run Pack scoring/action controls separately from end-to-end mechanism and stage ablations')
        if self.pack_mechanism_ablation!='none' and (self.pack.review_mode!='off' or self.pack.selector_mode!='combined'):
            raise ValueError('Pack scoring controls require combined selector and review off')
        for name, schema in (("candidates", CandidateConfig), ("route", RouteConfig),
                             ("plan", PlanConfig), ("pack", PackConfig)):
            if not isinstance(getattr(self, name), schema):
                raise TypeError(f"{name} must be {schema.__name__}; use from_file for dict overrides")
        if not 1 <= self.llm_prefetch_workers <= 32:
            raise ValueError("llm_prefetch_workers must be within 1..32")
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

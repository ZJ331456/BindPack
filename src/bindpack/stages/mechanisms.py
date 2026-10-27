"""One explicit, end-to-end switch for each mechanism experiment."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Mechanisms:
    source_bound_eligibility: bool = True
    dependency_closure: bool = True
    joint_selection: bool = True


def mechanisms(config):
    arm = getattr(config, "mechanism_ablation", "full")
    return Mechanisms(
        source_bound_eligibility=arm != "wo_source_bound_eligibility",
        dependency_closure=arm != "wo_dependency_closure",
        joint_selection=arm != "wo_joint_selection",
    )

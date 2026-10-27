"""Plan stage: build, validate and execute a source-bound dependency DAG."""
from __future__ import annotations

from .dependency_planner import DependencyPlanner
from .dependency_executor import DependencyExecutor
from .source_verifier import SourceVerifier
from ..mechanisms import mechanisms


class PlanStage:
    """Compose planning and citation verification without owning candidates."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.verifier = SourceVerifier(runtime)
        if mechanisms(runtime.cfg).dependency_closure:
            self.planner = DependencyPlanner(runtime)
            self.executor = DependencyExecutor(runtime, self.verifier)
        else:
            from .independent import IndependentPlanner, IndependentExecutor
            self.planner = IndependentPlanner(runtime)
            self.executor = IndependentExecutor(runtime, self.verifier)

    def process_window(self, states, executor=None):
        requests = []
        for state in states:
            context = state.get("base", (None, None, {}))[2]
            requests.append((self.planner.plan, (state["query"], int(context.get("hops", 1)))))
        for state, (plan, reason) in zip(states, self.runtime._collect_jobs(executor, requests)):
            trace = state["evidence_trace"]
            diagnostic = self.runtime._plan_diagnostics.get(state["query"], {"attempts": 1})
            trace["llm_plan_calls"] += diagnostic["attempts"]
            trace["planning_outputs"] = diagnostic
            trace["plan"] = plan
            state["_evidence_plan"] = plan
            if reason:
                trace["semantic_failures"].append({"node": "plan", "reason": reason})
        self.executor.execute(states, executor)

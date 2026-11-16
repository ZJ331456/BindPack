"""Opt-in observers for bounded Pack search; never supply ranking decisions."""
from collections import Counter
from copy import deepcopy
import time


class PackDiagnostics:
    """Record scores and first-discovery paths without changing search actions.

    ``seconds`` accounts for observer work and is excluded from the search
    deadline. Full score rows stay on the runtime sink; the retrieval trace gets
    a bounded summary. Counters describe observed attempts, not exhaustive paths.
    """

    def __init__(self, sink, arm, rank, budget):
        self.sink, self.arm, self.rank, self.budget = sink, arm, rank, budget
        self.capture_rows = sink.get("capture_score_rows", True)
        self.limit = max(0, min(16, int(sink.get("representative_limit", 12))))
        self.seconds = 0.0
        self.scores, self.last_scores, self.score_rows = {}, {}, []
        self.actions = {name: Counter({key: 0 for key in (
            "attempted", "new_unique", "duplicate", "budget_skip", "no_growth",
            "singleton_unit_attempts", "multi_passage_unit_attempts", "feasible_attempts",
            "multi_passage_growth_attempts", "multi_passage_growth_new_unique",
            "added_passages_in_new_states")}) for name in ("individual", "package")}
        self.frontiers, self.paths, self.package_paths = [], {}, set()
        self.multi_unit_paths, self.multi_growth_paths = set(), set()
        self.attempted_units = {name: set() for name in self.actions}
        self.productive_units = {name: set() for name in self.actions}
        self.timings, self.package_stats = {}, {}

    def ordered(self, selected):
        return sorted(selected, key=self.rank.__getitem__)

    def _score(self, row):
        stage = row["stage"]
        summary = self.scores.setdefault(stage, {
            "count": 0, "different_support_count": 0, "different_objective_count": 0,
            "sum_delta_support": 0.0, "sum_delta_objective": 0.0,
            "max_abs_delta_support": 0.0, "max_abs_delta_objective": 0.0,
            "representatives": [], "by_cardinality": {}})
        summary["count"] += 1
        dc, dj = row["delta_C_flat_minus_closed"], row["delta_J_flat_minus_full"]
        summary["different_support_count"] += abs(dc) > 1e-12
        summary["different_objective_count"] += abs(dj) > 1e-12
        summary["sum_delta_support"] += dc
        summary["sum_delta_objective"] += dj
        summary["max_abs_delta_support"] = max(summary["max_abs_delta_support"], abs(dc))
        summary["max_abs_delta_objective"] = max(summary["max_abs_delta_objective"], abs(dj))
        group = summary["by_cardinality"].setdefault(row["candidate_scope"],
            {"count": 0, "different_support_count": 0, "different_objective_count": 0,
             "max_abs_delta_objective": 0.0})
        group["count"] += 1
        group["different_support_count"] += abs(dc)>1e-12
        group["different_objective_count"] += abs(dj)>1e-12
        group["max_abs_delta_objective"] = max(group["max_abs_delta_objective"], abs(dj))
        examples = summary["representatives"]
        if len(examples) < self.limit:
            examples.append(deepcopy(row))
        elif examples and abs(dj) > min(abs(r["delta_J_flat_minus_full"]) for r in examples):
            weakest = min(range(len(examples)), key=lambda i: abs(examples[i]["delta_J_flat_minus_full"]))
            examples[weakest] = deepcopy(row)
        if self.capture_rows:
            self.score_rows.append(row)

    def score(self, selected, stage, closed, flat, value, full_value, flat_value,
              mode, visible, components):
        tick = time.perf_counter()
        row = dict(stage=stage, doc_ids=self.ordered(selected), cardinality=len(selected),
            candidate_scope="full" if len(selected)==self.budget else "partial",
            reader_visible=visible, scoring_mode=mode, used_objective=value,
            C_closed=closed, C_flat=flat, J_full=full_value, J_flat=flat_value,
            delta_C_flat_minus_closed=flat-closed,
            delta_J_flat_minus_full=flat_value-full_value, **components)
        self.last_scores[frozenset(selected)] = row
        self._score(row)
        self.seconds += time.perf_counter()-tick

    def seed(self, selected, origin):
        tick = time.perf_counter()
        self.paths[frozenset(selected)] = {"origin": origin, "parent": None}
        self.seconds += time.perf_counter()-tick

    def frontier(self, size, bucket, frontier):
        tick = time.perf_counter()
        self.frontiers.append({"size": size, "available_states": len(bucket),
            "retained_states": len(frontier),
            "retained_doc_ids": [self.ordered(s) for s in frontier],
            "retained_objectives": [bucket[s] for s in frontier]})
        for selected in frontier:
            row = dict(self.last_scores[frozenset(selected)], stage="frontier",
                       cached_frontier_objective=bucket[selected])
            self._score(row)
        self.seconds += time.perf_counter()-tick

    def action(self, selected, unit, new, kind, outcome):
        tick = time.perf_counter()
        counts = self.actions[kind]
        self.attempted_units[kind].add(frozenset(unit))
        counts["attempted"] += 1
        counts[outcome] += 1
        counts["singleton_unit_attempts" if len(unit)==1 else "multi_passage_unit_attempts"] += 1
        if outcome in {"new_unique", "duplicate"}:
            counts["feasible_attempts"] += 1
            counts["multi_passage_growth_attempts"] += len(new-selected)>1
        if outcome == "new_unique":
            self.productive_units[kind].add(frozenset(unit))
            counts["multi_passage_growth_new_unique"] += len(new-selected)>1
            counts["added_passages_in_new_states"] += len(new-selected)
        if outcome == "new_unique":
            self.paths[new] = {"origin": kind, "parent": selected,
                               "unit": self.ordered(unit), "added": self.ordered(new-selected)}
        if (outcome in {"new_unique", "duplicate"} and
                (kind == "package" or selected in self.package_paths)):
            self.package_paths.add(new)
        if outcome in {"new_unique", "duplicate"}:
            if (kind == "package" and len(unit)>1) or selected in self.multi_unit_paths:
                self.multi_unit_paths.add(new)
            if (kind == "package" and len(new-selected)>1) or selected in self.multi_growth_paths:
                self.multi_growth_paths.add(new)
        self.seconds += time.perf_counter()-tick

    def path(self, selected):
        selected = frozenset(selected)
        chain, current = [], selected
        while current in self.paths:
            item = self.paths[current]
            if item["parent"] is None:
                origin = item["origin"]
                break
            chain.append({"action": item["origin"], "unit": item["unit"],
                          "added": item["added"], "result": self.ordered(current)})
            current = item["parent"]
        else:
            origin = "unknown"
        chain.reverse()
        return {"origin": origin, "steps": chain, "length": len(chain),
                "package_steps": sum(s["action"]=="package" for s in chain),
                "multi_passage_package_steps": sum(s["action"]=="package" and len(s["added"])>1 for s in chain),
                "observed_package_path": selected in self.package_paths,
                "observed_multi_passage_package_path": selected in self.multi_unit_paths,
                "observed_multi_passage_growth_path": selected in self.multi_growth_paths,
                "semantics": "first-discovery path; observed_package_path also counts later duplicate package reaches"}

    def finish(self, *, weights, packages, bank, package_actions, finalists,
               chosen, s0, expanded, buckets, algorithm_seconds, wall_seconds):
        tick = time.perf_counter()
        selected = frozenset(chosen)
        def describe(values):
            return {"total": len(values), "singleton": sum(len(p)==1 for p in values),
                    "multi_passage": sum(len(p)>1 for p in values)}
        final_rows = [{"doc_ids": self.ordered(s), "path": self.path(s),
                       "scores": self.last_scores.get(frozenset(s)),
                       "complete_bank_packages": sum(p<=s for p in bank),
                       "complete_multi_passage_bank_packages": sum(len(p)>1 and p<=s for p in bank)}
                      for s in finalists]
        stages = deepcopy(self.scores)
        for summary in stages.values():
            n = max(1, summary["count"])
            summary["mean_delta_support"] = summary["sum_delta_support"]/n
            summary["mean_delta_objective"] = summary["sum_delta_objective"]/n
        action_rows = {name: dict(counts, unique_units_attempted=len(self.attempted_units[name]),
            unique_units_creating_new_states=len(self.productive_units[name]))
            for name, counts in self.actions.items()}
        payload = {"schema_version": 1, "arm": self.arm, "weights": weights,
            "score_comparison": stages,
            "reference_scores": self.last_scores.get(frozenset(s0)),
            "chosen_scores": self.last_scores.get(selected),
            "search": {"packages": dict(self.package_stats, feasible=describe(packages),
                capped_bank=describe(bank), active_actions=describe(package_actions)),
                "actions": action_rows,
                "frontiers": self.frontiers,
                "frontier_update_count": expanded,
                "unique_scored_states": sum(len(v) for v in buckets.values()),
                "unique_search_states": sum(len(v) for v in buckets.values()),
                "unique_scored_states_scope": "unique search-bucket states including seeds; excludes bank-only controls and repeated visible rescores",
                "expanded_states": expanded, "finalist_count": len(finalists),
                "reader_visible_evaluation_count": len(finalists), "finalists": final_rows,
                "package_first_discovery_finalists": sum(r["path"]["package_steps"]>0 for r in final_rows),
                "observed_package_path_finalists": sum(r["path"]["observed_package_path"] for r in final_rows),
                "observed_multi_passage_package_path_finalists": sum(r["path"]["observed_multi_passage_package_path"] for r in final_rows),
                "observed_multi_passage_growth_path_finalists": sum(r["path"]["observed_multi_passage_growth_path"] for r in final_rows),
                "chosen_path": self.path(selected),
                "chosen_equals_reference": selected==frozenset(s0),
                "chosen_complete_bank_packages": sum(p<=selected for p in bank),
                "chosen_complete_multi_passage_bank_packages": sum(len(p)>1 and p<=selected for p in bank),
                "reference_complete_multi_passage_bank_packages": sum(len(p)>1 and p<=set(s0) for p in bank)},
            "timings": dict(self.timings, algorithm_seconds=algorithm_seconds,
                            wall_seconds=wall_seconds),
            "score_row_scope": "same visited candidate set scored both ways; not exhaustive search or semantic proof",
            "search_budget_excludes_diagnostics": True}
        self.seconds += time.perf_counter()-tick
        payload["timings"]["diagnostic_seconds"] = self.seconds
        payload["timings"]["wall_seconds"] = algorithm_seconds+self.seconds
        self.sink.update(payload)
        if self.capture_rows:
            self.sink["score_rows"] = self.score_rows
        return payload

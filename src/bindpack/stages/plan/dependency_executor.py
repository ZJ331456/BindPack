"""Layered single-branch dependency search with source-bound answers."""
from __future__ import annotations

from typing import Dict, List
import numpy as np
from ...utils.evidence import normalized_text
from .answer_binder import bind_question
from .validation import plan_layers


class DependencyExecutor:
    """Search on the caller thread and verify with the supplied executor."""

    def __init__(self, runtime, verifier):
        self.runtime = runtime
        self.verifier = verifier
        self.config = runtime.cfg.plan
        self.beam_width = int(self.config.beam_width)

    @staticmethod
    def _beam_score(beam: dict, total: int) -> float:
        qualities = beam["qualities"]
        if not qualities:
            return 0.0
        coverage = len(beam["bindings"]) / max(1, total)
        return float(0.45 * min(qualities) + 0.35 * np.mean(qualities) + 0.20 * coverage)


    def _prune_beams(self, beams: List[dict], total: int) -> List[dict]:
        unique = {}
        for beam in beams:
            key = tuple(sorted((k, normalized_text(v)) for k, v in beam["bindings"].items()))
            if key not in unique or self._beam_score(beam, total) > self._beam_score(unique[key], total):
                unique[key] = beam
        ordered = sorted(unique.values(), key=lambda b: (
            -self._beam_score(b, total), tuple(sorted(b["bindings"].items()))))
        return ordered[:self.beam_width]


    def _verification_documents(self, ids: List[int]) -> Dict[int, str]:
        """The original verifier receives the first three bounded passages."""
        return {doc_id: self.runtime._document(doc_id)[:self.config.verification_passage_chars]
                for doc_id in ids[:self.config.verification_top_k]}


    def execute(self, states: List[dict], executor):
        active = [s for s in states if s.get("_evidence_plan")]
        for state in active:
            state["_evidence_layers"] = plan_layers(state["_evidence_plan"])
            state["_evidence_beams"] = [{"bindings": {}, "proofs": {}, "qualities": []}]
        depth = max((len(s["_evidence_layers"]) for s in active), default=0)
        for level in range(depth):
            tasks, task_lookup = [], {}
            # All embedding calls below execute on this caller thread.
            for state in active:
                if level >= len(state["_evidence_layers"]):
                    continue
                for node in state["_evidence_layers"][level]:
                    for beam_index, beam in enumerate(state["_evidence_beams"]):
                        question = bind_question(node["question"], beam["bindings"])
                        if question is None:
                            state["evidence_trace"]["semantic_failures"].append({
                                "node": node["id"], "reason": "unbound_dependency",
                                "requirements": list(node["depends_on"])})
                            continue
                        requirements = dict(beam["bindings"])
                        ids = self.runtime._add_route(state, f"dag:{node['id']}", f"dag:{node['id']}", question, requirements)
                        if not ids:
                            continue
                        # The verifier sees exactly these spans, not unbounded full documents.
                        docs = self._verification_documents(ids)
                        task = (state, node, beam_index, list(docs), question, docs)
                        task_lookup[(id(state), node["id"], beam_index)] = len(tasks)
                        tasks.append(task)
            results = self.runtime._collect_jobs(executor, [
                (self.verifier.verify, (question, node["answer_type"], docs))
                for state, node, beam_index, ids, question, docs in tasks])
            for task, (hypotheses, rejected, reason) in zip(tasks, results):
                state, node, beam_index, ids, question, docs = task
                trace = state["evidence_trace"]
                diagnostic_key = question + "::" + ",".join(map(str, docs))
                verification_diagnostic = self.runtime._verification_diagnostics.get(diagnostic_key, {"attempts": 1})
                trace["llm_verification_calls"] += verification_diagnostic["attempts"]
                trace.setdefault("verification_outputs", []).append(dict(verification_diagnostic, node=node["id"]))
                trace["rejected_hypotheses"].extend(dict(item, node=node["id"]) for item in rejected)
                if reason:
                    trace["semantic_failures"].append({"node": node["id"], "reason": reason})
                for hyp in hypotheses:
                    local_rank = ids.index(hyp["doc_id"]) + 1
                    hyp["quality"] = float(0.6 * self.runtime.rank_constant / (self.runtime.rank_constant + local_rank)
                                           + 0.4 * hyp["confidence"])
                hypotheses.sort(key=lambda h: (-h["quality"], h["doc_id"], normalized_text(h["answer"])))
                # Keep at most one proof per answer hypothesis to avoid a fake beam of duplicates.
                unique, seen = [], set()
                for hyp in hypotheses:
                    answer_key = normalized_text(hyp["answer"])
                    if answer_key not in seen:
                        unique.append(hyp)
                        seen.add(answer_key)
                results[task_lookup[(id(state), node["id"], beam_index)]] = unique[:self.beam_width]
                if not unique:
                    trace["semantic_failures"].append({"node": node["id"], "reason": "no_verified_answer"})
            # Cartesian expansion for independent nodes, preserving every ancestor binding.
            for state in active:
                if level >= len(state["_evidence_layers"]):
                    continue
                expanded = []
                for beam_index, ancestor in enumerate(state["_evidence_beams"]):
                    descendants = [ancestor]
                    for node in state["_evidence_layers"][level]:
                        index = task_lookup.get((id(state), node["id"], beam_index))
                        choices = results[index] if index is not None else []
                        if not choices:
                            continue
                        new = []
                        for descendant in descendants:
                            for hyp in choices:
                                child = {"bindings": dict(descendant["bindings"]),
                                         "proofs": dict(descendant["proofs"]),
                                         "qualities": list(descendant["qualities"])}
                                child["bindings"][node["id"]] = hyp["answer"]
                                child["proofs"][node["id"]] = dict(hyp)
                                child["qualities"].append(hyp["quality"])
                                new.append(child)
                        descendants = self._prune_beams(new, len(state["_evidence_plan"]))
                    expanded.extend(descendants)
                state["_evidence_beams"] = self._prune_beams(expanded, len(state["_evidence_plan"]))
        for state in active:
            beams = state["_evidence_beams"]
            total = len(state["_evidence_plan"])
            best = beams[0] if beams else {"bindings": {}, "proofs": {}, "qualities": []}
            trace = state["evidence_trace"]
            trace["bindings"] = dict(best["bindings"])
            trace["branch_scores"] = [
                {"bindings": dict(b["bindings"]), "score": self._beam_score(b, total),
                 "proofs": dict(b["proofs"])} for b in beams]
            state["_evidence_winning_bindings"] = best["bindings"]
            for node_id, hyp in best["proofs"].items():
                candidate = self.runtime._candidate(state, hyp["doc_id"])
                candidate["verified"].append(dict(hyp, goal=f"dag:{node_id}"))

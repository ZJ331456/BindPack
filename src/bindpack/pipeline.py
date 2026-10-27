"""Coordinate the shared candidate pool between Route, Plan and Pack.

Route produces the initial states and ranking. Plan executes dependency searches
before the final Route fusion, and Pack consumes that ranking plus shared evidence.
Dense searches and source reads remain on the caller thread.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .utils.evidence import normalized_text, parse_object, local_rank_scores
from .stages.plan.stage import PlanStage
from .stages.pack.stage import PackStage
from .stages.mechanisms import mechanisms
from dataclasses import asdict


class EvidencePipeline:
    """Own shared candidates, request diagnostics and explicit stage components."""

    def __init__(self, rag):
        self.rag, self.cfg = rag, rag.config
        self.top_k = int(self.cfg.candidates.top_k)
        self.pool_size = int(self.cfg.candidates.pool_size)
        self.max_searches = int(self.cfg.candidates.max_searches)
        self.rank_constant = float(self.cfg.candidates.rank_constant)
        self._documents: Dict[int, str] = {}
        self._plan_diagnostics: Dict[str, dict] = {}
        self._verification_diagnostics: Dict[str, dict] = {}
        self.plan = PlanStage(self)
        self.pack = PackStage(self)

    def _document(self, doc_id: int) -> str:
        if doc_id not in self._documents:
            key = self.rag.passage_node_keys[doc_id]
            row = self.rag.chunk_embedding_store.get_row(key)
            self._documents[doc_id] = str(row.get("content", "")) if isinstance(row, dict) else ""
        return self._documents[doc_id]


    def _trace(self) -> dict:
        trace = {"stages": {"route": self.cfg.stage_ablation != "wo_route",
                           "plan": self.cfg.stage_ablation != "wo_plan",
                           "pack": self.cfg.stage_ablation != "wo_pack"}, "candidate_count": 0, "search_count": 0,
                "llm_plan_calls": 0, "llm_verification_calls": 0,
                "semantic_failures": [], "rejected_hypotheses": [],
                "plan": [], "bindings": {}, "branch_scores": [],
                "routes": [], "selected_prefix": []}
        if getattr(self.cfg,'mechanism_ablation','full') != 'full':
            trace['mechanism_ablation'] = self.cfg.mechanism_ablation
            trace['mechanisms'] = asdict(mechanisms(self.cfg))
        return trace


    def _infer_object(self, messages: List[dict]) -> Tuple[Optional[dict], Optional[str]]:
        # HTTP failures deliberately propagate through future.result().
        ret = self.rag.llm_model.infer(messages=messages, temperature=0.0)
        response = ret[0] if isinstance(ret, tuple) else ret
        metadata = ret[1] if isinstance(ret, tuple) and len(ret) > 1 else {}
        if isinstance(metadata, dict) and metadata.get("finish_reason") == "length":
            return None, "truncated_response"
        text = self.rag._extract_llm_text(response)
        return parse_object(text)


    @staticmethod
    def _collect_jobs(executor, jobs: List[Tuple[Any, tuple]]) -> list:
        if executor is None:
            return [func(*args) for func, args in jobs]
        futures = [executor.submit(func, *args) for func, args in jobs]
        return [future.result() for future in futures]


    def _candidate(self, state: dict, doc_id: int) -> dict:
        candidates = state["evidence_candidates"]
        if doc_id not in candidates:
            candidates[doc_id] = {"doc_id": doc_id, "base_score": 0.0,
                                  "sources": [], "verified": []}
        return candidates[doc_id]


    def _add_route(self, state: dict, source: str, goal: str, query: str,
                   requirements: Optional[dict] = None) -> Optional[List[int]]:
        trace = state["evidence_trace"]
        requirements = dict(requirements or {})
        cache = state["_evidence_search_cache"]
        if query in cache:
            ids, raw = cache[query]
        else:
            if trace["search_count"] >= self.max_searches:
                trace["semantic_failures"].append({"node": goal, "reason": "search_budget_exhausted"})
                return None
            ids, raw = self.rag.dense_passage_retrieval(query)
            ids = [int(i) for i in np.asarray(ids)[:self.top_k]]
            raw = np.asarray(raw, dtype=float)[:len(ids)]
            cache[query] = (ids, raw)
            trace["search_count"] += 1
        local_scores = local_rank_scores(raw, self.rank_constant)
        for rank, (doc_id, score, original) in enumerate(zip(ids, local_scores, raw), 1):
            self._candidate(state, doc_id)["sources"].append({
                "source": source, "goal": goal, "rank": rank, "raw_score": float(original),
                "score": score, "question": query, "requirements": requirements})
        trace["routes"].append({"source": source, "goal": goal, "query": query,
                                "doc_ids": list(ids), "requirements": requirements})
        return ids


    @staticmethod
    def _goal_for_question(question: str, static_questions: List[str], fallback: str) -> str:
        tokens = set(normalized_text(question).split())
        best = (0.0, fallback)
        for index, static in enumerate(static_questions):
            other = set(normalized_text(static).split())
            sim = len(tokens & other) / max(1, len(tokens | other))
            if sim > best[0]:
                best = (sim, f"static:s{index + 1}")
        return best[1] if best[0] >= 0.25 else fallback


    def _prepare_state(self, state: dict):
        state["evidence_candidates"] = {}
        state["evidence_trace"] = self._trace()
        state["_evidence_search_cache"] = {}
        if "base" in state:
            ids, scores, ctx = state["base"]
            top_ids = np.asarray(ids)[:self.pool_size]
            top_scores = np.asarray(scores, dtype=float)[:len(top_ids)]
            values = local_rank_scores(top_scores, self.rank_constant)
            for doc_id, score in zip(top_ids, values):
                self._candidate(state, int(doc_id))["base_score"] = score
        static = [q for q in state.get("static_sub_questions", []) if isinstance(q, str) and q.strip()]
        for index, question in enumerate(static[:4]):
            self._add_route(state, f"static:{index + 1}", f"static:s{index + 1}", question)
        pcqd = state.get("pcqd_sub_questions") or []
        for index, item in enumerate(pcqd[:4]):
            question = item.get("question", "") if isinstance(item, dict) else str(item)
            if not question.strip():
                continue
            goal = self._goal_for_question(question, static, f"pcqd:p{index + 1}")
            self._add_route(state, f"pcqd:{index + 1}", goal, question)


    def _trim_pool(self, state: dict):
        candidates = state["evidence_candidates"]
        bindings = state.get("_evidence_winning_bindings", {})
        ranked = sorted(candidates, key=lambda d: (
            -bool(candidates[d]["verified"]),
            -(0.4 * candidates[d]["base_score"] +
              0.6 * max(self.pack.scorer.goal_scores(candidates[d], bindings).values(), default=0.0)), d))
        # Reserve one good candidate per subgoal before filling the shared pool.
        goals = defaultdict(list)
        for doc_id in ranked:
            for goal, score in self.pack.scorer.goal_scores(candidates[doc_id], bindings).items():
                goals[goal].append((score, doc_id))
        reserved = [d for d in ranked if candidates[d]["verified"]]
        reserved.extend(min(values, key=lambda x: (-x[0], x[1]))[1] for values in goals.values())
        keep = list(dict.fromkeys(reserved + ranked))[:self.pool_size]
        state["evidence_candidates"] = {d: candidates[d] for d in keep}
        state["evidence_trace"]["candidate_count"] = len(keep)


    def process_window(self, states: List[dict], executor=None):
        for state in states:
            self._prepare_state(state)
        if self.cfg.stage_ablation == "wo_plan":
            for state in states:
                state["_evidence_plan"] = []
                state["_evidence_layers"] = []
                state["_evidence_beams"] = []
                state["_evidence_winning_bindings"] = {}
                state["evidence_trace"]["plan_mode"] = "flat_route_goals"
        else:
            self.plan.process_window(states, executor)
        for state in states:
            self._trim_pool(state)
            state.pop("_evidence_search_cache", None)

    def finalize(self, query, ids, scores, context, state):
        return self.pack.finalize(query, ids, scores, context, state)

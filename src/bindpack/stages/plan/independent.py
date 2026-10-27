"""Dependency ablation: standalone planning and parallel, non-propagating execution.

No dependency planner, topological layering, answer substitution, or ancestor
requirements are used. Source verification is independently controlled.
"""
from __future__ import annotations
from copy import deepcopy
import re
from ...utils.evidence import normalized_text
from .question_structure import route_question_structure


class IndependentPlanner:
    def __init__(self, runtime):
        self.runtime = runtime

    def plan(self, query, hops):
        budget = int(self.runtime.cfg.plan.node_budget)
        routing = route_question_structure(query)
        if routing is not None and not routing['expand']:
            budget = max(1,min(4,int(hops)))
        messages = [
            {"role": "system", "content": (
                "Plan independent evidence retrieval using ONLY the input question. "
                "Output ONLY JSON {questions:[{question,answer_type}]}. Each question "
                "must be a standalone search request understandable without any other "
                "question's answer. Use named entities and descriptions from the input; "
                "never guess intermediate answers. Preserve nested relation, time and "
                "location qualifiers as descriptions when their entities are unknown. "
                "No DAG, dependency lists, node references, variables or answer placeholders. "
                "Do not add arithmetic or final comparison questions. Return the fewest "
                "independent requests that preserve the input's evidence requirements."
            )},
            {"role": "user", "content": (
                f"Question: {query}\nMaximum requests: {budget}\n"
                f"Benchmark evidence-depth hint: {max(1,int(hops))}\n"
                'Example format: {"questions":[{"question":"Where was the writer of Work X born?",'
                '"answer_type":"place"}]}'
            )},
        ]
        payload, reason = self.runtime._infer_object(messages)
        diagnostic = {"attempts": 1, "outputs": [payload], "validation_errors": [],
                      "execution_mode": "independent", "answer_propagation": False,
                      "node_budget":budget,"routing":routing}
        self.runtime._plan_diagnostics[query] = diagnostic
        items = payload.get("questions") if isinstance(payload, dict) else None
        if not reason and (not isinstance(items, list) or not 1 <= len(items) <= budget):
            reason = "invalid_independent_questions"
        nodes, seen = [], set()
        if not reason:
            for item in items:
                if not isinstance(item, dict):
                    reason = "invalid_independent_question"; break
                question, kind = item.get("question"), item.get("answer_type", "entity")
                if (not isinstance(question, str) or not question.strip() or
                        not isinstance(kind, str) or not kind.strip() or
                        item.get("depends_on") or re.search(r"\$\{|\b\w+\.answer\b",question)):
                    reason = "independent_question_contains_reference_or_invalid_field"; break
                key = normalized_text(question)
                if key in seen: continue
                seen.add(key)
                nodes.append({"id": f"s{len(nodes)+1}", "question": question.strip(),
                              "answer_type": kind.strip(), "depends_on": []})
        if reason:
            diagnostic["validation_errors"].append(reason)
            return [], reason
        return nodes, None


class IndependentExecutor:
    def __init__(self, runtime, verifier):
        self.runtime, self.verifier = runtime, verifier

    def execute(self, states, executor):
        tasks = []
        for state in states:
            state["_evidence_layers"] = []
            state["_evidence_winning_bindings"] = {}
            state["_evidence_beams"] = []
            state["evidence_trace"]["execution_mode"] = "independent_no_answer_propagation"
            state["evidence_trace"]["dependency_edges_executed"] = 0
            for node in state.get("_evidence_plan", []):
                assert not node["depends_on"] and "${" not in node["question"]
                ids = self.runtime._add_route(state,f"flat:{node['id']}",f"flat:{node['id']}",node["question"],{})
                if not ids: continue
                docs = {d:self.runtime._document(d)[:self.runtime.cfg.plan.verification_passage_chars]
                        for d in ids[:self.runtime.cfg.plan.verification_top_k]}
                tasks.append((state,node,list(docs),docs))
        results = self.runtime._collect_jobs(executor, [
            (self.verifier.verify,(node["question"],node["answer_type"],docs))
            for state,node,ids,docs in tasks])
        proofs_by_state = {id(s):{} for s in states}
        for (state,node,ids,docs),(hypotheses,rejected,reason) in zip(tasks,results):
            trace = state["evidence_trace"]
            key = node["question"]+"::"+",".join(map(str,docs))
            diagnostic = deepcopy(self.runtime._verification_diagnostics.get(key,{"attempts":1}))
            trace["llm_verification_calls"] += diagnostic["attempts"]
            trace.setdefault("verification_outputs",[]).append(dict(diagnostic,node=node["id"]))
            trace["rejected_hypotheses"].extend(dict(p,node=node["id"]) for p in rejected)
            if reason: trace["semantic_failures"].append({"node":node["id"],"reason":reason})
            for hyp in hypotheses:
                rank = ids.index(hyp["doc_id"])+1
                hyp["quality"] = float(.6*self.runtime.rank_constant/(self.runtime.rank_constant+rank)+.4*hyp["confidence"])
            hypotheses.sort(key=lambda p:(-p["quality"],p["doc_id"],normalized_text(p["answer"])))
            trace.setdefault("source_records",[]).extend(dict(p,goal="flat:"+node["id"],requirements={}) for p in hypotheses)
            if hypotheses:
                hyp = hypotheses[0]
                state["_evidence_winning_bindings"][node["id"]] = hyp["answer"]
                proofs_by_state[id(state)][node["id"]] = dict(hyp)
                self.runtime._candidate(state,hyp["doc_id"])["verified"].append(dict(hyp,goal="flat:"+node["id"]))
        for state in states:
            bindings = state["_evidence_winning_bindings"]
            # A container of independent facts, never an executed binding branch.
            state["_evidence_beams"] = [{"bindings":dict(bindings),"proofs":proofs_by_state[id(state)],"score":0.0}]
            state["evidence_trace"]["bindings"] = dict(bindings)
            state["evidence_trace"]["branch_scores"] = deepcopy(state["_evidence_beams"])

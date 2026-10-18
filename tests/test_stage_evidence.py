"""Stage deletion boundaries without model services or benchmark labels."""
import copy
import json
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


from bindpack import BindPackConfig
from bindpack.pipeline import EvidencePipeline as Engine
from bindpack.stages.pack import dag_package_selector as dag

QUESTION = "Where was the writer of Work X born?"
PLAN = {"nodes": [
    {"id": "s1", "question": "Who wrote Work X?", "depends_on": [], "answer_type": "person"},
    {"id": "s2", "question": "Where was ${s1.answer} born?",
     "depends_on": ["s1"], "answer_type": "place"},
]}
ANSWERS = [
    {"hypotheses": [{"answer": "Alpha", "doc_id": "D0", "confidence": .9,
                     "evidence": "Alpha wrote Work X in 1990."}]},
    {"hypotheses": [{"answer": "Rome", "doc_id": "D1", "confidence": .9,
                     "evidence": "Alpha was born in Rome in 1960."}]},
]


class FakeRAG:
    def __init__(self, mode="full", responses=()):
        self.config = BindPackConfig(stage_ablation=mode)
        self.config.pack.budget = 2
        self.responses = iter(copy.deepcopy(responses))
        self.searches, self.messages = [], []
        self.passage_node_keys = list(range(4))
        self.documents = {
            0: "Writer\nAlpha wrote Work X in 1990.",
            1: "Alpha\nAlpha was born in Rome in 1960.",
            2: "Work X\nAlpha wrote Work X in 1990.",
            3: "Other\nA different unrelated passage.",
        }
        self.chunk_embedding_store = SimpleNamespace(
            get_row=lambda key: {"content": self.documents[key]})
        self.llm_model = SimpleNamespace(infer=self.infer)

    def infer(self, messages, temperature):
        self.messages.append(copy.deepcopy(messages))
        return json.dumps(next(self.responses)), {"finish_reason": "stop"}

    def dense_passage_retrieval(self, question):
        self.searches.append(question)
        order = [1, 0, 2, 3] if "Alpha" in question else [0, 2, 1, 3]
        return np.asarray(order), np.asarray([.9, .8, .6, .4])

    @staticmethod
    def _extract_llm_text(response):
        return str(response)


def state_fixture():
    return {"query_idx": 0, "query": QUESTION,
            "base": (np.arange(4), np.asarray([.9, .8, .7, .6]), {"hops": 2}),
            "static_sub_questions": ["Who wrote Work X?"],
            "pcqd_sub_questions": [{"question": "Where was the writer born?"}]}


def selection_fixture():
    """Two near-duplicate routes and one complementary route."""
    candidates = {}
    for doc_id, goal in [(0, "static:s1"), (1, "static:s1"), (2, "static:s2")]:
        candidates[doc_id] = {"doc_id": doc_id, "base_score": .8, "verified": [],
                              "sources": [{"goal": goal, "score": 1., "requirements": {}}]}
    return {"evidence_candidates": candidates, "evidence_trace": {"selected_prefix": []}}


class StageEvidenceTests(unittest.TestCase):
    def test_without_plan_skips_planning_execution_and_clears_stale_dag(self):
        rag = FakeRAG("wo_plan")
        engine = Engine(rag)
        state = state_fixture()
        state.update(_evidence_plan=PLAN["nodes"], _evidence_layers=[[PLAN["nodes"][0]]],
                     _evidence_winning_bindings={"s1": "Old answer"},
                     _evidence_beams=[{"bindings": {"s1": "Old answer"}, "proofs": {"s1": {}}}],
                     evidence_candidates={99: {"verified": [{"goal": "dag:s1"}]}})
        with patch.object(engine.plan.planner, "plan", side_effect=AssertionError("Plan called")), \
                patch.object(engine.plan.executor, "execute", side_effect=AssertionError("DAG search called")):
            engine.process_window([state])
        self.assertEqual(rag.messages, [])
        self.assertEqual(rag.searches, ["Who wrote Work X?", "Where was the writer born?"])
        self.assertEqual(state["_evidence_plan"], [])
        self.assertEqual(state["_evidence_layers"], [])
        self.assertEqual(state["_evidence_beams"], [])
        self.assertEqual(state["_evidence_winning_bindings"], {})
        self.assertNotIn(99, state["evidence_candidates"])
        self.assertNotIn("_evidence_search_cache", state)
        trace = state["evidence_trace"]
        self.assertEqual((trace["llm_plan_calls"], trace["llm_verification_calls"]), (0, 0))
        self.assertEqual((trace["plan"], trace["bindings"], trace["branch_scores"]), ([], {}, []))
        for candidate in state["evidence_candidates"].values():
            self.assertEqual(candidate["verified"], [])
            for source in candidate["sources"]:
                self.assertFalse(source["goal"].startswith("dag:"))
                self.assertEqual(source["requirements"], {})

    def test_without_plan_retains_flat_coverage_and_redundancy_pack(self):
        rag = FakeRAG("wo_plan")
        rag.config.pack.coverage_weight = 3.
        engine = Engine(rag)
        state = selection_fixture()
        with patch.object(dag, "rerank_dag_packages", side_effect=AssertionError("DAG Pack called")):
            ids, _, trace = engine.finalize(QUESTION, np.arange(3), np.array([1., .95, .1]), {}, state)
        self.assertEqual(ids[:2].tolist(), [0, 2])
        self.assertGreater(trace["selected_prefix"][1]["coverage_gain"], 0.)
        self.assertIn("redundancy", trace["selected_prefix"][1])
        self.assertNotIn("improvement_dag_package", trace)

    def test_without_plan_with_no_subquestions_uses_only_base_candidates(self):
        rag = FakeRAG("wo_plan")
        engine = Engine(rag)
        state = state_fixture()
        state.update(static_sub_questions=[], pcqd_sub_questions=[])
        engine.process_window([state])
        self.assertEqual(rag.searches, [])
        self.assertEqual(set(state["evidence_candidates"]), {0, 1, 2, 3})
        self.assertEqual(state["evidence_trace"]["search_count"], 0)

    def test_without_pack_freezes_route_plan_candidates_and_requests(self):
        outputs = []
        for mode in ["full", "wo_pack", "wo_route"]:
            rag = FakeRAG(mode, [PLAN, *ANSWERS])
            engine = Engine(rag)
            state = state_fixture()
            engine.process_window([state])
            # The route ablation changes engine discovery upstream. Given the
            # same input here, Plan + candidate preparation must be identical.
            outputs.append((state, rag.searches, rag.messages))
        baseline_state, baseline_searches, baseline_messages = outputs[0]
        for state, searches, messages in outputs[1:]:
            self.assertEqual(searches, baseline_searches)
            self.assertEqual(messages, baseline_messages)
            for key in ["evidence_candidates", "_evidence_plan", "_evidence_layers",
                        "_evidence_beams", "_evidence_winning_bindings", "evidence_trace"]:
                left, right = copy.deepcopy(state[key]), copy.deepcopy(baseline_state[key])
                if key == "evidence_trace":
                    left.pop("stages")
                    right.pop("stages")
                self.assertEqual(left, right, key)
        self.assertEqual(baseline_state["_evidence_winning_bindings"], {"s1": "Alpha", "s2": "Rome"})

    def test_without_pack_uses_original_pointwise_fusion_without_document_reads(self):
        engine = Engine(FakeRAG("wo_pack"))
        state = selection_fixture()
        before = copy.deepcopy(state["evidence_candidates"])
        with patch.object(engine, "_document", side_effect=AssertionError("Redundancy read called")), \
                patch.object(dag, "rerank_dag_packages", side_effect=AssertionError("DAG Pack called")):
            ids, scores, trace = engine.finalize(QUESTION, np.arange(3), np.array([1., .95, .1]), {}, state)
        self.assertEqual(ids.tolist(), [0, 1, 2])
        np.testing.assert_allclose(scores, [.4 + .6, .4 * (.85 / .9) + .6, .6])
        self.assertEqual(state["evidence_candidates"], before)
        self.assertEqual(trace["covered_goals"], {})
        self.assertTrue(all("coverage_gain" not in p and "redundancy" not in p
                            for p in trace["selected_prefix"]))
        self.assertNotIn("improvement_dag_package", trace)

    def test_without_pack_keeps_route_tail_and_plan_only_documents(self):
        engine = Engine(FakeRAG("wo_pack"))
        state = selection_fixture()
        state["evidence_candidates"][3] = {
            "doc_id": 3, "base_score": .8, "verified": [],
            "sources": [{"goal": "dag:s3", "score": .9, "requirements": {}}]}
        ids, scores, _ = engine.finalize(QUESTION, np.array([0, 1, 2, 99]),
                                         np.array([1., .95, .1, 0.]), {}, state)
        self.assertEqual(set(ids), {0, 1, 2, 3, 99})
        self.assertEqual(len(set(ids)), len(ids))
        self.assertTrue(np.all(scores[:-1] >= scores[1:]))

    def test_without_pack_no_candidates_preserves_input_objects(self):
        engine = Engine(FakeRAG("wo_pack"))
        ids, scores = np.arange(4), np.array([.9, .8, .7, .6])
        actual_ids, actual_scores, _ = engine.finalize(QUESTION, ids, scores, {}, {})
        self.assertIs(actual_ids, ids)
        self.assertIs(actual_scores, scores)


if __name__ == "__main__":
    unittest.main()

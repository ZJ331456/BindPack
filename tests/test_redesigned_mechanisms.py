"""Causal isolation checks for the redesigned five mechanism variants."""
from copy import deepcopy
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
import numpy as np

from bindpack import BindPackConfig
from bindpack.stages.mechanisms import mechanisms
from bindpack.stages.pack.stage import PackStage
from bindpack.stages.pack.unbound_support import unbound_pack_state
from bindpack.stages.pack.budgeted_selector import BudgetedSelector
from bindpack.stages.pack.evidence_scorer import EvidenceScorer
from bindpack.stages.plan.answer_binder import describe_parent_questions
from bindpack.stages.plan.dependency_executor import DependencyExecutor
from test_evidence_dag_package import Engine, chain


class RedesignedMechanismTests(TestCase):
    def fixture(self, arm="full", eligibility=False):
        engine = Engine()
        state = chain(engine, count=2)
        for candidate in state["evidence_candidates"].values():
            candidate.update(sources=[], base_score=0.)
            for proof in candidate["verified"]:
                proof["quality"] = .95
        state["evidence_trace"]["extraction_inputs"] = [
            {"node": "a", "doc_ids": [0, 1, 3], "quality": .95},
            {"node": "b", "doc_ids": [2, 4, 6], "quality": .85}]
        cfg = BindPackConfig.from_file(mechanism_ablation=arm, source_bound_eligibility=eligibility)
        runtime = SimpleNamespace(cfg=cfg, _document=engine._document, _trace=lambda: {},
            _joint_visible=lambda q, ids: ({d: engine._document(d) for d in ids}, {}))
        return runtime, state

    def test_new_arms_are_independent_of_eligibility_and_stage_controls(self):
        for arm in ("wo_source_binding", "wo_parent_answer_propagation"):
            with self.subTest(arm=arm):
                cfg = BindPackConfig.from_file(mechanism_ablation=arm, source_bound_eligibility=True)
                flags = mechanisms(cfg)
                self.assertTrue(flags.source_bound_eligibility)
                self.assertTrue(flags.dependency_closure)
                self.assertTrue(flags.joint_selection)
                self.assertEqual(flags.source_binding, arm != "wo_source_binding")
                self.assertEqual(flags.parent_answer_propagation, arm != "wo_parent_answer_propagation")
                with self.assertRaises(ValueError):
                    BindPackConfig.from_file(mechanism_ablation=arm, stage_ablation="wo_plan")
                with self.assertRaises(ValueError):
                    BindPackConfig.from_file(mechanism_ablation=arm, pack_mechanism_ablation="singleton_search")

    def test_unbound_view_hides_attribution_and_uniformly_scores_extraction_pool(self):
        _, state = self.fixture()
        state["evidence_trace"].update(verification_outputs=[{"outputs": [{"doc_id": 0}]}],
            source_records=[{"doc_id": 0, "answer": "Alice", "requirements": {"secret": "answer"}}])
        original = deepcopy(state)
        clean = unbound_pack_state(state)
        self.assertEqual(state, original)
        self.assertEqual(clean["_evidence_plan"], state["_evidence_plan"])
        self.assertEqual(clean["_evidence_winning_bindings"], state["_evidence_winning_bindings"])
        self.assertEqual(clean["_evidence_beams"][0]["proofs"], {})
        self.assertNotIn("verification_outputs", clean["evidence_trace"])
        a = [r for r in clean["evidence_trace"]["source_records"] if r["goal"] == "dag:a"]
        self.assertEqual({r["doc_id"] for r in a}, {0, 1, 3})
        self.assertEqual({r["quality"] for r in a}, {.95})
        self.assertTrue(all(not r["requirements"] and "answer" not in r and "evidence" not in r for r in a))

    def test_source_binding_replay_preserves_full_reference_and_has_no_exact_proofs(self):
        runtime, state = self.fixture()
        captured = []
        runtime._capture_pack_reference = captured.append
        ids, scores = np.arange(220), np.linspace(1, 0, 220)
        full = PackStage(runtime).finalize("Sample Town", ids, scores, {}, state)
        self.assertEqual(len(captured), 1)
        runtime.cfg.mechanism_ablation = "wo_source_binding"
        runtime._pack_reference = captured[0]
        before = deepcopy(state)
        with patch.object(PackStage, "build_reference", side_effect=AssertionError("reference recomputed")):
            result = PackStage(runtime).finalize("Sample Town", ids, scores, {}, state)
        diag = result[2]["budgeted_selection"]
        self.assertEqual(diag["reference_doc_ids"], full[2]["budgeted_selection"]["reference_doc_ids"])
        self.assertFalse(diag["source_binding"])
        self.assertEqual({p["doc_id"] for p in diag["eligible_proofs"]["a"]}, {0, 1, 3})
        self.assertTrue(all(p["check"] == "unbound_extraction_candidate"
                            and "answer" not in p and "evidence" not in p
                            for values in diag["eligible_proofs"].values() for p in values))
        self.assertEqual(state, before)

    def test_reference_replay_does_not_search_or_render_visible_context(self):
        runtime, state = self.fixture("wo_joint_selection")
        ids, scores = np.arange(220), np.linspace(1, 0, 220)
        runtime._pack_reference = (ids[::-1], scores, {})
        with patch.object(BudgetedSelector, "select", side_effect=AssertionError("search executed")):
            result = PackStage(runtime).finalize("Sample Town", ids, scores, {}, state)
        self.assertEqual(result[0].tolist(), ids[::-1].tolist())
        self.assertTrue(result[2]["reference_only"])

    def test_flat_reward_is_mean_over_nodes_without_extra_leaf_weight(self):
        runtime, state = self.fixture()
        runtime.cfg.pack_mechanism_ablation = "flat_dependency_support"
        # Make only the root supported by S0: node mean=1/2; previous leaf-
        # weighted implementation gave 1/4 and changed the tested mechanism.
        state["evidence_candidates"][2]["verified"] = []
        state["_evidence_beams"][0]["proofs"].pop("b")
        ids, scores = np.arange(220), np.linspace(1, 0, 220)
        result = BudgetedSelector(runtime, EvidenceScorer(False)).select(
            "Other", ids, scores, (ids, scores, state["evidence_trace"]), state)
        diag = result[2]["budgeted_selection"]
        self.assertEqual(diag["reward_support"], .5)
        self.assertEqual(diag["closure_reward_mode"], "flat")

    def test_parent_descriptions_use_no_intermediate_answers(self):
        _, state = self.fixture()
        text = describe_parent_questions(state["_evidence_plan"][1]["question"], state["_evidence_plan"])
        self.assertIn("Who directed Sample Film", text)
        self.assertNotIn("Alice Example", text)
        self.assertNotIn("${", text)

    def test_parent_executor_keeps_dag_and_extraction_but_never_propagates_answers(self):
        routes = []
        cfg = BindPackConfig.from_file(mechanism_ablation="wo_parent_answer_propagation")
        runtime = SimpleNamespace(cfg=cfg, rank_constant=10., _verification_diagnostics={},
            _document=lambda d: "Alice Example is a person.",
            _collect_jobs=lambda executor, jobs: [f(*args) for f, args in jobs],
            _candidate=lambda s, d: s["evidence_candidates"].setdefault(d, {"verified": []}))
        def route(s, source, goal, query, requirements):
            routes.append((source, query, requirements))
            return [0, 1, 2]
        runtime._add_route = route
        verifier = SimpleNamespace(verify=lambda q, kind, docs: (
            [{"answer": "Alice Example", "doc_id": 0, "confidence": .9}], [], None))
        plan = [{"id": "a", "question": "Who directed Sample Film?", "depends_on": [], "answer_type": "person"},
                {"id": "b", "question": "Where was ${a.answer} born?", "depends_on": ["a"], "answer_type": "place"}]
        state = {"_evidence_plan": deepcopy(plan), "evidence_candidates": {},
                 "evidence_trace": {"llm_verification_calls": 0, "rejected_hypotheses": [], "semantic_failures": []}}
        DependencyExecutor(runtime, verifier).execute([state], None)
        self.assertEqual(state["_evidence_plan"], plan)
        self.assertEqual(len(routes), 2)
        self.assertTrue(all(not r[2] and "Alice Example" not in r[1] for r in routes))
        self.assertEqual(state["evidence_trace"]["dependency_edges_executed"], 1)
        self.assertFalse(state["evidence_trace"]["parent_answer_propagation"])
        self.assertEqual(state["evidence_trace"]["extraction_inputs"][1]["doc_ids"], [0, 1, 2])
        self.assertEqual(state["_evidence_winning_bindings"]["b"], "Alice Example")

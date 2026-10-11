"""Fixed-proposal review controls: no model services or accelerator needed."""

import copy
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType
import unittest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ModuleType("_review_ablation_test_package")
PACKAGE.__path__ = [str(ROOT / "src/bindpack")]
sys.modules[PACKAGE.__name__] = PACKAGE
replay = importlib.import_module(PACKAGE.__name__ + ".experiments.review_ablation")

QUERY = "Where was the director of Aster Film born?"
ROOT_QUOTE = "Aster Film was directed by Alice Example."
CANDIDATE_QUOTE = "Alice Example was born in Sample Town in 1960."


def fixture(*, approve=True, protect=False, omit_review=False, abstain=False,
            proposer_overflow=False, reviewer_overflow=False):
    documents = {i: f"Other {i}\nAn unrelated paragraph describes a different topic."
                 for i in range(20)}
    documents[0] = "Aster Film\n" + ROOT_QUOTE
    documents[5] = "Alice Example\n" + CANDIDATE_QUOTE
    plan = [
        {"id": "s1", "question": "Who directed Aster Film?", "depends_on": [], "answer_type": "person"},
        {"id": "s2", "question": "Where was ${s1.answer} born?", "depends_on": ["s1"], "answer_type": "place"},
    ]
    bindings = {"s1": "Alice Example"}
    proofs = {"s1": {"doc_id": 0, "answer": "Alice Example", "evidence": ROOT_QUOTE}}
    if protect:
        documents[4] = "Existing source\nAlice Example was born in Another Town."
        bindings["s2"] = "Another Town"
        proofs["s2"] = {"doc_id": 4, "answer": "Another Town",
                        "evidence": "Alice Example was born in Another Town."}
    evidence = {"plan": plan, "bindings": bindings,
                "branch_scores": [{"bindings": bindings, "proofs": proofs}]}
    state = {"evidence_trace": evidence, "_evidence_plan": plan,
             "_evidence_winning_bindings": bindings, "_evidence_beams": evidence["branch_scores"]}
    old = list(range(20))
    messages = replay.build_joint_prompt(QUERY, old, state, documents.__getitem__)
    response = json.dumps({"swap": None} if abstain else {"swap": {
        "promote": "D5", "replace": "D4", "confidence": .99, "root": "D0",
        "missing_condition": "Birthplace of the film director",
        "candidate_quote": CANDIDATE_QUOTE, "root_quote": ROOT_QUOTE,
        "reason": "The candidate adds birthplace; the selected film director evidence remains.",
    }})
    proposer = {"preflight": {"messages_sha256": replay._hash(messages),
        "allowed": not proposer_overflow, "prompt_tokens": 100,
        "context_limit": 8192, "output_token_budget": 2048},
        "output": None if proposer_overflow else {"response": response,
            "metadata": {"finish_reason": "stop", "prompt_tokens": 100, "completion_tokens": 80},
            "cache_hit": False}}
    proposed, proposal = replay.apply_joint_selection(
        QUERY, old, state, documents.__getitem__,
        response if not proposer_overflow else '{"swap":null}', "stop")
    if proposer_overflow:
        proposal = dict(proposal, abstain_reason="proposer_prompt_exceeds_context_limit")
    guard_allowed, guard = replay.support_guard(QUERY, old, state, documents.__getitem__, proposal)
    review_messages = replay.build_verifier_prompt(QUERY, old, state, documents.__getitem__, proposal)
    review_text = json.dumps({"root": "D0", "candidate": "D5", "victim": "D4",
        "same_entity_event": True, "relation_grounded": True,
        "adds_missing_constraint": True, "removal_safe": approve,
        "old_missing_fact": "The director birthplace is absent.",
        "candidate_supplied_fact": "Alice Example was born in Sample Town.",
        "victim_reason": "D0 retains the director and the other facts remain selected.",
        "reason": "The literal sources concern the same director and bridge."})
    reviewer = {"preflight": {"messages_sha256": replay._hash(review_messages),
        "allowed": not reviewer_overflow, "prompt_tokens": 70,
        "context_limit": 8192, "output_token_budget": 2048} if review_messages else None,
        "output": None if (omit_review or reviewer_overflow or proposed == old) else {
            "response": review_text,
            "metadata": {"finish_reason": "stop", "prompt_tokens": 70, "completion_tokens": 60},
            "cache_hit": True}}
    verified, verification = replay.validate_verification(
        QUERY, old, state, documents.__getitem__, proposal,
        reviewer["output"]["response"] if reviewer["output"] else '{}', "stop")
    if reviewer_overflow:
        verification = dict(verification, rejection_reason="reviewer_prompt_exceeds_context_limit")
    accepted = proposed != old and guard_allowed and verified
    evidence["improvement_support_semantic_veto"] = {
        "original_parent_doc_ids": old, "final_doc_ids": proposed if accepted else old,
        "accepted": accepted, "base_proposal": proposal, "support_guard": guard,
        "independent_verification": verification,
        "llm_stages": {"proposer": proposer, "reviewer": reviewer},
    }
    return {"question": QUERY, "retrieval_trace": {"evidence": evidence}}, documents, review_text


class FrozenReviewAblationTests(unittest.TestCase):
    def test_all_four_accept_same_safe_approved_source_validated_swap(self):
        row, documents, _ = fixture()
        result = replay.run_review_ablations(row, documents)
        self.assertEqual(result["status"], "resolved")
        self.assertTrue(result["both_matches_recorded"])
        self.assertFalse(result["gold_labels_used"])
        self.assertEqual(result["http_calls_issued_by_replay"], 0)
        for condition in result["conditions"].values():
            self.assertTrue(condition["accepted"])
            self.assertEqual(condition["final_doc_ids"][:5], [0, 1, 2, 3, 5])
            self.assertTrue(condition["top2_preserved"])
            self.assertTrue(condition["top10_set_preserved"])
            self.assertTrue(condition["suffix_preserved"])
        self.assertEqual(result["frozen_logical_requests"], 2)
        self.assertEqual(result["frozen_acquisition_cost"]["reviewer"]["completion_tokens"], 60)

    def test_semantic_veto_changes_only_review_enabled_conditions(self):
        row, documents, _ = fixture(approve=False)
        result = replay.run_review_ablations(row, documents)
        conditions = result["conditions"]
        self.assertTrue(conditions["none"]["accepted"])
        self.assertTrue(conditions["guard_only"]["accepted"])
        self.assertFalse(conditions["veto_only"]["accepted"])
        self.assertFalse(conditions["both"]["accepted"])
        self.assertTrue(result["both_matches_recorded"])

    def test_guard_veto_changes_only_guard_enabled_conditions(self):
        row, documents, _ = fixture(protect=True)
        result = replay.run_review_ablations(row, documents)
        conditions = result["conditions"]
        self.assertTrue(conditions["none"]["accepted"])
        self.assertFalse(conditions["guard_only"]["accepted"])
        self.assertTrue(conditions["veto_only"]["accepted"])
        self.assertFalse(conditions["both"]["accepted"])
        self.assertIn(4, result["support_guard"]["protected_source_doc_ids"])

    def test_guard_rejected_missing_review_is_unknown_for_veto_only(self):
        row, documents, _ = fixture(protect=True, omit_review=True)
        result = replay.run_review_ablations(row, documents)
        self.assertEqual(result["review_status"], "needs_review")
        self.assertEqual(result["conditions"]["veto_only"]["status"], "needs_review")
        self.assertIsNone(result["conditions"]["veto_only"]["final_doc_ids"])
        self.assertEqual(result["conditions"]["both"]["status"], "resolved")
        self.assertTrue(result["both_matches_recorded"])
        request = replay.build_frozen_review_request(row, documents)
        self.assertEqual(request["status"], "needs_review")
        self.assertIsNone(request["recorded_output"])
        self.assertEqual(request["request"]["response_format"], {"type": "json_object"})

    def test_guard_allowed_missing_review_leaves_both_unresolved(self):
        row, documents, _ = fixture(omit_review=True)
        result = replay.run_review_ablations(row, documents)
        self.assertIsNone(result["both_matches_recorded"])
        self.assertIsNone(result["conditions"]["both"]["accepted"])
        self.assertIsNone(result["conditions"]["both"]["final_doc_ids"])

    def test_real_supplementary_review_resolves_counterfactual_with_same_prompt(self):
        row, documents, review_text = fixture(protect=True, omit_review=True)
        request = replay.build_frozen_review_request(row, documents)
        row["frozen_review"] = {"messages_sha256": request["request"]["messages_sha256"],
            "output": {"response": review_text, "metadata": {
                "finish_reason": "stop", "prompt_tokens": 70, "completion_tokens": 60},
                "cache_hit": False}}
        result = replay.run_review_ablations(row, documents)
        self.assertEqual(result["status"], "resolved")
        self.assertEqual(result["review_origin"], "supplemental_frozen_response")
        self.assertTrue(result["conditions"]["veto_only"]["accepted"])
        self.assertTrue(result["both_matches_recorded"])

    def test_supplement_with_different_prompt_is_rejected(self):
        row, documents, review_text = fixture(protect=True, omit_review=True)
        row["frozen_review"] = {"messages_sha256": "wrong", "output": {"response": review_text}}
        with self.assertRaisesRegex(ValueError, "exact production prompt"):
            replay.run_review_ablations(row, documents)

    def test_abstention_retains_all_rankings_without_synthesizing_review(self):
        row, documents, _ = fixture(abstain=True)
        result = replay.run_review_ablations(row, documents)
        self.assertEqual(result["review_status"], "not_required")
        self.assertTrue(result["both_matches_recorded"])
        for condition in result["conditions"].values():
            self.assertEqual(condition["final_doc_ids"], list(range(20)))
            self.assertFalse(condition["accepted"])
        self.assertIsNone(replay.build_frozen_review_request(row, documents)["request"])

    def test_original_proposer_context_abstention_is_reproduced(self):
        row, documents, _ = fixture(proposer_overflow=True)
        result = replay.run_review_ablations(row, documents)
        self.assertTrue(result["both_matches_recorded"])
        self.assertEqual(result["frozen_logical_requests"], 0)
        self.assertEqual(result["base_proposal"]["abstain_reason"], "proposer_prompt_exceeds_context_limit")

    def test_original_reviewer_context_abstention_is_not_missing_response(self):
        row, documents, _ = fixture(reviewer_overflow=True)
        result = replay.run_review_ablations(row, documents)
        self.assertEqual(result["review_status"], "context_blocked")
        self.assertEqual(result["status"], "resolved")
        self.assertTrue(result["both_matches_recorded"])
        self.assertTrue(result["conditions"]["none"]["accepted"])
        self.assertFalse(result["conditions"]["both"]["accepted"])

    def test_mutated_source_is_rejected_by_original_prompt_identity(self):
        row, documents, _ = fixture()
        documents[3] += " Added text."
        with self.assertRaisesRegex(ValueError, "proposer prompt"):
            replay.run_review_ablations(row, documents)

    def test_missing_proposer_response_is_explicitly_unresolved(self):
        row, documents, _ = fixture()
        row["retrieval_trace"]["evidence"]["improvement_support_semantic_veto"]["llm_stages"]["proposer"]["output"] = None
        result = replay.run_review_ablations(row, documents)
        self.assertEqual(result["status"], "needs_proposal")
        self.assertIsNone(result["conditions"]["none"]["final_doc_ids"])

    def test_full_original_ranking_mismatch_is_rejected(self):
        row, documents, _ = fixture()
        row["retrieval_trace"]["evidence"]["improvement_support_semantic_veto"]["final_doc_ids"] = list(range(20))
        with self.assertRaisesRegex(ValueError, "differs from the original"):
            replay.run_review_ablations(row, documents)

    def test_json_document_keys_and_input_immutability(self):
        row, documents, _ = fixture()
        before = copy.deepcopy(row)
        direct = replay.run_review_ablations(row, documents)
        serialized = replay.run_review_ablations(row, {str(k): v for k, v in documents.items()})
        self.assertEqual(direct, serialized)
        self.assertEqual(row, before)


if __name__ == "__main__":
    unittest.main()

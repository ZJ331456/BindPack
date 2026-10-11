"""CPU mechanism tests over real package eligibility and frozen ranking controls."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
import unittest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ModuleType("fixed_selection_test_package")
PACKAGE.__path__ = [str(ROOT / "src/bindpack")]
sys.modules[PACKAGE.__name__] = PACKAGE
SPEC = importlib.util.spec_from_file_location(
    PACKAGE.__name__ + ".experiments.fixed_selection", ROOT / "src/bindpack/experiments/fixed_selection.py")
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def snapshot_for(nodes, bindings, rows, count=40):
    documents = {d: f"Unrelated title {d}\nAn unrelated passage number {d}." for d in range(count)}
    candidates = {d: {"verified": []} for d in range(count)}
    proofs = {}
    for nid, doc_id, title, quote in rows:
        documents[doc_id] = title + "\n" + quote
        proof = {"doc_id": doc_id, "answer": bindings[nid], "evidence": quote, "confidence": .95}
        proofs[nid] = proof
        candidates[doc_id]["verified"].append(dict(proof, goal="dag:" + nid))
    return {"schema_version": 1, "query": "A question without gold labels", "candidate_ids": list(range(count)),
        "candidate_scores": [1 - d / count for d in range(count)], "score_source": "pre_selection_merged",
        "documents": documents,
        "state": {"_evidence_plan": nodes, "_evidence_winning_bindings": bindings,
                  "_evidence_beams": [{"bindings": bindings, "proofs": proofs}], "evidence_candidates": candidates,
                  "evidence_trace": {"selected_prefix": [{"doc_id": d} for d in range(5)]}}}


def chain_with_orphan():
    nodes = [{"id": "a", "question": "Who directed Example Film?", "depends_on": [], "answer_type": "person"},
        {"id": "b", "question": "Where was ${a.answer} born?", "depends_on": ["a"], "answer_type": "place"},
        {"id": "c", "question": "Where is ${b.answer} located?", "depends_on": ["b"], "answer_type": "place"},
        {"id": "d", "question": "Who directed Another Film?", "depends_on": [], "answer_type": "person"}]
    bindings = {"a": "Alice Example", "b": "Sample Town", "c": "Sample County", "d": "Bob Example"}
    rows = [("a", 20, "Example Film", "Example Film was directed by Alice Example."),
            ("b", 21, "Alice Example", "Alice Example was born in Sample Town."),
            ("c", 2, "Sample Town", "Sample Town is located in Sample County."),
            ("d", 30, "Another Film", "Another Film was directed by Bob Example.")]
    return snapshot_for(nodes, bindings, rows)


class FixedSelectionTests(unittest.TestCase):
    def test_current_closed_exactly_replays_production_without_mutation(self):
        snapshot = chain_with_orphan()
        before = copy.deepcopy(snapshot)
        expected, diag = module.dag.rerank_dag_packages(snapshot["candidate_ids"], snapshot["state"], snapshot["documents"].get)
        output = module.run_fixed_selection(snapshot)
        actual = output["variants"]["current_closed"]
        self.assertEqual(actual["ids"], expected)
        self.assertEqual(actual["diagnostics"]["eligible_proofs"], diag["eligible_proofs"])
        self.assertEqual(actual["diagnostics"]["prefix_optimization"], diag["prefix_optimization"])
        self.assertTrue(actual["diagnostics"]["production_selector_replayed"])
        self.assertEqual(snapshot, before)

    def test_orphan_leaf_cannot_get_closed_reward_and_flat_is_real_control(self):
        output = module.run_fixed_selection(chain_with_orphan())["variants"]
        closed = output["current_closed"]
        flat = output["current_flat"]
        self.assertEqual(closed["ids"][:5], [0, 1, 2, 20, 21])
        self.assertEqual(flat["ids"][:5], [0, 1, 2, 20, 30])
        self.assertEqual(closed["diagnostics"]["prefix_optimization"][0]["coverage_before"], [])
        self.assertEqual(flat["diagnostics"]["prefix_optimization"][0]["coverage_before"], ["c"])
        self.assertEqual(closed["diagnostics"]["eligible_proofs"], flat["diagnostics"]["eligible_proofs"])
        self.assertEqual(closed["diagnostics"]["packages"], flat["diagnostics"]["packages"])

    def test_string_literal_relation_levels_do_not_fabricate_missing_proofs(self):
        snapshot = chain_with_orphan()
        # Existing proposal mentions the same answer, but its quotation is made up.
        for proof in [snapshot["state"]["_evidence_beams"][0]["proofs"]["b"],
                      snapshot["state"]["evidence_candidates"][21]["verified"][0]]:
            proof["evidence"] = "Alice Example was born in Sample Town in a made up quotation."
        variants = module.run_fixed_selection(snapshot)["variants"]
        self.assertIn("b", variants["weak_closed"]["diagnostics"]["eligible_proofs"])
        for name in ("literal_closed", "current_closed"):
            self.assertNotIn("b", variants[name]["diagnostics"]["eligible_proofs"])
            self.assertNotIn(21, variants[name]["ids"][:10])
        snapshot["state"]["_evidence_beams"][0]["proofs"].pop("b")
        snapshot["state"]["evidence_candidates"][21]["verified"] = []
        variants = module.run_fixed_selection(snapshot)["variants"]
        self.assertTrue(all("b" not in variants[n]["diagnostics"]["eligible_proofs"]
                            for n in variants if "eligible_proofs" in variants[n]["diagnostics"]))

    def test_unknown_relation_is_unsupported_not_factual_false(self):
        snapshot = chain_with_orphan()
        snapshot["state"]["_evidence_plan"][1]["question"] = "Which town did ${a.answer} mention?"
        output = module.run_fixed_selection(snapshot)["variants"]
        self.assertIn("b", output["literal_closed"]["diagnostics"]["eligible_proofs"])
        self.assertNotIn("b", output["current_closed"]["diagnostics"]["eligible_proofs"])
        rejected = output["current_closed"]["diagnostics"]["rejected_proofs"]
        self.assertTrue(any(r["node"] == "b" and r["epistemic_status"] == "unknown" for r in rejected))
        self.assertFalse(any(r.get("epistemic_status") == "false" for r in rejected))

    def test_json_roundtrip_restores_document_candidate_keys(self):
        snapshot = chain_with_orphan()
        actual = module.run_fixed_selection(json.loads(json.dumps(snapshot)))
        expected = module.run_fixed_selection(snapshot)
        for name in actual["variants"]:
            self.assertEqual(actual["variants"][name]["ids"], expected["variants"][name]["ids"])

    def test_missing_document_and_final_order_scores_are_rejected(self):
        snapshot = chain_with_orphan()
        snapshot["documents"].pop(3)
        with self.assertRaisesRegex(ValueError, "source document"):
            module.run_fixed_selection(snapshot)
        for source in (None, "final_output", "synthetic_order"):
            snapshot = chain_with_orphan()
            snapshot["score_source"] = source
            with self.assertRaisesRegex(ValueError, "pre-selection score_source"):
                module.run_fixed_selection(snapshot)

    def test_cross_branch_proof_cannot_become_weak_binding(self):
        snapshot = chain_with_orphan()
        for proof in [snapshot["state"]["_evidence_beams"][0]["proofs"]["b"],
                      snapshot["state"]["evidence_candidates"][21]["verified"][0]]:
            proof["requirements"] = {"a": "Different Person"}
        variants = module.run_fixed_selection(snapshot)["variants"]
        for name in ("weak_closed", "literal_closed", "current_closed"):
            self.assertNotIn("b", variants[name]["diagnostics"]["eligible_proofs"])
            self.assertTrue(any(r["epistemic_status"] == "inconsistent_with_frozen_binding"
                                for r in variants[name]["diagnostics"]["rejected_proofs"]))

    def test_relevance_mmr_keep_same_pool_and_fixed_front_without_gold(self):
        snapshot = chain_with_orphan()
        snapshot["candidate_scores"][30] = 2.
        snapshot["gold_documents"] = [11]  # Ignored by every selector.
        variants = module.run_fixed_selection(snapshot)["variants"]
        self.assertEqual(variants["relevance"]["ids"][:3], [0, 1, 30])
        for result in variants.values():
            self.assertEqual(result["ids"][:2], [0, 1])
            self.assertEqual(set(result["ids"]), set(range(40)))
            self.assertFalse(result["diagnostics"]["gold_labels_used"])
        self.assertFalse(variants["mmr_lexical"]["diagnostics"]["dense_mmr"])

    def test_invalid_plan_abstains_all_structure_controls(self):
        snapshot = chain_with_orphan()
        snapshot["state"]["_evidence_plan"][0]["depends_on"] = ["c"]
        variants = module.run_fixed_selection(snapshot)["variants"]
        for name in ("weak_flat", "weak_closed", "literal_flat", "literal_closed", "current_flat", "current_closed"):
            self.assertEqual(variants[name]["ids"], snapshot["candidate_ids"])
            self.assertEqual(variants[name]["diagnostics"]["plan_error"], "cyclic_dependencies")

    def test_ordinary_coverage_is_separate_reference_with_same_qualified_proofs(self):
        output = module.run_fixed_selection(chain_with_orphan())
        ordinary = output["variants"]["ordinary_node_coverage_greedy"]
        closed = output["variants"]["current_closed"]
        self.assertEqual(ordinary["diagnostics"]["eligible_proofs"], closed["diagnostics"]["eligible_proofs"])
        self.assertNotIn("ordinary_node_coverage_greedy", output["factorial_variant_names"])
        self.assertFalse(ordinary["diagnostics"]["controlled_factorial_member"])
        self.assertFalse(ordinary["diagnostics"]["package_feasibility_required"])
        self.assertFalse(ordinary["diagnostics"]["closure_required_for_reward"])
        self.assertEqual(ordinary["ids"][:2], [0, 1])
        self.assertIn(2, ordinary["ids"][:5])

    def test_frozen_hashes_ignore_gold_and_change_with_actual_input(self):
        snapshot = chain_with_orphan()
        first = module.run_fixed_selection(snapshot)["frozen_hashes"]
        snapshot["gold_documents"] = [9]
        self.assertEqual(first, module.run_fixed_selection(snapshot)["frozen_hashes"])
        snapshot["candidate_scores"][3] += .01
        second = module.run_fixed_selection(snapshot)["frozen_hashes"]
        self.assertNotEqual(first["candidate_scores_sha256"], second["candidate_scores_sha256"])
        self.assertNotEqual(first["snapshot_sha256"], second["snapshot_sha256"])
        self.assertEqual(first["candidate_pool_sha256"], second["candidate_pool_sha256"])


if __name__ == "__main__":
    unittest.main()

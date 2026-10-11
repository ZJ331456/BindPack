"""Offline metric identity, pairing and bootstrap contrast tests."""

import importlib
from pathlib import Path
import sys
from types import ModuleType
import unittest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ModuleType("_ablation_reporting_test_package")
PACKAGE.__path__ = [str(ROOT / "src/bindpack")]
sys.modules[PACKAGE.__name__] = PACKAGE
reporting = importlib.import_module(PACKAGE.__name__ + ".experiments.reporting")


def row(index, value):
    complete = value == 1.
    return {"query_index": index,
            "metrics": {**{f"Recall@{k}": value for k in reporting.RECALL_K},
                        "PoolAllGold@200": float(complete)},
            "all_gold_top5": complete, "all_gold_top10": complete,
            "pool_all_gold_top200": complete}


class RankingMetricTests(unittest.TestCase):
    def test_exact_full_source_matching_does_not_merge_same_titles(self):
        docs = {0: "Same title\nWrong paragraph", 1: "Same title\nGold paragraph",
                2: "Other title\nSecond gold paragraph"}
        result = reporting.score_ranking(7, [0, 1, 2], docs.__getitem__, [docs[1], docs[2]])
        self.assertEqual(result["metrics"]["Recall@1"], 0.)
        self.assertEqual(result["metrics"]["Recall@2"], .5)
        self.assertEqual(result["metrics"]["Recall@5"], 1.)
        self.assertTrue(result["all_gold_top5"])
        self.assertEqual(sorted(result["gold_document_ranks"]), [2, 3])

    def test_gold_duplicates_and_equal_source_physical_ids_count_only_once(self):
        docs = {0: "Gold first", 1: "Gold first", 2: "Gold second"}
        result = reporting.score_ranking(1, [0, 1], docs.__getitem__, ["Gold first", "Gold first", "Gold second"])
        self.assertEqual(result["n_gold_documents"], 2)
        self.assertEqual(result["metrics"]["Recall@2"], .5)
        self.assertFalse(result["pool_all_gold_top200"])

    def test_short_or_empty_rankings_are_valid_zero_coverage_cases(self):
        result = reporting.score_ranking(0, [], lambda _: "unused", ["gold source"])
        self.assertTrue(all(value == 0 for value in result["metrics"].values()))
        self.assertEqual(result["gold_document_ranks"], [None])
        self.assertFalse(result["all_gold_top10"])

    def test_pool_coverage_is_bounded_at_200_even_with_longer_input(self):
        docs = {i: f"source {i}" for i in range(201)}
        result = reporting.score_ranking(0, list(docs), docs.__getitem__, [docs[200]])
        self.assertEqual(result["metrics"]["Recall@200"], 0.)
        self.assertEqual(result["metrics"]["PoolAllGold@200"], 0.)
        self.assertEqual(result["gold_document_ranks"], [201])

    def test_invalid_gold_ids_or_missing_documents_fail_explicitly(self):
        for gold in ([], [""], ["  "], [None]):
            with self.subTest(gold=gold), self.assertRaises(ValueError):
                reporting.score_ranking(0, [], lambda _: "source", gold)
        for ids in ([0, 0], [True], ["0"], [-1]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                reporting.score_ranking(0, ids, lambda _: "source", ["gold"])
        with self.assertRaisesRegex(ValueError, "source text"):
            reporting.score_ranking(0, [0], lambda _: None, ["gold"])

    def test_aggregate_macro_average_and_unique_sample_count(self):
        rows = [row(4, .25), row(9, .75)]
        result = reporting.aggregate_rows(rows)
        self.assertEqual(result["n_samples"], 2)
        self.assertEqual(result["retrieval_metrics"]["Recall@5"], .5)
        self.assertEqual(result["pool_all_gold_top200"], 0.)
        self.assertEqual(result["per_question"], rows)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            reporting.aggregate_rows([row(4, .25), row(4, .75)])
        with self.assertRaises(ValueError):
            reporting.aggregate_rows([])


class PairedBootstrapTests(unittest.TestCase):
    def test_pairing_uses_query_id_and_reports_right_minus_left_pp(self):
        left = [row(9, .5), row(4, .25)]
        right = [row(4, .5), row(9, .75)]
        result = reporting.paired_bootstrap(left, right, n_bootstrap=200)
        metric = result["metrics"]["Recall@5"]
        self.assertEqual(result["query_indices"], [4, 9])
        self.assertEqual(metric["left"], .375)
        self.assertEqual(metric["right"], .625)
        self.assertEqual(metric["delta_pp"], 25.)
        self.assertEqual(metric["ci95_pp"], [25., 25.])
        self.assertTrue(result["small_sample_warning"])
        self.assertFalse(result["claims_statistical_significance"])

    def test_identical_pairings_have_zero_interval_even_with_query_variance(self):
        left = [row(i, i / 3) for i in range(4)]
        right = list(reversed(left))
        result = reporting.paired_bootstrap(left, right, n_bootstrap=200)
        for metric in result["metrics"].values():
            self.assertEqual(metric["delta_pp"], 0.)
            self.assertEqual(metric["ci95_pp"], [0., 0.])

    def test_partial_query_overlap_is_rejected_instead_of_silent_selection(self):
        with self.assertRaisesRegex(ValueError, "identical query_index sets"):
            reporting.paired_bootstrap([row(0, .5), row(1, .5)], [row(0, 1.)])

    def test_bootstrap_is_reproducible_and_reports_all_gold_intervals(self):
        left = [row(i, 0.) for i in range(50)]
        right = [row(i, 1. if i % 2 else 0.) for i in range(50)]
        a = reporting.paired_bootstrap(left, right, seed=17, n_bootstrap=200)
        b = reporting.paired_bootstrap(left, right, seed=17, n_bootstrap=200)
        self.assertEqual(a, b)
        self.assertEqual(a["n_samples"], 50)
        self.assertFalse(a["small_sample_warning"])
        self.assertEqual(a["metrics"]["AllGold@5"]["delta_pp"], 50.)
        self.assertEqual(a["metrics"]["PoolAllGold@200"]["delta_pp"], 50.)
        low, high = a["metrics"]["AllGold@5"]["ci95_pp"]
        self.assertLess(low, 50.)
        self.assertGreater(high, 50.)

    def test_joint_factorial_resampling_reports_d_minus_c_minus_b_plus_a(self):
        A = [row(i, .25) for i in range(4)]
        B = [row(i, .5) for i in reversed(range(4))]
        C = [row(i, .5) for i in range(4)]
        D = [row(i, 1.) for i in range(4)]
        result = reporting.interaction_bootstrap(A, B, C, D, n_bootstrap=200)
        metric = result["metrics"]["Recall@5"]
        self.assertEqual(result["contrast"], "D-C-B+A")
        self.assertEqual(metric["interaction_pp"], 25.)
        self.assertEqual(metric["ci95_pp"], [25., 25.])
        self.assertEqual(result["metrics"]["AllGold@5"]["interaction_pp"], 100.)

    def test_invalid_bootstrap_arguments_and_metrics_fail(self):
        for kwargs in ({"seed": -1}, {"seed": True}, {"n_bootstrap": 1}, {"n_bootstrap": 2.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                reporting.paired_bootstrap([row(0, .5)], [row(0, .5)], **kwargs)
        bad = row(0, .5)
        bad["metrics"]["Recall@5"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Invalid"):
            reporting.aggregate_rows([bad])


class AnnotationExportTests(unittest.TestCase):
    def test_checker_rejection_exports_blank_labels_without_claiming_false_fact(self):
        snapshot = {"query_index": 3, "query": "Original question", "documents": {5: "Full source"},
            "state": {"_evidence_plan": [{"id": "s1", "question": "Subquestion"}]},
            "variants": {
                "weak_closed": {"diagnostics": {"eligible_proofs": {"s1": [{"doc_id": 5}]}}},
                "current_closed": {"diagnostics": {"rejected_proofs": [
                    {"node": "s1", "doc_id": 5, "reason": "unsupported_relation"}]}}}}
        items = reporting.annotation_items(snapshot)
        self.assertEqual(len(items), 1)
        item = items[0]
        self.assertEqual(item["suggested_diagnostic_status"], "unknown")
        self.assertTrue(item["weak_eligible"])
        self.assertFalse(item["current_eligible"])
        self.assertEqual(item["source_text"], "Full source")
        self.assertEqual(item["node_question"], "Subquestion")
        self.assertEqual(item["human_source_support"], "")
        self.assertEqual(item["human_factual_correctness"], "")
        self.assertFalse(item["gold_labels_used"])

    def test_structural_rejection_and_duplicate_exports_are_distinguished(self):
        rejected = {"node": "s2", "doc_id": None, "reason": "invalid_proof_fields"}
        snapshot = {"variants": {
            "weak_closed": {"diagnostics": {"rejected_proofs": [rejected]}},
            "current_closed": {"diagnostics": {"rejected_proofs": [rejected]}}}}
        items = reporting.annotation_items(snapshot)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["suggested_diagnostic_status"], "structural_invalid")


if __name__ == "__main__":
    unittest.main()

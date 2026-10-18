"""Reporting integrity and non-interference checks, without model imports."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("bindpack_report_cli_tests", ROOT / "scripts/report_metrics.py")
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)
REPORT = CLI._load_report_module()


class MetricReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.case = self.root / "cases/hotpotqa/full"
        self.metadata = self.root / "metadata/batch/hotpotqa"
        self.case.mkdir(parents=True)
        self.metadata.mkdir(parents=True)
        self.data_path = self.root / "benchmark.json"
        self.source_corpus = self.root / "source_corpus.json"
        self.docs = ["A\nAlice works at Lab."] + [f"Filler{i}\nirrelevant" for i in range(5)] + ["B\nLab is in Paris."]
        self.sample = {"_id": "sample-1", "question": "Where is Alice's employer?", "answer": "Paris",
                       "context": [["A", ["Alice works at Lab."]], ["B", ["Lab is in Paris."]]],
                       "supporting_facts": [["A", 0], ["B", 0]]}
        self.write(self.data_path, [self.sample])
        corpus = [{"title": doc.partition("\n")[0], "text": doc.partition("\n")[2]} for doc in self.docs]
        self.write(self.source_corpus, corpus)
        self.write(self.metadata / "corpus.json", corpus)
        identity = {"dataset": "hotpotqa", "data_path": str(self.data_path),
            "data_sha256": REPORT._sha256(self.data_path), "selected_indices": [0], "sample_ids": ["sample-1"],
            "corpus_source": str(self.source_corpus), "corpus_source_sha256": REPORT._sha256(self.source_corpus),
            "corpus_docs_sha256": REPORT._digest(self.docs), "corpus_docs": len(self.docs)}
        self.write(self.metadata / "sample_manifest.json", identity)
        row = {"question": self.sample["question"], "query_index": 0, "sample_id": "sample-1",
            "answer": "The Paris", "gold_answers": ["Paris"], "gold_docs": [self.docs[0], self.docs[-1]],
            "docs": self.docs[:5], "candidate_docs": self.docs,
            "gold_document_ranks": [{"doc": self.docs[0], "rank": 1}, {"doc": self.docs[-1], "rank": 7}]}
        self.result = {"dataset": "hotpotqa", "stage_ablation": "full", "selected_indices": [0],
            "data_path": str(self.data_path), "corpus_path": str(self.metadata / "corpus.json"),
            "eval_mode": "rag_qa", "runtime_config": {"qa_top_k": 5}, "results": [row],
            "qa_metrics": {"ExactMatch": 1.0, "F1": 1.0}, "retrieval_metrics": {"Recall@5": .5}}
        self.write(self.case / "protocol.json", {"immutable": True})
        self.save_result()

    @staticmethod
    def write(path, value):
        Path(path).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def save_result(self):
        self.write(self.case / "result.json", self.result)
        self.write(self.case / "validated.ok", {"result_sha256": REPORT._sha256(self.case / "result.json"),
            "protocol_sha256": REPORT._digest(json.loads((self.case / "protocol.json").read_text()))})

    def test_report_aggregates_and_keeps_original_artifacts_unchanged(self):
        before = {name: (self.case / name).read_bytes() for name in ("result.json", "protocol.json", "validated.ok")}
        report = REPORT.write_metric_report(self.case)
        self.assertEqual(report["metrics"]["Recall@5"], .5)
        self.assertEqual(report["metrics"]["FCR@5"], 0)
        self.assertEqual(report["metrics"]["FCR@10"], 1)
        self.assertEqual(report["metrics"]["ExactMatch"], 1)
        self.assertEqual(report["metrics"]["F1"], 1)
        self.assertIsNone(report["metrics"]["DCC@5"])
        self.assertEqual(report["coverage"]["per_k"]["5"]["graph_n"], 0)
        self.assertIn("dataset_has_no_verified_gold_dependency_graph",
                      report["coverage"]["per_k"]["5"]["graph_unavailable_reasons"])
        self.assertTrue(report["sample_alignment"]["frozen_manifest_verified"])
        self.assertEqual(report["metrics_percent"]["Recall@5"], 50)
        for name, blob in before.items():
            self.assertEqual((self.case / name).read_bytes(), blob)
        details = json.loads((self.case / "metrics_details.json").read_text())
        self.assertEqual(details["per_query"][0]["qa"]["ExactMatch"], 1)

    def test_idempotent_refresh_does_not_rewrite_unchanged_report(self):
        first = REPORT.write_metric_report(self.case)
        state = REPORT._state(self.case / "report.json")
        second = REPORT.write_metric_report(self.case)
        self.assertEqual(first, second)
        self.assertEqual(state, REPORT._state(self.case / "report.json"))
        self.result["results"][0]["answer"] = "London"
        self.save_result()
        updated = REPORT.write_metric_report(self.case)
        self.assertEqual(updated["metrics"]["ExactMatch"], 0)
        self.assertNotEqual(updated["inputs"]["result"]["sha256"], first["inputs"]["result"]["sha256"])

    def test_unvalidated_export_waits_then_refreshes_after_receipt(self):
        (self.case / "validated.ok").unlink()
        with self.assertRaisesRegex(ValueError, "not validated"):
            REPORT.write_metric_report(self.case)
        provisional = REPORT.write_metric_report(self.case, require_validated=False)
        self.assertFalse(provisional["validated_result"])
        self.save_result()
        self.assertTrue(REPORT.write_metric_report(self.case)["validated_result"])

    def test_mismatched_result_hash_is_rejected(self):
        self.result["results"][0]["answer"] = "wrong"
        self.write(self.case / "result.json", self.result)
        with self.assertRaisesRegex(ValueError, "SHA-256 differs from validated"):
            REPORT.write_metric_report(self.case)

    def test_original_benchmark_changes_invalidate_cached_report(self):
        REPORT.write_metric_report(self.case)
        self.sample["answer"] = "London"
        self.write(self.data_path, [self.sample])
        with self.assertRaisesRegex(ValueError, "Benchmark SHA-256"):
            REPORT.write_metric_report(self.case)

    def test_sample_alignment_and_corpus_hash_are_enforced(self):
        self.result["results"][0]["sample_id"] = "different"
        self.save_result()
        with self.assertRaisesRegex(ValueError, "sample IDs/order"):
            REPORT.write_metric_report(self.case)
        self.result["results"][0]["sample_id"] = "sample-1"
        self.save_result()
        self.write(self.metadata / "corpus.json", [])
        with self.assertRaisesRegex(ValueError, "canonical corpus differs"):
            REPORT.write_metric_report(self.case)

    def test_hipporag_export_without_data_or_corpus_path(self):
        self.result.pop("data_path")
        self.result.pop("corpus_path")
        self.result["method"] = "hipporag2"
        self.save_result()
        report = REPORT.write_metric_report(self.case)
        self.assertEqual(report["method"], "hipporag2")
        self.assertEqual(report["metrics"]["FCR@10"], 1)

    def test_retrieve_only_does_not_report_fake_zero_answer_metrics(self):
        self.result["eval_mode"] = "retrieve"
        self.save_result()
        report = REPORT.write_metric_report(self.case)
        self.assertIsNone(report["metrics"]["ExactMatch"])
        self.assertEqual(report["coverage"]["qa_n"], 0)
        self.assertEqual(report["coverage"]["qa_unavailable_reasons"], {"answer_generation_not_run": 1})

    def test_smoke_corpus_source_sentinel_is_not_treated_as_a_file(self):
        path = self.metadata / "sample_manifest.json"
        identity = json.loads(path.read_text())
        identity.update(corpus_source="selected_sample_contexts", corpus_source_sha256=None)
        self.write(path, identity)
        self.assertTrue(REPORT.write_metric_report(self.case)["validated_result"])

    def test_standalone_entry_records_provenance_without_inventing_frozen_manifest(self):
        (self.metadata / "sample_manifest.json").unlink()
        (self.case / "validated.ok").unlink()
        report = REPORT.write_metric_report(self.case, require_validated=False)
        self.assertEqual(report["metrics"]["FCR@10"], 1)
        self.assertFalse(report["sample_alignment"]["frozen_manifest_verified"])
        self.assertIn("no prior frozen", report["sample_alignment"]["provenance_note"])

    def test_missing_prediction_is_failure_and_missing_answer_labels_are_unavailable(self):
        self.result["results"][0]["answer"] = None
        self.save_result()
        report = REPORT.write_metric_report(self.case)
        self.assertEqual(report["metrics"]["ExactMatch"], 0)
        self.assertEqual(report["coverage"]["empty_prediction_count"], 1)
        self.sample["answer"] = None
        self.write(self.data_path, [self.sample])
        identity_path = self.metadata / "sample_manifest.json"
        identity = json.loads(identity_path.read_text())
        identity["data_sha256"] = REPORT._sha256(self.data_path)
        self.write(identity_path, identity)
        self.result["results"][0]["gold_answers"] = []
        self.save_result()
        report = REPORT.write_metric_report(self.case)
        self.assertIsNone(report["metrics"]["ExactMatch"])
        self.assertEqual(report["coverage"]["qa_unavailable_reasons"], {"missing_answer_labels": 1})

    def test_cli_is_cpu_only_and_watcher_finishes_after_runner_exit(self):
        completed = subprocess.run([sys.executable, str(ROOT / "scripts/report_metrics.py"),
            "--case", str(self.case), "--watch", "--runner-pid", "99999999", "--interval", ".01"],
            capture_output=True, text=True, timeout=15)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue((self.case / "report.json").is_file())
        self.assertNotIn("torch", CLI._load_report_module().__dict__)

    def test_watcher_recovers_failed_postprocessing_and_clears_its_error_receipt(self):
        self.result["results"][0]["answer"] = "London"
        self.write(self.case / "result.json", self.result)
        failed = subprocess.run([sys.executable, str(ROOT / "scripts/report_metrics.py"),
            "--case", str(self.case)], capture_output=True, text=True, timeout=15)
        self.assertEqual(failed.returncode, 1)
        error = self.case / "report_error.json"
        self.assertTrue(json.loads(error.read_text())["postprocessing_only"])
        self.save_result()
        recovered = subprocess.run([sys.executable, str(ROOT / "scripts/report_metrics.py"),
            "--case", str(self.case), "--watch", "--runner-pid", "99999999"],
            capture_output=True, text=True, timeout=15)
        self.assertEqual(recovered.returncode, 0, recovered.stderr)
        self.assertFalse(error.exists())
        self.assertEqual(json.loads((self.case / "report.json").read_text())["metrics"]["ExactMatch"], 0)

    def test_discovery_ignores_historical_cases_and_popqa_aliases_decode(self):
        (self.case.parent / "old_adaptive_experiment").mkdir()
        (self.case.parent / "wo_pack").mkdir()
        self.assertEqual({path.name for path in REPORT.discover_cases(self.root / "cases")}, {"full", "wo_pack"})
        self.assertEqual(REPORT._answers({"obj": "Paris", "possible_answers": '["City of Paris"]'}),
                         ["City of Paris", "Paris"])

    def test_concurrent_hook_and_watcher_publish_one_complete_pair(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            reports = list(pool.map(REPORT.write_metric_report, [self.case, self.case]))
        self.assertEqual(reports[0], reports[1])
        details = self.case / "metrics_details.json"
        self.assertEqual(reports[0]["details"]["sha256"], REPORT._sha256(details))
        self.assertFalse(list(self.case.glob("*.tmp")))

    def test_musique_report_preserves_dependency_closure_and_matching_subset(self):
        self.docs = ["A\nroot evidence", "B\nsecond evidence", "C\nthird evidence"]
        self.sample = {"id": "sample-1", "question": "Where is Alice's employer?", "answer": "Paris",
            "paragraphs": [{"idx": i, "title": doc.split("\n")[0], "paragraph_text": doc.split("\n")[1],
                            "is_supporting": True} for i, doc in enumerate(self.docs)],
            "question_decomposition": [
                {"id": 100, "question": "Find root", "paragraph_support_idx": 0},
                {"id": 200, "question": "Use #1", "paragraph_support_idx": 1},
                {"id": 300, "question": "Use #2", "paragraph_support_idx": 2}]}
        self.write(self.data_path, [self.sample])
        corpus = [{"title": doc.partition("\n")[0], "text": doc.partition("\n")[2]} for doc in self.docs]
        self.write(self.source_corpus, corpus)
        self.write(self.metadata / "corpus.json", corpus)
        manifest = self.metadata / "sample_manifest.json"
        identity = json.loads(manifest.read_text())
        identity.update(dataset="musique", data_sha256=REPORT._sha256(self.data_path),
            corpus_source_sha256=REPORT._sha256(self.source_corpus), corpus_docs_sha256=REPORT._digest(self.docs),
            corpus_docs=3)
        self.write(manifest, identity)
        self.result["dataset"] = "musique"
        self.result["results"][0].update(candidate_docs=self.docs[1:], docs=self.docs[1:], gold_docs=self.docs,
            gold_document_ranks=[{"doc": self.docs[0], "rank": None}, {"doc": self.docs[1], "rank": 1},
                                 {"doc": self.docs[2], "rank": 2}])
        self.save_result()
        report = REPORT.write_metric_report(self.case, manifest_path=manifest)
        self.assertAlmostEqual(report["metrics"]["NodeCov@5"], 2 / 3)
        self.assertEqual(report["metrics"]["DCC@5"], 0)
        self.assertEqual(report["coverage"]["graph_labels_valid"], 1)
        self.assertEqual(report["dag_subset_document_metrics"]["5"]["n"], 1)
        self.assertAlmostEqual(report["dag_subset_document_metrics"]["5"]["Recall"], 2 / 3)


if __name__ == "__main__":
    unittest.main()

"""HoVer native identity, claim verification, and canonical evidence metrics."""
import ast
import __future__
from collections import Counter
import importlib.util
import json
import logging
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("bindpack_hover_report_tests", ROOT / "scripts/report_metrics.py")
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)
REPORT = CLI._load_report_module()
GOLD = __import__(REPORT.__package__ + ".gold_graph", fromlist=["gold_documents_from_sample"])


class HoverClassificationTests(unittest.TestCase):
    def test_native_labels_are_atomic_and_invalid_outputs_count_as_failures(self):
        metrics, coverage, details = REPORT.score_hover_classification(
            [" supported ", "NOT_SUPPORTED", "not_supported", "The claim is SUPPORTED."],
            ["SUPPORTED", "SUPPORTED", "NOT_SUPPORTED", "NOT_SUPPORTED"])
        self.assertEqual(metrics["ExactMatch"], .5)
        self.assertEqual(metrics["F1"], .5)
        self.assertEqual(metrics["ClassificationAccuracy"], .5)
        self.assertAlmostEqual(metrics["ClassificationMacroF1"], 7 / 12)
        self.assertEqual(coverage["classification"]["invalid_prediction_count"], 1)
        self.assertEqual(coverage["classification"]["confusion_matrix"]["NOT_SUPPORTED"]["INVALID"], 1)
        self.assertEqual(details[-1]["ExactMatch"], 0)
        self.assertFalse(details[-1]["prediction_valid"])
        self.assertIsNone(REPORT.normalize_hover_prediction("REFUTED"))
        self.assertIsNone(REPORT.normalize_hover_prediction("NOT SUPPORTED"))

    def test_missing_predictions_and_empty_run_do_not_gain_credit(self):
        metrics, coverage, _ = REPORT.score_hover_classification([None, " "], ["SUPPORTED", "NOT_SUPPORTED"])
        self.assertEqual(metrics["ClassificationAccuracy"], 0)
        self.assertEqual(metrics["ClassificationMacroF1"], 0)
        self.assertEqual(coverage["empty_prediction_count"], 2)
        metrics, coverage, details = REPORT.score_hover_classification([], [])
        self.assertIsNone(metrics["ClassificationAccuracy"])
        self.assertIsNone(metrics["ClassificationMacroF1"])
        self.assertEqual(coverage["qa_n"], 0)
        self.assertEqual(details, [])

    def test_malformed_native_labels_and_prediction_types_fail_fast(self):
        for label in ("supported", "REFUTED", None, ["SUPPORTED"]):
            with self.assertRaisesRegex(ValueError, "native HoVer label"):
                REPORT.score_hover_classification(["SUPPORTED"], [label])
        with self.assertRaisesRegex(ValueError, "must be a string"):
            REPORT.score_hover_classification([123], ["SUPPORTED"])
        with self.assertRaisesRegex(ValueError, "equal length"):
            REPORT.score_hover_classification([], ["SUPPORTED"])

    def test_reader_transport_accepts_only_native_label_or_exact_answer_object(self):
        for value, expected in [(" supported ", "SUPPORTED"),
                                ('{"answer":"NOT_SUPPORTED"}', "NOT_SUPPORTED"),
                                (' {"answer":"supported"} ', "SUPPORTED")]:
            self.assertEqual(REPORT.parse_hover_reader_response(value), expected)
        for value in ('{"answer":"REFUTED"}', '{"answer":"SUPPORTED","reason":"extra"}',
                      '["SUPPORTED"]', 'Answer: SUPPORTED', '```json\n{"answer":"SUPPORTED"}\n```',
                      '{"answer":true}', '{"answer":"The claim is SUPPORTED"}'):
            parsed = REPORT.parse_hover_reader_response(value)
            self.assertEqual(parsed, value.strip())
            self.assertIsNone(REPORT.normalize_hover_prediction(parsed))


class HoverReaderTests(unittest.TestCase):
    @staticmethod
    def base_reader():
        """Exercise the actual reader methods without importing GPU backends."""
        path = ROOT / "src/bindpack/core/base.py"
        tree = ast.parse(path.read_text())
        base = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BaseRAG")
        selected = ast.ClassDef(name="Reader", bases=[], keywords=[], decorator_list=[],
            body=[node for node in base.body if isinstance(node, ast.FunctionDef) and node.name in {"qa", "rag_qa"}])
        module = ast.fix_missing_locations(ast.Module(body=[selected], type_ignores=[]))
        namespace = {"tqdm": lambda value, **kwargs: value, "QuerySolution": SimpleNamespace,
            "parse_hover_reader_response": REPORT.parse_hover_reader_response,
            "score_hover_classification": REPORT.score_hover_classification,
            "np": SimpleNamespace(max=max), "logger": logging.getLogger(__name__), "json": json, "re": re}
        exec(compile(module, str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)
        return namespace["Reader"]()

    def test_native_reader_prompt_and_atomic_online_metrics_match_report(self):
        reader = self.base_reader()
        reader.global_config = SimpleNamespace(dataset="hover", qa_top_k=1)
        rendered = []
        reader.prompt_template_manager = SimpleNamespace(
            is_template_name_valid=lambda name: name == "rag_qa_hover",
            render=lambda **kwargs: rendered.append(kwargs) or kwargs)
        outputs = iter(['{"answer":"SUPPORTED"}', "Answer: NOT_SUPPORTED"])
        reader.llm_model = SimpleNamespace(infer=lambda messages: (next(outputs), {"stub": True}, False))
        queries = [SimpleNamespace(question="Claim one.", docs=["A\nevidence", "excluded\ntext"]),
                   SimpleNamespace(question="Claim two.", docs=["B\nevidence"])]
        solutions, responses, metadata, _, metrics = reader.rag_qa(
            queries, gold_answers=[["SUPPORTED"], ["NOT_SUPPORTED"]])
        self.assertEqual(metrics["ClassificationAccuracy"], .5)
        self.assertEqual(metrics["F1"], .5)
        self.assertEqual(solutions[0].answer, "SUPPORTED")
        self.assertEqual(solutions[1].answer, "Answer: NOT_SUPPORTED")
        self.assertEqual(metrics["ClassificationMacroF1"], round(1 / 2, 4))
        self.assertEqual(responses, ['{"answer":"SUPPORTED"}', "Answer: NOT_SUPPORTED"])
        self.assertEqual(len(metadata), 2)
        self.assertEqual(rendered[0]["name"], "rag_qa_hover")
        self.assertTrue(rendered[0]["prompt_user"].endswith("Claim: Claim one.\n"))
        self.assertNotIn("Thought:", rendered[0]["prompt_user"])
        self.assertNotIn("excluded", rendered[0]["prompt_user"])

    def test_missing_hover_template_fails_and_original_qa_prompt_parser_are_preserved(self):
        reader = self.base_reader()
        reader.global_config = SimpleNamespace(dataset="hover", qa_top_k=1)
        reader.prompt_template_manager = SimpleNamespace(is_template_name_valid=lambda name: False)
        query = SimpleNamespace(question="A claim.", docs=["A\ntext"])
        with self.assertRaisesRegex(ValueError, "requires the rag_qa_hover"):
            reader.qa([query])
        reader.global_config.dataset = "hotpotqa"
        rendered = []
        reader.prompt_template_manager = SimpleNamespace(is_template_name_valid=lambda name: True,
            render=lambda **kwargs: rendered.append(kwargs) or kwargs)
        reader.llm_model = SimpleNamespace(infer=lambda messages: ("Reasoning.\nAnswer: Paris", {}, False))
        solutions, _, _ = reader.qa([SimpleNamespace(question="Where?", docs=["A\ntext"])])
        self.assertEqual(solutions[0].answer, "Paris")
        self.assertTrue(rendered[0]["prompt_user"].endswith("Question: Where?\nThought: "))

    def test_stratified_hover_metrics_use_the_same_native_classification_rule(self):
        path = ROOT / "scripts/stratified_eval.py"
        tree = ast.parse(path.read_text())
        tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]

        class RetrievalStub:
            def __init__(self, **kwargs):
                pass

            def calculate_metric_scores(self, **kwargs):
                return {f"Recall@{k}": 1.0 for k in kwargs["k_list"]}, []

        namespace = {"score_hover_classification": REPORT.score_hover_classification,
            "np": SimpleNamespace(generic=complex, integer=int), "Counter": Counter,
            "RetrievalRecall": RetrievalStub}
        exec(compile(tree, str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)
        result = namespace["evaluate_stratified"](
            ["Claim one.", "Claim two."], [SimpleNamespace(docs=["a"]), SimpleNamespace(docs=["b"])],
            [["a"], ["b"]], [["SUPPORTED"], ["NOT_SUPPORTED"]], SimpleNamespace(dataset="hover"),
            predicted_answers=["SUPPORTED", "SUPPORTED"], query_hops=[3, 4])
        self.assertEqual(result["overall"]["qa"]["ClassificationAccuracy"], .5)
        self.assertAlmostEqual(result["overall"]["qa"]["ClassificationMacroF1"], 1 / 3)
        self.assertEqual(result["three_hop"]["qa"]["ClassificationAccuracy"], 1)
        self.assertEqual(result["four_hop"]["qa"]["ClassificationAccuracy"], 0)


class HoverMetricReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.case = self.root / "cases/hover/full"
        self.metadata = self.root / "metadata/batch/hover"
        self.case.mkdir(parents=True)
        self.metadata.mkdir(parents=True)
        self.sample = {"hover_id": "native-hover-id", "hotpotqa_id": "not-the-current-id",
            "_id": "legacy-source-id", "question": "Legacy source question must not be used.",
            "claim": "Alice's employer is in Paris.", "label": "NOT_SUPPORTED", "num_hops": 3,
            "hotpotqa_question": "Where is Alice's employer?", "hotpotqa_answer": "Paris",
            "supporting_facts": [["A", 0], ["A", 1], ["B", 0], ["C", 0]],
            "context": [["A", ["Alice works at Lab.", " Lab is a company."]],
                        ["B", ["Lab is in Lyon."]], ["C", ["Lyon is in France."]]]}
        self.docs = [title + "\n" + "".join(sentences) for title, sentences in self.sample["context"]]
        self.data_path = self.root / "hover.json"
        self.write(self.data_path, [self.sample])
        self.write(self.metadata / "corpus.json", self.docs)
        self.identity = {"dataset": "hover", "data_path": str(self.data_path),
            "data_sha256": REPORT._sha256(self.data_path), "selected_indices": [0],
            "sample_ids": [self.sample["hover_id"]], "corpus_docs": len(self.docs),
            "corpus_docs_sha256": REPORT._digest(self.docs)}
        self.write(self.metadata / "sample_manifest.json", self.identity)
        self.result = {"dataset": "hover", "selected_indices": [0], "eval_mode": "rag_qa",
            "data_path": str(self.data_path), "corpus_path": str(self.metadata / "corpus.json"),
            "results": [{"sample_id": self.sample["hover_id"], "query_index": 0,
                "question": self.sample["claim"], "answer": "not_supported",
                "gold_answers": ["NOT_SUPPORTED"], "gold_docs": self.docs, "candidate_docs": self.docs,
                "gold_document_ranks": [{"doc": doc, "rank": index} for index, doc in enumerate(self.docs, 1)]}]}
        self.write(self.case / "protocol.json", {"dataset": "hover", "immutable": True})
        self.save_result()

    @staticmethod
    def write(path, value):
        Path(path).write_text(json.dumps(value), encoding="utf-8")

    def save_result(self):
        self.write(self.case / "result.json", self.result)
        self.write(self.case / "validated.ok", {"result_sha256": REPORT._sha256(self.case / "result.json"),
            "protocol_sha256": REPORT._digest(json.loads((self.case / "protocol.json").read_text()))})

    def test_frozen_native_claim_labels_and_evidence_generate_report_without_mutation(self):
        before = {name: (self.case / name).read_bytes() for name in ("result.json", "protocol.json", "validated.ok")}
        report = REPORT.write_metric_report(self.case)
        self.assertEqual(report["metrics"]["FCR@5"], 1)
        self.assertEqual(report["metrics"]["Recall@5"], 1)
        self.assertEqual(report["metrics"]["ClassificationAccuracy"], 1)
        self.assertEqual(report["metrics"]["ExactMatch"], 1)
        self.assertIsNone(report["metrics"]["DCC@5"])
        self.assertEqual(report["coverage"]["per_k"]["5"]["graph_unavailable_reasons"],
                         {"dataset_has_no_verified_gold_dependency_graph": 1})
        self.assertEqual(report["protocol"]["answer_task"], "binary_claim_verification")
        self.assertIn("hotpotqa_answer is not", report["protocol"]["classification_gold"])
        details = json.loads((self.case / "metrics_details.json").read_text())
        self.assertEqual(details["per_query"][0]["qa"]["gold_label"], "NOT_SUPPORTED")
        self.assertEqual(details["per_query"][0]["gold_documents"]["docs"], self.docs)
        for name, value in before.items():
            self.assertEqual((self.case / name).read_bytes(), value)

    def test_wrong_qa_answer_alias_claim_or_native_id_is_rejected(self):
        self.result["results"][0]["gold_answers"] = ["Paris"]
        self.save_result()
        with self.assertRaisesRegex(ValueError, "answer labels differ"):
            REPORT.write_metric_report(self.case)
        self.result["results"][0]["gold_answers"] = ["NOT_SUPPORTED"]
        self.result["results"][0]["question"] = self.sample["hotpotqa_question"]
        self.save_result()
        with self.assertRaisesRegex(ValueError, "question text differs"):
            REPORT.write_metric_report(self.case)
        self.result["results"][0]["question"] = self.sample["claim"]
        self.result["results"][0]["sample_id"] = self.sample["hotpotqa_id"]
        self.save_result()
        with self.assertRaisesRegex(ValueError, "sample IDs/order differ"):
            REPORT.write_metric_report(self.case)

    def test_standalone_identity_and_retrieve_only_availability(self):
        (self.metadata / "sample_manifest.json").unlink()
        (self.case / "validated.ok").unlink()
        self.result["eval_mode"] = "retrieve"
        self.result["results"][0]["gold_answers"] = None
        self.write(self.case / "result.json", self.result)
        report = REPORT.write_metric_report(self.case, require_validated=False)
        self.assertFalse(report["sample_alignment"]["frozen_manifest_verified"])
        self.assertIsNone(report["metrics"]["ClassificationAccuracy"])
        self.assertIsNone(report["metrics"]["ClassificationMacroF1"])
        self.assertEqual(report["coverage"]["qa_unavailable_reasons"], {"answer_generation_not_run": 1})
        self.assertEqual(report["metrics"]["FCR@200"], 1)

    def test_hover_canonical_join_differs_from_2wiki_and_missing_gold_stays_a_miss(self):
        self.assertEqual(GOLD.gold_documents_from_sample(self.sample, "hover")["docs"], self.docs)
        self.assertNotEqual(GOLD.gold_documents_from_sample(self.sample, "2wikimultihopqa")["docs"], self.docs)
        row = self.result["results"][0]
        row["candidate_docs"] = self.docs[:-1]
        row["gold_document_ranks"][-1]["rank"] = None
        self.save_result()
        report = REPORT.write_metric_report(self.case)
        self.assertEqual(report["metrics"]["FCR@200"], 0)
        self.assertAlmostEqual(report["metrics"]["Recall@200"], 2 / 3)
        self.assertEqual(report["coverage"]["per_k"]["200"]["document_n"], 1)


if __name__ == "__main__":
    unittest.main()

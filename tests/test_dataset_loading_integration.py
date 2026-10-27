"""Exercise the real dataset entry and metric export without loading models."""
from contextlib import ExitStack
from dataclasses import dataclass, fields
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class StubConfig:
    dataset: str = ""
    stage_ablation: str = "full"
    retrieval_top_k: int = 200
    corpus_len: int = 0
    save_dir: str = ""

    @classmethod
    def from_file(cls, path, **overrides):
        names = {field.name for field in fields(cls)}
        return cls(**{name: value for name, value in overrides.items() if name in names})


class StubSolution:
    def __init__(self, question, docs, answers=None):
        self.question = question
        self.docs = docs
        self.doc_scores = [1.0] * len(docs)
        self.gold_answers = answers
        self.answer = answers[0] if answers else None

    def to_dict(self, top_k):
        return {"question": self.question, "docs": self.docs[:top_k],
                "doc_scores": self.doc_scores[:top_k], "answer": self.answer,
                "gold_answers": self.gold_answers, "retrieval_trace": {}}


class StubBindPack:
    instances = []

    def __init__(self, config):
        self.config = config
        self.llm_model = types.SimpleNamespace()
        self.__class__.instances.append(self)

    def index(self, docs):
        self.docs = docs

    def set_query_hop_overrides(self, hops, source):
        self.hops = list(hops)
        self.hop_source = source

    def retrieve(self, queries, num_to_retrieve, gold_docs):
        self.queries = queries
        return [StubSolution(query, docs) for query, docs in zip(queries, gold_docs)], {"Recall@5": 1.0}

    def rag_qa(self, queries, gold_docs, gold_answers):
        self.queries = queries
        self.answers = gold_answers
        solutions = [StubSolution(query, docs, answers)
                     for query, docs, answers in zip(queries, gold_docs, gold_answers)]
        return solutions, [], [], {"Recall@5": 1.0}, {"ExactMatch": 1.0, "F1": 1.0}


class DatasetLoadingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        temporary = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.base = Path(temporary)
        self.datasets = self.base / "datasets"
        self.datasets.mkdir()
        self.method = self.base / "method.json"
        self.method.write_text("{}", encoding="utf-8")
        self.stack.enter_context(patch.object(sys, "path", list(sys.path)))
        sys.path.insert(0, str(ROOT / "scripts"))
        helper = load_module("_integration_eval_utils", ROOT / "scripts/eval_utils.py")
        result_export = load_module("_integration_result_export", ROOT / "src/bindpack/evaluation/result_export.py")
        publication = load_module("_integration_publication", ROOT / "src/bindpack/index/publication_policy.py")
        report_cli = load_module("_integration_report_cli", ROOT / "scripts/report_metrics.py")
        self.report = report_cli._load_report_module()
        package = types.ModuleType("bindpack")
        package.BindPack = StubBindPack
        package.BindPackConfig = StubConfig
        stratified = types.ModuleType("stratified_eval")
        for name in ("evaluate_stratified", "print_stratified_results", "save_stratified_results"):
            setattr(stratified, name, lambda *args, **kwargs: self.fail("Unexpected stratified execution"))
        self.stack.enter_context(patch.dict(sys.modules, {
            "bindpack": package, "bindpack.evaluation": types.ModuleType("bindpack.evaluation"),
            "bindpack.evaluation.result_export": result_export,
            "bindpack.evaluation.metric_report": self.report,
            "bindpack.index": types.ModuleType("bindpack.index"),
            "bindpack.index.publication_policy": publication, "eval_utils": helper,
            "stratified_eval": stratified,
        }))
        self.entry = load_module("_integration_eval_dataset", ROOT / "scripts/eval_dataset.py")
        StubBindPack.instances = []

    def _dataset(self, dataset, samples):
        corpus = {}
        for sample in samples:
            for title, sentences in sample["context"]:
                separator = "" if dataset in ("hotpotqa", "hover") else " "
                corpus[title] = separator.join(sentences)
        (self.datasets / f"{dataset}.json").write_text(json.dumps(samples), encoding="utf-8")
        (self.datasets / f"{dataset}_corpus.json").write_text(json.dumps([
            {"title": title, "text": text} for title, text in corpus.items()]), encoding="utf-8")
        self.stack.enter_context(patch.object(self.entry, "DEFAULT_DATA_DIR", self.datasets))

    def _args(self, dataset, *extra):
        self.case = self.base / "standalone" / dataset
        self.case.mkdir(parents=True, exist_ok=True)
        return self.entry.build_parser().parse_args([
            "--dataset", dataset, "--save_dir", str(self.case / "index"),
            "--output", str(self.case / "result.json"), "--method_config", str(self.method), *extra])

    @staticmethod
    def _wiki_sample(identifier, titles):
        return {"_id": identifier, "question": f"Question {identifier}?", "answer": "Answer",
                "type": "bridge_comparison", "supporting_facts": [[title, 0] for title in titles],
                "context": [[title, [f"{title} evidence."]] for title in dict.fromkeys(titles)]}

    def test_default_entry_data_root_is_inside_repository(self):
        self.assertEqual(self.entry.DEFAULT_DATA_DIR, ROOT / "datasets")

    def test_2wiki_per_question_hops_are_forwarded_in_selected_order_and_exported(self):
        samples = [self._wiki_sample("two", ["A", "B"]),
                   self._wiki_sample("four", ["C", "D", "E", "F"]),
                   self._wiki_sample("five", ["G", "H", "I", "J", "J"])]
        self._dataset("2wikimultihopqa", samples)
        selected = self.base / "selected.json"
        selected.write_text("[2, 0]", encoding="utf-8")
        args = self._args("2wikimultihopqa", "--sample_indices_file", str(selected))
        output = self.entry.run_eval("2wikimultihopqa", args)
        rag = StubBindPack.instances[-1]
        self.assertEqual(rag.hops, [4, 2])
        self.assertEqual(rag.hop_source, "benchmark")
        self.assertEqual(rag.queries, [samples[2]["question"], samples[0]["question"]])
        result = json.loads(Path(output).read_text())
        self.assertEqual(result["data_path"], str(self.datasets / "2wikimultihopqa.json"))
        self.assertEqual(result["hop_distribution"], {"4": 1, "2": 1})
        self.assertEqual([row["benchmark_hops"] for row in result["results"]], [4, 2])
        self.assertEqual([row["sample_id"] for row in result["results"]], ["five", "two"])
        self.assertTrue(result["hop_provenance"]["oracle"])
        self.assertEqual(result["hop_provenance"]["validated_samples"], 3)
        self.assertEqual(result["hop_provenance"]["support_count_capped_to_four"], 1)
        report = json.loads((self.case / "report.json").read_text())
        self.assertEqual(report["metrics"]["FCR@5"], 1.0)
        self.assertFalse((self.case / "report_error.json").exists())

    def test_bad_2wiki_annotation_outside_selected_sample_fails_before_model_loading(self):
        self._dataset("2wikimultihopqa", [self._wiki_sample("valid", ["A", "B"]),
                                         self._wiki_sample("invalid", ["C", "D", "E"])])
        args = self._args("2wikimultihopqa", "--sample_index", "0")
        with self.assertRaisesRegex(ValueError, "3 supporting_facts"):
            self.entry.run_eval("2wikimultihopqa", args)
        self.assertEqual(StubBindPack.instances, [])
        self.assertFalse((self.case / "result.json").exists())

    def test_hotpot_hop_reads_distinct_support_titles(self):
        sample = self._wiki_sample("hotpot", ["A", "A", "B", "B", "B"])
        self._dataset("hotpotqa", [sample])
        output = self.entry.run_eval("hotpotqa", self._args("hotpotqa"))
        self.assertEqual(StubBindPack.instances[-1].hops, [2])
        result = json.loads(Path(output).read_text())
        self.assertEqual(result["hop_provenance"]["field"], "supporting_facts distinct titles")
        self.assertTrue(result["hop_provenance"]["oracle"])
        self.assertEqual(result["hop_provenance"]["raw_supporting_facts_distribution"], {"5": 1})

    def test_hover_forwards_claims_native_labels_hops_and_native_ids_through_report(self):
        samples = []
        for hop, label in ((3, "SUPPORTED"), (4, "NOT_SUPPORTED")):
            sample = self._wiki_sample(f"wrong-source-id-{hop}", [f"D{hop}-{index}" for index in range(hop)])
            sample.update(hover_id=f"hover-{hop}", claim=f"Native {hop}-hop claim.",
                          label=label, num_hops=hop, question="Wrong source question?",
                          hotpotqa_answer="Wrong source answer")
            samples.append(sample)
        self._dataset("hover", samples)
        output = self.entry.run_eval("hover", self._args("hover", "--eval_mode", "rag_qa"))
        rag = StubBindPack.instances[-1]
        self.assertEqual(rag.hops, [3, 4])
        self.assertEqual(rag.queries, [sample["claim"] for sample in samples])
        self.assertEqual(rag.answers, [["SUPPORTED"], ["NOT_SUPPORTED"]])
        result = json.loads(Path(output).read_text())
        self.assertEqual(result["task"], "fact_verification")
        self.assertEqual([row["sample_id"] for row in result["results"]], ["hover-3", "hover-4"])
        self.assertEqual([row["question"] for row in result["results"]], rag.queries)
        report = json.loads((self.case / "report.json").read_text())
        self.assertEqual(report["metrics"]["ClassificationAccuracy"], 1.0)
        self.assertEqual(report["metrics"]["ClassificationMacroF1"], 1.0)
        self.assertEqual(report["metrics"]["FCR@5"], 1.0)
        self.assertIsNone(report["metrics"]["DCC@5"])
        details = json.loads((self.case / "metrics_details.json").read_text())
        self.assertEqual([row["sample_id"] for row in details["per_query"]], ["hover-3", "hover-4"])
        self.assertFalse((self.case / "report_error.json").exists())

    def test_unselected_hover_bad_label_or_id_fails_before_model_loading(self):
        valid = self._wiki_sample("source", ["A", "B", "C"])
        valid.update(hover_id="hover-valid", claim="Valid claim.", label="SUPPORTED", num_hops=3)
        for field, bad in (("hover_id", ""), ("label", "REFUTED"), ("claim", "")):
            with self.subTest(field=field):
                invalid = {**valid, field: bad}
                self._dataset("hover", [valid, invalid])
                args = self._args("hover", "--sample_index", "0")
                with self.assertRaises(ValueError):
                    self.entry.run_eval("hover", args)
                self.assertEqual(StubBindPack.instances, [])
                self.assertFalse((self.case / "result.json").exists())

    def test_hover_retrieval_only_exports_report_without_qa_predictions(self):
        sample = self._wiki_sample("wrong-source-id", ["A", "B", "C"])
        sample.update(hover_id="hover-retrieval", claim="Native claim.",
                      label="NOT_SUPPORTED", num_hops=3, question="Wrong source question?")
        self._dataset("hover", [sample])
        output = self.entry.run_eval("hover", self._args("hover"))
        result = json.loads(Path(output).read_text())
        self.assertEqual(result["eval_mode"], "retrieve")
        self.assertIsNone(result["results"][0]["gold_answers"])
        report = json.loads((self.case / "report.json").read_text())
        self.assertEqual(report["metrics"]["FCR@5"], 1.0)
        self.assertIsNone(report["metrics"]["ClassificationAccuracy"])
        self.assertIsNone(report["metrics"]["ClassificationMacroF1"])
        self.assertFalse((self.case / "report_error.json").exists())

    def test_external_hipporag_hover_is_rejected_before_service_or_context_loading(self):
        runner = load_module("_integration_stage_runner", ROOT / "scripts/utils/stage_ablations.py")
        baseline = self.base / "external_baseline.py"
        baseline.write_text("pass\n", encoding="utf-8")
        with patch.object(runner, "BASELINE_MAIN", baseline), \
                patch.object(runner, "check_service", side_effect=AssertionError("Service must not be queried")), \
                patch.object(runner, "prepare_context", side_effect=AssertionError("Context must not be loaded")):
            with self.assertRaisesRegex(ValueError, "does not support HoVer fact verification"):
                runner.main(["--datasets", "hover", "--include-hipporag", "--dry-run"])

    def test_old_shared_index_rejects_changed_musique_benchmark_sha_before_asset_reads(self):
        runner = load_module("_integration_stage_runner", ROOT / "scripts/utils/stage_ablations.py")
        config = {"runtime": {"llm_name": "qwen3-8b", "embedding_model_name": "Qwen3-Embedding-8B"}}
        index = self.base / "old_shared_indexes/musique"
        metadata = index.parent.parent / "metadata/musique"
        model_name = runner.model_dir(config["runtime"])
        metadata.mkdir(parents=True)
        (metadata / "manifest.json").write_text(json.dumps({
            "dataset": "musique", "index_build_status": "ready", "model_dir": model_name,
            "runtime": config["runtime"], "data_sha256": "a" * 64,
            "corpus_sha256": "c" * 64, "indexed_docs": 1}), encoding="utf-8")
        model = index / model_name
        model.mkdir(parents=True)
        (model / "index_manifest.json").write_text(json.dumps({
            "embedding": {"model_name": config["runtime"]["embedding_model_name"]},
            "openie": {"identity": {"model_name": config["runtime"]["llm_name"]}}}), encoding="utf-8")
        identity = {"dataset": "musique", "data_path": str(self.datasets / "musique.json"),
                    "data_sha256": "b" * 64, "corpus_source_sha256": "c" * 64, "corpus_docs": 1}
        with patch.object(runner, "index_hashes", side_effect=AssertionError("Assets must not be read")):
            with self.assertRaisesRegex(ValueError, "corpus/benchmark identity mismatch"):
                runner.validate_shared_index({"index": index, "identity": identity}, config)


if __name__ == "__main__":
    unittest.main()

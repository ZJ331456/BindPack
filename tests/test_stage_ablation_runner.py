"""Exercise four-case orchestration and resume without model/GPU requests."""
import contextlib
import copy
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from utils import stage_ablations as runner


class StageAblationRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / "outputs", prefix="test_stage_runner_")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.output = self.base / "run_smoke"
        self.data = self.base / "data.json"
        self.corpus = self.base / "corpus.json"
        runner.write_json(self.data, [{"_id": "question-1", "question": "Where is Alice's employer based?",
            "answer": "Paris", "context": [["Alice", ["Alice works at Lab."]], ["Lab", ["Lab is in Paris."]]],
            "supporting_facts": [["Alice", 0], ["Lab", 0]]}])
        runner.write_json(self.corpus, [{"title": "Alice", "text": "Alice works at Lab."},
                                      {"title": "Lab", "text": "Lab is in Paris."}])
        self.config = runner.read_json(ROOT / "configs/ablations_qwen3.json")
        self.config["runtime"].update(python="/fake/python", runtime_deps="",
            embedding_model_name=str(self.base / "Qwen3-Embedding-8B"),
            llm_model_path=str(self.base / "Qwen3-8B"))
        for key in ("embedding_model_name", "llm_model_path"):
            runner.write_json(Path(self.config["runtime"][key]) / "config.json", {"model_type": "qwen3"})
        self.method_file = self.base / "method.json"
        runner.write_json(self.method_file, runner.read_json(ROOT / self.config["method_config"]))
        self.config["method_config"] = str(self.method_file)
        self.config_file = self.base / "config.json"
        runner.write_json(self.config_file, self.config)
        self.argv = ["--config", str(self.config_file), "--out-root", str(self.output),
            "--datasets", "hotpotqa", "--data-path", str(self.data), "--corpus-path", str(self.corpus),
            "--smoke", "--sample-size", "1"]
        self.calls = []

    def fake_execute(self, argv, logfile, env):
        self.calls.append(argv)
        tokens = argv[4:]
        values = {}
        index = 0
        while index < len(tokens):
            if tokens[index] in ("--save_dir_exact", "--reuse_index", "--index_only", "--no_ensure_gold_in_corpus"):
                values[tokens[index][2:]] = True
                index += 1
            else:
                values[tokens[index][2:]] = tokens[index + 1]
                index += 2
        directory = Path(values["save_dir"])
        output = Path(values["output"])
        model = directory / runner.model_dir(self.config["runtime"])
        indices = runner.read_json(values["sample_indices_file"])
        samples = runner.read_json(self.data)
        exported_rows = []
        for number in indices:
            sample = samples[number]
            gold = runner.get_gold_docs([sample], "hotpotqa")[0]
            exported_rows.append({"query_index": number, "sample_id": sample["_id"],
                "question": sample["question"], "answer": sample["answer"],
                "gold_answers": [sample["answer"]], "gold_docs": gold,
                "docs": gold, "candidate_docs": gold,
                "gold_document_ranks": [{"doc": doc, "rank": rank}
                                        for rank, doc in enumerate(gold, 1)]})
        hippo = str(argv[3]).endswith("HippoRAG/main.py")
        if "--index_only" in argv:
            method = {"sha256": runner.sha256(values["method_config"]),
                      "parameters": runner.read_json(values["method_config"])}
            runtime = {"llm_name": self.config["runtime"]["llm_name"],
                       "embedding_model_name": self.config["runtime"]["embedding_model_name"],
                       "stage_ablation": values["stage_ablation"]}
            for relative in ("graph.pickle", "chunk_embeddings/vdb_chunk.parquet"):
                path = model / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"frozen asset: " + relative.encode())
            runner.write_json(directory / "openie_results_ner_qwen3-8b.json", {"docs": []})
            (model / "openie_progress.sqlite").write_bytes(b"build journal")
            runner.write_json(output, {"index_build_complete": True, "indexed_docs": 2,
                                       "runtime_config": runtime, "method_config": method})
        elif hippo:
            self.assertFalse((directory / "llm_cache").exists())
            runner.write_json(output, {"dataset": "hotpotqa", "method": "hipporag2", "reuse_index": True,
                "llm_name": self.config["runtime"]["llm_name"],
                "embedding_name": self.config["runtime"]["embedding_model_name"],
                "selected_indices": indices, "sample_size_effective": len(indices),
                "eval_mode": values["eval_mode"], "retrieval_metrics": {"Recall@1": .4, "Recall@5": .9, "Recall@10": .95},
                "qa_metrics": {"ExactMatch": .5, "F1": .5} if values["eval_mode"] == "rag_qa" else {},
                "results": exported_rows})
        else:
            method = {"sha256": runner.sha256(values["method_config"]),
                      "parameters": runner.read_json(values["method_config"])}
            runtime = {"llm_name": self.config["runtime"]["llm_name"],
                       "embedding_model_name": self.config["runtime"]["embedding_model_name"],
                       "stage_ablation": values["stage_ablation"]}
            # A new stage must not inherit the index builder/previous stage cache.
            self.assertFalse((directory / "llm_cache").exists())
            self.assertFalse((model / "openie_progress.sqlite").exists())
            runner.write_json(output, {"dataset": "hotpotqa", "stage_ablation": values["stage_ablation"],
                "runtime_config": runtime, "method_config": method, "selected_indices": indices, "sample_size_effective": len(indices),
                "eval_mode": values["eval_mode"], "retrieval_metrics": {"Recall@1": .5, "Recall@5": 1., "Recall@10": 1.},
                "qa_metrics": {"ExactMatch": .5, "F1": .5} if values["eval_mode"] == "rag_qa" else {},
                "results": exported_rows})
        cache = directory / "llm_cache/cache.json"
        owner = "builder" if "--index_only" in argv else ("hippporag2_reproduce" if hippo else values["stage_ablation"])
        runner.write_json(cache, {"owner": owner})
        return 1.25

    def invoke(self, extra=()):
        with patch.object(runner, "execute", side_effect=self.fake_execute), \
                patch.object(runner, "check_service"), patch.object(runner.fcntl, "flock"), \
                contextlib.redirect_stdout(io.StringIO()):
            return runner.main(self.argv + list(extra))

    def historical_index(self):
        historical = self.base / "historical"
        source = historical / "shared_indexes/hotpotqa"
        model = source / runner.model_dir(self.config["runtime"])
        asset_hashes = {}
        for relative in ("graph.pickle", "chunk_embeddings/vdb_chunk.parquet",
                         "entity_embeddings/vdb_entity.parquet", "fact_embeddings/vdb_fact.parquet"):
            path = model / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"historical asset: " + relative.encode())
            asset_hashes[relative] = runner.sha256(path)
        runner.write_json(model / "index_manifest.json", {
            "embedding": {"model_name": self.config["runtime"]["embedding_model_name"]},
            "openie": {"identity": {"model_name": self.config["runtime"]["llm_name"]}}})
        runner.write_json(model / "openie_state.json", {"docs": []})
        runner.write_json(source / "openie_results_ner_qwen3-8b.json", {"docs": []})
        runner.write_json(source / "llm_cache/cache.json", {"owner": "historical builder"})
        runner.write_json(model / "route_cache.json", {"owner": "historical route"})
        runner.write_json(historical / "metadata/hotpotqa/manifest.json", {
            "dataset": "hotpotqa", "index_build_status": "ready", "model_dir": model.name,
            "runtime": self.config["runtime"], "data_sha256": runner.sha256(self.data),
            "corpus_sha256": runner.sha256(self.corpus), "indexed_docs": 2,
            "selected_indices": [0], "selected_sample_ids": ["question-1"],
            "source_asset_sha256": asset_hashes})
        runner.write_json(historical / "metadata/hotpotqa/selected_indices.json", [0])
        runner.write_json(historical / "protocol.json", {"cases": ["historical_case"]})
        runner.write_json(historical / "cases/hotpotqa/historical_case/result.json", {"historical": True})
        return historical, source

    def test_one_index_four_cases_and_resume_preserve_samples_and_isolate_caches(self):
        self.assertEqual(self.invoke(), 0)
        self.assertEqual(len(self.calls), 5)
        self.assertIn("--index_only", self.calls[0])
        cases = self.calls[1:]
        self.assertEqual([command[command.index("--stage_ablation") + 1] for command in cases], list(runner.CASES))
        self.assertEqual(len({command[command.index("--sample_indices_file") + 1] for command in cases}), 1)
        manifest = runner.read_json(self.output / "metadata/hotpotqa/sample_manifest.json")
        self.assertEqual(manifest["sample_ids"], ["question-1"])
        self.assertEqual(runner.read_json(self.output / "completed.ok")["validated_case_count"], 4)
        for name in runner.CASES:
            case = self.output / "cases/hotpotqa" / name
            report = runner.read_json(case / "report.json")
            self.assertEqual(report["metrics"]["FCR@5"], 1.0)
            self.assertIsNone(report["metrics"]["DCC@5"])
            self.assertEqual(report["inputs"]["result"]["sha256"],
                             runner.read_json(case / "validated.ok")["result_sha256"])
        self.assertEqual(self.invoke(["--resume"]), 0)
        self.assertEqual(len(self.calls), 5, "validated resume must not run models again")
        caches = [runner.read_json(self.output / "cases/hotpotqa" / case / "index/llm_cache/cache.json")
                  for case in runner.CASES]
        self.assertEqual([cache["owner"] for cache in caches], list(runner.CASES))

    def test_report_failure_is_retried_offline_without_rerunning_validated_cases(self):
        with patch.object(runner, "write_metric_report", side_effect=OSError("report storage unavailable")), \
                self.assertLogs(runner.LOG, level="ERROR"):
            self.assertEqual(self.invoke(), 0)
        for name in runner.CASES:
            case = self.output / "cases/hotpotqa" / name
            self.assertTrue((case / "validated.ok").is_file())
            self.assertTrue(runner.read_json(case / "report_error.json")["postprocessing_only"])
        self.assertEqual(self.invoke(["--resume"]), 0)
        self.assertEqual(len(self.calls), 5)
        for name in runner.CASES:
            case = self.output / "cases/hotpotqa" / name
            self.assertTrue((case / "report.json").is_file())
            self.assertFalse((case / "report_error.json").exists())

    def test_dry_run_is_read_only_and_never_contacts_service(self):
        with patch.object(runner, "execute") as execute, patch.object(runner, "check_service") as service, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(self.argv + ["--dry-run"]), 0)
        self.assertFalse(self.output.exists())
        execute.assert_not_called()
        service.assert_not_called()

    def test_resume_rejects_changed_result_even_with_valid_shape(self):
        self.invoke()
        result_path = self.output / "cases/hotpotqa/full/result.json"
        payload = runner.read_json(result_path)
        payload["retrieval_metrics"]["Recall@5"] = .5
        runner.write_json(result_path, payload)
        with self.assertRaisesRegex(ValueError, "Validated result changed"):
            self.invoke(["--resume"])

    def test_resume_rejects_frozen_and_private_graph_drift(self):
        self.invoke()
        name = runner.model_dir(self.config["runtime"])
        for location, message in (("shared_indexes/hotpotqa", "Frozen index or corpus changed"),
                                  ("cases/hotpotqa/full/index", "Private index changed")):
            graph = self.output / location / name / "graph.pickle"
            original = graph.read_bytes()
            graph.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, message):
                self.invoke(["--resume"])
            graph.write_bytes(original)

    def test_resume_rejects_model_configuration_or_sample_drift(self):
        self.invoke()
        changed = copy.deepcopy(self.config)
        changed["evaluation"]["qa_top_k"] = 4
        runner.write_json(self.config_file, changed)
        with self.assertRaisesRegex(ValueError, "inputs/configuration/code differ"):
            self.invoke(["--resume"])
        runner.write_json(self.config_file, self.config)
        data = runner.read_json(self.data)
        data[0]["_id"] = "another-question"
        runner.write_json(self.data, data)
        with self.assertRaisesRegex(ValueError, "inputs/configuration/code differ"):
            self.invoke(["--resume"])

    def test_invalid_case_matrix_and_unsafe_outputs_are_rejected(self):
        changed = copy.deepcopy(self.config)
        changed["cases"].append("legacy_component")
        runner.write_json(self.config_file, changed)
        with self.assertRaisesRegex(ValueError, "Exactly four"):
            self.invoke(["--dry-run"])
        with self.assertRaisesRegex(ValueError, "dedicated output directory"):
            runner.safe_output(ROOT / "outputs")

    def test_resume_rejects_eval_helper_code_and_method_parameter_drift(self):
        self.invoke()
        protocol = runner.read_json(self.output / "protocol.json")
        self.assertIn("scripts/eval_utils.py", protocol["code_sha256"])
        self.assertIn("scripts/stratified_eval.py", protocol["code_sha256"])
        original_sha256 = runner.sha256

        def changed_helper(path):
            return "changed-helper" if Path(path) == ROOT / "scripts/eval_utils.py" else original_sha256(path)

        with patch.object(runner, "sha256", side_effect=changed_helper):
            with self.assertRaisesRegex(ValueError, "inputs/configuration/code differ"):
                self.invoke(["--resume"])
        method = runner.read_json(self.method_file)
        method["pack"]["coverage_weight"] += .01
        runner.write_json(self.method_file, method)
        with self.assertRaisesRegex(ValueError, "inputs/configuration/code differ"):
            self.invoke(["--resume"])

    def test_partial_method_is_resolved_and_default_drift_rejects_resume(self):
        runner.write_json(self.method_file, {"pack": {"coverage_weight": .7}})
        self.invoke()
        frozen = runner.read_json(self.output / "method_config.json")
        defaults = ROOT / "src/bindpack/configuration/defaults.json"
        self.assertEqual(frozen["pack"]["coverage_weight"], .7)
        self.assertEqual(frozen["route"], runner.read_json(defaults)["route"])
        self.assertEqual(frozen["candidates"], runner.read_json(defaults)["candidates"])
        expected = runner.sha256(self.output / "method_config.json")
        protocol = runner.read_json(self.output / "protocol.json")
        self.assertEqual(protocol["config"]["method_source"]["sha256"], expected)
        original = runner.sha256
        with patch.object(runner, "sha256", side_effect=lambda p: "new-defaults" if Path(p) == defaults else original(p)):
            with self.assertRaisesRegex(ValueError, "inputs/configuration/code differ"):
                self.invoke(["--resume"])

    def test_shared_index_reuse_is_read_only_releases_private_inputs_and_resume_never_clones(self):
        historical, source = self.historical_index()
        self.argv += ["--reuse-shared-index-root", str(source.parent)]
        originals = {str(path.relative_to(historical)): runner.sha256(path)
                     for path in historical.rglob("*") if path.is_file()}
        self.assertEqual(self.invoke(), 0)
        self.assertEqual(len(self.calls), 4)
        self.assertTrue(all("--reuse_index" in command and "--index_only" not in command for command in self.calls))
        batch = self.output / "metadata/stage_ablations_qwen3_reuse"
        self.assertEqual(runner.read_json(batch / "completed.ok")["validated_case_count"], 4)
        self.assertFalse((self.output / "protocol.json").exists())
        for stage in runner.CASES:
            case = self.output / "cases/hotpotqa" / stage
            self.assertTrue(runner.read_json(case / "validated.ok")["private_assets_released"])
            self.assertEqual(runner.read_json(case / "index/llm_cache/cache.json")["owner"], stage)
            self.assertFalse((case / "index" / runner.model_dir(self.config["runtime"]) / "graph.pickle").exists())
            self.assertFalse((case / "index" / runner.model_dir(self.config["runtime"]) / "route_cache.json").exists())
        with patch.object(runner, "clone_shared_assets", side_effect=AssertionError("completed resume must not clone")):
            self.assertEqual(self.invoke(["--resume"]), 0)
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(originals, {str(path.relative_to(historical)): runner.sha256(path)
                                   for path in historical.rglob("*") if path.is_file()})

    def test_shared_index_dry_run_never_builds_and_preserves_history(self):
        historical, source = self.historical_index()
        self.argv += ["--reuse-shared-index-root", str(source.parent)]
        with patch.object(runner, "execute") as execute, patch.object(runner, "check_service") as service, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(runner.main(self.argv + ["--dry-run"]), 0)
        self.assertIn("REUSE VERIFIED SHARED INDEX", output.getvalue())
        self.assertNotIn("--index_only", output.getvalue())
        self.assertFalse(self.output.exists())
        execute.assert_not_called()
        service.assert_not_called()

    def test_shared_index_rejects_wrong_model_corpus_sample_or_asset_without_writes(self):
        historical, source = self.historical_index()
        self.argv += ["--reuse-shared-index-root", str(source.parent)]
        manifest_path = historical / "metadata/hotpotqa/manifest.json"
        original = runner.read_json(manifest_path)
        mutations = (("runtime", {**original["runtime"], "llm_name": "wrong-model"}, "model/dataset"),
                     ("corpus_sha256", "different-corpus", "corpus/benchmark"),
                     ("selected_sample_ids", ["different-question"], "Selected samples"),
                     ("source_asset_sha256", {}, "historical manifest"))
        for key, value, message in mutations:
            with self.subTest(key=key):
                runner.write_json(manifest_path, {**original, key: value})
                with self.assertRaisesRegex(ValueError, message):
                    self.invoke(["--dry-run"])
                self.assertFalse(self.output.exists())
        runner.write_json(manifest_path, original)

    def test_shared_index_failed_case_keeps_private_inputs_and_resume_cache(self):
        historical, source = self.historical_index()
        self.argv += ["--reuse-shared-index-root", str(source.parent)]
        with patch.object(runner, "execute", side_effect=RuntimeError("simulated model failure")), \
                patch.object(runner, "check_service"), patch.object(runner.fcntl, "flock"):
            with self.assertRaisesRegex(RuntimeError, "simulated model failure"):
                runner.main(self.argv)
        private = self.output / "cases/hotpotqa/full/index"
        self.assertTrue((private / runner.model_dir(self.config["runtime"]) / "graph.pickle").exists())
        self.assertFalse((private.parent / "validated.ok").exists())
        self.assertEqual(self.invoke(["--resume"]), 0)
        self.assertEqual(len(self.calls), 4)

    def test_shared_index_refuses_existing_case_before_model_requests(self):
        historical, source = self.historical_index()
        self.argv += ["--reuse-shared-index-root", str(source.parent)]
        runner.write_json(self.output / "cases/hotpotqa/full/result.json", {"untracked": True})
        with self.assertRaisesRegex(ValueError, "Refusing existing untracked case"):
            self.invoke(["--dry-run"])
        self.assertFalse((self.output / "metadata/stage_ablations_qwen3_reuse").exists())

    def test_shared_index_adds_hipporag_reproduce_with_retrieval_and_qa(self):
        historical, source = self.historical_index()
        self.argv += ["--reuse-shared-index-root", str(source.parent), "--include-hipporag", "--eval-mode", "rag_qa"]
        baseline = self.base / "baseline/HippoRAG/main.py"
        baseline.parent.mkdir(parents=True)
        baseline.write_text("# Subprocess execution is replaced by fake_execute.\n")
        with patch.object(runner, "BASELINE_MAIN", baseline):
            self.assertEqual(self.invoke(), 0)
        self.assertEqual(len(self.calls), 5)
        hippo = self.calls[-1]
        self.assertEqual(Path(hippo[3]), baseline)
        self.assertTrue(str(hippo[3]).endswith("HippoRAG/main.py"))
        self.assertIn("--reuse_index", hippo)
        self.assertNotIn("--index_only", hippo)
        self.assertEqual(hippo[hippo.index("--eval_mode") + 1], "rag_qa")
        case = self.output / "cases/hotpotqa/hippporag2_reproduce"
        result = runner.read_json(case / "result.json")
        self.assertEqual(result["qa_metrics"]["ExactMatch"], .5)
        self.assertTrue(runner.read_json(case / "validated.ok")["private_assets_released"])
        self.assertFalse((case / "index" / runner.model_dir(self.config["runtime"]) / "graph.pickle").exists())
        batch = self.output / "metadata/stage_ablations_qwen3_reuse"
        self.assertEqual(runner.read_json(batch / "completed.ok")["validated_case_count"], 5)
        self.assertIn("hippporag2_reproduce", runner.read_json(batch / "comparison.json")["cases"])

    def test_reuse_can_add_four_cases_to_historical_output_without_overwriting_old_protocol(self):
        historical, source = self.historical_index()
        self.argv.remove("--smoke")
        self.argv[self.argv.index("--out-root") + 1] = str(historical)
        self.argv += ["--reuse-shared-index-root", str(source.parent)]
        old_protocol = (historical / "protocol.json").read_bytes()
        old_result = (historical / "cases/hotpotqa/historical_case/result.json").read_bytes()
        self.assertEqual(self.invoke(), 0)
        self.assertEqual((historical / "protocol.json").read_bytes(), old_protocol)
        self.assertEqual((historical / "cases/hotpotqa/historical_case/result.json").read_bytes(), old_result)
        self.assertTrue((historical / "metadata/stage_ablations_qwen3_reuse/completed.ok").is_file())
        self.assertEqual([path.name for path in sorted((historical / "cases/hotpotqa").iterdir())],
                         ["full", "historical_case", "wo_pack", "wo_plan", "wo_route"])

    def test_shared_index_resume_rejects_source_graph_drift_before_any_model_request(self):
        historical, source = self.historical_index()
        self.argv += ["--reuse-shared-index-root", str(source.parent)]
        self.invoke()
        graph = source / runner.model_dir(self.config["runtime"]) / "graph.pickle"
        graph.write_bytes(b"source changed after original experiment")
        with self.assertRaisesRegex(ValueError, "historical manifest"):
            self.invoke(["--resume"])
        self.assertEqual(len(self.calls), 4)

    def test_copy_asset_falls_back_to_independent_copy_when_reflink_unavailable(self):
        source, target = self.base / "original.bin", self.base / "copy.bin"
        source.write_bytes(b"immutable")
        with patch.object(runner.fcntl, "ioctl", side_effect=OSError("no reflink")):
            runner.copy_asset(source, target)
        self.assertNotEqual(source.stat().st_ino, target.stat().st_ino)
        target.write_bytes(b"case changed")
        self.assertEqual(source.read_bytes(), b"immutable")

    def test_execute_streams_child_output_to_main_and_case_logs_and_reports_failure(self):
        log = self.base / "case/run.log"
        with contextlib.redirect_stdout(io.StringIO()) as output:
            runner.execute([sys.executable, "-u", "-c", "print('retrieval progress')"], log, {})
        self.assertIn("retrieval progress", output.getvalue())
        self.assertIn("retrieval progress", log.read_text())
        with self.assertRaisesRegex(RuntimeError, "Evaluation failed \\(7\\)"):
            runner.execute([sys.executable, "-c", "raise SystemExit(7)"], log, {})


if __name__ == "__main__":
    unittest.main()

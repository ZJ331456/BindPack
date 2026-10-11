"""CPU-only validation gates for online and frozen-input ablation exports."""
import gzip
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from utils import bindpack_ablations as driver

ISOLATED = ModuleType("_orchestration_cpu_bindpack")
ISOLATED.__path__ = [str(ROOT / "src/bindpack")]
sys.modules[ISOLATED.__name__] = ISOLATED
reporting = importlib.import_module(ISOLATED.__name__ + ".experiments.reporting")
selection = importlib.import_module(ISOLATED.__name__ + ".experiments.fixed_selection")


class AblationOrchestrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "cases"
        self.args = SimpleNamespace(python="/python", llm_base_url="http://local/v1",
            vllm_log=str(self.root / "vllm.log"), metadata=self.root / "metadata")
        Path(self.args.vllm_log).write_text("")
        (self.args.metadata / "hotpotqa").mkdir(parents=True)
        self.indices = [2, 0]  # Deliberately preserve a non-sorted selected order.
        self.documents = [f"Doc{i}\nActual source passage {i}." for i in range(205)]
        data = [{"_id": f"sample{i}", "question": f"Original question {i}?",
                 "supporting_facts": [[f"Doc{i}", 0]],
                 "context": [[f"Doc{i}", [f"Actual source passage {i}."]]]} for i in range(3)]
        self.asset_hashes = {name: "frozen-" + name for name in driver.assets.ASSETS}
        manifest = {"dataset": "hotpotqa", "model_dir": "nv2-model-dir",
            "runtime": {"embedding_batch_size": 4, "llm_prefetch_workers": 8,
                        "openie_max_workers": 8, "embedding_model_name": "/root/models/NV-Embed-v2"},
            "data_path": "/data/hotpotqa.json", "corpus_path": "/data/hotpotqa_corpus.json",
            "benchmark_hop_policy": driver.assets.benchmark_hop_policy("hotpotqa"),
            "source_asset_sha256": self.asset_hashes,
            "selected_indices": [0, 1, 2], "selected_sample_ids": [f"sample{i}" for i in range(3)]}
        self.context = {"dataset": "hotpotqa", "manifest": manifest,
            "data": data, "corpus": set(self.documents), "hops": [2, 2, 2],
            "source": self.root / "shared_indexes/hotpotqa", "cache_policy": "private frozen cache"}
        self.subset = driver.wf.subset_manifest(manifest, data, self.context["hops"], self.indices)
        self.source = self.output / "hotpotqa/abla_capture_full"
        self.receipt = {"completed": True, "seconds": 1, "log_start": 0, "log_end": 0}
        assets = patch.object(driver.assets, "asset_hashes", return_value=self.asset_hashes)
        assets.start()
        self.addCleanup(assets.stop)

        def initialize(context, case):
            if case.exists():
                raise ValueError("unvalidated case exists")
            case.mkdir(parents=True)
            driver.write_json(case / "before.json", {
                "asset_sha256": self.asset_hashes, "initial_cache_sha256": {"cache": "frozen"}})

        initialize_patch = patch.object(driver.wf, "initialize_case", side_effect=initialize)
        initialize_patch.start()
        self.addCleanup(initialize_patch.stop)
        # Alias only the two CPU experiment modules, avoiding package __init__
        # and any transformer, embedding or GPU setup.
        aliases = patch.dict(sys.modules, {
            "bindpack.experiments.reporting": reporting,
            "bindpack.experiments.fixed_selection": selection})
        aliases.start()
        self.addCleanup(aliases.stop)

    def result(self):
        runtime = dict(self.subset["runtime"], llm_name="qwen3-8b", max_new_tokens=2048,
                       improvement_stage=4, evidence_improvements=",".join(driver.wf.FLAGS))
        rows = []
        for index in self.indices:
            gold = self.documents[index]
            candidates = self.documents[:200]
            metrics = {f"Recall@{k}": float(gold in candidates[:k]) for k in (1, 2, 5, 10, 20, 200)}
            rows.append({"query_index": index, "sample_id": f"sample{index}",
                "question": self.context["data"][index]["question"], "benchmark_hops": 2,
                "docs": candidates[:10], "candidate_docs": candidates,
                "doc_scores": [float(i) for i in range(10, 0, -1)],
                "candidate_doc_scores": [float(i) for i in range(200, 0, -1)],
                "gold_docs": [gold], "retrieval_metrics": metrics,
                "gold_document_ranks": [{"doc": gold, "rank": index + 1}],
                "all_gold_in_top5": True, "all_gold_in_top10": True,
                "retrieval_trace": {"budget_control": {"query_index": index}}})
        return {"dataset": "hotpotqa", "eval_mode": "retrieve", "indexed_docs": 205,
            "selected_indices": self.indices, "sample_size_effective": 2,
            "result_top_k": 10, "candidate_output_top_k": 200, "runtime_config": runtime,
            "hop_source": "benchmark", "hop_distribution": self.subset["hop_distribution"],
            "llm_request_stats": {"failures": 0, "max_in_flight": 8, "http_attempts": 0},
            "retrieval_metrics": {key: sum(row["retrieval_metrics"][key] for row in rows) / 2
                                  for key in rows[0]["retrieval_metrics"]},
            "results": rows}

    def prepare(self, name="abla_budget_qd_100", mode="qd", result=None, complete=True):
        driver.online_case(self.args, self.context, self.indices, self.output,
            name, mode, self.source if mode in ("qd", "iterative") else None,
            prepare_only=True)
        case = self.output / "hotpotqa" / name
        driver.write_json(case / "result.json", result or self.result())
        driver.write_json(case / "request_ledger.json", {
            "summary": {"complete": complete, "total_tokens": 100}, "requests": []})
        return case

    def validate(self, name="abla_budget_qd_100", mode="qd"):
        return driver.online_case(self.args, self.context, self.indices, self.output,
            name, mode, self.source if mode in ("qd", "iterative") else None,
            execution_receipt=self.receipt)

    def test_budget_case_keeps_selected_order_and_uses_unified_validation(self):
        case = self.prepare()
        with patch.object(driver.wf, "validate_case", wraps=driver.wf.validate_case) as validate:
            report = self.validate()
        self.assertEqual(validate.call_count, 1)
        self.assertEqual([r["query_index"] for r in report["per_question"]], self.indices)
        self.assertEqual(report["n_samples"], 2)
        self.assertEqual(report["request_cost"]["total_tokens"], 100)
        self.assertEqual(driver.read_json(case / "validated.ok")["ledger_sha256"],
                         driver.assets.sha256(case / "request_ledger.json"))

    def test_budget_case_rejects_dropped_reordered_and_misattributed_samples(self):
        changes = {
            "dropped": lambda r: r["results"].pop(),
            "reordered": lambda r: r["results"].reverse(),
            "question": lambda r: r["results"][0].update(question="Wrong original question"),
            "duplicate_id": lambda r: r["results"][0].update(query_index=0),
            "wrong_hop": lambda r: r["results"][0].update(benchmark_hops=4),
            "wrong_batch": lambda r: r["runtime_config"].update(embedding_batch_size=8),
        }
        for label, change in changes.items():
            with self.subTest(label=label):
                result = self.result()
                change(result)
                name = "invalid_" + label
                case = self.prepare(name=name, result=result)
                with self.assertRaises(ValueError):
                    self.validate(name=name)
                self.assertFalse((case / "validated.ok").exists())

    def test_incomplete_ledger_never_publishes_a_validation_marker(self):
        for mode in ("capture", "qd", "iterative"):
            with self.subTest(mode=mode):
                name = "incomplete_" + mode
                case = self.prepare(name=name, mode=mode, complete=False)
                with self.assertRaisesRegex(ValueError, "ledger is incomplete"):
                    self.validate(name=name, mode=mode)
                self.assertFalse((case / "validated.ok").exists())

    def test_resume_rejects_ledger_modification_even_when_complete_remains_true(self):
        case = self.prepare()
        self.validate()
        ledger = driver.read_json(case / "request_ledger.json")
        ledger["summary"]["total_tokens"] += 1
        driver.write_json(case / "request_ledger.json", ledger)
        with self.assertRaisesRegex(ValueError, "ledger changed"):
            self.validate()

    def test_resume_rejects_selected_subset_and_budget_fraction_drift(self):
        self.prepare()
        self.validate()
        for indices, fraction in (([0, 2], 1.0), (self.indices, .5)):
            with self.subTest(indices=indices, fraction=fraction), self.assertRaisesRegex(
                    ValueError, "configuration differs"):
                driver.online_case(self.args, self.context, indices, self.output,
                    "abla_budget_qd_100", "qd", self.source, fraction)

    def test_unchanged_failed_retry_keeps_cache_and_archives_failure_accounting(self):
        case = self.prepare(name="retry_unchanged", mode="capture", complete=False)
        cache = case / "index/llm_cache/qwen3-8b_cache.sqlite"
        cache.parent.mkdir(parents=True)
        cache.write_bytes(b"private request cache from failed attempt\x00\x01")
        (case / "run.log").write_text("original failed-attempt log\n")
        preserved = {name: (case / name).read_bytes()
                     for name in ("request_ledger.json", "run.log", "before.json")}
        cache_bytes, cache_hash = cache.read_bytes(), driver.assets.sha256(cache)
        manifest = driver.read_json(case / "manifest.json")
        protocol = driver.read_json(case / "experiment_protocol.json")

        driver.prepare_failed_retry(self.context, case, manifest, protocol)

        attempts = list((case / "attempts").iterdir())
        self.assertEqual(len(attempts), 1)
        archive = attempts[0]
        for name, original in preserved.items():
            self.assertEqual((archive / name).read_bytes(), original)
            self.assertEqual((case / name).read_bytes(), original)
        self.assertEqual(cache.read_bytes(), cache_bytes)
        receipt = driver.read_json(archive / "retry_receipt.json")
        self.assertTrue(receipt["warm_cache"])
        self.assertEqual(receipt["reason"], "retry_incomplete_run_after_observer_fix")
        self.assertFalse(receipt["prior_logical_request_cost"]["complete"])
        self.assertEqual(receipt["cache_sha256_before_retry"], {cache.name: cache_hash})
        self.assertEqual(receipt["source_asset_sha256"], self.asset_hashes)
        self.assertFalse((case / "validated.ok").exists())

    def test_failed_retry_rejects_input_protocol_and_private_or_public_asset_drift(self):
        for drift in ("manifest", "protocol", "private_index", "public_index"):
            with self.subTest(drift=drift):
                case = self.prepare(name="retry_changed_" + drift, mode="capture", complete=False)
                manifest = driver.read_json(case / "manifest.json")
                protocol = driver.read_json(case / "experiment_protocol.json")
                if drift == "manifest":
                    manifest["selected_indices"] = list(reversed(self.indices))
                elif drift == "protocol":
                    protocol["llm_concurrency"] = 4
                changed_hashes = dict(self.asset_hashes)
                changed_hashes[next(iter(changed_hashes))] = "modified-index-asset"

                def hashes(index, model_dir=None):
                    changed = ((drift == "private_index" and Path(index) == case / "index") or
                               (drift == "public_index" and Path(index) == self.context["source"]))
                    return changed_hashes if changed else self.asset_hashes

                with patch.object(driver.assets, "asset_hashes", side_effect=hashes):
                    with self.assertRaisesRegex(ValueError, "differ|changed private/public index"):
                        driver.prepare_failed_retry(self.context, case, manifest, protocol)
                self.assertFalse((case / "attempts").exists())
                self.assertFalse((case / "validated.ok").exists())
                self.assertTrue((case / "request_ledger.json").is_file())

    def frozen(self, expected_ids=None):
        directory = self.source / "snapshots"
        directory.mkdir(parents=True)
        entries = []
        for index in self.indices:
            snapshot = {"schema_version": 1, "query": self.context["data"][index]["question"],
                "query_index": index, "candidate_ids": list(range(200)),
                "candidate_scores": [1.0 / (d + 1) for d in range(200)],
                "score_source": "pre_selection_merged", "state": {},
                "documents": dict(enumerate(self.documents[:200])),
                "expected_dag_ids": list(range(200)) if expected_ids is None else expected_ids}
            blob = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")).encode()
            path = directory / f"{index:05d}.json.gz"
            with gzip.open(path, "wb") as stream:
                stream.write(blob)
            entries.append({"query_index": index, "input_sha256": hashlib.sha256(blob).hexdigest()})
        driver.write_json(directory / "manifest.json", {"snapshots": entries})
        return directory

    def test_frozen_replay_validates_real_production_and_preserves_sample_order(self):
        self.frozen()
        reports = driver.fixed_replay(self.args, self.context, self.indices, self.source, self.output)
        self.assertEqual(len(reports), 9)
        self.assertEqual([r["query_index"] for r in reports["current_closed"]["per_question"]], self.indices)
        self.assertIn("shared_closed_package_space", reports["current_flat"]["experiment_scope"])
        self.assertEqual(reports["ordinary_node_coverage_greedy"]["experiment_scope"],
                         "ordinary_node_coverage_greedy_reference")

    def test_frozen_replay_rejects_production_mismatch_before_publishing(self):
        self.frozen(expected_ids=list(reversed(range(200))))
        with self.assertRaisesRegex(ValueError, "real DAG replay differs"):
            driver.fixed_replay(self.args, self.context, self.indices, self.source, self.output)
        self.assertFalse((self.output / "hotpotqa/abla_fixed_current_closed").exists())

    def test_frozen_replay_rejects_changed_snapshot_bytes(self):
        directory = self.frozen()
        path = directory / f"{self.indices[0]:05d}.json.gz"
        with gzip.open(path, "rt") as stream:
            snapshot = json.load(stream)
        snapshot["candidate_scores"][0] += .01
        with gzip.open(path, "wt") as stream:
            json.dump(snapshot, stream)
        with self.assertRaisesRegex(ValueError, "snapshot hash changed"):
            driver.fixed_replay(self.args, self.context, self.indices, self.source, self.output)


if __name__ == "__main__":
    unittest.main()

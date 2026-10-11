"""CPU-only duplicate-query tests with actual capture and logical-cost ledgers."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ModuleType("duplicate_capture_tests_package")
PACKAGE.__path__ = [str(ROOT / "src/bindpack")]
sys.modules[PACKAGE.__name__] = PACKAGE


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return [1] * 10


class FakeLLM:
    def __init__(self):
        self.llm_config = SimpleNamespace(generate_params={"max_completion_tokens": 2048})
        self.calls = []
        self.fail_after = None

    def infer(self, messages, **kwargs):
        self.calls.append(messages)
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            raise RuntimeError("fake transport failure")
        content = messages[0]["content"]
        completion = int(content.rsplit("hop=", 1)[1]) if "hop=" in content else 1
        return "{}", {"prompt_tokens": 10, "completion_tokens": completion, "finish_reason": "stop"}, False


class FakeEvidence:
    def finalize(self, query, ids, scores, ctx, state):
        return ids, scores, state["evidence_trace"]


class FakeRuntime(FakeEvidence):
    def __init__(self, rag):
        self.rag, self.cfg = rag, rag.global_config

    def _plan(self, query, hop):
        return self.rag.llm_model.infer([{"role": "user", "content": f"{query}; hop={hop}"}])

    def _add_route(self, *args, **kwargs):
        return [0, 1, 2]

    def _collect_jobs(self, executor, jobs):
        if executor is not None:
            futures = [executor.submit(func, *args) for func, args in jobs]
            return [future.result() for future in futures]
        return [func(*args) for func, args in jobs]

    def _verify(self, question, answer_type, documents):
        return self.rag.llm_model.infer([{"role": "user", "content": question}])

    def _document(self, doc_id):
        if doc_id == 0:
            return "Example Film\nExample Film was directed by Alice Example."
        if doc_id == 20:
            return "Alice Example\nAlice Example was born in Sample Town."
        return f"Title {doc_id}\nA genuine fake passage {doc_id}."

    def _goal_scores(self, candidate, bindings):
        return {proof["goal"]: proof["quality"] for proof in candidate["verified"]}


class FakeMetric:
    def __init__(self, **kwargs):
        pass

    def calculate_metric_scores(self, gold_docs, retrieved_docs, k_list):
        values = {f"Recall@{k}": sum(len(set(gold) & set(docs[:k])) / len(set(gold))
            for gold, docs in zip(gold_docs, retrieved_docs)) / len(gold_docs) for k in k_list}
        return values, []


class FakeBase:
    def __init__(self, *args, **kwargs):
        self.global_config = SimpleNamespace(evidence_base_weight=.4, max_new_tokens=2048)
        self.llm_model = FakeLLM()
        self.rerank_filter = SimpleNamespace(llm_infer_fn=self.llm_model.infer)
        self.evidence_runtime = FakeRuntime(self)
        self._query_hop_overrides = None
        self._query_hop_override_source = None
        self.parent_batches = []
        self.all_retrieval_time = 0.
        self.retrieval_diagnostics = {}
        self.passage_node_keys = list(range(40))
        self.chunk_embedding_store = SimpleNamespace(get_row=lambda d: {"content": self.evidence_runtime._document(d)})

    def set_query_hop_overrides(self, hops, source):
        self._query_hop_overrides = list(hops)
        self._query_hop_override_source = source

    def rerank_facts(self, query, *args):
        return self.llm_model.infer([{"role": "user", "content": query}])

    def _decompose_query(self, *args, **kwargs):
        return []

    def _decompose_query_with_hints(self, *args, **kwargs):
        return []

    def _path_graph_search(self, query, **kwargs):
        return self.run_ppr()

    def _build_qd_track_ranking(self, query, **kwargs):
        return []

    def _query_decomposition_retrieval(self, query, **kwargs):
        return []

    def dense_passage_retrieval(self, query):
        return self.run_ppr()

    def run_ppr(self, *args, **kwargs):
        return np.arange(40), np.arange(40, 0, -1, dtype=float)

    def retrieve(self, queries, num_to_retrieve=None, gold_docs=None):
        self.parent_batches.append((list(queries), list(self._query_hop_overrides)))
        solutions = []
        for local_index, (query, hop) in enumerate(zip(queries, self._query_hop_overrides)):
            self.rerank_facts(query)
            ids, scores = self._path_graph_search(query=query)
            self.evidence_runtime._plan(query, hop)
            plan = [{"id": "a", "question": "Who directed Example Film?", "depends_on": [], "answer_type": "person"},
                    {"id": "b", "question": "Where was ${a.answer} born?", "depends_on": ["a"], "answer_type": "place"}]
            bindings = {"a": "Alice Example", "b": "Sample Town"}
            proofs = {nid: {"doc_id": doc_id, "answer": bindings[nid], "confidence": .95,
                            "evidence": self.evidence_runtime._document(doc_id).split("\n", 1)[1]}
                      for nid, doc_id in (("a", 0), ("b", 20))}
            candidates = {proof["doc_id"]: {"base_score": .1, "sources": [],
                "verified": [dict(proof, goal="dag:" + nid, quality=.95)]} for nid, proof in proofs.items()}
            state = {"base": (ids, scores, {"hops": hop}), "evidence_candidates": candidates,
                     "_evidence_plan": plan, "_evidence_winning_bindings": bindings,
                     "_evidence_beams": [{"bindings": bindings, "proofs": proofs}],
                     "evidence_trace": {"selected_prefix": [], "plan": plan, "bindings": bindings}}
            self.evidence_runtime.finalize(query, ids, scores, {}, state)
            selected, diag = REAL_SELECTOR.dag.rerank_dag_packages(ids.tolist(), state, self.evidence_runtime._document)
            trace = {"evidence": dict(state["evidence_trace"], improvement_dag_package=diag,
                improvement_support_semantic_veto={"original_parent_doc_ids": selected})}
            solutions.append(SimpleNamespace(question=query,
                docs=[self.evidence_runtime._document(d) for d in selected], doc_scores=scores, retrieval_trace=trace))
        self.all_retrieval_time += len(queries)
        self.retrieval_diagnostics = {"total_queries": len(queries),
            "hop_counter": dict(Counter(str(h) for h in self._query_hop_overrides)),
            "avg_qd_sub_questions": sum(self._query_hop_overrides) / len(queries),
            "module_usage": {"planning": 1}, "per_query": [{"query_idx": i} for i in range(len(queries))]}
        if gold_docs is not None:
            return solutions, FakeMetric().calculate_metric_scores(gold_docs, [s.docs for s in solutions], [1, 5])[0]
        return solutions


def stub(name, **values):
    module = ModuleType(PACKAGE.__name__ + name)
    for key, value in values.items():
        setattr(module, key, value)
    sys.modules[module.__name__] = module


stub(".retrieval.engine", BindPack=FakeBase)
stub(".utils.misc_utils", QuerySolution=SimpleNamespace)
stub(".evaluation.retrieval_eval", RetrievalRecall=FakeMetric)
stub(".evidence.evidence_retrieval", EvidenceRetrieval=FakeEvidence,
     branch_matches=lambda requirements, bindings: all(
         bindings.get(key) == value for key, value in requirements.items()))
SPEC = importlib.util.spec_from_file_location(PACKAGE.__name__ + ".experiments.online",
                                            ROOT / "src/bindpack/experiments/online.py")
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)
RealLedger = module.LLMCostLedger

REAL_PACKAGE = ModuleType("duplicate_capture_real_selector")
REAL_PACKAGE.__path__ = [str(ROOT / "src/bindpack")]
sys.modules[REAL_PACKAGE.__name__] = REAL_PACKAGE
REAL_SPEC = importlib.util.spec_from_file_location(REAL_PACKAGE.__name__ + ".experiments.fixed_selection",
                                                 ROOT / "src/bindpack/experiments/fixed_selection.py")
REAL_SELECTOR = importlib.util.module_from_spec(REAL_SPEC)
sys.modules[REAL_SPEC.name] = REAL_SELECTOR
REAL_SPEC.loader.exec_module(REAL_SELECTOR)


class TestingLedger(RealLedger):
    def __init__(self, **kwargs):
        kwargs["tokenizer"] = kwargs.get("tokenizer") or FakeTokenizer()
        super().__init__(**kwargs)


class DuplicateCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.patch_ledger = patch.object(module, "LLMCostLedger", TestingLedger)
        self.patch_ledger.start()
        self.addCleanup(self.patch_ledger.stop)

    def worker(self, mode="capture", indices=(176, 300, 525), hops=(2, 3, 4)):
        options = SimpleNamespace(mode=mode, snapshot_dir=str(self.root / "snapshots"),
                                  ledger_path=str(self.root / "ledger.json"),
                                  selected_indices=list(indices), budget_fraction=1.)
        worker = module.worker_class(options)()
        worker.set_query_hop_overrides(list(hops), "benchmark")
        return worker, options

    def test_duplicate_capture_keeps_all_samples_and_separate_costs(self):
        worker, options = self.worker()
        gold = [[worker.evidence_runtime._document(0)], [worker.evidence_runtime._document(0)], ["absent gold"]]
        solutions, metrics = worker.retrieve(["same", "different", "same"], 40, gold)
        self.assertEqual([s.question for s in solutions], ["same", "different", "same"])
        self.assertEqual(worker.parent_batches, [(["same", "different"], [2, 3]), (["same"], [4])])
        self.assertAlmostEqual(metrics["Recall@5"], 2 / 3)
        self.assertEqual(worker._query_hop_overrides, [2, 3, 4])
        self.assertEqual(worker._query_hop_override_source, "benchmark")
        self.assertEqual(worker.retrieval_diagnostics["total_queries"], 3)
        self.assertEqual(worker.retrieval_diagnostics["hop_counter"], {"2": 1, "3": 1, "4": 1})
        self.assertEqual([r["query_idx"] for r in worker.retrieval_diagnostics["per_query"]], [0, 1, 2])
        self.assertEqual(worker.retrieval_diagnostics["avg_qd_sub_questions"], 3.)
        self.assertEqual(worker.all_retrieval_time, 3.)
        ledger = json.loads(Path(options.ledger_path).read_text())
        self.assertEqual(ledger["summary"]["logical_requests"], 6)
        self.assertEqual(ledger["queries"]["176"]["total_tokens"], 23)
        self.assertEqual(ledger["queries"]["525"]["total_tokens"], 25)
        self.assertEqual(ledger["summary"]["total_tokens"], 72)
        self.assertEqual([r["query_index"] for r in ledger["requests"]], [176, 176, 300, 300, 525, 525])
        self.assertEqual([r["request_id"] for r in ledger["requests"]], list(range(6)))
        first = module.read_snapshot(options.snapshot_dir, 176)
        last = module.read_snapshot(options.snapshot_dir, 525)
        self.assertEqual(first["hop"], 2)
        self.assertEqual(last["hop"], 4)
        self.assertEqual(first["request_cost"]["total_tokens"], 23)
        self.assertEqual(last["request_cost"]["total_tokens"], 25)
        self.assertEqual(len(first["retrieval_calls"]), 1)
        self.assertEqual(len(last["retrieval_calls"]), 1)
        for snapshot in (first, last):
            replay = REAL_SELECTOR.run_fixed_selection(snapshot)["variants"]["current_closed"]["ids"]
            self.assertEqual(replay, snapshot["expected_dag_ids"])
            self.assertIn(20, replay[:5])  # Actual production package promotion survives capture.
        manifest = json.loads((Path(options.snapshot_dir) / "manifest.json").read_text())
        self.assertEqual([r["query_index"] for r in manifest["snapshots"]], [176, 300, 525])
        for item in manifest["snapshots"]:
            with gzip.open(Path(options.snapshot_dir) / item["file"], "rb") as stream:
                self.assertEqual(hashlib.sha256(stream.read()).hexdigest(), item["input_sha256"])

    def test_no_plan_prune_also_retains_duplicate_and_without_gold_return_shape(self):
        worker, options = self.worker(mode="no_plan_prune")
        solutions = worker.retrieve(["same", "different", "same"], 40)
        self.assertIsInstance(solutions, list)
        self.assertEqual(len(solutions), 3)
        self.assertEqual(len(worker.experiment_capture.saved), 3)

    def test_unique_query_execution_remains_one_original_parent_call(self):
        worker, options = self.worker(indices=(5, 9), hops=(2, 3))
        solutions = worker.retrieve(["one", "two"], 40)
        self.assertEqual(worker.parent_batches, [(["one", "two"], [2, 3])])
        self.assertEqual(len(solutions), 2)
        ledger = json.loads(Path(options.ledger_path).read_text())
        self.assertEqual(ledger["summary"]["logical_requests"], 4)
        self.assertEqual(set(ledger["queries"]), {"one", "two"})
        self.assertNotIn("split_duplicate_question_batches", worker.retrieval_diagnostics)

    def test_failure_is_not_hidden_and_restores_hop_labels_and_manifest(self):
        worker, options = self.worker()
        worker.llm_model.fail_after = 4
        with self.assertRaisesRegex(RuntimeError, "fake transport failure"):
            worker.retrieve(["same", "different", "same"], 40)
        self.assertEqual(worker._query_hop_overrides, [2, 3, 4])
        self.assertEqual(worker.experiment_capture.selected_indices, [176, 300, 525])
        ledger = json.loads(Path(options.ledger_path).read_text())
        self.assertFalse(ledger["summary"]["complete"])
        self.assertEqual(ledger["summary"]["failed_requests"], 1)
        self.assertEqual(ledger["requests"][-1]["query_index"], 525)
        self.assertTrue((Path(options.snapshot_dir) / "manifest.json").is_file())

    def test_contiguous_split_does_not_drop_multiple_repetitions(self):
        self.assertEqual(module._unique_question_batches(["a", "b", "a", "a", "b"]),
                         [[0, 1], [2], [3, 4]])

    def test_budget_duplicate_questions_have_unique_index_owners(self):
        worker, options = self.worker(mode="qd", indices=(176, 525), hops=(3, 3))
        Path(options.snapshot_dir).mkdir()
        for index in options.selected_indices:
            snapshot = {"query": "same", "request_cost": {"total_tokens": 10000, "logical_requests": 8},
                        "common_base_cost": {"total_tokens": 11, "logical_requests": 1},
                        "common_base_ids": list(range(40)), "common_base_scores": [1.] * 40,
                        "hop": 3, "retrieval_calls": [{"stage": "shared_base_retrieval"},
                                                      {"stage": "dependency_search"}]}
            with gzip.open(Path(options.snapshot_dir) / f"{index:05d}.json.gz", "wt") as stream:
                json.dump(snapshot, stream)

        class FakeControl:
            def __init__(self, rag, *args, **kwargs):
                self.rag = rag

            def retrieve_one(self, question, ids, scores, hop):
                self.rag.llm_model.infer([{"role": "user", "content": question}])
                return np.asarray(ids), np.asarray(scores), {"search_count": 1}

        with patch.object(module, "BudgetRetrievalControl", FakeControl):
            solutions = worker.retrieve(["same", "same"], 40)
        self.assertEqual([s.question for s in solutions], ["same", "same"])
        ledger = json.loads(Path(options.ledger_path).read_text())
        self.assertEqual(set(ledger["queries"]), {"176::same", "525::same"})
        self.assertEqual(sorted(r["query_index"] for r in ledger["requests"]), [176, 525])
        self.assertTrue(all(r["original_question"] == "same" for r in ledger["requests"]))
        for solution in solutions:
            diagnostic = solution.retrieval_trace["budget_control"]
            self.assertEqual(diagnostic["end_to_end_logical_tokens"], 22)

    def test_real_capture_hooks_attribute_fact_and_subquestion_jobs_inside_threads(self):
        worker, options = self.worker(indices=(5, 9), hops=(2, 3))
        capture = worker.experiment_capture
        capture.register_queries(["original one", "original two"])
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(worker.rerank_facts, query)
                           for query in ("original one", "original two")]
                for future in futures:
                    future.result()
                jobs = []
                for original, subquestion in (("original one", "Who directed Example Film?"),
                                              ("original two", "Where was Alice Example born?")):
                    ids = worker.evidence_runtime._add_route({"query": original}, "dag:a", "dag:a", subquestion)
                    docs = {d: worker.evidence_runtime._document(d) for d in ids}
                    jobs.append((worker.evidence_runtime._verify, (subquestion, "person", docs)))
                worker.evidence_runtime._collect_jobs(executor, jobs)
            for query in ("original one", "original two"):
                summary = worker.experiment_ledger.get_summary(query)
                self.assertEqual(summary["logical_requests"], 2)
                self.assertEqual(summary["by_stage"]["shared_base"]["logical_requests"], 1)
                self.assertEqual(summary["by_stage"]["dependency_verification"]["logical_requests"], 1)
        finally:
            capture.close()


if __name__ == "__main__":
    unittest.main()

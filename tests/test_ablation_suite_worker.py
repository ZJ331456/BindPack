"""Suite cache identity and per-job isolation without imports of model code."""

from dataclasses import dataclass
import importlib.util
import io
import json
import logging
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("_ablation_suite_worker", ROOT / "scripts/ablation_suite_worker.py")
suite = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(suite)


@dataclass
class FakeConfig:
    embedding_model_name: str = "/root/models/NV-Embed-v2"
    embedding_batch_size: int = 4
    embedding_return_as_normalized: bool = True
    embedding_max_seq_len: int = 32768
    embedding_model_dtype: str = "bfloat16"
    embedding_future_option: str = "strict"
    save_dir: str = "/private/first"
    dataset: str = "hotpotqa"


class FakeNV:
    constructed = 0

    def __init__(self, global_config=None, embedding_model_name=None):
        type(self).constructed += 1
        self.global_config = global_config
        self.embedding_model_name = embedding_model_name or global_config.embedding_model_name
        self._init_embedding_config()
        self.model_weights = object()

    def _init_embedding_config(self):
        cfg = self.global_config
        value = {"name": self.embedding_model_name, "norm": cfg.embedding_return_as_normalized,
                 "dtype": cfg.embedding_model_dtype,
                 "encode_params": {"batch_size": cfg.embedding_batch_size,
                                   "max_length": cfg.embedding_max_seq_len}}
        self.embedding_config = SimpleNamespace(to_dict=lambda: value)


def original_factory(embedding_model_name):
    return FakeNV


class FrozenEmbeddingFactoryTests(unittest.TestCase):
    def setUp(self):
        FakeNV.constructed = 0

    def test_identical_embedding_settings_reuse_weights_with_private_case_config(self):
        factory = suite.FrozenNV2Factory(original_factory)
        first_config = FakeConfig()
        second_config = FakeConfig(save_dir="/private/second", dataset="musique")
        first = factory(first_config.embedding_model_name)(global_config=first_config,
                                                         embedding_model_name=first_config.embedding_model_name)
        first._nvembed_execution_stats = {"encoded_texts": 25}
        second = factory(second_config.embedding_model_name)(global_config=second_config,
                                                            embedding_model_name=second_config.embedding_model_name)
        self.assertIs(first, second)
        self.assertIs(second.global_config, second_config)
        self.assertFalse(hasattr(second, "_nvembed_execution_stats"))
        self.assertEqual(FakeNV.constructed, 1)
        self.assertEqual(factory.constructed, 1)
        self.assertEqual(factory.reused, 1)
        self.assertEqual(len(factory.config_sha256), 64)

    def test_embedding_field_conflicts_never_load_second_model(self):
        for field, value in (("embedding_batch_size", 8), ("embedding_model_dtype", "float16"),
                             ("embedding_max_seq_len", 8192), ("embedding_return_as_normalized", False),
                             ("embedding_future_option", "changed")):
            with self.subTest(field=field):
                FakeNV.constructed = 0
                factory = suite.FrozenNV2Factory(original_factory)
                constructor = factory()
                constructor(global_config=FakeConfig())
                changed = FakeConfig()
                setattr(changed, field, value)
                with self.assertRaisesRegex(ValueError, "configuration changed"):
                    constructor(global_config=changed)
                self.assertEqual(FakeNV.constructed, 1)

    def test_runtime_environment_changes_are_rejected(self):
        factory = suite.FrozenNV2Factory(original_factory)
        constructor = factory()
        with patch.dict(suite.os.environ, {"PATHCONDRAG_NVEMBED_OOM_SPLIT": "false"}):
            constructor(global_config=FakeConfig())
        with patch.dict(suite.os.environ, {"PATHCONDRAG_NVEMBED_OOM_SPLIT": "true"}):
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                constructor(global_config=FakeConfig())

    def test_full_derived_configuration_mutation_is_rejected(self):
        factory = suite.FrozenNV2Factory(original_factory)
        constructor = factory()
        model = constructor(global_config=FakeConfig())
        model.embedding_config.to_dict()["encode_params"]["batch_size"] = 8
        with self.assertRaisesRegex(ValueError, "Full derived"):
            constructor(global_config=FakeConfig())

    def test_non_nv2_models_are_explicitly_unsupported(self):
        factory = suite.FrozenNV2Factory(original_factory)
        with self.assertRaisesRegex(ValueError, "NV-Embed-v2 only"):
            factory("/root/models/Qwen3-Embedding-8B")

    def test_factory_patch_restores_even_after_job_error(self):
        core = SimpleNamespace(_get_embedding_model_class=original_factory)
        with self.assertRaisesRegex(RuntimeError, "failed"):
            with suite.shared_nv2_factory(core):
                self.assertIsInstance(core._get_embedding_model_class, suite.FrozenNV2Factory)
                raise RuntimeError("failed")
        self.assertIs(core._get_embedding_model_class, original_factory)


class SuiteJobIsolationTests(unittest.TestCase):
    def test_argv_accepts_driver_and_normal_python_formats(self):
        flags = ["--ablation-mode", "capture", "--dataset", "hotpotqa"]
        expected = [str(suite.WORKER_PATH), *flags]
        self.assertEqual(suite.normalize_job_argv(expected), expected)
        self.assertEqual(suite.normalize_job_argv([sys.executable, "-B", "-u", *expected]), expected)
        with self.assertRaisesRegex(ValueError, "this project's"):
            suite.normalize_job_argv(["/elsewhere/ablation_worker.py", *flags])

    def test_sequential_jobs_use_private_logs_receipts_and_restore_argv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            server = root / "vllm.log"
            server.write_text("existing requests\n")
            initial_argv = sys.argv
            original_class = object()
            original_finalize = lambda: None
            evaluation = SimpleNamespace(BindPack=original_class)
            evidence = SimpleNamespace(finalize=original_finalize)
            seen = []
            jobs = [{"argv": [str(suite.WORKER_PATH), "--dataset", name],
                     "log_path": str(root / (name + ".log")), "vllm_log": str(server),
                     "receipt_path": str(root / (name + ".receipt.json"))}
                    for name in ("hotpotqa", "musique")]

            def worker():
                self.assertIs(evaluation.BindPack, original_class)
                evaluation.BindPack = object()
                seen.append(sys.argv[-1])
                print("private output " + sys.argv[-1])
                with server.open("a") as stream:
                    stream.write("new HTTP 200\n")

            with patch("sys.stdout", new=io.StringIO()) as stream:
                suite.run_jobs(jobs, worker, eval_module=evaluation, evidence_class=evidence)
                self.assertIn("[suite-done] job=2/2", stream.getvalue())
            self.assertEqual(seen, ["hotpotqa", "musique"])
            self.assertIs(sys.argv, initial_argv)
            self.assertIs(evaluation.BindPack, original_class)
            first = json.loads((root / "hotpotqa.receipt.json").read_text())
            second = json.loads((root / "musique.receipt.json").read_text())
            self.assertTrue(first["completed"])
            self.assertTrue(second["completed"])
            self.assertEqual(first["log_end"], second["log_start"])
            self.assertGreater(first["log_end"], first["log_start"])
            self.assertIn("private output hotpotqa", (root / "hotpotqa.log").read_text())
            self.assertNotIn("musique", (root / "hotpotqa.log").read_text())

    def test_failure_stops_suite_and_preserves_error_and_global_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = lambda: None
            evidence = SimpleNamespace(finalize=original)
            jobs = [{"argv": [str(suite.WORKER_PATH), "first"],
                     "log_path": str(root / "first.log"), "receipt_path": str(root / "first.json")},
                    {"argv": [str(suite.WORKER_PATH), "second"]}]
            seen = []

            def fail():
                seen.append(sys.argv[-1])
                evidence.finalize = lambda: "leaked"
                raise ConnectionError("HTTP rejected")

            with patch("sys.stdout", new=io.StringIO()), self.assertRaisesRegex(ConnectionError, "HTTP rejected"):
                suite.run_jobs(jobs, fail, evidence_class=evidence)
            self.assertEqual(seen, ["first"])
            self.assertIs(evidence.finalize, original)
            receipt = json.loads((root / "first.json").read_text())
            self.assertFalse(receipt["completed"])
            self.assertEqual(receipt["error"]["type"], "ConnectionError")
            self.assertIn("HTTP rejected", (root / "first.log").read_text())

    def test_success_with_capture_hook_leak_is_rejected(self):
        evidence = SimpleNamespace(finalize=lambda: None)
        original = evidence.finalize
        job = {"argv": [str(suite.WORKER_PATH)]}

        def bad_worker():
            evidence.finalize = lambda: None

        with patch("sys.stdout", new=io.StringIO()), patch("sys.stderr", new=io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "capture finalize hook leaked"):
                suite.run_jobs([job], bad_worker, evidence_class=evidence)
        self.assertIs(evidence.finalize, original)

    def test_lazy_logger_does_not_retain_closed_case_file(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = logging.getLogger("ablation-suite-lazy-handler-test")
            logger.setLevel(logging.INFO)
            logger.propagate = False
            logger.handlers.clear()
            root = Path(directory)
            jobs = [{"argv": [str(suite.WORKER_PATH), name], "log_path": str(root / (name + ".log"))}
                    for name in ("first", "second")]

            def worker():
                if not logger.handlers:
                    logger.addHandler(logging.StreamHandler(sys.stderr))
                logger.info("message " + sys.argv[-1])

            try:
                with patch("sys.stdout", new=io.StringIO()), patch("sys.stderr", new=io.StringIO()):
                    suite.run_jobs(jobs, worker)
                    self.assertFalse(logger.handlers[0].stream.closed)
                self.assertIn("message first", (root / "first.log").read_text())
                self.assertIn("message second", (root / "second.log").read_text())
            finally:
                logger.handlers.clear()

    def test_existing_closed_logger_stream_can_be_replaced_without_flush(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = logging.getLogger("ablation-suite-closed-handler-test")
            logger.setLevel(logging.INFO)
            logger.propagate = False
            closed = io.StringIO()
            closed.close()
            handler = logging.StreamHandler(closed)
            logger.handlers = [handler]
            destination = Path(directory) / "case.log"
            try:
                with suite._job_output(destination):
                    logger.info("case logs must survive closed previous streams")
                self.assertIs(handler.stream, closed)
                self.assertIn("survive closed", destination.read_text())
            finally:
                logger.handlers.clear()

    def test_read_only_dynamic_stderr_handler_uses_context_redirection(self):
        class DynamicStderrHandler(logging.StreamHandler):
            def __init__(self):
                logging.Handler.__init__(self)

            @property
            def stream(self):
                return sys.stderr

        with tempfile.TemporaryDirectory() as directory:
            logger = logging.getLogger("ablation-suite-readonly-handler-test")
            logger.setLevel(logging.INFO)
            logger.propagate = False
            handler = DynamicStderrHandler()
            logger.handlers = [handler]
            destination = Path(directory) / "case.log"
            try:
                original = sys.stderr
                with suite._job_output(destination):
                    logger.info("read-only stream follows redirected stderr")
                self.assertIs(handler.stream, original)
                self.assertIn("follows redirected", destination.read_text())
            finally:
                logger.handlers.clear()


if __name__ == "__main__":
    unittest.main()

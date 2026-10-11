#!/usr/bin/env python3
"""Run smoke ablation jobs sequentially while retaining one NV2 model.

``--job-file`` is a JSON array of objects with ``argv``, ``log_path``,
``vllm_log`` and ``receipt_path``. ``argv`` accepts either the normal complete
Python command or ``[".../ablation_worker.py", ...normal flags...]``. Indexes,
LLM clients, caches, capture sessions and result files are still constructed by
the normal worker for each case. Only an identically configured NV2 embedding
wrapper is shared. This helper is intended for smoke checks, not formal timing.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import asdict, is_dataclass
import gc
import hashlib
import importlib
import inspect
import json
import logging
import os
from pathlib import Path
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = ROOT / "scripts/ablation_worker.py"


def _serialized(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


class FrozenNV2Factory:
    """Proxy constructor that refuses any embedding configuration drift.

    No RAG, index, graph or LLM instance is cached. Reused wrappers are rebound
    to the current case configuration and their execution counters are reset.
    The full derived EmbeddingConfig is recomputed without loading weights.
    """

    def __init__(self, original_factory):
        self.original_factory = original_factory
        self.wrapper = None
        self.signature = None
        self.embedding_config = None
        self.constructed = 0
        self.reused = 0

    def __call__(self, embedding_model_name="nvidia/NV-Embed-v2"):
        if not isinstance(embedding_model_name, str) or "NV-Embed-v2" not in embedding_model_name:
            raise ValueError("Smoke suite model sharing supports NV-Embed-v2 only")
        real_class = self.original_factory(embedding_model_name=embedding_model_name)
        cache = self

        class ReusedNV2:
            def __new__(cls, *args, **kwargs):
                return cache._construct(real_class, embedding_model_name, args, kwargs)

        return ReusedNV2

    @staticmethod
    def _config_dict(config):
        if is_dataclass(config):
            return asdict(config)
        if hasattr(config, "__dict__"):
            return vars(config)
        raise TypeError("A complete dataclass or namespace embedding configuration is required")

    def _construct(self, real_class, name, args, kwargs):
        bound = inspect.signature(real_class).bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        config = arguments.get("global_config")
        if config is None:
            raise ValueError("Suite embedding construction requires an explicit case global_config")
        complete = self._config_dict(config)
        embedding_fields = {key: value for key, value in complete.items()
                            if key.startswith("embedding") or key.startswith("azure_embedding")}
        constructor_fields = {key: value for key, value in arguments.items() if key != "global_config"}
        env_names = {
            "CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF",
            "PATHCONDRAG_NVEMBED_OOM_SPLIT", "HF_HOME", "TRANSFORMERS_CACHE",
        }
        env_names.update(key for key in os.environ if key.startswith("HIPPO_EMBEDDING_"))
        environment = {key: os.environ.get(key) for key in sorted(env_names)}
        signature = _serialized({"model_name": name,
            "model_class": real_class.__module__ + "." + real_class.__qualname__,
            "embedding_fields": embedding_fields, "constructor_fields": constructor_fields,
            "environment": environment})
        if self.wrapper is None:
            wrapper = real_class(*args, **kwargs)
            if not hasattr(wrapper, "embedding_config") or not hasattr(wrapper.embedding_config, "to_dict"):
                raise TypeError("NV2 wrapper must expose its complete EmbeddingConfig")
            self.wrapper, self.signature = wrapper, signature
            self.embedding_config = deepcopy(wrapper.embedding_config.to_dict())
            self.constructed += 1
            return wrapper
        if signature != self.signature:
            raise ValueError("NV2 embedding configuration changed across smoke suite jobs")
        if not hasattr(real_class, "_init_embedding_config"):
            raise TypeError("NV2 wrapper needs _init_embedding_config for full configuration validation")
        probe = object.__new__(real_class)
        probe.global_config = config
        probe.embedding_model_name = arguments.get("embedding_model_name") or complete.get("embedding_model_name", name)
        probe._init_embedding_config()
        expected = probe.embedding_config.to_dict()
        if (_serialized(expected) != _serialized(self.embedding_config)
                or _serialized(self.wrapper.embedding_config.to_dict()) != _serialized(self.embedding_config)):
            raise ValueError("Full derived NV2 EmbeddingConfig differs from the frozen model configuration")
        self.wrapper.global_config = config
        if hasattr(self.wrapper, "_nvembed_execution_stats"):
            del self.wrapper._nvembed_execution_stats
        self.reused += 1
        return self.wrapper

    @property
    def config_sha256(self):
        return hashlib.sha256(self.signature.encode()).hexdigest() if self.signature is not None else None


@contextmanager
def shared_nv2_factory(core_module):
    original = core_module._get_embedding_model_class
    factory = FrozenNV2Factory(original)
    core_module._get_embedding_model_class = factory
    try:
        yield factory
    finally:
        core_module._get_embedding_model_class = original


def normalize_job_argv(argv):
    if not isinstance(argv, list) or not argv or any(not isinstance(part, str) for part in argv):
        raise ValueError("Each suite job argv must be a nonempty string array")
    worker_positions = [i for i, part in enumerate(argv) if Path(part).name == "ablation_worker.py"]
    if len(worker_positions) != 1:
        raise ValueError("Suite jobs must call ablation_worker.py exactly once")
    position = worker_positions[0]
    if Path(argv[position]).resolve() != WORKER_PATH.resolve():
        raise ValueError("Suite jobs must use this project's ablation_worker.py")
    prefix = argv[:position]
    if prefix and ("python" not in Path(prefix[0]).name or any(part not in {"-B", "-u"} for part in prefix[1:])):
        raise ValueError("Only normal Python -B/-u prefixes are supported by suite jobs")
    return [str(WORKER_PATH), *argv[position + 1:]]


def _mutable_stream_handler(handler):
    if not isinstance(handler, logging.StreamHandler) or isinstance(handler, logging.FileHandler):
        return False
    descriptor = inspect.getattr_static(type(handler), "stream", None)
    # logging._StderrHandler and some third-party handlers resolve sys.stderr
    # dynamically through a read-only property. redirect_stderr handles these;
    # assigning stream would itself fail before the worker can start.
    return not (isinstance(descriptor, property) and descriptor.fset is None)


def _switch_stream(handler, destination):
    if getattr(handler.stream, "closed", False):
        # setStream flushes the old stream. Another test/library may have kept
        # its handler after closing that stream, so flushing it is invalid.
        handler.acquire()
        try:
            handler.stream = destination
        finally:
            handler.release()
    else:
        handler.setStream(destination)


@contextmanager
def _job_output(log_path):
    if not log_path:
        yield
        return
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    outer_stderr = sys.stderr
    with path.open("a", encoding="utf-8", buffering=1) as stream:
        previous = []
        loggers = [logging.getLogger()] + [logger for logger in logging.Logger.manager.loggerDict.values()
                                           if isinstance(logger, logging.Logger)]
        seen = set()
        for logger in loggers:
            for handler in logger.handlers:
                if id(handler) not in seen and _mutable_stream_handler(handler):
                    previous.append((handler, handler.stream))
                    _switch_stream(handler, stream)
                    seen.add(id(handler))
        try:
            with redirect_stdout(stream), redirect_stderr(stream):
                yield
        finally:
            for handler, original_stream in previous:
                _switch_stream(handler, original_stream)
            # Some libraries attach a logger handler lazily during their first
            # request. Never leave such a handler pointing to this closed file.
            current_loggers = [logging.getLogger()] + [
                logger for logger in logging.Logger.manager.loggerDict.values()
                if isinstance(logger, logging.Logger)]
            for logger in current_loggers:
                for handler in logger.handlers:
                    if (id(handler) not in seen and _mutable_stream_handler(handler)
                            and handler.stream is stream):
                        _switch_stream(handler, outer_stderr)


def _log_size(path):
    if not path:
        return 0
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"vLLM request log is missing: {source}")
    return source.stat().st_size


def _receipt(path, data):
    if not path:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(destination)


def run_jobs(jobs, worker_main, *, eval_module=None, evidence_class=None, factory=None):
    """Execute jobs in order; an exception publishes a failed receipt and stops."""
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Suite job file must contain a nonempty JSON array")
    normalized = []
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("Each suite job must be an object")
        normalized.append((job, normalize_job_argv(job.get("argv"))))
    for index, (job, argv) in enumerate(normalized, 1):
        print(f"[suite-start] job={index}/{len(normalized)} log={job.get('log_path', '-')}", flush=True)
        started = time.monotonic()
        log_start = _log_size(job.get("vllm_log"))
        old_argv = sys.argv
        old_bindpack = eval_module.BindPack if eval_module is not None else None
        old_finalize = evidence_class.finalize if evidence_class is not None else None
        completed = False
        error = None
        try:
            with _job_output(job.get("log_path")):
                try:
                    sys.argv = argv
                    worker_main()
                    if evidence_class is not None and evidence_class.finalize is not old_finalize:
                        raise RuntimeError("Experimental capture finalize hook leaked between suite jobs")
                    completed = True
                except BaseException:
                    traceback.print_exc()
                    raise
        except BaseException as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            sys.argv = old_argv
            if eval_module is not None:
                eval_module.BindPack = old_bindpack
            if evidence_class is not None:
                evidence_class.finalize = old_finalize
            elapsed = time.monotonic() - started
            log_end = _log_size(job.get("vllm_log"))
            if log_end < log_start:
                completed = False
                error = {"type": "LogRotationError", "message": "vLLM log shrank during suite job"}
            receipt = {"seconds": elapsed, "log_start": log_start, "log_end": log_end,
                       "completed": completed, "smoke_suite_shared_model": True}
            if factory is not None:
                receipt["embedding_reuse"] = {"constructed": factory.constructed, "reused": factory.reused,
                                             "config_sha256": factory.config_sha256}
            if error:
                receipt["error"] = error
            _receipt(job.get("receipt_path"), receipt)
            gc.collect()
            if log_end < log_start:
                raise RuntimeError("vLLM log shrank during suite job")
        print(f"[suite-done] job={index}/{len(normalized)} seconds={elapsed:.2f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-file", required=True)
    args = parser.parse_args()
    jobs = json.loads(Path(args.job_file).read_text(encoding="utf-8"))
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    # Import before per-case redirection so shared logging handlers retain a
    # valid outer stream once each case's private log file is closed.
    worker = importlib.import_module("ablation_worker")
    core = importlib.import_module("bindpack.core.base")
    evidence = importlib.import_module("bindpack.evidence.evidence_retrieval")
    with shared_nv2_factory(core) as factory:
        run_jobs(jobs, worker.main, eval_module=worker.eval_dataset,
                 evidence_class=evidence.EvidenceRetrieval, factory=factory)


if __name__ == "__main__":
    main()

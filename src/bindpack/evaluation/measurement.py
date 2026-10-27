"""Opt-in observation of native execution; never changes prompts or model outputs."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid


def digest(value):
    # Native traces may contain tuples/integer dictionary keys. Hash the actual
    # JSON representation, so a persisted checkpoint has the identical digest.
    value = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def jsonl(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []


def distribution(values):
    values = sorted(float(v) for v in values if v is not None and math.isfinite(v))
    if not values:
        return dict(n=0, total=0, mean=None, p50=None, p95=None, min=None, max=None)
    def percentile(q):
        position = (len(values) - 1) * q
        low = math.floor(position)
        high = math.ceil(position)
        return values[low] + (values[high] - values[low]) * (position - low)
    return dict(n=len(values), total=sum(values), mean=sum(values)/len(values),
                p50=percentile(.5), p95=percentile(.95), min=values[0], max=values[-1])


def usage(events):
    return dict(http_attempts=len(events),
        successful_calls=sum(e["status"] == "success" for e in events),
        failed_attempts=sum(e["status"] != "success" for e in events),
        input_tokens=sum(e.get("prompt_tokens") or 0 for e in events),
        output_tokens=sum(e.get("completion_tokens") or 0 for e in events),
        attempts_with_unknown_usage=sum(e.get("prompt_tokens") is None or
                                        e.get("completion_tokens") is None for e in events))


class Recorder:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.phase, self.query_id = "model_loading", None
        self.lock = threading.RLock()
        self.local = threading.local()
        self.phases = []
        self.restorations = []

    def append(self, name, event):
        with self.lock, (self.directory / name).open("a") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def tags(self):
        return dict(phase=self.phase, query_id=self.query_id)

    @contextmanager
    def section(self, phase, query_id=None):
        previous = self.phase, self.query_id
        self.phase, self.query_id = phase, query_id
        start = time.perf_counter()
        try:
            yield
        finally:
            self.phases.append(dict(phase=phase, query_id=query_id,
                                    seconds=time.perf_counter()-start))
            self.phase, self.query_id = previous

    def replace(self, obj, name, replacement):
        original = getattr(obj, name)
        self.restorations.append((obj, name, original))
        setattr(obj, name, replacement)
        return original

    def install_llm(self, llm, tokenizer):
        """Observe physical SDK attempts below native retries and cache lookups."""
        transport = llm.openai_client.chat.completions
        if getattr(llm.openai_client, "max_retries", 0) != 0:
            raise ValueError("SDK retries must be disabled to count every physical attempt")
        original_create, original_infer = transport.create, llm.infer

        @functools.wraps(original_create)
        def create(*args, **kwargs):
            event = dict(self.tags(), call_id=getattr(self.local, "call_id", None),
                model=kwargs.get("model"), prompt_sha256=digest(kwargs.get("messages")),
                max_tokens=kwargs.get("max_tokens", kwargs.get("max_completion_tokens")),
                status="error", prompt_tokens=None, completion_tokens=None,
                started_unix=time.time())
            start = time.perf_counter()
            try:
                response = original_create(*args, **kwargs)
                data = getattr(response, "usage", None)
                event.update(status="success", prompt_tokens=getattr(data, "prompt_tokens", None),
                    completion_tokens=getattr(data, "completion_tokens", None),
                    response_id=getattr(response, "id", None),
                    finish_reason=response.choices[0].finish_reason)
                return response
            except Exception as exc:
                event.update(error_type=type(exc).__name__, error=str(exc)[:500])
                raise
            finally:
                event["seconds"] = time.perf_counter() - start
                self.append("llm_calls.jsonl", event)

        @functools.wraps(original_infer)
        def infer(*args, **kwargs):
            call_id = uuid.uuid4().hex
            previous = getattr(self.local, "call_id", None)
            self.local.call_id = call_id
            event = dict(self.tags(), call_id=call_id, status="error", cache_hit=None)
            messages = args[0] if args else kwargs.get("messages")
            event["prompt_sha256"] = digest(messages)
            if self.phase == "online_reader":
                event["prompt_tokens_local"] = len(tokenizer.apply_chat_template(messages,
                    tokenize=True, add_generation_prompt=True, enable_thinking=False))
            start = time.perf_counter()
            try:
                response = original_infer(*args, **kwargs)
                event.update(status="success", cache_hit=bool(response[2]), metadata=response[1])
                return response
            except Exception as exc:
                event.update(error_type=type(exc).__name__, error=str(exc)[:500])
                raise
            finally:
                event["seconds"] = time.perf_counter() - start
                self.append("llm_logical_calls.jsonl", event)
                self.local.call_id = previous
        self.replace(transport, "create", create)
        self.replace(llm, "infer", infer)

    def install_llm_factory(self, llm_class, tokenizer):
        """Install before RAG initialization so captured native callbacks are observed."""
        original = llm_class.__init__
        @functools.wraps(original)
        def initialize(instance, *args, **kwargs):
            original(instance, *args, **kwargs)
            self.install_llm(instance, tokenizer)
        self.replace(llm_class, "__init__", initialize)

    def install_embedding(self, embedding):
        original = embedding.batch_encode
        tokenizer = embedding.model.tokenizer
        @functools.wraps(original)
        def encode(texts, **kwargs):
            values = [texts] if isinstance(texts, str) else list(texts)
            instruction = kwargs.get("instruction", "") or ""
            if instruction:
                prefix = f"Instruct: {instruction}\nQuery:" if embedding._is_qwen3_embedding else f"Instruct: {instruction}\nQuery: "
                values = [prefix + text for text in values]
            lengths = [len(ids) for ids in tokenizer(values, truncation=False,
                add_special_tokens=True)["input_ids"]]
            limit = embedding.model.max_seq_length
            event = dict(self.tags(), texts=len(values), effective_tokens=sum(min(n, limit) for n in lengths),
                truncated_texts=sum(n > limit for n in lengths), status="error",
                instruction=instruction)
            start = time.perf_counter()
            try:
                result = original(texts, **kwargs)
                event["status"] = "success"
                return result
            finally:
                event["seconds"] = time.perf_counter() - start
                self.append("embedding_calls.jsonl", event)
        self.replace(embedding, "batch_encode", encode)

    def time_method(self, obj, name, label):
        original = getattr(obj, name)
        @functools.wraps(original)
        def observed(*args, **kwargs):
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.append("stage_timings.jsonl", dict(self.tags(), stage=label,
                    seconds=time.perf_counter()-start))
        self.replace(obj, name, observed)

    def install_index(self, rag):
        self.time_method(rag.openie, "batch_openie", "openie")
        for name in ("chunk", "entity", "fact"):
            self.time_method(getattr(rag, name + "_embedding_store"), "insert_strings", name + "_embedding_and_store")
        for name in ("add_fact_edges", "add_passage_edges", "add_synonymy_edges",
                     "augment_graph", "save_igraph", "_build_index_side_artifacts"):
            self.time_method(rag, name, name)


class ResourceMonitor:
    """Sample the process tree and shared GPU; GPU numbers are never exclusive cost."""
    def __init__(self, recorder, interval=2.0):
        import psutil
        self.recorder, self.interval = recorder, interval
        self.process = psutil.Process()
        self.process.cpu_percent(None)
        self.cpu_start = sum(self.process.cpu_times()[:2])
        self.start = time.monotonic()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def sample(self):
        import psutil
        event = dict(self.recorder.tags(), unix_seconds=time.time(), pid=os.getpid(),
            elapsed_seconds=time.monotonic()-self.start,
            process_cpu_percent=self.process.cpu_percent(None),
            process_rss_bytes=self.process.memory_info().rss,
            process_cpu_seconds=sum(self.process.cpu_times()[:2])-self.cpu_start,
            system_cpu_percent=psutil.cpu_percent(None),
            system_memory_used_bytes=psutil.virtual_memory().used)
        children = self.process.children(recursive=True)
        event["process_tree_rss_bytes"] = event["process_rss_bytes"]
        for child in children:
            try:
                event["process_tree_rss_bytes"] += child.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_initialized():
            event.update(cuda_allocated_bytes=torch.cuda.memory_allocated(),
                cuda_reserved_bytes=torch.cuda.memory_reserved(),
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3, check=True).stdout
            event["gpus"] = []
            for line in out.strip().splitlines():
                index, name, util, used, total, power = [s.strip() for s in line.split(",")]
                def number(s):
                    try: return float(s)
                    except ValueError: return None
                event["gpus"].append(dict(index=int(index), name=name, utilization_percent=number(util),
                    memory_used_mib=number(used), memory_total_mib=number(total), power_watts=number(power)))
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            event.update(gpus=[], gpu_unavailable_reason=str(exc)[:200])
        self.recorder.append("resource_samples.jsonl", event)

    def _run(self):
        while not self.stop.is_set():
            try:
                self.sample()
            except Exception as exc:
                self.recorder.append("monitor_errors.jsonl", dict(error=str(exc), unix_seconds=time.time()))
            self.stop.wait(self.interval)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=5)
        self.sample()

    def summary(self):
        rows = jsonl(self.recorder.directory / "resource_samples.jsonl")
        values = {key: distribution(row.get(key) for row in rows) for key in (
            "process_cpu_percent", "process_rss_bytes", "process_tree_rss_bytes", "system_cpu_percent",
            "cuda_allocated_bytes", "cuda_reserved_bytes")}
        gpu_rows = [g for r in rows for g in r.get("gpus", [])]
        values.update(samples=len(rows), interval_seconds=self.interval,
            process_cpu_seconds=sum(max(r["process_cpu_seconds"] for r in rows if r.get('pid')==pid)
                for pid in {r.get('pid') for r in rows}),
            gpu_utilization_percent=distribution(g.get("utilization_percent") for g in gpu_rows),
            gpu_memory_used_mib=distribution(g.get("memory_used_mib") for g in gpu_rows),
            gpu_power_watts=distribution(g.get("power_watts") for g in gpu_rows),
            process_cuda_peak_allocated_bytes=max((r.get("cuda_peak_allocated_bytes",0) for r in rows),default=0) or None,
            process_cuda_peak_reserved_bytes=max((r.get("cuda_peak_reserved_bytes",0) for r in rows),default=0) or None,
            process_cuda_scope="PyTorch allocator in the BindPack process, including embedding weights; excludes vLLM in another process",
            gpu_scope="whole physical GPU including shared vLLM and embedding; not exclusive BindPack GPU cost",
            cpu_scope="process CPU percent: 100% equals one logical CPU; process-tree RSS includes descendants",
            peak_scope="sampled maximum, not guaranteed instantaneous high-water mark",
            monitor_errors=len(jsonl(self.recorder.directory / "monitor_errors.jsonl")))
        values["by_phase"] = {phase: {"samples":len(selected),
            "process_rss_bytes":distribution(r["process_rss_bytes"] for r in selected),
            "gpu_utilization_percent":distribution(g.get("utilization_percent") for r in selected for g in r.get("gpus", []))}
            for phase in sorted({r["phase"] for r in rows})
            for selected in [[r for r in rows if r["phase"] == phase]]}
        return values

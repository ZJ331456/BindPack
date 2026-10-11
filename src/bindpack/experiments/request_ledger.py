"""Observe per-question logical LLM costs without altering model requests.

Provider usage, including cached usage, measures equivalent logical work. It
does not measure GPU execution, HTTP retries, or billing. Those are exported
separately by the cached client's counters and workflow timing.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import functools
import hashlib
import json
from numbers import Integral
from pathlib import Path
import threading
import time


def normalize_usage(metadata: dict) -> dict:
    """Require provider counts rather than estimating missing response usage."""
    if not isinstance(metadata, dict):
        raise ValueError("LLM metadata must be a dictionary with actual token usage")
    usage = {}
    for key in ("prompt_tokens", "completion_tokens"):
        value = metadata.get(key)
        if not isinstance(value, Integral) or isinstance(value, bool) or value < 0:
            raise ValueError(f"LLM usage missing or invalid {key}")
        usage[key] = int(value)
    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    if "total_tokens" in metadata:
        declared = metadata["total_tokens"]
        if not isinstance(declared, Integral) or isinstance(declared, bool) or declared != usage["total_tokens"]:
            raise ValueError("LLM total_tokens disagrees with prompt/completion counts")
    return usage


class LLMCostLedger:
    """Record exact logical usage with explicit question and stage ownership.

    Scope is a ContextVar, so concurrent tasks cannot overwrite each other's
    owner. ThreadPoolExecutor jobs must enter scope in the worker (or use an
    explicit message resolver); Python does not inherit ContextVars into newly
    created worker threads automatically. Requests and returned tuples are
    forwarded unchanged. This observer does not enforce a token budget.
    """

    def __init__(self, *, tokenizer=None, tokenizer_name: str = "/root/models/Qwen3-8B", resolver=None):
        self.tokenizer_name = str(tokenizer_name)
        self._tokenizer = tokenizer
        self.resolver = resolver
        self._scope = ContextVar(f"llm_cost_ledger_scope_{id(self)}", default=None)
        self._lock = threading.RLock()
        self._records = []
        self._next_id = 0

    @staticmethod
    def _owner(value):
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError("LLM cost owner must be (question, stage)")
        query, stage = value
        if not isinstance(query, str) or not query.strip() or not isinstance(stage, str) or not stage.strip():
            raise ValueError("LLM cost owner question and stage must be nonempty strings")
        return query, stage

    @contextmanager
    def scope(self, query: str, stage: str):
        token = self._scope.set(self._owner((query, stage)))
        try:
            yield self
        finally:
            self._scope.reset(token)

    def current_owner(self):
        return self._scope.get()

    def _prompt_tokens(self, messages) -> int:
        # Initialization and template use share the lock, avoiding duplicate
        # loads and unsafe concurrent tokenizer mutation in worker threads.
        with self._lock:
            if self._tokenizer is None:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.tokenizer_name, local_files_only=True, trust_remote_code=True)
            return len(self._tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, enable_thinking=False))

    @property
    def tokenizer(self):
        """Use the same local chat tokenizer for online budget admission."""
        with self._lock:
            if self._tokenizer is None:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.tokenizer_name, local_files_only=True, trust_remote_code=True)
            return self._tokenizer

    @staticmethod
    def _request_output_allowance(infer, kwargs):
        owner = getattr(infer, "__self__", None)
        config = getattr(owner, "llm_config", None)
        defaults = getattr(config, "generate_params", {}) or {}
        effective = dict(defaults)
        effective.update(kwargs)
        extra = effective.get("extra_body") or {}
        chat_kwargs = extra.get("chat_template_kwargs", {})
        if chat_kwargs.get("enable_thinking", False):
            raise ValueError("Thinking must remain disabled during cost observation")
        # Observe the existing request; base fact filtering has its own original
        # 512 allowance. This ledger must not replace it with the main 2048 cap.
        # CachedOpenAI translates max_completion_tokens to max_tokens for vLLM.
        return next((effective[key] for key in ("max_completion_tokens", "max_tokens", "max_new_tokens")
                     if effective.get(key) is not None), None)

    def wrap_infer(self, infer, resolver=None):
        """Return a wrapper; installing it on the model is the caller's choice."""
        @functools.wraps(infer)
        def observed(*args, **kwargs):
            messages = args[0] if args else kwargs.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError("LLM infer must receive a nonempty messages list")
            owner = self._scope.get()
            if owner is None:
                # Capture sessions may install their lookup after wrapping the
                # client. Consult the current public resolver on each request.
                resolve = resolver if resolver is not None else self.resolver
                if resolve is None:
                    raise ValueError("Unattributed LLM request: enter ledger.scope or provide a resolver")
                owner = resolve(messages)
            query, stage = self._owner(owner)
            output_allowance = self._request_output_allowance(infer, kwargs)
            prompt_tokens = self._prompt_tokens(messages)
            message_hash = hashlib.sha256(json.dumps(
                messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            with self._lock:
                request_id = self._next_id
                self._next_id += 1
                record = {"request_id": request_id, "query": query, "stage": stage,
                          "messages_sha256": message_hash, "local_prompt_tokens": prompt_tokens,
                          "max_output_tokens": output_allowance, "thinking": False,
                          "status": "pending", "cache_hit": None, "usage": None,
                          "prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
                self._records.append(record)
            started = time.perf_counter()
            try:
                result = infer(*args, **kwargs)
                if not isinstance(result, tuple) or len(result) not in (2, 3):
                    raise ValueError("LLM must return a response/metadata tuple of length 2 or 3")
                usage = normalize_usage(result[1])
                if len(result) == 3 and not isinstance(result[2], bool):
                    raise ValueError("LLM cache-hit flag must be boolean when present")
            except BaseException as exc:
                with self._lock:
                    record.update(status="failed", elapsed_seconds=time.perf_counter() - started,
                                  error_type=type(exc).__name__, error=str(exc))
                raise
            with self._lock:
                record.update(status="success", elapsed_seconds=time.perf_counter() - started,
                              cache_hit=result[2] if len(result) == 3 else None, usage=usage,
                              finish_reason=result[1].get("finish_reason"),
                              prompt_count_delta=usage["prompt_tokens"] - prompt_tokens, **usage)
            return result

        return observed

    def get_request_records(self, query: str | None = None) -> list[dict]:
        with self._lock:
            return deepcopy([record for record in self._records if query is None or record["query"] == query])

    @staticmethod
    def _aggregate(records) -> dict:
        status = Counter(record["status"] for record in records)
        successful = [record for record in records if record["status"] == "success"]
        hits = sum(record["cache_hit"] is True for record in successful)
        misses = sum(record["cache_hit"] is False for record in successful)
        return {"logical_requests": len(records), "successful_requests": status["success"],
                "failed_requests": status["failed"], "pending_requests": status["pending"],
                "unmeasured_requests": len(records) - len(successful),
                "complete": len(records) == len(successful),
                "cache_hits": hits, "cache_misses": misses,
                "unknown_cache_status": len(successful) - hits - misses,
                "prompt_tokens": sum(record["usage"]["prompt_tokens"] for record in successful),
                "completion_tokens": sum(record["usage"]["completion_tokens"] for record in successful),
                "total_tokens": sum(record["usage"]["total_tokens"] for record in successful),
                "request_elapsed_seconds": sum(record.get("elapsed_seconds", 0.0) for record in records)}

    def get_summary(self, query: str | None = None) -> dict:
        records = self.get_request_records(query)
        summary = self._aggregate(records)
        summary["query"] = query
        summary["by_stage"] = {stage: self._aggregate([r for r in records if r["stage"] == stage])
                               for stage in sorted({r["stage"] for r in records})}
        summary["cost_semantics"] = "provider logical tokens including cached responses; not physical GPU/HTTP cost"
        return summary

    def write_json(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Hold the lock so summaries and records describe the same snapshot.
        with self._lock:
            records = self.get_request_records()
            payload = {"schema_version": 1, "tokenizer": self.tokenizer_name,
                       "summary": self.get_summary(),
                       "queries": {query: self.get_summary(query) for query in sorted({r["query"] for r in records})},
                       "requests": records}
            temporary = path.with_name(path.name + ".tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(path)
        return payload

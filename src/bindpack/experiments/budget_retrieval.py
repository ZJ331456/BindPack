"""Budgeted plain QD and iterative retrieval controls, outside the main method.

These controls reuse the same existing index, initial ranking, cached LLM and
dense retriever. They do not implement another paper's named baseline. Both
use identical unweighted RRF and omit source binding and evidence selection.
The initial supplied retrieval counts as one search. Token budgets account for
logical work, including cached calls; physical HTTP statistics belong to the
workflow's cached-client counters.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

import numpy as np


OUTPUT_TOKENS = 2048


def _object(text: str) -> dict | None:
    """Parse an object without repairing or guessing its contents."""
    text = str(text or "").strip()
    variants = [text]
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        variants.append(text[start:end + 1])
    for candidate in variants:
        try:
            value = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict):
            return value
    return None


def _query(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = " ".join(value.split())
    # Ordinary controls must emit usable text, not unresolved DAG references.
    if not value or len(value) > 1500 or re.search(r"\$\{|\b(?:entity-)?[0-9a-f]{32}\b", value, re.I):
        return None
    return value


class BudgetRetrievalControl:
    """Run one question with fixed maximum logical token/call/search budgets.

    ``mode`` is ``qd`` or ``iterative``. Budget admission reserves the exact
    local chat-template input count plus the unchanged 2048 output allowance.
    Insufficient budget abstains; transport failures and missing usage metadata
    raise, so an unsuccessful request cannot masquerade as a valid result.
    The object carries no per-query mutable budget state.
    """

    def __init__(self, rag, mode: str, token_budget: int, request_budget: int,
                 search_budget: int, *, tokenizer=None, candidate_top_k: int = 200,
                 rrf_constant: float = 60.0):
        aliases = {"plain_qd": "qd", "budget_qd": "qd", "budget_iterative": "iterative"}
        self.mode = aliases.get(mode, mode)
        if self.mode not in {"qd", "iterative"}:
            raise ValueError("mode must be qd or iterative")
        self.rag = rag
        self.token_budget = int(token_budget)
        self.request_budget = int(request_budget)
        self.search_budget = int(search_budget)
        self.candidate_top_k = int(candidate_top_k)
        self.rrf_constant = float(rrf_constant)
        if self.token_budget < 0 or self.request_budget < 0 or self.search_budget < 1:
            raise ValueError("budgets must be nonnegative, and searches include the initial search")
        if self.candidate_top_k < 1 or not math.isfinite(self.rrf_constant) or self.rrf_constant <= 0:
            raise ValueError("candidate_top_k and rrf_constant must be positive")
        self.cfg = getattr(rag, "pcrag_config", getattr(rag, "global_config", None))
        if int(getattr(self.cfg, "max_new_tokens", OUTPUT_TOKENS)) != OUTPUT_TOKENS:
            raise ValueError("Online controls require the unchanged 2048 output allowance")
        self._tokenizer = tokenizer

    def _prompt_tokens(self, messages: list[dict]) -> int:
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                getattr(self.cfg, "evidence_review_tokenizer", "/root/models/Qwen3-8B"),
                trust_remote_code=True, local_files_only=True)
        return len(self._tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False))

    def _request(self, messages: list[dict], trace: dict, purpose: str) -> dict | None:
        admission = {"purpose": purpose, "max_output_tokens": OUTPUT_TOKENS,
                     "messages_sha256": hashlib.sha256(json.dumps(
                         messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest()}
        remaining = self.token_budget - trace["total_tokens"]
        if trace["logical_calls"] >= self.request_budget:
            admission.update(allowed=False, reason="request_budget_exhausted")
        elif remaining < OUTPUT_TOKENS:
            admission.update(allowed=False, reason="token_budget_exhausted")
        else:
            prompt_tokens = self._prompt_tokens(messages)
            required = prompt_tokens + OUTPUT_TOKENS
            admission.update(prompt_tokens=prompt_tokens, reserved_tokens=required,
                             remaining_tokens=remaining)
            context_limit = int(getattr(self.cfg, "evidence_review_context_length", 8192))
            if required > context_limit:
                admission.update(allowed=False, reason="context_limit")
            elif required > remaining:
                admission.update(allowed=False, reason="token_budget_exhausted")
            else:
                admission.update(allowed=True, reason=None)
        trace["admissions"].append(admission)
        if not admission["allowed"]:
            return None
        result = self.rag.llm_model.infer(
            messages=messages, temperature=0.0, max_completion_tokens=OUTPUT_TOKENS,
            response_format={"type": "json_object"},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        if not isinstance(result, tuple) or len(result) < 2 or not isinstance(result[1], dict):
            raise RuntimeError("Cached LLM must return actual token usage metadata")
        response, metadata = result[:2]
        usage = {}
        for field in ("prompt_tokens", "completion_tokens"):
            value = metadata.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise RuntimeError(f"LLM returned missing or invalid {field}")
            usage[field] = value
            trace[field] += value
        trace["logical_calls"] += 1
        trace["total_tokens"] = trace["prompt_tokens"] + trace["completion_tokens"]
        if trace["total_tokens"] > self.token_budget or usage["completion_tokens"] > OUTPUT_TOKENS:
            raise RuntimeError("Actual LLM usage exceeded the admitted budget")
        raw = self.rag._extract_llm_text(response)
        event = dict(usage, purpose=purpose, response=raw,
                     finish_reason=metadata.get("finish_reason"),
                     cache_hit=bool(result[2]) if len(result) > 2 else None)
        trace["requests"].append(event)
        if metadata.get("finish_reason") == "length":
            trace["semantic_failures"].append({"purpose": purpose, "reason": "truncated_response"})
            return None
        payload = _object(raw)
        if payload is None:
            trace["semantic_failures"].append({"purpose": purpose, "reason": "invalid_json_object"})
        return payload

    @staticmethod
    def _rank(ids, scores, top_k: int) -> tuple[list[int], list[float]]:
        ids, scores = list(ids), list(scores)
        if len(ids) != len(scores) or len(ids) != len(set(int(d) for d in ids)):
            raise ValueError("Retrieval IDs and scores must be aligned and unique")
        if any(not math.isfinite(float(score)) for score in scores):
            raise ValueError("Retrieval scores must be finite")
        return [int(doc) for doc in ids[:top_k]], [float(score) for score in scores[:top_k]]

    def _search(self, query: str, routes: list[dict], trace: dict) -> bool:
        if any(route["query"] == query for route in routes):
            trace["skipped_searches"].append({"query": query, "reason": "duplicate_query"})
            return False
        if trace["search_count"] >= self.search_budget:
            trace["skipped_searches"].append({"query": query, "reason": "search_budget_exhausted"})
            return False
        ids, scores = self.rag.dense_passage_retrieval(query)
        ids, scores = self._rank(ids, scores, self.candidate_top_k)
        routes.append({"query": query, "doc_ids": ids, "raw_scores": scores, "source": self.mode})
        trace["search_count"] += 1
        return True

    def _fuse(self, routes: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        values = {}
        for route in routes:
            for rank, doc_id in enumerate(route["doc_ids"], 1):
                values[doc_id] = values.get(doc_id, 0.0) + 1.0 / (self.rrf_constant + rank)
        ordered = sorted(values, key=lambda doc: (-values[doc], doc))[:self.candidate_top_k]
        return np.asarray(ordered, dtype=int), np.asarray([values[doc] for doc in ordered], dtype=float)

    def _documents(self, ids: list[int]) -> list[dict]:
        output = []
        for doc_id in ids[:3]:
            row = self.rag.chunk_embedding_store.get_row(self.rag.passage_node_keys[doc_id])
            if not isinstance(row, dict) or "content" not in row:
                raise ValueError(f"No source document for physical ID {doc_id}")
            output.append({"doc_id": f"D{doc_id}", "text": str(row["content"])[:4000]})
        return output

    def _qd(self, question: str, hop: int, routes: list[dict], trace: dict):
        if self.search_budget <= trace["search_count"]:
            return
        messages = [
            {"role": "system", "content": (
                "Decompose a question into standalone retrieval subquestions. Output ONLY JSON "
                'as {"subquestions":["question",...]}. Use only the input question. '
                "Do not invent intermediate answers or entities and do not use variable placeholders. "
                "Preserve relation and location qualifiers. Each query must be ordinary text.")},
            {"role": "user", "content": f"Question: {question}\nMaximum subquestions: {hop}"},
        ]
        payload = self._request(messages, trace, "plain_qd_plan")
        if payload is None:
            return
        items = payload.get("subquestions")
        if not isinstance(items, list) or not items or len(items) > hop or any(_query(q) is None for q in items):
            trace["semantic_failures"].append({"purpose": "plain_qd_plan", "reason": "invalid_subquestions"})
            return
        for query in dict.fromkeys(_query(item) for item in items):
            self._search(query, routes, trace)

    def _iterative(self, question: str, hop: int, routes: list[dict], trace: dict):
        for step in range(1, hop + 1):
            if trace["search_count"] >= self.search_budget:
                break
            ids, _ = self._fuse(routes)
            documents = self._documents(ids.tolist())
            messages = [
                {"role": "system", "content": (
                    "Choose one next retrieval query using the question and retrieved text. Output ONLY JSON "
                    'as {"query":"ordinary retrieval question"} or {"query":null} when further search '
                    "is unnecessary. Reuse entities stated in the retrieved text when useful. Do not invent "
                    "facts, use placeholders, output an answer, or repeat previous queries. This is ordinary "
                    "iterative retrieval; do not output proofs, source bindings, or a dependency graph.")},
                {"role": "user", "content": json.dumps({
                    "question": question, "step": step, "maximum_steps": hop,
                    "previous_queries": [route["query"] for route in routes],
                    "retrieved_passages": documents}, ensure_ascii=False)},
            ]
            payload = self._request(messages, trace, f"iterative_query_{step}")
            if payload is None:
                break
            if payload.get("query", "missing") is None:
                trace["early_stop"] = "llm_no_further_search"
                break
            query = _query(payload.get("query"))
            if query is None:
                trace["semantic_failures"].append({"purpose": f"iterative_query_{step}",
                                                   "reason": "invalid_query"})
                break
            if not self._search(query, routes, trace):
                break

    def retrieve_one(self, question: str, base_ids, base_scores, hop: int):
        if _query(question) is None or isinstance(hop, bool) or int(hop) != hop or not 1 <= int(hop) <= 4:
            raise ValueError("A usable question and explicit benchmark/prior hop in [1,4] are required")
        question, hop = _query(question), int(hop)
        ids, scores = self._rank(base_ids, base_scores, self.candidate_top_k)
        routes = [{"query": question, "doc_ids": ids, "raw_scores": scores, "source": "common_initial"}]
        trace = {"control": self.mode, "hop": hop, "token_budget": self.token_budget,
                 "request_budget": self.request_budget, "search_budget": self.search_budget,
                 "selector": "unweighted_rrf", "rrf_constant": self.rrf_constant,
                 "max_output_tokens": OUTPUT_TOKENS, "thinking": False,
                 "logical_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                 "initial_search_count": 1, "search_count": 1, "routes": routes,
                 "admissions": [], "requests": [], "semantic_failures": [], "skipped_searches": []}
        if self.mode == "qd":
            self._qd(question, hop, routes, trace)
        else:
            self._iterative(question, hop, routes, trace)
        ids, scores = self._fuse(routes)
        trace["candidate_count"] = len(ids)
        trace["candidate_ids"] = ids.tolist()
        trace["remaining_tokens"] = self.token_budget - trace["total_tokens"]
        return ids, scores, trace

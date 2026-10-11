"""CPU verification of budget admission and plain online controls."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "src/bindpack/experiments/budget_retrieval.py"
SPEC = importlib.util.spec_from_file_location("budget_retrieval_under_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Tokenizer:
    def __init__(self, count=10):
        self.count, self.calls = count, []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return list(range(self.count))


class LLM:
    def __init__(self, responses):
        self.responses, self.calls = iter(responses), []

    def infer(self, **kwargs):
        self.calls.append(kwargs)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, tuple):
            return response
        return json.dumps(response), {"prompt_tokens": 10, "completion_tokens": 20,
                                     "finish_reason": "stop"}, False


class RAG:
    def __init__(self, responses=()):
        self.pcrag_config = SimpleNamespace(max_new_tokens=2048, evidence_review_context_length=8192)
        self.llm_model, self.searches = LLM(responses), []
        self.passage_node_keys = [f"chunk-{i}" for i in range(6)]
        self.chunk_embedding_store = SimpleNamespace(get_row=lambda key: {
            "content": f"{key}\nAlpha wrote Work X. Alpha was born in Rome."})

    def dense_passage_retrieval(self, query):
        self.searches.append(query)
        return np.array([2, 1, 0]), np.array([.95, .8, .6])

    @staticmethod
    def _extract_llm_text(response):
        return str(response)


QUESTION = "Where was the writer of Work X born?"


def control(rag, mode="qd", tokens=10000, calls=8, searches=8, tokenizer=None, **kwargs):
    return MODULE.BudgetRetrievalControl(
        rag, mode, tokens, calls, searches, tokenizer=tokenizer or Tokenizer(), **kwargs)


def retrieve(c):
    return c.retrieve_one(QUESTION, [0, 1, 2], [.9, .8, .7], 2)


class BudgetRetrievalTests(unittest.TestCase):
    def test_qd_uses_one_plan_and_real_hop_then_shared_rrf(self):
        rag = RAG([{"subquestions": ["Who wrote Work X?", "Where was this writer born?"]}])
        tokenizer = Tokenizer()
        ids, scores, trace = retrieve(control(rag, tokenizer=tokenizer))
        self.assertEqual(rag.searches, ["Who wrote Work X?", "Where was this writer born?"])
        self.assertEqual(trace["logical_calls"], 1)
        self.assertEqual(trace["total_tokens"], 30)
        self.assertEqual(trace["search_count"], 3)
        self.assertEqual(ids.tolist(), [2, 1, 0])
        self.assertAlmostEqual(scores[0], 1 / 63 + 2 / 61)
        self.assertIn("Maximum subquestions: 2", rag.llm_model.calls[0]["messages"][1]["content"])
        self.assertFalse(tokenizer.calls[0][1]["enable_thinking"])
        self.assertEqual(rag.llm_model.calls[0]["max_completion_tokens"], 2048)
        self.assertFalse(rag.llm_model.calls[0]["extra_body"]["chat_template_kwargs"]["enable_thinking"])

    def test_admission_reserves_full_2048_without_lowering_output(self):
        rag = RAG()
        ids, _, trace = retrieve(control(rag, tokens=2057))
        self.assertEqual(ids.tolist(), [0, 1, 2])
        self.assertEqual(rag.llm_model.calls, [])
        self.assertEqual(trace["admissions"][0]["reserved_tokens"], 2058)
        self.assertEqual(trace["admissions"][0]["reason"], "token_budget_exhausted")
        self.assertEqual(trace["max_output_tokens"], 2048)

    def test_success_usage_releases_unused_output_reservation(self):
        rag = RAG([{"query": "Who wrote Work X?"}, {"query": "Where was Alpha born?"}])
        _, _, trace = retrieve(control(rag, mode="iterative", tokens=2088))
        self.assertEqual(trace["logical_calls"], 2)
        self.assertEqual(trace["total_tokens"], 60)
        self.assertEqual(trace["remaining_tokens"], 2028)
        self.assertEqual(len(rag.searches), 2)

    def test_cached_response_counts_logical_tokens(self):
        response = json.dumps({"subquestions": ["Who wrote Work X?"]})
        rag = RAG([(response, {"prompt_tokens": 10, "completion_tokens": 25, "finish_reason": "stop"}, True)])
        _, _, trace = retrieve(control(rag))
        self.assertEqual(trace["total_tokens"], 35)
        self.assertTrue(trace["requests"][0]["cache_hit"])

    def test_zero_budget_and_zero_calls_do_not_load_tokenizer_or_call_llm(self):
        for tokens, calls in ((0, 8), (10000, 0)):
            rag, tokenizer = RAG(), Tokenizer()
            _, _, trace = retrieve(control(rag, tokens=tokens, calls=calls, tokenizer=tokenizer))
            self.assertEqual(tokenizer.calls, [])
            self.assertEqual(rag.llm_model.calls, [])
            self.assertEqual(trace["logical_calls"], 0)

    def test_context_limit_is_independent_of_token_budget(self):
        rag = RAG()
        rag.pcrag_config.evidence_review_context_length = 2050
        _, _, trace = retrieve(control(rag))
        self.assertEqual(trace["admissions"][0]["reason"], "context_limit")
        self.assertEqual(rag.llm_model.calls, [])

    def test_search_budget_includes_initial_route(self):
        rag = RAG([{"subquestions": ["Who wrote Work X?", "Where was this writer born?"]}])
        _, _, trace = retrieve(control(rag, searches=2))
        self.assertEqual(len(rag.searches), 1)
        self.assertEqual(trace["search_count"], 2)
        self.assertEqual(trace["skipped_searches"][0]["reason"], "search_budget_exhausted")
        rag = RAG()
        _, _, trace = retrieve(control(rag, searches=1))
        self.assertEqual(rag.llm_model.calls, [])

    def test_duplicate_queries_receive_no_double_fusion_credit(self):
        rag = RAG([{"subquestions": ["Who wrote Work X?", "Who wrote Work X?"]}])
        _, _, trace = retrieve(control(rag))
        self.assertEqual(len(rag.searches), 1)
        self.assertEqual(trace["search_count"], 2)
        rag = RAG([{"query": QUESTION}])
        _, _, trace = retrieve(control(rag, mode="iterative"))
        self.assertEqual(rag.searches, [])
        self.assertEqual(trace["skipped_searches"][0]["reason"], "duplicate_query")

    def test_iterative_gets_current_documents_and_can_stop_early(self):
        rag = RAG([{"query": "Who wrote Work X?"}, {"query": None}])
        _, _, trace = retrieve(control(rag, mode="iterative"))
        payload = json.loads(rag.llm_model.calls[1]["messages"][1]["content"])
        self.assertEqual(payload["previous_queries"], [QUESTION, "Who wrote Work X?"])
        self.assertEqual(len(payload["retrieved_passages"]), 3)
        self.assertEqual(trace["early_stop"], "llm_no_further_search")
        self.assertEqual(trace["logical_calls"], 2)

    def test_truncated_or_invalid_payload_is_reported_without_fake_success(self):
        for response, reason in (
            (("{}", {"prompt_tokens": 10, "completion_tokens": 20, "finish_reason": "length"}),
             "truncated_response"),
            (("bad", {"prompt_tokens": 10, "completion_tokens": 20, "finish_reason": "stop"}),
             "invalid_json_object"),
            ({"subquestions": ["Where was ${s1.answer} born?"]}, "invalid_subquestions"),
        ):
            rag = RAG([response])
            ids, _, trace = retrieve(control(rag))
            self.assertEqual(ids.tolist(), [0, 1, 2])
            self.assertEqual(trace["semantic_failures"][0]["reason"], reason)
            self.assertEqual(trace["logical_calls"], 1)
            self.assertEqual(trace["search_count"], 1)

    def test_transport_error_propagates(self):
        with self.assertRaisesRegex(RuntimeError, "transport"):
            retrieve(control(RAG([RuntimeError("transport")])) )

    def test_missing_usage_and_actual_overspend_raise(self):
        for metadata in ({"finish_reason": "stop"},
                         {"prompt_tokens": 10, "completion_tokens": 2049, "finish_reason": "stop"},
                         {"prompt_tokens": 2060, "completion_tokens": 2, "finish_reason": "stop"}):
            with self.assertRaises(RuntimeError):
                retrieve(control(RAG([('{}', metadata)]), tokens=2058))

    def test_query_budgets_reset_between_questions(self):
        rag = RAG([{"subquestions": ["Who wrote Work X?"]}, {"subquestions": ["Who wrote Work X?"]}])
        c = control(rag, calls=1)
        for _ in range(2):
            _, _, trace = retrieve(c)
            self.assertEqual(trace["logical_calls"], 1)
        self.assertEqual(len(rag.llm_model.calls), 2)

    def test_validates_inputs_and_preserves_candidate_bound(self):
        rag = RAG([{"subquestions": ["Who wrote Work X?"]}])
        ids, _, trace = retrieve(control(rag, candidate_top_k=2))
        self.assertEqual(len(ids), 2)
        self.assertEqual(trace["candidate_count"], 2)
        with self.assertRaises(ValueError):
            control(rag, searches=0)
        with self.assertRaises(ValueError):
            control(rag).retrieve_one(QUESTION, [0, 0], [.9, .8], 2)
        with self.assertRaises(ValueError):
            control(rag).retrieve_one(QUESTION, [0], [.9], 0)
        rag.pcrag_config.max_new_tokens = 200
        with self.assertRaises(ValueError):
            control(rag)


if __name__ == "__main__":
    unittest.main()

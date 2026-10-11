"""CPU checks for exact, attributable and thread-safe logical LLM costs."""

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


PATH = Path(__file__).resolve().parents[1] / "src/bindpack/experiments/request_ledger.py"
SPEC = importlib.util.spec_from_file_location("request_ledger_under_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


MESSAGES = [{"role": "user", "content": "Where was Alpha born?"}]


class Tokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return [1] * 12


class LLM:
    def __init__(self, result=None):
        self.result = result or ("Rome", {"prompt_tokens": 12, "completion_tokens": 2,
                                          "finish_reason": "stop"}, False)
        self.llm_config = SimpleNamespace(generate_params={"max_completion_tokens": 2048})
        self.calls = []

    def infer(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class CostLedgerTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer()
        self.ledger = MODULE.LLMCostLedger(tokenizer=self.tokenizer)

    def test_original_arguments_and_return_tuple_are_unchanged(self):
        llm = LLM()
        observed = self.ledger.wrap_infer(llm.infer)
        kwargs = {"temperature": 0.0, "max_completion_tokens": 2048,
                  "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
        with self.ledger.scope("question", "base_fact_filter"):
            result = observed(MESSAGES, **kwargs)
        self.assertIs(result, llm.result)
        self.assertIs(llm.calls[0][0][0], MESSAGES)
        self.assertEqual(llm.calls[0][1], kwargs)
        self.assertEqual(self.tokenizer.calls[0][1], {
            "tokenize": True, "add_generation_prompt": True, "enable_thinking": False})
        summary = self.ledger.get_summary("question")
        self.assertEqual(summary["total_tokens"], 14)
        self.assertEqual(summary["by_stage"]["base_fact_filter"]["logical_requests"], 1)
        record = self.ledger.get_request_records("question")[0]
        self.assertEqual(record["prompt_tokens"], 12)
        self.assertEqual(record["completion_tokens"], 2)

    def test_keyword_messages_and_tuple_two_have_unknown_cache_status(self):
        llm = LLM(("Rome", {"prompt_tokens": 12, "completion_tokens": 2}))
        with self.ledger.scope("q", "plan"):
            result = self.ledger.wrap_infer(llm.infer)(messages=MESSAGES)
        self.assertIs(result, llm.result)
        self.assertEqual(self.ledger.get_summary("q")["unknown_cache_status"], 1)

    def test_unknown_ownership_fails_before_any_model_request(self):
        llm = LLM()
        with self.assertRaisesRegex(ValueError, "Unattributed"):
            self.ledger.wrap_infer(llm.infer)(MESSAGES)
        self.assertEqual(llm.calls, [])
        self.assertEqual(self.ledger.get_request_records(), [])
        self.assertEqual(self.tokenizer.calls, [])

    def test_resolver_and_nested_scope_restore_parent_owner(self):
        llm = LLM()
        observed = self.ledger.wrap_infer(llm.infer, resolver=lambda messages: ("resolved", "fallback"))
        observed(MESSAGES)
        with self.ledger.scope("outer", "base"):
            observed(MESSAGES)
            with self.ledger.scope("inner", "verify"):
                observed(MESSAGES)
            observed(MESSAGES)
        observed(MESSAGES)
        self.assertEqual(self.ledger.get_summary("outer")["logical_requests"], 2)
        self.assertEqual(self.ledger.get_summary("inner")["logical_requests"], 1)
        self.assertEqual(self.ledger.get_summary("resolved")["logical_requests"], 2)

    def test_scope_reset_after_exception(self):
        with self.assertRaises(RuntimeError):
            with self.ledger.scope("q", "stage"):
                raise RuntimeError("stop")
        with self.assertRaisesRegex(ValueError, "Unattributed"):
            self.ledger.wrap_infer(LLM().infer)(MESSAGES)

    def test_latest_public_resolver_is_used_after_wrapper_installation(self):
        observed = self.ledger.wrap_infer(LLM().infer)
        self.ledger.resolver = lambda messages: ("first", "plan")
        observed(MESSAGES)
        self.ledger.resolver = lambda messages: ("second", "base_fact_filter")
        observed(MESSAGES)
        self.assertEqual(self.ledger.get_summary("first")["logical_requests"], 1)
        self.assertEqual(self.ledger.get_summary("second")["logical_requests"], 1)

    def test_cached_calls_count_logical_work_not_gpu_execution(self):
        llm = LLM(("Rome", {"prompt_tokens": 12, "completion_tokens": 2}, True))
        with self.ledger.scope("q", "plan"):
            self.ledger.wrap_infer(llm.infer)(MESSAGES)
        summary = self.ledger.get_summary("q")
        self.assertEqual(summary["cache_hits"], 1)
        self.assertEqual(summary["total_tokens"], 14)
        self.assertIn("not physical", summary["cost_semantics"])

    def test_transport_error_is_recorded_and_propagated(self):
        llm = LLM(RuntimeError("transport failure"))
        with self.ledger.scope("q", "verify"):
            with self.assertRaisesRegex(RuntimeError, "transport failure"):
                self.ledger.wrap_infer(llm.infer)(MESSAGES)
        summary = self.ledger.get_summary("q")
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["failed_requests"], 1)
        self.assertEqual(summary["unmeasured_requests"], 1)
        record = self.ledger.get_request_records("q")[0]
        self.assertEqual(record["error_type"], "RuntimeError")
        self.assertGreaterEqual(record["elapsed_seconds"], 0)

    def test_missing_usage_is_never_replaced_by_local_estimate(self):
        llm = LLM(("Rome", {"completion_tokens": 2}))
        with self.ledger.scope("q", "verify"):
            with self.assertRaisesRegex(ValueError, "prompt_tokens"):
                self.ledger.wrap_infer(llm.infer)(MESSAGES)
        self.assertFalse(self.ledger.get_summary("q")["complete"])
        self.assertEqual(self.ledger.get_summary("q")["prompt_tokens"], 0)

    def test_actual_server_tokens_override_local_diagnostic_count(self):
        llm = LLM(("Rome", {"prompt_tokens": 15, "completion_tokens": 2}))
        with self.ledger.scope("q", "base"):
            self.ledger.wrap_infer(llm.infer)(MESSAGES)
        self.assertEqual(self.ledger.get_summary("q")["total_tokens"], 17)
        self.assertEqual(self.ledger.get_request_records("q")[0]["prompt_count_delta"], 3)

    def test_invalid_token_values_and_inconsistent_total_raise(self):
        for metadata in ({"prompt_tokens": -1, "completion_tokens": 2},
                         {"prompt_tokens": True, "completion_tokens": 2},
                         {"prompt_tokens": 1.0, "completion_tokens": 2},
                         {"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 99}):
            with self.assertRaises(ValueError):
                MODULE.normalize_usage(metadata)

    def test_output_allowance_is_observed_and_thinking_cannot_be_enabled(self):
        llm = LLM()
        observed = self.ledger.wrap_infer(llm.infer)
        with self.ledger.scope("q", "plan"):
            with self.assertRaises(ValueError):
                observed(MESSAGES, extra_body={"chat_template_kwargs": {"enable_thinking": True}})
        self.assertEqual(llm.calls, [])

    def test_original_fact_filter_512_is_forwarded_without_any_override(self):
        llm = LLM(("Rome", {"prompt_tokens": 12, "completion_tokens": 500}, False))
        observed = self.ledger.wrap_infer(llm.infer)
        with self.ledger.scope("q", "base_fact_filter"):
            result = observed(messages=MESSAGES, max_completion_tokens=512)
        self.assertIs(result, llm.result)
        self.assertEqual(llm.calls[0][1], {"messages": MESSAGES, "max_completion_tokens": 512})
        self.assertEqual(self.ledger.get_request_records("q")[0]["max_output_tokens"], 512)
        self.assertEqual(self.ledger.get_summary("q")["total_tokens"], 512)
        self.assertTrue(self.ledger.get_summary("q")["complete"])

    def test_thread_scopes_do_not_contaminate_owner_or_lose_records(self):
        llm = LLM()
        observed = self.ledger.wrap_infer(llm.infer)

        def job(index):
            with self.ledger.scope(f"q{index % 5}", f"stage{index % 2}"):
                return observed(MESSAGES)

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(job, range(100)))
        self.assertEqual(self.ledger.get_summary()["logical_requests"], 100)
        self.assertEqual(self.ledger.get_summary()["total_tokens"], 1400)
        self.assertEqual(len({r["request_id"] for r in self.ledger.get_request_records()}), 100)
        for index in range(5):
            summary = self.ledger.get_summary(f"q{index}")
            self.assertEqual(summary["logical_requests"], 20)
            self.assertEqual(summary["by_stage"]["stage0"]["logical_requests"], 10)
            self.assertEqual(summary["by_stage"]["stage1"]["logical_requests"], 10)

    def test_records_are_copies_and_json_is_a_complete_snapshot(self):
        with self.ledger.scope("q", "base"):
            self.ledger.wrap_infer(LLM().infer)(MESSAGES)
        records = self.ledger.get_request_records("q")
        records[0]["usage"]["total_tokens"] = 999
        self.assertEqual(self.ledger.get_summary("q")["total_tokens"], 14)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "costs/request_ledger.json"
            payload = self.ledger.write_json(path)
            self.assertEqual(json.loads(path.read_text()), payload)
            self.assertEqual(payload["queries"]["q"]["total_tokens"], 14)
            self.assertEqual(len(payload["requests"]), 1)
            self.assertFalse(path.with_name(path.name + ".tmp").exists())


if __name__ == "__main__":
    unittest.main()

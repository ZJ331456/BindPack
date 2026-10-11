"""Exercise actual review fetch/thread/ledger paths without a model or GPU."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import ModuleType, SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ModuleType('review_cost_scope_test_package')
PACKAGE.__path__ = [str(ROOT / 'src/bindpack')]
sys.modules[PACKAGE.__name__] = PACKAGE


class FakeEvidence:
    def finalize(self, *args, **kwargs):
        return None


evidence = ModuleType(PACKAGE.__name__ + '.evidence.evidence_retrieval')
evidence.EvidenceRetrieval = FakeEvidence
evidence.branch_matches = lambda requirements, bindings: all(
    bindings.get(key) == value for key, value in requirements.items())
sys.modules[evidence.__name__] = evidence
spec = importlib.util.spec_from_file_location(
    PACKAGE.__name__ + '.experiments.capture', ROOT / 'src/bindpack/experiments/capture.py')
capture = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = capture
spec.loader.exec_module(capture)
ledger_spec = importlib.util.spec_from_file_location(
    PACKAGE.__name__ + '.experiments.request_ledger',
    ROOT / 'src/bindpack/experiments/request_ledger.py')
ledger_module = importlib.util.module_from_spec(ledger_spec)
sys.modules[ledger_spec.name] = ledger_module
ledger_spec.loader.exec_module(ledger_module)


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return [1] * len(messages[0]['content'])


class FakeLLM:
    def __init__(self):
        self.llm_config = SimpleNamespace(generate_params={'max_completion_tokens': 2048})
        self.calls = []
        self.barrier = None
        self.error = None
        self.lock = threading.Lock()
        self.metadata = {'prompt_tokens': 17, 'completion_tokens': 3, 'finish_reason': 'stop'}

    def infer(self, messages, **kwargs):
        if self.barrier is not None:
            self.barrier.wait(timeout=5)
        if self.error is not None:
            raise self.error
        with self.lock:
            self.calls.append((messages, kwargs))
        return messages[0]['content'], self.metadata, False


class ReviewCostScopeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.ledger = ledger_module.LLMCostLedger(tokenizer=FakeTokenizer(),
                                               resolver=self.forbid_prompt_resolution)
        self.model = FakeLLM()
        self.model.infer = self.ledger.wrap_infer(self.model.infer)
        self.rag = SimpleNamespace(llm_model=self.model, _extract_llm_text=lambda response: response)
        self.runtime = SimpleNamespace(rag=self.rag,
            cfg=SimpleNamespace(evidence_review_context_length=4096, max_new_tokens=2048),
            _support_semantic_tokenizer=FakeTokenizer())
        self.rag.evidence_runtime = self.runtime
        self.original_fetch = capture.support_review._fetch
        self.session = capture.CaptureSession(self.rag, self.temp.name, self.ledger)
        self.session._install_review_scopes()
        self.addCleanup(self.session.close)

    @staticmethod
    def forbid_prompt_resolution(messages):
        raise AssertionError('Review ownership must come from its actual task, never prompt text')

    @staticmethod
    def context(query, stage, messages=None):
        if messages is None:
            messages = [{'role': 'user', 'content': json.dumps({'question': query})}]
        return {'solution': SimpleNamespace(question=query), stage + '_messages': messages}

    def test_json_escaped_queries_get_exact_owners_in_real_worker_threads(self):
        queries = ['Who wrote "A quoted title"?', 'Which person\nwas born there?',
                   'Which path contains C:\\tmp\\file and \u4e2d\u6587?']
        self.model.barrier = threading.Barrier(len(queries))
        with ThreadPoolExecutor(max_workers=len(queries)) as executor:
            for stage in ('proposer', 'reviewer'):
                contexts = [self.context(query, stage) for query in queries]
                for ctx in contexts:
                    self.assertNotIn(ctx['solution'].question, ctx[stage + '_messages'][0]['content'])
                # The caller's context also must not leak into worker records.
                with self.ledger.scope('unrelated caller question', 'caller'):
                    result = capture.support_review._fetch(self.runtime, executor, contexts, stage)
                self.assertIsNone(result)  # Original _fetch API is preserved.
                for ctx in contexts:
                    self.assertTrue(ctx[stage + '_preflight']['allowed'])
                    output = ctx[stage + '_response']
                    self.assertEqual(output['response'], ctx[stage + '_messages'][0]['content'])
                    self.assertIs(output['metadata'], self.model.metadata)
                    self.assertFalse(output['cache_hit'])
        records = self.ledger.get_request_records()
        self.assertEqual({(row['query'], row['stage']) for row in records},
                         {(query, stage) for query in queries for stage in ('proposer', 'reviewer')})
        self.assertEqual(len(records), 6)
        self.assertTrue(self.ledger.get_summary()['complete'])
        self.assertEqual(self.ledger.get_summary()['total_tokens'], 120)
        for messages, kwargs in self.model.calls:
            self.assertEqual(kwargs, {'response_format': {'type': 'json_object'}})
        self.assertIsNone(self.ledger.current_owner())

    def test_skipped_and_overlength_contexts_still_use_original_preflight(self):
        missing = {'solution': SimpleNamespace(question='no message')}
        denied = self.context('too long', 'proposer', [{'role': 'user', 'content': 'x' * 4096}])
        allowed = self.context('answer "this"', 'proposer')
        with ThreadPoolExecutor(max_workers=1) as executor:
            capture.support_review._fetch(self.runtime, executor, [missing, denied, allowed], 'proposer')
            # Each job resets ContextVar even when a pool thread is reused.
            self.assertIsNone(executor.submit(self.ledger.current_owner).result())
        self.assertNotIn('proposer_response', missing)
        self.assertNotIn('proposer_preflight', missing)
        self.assertFalse(denied['proposer_preflight']['allowed'])
        self.assertNotIn('proposer_response', denied)
        self.assertEqual(self.ledger.get_summary()['logical_requests'], 1)
        self.assertEqual(self.ledger.get_request_records()[0]['query'], 'answer "this"')

    def test_actual_context_works_even_if_multiple_tasks_share_one_message_object(self):
        messages = [{'role': 'user', 'content': '{"question":"identical frozen prompt"}'}]
        contexts = [self.context('first real task', 'proposer', messages),
                    self.context('second real task', 'proposer', messages)]
        with ThreadPoolExecutor(max_workers=2) as executor:
            capture.support_review._fetch(self.runtime, executor, contexts, 'proposer')
        self.assertEqual({r['query'] for r in self.ledger.get_request_records()},
                         {'first real task', 'second real task'})
        self.assertTrue(all('proposer_response' in ctx for ctx in contexts))

    def test_duplicate_batch_ledger_replacement_is_read_at_execution(self):
        question = 'Same "duplicate" question'
        with ThreadPoolExecutor(max_workers=1) as executor:
            capture.support_review._fetch(self.runtime, executor,
                                         [self.context(question, 'proposer')], 'proposer')
            second_ledger = ledger_module.LLMCostLedger(tokenizer=FakeTokenizer(),
                                                       resolver=self.forbid_prompt_resolution)
            original_infer = self.model.infer.__wrapped__
            self.model.infer = second_ledger.wrap_infer(original_infer)
            self.session.ledger = second_ledger
            capture.support_review._fetch(self.runtime, executor,
                                         [self.context(question, 'reviewer')], 'reviewer')
        self.assertEqual(self.ledger.get_summary()['logical_requests'], 1)
        self.assertEqual(second_ledger.get_summary()['logical_requests'], 1)
        self.assertEqual(second_ledger.get_request_records()[0]['stage'], 'reviewer')

    def test_failure_propagates_and_close_restores_module_fetch(self):
        self.model.error = RuntimeError('transport failed')
        with ThreadPoolExecutor(max_workers=1) as executor:
            with self.assertRaisesRegex(RuntimeError, 'transport failed'):
                capture.support_review._fetch(self.runtime, executor,
                                             [self.context('Who said "hello"?', 'proposer')], 'proposer')
        self.assertEqual(self.ledger.get_summary()['failed_requests'], 1)
        self.session.close()
        self.assertIs(capture.support_review._fetch, self.original_fetch)
        self.assertTrue((Path(self.temp.name) / 'manifest.json').is_file())


if __name__ == '__main__':
    unittest.main()

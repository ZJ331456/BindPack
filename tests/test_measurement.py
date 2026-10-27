"""Physical request accounting, native-call noninterference and context budgets."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
namespace=types.ModuleType('_measurement_tests')
namespace.__path__=[str(ROOT/'src/bindpack/evaluation')]
sys.modules.setdefault('_measurement_tests',namespace)
def load(name):
    spec=importlib.util.spec_from_file_location('_measurement_tests.'+name, ROOT/'src/bindpack/evaluation'/f'{name}.py')
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module
M=load('measurement');B=load('measured_reader')


class Tokenizer:
    def encode(self,text,**kwargs):return list(text)
    def decode(self,tokens,**kwargs):return ''.join(tokens)
    def apply_chat_template(self,messages,**kwargs):return list(''.join(m['content'] for m in messages))


class MeasurementTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name);self.recorder=M.Recorder(self.root)

    def llm(self,fail_first=False,always_cached=False):
        seen=[]
        response=types.SimpleNamespace(usage=types.SimpleNamespace(prompt_tokens=11,completion_tokens=7),
            choices=[types.SimpleNamespace(finish_reason='stop')],id='response')
        def create(**kwargs):
            seen.append(kwargs)
            if fail_first and len(seen)==1:raise TimeoutError('transport timeout')
            return response
        client=types.SimpleNamespace(max_retries=0,chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
        llm=types.SimpleNamespace(openai_client=client)
        def infer(messages,**kwargs):
            if not always_cached:
                for i in range(2):
                    try:
                        llm.openai_client.chat.completions.create(model='qwen3-8b',messages=messages,**kwargs)
                        break
                    except TimeoutError:
                        if i==1:raise
            return 'unchanged native output',{'prompt_tokens':11,'completion_tokens':7},always_cached
        llm.infer=infer
        self.recorder.install_llm(llm,Tokenizer())
        return llm,seen

    def test_native_arguments_and_output_unchanged_real_usage_logged(self):
        llm,seen=self.llm()
        messages=[{'role':'user','content':'query'}]
        with self.recorder.section('online_reader','q1'):
            result=llm.infer(messages,max_tokens=2048)
        self.assertEqual(result[0],'unchanged native output')
        self.assertEqual(seen,[dict(model='qwen3-8b',messages=messages,max_tokens=2048)])
        events=M.jsonl(self.root/'llm_calls.jsonl')
        self.assertEqual(M.usage(events)['input_tokens'],11)
        self.assertEqual(events[0]['query_id'],'q1')
        self.assertEqual(M.jsonl(self.root/'llm_logical_calls.jsonl')[0]['prompt_tokens_local'],5)

    def test_failed_attempt_retained_without_inventing_token_usage(self):
        llm,_=self.llm(fail_first=True)
        llm.infer([{'role':'user','content':'q'}])
        events=M.jsonl(self.root/'llm_calls.jsonl');usage=M.usage(events)
        self.assertEqual(len({e['call_id'] for e in events}),1)
        self.assertEqual(usage['http_attempts'],2)
        self.assertEqual(usage['failed_attempts'],1)
        self.assertEqual(usage['attempts_with_unknown_usage'],1)
        self.assertEqual(usage['input_tokens'],11)

    def test_cache_response_does_not_count_as_physical_tokens(self):
        llm,seen=self.llm(always_cached=True)
        llm.infer([{'role':'user','content':'q'}])
        self.assertEqual(seen,[])
        self.assertEqual(M.usage(M.jsonl(self.root/'llm_calls.jsonl'))['input_tokens'],0)
        self.assertTrue(M.jsonl(self.root/'llm_logical_calls.jsonl')[0]['cache_hit'])

    def test_parallel_native_calls_keep_separate_identifiers(self):
        llm,_=self.llm()
        with ThreadPoolExecutor(max_workers=8) as pool:
            values=list(pool.map(lambda n:llm.infer([{'role':'user','content':str(n)}]),range(24)))
        events=M.jsonl(self.root/'llm_calls.jsonl')
        self.assertEqual(len(events),24)
        self.assertEqual(len({e['call_id'] for e in events}),24)
        self.assertEqual(len(values),24)

    def test_native_callback_captured_during_initialization_is_also_observed(self):
        response=types.SimpleNamespace(usage=types.SimpleNamespace(prompt_tokens=11,completion_tokens=7),
            choices=[types.SimpleNamespace(finish_reason='stop')],id='response')
        class NativeLLM:
            def __init__(self):
                self.openai_client=types.SimpleNamespace(max_retries=0,chat=types.SimpleNamespace(
                    completions=types.SimpleNamespace(create=lambda **kwargs:response)))
            def infer(self,messages,**kwargs):
                self.openai_client.chat.completions.create(model='qwen3-8b',messages=messages)
                return 'answer',{},False
        self.recorder.install_llm_factory(NativeLLM,Tokenizer())
        llm=NativeLLM()
        # DSPyFilter captures this callback in BaseRAG.__init__, before retrieval.
        native_reranker_callback=llm.infer
        native_reranker_callback([{'role':'user','content':'rank facts'}])
        physical=M.jsonl(self.root/'llm_calls.jsonl')
        logical=M.jsonl(self.root/'llm_logical_calls.jsonl')
        self.assertEqual(physical[0]['call_id'],logical[0]['call_id'])
        self.assertIsNotNone(physical[0]['call_id'])

    def test_embedding_observer_keeps_inputs_and_vectors(self):
        calls=[];vectors=object()
        def encode(texts,**kw):calls.append((texts,kw));return vectors
        embedding=types.SimpleNamespace(batch_encode=encode,model=types.SimpleNamespace(tokenizer=lambda vs,**kw:{'input_ids':[list(v) for v in vs]},max_seq_length=4),_is_qwen3_embedding=True)
        self.recorder.install_embedding(embedding)
        self.assertIs(embedding.batch_encode(['abc','123456']),vectors)
        self.assertEqual(calls,[(['abc','123456'],{})])
        event=M.jsonl(self.root/'embedding_calls.jsonl')[0]
        self.assertEqual(event['effective_tokens'],7);self.assertEqual(event['truncated_texts'],1)

    def test_resource_sampling_is_explicitly_shared_and_missing_gpu_is_na(self):
        monitor=M.ResourceMonitor(self.recorder)
        with patch.object(M.subprocess,'run',return_value=types.SimpleNamespace(stdout='0, GPU, 40, 100, 200, 20\n')):
            monitor.sample()
        summary=monitor.summary()
        self.assertEqual(summary['gpu_utilization_percent']['mean'],40)
        self.assertIn('not exclusive',summary['gpu_scope'])
        self.assertIsNone(M.distribution([])['mean'])

    def test_native_trace_numeric_keys_and_tuples_survive_checkpoint_serialization(self):
        value={'trace':{2:('entity',.7),10:('branch',.3)},'docs':['A\nbody']}
        persisted=json.loads(json.dumps(value))
        self.assertEqual(M.digest(value),M.digest(persisted))
        self.recorder.append('progress.jsonl',dict(sha256=M.digest(value),row=value))
        event=M.jsonl(self.root/'progress.jsonl')[0]
        self.assertEqual(event['sha256'],M.digest(event['row']))


class ReaderBudgetTests(unittest.TestCase):
    def rag(self):
        manager=types.SimpleNamespace(is_template_name_valid=lambda **kw:True,
            render=lambda **kw:[{'role':'user','content':kw['prompt_user']}])
        return types.SimpleNamespace(prompt_template_manager=manager)

    def test_reader_budget_never_mutates_the_retrieval_ranking(self):
        original=types.SimpleNamespace(question='claim',docs=['A\nshort','B\n'+('long '*1000),'C\nthird'])
        clone,budget=B.prepare(self.rag(),original,Tokenizer(),'hover',context_window=2600,output_tokens=2048)
        self.assertEqual(len(original.docs[1]),5002)
        self.assertLessEqual(budget['prompt_tokens_local']+2048,2600)
        self.assertEqual(budget['partial_passage_ranks'],[2])
        self.assertEqual(budget['dropped_passage_ranks'],[3])
        self.assertTrue(clone.docs[-1].startswith('B\n'))
        self.assertEqual(budget['prompt_sha256'],M.digest(B.render(self.rag(),original.question,clone.docs,'hover')))

    def test_short_evidence_unchanged_and_uses_baseline_evidence_token_definition(self):
        original=types.SimpleNamespace(question='claim',docs=['A\nshort','B\ntext'])
        clone,budget=B.prepare(self.rag(),original,Tokenizer(),'hover')
        self.assertEqual(clone.docs,original.docs)
        self.assertEqual(budget['evidence_tokens'],len('Wikipedia Title: A\nshort\n\nWikipedia Title: B\ntext\n\n'))
        self.assertEqual(budget['partial_passage_ranks'],[])


if __name__=='__main__':unittest.main()

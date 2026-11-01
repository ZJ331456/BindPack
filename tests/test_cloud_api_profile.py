"""Cloud wire format, model identity and cache-aware monetary estimates."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch,Mock
import tempfile
import httpx
import openai

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from bindpack.llm.openai_gpt import _effective_generation_params,_is_dashscope_endpoint,_content_rejection_metadata
from utils.api_cost import cost_for_usage,summarize_calls
from bindpack.index.openie.openie_structured_output import structured_request_parameters
from bindpack.index.openie.structural_openie import StructuralOpenIE
from bindpack.index.openie_build_queue import run_openie_queue
from bindpack.configuration.runtime import RuntimeConfig
from bindpack.llm.openai_gpt import CacheOpenAI
from bindpack.evaluation.measurement import Recorder,jsonl,usage


class CloudAPIProfileTests(unittest.TestCase):
    def llm(self,endpoint):
        return SimpleNamespace(llm_name='qwen3.8-flash',llm_base_url=endpoint,
            llm_config=SimpleNamespace(generate_params={'model':'qwen3.8-flash','temperature':0,
                'extra_body':{'chat_template_kwargs':{'enable_thinking':False}}}))

    def test_dashscope_uses_top_level_flag_and_disables_thinking_overrides(self):
        llm=self.llm('https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1')
        actual=_effective_generation_params(llm,{'extra_body':{'enable_thinking':True}})
        self.assertEqual(actual['extra_body'],{'enable_thinking':False})
        self.assertEqual(llm.llm_config.generate_params['extra_body'],{'chat_template_kwargs':{'enable_thinking':False}})

    def test_local_server_keeps_its_existing_template_flag(self):
        llm=self.llm('http://127.0.0.1:8035/v1')
        self.assertEqual(_effective_generation_params(llm,{})['extra_body'],{'chat_template_kwargs':{'enable_thinking':False}})
        self.assertFalse(_is_dashscope_endpoint('https://maas.aliyuncs.com.attacker.example'))
        self.assertFalse(_is_dashscope_endpoint('https://notdashscope.aliyuncs.com'))

    def test_caching_discount_applies_only_to_reported_cached_input(self):
        usage=cost_for_usage(1000000,1000000,cached_tokens=250000)
        self.assertAlmostEqual(usage['estimated_cny'],.75*.8+.25*.1+2.7)
        self.assertAlmostEqual(usage['all_input_at_standard_rate_cny'],3.5)
        self.assertAlmostEqual(cost_for_usage(1000000,0,created_tokens=1000000)['estimated_cny'],1.25)
        with self.assertRaises(ValueError):cost_for_usage(100,10,cached_tokens=101)

    def test_cloud_extraction_uses_schema_without_vllm_grammar(self):
        llm=self.llm('https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1')
        schema={'type':'object','properties':{'named_entities':{'type':'array','items':{'type':'string'}}},
                'required':['named_entities'],'additionalProperties':False}
        actual=structured_request_parameters(llm,schema,{'guided_grammar':'local-only'})
        self.assertEqual(actual['extra_body'],{'enable_thinking':False})
        self.assertEqual(actual['response_format']['json_schema']['schema'],schema)
        self.assertTrue(actual['response_format']['json_schema']['strict'])
        actual['response_format']['json_schema']['schema']['properties'].clear()
        self.assertIn('named_entities',schema['properties'])

    def test_cost_retains_failures_and_unknown_usage_without_fabricating_discount(self):
        events=[{'status':'success','prompt_tokens':1000,'completion_tokens':200,
                 'usage_details':{'prompt_tokens_details':{'cached_tokens':400}}},
                {'status':'success','prompt_tokens':500,'completion_tokens':100},
                {'status':'error','prompt_tokens':None,'completion_tokens':None}]
        result=summarize_calls(events)
        self.assertEqual(result['physical_requests'],3);self.assertEqual(result['failed_attempts'],1)
        self.assertEqual(result['unknown_usage_attempts'],1)
        self.assertEqual(result['cache_usage_unreported_attempts'],1)
        self.assertEqual(result['cached_tokens'],400)
        self.assertAlmostEqual(result['estimated_cny'],(1100*.8+400*.1+300*2.7)/1000000)

    def test_only_explicit_cloud_content_rejection_can_abstain(self):
        llm=self.llm('https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1')
        response=httpx.Response(400,request=httpx.Request('POST',llm.llm_base_url))
        rejected=openai.BadRequestError('content rejected',response=response,
            body={'error':{'code':'data_inspection_failed'}})
        invalid=openai.BadRequestError('invalid parameter',response=response,
            body={'error':{'code':'invalid_parameter'}})
        with patch.dict('os.environ',{'BINDPACK_PROVIDER_REJECTION_POLICY':'abstain'}):
            result=_content_rejection_metadata(llm,rejected)
            self.assertTrue(result['non_retryable'])
            self.assertIsNone(result['prompt_tokens'])
            self.assertIsNone(_content_rejection_metadata(llm,invalid))
            llm.llm_base_url='http://127.0.0.1:8035/v1'
            self.assertIsNone(_content_rejection_metadata(llm,rejected))
        with patch.dict('os.environ',{'BINDPACK_PROVIDER_REJECTION_POLICY':'stop'}):
            self.assertIsNone(_content_rejection_metadata(llm,rejected))

    def test_terminal_rejection_is_one_physical_failure_not_zero_token_success(self):
        endpoint='https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1'
        response=httpx.Response(400,request=httpx.Request('POST',endpoint))
        exception=openai.BadRequestError('content rejected',response=response,
            body={'error':{'code':'data_inspection_failed'}})
        with tempfile.TemporaryDirectory() as directory,patch.dict('os.environ',
                {'OPENAI_API_KEY':'test-not-a-real-key','BINDPACK_PROVIDER_REJECTION_POLICY':'abstain'}):
            llm=CacheOpenAI(cache_dir=directory,global_config=RuntimeConfig(
                llm_name='qwen3.8-flash',llm_base_url=endpoint))
            native=Mock(side_effect=exception)
            llm.openai_client.chat.completions.create=native
            recorder=Recorder(directory);recorder.install_llm(llm,None)
            with recorder.section('offline_index','one-passage'):
                first=llm.infer([{'role':'user','content':'unchanged original source'}])
                cached=llm.infer([{'role':'user','content':'unchanged original source'}])
            self.assertEqual(first[0],'');self.assertTrue(first[1]['provider_rejected'])
            self.assertTrue(cached[2]);self.assertEqual(native.call_count,1)
            physical=jsonl(Path(directory)/'llm_calls.jsonl')
            self.assertEqual(usage(physical)['failed_attempts'],1)
            self.assertEqual(usage(physical)['attempts_with_unknown_usage'],1)
            self.assertIsNone(physical[0]['prompt_tokens'])

    def test_rejected_extraction_never_splits_source_or_retries_queue(self):
        llm=self.llm('https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1')
        metadata=dict(error='provider_content_rejected',provider_rejected=True,
            non_retryable=True,finish_reason='content_filter',prompt_tokens=None,completion_tokens=None)
        llm.infer=Mock(return_value=('',metadata,False))
        extractor=StructuralOpenIE(llm_model=llm,max_workers=1)
        with patch('bindpack.index.openie.openie_semantic_validation._prompt_tokens',return_value=10):
            ner,triples=run_openie_queue(extractor,{'one':{'content':'Original unchanged source.'}})
            self.assertEqual(llm.infer.call_count,1)
            self.assertTrue(ner['one'].metadata['non_retryable'])
            self.assertFalse(ner['one'].metadata['complete'])
            self.assertEqual(triples['one'].triples,[])
            llm.infer.reset_mock()
            result=extractor.triple_extraction('two','Another unchanged source.',['Another'])
            self.assertEqual(llm.infer.call_count,1)
            self.assertTrue(result.metadata['non_retryable'])
            self.assertIs(extractor.recover_pending_triples('two','Another unchanged source.',[],result,2),result)
            self.assertEqual(llm.infer.call_count,1)


if __name__=='__main__':unittest.main()

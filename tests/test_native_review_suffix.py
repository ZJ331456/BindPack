import json
from pathlib import Path
import unittest

from bindpack import BindPackConfig
from bindpack.stages.pack.verification import support_semantic_veto as native
from test_evidence_support import fixture


class NativeSuffixReviewTests(unittest.TestCase):
    def runtime(self,semantic=True):
        runtime,solution,llm,docs=fixture(semantic=semantic)
        runtime.cfg.pack.review_mode='suffix_stop'
        original=llm.infer
        def infer(messages,stop,**kwargs):
            raw,metadata=original(messages,**kwargs)
            hits=[raw.find(value) for value in stop if raw.find(value)>=0]
            return raw[:min(hits)] if hits else raw,metadata
        llm.infer=infer
        return runtime,solution,llm,docs

    def test_suffix_config_is_explicit_and_preserves_weights(self):
        root=Path(__file__).resolve().parents[1]
        default=BindPackConfig()
        cfg=BindPackConfig.from_file(root/'configs/pack_review_suffix.json')
        self.assertEqual(default.pack.review_mode,'off')
        self.assertEqual(cfg.pack.review_mode,'suffix_stop')
        self.assertEqual(cfg.pack.relation_weight,.6)
        self.assertEqual(cfg.pack.budget,5)
        with self.assertRaisesRegex(ValueError,'pack.review_mode'):
            BindPackConfig.from_file(pack={'review_mode':'unknown'})

    def test_native_engine_accepts_complete_source_and_decision_prefix(self):
        runtime,solution,llm,docs=self.runtime()
        stats=native.postprocess_support_semantic_veto(runtime,[(solution,list(range(20)))])
        trace=solution.retrieval_trace['evidence']['improvement_support_semantic_veto']
        self.assertEqual(stats['applied'],1)
        self.assertTrue(trace['accepted']);self.assertEqual(len(llm.calls),2)
        self.assertTrue(trace['top2_preserved']);self.assertTrue(trace['protected_support_preserved'])
        out=trace['llm_stages']['proposer']['output']
        self.assertTrue(out['json_closure_reconstructed'])
        with self.assertRaises(json.JSONDecodeError):json.loads(out['raw_response'])
        self.assertIsNone(trace['base_proposal']['promotions'][0]['reason'])

    def test_native_negative_decision_keeps_original_context(self):
        runtime,solution,llm,docs=self.runtime(semantic=False);before=list(solution.docs)
        native.postprocess_support_semantic_veto(runtime,[(solution,list(range(20)))])
        trace=solution.retrieval_trace['evidence']['improvement_support_semantic_veto']
        self.assertFalse(trace['accepted']);self.assertEqual(solution.docs,before)

    def test_empty_candidate_quote_is_not_filled_or_retried(self):
        runtime,solution,llm,docs=self.runtime();before=list(solution.docs)
        def infer(messages,stop,**kwargs):
            llm.calls.append(messages)
            return '{"swap":{"promote":"D5","replace":"D4","root":"D0","confidence":0.99,"missing_condition":"A real missing birthplace",', {'finish_reason':'stop'}
        llm.infer=infer
        native.postprocess_support_semantic_veto(runtime,[(solution,list(range(20)))])
        self.assertEqual(solution.docs,before);self.assertEqual(len(llm.calls),1)
        trace=solution.retrieval_trace['evidence']['improvement_support_semantic_veto']
        self.assertFalse(trace['accepted'])


if __name__=='__main__':unittest.main()

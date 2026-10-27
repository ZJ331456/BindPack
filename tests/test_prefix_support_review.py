import json
import unittest
import prefix_support_review as prefix
from test_evidence_support import fixture


class PrefixReviewTests(unittest.TestCase):
    def test_exact_prefix_closure_reconstruction(self):
        self.assertEqual(prefix.decode_prefix('{"swap":{"root":"D1", ','proposer'),{'swap':{'root':'D1'}})
        self.assertEqual(prefix.decode_prefix('{"root":"D1", ','reviewer'),{'root':'D1'})
        self.assertEqual(prefix.decode_prefix('{"swap":null}','proposer'),{'swap':None})
        self.assertIsNone(prefix.decode_prefix('{"swap":{"root":"unclosed','proposer'))

    def runtime(self,semantic=True):
        runtime,solution,llm,docs=fixture(semantic=semantic);original=llm.infer
        def infer(messages,stop,**kwargs):
            raw,meta=original(messages,**kwargs)
            positions=[raw.find(s) for s in stop if raw.find(s)>=0]
            if positions:raw=raw[:min(positions)]
            return raw,meta,False
        llm.infer=infer
        return runtime,solution,llm,docs

    def test_real_source_checks_and_decisions_survive_suffix_stop(self):
        runtime,solution,llm,docs=self.runtime()
        d=prefix.run(runtime,solution,list(range(20)))
        self.assertTrue(d['accepted']);self.assertEqual(d['extra_requests'],2)
        self.assertFalse(d['base_proposal']['reason_suffix_generated'])
        self.assertTrue(d['llm_stages']['proposer']['output']['json_closure_reconstructed'])
        self.assertTrue(d['top2_preserved']);self.assertTrue(d['protected_support_preserved'])

    def test_negative_review_still_rejects(self):
        runtime,solution,llm,docs=self.runtime(semantic=False);before=list(solution.docs)
        d=prefix.run(runtime,solution,list(range(20)))
        self.assertFalse(d['accepted']);self.assertEqual(solution.docs,before)

    def test_early_incomplete_negative_prefix_never_approves(self):
        runtime,solution,llm,docs=self.runtime();order=list(range(20));state=prefix._state(solution.retrieval_trace['evidence'])
        proposal={'promotions':[{'doc_id':5,'victim':4,'root_doc_id':0}]}
        output={'raw_response':'{"root":"D0","candidate":"D5","victim":"D4",', 'metadata':{'finish_reason':'stop'}}
        okay,diag=prefix.check_review(order,state,proposal,output)
        self.assertFalse(okay)

    def test_stop_never_supplies_missing_source_or_missing_fact_fields(self):
        runtime,solution,llm,docs=self.runtime();order=list(range(20));state=prefix._state(solution.retrieval_trace['evidence'])
        proposed,diag=prefix.validate_source_swap(solution.question,order,state,docs.__getitem__,
            '{"swap":{"promote":"D5","replace":"D4","root":"D0","confidence":0.99}}')
        self.assertEqual(proposed,order);self.assertFalse(diag['promotions'])


if __name__=='__main__':unittest.main()

import json
import unittest
import tight_support_review as tight
from test_evidence_support import fixture


class TightReviewTests(unittest.TestCase):
    def test_proposal_does_not_require_unused_generator_explanations(self):
        runtime,solution,llm,docs=fixture()
        state=tight._state(solution.retrieval_trace['evidence']);order=list(range(20))
        _,guard=tight.support_guard(solution.question,order,state,docs.__getitem__,None)
        prompt=tight.prompt(solution.question,order,state,docs.__getitem__,guard)
        self.assertNotIn('"missing_condition":',prompt[0]['content'])
        self.assertNotIn('"reason":',prompt[0]['content'])
        raw,meta=llm.infer(prompt,response_format={'type':'json_object'})
        obj=json.loads(raw);obj['swap'].pop('reason');obj['swap'].pop('missing_condition')
        final,diag=tight.validate(solution.question,order,state,docs.__getitem__,json.dumps(obj))
        self.assertNotEqual(final,order);self.assertTrue(diag['promotions'])

    def test_invalid_quote_confidence_and_scope_keep_parent(self):
        runtime,solution,llm,docs=fixture();state=tight._state(solution.retrieval_trace['evidence']);order=list(range(20))
        _,guard=tight.support_guard(solution.question,order,state,docs.__getitem__,None)
        prompt=tight.prompt(solution.question,order,state,docs.__getitem__,guard)
        raw,_=llm.infer(prompt,response_format={'type':'json_object'});base=json.loads(raw)
        for field,value in [('confidence',.2),('promote','D0'),('replace','D1'),('candidate_quote','An invented unsupported fact.')]:
            obj=json.loads(json.dumps(base));obj['swap'][field]=value
            final,diag=tight.validate(solution.question,order,state,docs.__getitem__,json.dumps(obj))
            self.assertEqual(final,order);self.assertFalse(diag['promotions'])

    def test_full_engine_keeps_native_independent_audit(self):
        runtime,solution,llm,docs=fixture();original=llm.infer
        def infer(messages,**kwargs):
            answer,meta=original(messages,**kwargs);return answer,meta,False
        llm.infer=infer
        d=tight.run(runtime,solution,list(range(20)))
        self.assertTrue(d['accepted']);self.assertEqual(d['extra_requests'],2)
        reviewer=llm.calls[-1][0]['content']
        self.assertIn('CURRENT_TOP_FIVE',reviewer)
        self.assertIn('root_quote, candidate_quote',reviewer)

    def test_prompt_uses_real_support_guard_protection(self):
        runtime,solution,llm,docs=fixture(protect_victim=True)
        state=tight._state(solution.retrieval_trace['evidence']);order=list(range(20))
        _,guard=tight.support_guard(solution.question,order,state,docs.__getitem__,None)
        p=tight.prompt(solution.question,order,state,docs.__getitem__,guard)
        data=json.loads(p[1]['content']);self.assertNotIn('D4',data['allowed_replace'])
        self.assertTrue(data['passages'][4]['protected'])


if __name__=='__main__':unittest.main()

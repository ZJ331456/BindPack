import json
import unittest

import compact_support_review as compact
from test_evidence_support import FakeLLM, fixture


class CompactLLM(FakeLLM):
    def __init__(self, semantic=True, bad_span=False):
        super().__init__(semantic)
        self.bad_span=bad_span

    def infer(self,messages,response_format):
        response,metadata=super().infer(messages,response_format)
        value=json.loads(response)
        if 'swap' in value:
            swap=value['swap']
            swap.update(promote='C1',replace='S5',root='S1',root_span=[1,1],
                candidate_span=[999,999] if self.bad_span else [1,1])
            swap.pop('root_quote');swap.pop('candidate_quote')
        return json.dumps(value),metadata,False


class CompactReviewTests(unittest.TestCase):
    def build(self,**kwargs):
        runtime,solution,_,documents=fixture(**kwargs)
        runtime.rag.llm_model=CompactLLM(kwargs.get('semantic',True))
        return runtime,solution,documents

    def test_partition_and_quote_preserve_visible_source(self):
        source='Topic\nFirst fact.  Second fact?\nThird fact.'
        parts=compact.segments(source)
        self.assertEqual(''.join(p[2] for p in parts),source)
        self.assertEqual(compact.quote(parts,[1,2]),'First fact.  Second fact?')
        self.assertEqual(''.join(p[2] for p in compact.segments('a'*2000)),'a'*1400)

    def test_invalid_or_noncontiguous_boundaries_are_rejected(self):
        parts=compact.segments('Topic\nA real fact.')
        for span in ([True,1],[-1,0],[1,0],[1,99],None,[1]):
            with self.subTest(span=span),self.assertRaises(ValueError):compact.quote(parts,span)

    def test_winning_source_is_protected_in_actual_prompt(self):
        runtime,solution,documents=self.build(protect_victim=True)
        state=compact._state(solution.retrieval_trace['evidence'])
        _,guard=compact.support_guard(solution.question,list(range(20)),state,documents.__getitem__,None)
        messages,_,_,_=compact.build_prompt(solution.question,list(range(20)),state,documents.__getitem__,guard)
        payload=json.loads(messages[1]['content'])
        self.assertTrue(payload['selected'][4]['protected'])
        self.assertNotIn('S5',payload['allowed_victims'])

    def test_approved_swap_retains_all_native_invariants(self):
        runtime,solution,_=self.build()
        original=list(solution.docs)
        d=compact.run(runtime,solution,list(range(20)))
        self.assertTrue(d['accepted']);self.assertEqual(len(runtime.rag.llm_model.calls),2)
        self.assertEqual(solution.docs[:2],original[:2]);self.assertEqual(solution.docs[10:],original[10:])
        self.assertTrue(d['protected_support_preserved'])

    def test_reviewer_can_reject_without_changing_ranking(self):
        runtime,solution,_=self.build(semantic=False)
        original=list(solution.docs)
        d=compact.run(runtime,solution,list(range(20)))
        self.assertFalse(d['accepted']);self.assertEqual(solution.docs,original)

    def test_source_only_arm_uses_one_call(self):
        runtime,solution,_=self.build()
        d=compact.run(runtime,solution,list(range(20)),reviewer=False)
        self.assertTrue(d['accepted']);self.assertEqual(d['extra_requests'],1)
        self.assertFalse(d['independent_verification']['independent_review_requested'])

    def test_invalid_span_skips_without_retry(self):
        runtime,solution,_=self.build()
        runtime.rag.llm_model=CompactLLM(bad_span=True)
        original=list(solution.docs)
        d=compact.run(runtime,solution,list(range(20)))
        self.assertFalse(d['accepted']);self.assertEqual(solution.docs,original)
        self.assertEqual(d['extra_requests'],1)

    def test_native_type_scope_still_skips_request(self):
        runtime,solution,_=self.build(question='How many directors worked on Aster Film?')
        d=compact.run(runtime,solution,list(range(20)))
        self.assertFalse(d['accepted']);self.assertEqual(runtime.rag.llm_model.calls,[])


if __name__=='__main__':unittest.main()

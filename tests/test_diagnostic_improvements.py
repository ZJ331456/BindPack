"""Invariants of isolated diagnostic changes; no model requests."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from diagnostic_improvements import (fact_key,FactCoverageSelector,recover_plan,
    positive_path_fusion,complete_chain_package)
from bindpack.stages.pack.evidence_scorer import EvidenceScorer
from bindpack.stages.pack.coverage_selector import CoverageSelector


class CandidateTests(unittest.TestCase):
    def test_fact_identity_preserves_requested_relation_and_context(self):
        self.assertEqual(fact_key('Who directed Work X?',{}),fact_key('Who was the director of Work X?',{}))
        self.assertNotEqual(fact_key('When was A born?',{}),fact_key('Where was A born?',{}))
        self.assertNotEqual(fact_key('When was A born?',{}),fact_key('When did A die?',{}))
        self.assertNotEqual(fact_key('Who wrote X?',{'s1':'A'}),fact_key('Who wrote X?',{'s1':'B'}))

    def test_plan_recovery_never_invents_missing_references_or_accepts_cycle(self):
        cfg=SimpleNamespace(node_budget=6,depth_budget=4)
        good=dict(nodes=[dict(id='s1',question='Who wrote X?',depends_on=[],answer_type='person'),
                         dict(id='s2',question='Where was ${s1.answer} born?',depends_on=[],answer_type='place')])
        nodes,diag=recover_plan('Where was the author of X born?',2,dict(outputs=[good]),cfg)
        self.assertTrue(diag['recovered']);self.assertEqual(nodes[1]['depends_on'],['s1'])
        self.assertEqual(nodes[1]['question'],good['nodes'][1]['question'])
        missing=deepcopy(good);missing['nodes'][1].update(question='Where was the author born?',depends_on=['s1'])
        self.assertFalse(recover_plan('Q',2,dict(outputs=[missing]),cfg)[1]['recovered'])
        cycle=deepcopy(good);cycle['nodes'][0].update(question='Who wrote ${s2.answer}?',depends_on=['s2'])
        self.assertFalse(recover_plan('Q',2,dict(outputs=[cycle]),cfg)[1]['recovered'])

    def test_positive_path_fusion_preserves_unselected_and_removes_penalty(self):
        # Base normalization is [1,.5,0]; path fusion lowered d0 and raised d1.
        ids,scores,diag=positive_path_fusion(np.array([0,1,2]),np.array([2.,1.,0.]),
            np.array([0,1,2]),np.array([.7,.65,0.],dtype=np.float32),3)
        by_id=dict(zip(ids,scores))
        self.assertEqual(float(by_id[0]),1.0)
        self.assertAlmostEqual(float(by_id[1]),.65,places=6)
        self.assertEqual(float(by_id[2]),0)
        self.assertEqual(diag['restored_penalized_passages'],1)

    def test_coverage_variant_keeps_pointwise_ablation_exact(self):
        pack=SimpleNamespace(base_weight=.4,relation_weight=.6,coverage_weight=.45,redundancy_weight=.15,budget=5)
        runtime=SimpleNamespace(cfg=SimpleNamespace(pack=pack,stage_ablation='wo_pack'),
            _trace=lambda:{},_document=lambda i:f'Doc {i}\nBody')
        state=dict(evidence_trace={},_evidence_winning_bindings={},evidence_candidates={
            i:dict(base_score=0,sources=[dict(goal=f'g{i}',requirements={},question=f'Q{i}',score=.9)],verified=[])
            for i in range(6)})
        a=CoverageSelector(runtime,EvidenceScorer()).select('Q',np.arange(6),np.arange(6)[::-1],{},deepcopy(state))
        b=FactCoverageSelector(runtime,EvidenceScorer()).select('Q',np.arange(6),np.arange(6)[::-1],{},deepcopy(state))
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])

    def test_incomplete_or_ungrounded_chain_cannot_promote(self):
        result=(np.arange(6),np.arange(6)[::-1],{})
        runtime=SimpleNamespace(_document=lambda i:'X\nA literal document.')
        state=dict(_evidence_plan=[dict(id='s1'),dict(id='s2')],_evidence_winning_bindings={})
        output=complete_chain_package(runtime,result,state,'Q',np.arange(6),np.arange(6))
        np.testing.assert_array_equal(output[0],result[0])
        self.assertFalse(output[2]['complete_chain_package']['applied'])

    def test_complete_chain_is_bounded_and_requires_real_source_quotes(self):
        from bindpack.stages.pack.evidence_scorer import EvidenceScorer
        docs={i:f'Document {i}\nUnrelated background.' for i in range(6)}
        docs[4]='Person A\nPerson A is a writer.'
        docs[5]='Town B\nPerson A was born in Town B.'
        plan=[dict(id='s1',question='Who wrote X?',depends_on=[],answer_type='unknown'),
              dict(id='s2',question='Where was ${s1.answer} born?',depends_on=['s1'],answer_type='unknown')]
        proofs={'s1':dict(doc_id=4,confidence=.9,evidence='Person A is a writer.',answer='Person A',quality=.9),
                's2':dict(doc_id=5,confidence=.9,evidence='Person A was born in Town B.',answer='Town B',quality=.9)}
        bindings={'s1':'Person A','s2':'Town B'}
        candidates={i:dict(base_score=0,sources=[],verified=[]) for i in range(6)}
        for k,p in proofs.items():candidates[p['doc_id']]['verified']=[dict(p,goal='dag:'+k)]
        state=dict(_evidence_plan=plan,_evidence_winning_bindings=bindings,evidence_candidates=candidates,
                   _evidence_beams=[dict(bindings=bindings,proofs=proofs)])
        runtime=SimpleNamespace(_document=lambda i:docs[i],cfg=SimpleNamespace(pack=SimpleNamespace(
            base_weight=.4,relation_weight=.6,coverage_weight=.45)),pack=SimpleNamespace(scorer=EvidenceScorer()))
        result=(np.arange(6),np.arange(6)[::-1],{})
        out=complete_chain_package(runtime,result,state,'Original question',np.arange(6),np.arange(6)[::-1])
        self.assertTrue(out[2]['complete_chain_package']['applied'])
        self.assertTrue({4,5}<=set(out[0][:5]));self.assertEqual(len(set(out[0])),6)
        bad=deepcopy(state);bad['_evidence_beams'][0]['proofs']['s2']['evidence']='An invented quotation.'
        out=complete_chain_package(runtime,result,bad,'Original question',np.arange(6),np.arange(6)[::-1])
        self.assertFalse(out[2]['complete_chain_package']['applied'])


if __name__=='__main__':unittest.main()

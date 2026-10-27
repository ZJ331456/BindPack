"""Keep complementary evidence despite shared wording; retain pointwise ablation."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
import numpy as np

from bindpack.stages.pack.coverage_selector import CoverageSelector
from bindpack.stages.pack.evidence_scorer import EvidenceScorer


def fixture(second_goal='birthplace', verified=False, ablation='full'):
    config=SimpleNamespace(base_weight=.4,relation_weight=.6,coverage_weight=.45,redundancy_weight=.15,budget=5)
    documents={0:'Alice Example\nAlice Example shared entity name and biographical evidence.',
               1:'Alice Example\nAlice Example shared entity name and birthplace evidence.'}
    candidates={0:dict(base_score=1.,sources=[dict(goal='director',score=1.,requirements={})],verified=[]),
                1:dict(base_score=0.,sources=[dict(goal=second_goal,score=1.,requirements={})],verified=[])}
    if verified:
        candidates[1]['sources'][0]['goal']='director'
        candidates[1]['verified']=[dict(goal='birthplace',quality=.9)]
    runtime=SimpleNamespace(cfg=SimpleNamespace(pack=config,stage_ablation=ablation),
        _document=documents.__getitem__,_trace=lambda:{})
    state=dict(evidence_candidates=candidates,evidence_trace={},_evidence_winning_bindings={})
    return runtime,state


class GoalRedundancyTests(unittest.TestCase):
    def test_shared_entity_different_goal_keeps_complementary_evidence_unpenalized(self):
        runtime,state=fixture();before=deepcopy(state['evidence_candidates'])
        ids,_,trace=CoverageSelector(runtime,EvidenceScorer()).select('Q',np.arange(2),np.array([1.,0.]),{},state)
        self.assertEqual(ids.tolist(),[0,1])
        self.assertEqual(trace['selected_prefix'][1]['redundancy'],0.)
        self.assertEqual(before,state['evidence_candidates'])

    def test_same_goal_lexical_duplicate_still_has_penalty(self):
        runtime,state=fixture(second_goal='director')
        _,_,trace=CoverageSelector(runtime,EvidenceScorer()).select('Q',np.arange(2),np.array([1.,0.]),{},state)
        self.assertGreater(trace['selected_prefix'][1]['redundancy'],0.)

    def test_new_verified_goal_is_exempt_even_when_retrieval_goal_overlaps(self):
        runtime,state=fixture(verified=True)
        _,_,trace=CoverageSelector(runtime,EvidenceScorer()).select('Q',np.arange(2),np.array([1.,0.]),{},state)
        self.assertEqual(trace['selected_prefix'][0]['doc_id'],0)
        self.assertEqual(trace['selected_prefix'][1]['redundancy'],0.)
        self.assertAlmostEqual(trace['selected_prefix'][1]['utility'],.6+.45*.9/2)

    def test_wo_pack_preserves_original_pointwise_scores(self):
        runtime,state=fixture(verified=True,ablation='wo_pack')
        ids,scores,trace=CoverageSelector(runtime,EvidenceScorer()).select('Q',np.arange(2),np.array([1.,0.]),{},state)
        self.assertEqual(ids.tolist(),[0,1]);np.testing.assert_allclose(scores,[1.,.6])
        self.assertEqual(trace['pack_mode'],'pointwise_base_goal')


if __name__=='__main__':unittest.main()

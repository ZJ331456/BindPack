import copy
import unittest
from types import SimpleNamespace
import numpy as np

from bindpack import BindPackConfig
from bindpack.stages.pack.budgeted_selector import BudgetedSelector, closed_nodes, residual_features
from bindpack.stages.pack.evidence_scorer import EvidenceScorer
from test_evidence_dag_package import Engine, chain


class BudgetedTests(unittest.TestCase):
    def runtime(self, engine, mode='closed', visible=None):
        cfg = BindPackConfig.from_file(pack={'selector_mode':mode,'joint_search_seconds':1.0,
                                           'joint_fixed_prefix':2,'joint_state_cap':2048})
        return SimpleNamespace(cfg=cfg,_document=engine._document,
            _joint_visible=visible or (lambda q, ids: ({d:engine._document(d) for d in ids},{})))

    def test_joint_chain_and_immutable_input(self):
        e=Engine();s=chain(e)
        for c in s['evidence_candidates'].values():
            c['sources']=[];c['base_score']=0
            for p in c['verified']:p['quality']=.95
        old=copy.deepcopy(s);ids=np.arange(220);scores=np.linspace(1,0,220)
        result=BudgetedSelector(self.runtime(e),EvidenceScorer()).select(
            'Where is the county containing Sample Town?',ids,scores,(ids,scores,s['evidence_trace']),s)
        self.assertEqual(s,old)
        self.assertEqual(result[0][:2].tolist(),[0,1])
        self.assertTrue({0,2,5,15}<=set(result[0][:5]))
        self.assertEqual(set(result[0][:200]),set(range(200)))
        self.assertEqual(result[2]['budgeted_selection']['model_requests'],0)

    def test_and_ancestors_or_alternative_sources(self):
        choices={'a':[{'doc_id':1,'evidence':'a'}],
                 'b':[{'doc_id':2,'evidence':'b'},{'doc_id':3,'evidence':'c'}]}
        ancestors={'a':set(),'b':{'a'}}
        self.assertEqual(closed_nodes({2},choices,ancestors,['a','b']),set())
        self.assertEqual(closed_nodes({1,3},choices,ancestors,['a','b']),{'a','b'})
        self.assertEqual(closed_nodes({1,3},choices,ancestors,['a','b'],{1:'a',3:'hidden'}),{'a'})

    def test_invisible_proof_has_no_final_closure_credit(self):
        e=Engine();s=chain(e)
        for c in s['evidence_candidates'].values():
            c['sources']=[];c['base_score']=0
            for p in c['verified']:p['quality']=.95
        ids=np.arange(220);scores=np.linspace(1,0,220)
        runtime=self.runtime(e,visible=lambda q,ids:({d:'Title\nTruncated content.' for d in ids},{}))
        result=BudgetedSelector(runtime,EvidenceScorer()).select('A query',ids,scores,(ids,scores,s['evidence_trace']),s)
        self.assertEqual(result[2]['budgeted_selection']['closed_nodes'],[])

    def test_soft_facets_never_join_different_sentences(self):
        f,m,w=residual_features('bluegrass singer',['A bluegrass band. A singer.', 'A bluegrass singer.'])
        self.assertEqual(m[0,f.index(('bluegrass','singer'))],0)
        self.assertEqual(m[1,f.index(('bluegrass','singer'))],1)

    def test_missing_proof_uses_soft_fallback_without_fabricating_closure(self):
        e=Engine();e.documents[5]='Music\nA bluegrass singer performs here.'
        state={'evidence_candidates':{d:{'sources':[],'verified':[],'base_score':0} for d in range(8)},
            '_evidence_plan':[],'_evidence_winning_bindings':{},'_evidence_beams':[],
            'evidence_trace':{'selected_prefix':[{'doc_id':d} for d in range(5)]}}
        ids=np.arange(8);scores=np.linspace(1,0,8)
        result=BudgetedSelector(self.runtime(e,'residual'),EvidenceScorer()).select(
            'Which bluegrass singer?',ids,scores,(ids,scores,state['evidence_trace']),state)
        self.assertIn(5,result[0][:5])
        diag=result[2]['budgeted_selection']
        self.assertEqual(diag['closed_nodes'],[])
        self.assertEqual(diag['soft_weight_effective'],1)
        self.assertFalse(diag['entailment_guaranteed'])

    def test_experimental_selector_requires_review_off(self):
        with self.assertRaises(ValueError):
            BindPackConfig.from_file(pack={'selector_mode':'combined','review_mode':'standard'})

    def test_source_collection_keeps_same_answer_alternatives_without_changing_branch(self):
        from test_stage_regression import FrozenRAG
        from bindpack.pipeline import EvidencePipeline
        def execute(enabled):
            rag=FrozenRAG();responses=list(rag.responses)
            rag.documents['d2']='Biography\nAlpha wrote Work X in 1990.'
            responses[1]['hypotheses'].append({'answer':'Alpha','doc_id':'D2',
                'evidence':'Alpha wrote Work X in 1990.','confidence':.8})
            rag.responses=iter(responses)
            runtime=EvidencePipeline(rag);runtime.capture_source_alternatives=enabled
            state={'query':'Where was the writer of Work X born?',
                'base':(np.arange(4),np.array([.9,.8,.7,.6]),{'hops':2}),
                'static_sub_questions':[],'pcqd_sub_questions':[]}
            runtime.process_window([state])
            return rag,state
        plain,a=execute(False);recorded,b=execute(True)
        self.assertEqual(plain.calls,recorded.calls)
        self.assertEqual(a['_evidence_winning_bindings'],b['_evidence_winning_bindings'])
        self.assertEqual(a['_evidence_beams'],b['_evidence_beams'])
        self.assertNotIn('source_records',a['evidence_trace'])
        self.assertEqual({r['doc_id'] for r in b['evidence_trace']['source_records'] if r['goal']=='dag:s1'},{0,2})

    def test_free_prefix_can_fit_chain_blocked_by_two_unrelated_anchors(self):
        from test_evidence_dag_package import state_for
        e=Engine();original=chain(e);texts=copy.deepcopy(e.documents)
        mapping={0:2,2:3,5:4,15:5}
        e.documents={d:f'Other {d}\nAn unrelated article.' for d in range(8)}
        proofs=[]
        for nid,p in original['_evidence_beams'][0]['proofs'].items():
            d=mapping[p['doc_id']];e.documents[d]=texts[p['doc_id']]
            proofs.append((nid,d,p['evidence']))
        s=state_for(e,original['_evidence_plan'],original['_evidence_winning_bindings'],proofs)
        for c in s['evidence_candidates'].values():
            c['sources']=[];c['base_score']=0
            for p in c['verified']:p['quality']=.95
        ids=np.arange(8);scores=np.linspace(1,0,8)
        locked=self.runtime(e);free=self.runtime(e)
        free.cfg.pack.joint_fixed_prefix=0;free.cfg.pack.joint_state_cap=4096
        a=BudgetedSelector(locked,EvidenceScorer()).select('Sample Region',ids,scores,(ids,scores,s['evidence_trace']),s)
        b=BudgetedSelector(free,EvidenceScorer()).select('Sample Region',ids,scores,(ids,scores,s['evidence_trace']),s)
        self.assertLess(len(a[2]['budgeted_selection']['closed_nodes']),4)
        self.assertEqual(len(b[2]['budgeted_selection']['closed_nodes']),4)
        self.assertTrue({2,3,4,5}<=set(b[0][:5]))
        self.assertFalse(b[2]['budgeted_selection']['top2_preserved'])

if __name__=='__main__':unittest.main()

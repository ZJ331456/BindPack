"""Intervention tests: disabled code paths must never execute."""
from copy import deepcopy
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
import numpy as np

from bindpack import BindPackConfig
from bindpack.stages.plan.stage import PlanStage
from bindpack.stages.plan.independent import IndependentPlanner,IndependentExecutor
from bindpack.stages.plan.source_verifier import SourceVerifier,extract_unverified_hypotheses,verify_hypotheses
from bindpack.stages.pack.stage import PackStage
from bindpack.stages.pack.evidence_scorer import EvidenceScorer
from bindpack.stages.pack.dag_package_selector import _collect_proofs
from test_evidence_dag_package import Engine,chain


class CompleteMechanismTests(TestCase):
    def test_new_full_and_legacy_unqualified_arm_have_identical_mechanisms(self):
        from bindpack.stages.mechanisms import mechanisms
        new=BindPackConfig.from_file('configs/full_no_source_eligibility.json')
        old=BindPackConfig.from_file(mechanism_ablation='wo_source_bound_eligibility')
        self.assertEqual(mechanisms(new),mechanisms(old))
        self.assertEqual(new.pack,old.pack)
        with self.assertRaises(ValueError):
            BindPackConfig.from_file(source_bound_eligibility='off')

    def test_both_new_ablations_keep_source_verification_disabled(self):
        from bindpack.stages.mechanisms import mechanisms
        for arm in ('full','wo_dependency_closure','wo_joint_selection'):
            with self.subTest(arm=arm):
                cfg=BindPackConfig.from_file(mechanism_ablation=arm,source_bound_eligibility=False)
                flags=mechanisms(cfg)
                self.assertFalse(flags.source_bound_eligibility)
                self.assertEqual(flags.dependency_closure,arm!='wo_dependency_closure')
                self.assertEqual(flags.joint_selection,arm!='wo_joint_selection')
                runtime=SimpleNamespace(cfg=cfg,_verification_diagnostics={},
                    _infer_object=lambda messages:({'hypotheses':[{'answer':'Claim','doc_id':'D1','confidence':.5}]},None))
                with patch('bindpack.stages.plan.source_verifier.verify_hypotheses',side_effect=AssertionError('strict source gate leaked')):
                    accepted,_,_=SourceVerifier(runtime).verify('Q?','entity',{1:'Unrelated text.'})
                self.assertFalse(accepted[0]['source_verified'])

    def test_new_full_pack_ranking_matches_legacy_source_ablation(self):
        old=self.runtime('wo_source_bound_eligibility')
        state=self.state()
        new=SimpleNamespace(**vars(old))
        new.cfg=BindPackConfig.from_file(source_bound_eligibility=False)
        ids=np.arange(220);scores=np.linspace(1,0,220)
        a=PackStage(old).finalize('Sample Region',ids,scores,{},deepcopy(state))
        b=PackStage(new).finalize('Sample Region',ids,scores,{},deepcopy(state))
        self.assertEqual(a[0].tolist(),b[0].tolist())
        for key in ('reference_doc_ids','candidate_domain','objective','selected_doc_ids'):
            self.assertEqual(a[2]['budgeted_selection'][key],b[2]['budgeted_selection'][key])

    def test_combined_no_source_no_dependency_never_invokes_strict_gate_or_closure(self):
        runtime=self.runtime('wo_dependency_closure')
        runtime.cfg.source_bound_eligibility=False
        state=self.state()
        for node in state['_evidence_plan']:
            node.update(depends_on=[],question='Independent question about '+node['id'])
        forbidden=AssertionError('disabled mechanism leaked')
        with patch('bindpack.stages.pack.dag_package_selector.DAGPackageSelector.select',side_effect=forbidden),\
             patch('bindpack.stages.pack.budgeted_selector._collect_proofs',side_effect=forbidden),\
             patch('bindpack.stages.pack.budgeted_selector._ancestors',side_effect=forbidden):
            result=PackStage(runtime).finalize('Sample Region',np.arange(220),np.linspace(1,0,220),{},state)
        self.assertEqual(result[2]['budgeted_selection']['package_count'],0)

    def test_combined_no_source_no_joint_never_invokes_joint_solver(self):
        runtime=self.runtime('wo_joint_selection')
        runtime.cfg.source_bound_eligibility=False
        with patch('bindpack.stages.pack.budgeted_selector.BudgetedSelector.select',side_effect=AssertionError('joint leaked')):
            stage=PackStage(runtime)
            self.assertFalse(stage.mechanisms.source_bound_eligibility)
            result=stage.finalize('Sample Region',np.arange(220),np.linspace(1,0,220),{},self.state())
        self.assertNotIn('budgeted_selection',result[2])

    def runtime(self,arm):
        self.engine=Engine()
        return SimpleNamespace(cfg=BindPackConfig.from_file(mechanism_ablation=arm),
            _document=self.engine._document,_trace=lambda:{},
            _joint_visible=lambda q,ids:({d:self.engine._document(d) for d in ids},{}))

    def state(self):
        state=chain(self.engine)
        for c in state['evidence_candidates'].values():
            c.update(sources=[],base_score=0.)
            for p in c['verified']:p['quality']=.95
        return state

    def test_configuration_rejects_mixed_stage_and_unknown_mechanism(self):
        with self.assertRaises(ValueError):BindPackConfig.from_file(mechanism_ablation='unknown')
        with self.assertRaises(ValueError):BindPackConfig.from_file(mechanism_ablation='wo_dependency_closure',stage_ablation='wo_plan')

    def test_ordinary_extraction_has_no_literal_quote_gate_but_validates_ids(self):
        payload={'hypotheses':[{'answer':'Different Answer','doc_id':'D1','confidence':.2}]}
        docs={1:'A passage that never contains that answer.'}
        self.assertEqual(verify_hypotheses(payload,docs)[0],[])
        accepted,_=extract_unverified_hypotheses(payload,docs)
        self.assertEqual(accepted[0]['answer'],'Different Answer')
        self.assertFalse(accepted[0]['source_verified'])
        payload['hypotheses'][0]['doc_id']='D999'
        self.assertEqual(extract_unverified_hypotheses(payload,docs)[0],[])

    def test_plan_source_gate_is_bypassed_without_format_retry(self):
        calls=[]
        runtime=SimpleNamespace(cfg=BindPackConfig.from_file(mechanism_ablation='wo_source_bound_eligibility'),
            _verification_diagnostics={},_infer_object=lambda messages:(calls.append(messages) or
                {'hypotheses':[{'answer':'Claim','doc_id':'D1','confidence':.5}]},None))
        with patch('bindpack.stages.plan.source_verifier.verify_hypotheses',side_effect=AssertionError('strict gate leaked')):
            result=SourceVerifier(runtime).verify('Question?','entity',{1:'Some text.'})
        self.assertEqual(len(calls),1)
        self.assertEqual(result[0][0]['answer'],'Claim')
        self.assertEqual(next(iter(runtime._verification_diagnostics.values()))['mode'],'unverified_answer_extraction')

    def test_numeric_answers_keep_full_schema_conversion_and_answer_budget(self):
        docs={1:'A source passage.'}
        accepted,rejected=extract_unverified_hypotheses({'hypotheses':[
            {'answer':1966,'doc_id':1,'confidence':.9},
            {'answer':'x'*201,'doc_id':1,'confidence':.9}]},docs)
        self.assertEqual([p['answer'] for p in accepted],['1966'])
        self.assertEqual(len(rejected),1)

    def test_independent_planning_keeps_full_per_query_node_capacity(self):
        calls=[]
        runtime=SimpleNamespace(cfg=BindPackConfig.from_file(mechanism_ablation='wo_dependency_closure'),
            _plan_diagnostics={},_infer_object=lambda messages:(calls.append(messages) or {'questions':[
                {'question':'Where was the writer of Book born?','answer_type':'place'}]},None))
        nodes,reason=IndependentPlanner(runtime).plan('Where was the writer of Book born?',2)
        self.assertIsNone(reason)
        self.assertIn('Maximum requests: 2',calls[0][1]['content'])
        self.assertEqual(next(iter(runtime._plan_diagnostics.values()))['node_budget'],2)

    def test_source_gate_is_removed_from_collector_and_route_scorer(self):
        runtime=self.runtime('wo_source_bound_eligibility');state=self.state()
        state['_evidence_winning_bindings']['a']='Other binding'
        for proof in [state['_evidence_beams'][0]['proofs']['a'],state['evidence_candidates'][0]['verified'][0]]:
            proof['evidence']='No literal quote in this document'
            proof['requirements']={'different':'conflicting branch'}
        args=({n['id']:n for n in state['_evidence_plan']},state['_evidence_winning_bindings'],
              state['_evidence_beams'][0],state['evidence_candidates'],{d:d+1 for d in range(220)},runtime._document)
        with patch('bindpack.stages.pack.dag_package_selector._proof_relation',side_effect=AssertionError('relation gate leaked')):
            choices=_collect_proofs(*args,{'rejected_proofs':[]},include_citations=True,source_bound_eligibility=False)
        self.assertIn('a',choices)
        self.assertEqual(choices['a'][0]['check'],'eligibility_disabled_unverified')
        candidate={'sources':[{'goal':'g','score':.7,'requirements':{'a':'conflict'}}],'verified':[]}
        self.assertEqual(EvidenceScorer.goal_scores(candidate,{'a':'value'}),{})
        self.assertEqual(EvidenceScorer(False).goal_scores(candidate,{'a':'value'}),{'g':.7})

    def test_dependency_ablation_selects_independent_native_components(self):
        runtime=self.runtime('wo_dependency_closure')
        stage=PlanStage(runtime)
        self.assertIsInstance(stage.planner,IndependentPlanner)
        self.assertIsInstance(stage.executor,IndependentExecutor)

    def test_independent_planner_rejects_answer_references(self):
        runtime=SimpleNamespace(cfg=BindPackConfig.from_file(mechanism_ablation='wo_dependency_closure'),
            _plan_diagnostics={},_infer_object=lambda messages:({'questions':[
                {'question':'Where was ${s1.answer} born?','answer_type':'place'}]},None))
        nodes,reason=IndependentPlanner(runtime).plan('A question',2)
        self.assertEqual(nodes,[]);self.assertIsNotNone(reason)

    def test_independent_executor_does_not_propagate_an_earlier_answer(self):
        routes=[]
        def add(state,source,goal,question,requirements):
            routes.append((source,question,requirements));return [0]
        runtime=SimpleNamespace(cfg=BindPackConfig.from_file(mechanism_ablation='wo_dependency_closure'),
            rank_constant=10.,_add_route=add,_document=lambda d:'Answer A is here.',
            _verification_diagnostics={},_collect_jobs=lambda executor,jobs:[f(*a) for f,a in jobs],
            _candidate=lambda s,d:s.setdefault('evidence_candidates',{}).setdefault(d,{'verified':[]}))
        verifier=SimpleNamespace(verify=lambda *args:([{'answer':'Answer A','doc_id':0,
            'evidence':'Answer A is here.','confidence':.9}],[],None))
        state={'_evidence_plan':[{'id':'s1','question':'Who wrote Book?','depends_on':[],'answer_type':'person'},
            {'id':'s2','question':'Where was the writer of Book born?','depends_on':[],'answer_type':'place'}],
            'evidence_trace':{'llm_verification_calls':0,'rejected_hypotheses':[],'semantic_failures':[]}}
        IndependentExecutor(runtime,verifier).execute([state],None)
        self.assertEqual(routes[1][1],'Where was the writer of Book born?')
        self.assertTrue(all(not r[2] and r[0].startswith('flat:') for r in routes))
        self.assertTrue(all(not p['requirements'] for p in state['evidence_trace']['source_records']))

    def test_dependency_ablation_never_invokes_old_or_new_closure(self):
        runtime=self.runtime('wo_dependency_closure');state=self.state()
        for node in state['_evidence_plan']:
            node['depends_on']=[];node['question']='A standalone question about '+node['id']
        ids=np.arange(220);scores=np.linspace(1,0,220)
        forbidden=AssertionError('dependency closure leaked')
        with patch('bindpack.stages.pack.dag_package_selector.DAGPackageSelector.select',side_effect=forbidden),\
             patch('bindpack.stages.pack.budgeted_selector._ancestors',side_effect=forbidden),\
             patch('bindpack.stages.pack.budgeted_selector._collect_proofs',side_effect=forbidden),\
             patch('bindpack.stages.pack.budgeted_selector.closed_nodes',side_effect=forbidden),\
             patch('bindpack.stages.pack.budgeted_selector._package_sets',side_effect=forbidden):
            result=PackStage(runtime).finalize('Sample Region',ids,scores,{},state)
        diag=result[2]['budgeted_selection']
        self.assertEqual(diag['eligible_proofs'],{})
        self.assertEqual(diag['package_count'],0)
        self.assertEqual(diag['predicted_closure'],0)

    def test_joint_ablation_never_invokes_budgeted_search(self):
        runtime=self.runtime('wo_joint_selection');state=self.state()
        with patch('bindpack.stages.pack.budgeted_selector.BudgetedSelector.select',side_effect=AssertionError('joint leaked')):
            result=PackStage(runtime).finalize('Sample Region',np.arange(220),np.linspace(1,0,220),{},state)
        self.assertNotIn('budgeted_selection',result[2])
        self.assertIn('improvement_dag_package',result[2])

"""Matched-domain score/action interventions must preserve the Full controls."""
from copy import deepcopy
from types import SimpleNamespace
from unittest import TestCase
import numpy as np
from bindpack import BindPackConfig
from bindpack.stages.pack.budgeted_selector import BudgetedSelector,closed_nodes,local_supported_nodes
from bindpack.stages.pack.evidence_scorer import EvidenceScorer
from test_evidence_dag_package import Engine,chain


class JointComponentTests(TestCase):
    def run_arm(self,arm):
        engine=Engine();state=chain(engine)
        for candidate in state['evidence_candidates'].values():
            candidate.update(sources=[],base_score=0.)
            for proof in candidate['verified']:proof.update(quality=.95,source_verified=False)
        cfg=BindPackConfig.from_file(pack_mechanism_ablation=arm)
        runtime=SimpleNamespace(cfg=cfg,_document=engine._document,
            _joint_visible=lambda q,ids:({d:engine._document(d) for d in ids},{}))
        ids=np.arange(220);scores=np.linspace(1,0,220);original=deepcopy(state)
        result=BudgetedSelector(runtime,EvidenceScorer(False)).select('Sample Region',ids,scores,
            (ids,scores,state['evidence_trace']),state)
        self.assertEqual(state,original)
        self.assertEqual(set(result[0][:200]),set(ids[:200]))
        return result[2]['budgeted_selection']

    def test_scoring_controls_keep_full_domain_rho_and_capped_action_bank(self):
        baseline=self.run_arm('none')
        for arm in ('flat_dependency_support','wo_support_retention','wo_residual_coverage'):
            with self.subTest(arm=arm):
                variant=self.run_arm(arm)
                for key in ('candidate_domain','reference_doc_ids','soft_weight_effective','full_package_bank','package_action_count'):
                    self.assertEqual(variant[key],baseline[key],key)
                self.assertEqual(variant['reward_loss_weight'],0. if arm=='wo_support_retention' else .1)
                self.assertEqual(variant['residual_reward_enabled'],arm!='wo_residual_coverage')

    def test_action_control_removes_packages_without_removing_support_reward(self):
        base=self.run_arm('none');variant=self.run_arm('singleton_search')
        self.assertTrue(base['full_package_bank'])
        self.assertEqual(base['full_package_bank'],variant['full_package_bank'])
        self.assertEqual(variant['package_action_count'],0)
        self.assertEqual(variant['closure_reward_mode'],'ancestor_closed')
        self.assertEqual(variant['reward_loss_weight'],base['reward_loss_weight'])

    def test_flat_support_can_cover_child_without_ancestor(self):
        choices={'a':[{'doc_id':0,'evidence':'a'}],'b':[{'doc_id':1,'evidence':'b'}]}
        self.assertEqual(closed_nodes({1},choices,{'a':set(),'b':{'a'}},['a','b']),set())
        self.assertEqual(local_supported_nodes({1},choices,['a','b']),{'b'})

    def test_combined_end_to_end_and_scoring_interventions_are_rejected(self):
        with self.assertRaises(ValueError):
            BindPackConfig.from_file(mechanism_ablation='wo_dependency_closure',pack_mechanism_ablation='wo_residual_coverage')

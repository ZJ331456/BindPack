"""A fixed two-factor intervention and observational Pack instrumentation."""
from copy import deepcopy
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
import time
import numpy as np

from bindpack import BindPackConfig
from bindpack.stages.pack.budgeted_selector import BudgetedSelector
from bindpack.stages.pack.evidence_scorer import EvidenceScorer
from test_evidence_dag_package import Engine, chain


class ClosureInteractionTests(TestCase):
    def run_arm(self, arm='none', diagnostics=None, weights=None):
        engine = Engine()
        state = chain(engine)
        for candidate in state['evidence_candidates'].values():
            candidate.update(sources=[], base_score=0.)
            for proof in candidate['verified']:
                proof.update(quality=.95, source_verified=False)
        cfg = BindPackConfig.from_file(pack_mechanism_ablation=arm, pack=weights or {})
        runtime = SimpleNamespace(cfg=cfg, _document=engine._document,
            _joint_visible=lambda q, ids: ({d: engine._document(d) for d in ids}, {}))
        if diagnostics is not None:
            runtime._pack_diagnostics = diagnostics
        ids, scores = np.arange(220), np.linspace(1, 0, 220)
        before = deepcopy(state)
        result = BudgetedSelector(runtime, EvidenceScorer(False)).select(
            'Sample Region', ids, scores, (ids, scores, state['evidence_trace']), state)
        self.assertEqual(state, before)
        return result, runtime

    def test_combo_is_explicit_and_keeps_other_controls(self):
        full, _ = self.run_arm()
        combo, _ = self.run_arm('wo_closure_aware_packaging')
        a, b = full[2]['budgeted_selection'], combo[2]['budgeted_selection']
        for key in ('candidate_domain', 'reference_doc_ids', 'eligible_proofs', 'full_package_bank',
                    'soft_weight_effective', 'reward_loss_weight', 'residual_reward_enabled'):
            self.assertEqual(a[key], b[key], key)
        self.assertEqual(b['closure_reward_mode'], 'flat')
        self.assertEqual(b['package_action_count'], 0)
        self.assertTrue(b['full_package_bank'])
        with self.assertRaises(ValueError):
            BindPackConfig.from_file(pack_mechanism_ablation='wo_closure_aware_packaging',
                                    mechanism_ablation='wo_source_binding')
        with self.assertRaises(ValueError):
            BindPackConfig.from_file(pack_mechanism_ablation='flat_dependency_support+singleton_search')

    def test_observer_preserves_complete_rank_score_objective_domain_and_search_count(self):
        for arm in ('none', 'flat_dependency_support', 'singleton_search', 'wo_closure_aware_packaging'):
            with self.subTest(arm=arm):
                plain, _ = self.run_arm(arm)
                observed, runtime = self.run_arm(arm, {})
                np.testing.assert_array_equal(plain[0], observed[0])
                np.testing.assert_array_equal(plain[1], observed[1])
                old, new = deepcopy(plain[2]['budgeted_selection']), deepcopy(observed[2]['budgeted_selection'])
                old.pop('seconds'); new.pop('seconds'); new.pop('pack_diagnostics')
                self.assertEqual(old, new)
                self.assertGreater(len(runtime._pack_diagnostics['score_rows']), 0)
                self.assertNotIn('score_rows', observed[2]['budgeted_selection']['pack_diagnostics'])
                self.assertNotIn('pack_diagnostics', plain[2]['budgeted_selection'])

    def test_flat_objective_is_used_at_every_decision_stage(self):
        for arm in ('flat_dependency_support', 'wo_closure_aware_packaging'):
            with self.subTest(arm=arm):
                _, runtime = self.run_arm(arm, {})
                probe = runtime._pack_diagnostics
                self.assertEqual(set(probe['score_comparison']),
                    {'package_bank_control_full', 'initial', 'frontier', 'expansion', 'reader_visible'})
                self.assertGreater(probe['score_comparison']['expansion']['different_objective_count'], 0)
                self.assertIn('partial', probe['score_comparison']['expansion']['by_cardinality'])
                self.assertIn('full', probe['score_comparison']['reader_visible']['by_cardinality'])
                for row in probe['score_rows']:
                    if row['stage'] == 'package_bank_control_full':
                        self.assertEqual(row['used_objective'], row['J_full'])
                    else:
                        self.assertEqual(row['used_objective'], row['J_flat'])
                    self.assertAlmostEqual(row['delta_J_flat_minus_full'],
                        probe['weights']['closure_coefficient']*row['delta_C_flat_minus_closed'], places=12)
                self.assertTrue(probe['chosen_scores']['reader_visible'])
                self.assertTrue(probe['reference_scores']['reader_visible'])

    def test_actual_weights_and_bounded_representatives_are_reported(self):
        _, runtime = self.run_arm(diagnostics={'capture_score_rows': False, 'representative_limit': 3},
                                 weights={'coverage_weight': .6, 'joint_soft_weight': .2})
        probe = runtime._pack_diagnostics
        self.assertEqual(probe['weights']['lambda'], .6)
        self.assertEqual(probe['weights']['eta_configured'], .2)
        self.assertEqual(probe['weights']['rho_effective'], .2)
        self.assertAlmostEqual(probe['weights']['closure_coefficient'], .48)
        self.assertNotIn('score_rows', probe)
        for stage in probe['score_comparison'].values():
            self.assertLessEqual(len(stage['representatives']), 3)

    def test_package_attempt_counts_and_paths_do_not_confuse_containment_with_origin(self):
        full, runtime = self.run_arm(diagnostics={})
        probe = runtime._pack_diagnostics
        search = probe['search']; actions = search['actions']; packages = search['packages']
        self.assertEqual(sum(a['new_unique'] for a in actions.values()), search['expanded_states'])
        self.assertEqual(search['frontier_update_count'], full[2]['budgeted_selection']['expanded_states'])
        for a in actions.values():
            self.assertEqual(a['attempted'], sum(a[k] for k in ('new_unique', 'duplicate', 'budget_skip', 'no_growth')))
        self.assertGreater(packages['capped_bank']['multi_passage'], 0)
        self.assertGreater(actions['package']['multi_passage_growth_new_unique'], 0)
        self.assertEqual(packages['generated_assignments']-packages['duplicate_assignments'],
                         packages['unique_generated_packages'])
        _, runtime = self.run_arm('singleton_search', {})
        single = runtime._pack_diagnostics['search']
        self.assertEqual(single['actions']['package']['attempted'], 0)
        self.assertEqual(single['chosen_path']['package_steps'], 0)
        self.assertEqual(single['package_first_discovery_finalists'], 0)
        self.assertGreater(single['chosen_complete_multi_passage_bank_packages'], 0)

    def test_diagnostic_time_is_separate_from_algorithm_time(self):
        from bindpack.stages.pack.search_diagnostics import PackDiagnostics
        original = PackDiagnostics._score
        def delayed(observer, row):
            time.sleep(.00001)
            original(observer, row)
        with patch.object(PackDiagnostics, '_score', delayed):
            _, runtime = self.run_arm(diagnostics={'capture_score_rows': False})
        timings = runtime._pack_diagnostics['timings']
        self.assertGreater(timings['diagnostic_seconds'], .01)
        self.assertAlmostEqual(timings['algorithm_seconds']+timings['diagnostic_seconds'], timings['wall_seconds'])
        self.assertTrue(runtime._pack_diagnostics['search_budget_excludes_diagnostics'])

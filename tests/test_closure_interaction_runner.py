"""Cost provenance, historical parity and diagnostic aggregation contracts."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

PATH=Path(__file__).resolve().parents[1]/'scripts/run_closure_interaction.py'
SPEC=importlib.util.spec_from_file_location('closure_interaction_runner',PATH)
runner=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(runner)


class ClosureInteractionRunnerTests(unittest.TestCase):
    def test_reuse_requires_exact_full_prompt_and_generation_settings(self):
        budget=dict(prompt_sha256='exact',context_window=8192,reserved_output_tokens=2048)
        row=dict(reader=dict(budget=budget))
        config=dict(llm_name='qwen3-8b',llm_base_url='http://127.0.0.1:8035/v1',temperature=0,
            seed=None,num_gen_choices=1,max_new_tokens=2048)
        self.assertTrue(runner.reusable_full_response(row,dict(budget),config,dict(config)))
        self.assertFalse(runner.reusable_full_response(row,dict(budget,prompt_sha256='other'),config,config))
        self.assertFalse(runner.reusable_full_response(row,budget,config,dict(config,temperature=.2)))
        self.assertFalse(runner.reusable_full_response(row,budget,config,dict(config,seed=17)))

    def test_historical_parity_is_required_for_order_and_objective(self):
        diag=dict(reference_doc_ids=[1,2],candidate_domain=[1,2,3],full_package_bank=[[1,2]],
            expanded_states=8,objective=.8,reference_objective=.7,soft_weight_effective=.5,
            search_deadline_reached=False)
        old=dict(diag,arm='full',ids=[1,2,3])
        runner.parity([1,2,3],dict(budgeted_selection=diag),old)
        with self.assertRaises(AssertionError): runner.parity([2,1,3],dict(budgeted_selection=diag),old)
        with self.assertRaises(AssertionError): runner.parity([1,2,3],dict(budgeted_selection=dict(diag,objective=.81)),old)

    def test_partial_full_score_and_nested_package_aggregation(self):
        scores=dict(C_closed=.5,C_flat=.75,J_full=1.,J_flat=1.1,used_objective=1.)
        stage=dict(count=10,different_support_count=3,different_objective_count=3,
            max_abs_delta_support=.25,max_abs_delta_objective=.1,representatives=[scores],
            by_cardinality=dict(partial=dict(count=8,different_support_count=3,
                different_objective_count=3,max_abs_delta_objective=.1),
                full=dict(count=2,different_support_count=0,different_objective_count=0,max_abs_delta_objective=0.)))
        detail=dict(score_comparison=dict(expansion=stage),reference_scores=scores,chosen_scores=scores,
            weights=dict(closure_coefficient=.225),timings=dict(bounded_search_seconds=.4),
            search=dict(packages=dict(feasible=dict(total=7,singleton=2,multi_passage=5)),
                actions=dict(package=dict(new_unique=3)),chosen_path=dict(package_steps=1,
                    multi_passage_package_steps=1,observed_package_path=True),
                chosen_complete_multi_passage_bank_packages=1))
        cell=dict(ids=[1,2],trace=dict(budgeted_selection=dict(objective=1.)))
        record=dict(cells={a:copy.deepcopy(cell) for a in runner.CELLS},
            diagnostics_summary={a:copy.deepcopy(detail) for a in runner.CELLS})
        result=runner.diagnostic_aggregate([record,record])['full']
        self.assertEqual(result['score_comparison']['expansion']['different_objective_count'],6)
        self.assertEqual(result['score_comparison']['expansion']['by_cardinality']['partial']['count'],16)
        self.assertEqual(result['score_comparison']['expansion']['by_cardinality']['full']['different_objective_count'],0)
        self.assertEqual(result['package_totals']['feasible']['multi_passage'],10)
        self.assertEqual(result['chosen_path_counts']['observed_package_path'],2)
        self.assertAlmostEqual(result['timing']['bounded_search_seconds']['mean'],.4)

    def test_interaction_and_inclusion_samples(self):
        metrics={a:{'Recall@5':v,'DCC@5':None} for a,v in zip(runner.CELLS,[.8,.7,.75,.6])}
        self.assertAlmostEqual(runner.interaction_metrics(metrics)['Recall@5'],-5.)
        self.assertIsNone(runner.interaction_metrics(metrics)['DCC@5'])
        self.assertEqual(runner.parse_inclusions('musique:567,hover:317')['musique'],{567})
        with self.assertRaises(ValueError): runner.parse_inclusions('musique:unknown')

    def test_observer_off_probe_is_balanced_and_checks_every_historical_result(self):
        diag=dict(reference_doc_ids=[1,2],candidate_domain=[1,2,3],full_package_bank=[[1,2]],
            expanded_states=8,objective=.8,reference_objective=.7,soft_weight_effective=.5,
            search_deadline_reached=False,seconds=.25)
        historical={arm:dict(diag,arm=arm,ids=[1,2,3]) for arm in ('full','singleton_search')}
        calls=[]
        def prepare(arm):
            def select():
                calls.append(arm)
                return [1,2,3],[],dict(budgeted_selection=dict(diag))
            return select
        with patch.object(runner.gc,'collect') as collect:
            probe=runner.observer_off_probe(prepare,historical,'stable_id',lambda:dict(residual=0,reader_visible=0))
        self.assertEqual(calls[:2],['full','singleton_search'])
        self.assertEqual(calls[2:],probe['order'])
        self.assertEqual(collect.call_count,6)
        self.assertEqual(sorted(probe['order']),['full','full','singleton_search','singleton_search'])
        self.assertTrue(probe['fully_warm_shared_caches'])
        self.assertFalse(probe['observer_enabled'])
        self.assertTrue(all(row['parity_passed'] for row in probe['measurements']))
        bad=copy.deepcopy(historical);bad['full']['objective']=99
        with self.assertRaises(AssertionError):
            runner.observer_off_probe(prepare,bad,'stable_id',lambda:dict(residual=0,reader_visible=0))

    def test_timing_cohort_is_outcome_blind_and_aggregate_keeps_cache_misses_visible(self):
        indices=list(range(40));ids=['sid'+str(i) for i in indices]
        cohort=runner.observer_off_cohort(indices,ids)
        permuted=runner.observer_off_cohort(indices[::-1],ids[::-1])
        self.assertEqual(cohort,permuted)
        self.assertEqual(len(cohort['selected_indices']),32)
        self.assertEqual(len(runner.observer_off_cohort(indices[:3],ids[:3])['selected_indices']),3)
        probe=dict(observer_enabled=False,parity_passed=True,order=['full','singleton_search','singleton_search','full'],
            measurements=[dict(arm=arm,wall_seconds=wall,process_cpu_seconds=wall,
                native_selector_seconds=wall,parity_passed=True,fully_warm_shared_caches=True)
                for arm,wall in zip(['full','singleton_search','singleton_search','full'],[.3,.2,.2,.3])])
        warm=runner.aggregate_observer_off_timing([dict(observer_off_timing=probe)])
        self.assertEqual(warm['full']['sample_count'],1)
        self.assertEqual(warm['singleton_search']['measurement_count'],2)
        self.assertAlmostEqual(warm['full']['paired_full_minus_singleton_wall_seconds']['mean'],.1)
        probe['measurements'][1]['fully_warm_shared_caches']=False
        cold=runner.aggregate_observer_off_timing([dict(observer_off_timing=probe)])
        self.assertEqual(cold['singleton_search']['shared_cache_miss_measurement_count'],1)
        self.assertEqual(cold['singleton_search']['wall_seconds']['n'],2)
        self.assertEqual(cold['singleton_search']['fully_warm_wall_seconds']['n'],1)
        self.assertEqual(cold['full']['paired_full_minus_singleton_wall_seconds']['n'],0)


if __name__=='__main__': unittest.main()

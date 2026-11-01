import json
from pathlib import Path
import tempfile
import unittest

from utils.serial_cost import DATASETS, aggregate, distribution, sha, stratified_sample, write


class SerialCostTests(unittest.TestCase):
    def test_sampling_is_reproducible_proportional_and_not_first400(self):
        records = [dict(sample_id=str(i), hop=2 if i < 765 else 4, label='') for i in range(1000)]
        selected, strata = stratified_sample('2wikimultihopqa', records)
        self.assertEqual(len(set(selected)), 400)
        self.assertEqual([r['selected'] for r in strata], [306, 94])
        self.assertEqual((selected, strata), stratified_sample('2wikimultihopqa', records))
        self.assertNotEqual(selected, list(range(400)))
        changed_outcomes = [dict(row, answer='a completely different prediction', elapsed=10000) for row in records]
        self.assertEqual(selected, stratified_sample('2wikimultihopqa', changed_outcomes)[0])

    def test_pooled_means_and_p95_use_all1600_queries(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for group, ds in enumerate(DATASETS, 1):
                case = root / 'runs' / ds / 'cases' / ds / 'full'; case.mkdir(parents=True)
                write(root / 'indices' / (ds + '.json'), list(range(400)))
                write(root / 'samples' / (ds + '.json'), dict(indices=list(range(400)),
                    sample_ids=[str(i) for i in range(400)]))
                cfg = dict(stage_ablation='full', mechanism_ablation='full', source_bound_eligibility=False,
                    llm_prefetch_workers=1, qa_top_k=5,
                    pack=dict(selector_mode='combined', review_mode='off', relation_weight=.6))
                write(case / 'protocol.json', dict(configuration=cfg, query_concurrency=1,
                    qa_executed=True, dataset=dict(selected_indices=list(range(400)))))
                write(case / 'audit.json', dict(status='passed')); write(case / 'result.json', {})
                with (case / 'progress.jsonl').open('w') as p, (case / 'llm_calls.jsonl').open('w') as c, (case / 'llm_logical_calls.jsonl').open('w') as l:
                    for i in range(400):
                        p.write(json.dumps(dict(row=dict(sample_id=str(i), timing=dict(execution_mode='serial',
                            end_to_end_seconds=group*2., retrieval_seconds=group*2.-1, reader_seconds=1.))))+'\n')
                        c.write(json.dumps(dict(phase='online_reader', started_unix=group*10000+i,
                            seconds=.1, prompt_tokens=group*100, completion_tokens=group*10))+'\n')
                        l.write(json.dumps(dict(phase='online_reader'))+'\n')
                write(case / 'report.json', dict(coverage=dict(sample_count=400, qa_n=400),
                    efficiency=dict(query_concurrency=1,native_llm_concurrency_limit=1),
                    cost=dict(offline_index=dict(http_attempts=0),llm_logical_calls=400,llm_http_attempts=400,
                        input_tokens_known_total=group*40000,output_tokens_known_total=group*4000,
                        attempts_with_unknown_token_usage=0,llm_failed_attempts=0),
                    resources={},evidence_budget={},metrics={}))
                write(case / 'completed.ok', dict(query_count=400,report_sha256=sha(case/'report.json'),
                    result_sha256=sha(case/'result.json'),llm_calls_sha256=sha(case/'llm_calls.jsonl')))
            result = aggregate(root)
            self.assertEqual(result['n'], 1600)
            self.assertEqual(result['means']['input_tokens_per_query'], 250)
            self.assertEqual(result['means']['output_tokens_per_query'], 25)
            self.assertEqual(result['means']['logical_calls_per_query'], 1)
            self.assertEqual(result['means']['wall_seconds_per_query'], 5)
            self.assertEqual(result['latency']['p95'], 8)
            self.assertNotEqual(result['latency']['p95'], sum(row['latency']['p95'] for row in result['per_dataset'])/4)
            # Refuse to aggregate a missing measurement instead of changing denominator.
            (root/'runs/hover/cases/hover/full/completed.ok').unlink()
            with self.assertRaises(FileNotFoundError):
                aggregate(root)


if __name__ == '__main__':
    unittest.main()

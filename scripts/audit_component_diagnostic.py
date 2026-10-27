#!/usr/bin/env python3
"""Independent saved-artifact audit after the component experiment completes."""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path


def read(path):return json.loads(Path(path).read_text())
def digest(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def jsonl(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s] if Path(path).exists() else []


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('/root/BindPack/outputs/component_diagnostic_r060_20261026'))
    args=p.parse_args();root=args.output
    completed=read(root/'completed.ok');assert completed['status']=='complete'
    selection=read(root/'selection_manifest.json');lock=read(root/'selection_locked.ok')
    assert digest(selection)==lock['sha256']=='719599596273100e9830e7482ef00be1afd75bbb862dc1b266cd1e04b7ea038e'
    protocol=read(root/'experiment.json');cost=Counter();budgets=Counter();paired={};reports=0
    for ds,spec in selection['datasets'].items():
        expected=[r['sample_id'] for r in spec['rows']]
        assert len(expected)==len(set(expected))==50
        receipt=read(root/'runtime'/ds/'parity_r060.ok')
        assert receipt['checked']==50 and receipt['new_http_requests']==0
        base=read(root/'cases'/ds/'baseline/result.json')['results']
        bdetails=read(root/'cases'/ds/'baseline/metrics_details.json')['per_query']
        paired[ds]={}
        for arm in ('baseline','coverage_facts','plan_repair','chain_package','route_fusion','hover_reader','combined'):
            case=root/'cases'/ds/arm;result=read(case/'result.json');report=read(case/'report.json')
            details=read(case/'metrics_details.json')['per_query'];rows=result['results'];c=Counter()
            assert [r['sample_id'] for r in rows]==expected
            assert [r['sample_id'] for r in details]==expected
            assert result['runtime_config']['pack']['relation_weight']==.6
            assert report['coverage']['sample_count']==50
            assert sha(case/'result.json')==read(case/'validated.ok')['result_sha256']
            for row,before,d,bd in zip(rows,base,details,bdetails):
                token=hashlib.sha256(str(row['sample_id']).encode()).hexdigest()+'.json'
                checkpoint=read(case/'checkpoints'/token)
                assert digest(row)==checkpoint['sha256']==digest(checkpoint['row'])
                frozen=read(root/'frozen_inputs'/ds/arm/token)
                assert digest(frozen['input'])==frozen['input_sha256']==row['frozen_pack_input_sha256']
                assert len(row['candidate_docs'])==len(set(row['candidate_docs']))==200
                c['changed_ordered_top5']+=row['candidate_docs'][:5]!=before['candidate_docs'][:5]
                c['changed_top5_set']+=set(row['candidate_docs'][:5])!=set(before['candidate_docs'][:5])
                c['fcr_improved']+=d['at_k']['5']['FCR']>bd['at_k']['5']['FCR']
                c['fcr_worsened']+=d['at_k']['5']['FCR']<bd['at_k']['5']['FCR']
                c['correctness_improved']+=d['qa']['ExactMatch']>bd['qa']['ExactMatch']
                c['correctness_worsened']+=d['qa']['ExactMatch']<bd['qa']['ExactMatch']
                budget=row['reader']['budget']
                budgets['partial_passages']+=len(budget['partial_passage_ranks'])
                budgets['dropped_passages']+=len(budget['dropped_passage_ranks'])
            paired[ds][arm]=dict(c);reports+=1
        calls=jsonl(root/'runtime'/ds/'accounting/llm_calls.jsonl')
        logical=jsonl(root/'runtime'/ds/'accounting/llm_logical_calls.jsonl')
        cost['physical_http_attempts']+=len(calls)
        cost['failed_http_attempts']+=sum(c['status']!='success' for c in calls)
        cost['unknown_usage']+=sum(c.get('prompt_tokens') is None or c.get('completion_tokens') is None for c in calls)
        cost['input_tokens']+=sum(c.get('prompt_tokens') or 0 for c in calls)
        cost['output_tokens']+=sum(c.get('completion_tokens') or 0 for c in calls)
        cost['logical_calls']+=len(logical)
        cost['cache_hits']+=sum(bool(c.get('cache_hit')) for c in logical)
        original=Path('/root/BindPack/outputs/pack_weight_diagnostic_20261025/cases')/ds/'full_r060/result.json'
        assert sha(original)==protocol['original_results'][ds]
    assert reports==28
    for rel,value in read(root/'source_fingerprint.json').items():assert sha(root/'source_snapshot'/rel)==value
    promotion=read(root/'production_promotion.json')
    if promotion['status']=='applied_and_verified':
        parity=read(root/'production_parity.ok');assert parity['checked']==200 and parity['new_requests']==0
        for rel,value in promotion['changes'].items():assert sha(Path('/root/BindPack')/rel)==value
    assert read('/root/BindPack/src/bindpack/configuration/defaults.json')['pack']['relation_weight']==.6
    audit=dict(status='passed',unique_questions=200,quality_reports=28,evaluated_rows=1400,
        original_baseline_parity_questions=200,source_snapshot_unchanged=True,
        original_r060_results_unchanged=4,weights_all_r060=True,incremental_cost=dict(cost),
        reader_budget_counts=dict(budgets),production=promotion['status'])
    (root/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    (root/'paired_changes.json').write_text(json.dumps(paired,ensure_ascii=False,indent=2)+'\n')
    (root/'status.json').write_text(json.dumps(dict(phase='complete',final_arm=completed['final_arm'],
        features=completed['features'],production=promotion['status']),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(audit,ensure_ascii=False))


if __name__=='__main__':main()

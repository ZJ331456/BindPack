#!/usr/bin/env python3
"""Audit completed paired Pack diagnostics and render cohort-aware comparisons."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

PROJECT=Path('/root/BindPack')
ARCHIVE=PROJECT/'outputs/4_datasets_result'
DATASETS=('hotpotqa','2wikimultihopqa','musique','hover')
CASES=('full_r025','full_r060')
GROUPS=('similar','expected_r025','expected_r060')
LABELS={'similar':'预计接近','expected_r025':'预计偏向 0.25','expected_r060':'预计偏向 0.60'}


def read(path):return json.loads(path.read_text())


def digest(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=PROJECT/'outputs/pack_weight_diagnostic_20261025')
    args=parser.parse_args();root=args.output.resolve()
    selection=read(root/'selection_manifest.json');lock=read(root/'selection_locked.ok')
    assert digest(selection)==lock['sha256']
    all_rows=[];datasets=[];costs=Counter();counts=Counter()
    for ds in DATASETS:
        reports=[read(root/'cases'/ds/case/'report.json') for case in CASES]
        details=[read(root/'cases'/ds/case/'metrics_details.json')['per_query'] for case in CASES]
        pairs=[json.loads(line) for line in (root/'runtime'/ds/'pairs.jsonl').open() if line.strip()]
        expected=selection['datasets'][ds]['rows']
        assert len(pairs)==len(expected)==50
        assert Counter(row['group'] for row in expected)==dict(similar=10,expected_r025=20,expected_r060=20)
        assert reports[0]['sample_alignment']['sample_ids_sha256']==reports[1]['sample_alignment']['sample_ids_sha256']
        assert all(r['coverage']['qa_n']==r['coverage']['sample_count']==r['coverage']['document_labels_valid']==50 for r in reports)
        params=[read(root/'metadata'/ds/(case+'_method.json')) for case in CASES]
        assert params[0]['pack']['relation_weight']==.25 and params[1]['pack']['relation_weight']==.60
        params[1]['pack']['relation_weight']=.25;assert params[0]==params[1]
        dataset_rows=[];stages={case:Counter() for case in CASES}
        for pair,chosen,a,b in zip(pairs,expected,*details):
            assert pair['sample_id']==chosen['sample_id']==a['sample_id']==b['sample_id']
            assert digest(pair['payload'])==pair['sha256']
            qid=chosen['sample_id']
            frozen=read(root/'frozen_inputs'/ds/(hashlib.sha256(str(qid).encode()).hexdigest()+'.json'))
            fingerprint=digest(frozen['input']);assert fingerprint==frozen['input_sha256']==pair['payload']['frozen_pack_input_sha256']
            row=dict(dataset=ds,sample_id=qid,cohort=chosen['group'],selection_reason=chosen['reason'],
                benchmark_hops=chosen['benchmark_hops'],pack_input_sha256=fingerprint,
                r025_recall5=a['at_k']['5']['Recall'],r060_recall5=b['at_k']['5']['Recall'],
                r025_fcr5=a['at_k']['5']['FCR'],r060_fcr5=b['at_k']['5']['FCR'],
                r025_em=a['qa']['ExactMatch'],r060_em=b['qa']['ExactMatch'],
                r025_answer_f1=a['qa']['F1'],r060_answer_f1=b['qa']['F1'],
                r025_dcc5=a['at_k']['5']['DCC'],r060_dcc5=b['at_k']['5']['DCC'])
            for case,weight in zip(CASES,(.25,.60)):
                raw=pair['payload']['rows'][case]
                assert raw['pack_relation_weight']==weight
                assert raw['frozen_pack_input_sha256']==fingerprint
                assert raw['retrieval_trace']['evidence']['stages']==dict(route=True,plan=True,pack=True)
                assert len(raw['candidate_docs'])==len(set(raw['candidate_docs']))==200
                budget=raw['reader']['budget'];assert budget['used_passages']<=5
                counts['reader_partial']+=bool(budget['partial_passage_ranks'])
                counts['reader_dropped']+=bool(budget['dropped_passage_ranks'])
                sem=raw['retrieval_trace']['evidence']['improvement_support_semantic_veto']
                docs=dict(zip(sem['final_doc_ids'],raw['candidate_docs']))
                gold=set(raw['gold_docs'])
                before=gold<=set(docs[d] for d in sem['original_parent_doc_ids'][:5])
                after=bool(raw['all_gold_in_top5'])
                stages[case]['before_review_fcr_count']+=before
                stages[case]['after_review_fcr_count']+=after
                stages[case]['review_fcr_gains']+=after>before
                stages[case]['review_fcr_losses']+=after<before
            dataset_rows.append(row);all_rows.append(row)
        groups=[]
        for group in GROUPS:
            rows=[r for r in dataset_rows if r['cohort']==group]
            groups.append(dict(group=group,n=len(rows),
                fcr_r025=sum(r['r025_fcr5'] for r in rows)/len(rows),fcr_r060=sum(r['r060_fcr5'] for r in rows)/len(rows),
                fcr_r060_wins=sum(r['r060_fcr5']>r['r025_fcr5'] for r in rows),
                fcr_r025_wins=sum(r['r060_fcr5']<r['r025_fcr5'] for r in rows),
                fcr_ties=sum(r['r060_fcr5']==r['r025_fcr5'] for r in rows),
                em_r025=sum(r['r025_em'] for r in rows)/len(rows),em_r060=sum(r['r060_em'] for r in rows)/len(rows),
                em_r060_wins=sum(r['r060_em']>r['r025_em'] for r in rows),
                em_r025_wins=sum(r['r060_em']<r['r025_em'] for r in rows)))
        datasets.append(dict(dataset=ds,n=50,r025=reports[0]['metrics'],r060=reports[1]['metrics'],groups=groups,
            stages={case:dict(value) for case,value in stages.items()},
            em_r060_wins=sum(r['r060_em']>r['r025_em'] for r in dataset_rows),em_r025_wins=sum(r['r060_em']<r['r025_em'] for r in dataset_rows),
            fcr_r060_wins=sum(r['r060_fcr5']>r['r025_fcr5'] for r in dataset_rows),fcr_r025_wins=sum(r['r060_fcr5']<r['r025_fcr5'] for r in dataset_rows)))
        calls_path=root/'runtime'/ds/'accounting/llm_calls.jsonl'
        calls=[json.loads(s) for s in calls_path.open() if s.strip()] if calls_path.exists() else []
        assert not any('offline' in c['phase'] or c['phase']=='model_and_index_loading' for c in calls)
        costs['http_attempts']+=len(calls)
        costs['failed_attempts']+=sum(c.get('status')!='success' for c in calls)
        costs['unknown_usage']+=sum(c.get('prompt_tokens') is None or c.get('completion_tokens') is None for c in calls)
        costs['input_tokens']+=sum(c.get('prompt_tokens') or 0 for c in calls)
        costs['output_tokens']+=sum(c.get('completion_tokens') or 0 for c in calls)
    assert len(all_rows)==200
    with (root/'paired_queries.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(all_rows[0]));writer.writeheader()
        writer.writerows({k:'N/A' if v is None else v for k,v in row.items()} for row in all_rows)
    write(root/'detailed_comparison.json',dict(diagnostic_only=True,selection_sha256=lock['sha256'],datasets=datasets,incremental_cost=dict(costs)))
    labels={'hotpotqa':'HotpotQA','2wikimultihopqa':'2WikiMultiHopQA','musique':'MuSiQue','hover':'HoVer'}
    lines=['# 200 题 Full Pack 配对诊断结果','','200 道固定问题、400 份回答已完成。每个数据集按历史记录预选 10 条接近、20 条预计偏向 0.25、20 条预计偏向 0.60；样本清单在新推理前冻结，实际结果不满足预期也全部保留。**这是诊断集，不能把汇总差值当作全量或无偏主实验提升。**','',
        '两个设置均为 Full，唯一参数差异是 Pack 的 relation_weight。每题 Route+Plan 输入完全相同，使用原生 DAG、末端语义审核和 Reader@5。前三个数据集的后两列为 EM / token F1，HoVer 为 Accuracy / Macro-F1。','',
        '| 数据集 | R 权重 | Recall@5 | FCR@5 | EM / Accuracy | token F1 / Macro-F1 | DCC@5 | NodeCov@5 | ClosureGap@5 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in datasets:
        for key,weight,case in [('r025',.25,CASES[0]),('r060',.60,CASES[1])]:
            m=r[key];ds=r['dataset'];qa='ClassificationAccuracy' if ds=='hover' else 'ExactMatch';f1='ClassificationMacroF1' if ds=='hover' else 'F1'
            values=[m.get(k) for k in ['Recall@5','FCR@5',qa,f1,'DCC@5','NodeCov@5','ClosureGap@5']]
            lines.append(f'| [{labels[ds]}](cases/{ds}/{case}/report.json) | {weight:.2f} | '+' | '.join('N/A' if x is None else f'{x:.6f}' for x in values)+' |')
    lines+=['','## 逐题净变化','','| 数据集 | 仅 0.60 证据齐全 | 仅 0.25 证据齐全 | 仅 0.60 答对 | 仅 0.25 答对 |','|---|---:|---:|---:|---:|']
    for r in datasets:lines.append(f'| {labels[r["dataset"]]} | {r["fcr_r060_wins"]} | {r["fcr_r025_wins"]} | {r["em_r060_wins"]} | {r["em_r025_wins"]} |')
    lines+=['','## 预选分组与实际结果','','“预计偏向”只表示选题依据，不强制实际偏好。表中胜负按 FCR@5 判断。','','| 数据集 | 历史预选组 | n | 0.25 FCR@5 | 0.60 FCR@5 | 0.60 胜 / 0.25 胜 / 相同 |','|---|---|---:|---:|---:|---|']
    for r in datasets:
        for g in r['groups']:lines.append(f'| {labels[r["dataset"]]} | {LABELS[g["group"]]} | {g["n"]} | {g["fcr_r025"]:.3f} | {g["fcr_r060"]:.3f} | {g["fcr_r060_wins"]} / {g["fcr_r025_wins"]} / {g["fcr_ties"]} |')
    lines+=['','## 审核之前与之后','','下表的“审核前”包含原生贪心与 DAG package，不包含最后的提案/语义审核。','','| 数据集 | R 权重 | 审核前 FCR@5 | 最终 FCR@5 | 审核净修复 |','|---|---:|---:|---:|---:|']
    for r in datasets:
        for case,weight in zip(CASES,(.25,.60)):
            s=r['stages'][case];lines.append(f'| {labels[r["dataset"]]} | {weight:.2f} | {s["before_review_fcr_count"]/50:.3f} | {s["after_review_fcr_count"]/50:.3f} | {s["review_fcr_gains"]-s["review_fcr_losses"]:+d} |')
    lines+=['','## 使用范围与复查','','- 全部 200 对共享输入 SHA256、样本身份、完整 200 篇排名、两组参数唯一差异均已校验。原项目 defaults.json 仍保持 0.25，本轮只使用输出目录中的独立实验配置。',
        '- 历史 HotpotQA / 2Wiki 的 hop prior=2 保持一致，MuSiQue 使用分解长度，HoVer 使用 num_hops；这是为隔离 Pack 参数，不是修复 hop 后的正式重跑。',
        '- 索引直接复用 4_datasets_result/shared_indexes。历史 Full 缓存用于共享 Route+Plan 和完全相同的模型提示；增量物理请求、cache hits、token 和资源账本保留。它们不代表独立冷缓存的新框架成本。',
        '- 分组按照历史结果选择，样本较小；模型提示及语义审核会随排名变化。本次结果可以验证机制和修复方向，仍需在独立开发样本上决定最终参数，正式测试集使用锁定后的配置。',
        '- 指标原始值见各 case 的 report.json / metrics_details.json；逐题对照见 [paired_queries.csv](paired_queries.csv)，分组及内部阶段见 [detailed_comparison.json](detailed_comparison.json)。']
    (root/'comparison_detailed.md').write_text('\n'.join(lines)+'\n')
    manifest=read(ARCHIVE/'manifest.json');verified=0
    for entry in manifest['files']:
        if entry['path'].startswith('shared_indexes/') and (entry['path'].endswith(('.parquet','graph.pickle')) or
                Path(entry['path']).name.startswith('openie_results_') and entry['path'].endswith('.json')):
            assert sha(ARCHIVE/entry['path'])==entry['sha256'],entry['path'];verified+=1
    audit=dict(status='passed',paired_queries=200,answer_evaluations=400,all_shared_input_hashes_match=True,
        all_selection_counts_match=True,all_parameter_pairs_differ_only_in_relation_weight=True,
        shared_index_assets_unchanged=verified,index_build_llm_calls=0,incremental_cost=dict(costs),reader_budget_counts=dict(counts))
    write(root/'audit.json',audit)
    print(json.dumps(dict(audit=audit,datasets=[{k:r[k] for k in ('dataset','fcr_r060_wins','fcr_r025_wins','em_r060_wins','em_r025_wins')} for r in datasets]),ensure_ascii=False))


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Audit the paired 150-per-dataset experiment and report pooled/stratum changes."""
from collections import Counter
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

PROJECT=Path('/root/BindPack')
DEFAULT=PROJECT/'outputs/pack_150_before_after_20261026'
DATASETS=('hotpotqa','2wikimultihopqa','musique','hover')
METRICS=('Recall@5','FCR@5','DCC@5','NodeCov@5','ClosureGap@5')


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def jsonl(p):return [json.loads(v) for v in Path(p).read_text().splitlines() if v] if Path(p).exists() else []
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def exact_p(wins,losses):
    n=wins+losses
    return 1. if n==0 else min(1.,2.*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n)


def qa_names(ds):return ('ClassificationAccuracy','ClassificationMacroF1') if ds=='hover' else ('ExactMatch','F1')
def case(root,ds,cohort,arm):return root/'cases'/ds/arm if cohort=='combined150' else root/'strata'/ds/cohort/arm
def fmt(v):return 'N/A' if v is None else f'{v:.6f}'
def delta_fmt(v):return 'N/A' if v is None else f'{v:+.6f}'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT);args=p.parse_args();root=args.output
    assert read(root/'completed.ok')['status']=='complete'
    selected=read(root/'selection_manifest.json');assert digest(selected)==read(root/'selection_locked.ok')['sha256']
    protocol=read(root/'experiment.json');fingerprints=read(root/'source_fingerprint.json')
    for arm,hashes in fingerprints.items():
        for rel,value in hashes.items():
            assert sha(root/'source_snapshot'/arm/'src'/rel)==value
            if arm=='after':assert sha(PROJECT/'src'/rel)==value,('Production source changed',rel)
    paired=[];findings={};costs={};totals=Counter();reports=0;parity=Counter();budget=Counter()
    lines=['# 固定150条：Pack优化前后变化','',
        '每个数据集150条，共600条；每题固定同一Route+Plan输入。前后权重均为0.60，使用相同当前reader、模型和预算；after仅包含目标感知去重、无效请求短路。索引和embedding结果全部复用。','',
        '150条由原诊断50条与此前选定的额外100条组成，没有重新挑题。前50条参与过方案筛选，因此同时单列新增100条；该批仍是开发诊断，不能视为独立正式测试。','',
        '## 合并150条的指标','',
        '| 数据集 | 版本 | Recall@5 | FCR@5 | EM / Accuracy | F1 / Macro-F1 | DCC@5 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for ds in DATASETS:
        chosen=selected['datasets'][ds]['rows'];ids=[q['sample_id'] for q in chosen]
        assert len(ids)==len(set(ids))==150
        assert Counter(q['cohort'] for q in chosen)==dict(old50=50,new100=100)
        inputs=protocol['inputs'][ds];assert sha(root/'metadata'/ds/'benchmark.json')==inputs['benchmark_sha256']
        assert sha(root/'metadata'/ds/'corpus.json')==inputs['corpus_sha256']
        for cohort,refs in inputs['references'].items():
            for ref in refs.values():assert sha(ref['path'])==ref['sha256']
        findings[ds]={};costs[ds]={}
        for cohort in ('combined150','old50','new100'):
            expected_ids=ids if cohort=='combined150' else [q['sample_id'] for q in chosen if q['cohort']==cohort]
            data={};details={};metrics={};summed={}
            for arm in ('before','after'):
                directory=case(root,ds,cohort,arm);result=read(directory/'result.json');rows=result['results']
                report=read(directory/'report.json');detail=read(directory/'metrics_details.json')['per_query']
                assert [v['sample_id'] for v in rows]==expected_ids==[v['sample_id'] for v in detail]
                assert sha(directory/'result.json')==read(directory/'validated.ok')['result_sha256']
                assert report['coverage']['sample_count']==len(expected_ids)
                assert result['runtime_config']['pack']['relation_weight']==.6 and result['runtime_config']['qa_top_k']==5
                cache_cohorts=('old50','new100') if cohort=='combined150' else (cohort,)
                caches={c:str(root/'runtime'/ds/c/'index') for c in cache_cohorts}
                assert result['cache_directories_by_cohort']==caches
                assert result['runtime_config']['save_dir']==(None if cohort=='combined150' else caches[cohort])
                data[arm]=rows;details[arm]=detail;metrics[arm]=report['metrics'];summed[arm]=read(directory/'cost.json')['committed_rows'];reports+=1
                for name in METRICS:
                    values=[q['at_k']['5'][name.removesuffix('@5')] for q in detail if q['at_k']['5'][name.removesuffix('@5')] is not None]
                    assert (metrics[arm][name] is None) if not values else abs(metrics[arm][name]-sum(values)/len(values))<1e-12
                correctness,f1=qa_names(ds)
                assert abs(metrics[arm][correctness]-sum(q['qa']['ExactMatch'] for q in detail)/len(detail))<1e-12
                if ds=='hover':
                    class_f1=[]
                    for label in ('SUPPORTED','NOT_SUPPORTED'):
                        tp=sum(q['qa']['gold_label']==label and q['qa']['predicted_label']==label for q in detail)
                        fp=sum(q['qa']['gold_label']!=label and q['qa']['predicted_label']==label for q in detail)
                        fn=sum(q['qa']['gold_label']==label and q['qa']['predicted_label']!=label for q in detail)
                        class_f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
                    assert abs(metrics[arm][f1]-sum(class_f1)/2)<1e-12
            correctness,f1=qa_names(ds);keys=METRICS+(correctness,f1)
            differences={k:None if metrics['before'][k] is None else metrics['after'][k]-metrics['before'][k] for k in keys}
            stats=Counter()
            for before,after,bd,ad in zip(data['before'],data['after'],details['before'],details['after']):
                assert before['frozen_pack_input_sha256']==after['frozen_pack_input_sha256']
                stats['top5_order_changes']+=before['candidate_docs'][:5]!=after['candidate_docs'][:5]
                stats['top5_set_changes']+=set(before['candidate_docs'][:5])!=set(after['candidate_docs'][:5])
                stats['fcr_wins']+=ad['at_k']['5']['FCR']>bd['at_k']['5']['FCR']
                stats['fcr_losses']+=ad['at_k']['5']['FCR']<bd['at_k']['5']['FCR']
                stats['correctness_wins']+=ad['qa']['ExactMatch']>bd['qa']['ExactMatch']
                stats['correctness_losses']+=ad['qa']['ExactMatch']<bd['qa']['ExactMatch']
                if cohort!='combined150':continue
                chosen_row=next(q for q in chosen if q['sample_id']==before['sample_id'])
                assert before['frozen_pack_input_sha256']==chosen_row['frozen_pack_input_sha256']
                assert sha(chosen_row['frozen_path'])==chosen_row['frozen_file_sha256']
                for arm,row in (('before',before),('after',after)):
                    saved=read(root/'checkpoints'/ds/arm/(hashlib.sha256(str(row['sample_id']).encode()).hexdigest()+'.json'))
                    assert digest(row)==saved['sha256']==digest(saved['row'])
                    assert len(row['candidate_docs'])==200
                    sem=row['retrieval_trace']['evidence']['improvement_support_semantic_veto']
                    assert len(sem['final_doc_ids'])==len(set(sem['final_doc_ids']))==200
                    assert all(sem[k] for k in ('top2_preserved','top10_set_preserved','top200_set_preserved','suffix_preserved','unique_documents','protected_support_preserved'))
                    budget['partial_passages']+=len(row['reader']['budget']['partial_passage_ranks'])
                    budget['dropped_passages']+=len(row['reader']['budget']['dropped_passage_ranks'])
                    ref=inputs['references'][row['comparison_cohort']][arm]
                    # References are shared by 50/100 rows, cached below after first read.
                    cache_key=(ds,row['comparison_cohort'],arm)
                    if cache_key not in reference_cache:reference_cache[cache_key]={q['sample_id']:q for q in read(ref['path'])['results']}
                    old=reference_cache[cache_key][row['sample_id']]
                    if arm=='before' or row['comparison_cohort']=='old50':
                        assert row['candidate_docs']==old['candidate_docs'];parity['historical_ranking_checks']+=1
                        if row['comparison_cohort']=='old50' or ds!='hover':
                            assert row['answer']==old['answer'] and row['reader']['budget']['prompt_sha256']==old['reader']['budget']['prompt_sha256']
                            parity['historical_answer_prompt_checks']+=1
                        else:parity['hover_new100_common_reader_reanswers']+=1
                record=dict(dataset=ds,cohort=before['comparison_cohort'],sample_id=before['sample_id'],question=before['question'],
                    before_answer=before['answer'],after_answer=after['answer'])
                for name in METRICS:
                    record['before_'+name]=bd['at_k']['5'][name.removesuffix('@5')]
                    record['after_'+name]=ad['at_k']['5'][name.removesuffix('@5')]
                record.update(before_correct=bd['qa']['ExactMatch'],after_correct=ad['qa']['ExactMatch'])
                paired.append(record)
            findings[ds][cohort]=dict(before=metrics['before'],after=metrics['after'],delta=differences,
                paired_counts=dict(stats),fcr_p_exact=exact_p(stats['fcr_wins'],stats['fcr_losses']),
                correctness_p_exact=exact_p(stats['correctness_wins'],stats['correctness_losses']),
                logical_calls_before=summed['before']['logical_calls'],logical_calls_after=summed['after']['logical_calls'])
            if cohort=='combined150':
                for arm in ('before','after'):
                    m=metrics[arm];lines.append(f'| {ds} | {arm} | '+' | '.join(fmt(m[k]) for k in ('Recall@5','FCR@5',correctness,f1,'DCC@5'))+' |')
        for arm in ('before','after'):
            c=Counter();resources={}
            for cohort in ('old50','new100'):
                directory=root/'accounting'/ds/arm/cohort;http=jsonl(directory/'llm_calls.jsonl');logical=jsonl(directory/'llm_logical_calls.jsonl')
                if cohort=='old50':assert not http
                if arm=='before' and ds!='hover':assert not http
                if arm=='before' and ds=='hover':assert all(v['phase'].startswith('reader_') for v in http)
                c.update(dict(http_attempts=len(http),successful_http=sum(v['status']=='success' for v in http),failed_http=sum(v['status']!='success' for v in http),
                    unknown_usage=sum(v.get('prompt_tokens') is None or v.get('completion_tokens') is None for v in http),
                    physical_input_tokens=sum(v.get('prompt_tokens') or 0 for v in http),physical_output_tokens=sum(v.get('completion_tokens') or 0 for v in http),
                    logical_calls=len(logical),cache_hits=sum(bool(v.get('cache_hit')) for v in logical)))
                resources[cohort]=read(directory/'resource_summary.json')
            costs[ds][arm]=dict(actual_execution=dict(c),committed_rows=read(root/'cases'/ds/arm/'cost.json')['committed_rows'],resources=resources)
            totals.update(c)
    assert reports==24 and len(paired)==600 and not budget['partial_passages'] and not budget['dropped_passages']
    assert parity==dict(historical_ranking_checks=800,historical_answer_prompt_checks=700,hover_new100_common_reader_reanswers=100)
    write(root/'analysis.json',findings);write(root/'costs.json',costs)
    with (root/'paired_queries.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
    for cohort,title in (('new100','新增100条：单列扩展验证'),('old50','原50条：保留项的诊断集复现')):
        lines+=['','## '+title,'','| 数据集 | ΔRecall@5 | ΔFCR@5 | ΔEM / Accuracy | ΔF1 / Macro-F1 | ΔDCC@5 |','|---|---:|---:|---:|---:|---:|']
        for ds in DATASETS:
            correctness,f1=qa_names(ds);d=findings[ds][cohort]['delta']
            lines.append(f'| {ds} | '+' | '.join(delta_fmt(d[k]) for k in ('Recall@5','FCR@5',correctness,f1,'DCC@5'))+' |')
    lines+=['','## 合并150条的结构与成本','',
        '| 数据集 | NodeCov@5 前→后 | DCC@5 前→后 | ClosureGap@5 前→后 | Pack+Reader逻辑调用 前→后 |',
        '|---|---|---|---|---|']
    for ds in DATASETS:
        a=findings[ds]['combined150'];b=a['before'];c=a['after']
        lines.append(f'| {ds} | '+' | '.join(f'{fmt(b[k])} → {fmt(c[k])}' for k in ('NodeCov@5','DCC@5','ClosureGap@5'))+f' | {a["logical_calls_before"]} → {a["logical_calls_after"]} |')
    lines+=['','## 逐题配对','',
        '| 数据集 | FCR改善 / 损坏 | 正确率改善 / 损坏 | FCR精确p | 正确率精确p |',
        '|---|---|---|---:|---:|']
    for ds in DATASETS:
        d=findings[ds]['combined150'];c=d['paired_counts']
        lines.append(f'| {ds} | {c["fcr_wins"]} / {c["fcr_losses"]} | {c["correctness_wins"]} / {c["correctness_losses"]} | {d["fcr_p_exact"]:.6f} | {d["correctness_p_exact"]:.6f} |')
    lines+=['','精确p为配对二项检验/McNemar精确形式，未做跨数据集及多轮筛选校正；本报告不据此声称正式显著性。',
        '','## 如何理解这次扩展','',
        '合并150条后，HotpotQA、MuSiQue、HoVer的回答主指标提高，2Wiki不变；HoVer的Recall@5略降，因此不能称为所有指标均不退化。',
        '','新增100条单独看，四个数据集FCR@5均没有净提升；MuSiQue和HoVer的Recall@5略降，但回答指标有小幅收益。合并表中的FCR净增来自已用于筛选的原50条。该结果支持继续评估回答收益，但尚不足以证明证据完整率提升具有稳定的外推性。',
        '','## 协议与计量说明','',
        '- before为优化前的0.60 Pack，并非旧全量目录的0.25 Full；after为当前已采用的两个改动。本轮没有根据新结果继续调参，也没有修改生产算法。',
        '- HoVer新增100条原来使用旧reader。本轮对照双方统一为当前原子核验reader，因此这100条的before QA与历史文件可能不同；其检索排名必须逐题一致。旧文件保持原样。HoVer合并Macro-F1直接由150条预测重算，不平均两组Macro-F1。',
        '- 冻结输入沿用历史hop条件：HotpotQA/2Wiki prior=2，MuSiQue/HoVer沿用原辅助条件；不代表修复hop后的正式主实验。',
        '- 只有MuSiQue有可信gold DAG，因此其余数据集的NodeCov/DCC/ClosureGap为N/A。',
        '- 时间、资源和tokens均为本次缓存复用的Pack+Reader增量执行，不包含冻结的Route/Plan和离线建库；无法据此推断冷缓存端到端加速。GPU采样含共享vLLM负载，不是独占GPU成本。',
        '',f'本次实际新增HTTP尝试 {totals["http_attempts"]}，其中失败 {totals["failed_http"]}；输入 {totals["physical_input_tokens"]}、输出 {totals["physical_output_tokens"]} tokens，usage缺失 {totals["unknown_usage"]}。',
        '','完整分层指标：[comparison.md](comparison.md)；逐题配对：[paired_queries.csv](paired_queries.csv)；机器可读分析：[analysis.json](analysis.json)；调用/资源账本：[costs.json](costs.json)；核验：[audit.json](audit.json)。']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    audit=dict(status='passed',unique_questions=600,paired_predictions=1200,quality_reports=reports,
        selection_sha256=digest(selected),source_snapshots_unchanged=True,production_source_unchanged=True,
        original_result_files_unchanged=True,index_rebuilt=False,embedding_inference=False,route_plan_frozen=True,
        historical_parity=dict(parity),reader_budget=dict(budget),incremental_cost=dict(totals))
    write(root/'audit.json',audit);print(json.dumps(audit,ensure_ascii=False))


reference_cache={}
if __name__=='__main__':main()

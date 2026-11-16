#!/usr/bin/env python3
"""Readable exports for saved closure/packaging diagnostic summaries."""
import argparse
import csv
import json
from pathlib import Path

DATASETS=('hotpotqa','2wikimultihopqa','musique','hover')
CELLS=('full','flat_dependency_support','singleton_search','wo_closure_aware_packaging')
LABELS=('Full','Flat Support','Singleton Search','w/o Closure-aware Packaging')


def read(path):
    return json.loads(Path(path).read_text())


def write_csv(path,columns,rows):
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)
    temporary.replace(path)


def refresh(root):
    root=Path(root);output=root/'analysis';output.mkdir(parents=True,exist_ok=True)
    scores=[];packages=[];timings=[];observer_off=[];aggregates={}
    for ds in DATASETS:
        path=root/'diagnostics'/ds/'aggregate.json'
        if not path.exists():continue
        values=read(path);aggregates[ds]=values
        for arm,label in zip(CELLS,LABELS):
            cell=values[arm];n=cell['queries'];p=cell['package_totals'];a=cell['action_totals'].get('package',{})
            paths=cell.get('chosen_path_counts',{});totals=cell.get('search_totals',{})
            packages.append(dict(dataset=ds,variant=label,n=n,
                generated_assignments=p.get('generated_assignments',0),
                unique_generated_packages=p.get('unique_generated_packages',0),
                feasible_packages=p.get('feasible',{}).get('total',0),
                capped_bank_packages=p.get('capped_bank',{}).get('total',0),
                capped_multi_passage_packages=p.get('capped_bank',{}).get('multi_passage',0),
                active_package_actions=p.get('active_actions',{}).get('total',0),
                attempted=a.get('attempted',0),new_unique=a.get('new_unique',0),
                duplicate=a.get('duplicate',0),budget_skip=a.get('budget_skip',0),no_growth=a.get('no_growth',0),
                multi_passage_growth_new_unique=a.get('multi_passage_growth_new_unique',0),
                first_discovery_package_chosen=paths.get('package_first_discovery',0),
                first_discovery_multi_package_chosen=paths.get('multi_passage_package_first_discovery',0),
                observed_multi_package_path=paths.get('observed_multi_passage_package_path',0),
                observed_multi_growth_path=paths.get('observed_multi_passage_growth_path',0),
                contains_multi_package=paths.get('contains_multi_passage_bank_package',0),
                package_first_discovery_finalists=totals.get('package_first_discovery_finalists',0),
                total_expanded_states=totals.get('expanded_states',0),
                mean_expanded_states=totals.get('expanded_states',0)/max(1,n),
                mean_finalists=totals.get('finalist_count',0)/max(1,n)))
            for metric,distribution in cell.get('timing',{}).items():
                timings.append(dict(dataset=ds,variant=label,n=n,metric=metric,
                    **{k+'_ms':None if distribution.get(k) is None else 1000*distribution[k]
                       for k in ('mean','p50','p95','min','max','total')}))
            probe=cell.get('observer_off_timing',{})
            if probe.get('sample_count',0):
                row=dict(dataset=ds,variant=label,
                    **{key:probe.get(key,0) for key in ('sample_count','measurement_count',
                        'fully_warm_measurement_count','shared_cache_miss_measurement_count','fully_warm_sample_count')})
                for clock in ('wall_seconds','process_cpu_seconds','native_selector_seconds',
                        'fully_warm_wall_seconds','fully_warm_process_cpu_seconds',
                        'paired_full_minus_singleton_wall_seconds','paired_full_minus_singleton_process_cpu_seconds'):
                    for quantile in ('mean','p50','p95'):
                        value=probe.get(clock,{}).get(quantile)
                        row[clock+'_'+quantile+'_ms']=None if value is None else value*1000
                observer_off.append(row)
            for stage,summary in cell.get('score_comparison',{}).items():
                for row in summary.get('representatives',[]):
                    item=dict(row,dataset=ds,variant=label,stage=stage)
                    item['doc_ids']=json.dumps(row.get('doc_ids',[]),separators=(',',':'))
                    scores.append(item)
    # Initial independently audited examples remain useful before the first
    # formal window has finished; their provenance is explicit in the export.
    if not scores and (output/'score_probe_preflight.json').exists():
        for example in read(output/'score_probe_preflight.json').get('score_examples',[]):
            for row in example['score_examples']:
                item=dict(row,dataset=example['dataset'],query_index=example['query_index'],
                    variant=LABELS[CELLS.index(example['arm'])],provenance='independent_preflight_replay')
                item['doc_ids']=json.dumps(row.get('doc_ids',[]),separators=(',',':'));scores.append(item)
    write_csv(output/'CANDIDATE_SCORE_COMPARISONS.csv',[
        'dataset','query_index','sample_id','variant','stage','candidate_scope','reader_visible','doc_ids',
        'C_closed','C_flat','J_full','J_flat','delta_J_flat_minus_full','used_objective','effective_mode',
        'relevance','H','redundancy','support_loss','provenance'],scores)
    package_columns=['dataset','variant','n','generated_assignments','unique_generated_packages','feasible_packages',
        'capped_bank_packages','capped_multi_passage_packages','active_package_actions','attempted','new_unique',
        'multi_passage_growth_new_unique','duplicate','budget_skip','no_growth','first_discovery_package_chosen',
        'first_discovery_multi_package_chosen','observed_multi_package_path','observed_multi_growth_path',
        'contains_multi_package','package_first_discovery_finalists','total_expanded_states',
        'mean_expanded_states','mean_finalists']
    write_csv(output/'PACKAGE_ACTION_STATISTICS.csv',package_columns,packages)
    write_csv(output/'SEARCH_TIMING.csv',['dataset','variant','n','metric','mean_ms','p50_ms','p95_ms',
        'min_ms','max_ms','total_ms'],timings)
    clocks=('wall_seconds','process_cpu_seconds','native_selector_seconds','fully_warm_wall_seconds',
        'fully_warm_process_cpu_seconds','paired_full_minus_singleton_wall_seconds',
        'paired_full_minus_singleton_process_cpu_seconds')
    write_csv(output/'OBSERVER_OFF_TIMING.csv',['dataset','variant','sample_count','measurement_count',
        'fully_warm_measurement_count','shared_cache_miss_measurement_count','fully_warm_sample_count']+
        [c+'_'+q+'_ms' for c in clocks for q in ('mean','p50','p95')],observer_off)
    lines=['# 闭包评分与来源包动作诊断','',
        '本表随正式运行更新。旧Full/Flat/Singleton只作冻结输入的离线诊断重放；仅新增组合组执行检索选择与QA评测。', '',
        '## 原始4000题的支持饱和证据','']
    initial=output/'initial_trace_audit.json'
    if initial.exists():
        audit=read(initial);lines+=['|数据集|题数|有来源支持|S₀已C=1|Full最终C=1|有多篇包|触及状态上限|',
            '|---|---:|---:|---:|---:|---:|---:|']
        for ds in DATASETS:
            c=audit['datasets'][ds]['full']['counts']
            lines.append(f'|{ds}|{c["queries"]}|{c["queries"]-c["no_source_choices"]}|{c["s0_closure_full"]}|'
                f'{c["selected_closure_full"]}|{c["multi_passage_bank_queries"]}|{c["state_cap_reached"]}|')
        lines+=['','3653道有来源支持题中，S₀已满分3612题（98.88%），Full最终满分3641题（99.67%）。'
            '来源资格严格核验在原Full中关闭，预测支持与独立金标证据完整性不是同一指标。', '']
    lines+=['## 同一候选集合的两种目标分数','',
        '详细代表集见 [CANDIDATE_SCORE_COMPARISONS.csv](CANDIDATE_SCORE_COMPARISONS.csv)。'
        '每题所有被评估集合保存在 `../diagnostics/<dataset>/<query_index>.json.gz` 的四个cell的score_rows中。', '',
        '|数据集/题号|阶段|文档ID集合|C_closed|C_flat|J_full|J_flat|ΔJ|',
        '|---|---|---|---:|---:|---:|---:|---:|']
    example_rows=[r for r in scores if r.get('stage')=='expansion' and
        abs(r.get('delta_J_flat_minus_full',0))>1e-12][:6]
    for row in example_rows:
        lines.append('|'+ '|'.join([f'{row["dataset"]}/{row.get("query_index","?")}',row['stage'],row['doc_ids']]+[
            f'{row[key]:.6f}' for key in ('C_closed','C_flat','J_full','J_flat','delta_J_flat_minus_full')])+'|')
    lines+=['','ΔJ=J_flat−J_full。实际闭包系数有来源支持时为0.225，无来源通道时按原算法变为0。',
        '包候选库的共同截断排序保留Full目标，阶段标为package_bank_control_full；'
        '这用于固定四cell的动作库。实际initial/frontier/expansion/reader_visible阶段使用各自目标，Flat/组合组均使用J_flat。', '',
        '## 包生成、扩展和最终路径','',
        '|数据集|条件|已诊断题数|多篇bank包累计|包尝试|包新增唯一状态|真正多篇增长新增|最终首发现经过多篇包|最终仅包含多篇包|平均扩展数|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in packages:
        lines.append('|'+ '|'.join(str(row[k]) if k!='mean_expanded_states' else f'{row[k]:.2f}' for k in (
            'dataset','variant','n','capped_multi_passage_packages','attempted','new_unique',
            'multi_passage_growth_new_unique','first_discovery_multi_package_chosen','contains_multi_package',
            'mean_expanded_states'))+'|')
    lines+=['','首发现路径、后续重复到达的多篇包路径、最终集合包含包分别统计，不能互相当作因果贡献。'
        'Singleton/组合组仍构建共同bank以保持控制输入，但实际包动作数及包扩展必须为0。', '',
        '## 搜索耗时（毫秒）','',
        '|数据集|条件|题数|有界搜索均值|P50|P95|包生成均值|诊断开销均值|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for ds,cellmap in aggregates.items():
        for arm,label in zip(CELLS,LABELS):
            cell=cellmap[arm];t=cell['timing'];search=t.get('bounded_search_seconds',{})
            ms=lambda v:'N/A' if v is None else f'{1000*v:.3f}'
            lines.append('|'+ '|'.join([ds,label,str(cell['queries']),ms(search.get('mean')),
                ms(search.get('p50')),ms(search.get('p95')),ms(t.get('package_generation_seconds',{}).get('mean')),
                ms(t.get('diagnostic_seconds',{}).get('mean'))])+'|')
    lines+=['','有界搜索、包生成和bank排序的计时已扣除直接观测器开销，但仍可能受诊断对象保留、GC、固定执行次序和CPU调度影响。'
        '共享残余特征与上下文缓存使Full-first的setup/总时间与后续cell不同，'
        '不能单用algorithm_seconds得出工程加速结论；原始历史native selector耗时也保留在CSV中，属于独立观测。', '',
        '## 观测器关闭、次序平衡的核心计时', '',
        '每数据集按sample_id的SHA确定性选32题，不读取质量或耗时结果。Full与Singleton各预热一次，'
        '再按F-S-S-F或S-F-F-S测四轮。状态准备、gc.collect、历史一致性核对在计时外；'
        '完整JSON保留缓存未命中，纯热缓存配对对比排除存在未命中的题。'
        '这是热特征选择器内核计时，不能当作冷缓存或部署端到端延迟。', '',
        '|数据集|条件|样本题数|测量次数|热缓存测量数|壁钟均值ms|进程CPU均值ms|配对Full−Singleton均值ms|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in observer_off:
        ms=lambda v:'N/A' if v is None else f'{v:.3f}'
        lines.append('|'+ '|'.join([row['dataset'],row['variant'],str(row['sample_count']),
            str(row['measurement_count']),str(row['fully_warm_measurement_count']),ms(row.get('fully_warm_wall_seconds_mean_ms')),
            ms(row.get('fully_warm_process_cpu_seconds_mean_ms')),ms(row.get('paired_full_minus_singleton_wall_seconds_mean_ms'))])+'|')
    lines+=['',
        '组合消融的2×2检索、QA、DCC及交互差值见 [RESULTS.md](../RESULTS.md)。'
        '四条件仍共享已有结构化S₀，结论只针对联合搜索内的闭包评分与包动作。']
    target=output/'DIAGNOSTICS.md';temporary=target.with_name(target.name+'.tmp')
    temporary.write_text('\n'.join(lines)+'\n');temporary.replace(target)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    refresh(parser.parse_args().root)

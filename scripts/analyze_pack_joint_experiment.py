#!/usr/bin/env python3
"""Paired quality/cost analysis; preserve negative results and both references."""
from pathlib import Path
import argparse,json,math,sys
PROJECT=Path('/root/BindPack');sys.path.insert(0,str(PROJECT/'scripts'))
from capture_joint_inputs import read,write,DATASETS

def exact_p(gain,loss):
    n=gain+loss
    return 1.0 if not n else min(1.,2*sum(math.comb(n,i) for i in range(min(gain,loss)+1))/2**n)

def analyze(root):
    retrieval_priority=(root/'retrieval_priority_update.json').exists()
    results={};lines=['# Pack新方案：配对效果与开销','',
        '所有值为0–1小数。pilot100与validation100互不重叠；原核验使用同一次上游作为正对照。',
        '这是开发诊断，不能当未调参的论文主实验。新增Pack方法不调用模型。'+
        ('后续按用户要求停止新QA生成，未完成整批QA的回答指标为N/A；此前答案及成本保留。' if retrieval_priority else 'Reader生成或在严格相同prompt时复用。'),'']
    for phase in ['pilot','validation']:
        lines += [f'## {phase}', '', '|数据集|方案|Recall@5|FCR@5|EM/Accuracy|F1/Macro-F1|NodeCov@5|DCC@5|ClosureGap@5|',
                  '|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for ds in DATASETS:
            base_path=root/'cases'/ds/(phase+'_baseline')
            if not (base_path/'completed.ok').exists():continue
            baseline=read(base_path/'report.json')
            for folder in sorted((root/'cases'/ds).glob(phase+'_*')):
                if not (folder/'completed.ok').exists():continue
                report=read(folder/'report.json');m=report['metrics'];arm=read(folder/'protocol.json')['variant']
                keys=['Recall@5','FCR@5','ClassificationAccuracy' if ds=='hover' else 'ExactMatch',
                      'ClassificationMacroF1' if ds=='hover' else 'F1','NodeCov@5','DCC@5','ClosureGap@5']
                lines.append('|'+ '|'.join([ds,arm]+['N/A' if m.get(k) is None else f'{m[k]:.6f}' for k in keys])+'|')
                entry={'metrics':{k:m.get(k) for k in keys},'versus':{}}
                own=read(folder/'metrics_details.json')['per_query'];own={q['sample_id']:q for q in own}
                for reference in ['baseline','review']:
                    ref=root/'cases'/ds/(phase+'_'+reference)
                    if not (ref/'completed.ok').exists():continue
                    refm=read(ref/'report.json')['metrics'];old=read(ref/'metrics_details.json')['per_query']
                    assert set(own)=={q['sample_id'] for q in old}
                    gains=sum(own[q['sample_id']]['at_k']['5']['FCR']>q['at_k']['5']['FCR'] for q in old)
                    losses=sum(own[q['sample_id']]['at_k']['5']['FCR']<q['at_k']['5']['FCR'] for q in old)
                    delta={k:None if m.get(k) is None or refm.get(k) is None else m[k]-refm[k] for k in keys}
                    safe=all(v>=-1e-10 for k,v in delta.items() if k in {'Recall@5','FCR@5'}) if retrieval_priority else all(v is None or (v<=1e-10 if k=='ClosureGap@5' else v>=-1e-10) for k,v in delta.items())
                    entry['versus'][reference]={'delta':delta,'nonregressing_all_reported_metrics':safe,
                        'FCR_gains':gains,'FCR_losses':losses,'FCR_exact_p':exact_p(gains,losses),
                        'p_scope':'descriptive unadjusted paired exact test; no multiple-comparison correction'}
                entry['pack_seconds']=report['efficiency']['pack_seconds']
                entry['isolated_pack_replay_seconds']=report['efficiency'].get('isolated_pack_replay_seconds')
                entry['window_query_count']=report['efficiency'].get('window_query_count',0)
                entry['cost']=report['cost'];results[f'{phase}/{ds}/{arm}']=entry
    lines += ['','## Pack开销与Reader预算','',
        '|阶段/数据集/方案|Pack均值秒|Pack P95秒|核验逻辑请求|本轮新Reader请求|相同prompt复用答案|',
        '|---|---:|---:|---:|---:|---:|']
    for key,r in results.items():
        c=r['cost'];t=r['pack_seconds']
        suffix='*' if r.get('window_query_count',0) else ''
        lines.append(f"|{key}|{t['mean']:.4f}{suffix}|{t['p95']:.4f}{suffix}|{c['pack_model_calls']}|{c['new_reader_http_attempts']}|{c['answers_reused_identical_prompt']}|")
    lines += ['','Pack开销包括选包及实际Reader预算下的来源可见性检查。原核验正对照还包括其模型等待/生成。',
        '新Reader请求量是本轮研究实际开销，不能当方法部署时的总请求数；相同prompt复用已分列。',
        '本表使用各case原100题组件计时；额外固定20题CPU回放仅用于排名一致性审计，另存report。未测完整Route到Reader端到端加速。共享GPU采样不代表方法独占成本。',
        '原核验存在保留的串行前缀与原生8题窗口；窗口均摊值不是单题延迟，须以各report的pack_time_scope与window_query_count为准，不能混合计算加速比。']
    if (root/'decision.json').exists():
        d=read(root/'decision.json')
        lines += ['','## 决策','',f"两阶段均通过全数据集{'Recall/FCR' if retrieval_priority else '原'}门槛的候选：`{d['validated_global_candidates']}`。",'生产默认没有自动修改；是否追回原核验收益需同时看上面对review的配对差值。']
    write(root/'analysis.json',results);(root/'ANALYSIS.md').write_text('\n'.join(lines)+'\n')
    if retrieval_priority:
        compact=['# Recall@5与FCR@5优先：两批100题配对结果','',
            '按用户后续要求，只有Recall@5与FCR@5参与筛选；不因EM/F1下降淘汰。',
            '每个数据集pilot100与validation100互不重叠。所有方法共用同题上游，权重0.60。','',
            '|阶段|数据集|方案|Recall@5|FCR@5|ΔRecall对关闭版|ΔFCR对关闭版|ΔRecall对原核验|ΔFCR对原核验|',
            '|---|---|---|---:|---:|---:|---:|---:|---:|']
        for key,value in results.items():
            phase,ds,arm=key.split('/')
            vals=[value['metrics']['Recall@5'],value['metrics']['FCR@5']]
            for ref in ['baseline','review']:
                d=value['versus'].get(ref,{}).get('delta',{})
                vals.extend([d.get('Recall@5'),d.get('FCR@5')])
            compact.append('|'+ '|'.join([phase,ds,arm]+['N/A' if x is None else f'{x:.6f}' for x in vals])+'|')
        (root/'RETRIEVAL_RESULTS.md').write_text('\n'.join(compact)+'\n')
        if (root/'completed.ok').exists():
            conclude(root,results)
    print('ANALYSIS',len(results),'completed cases',flush=True)

def conclude(root,results):
    """Apply the declared two-cohort, four-dataset gate without QA selection."""
    candidates=sorted({key.split('/')[-1] for key in results}-{'baseline','review'})
    qualified=[];assessments={}
    for arm in candidates:
        keys=[f'{phase}/{ds}/{arm}' for phase in ['pilot','validation'] for ds in DATASETS]
        complete=all(key in results for key in keys)
        safe={ref:complete and all(results[key]['versus'][ref]['nonregressing_all_reported_metrics']
                                  for key in keys) for ref in ['baseline','review']}
        phase_gains={phase:any(any((results.get(f'{phase}/{ds}/{arm}',{}).get('versus',{}).get('baseline',{}).get('delta',{}).get(k) or 0)>1e-10
                                  for k in ['Recall@5','FCR@5']) for ds in DATASETS)
                     for phase in ['pilot','validation']}
        qualifies=safe['baseline'] and all(phase_gains.values())
        assessment={'complete_two_cohorts':complete,'nonregressing':safe,
                    'gain_in_phase':phase_gains,'qualified_vs_no_review':qualifies,
                    'recovers_review_on_every_cohort_and_dataset':safe['review']}
        if complete:
            assessment['macro_FCR']=sum(results[key]['metrics']['FCR@5'] for key in keys)/len(keys)
            assessment['macro_Recall']=sum(results[key]['metrics']['Recall@5'] for key in keys)/len(keys)
            assessment['mean_pack_cpu_seconds']=sum(results[key]['pack_seconds']['mean'] for key in keys)/len(keys)
        assessments[arm]=assessment
        if qualifies:qualified.append(arm)
    qualified.sort(key=lambda arm:(not assessments[arm]['recovers_review_on_every_cohort_and_dataset'],
        -assessments[arm]['macro_FCR'],-assessments[arm]['macro_Recall'],assessments[arm]['mean_pack_cpu_seconds'],arm))
    chosen=qualified[0] if qualified else None
    complete_candidates=[arm for arm,a in assessments.items() if a['complete_two_cohorts']]
    descriptive=min(complete_candidates,key=lambda arm:(-assessments[arm]['macro_FCR'],
        -assessments[arm]['macro_Recall'],assessments[arm]['mean_pack_cpu_seconds'],arm)) if complete_candidates else None
    value={'primary_metrics':['Recall@5','FCR@5'],'qa_is_selection_gate':False,
        'qualified_candidates':qualified,'recommended_candidate':chosen,
        'descriptive_macro_leader':descriptive,'assessments':assessments,
        'production_default_changed':False,'scope':'development diagnosis on two disjoint cohorts; not an untouched paper test'}
    write(root/'retrieval_conclusion.json',value)
    lines=['# Recall/FCR优先：最终结论','',
        '每个数据集两批互不重叠100题，共800道不同问题。'+
        (f'通过保留门槛的首选：`{chosen}`。' if chosen else '没有统一候选通过保留门槛，因此不切换默认。'),
        '筛选要求：相对关闭末端核验的当前Pack，两项指标在四数据集、两批中均不下降，每批至少一个指标提升。另独立检查是否追回原末端核验。',
        'QA没有参与保留门槛；后续未运行的QA为N/A。权重0.60，所有新方法Pack的LLM/embedding新增调用均为0。',
        '这两批属于开发诊断；反复比较后的胜出结果不构成独立论文测试。生产默认仍为current。','',
        '|候选|两批齐全|对关闭版全通过|对原核验全不退|', '|---|---|---|---|']
    for arm,a in assessments.items():
        lines.append(f"|{arm}|{a['complete_two_cohorts']}|{a['qualified_vs_no_review']}|{a['recovers_review_on_every_cohort_and_dataset']}|")
    displayed=chosen or descriptive
    if chosen is None and descriptive:
        lines+=['',f'下面同时展示宏平均表现最好的诊断候选 `{descriptive}`；它没有通过逐批全数据集门槛，不能据合并均值采用。']
    arms=['baseline','review']+([displayed] if displayed else [])
    lines+=['','## 两批合并，仅用于描述','',
        '|数据集|方案|题数|Recall@5|FCR@5|FCR较关闭版净增完整题|', '|---|---|---:|---:|---:|---:|']
    for ds in DATASETS:
        for arm in arms:
            values=[results[f'{phase}/{ds}/{arm}'] for phase in ['pilot','validation']]
            recall=sum(v['metrics']['Recall@5'] for v in values)/2
            fcr=sum(v['metrics']['FCR@5'] for v in values)/2
            net=sum(v['versus']['baseline']['FCR_gains']-v['versus']['baseline']['FCR_losses'] for v in values)
            lines.append(f'|{ds}|{arm}|200|{recall:.6f}|{fcr:.6f}|{net:+d}|')
    lines+=['','## 四数据集等权平均，仅用于描述','',
            '|方案|Recall@5|FCR@5|', '|---|---:|---:|']
    for arm in arms:
        vals=[results[f'{phase}/{ds}/{arm}']['metrics'] for phase in ['pilot','validation'] for ds in DATASETS]
        lines.append(f"|{arm}|{sum(m['Recall@5'] for m in vals)/len(vals):.6f}|{sum(m['FCR@5'] for m in vals)/len(vals):.6f}|")
    lines+=['','各批逐项配对差值：[RETRIEVAL_RESULTS.md](RETRIEVAL_RESULTS.md)。完整指标、资源与成本：[ANALYSIS.md](ANALYSIS.md)。',
        '合并均值不能掩盖任意单批下降，资格判断使用上面的逐批门槛。显著性检验若报告，仅为未校正的描述性配对统计。',
        'Pack时间包括本地Reader上下文可见性检查。原核验的8题窗口均摊为吞吐指标，不能直接当作单题延迟或与新CPU均值相除声称端到端加速。']
    (root/'CONCLUSION.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();analyze(a.output)

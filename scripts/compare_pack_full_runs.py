#!/usr/bin/env python3
"""Read-only comparison of old no-review Full and the new joint Pack Full."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,sys
from pathlib import Path

PROJECT=Path('/root/BindPack')
sys.path.insert(0,str(PROJECT/'src'))
from bindpack.evaluation.structural_metrics import evaluate_query_metrics

DATASETS=('hotpotqa','2wikimultihopqa','musique','hover')
BEFORE=PROJECT/'outputs/4_datasets_result/full_no_review_20261026'
AFTER=PROJECT/'outputs/4_datasets_result/pack_nine_cases_20261027/new_scheme'
OUTPUT=AFTER.parent/'analysis/full_before_after'
KEYS=('Recall@5','FCR@5','ExactMatch','F1','ClassificationAccuracy','ClassificationMacroF1','NodeCov@5','DCC@5','ClosureGap@5')

def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def rows(p):
    with Path(p).open() as f:
        for line in f:
            item=json.loads(line);assert digest(item['row'])==item['sha256'];yield item['row']
def exact(g,l):
    n=g+l
    return 1.0 if not n else min(1.,2*sum(math.comb(n,i) for i in range(min(g,l)+1))/2**n)
def upstream(row):
    e=row['retrieval_trace']['evidence']
    return digest({k:e.get(k) for k in ('plan','bindings','routes','branch_scores')})
def validate_case(case):
    marker=read(case/'completed.ok')
    assert marker['query_count']==1000 and read(case/'audit.json')['status']=='passed'
    for name,suffix in [('report','.json'),('result','.json'),('llm_calls','.jsonl')]:
        assert sha(case/(name+suffix))==marker[name+'_sha256']
    report=read(case/'report.json');assert report['coverage']['sample_count']==1000
    return report
def summary(row):
    return {'sample_id':row['sample_id'],'query_index':row['query_index'],'question':row['question'],
            'upstream_sha256':upstream(row)}
def fmt(v):return 'N/A' if v is None else f'{v:.6f}'


def compare(output):
    output.mkdir(parents=True,exist_ok=True);result={'datasets':{},'pending':[],'model_calls':0,'method':'current 0.60 no-review Full versus joint_free 0.60 no-review Full'}
    for ds in DATASETS:
        old,new=BEFORE/'cases'/ds/'full',AFTER/'cases'/ds/'full'
        if not (new/'completed.ok').exists():result['pending'].append(ds);continue
        b,a=validate_case(old),validate_case(new);bp,ap=read(old/'protocol.json'),read(new/'protocol.json')
        for k in ('data_sha256','corpus_docs_sha256','selected_indices','sample_ids','oracle_hops','hop_provenance'):
            assert bp['dataset'][k]==ap['dataset'][k],(ds,k)
        assert bp['models']==ap['models'] and b['evaluator']==a['evaluator']
        for p in (bp,ap):assert p['configuration']['pack']['review_mode']=='off' and p['configuration']['pack']['relation_weight']==.6
        samples=read(AFTER/'runs'/ds/(ds+'.json'))
        corpus=read(AFTER/'runs'/ds/'metadata'/ds/'corpus.json')
        old_details={q['sample_id']:q for q in read(old/'metrics_details.json')['per_query']}
        new_details={q['sample_id']:q for q in read(new/'metrics_details.json')['per_query']}
        old_rows={r['sample_id']:summary(r) for r in rows(old/'progress.jsonl')}
        assert set(old_rows)==set(old_details)==set(new_details)
        paired=[];cf=[];actual=[];g=l=cg=cl=matched=0;selected_changed=0
        for row in rows(new/'progress.jsonl'):
            qid=row['sample_id'];old_row=old_rows[qid]
            assert (row['query_index'],row['question'])==(old_row['query_index'],old_row['question'])
            trace=row['retrieval_trace']['evidence'];selection=trace['budgeted_selection'];review=trace['improvement_support_semantic_veto']
            ids=review['final_doc_ids'];docs=row['candidate_docs'];assert len(ids)==len(docs)==200
            lookup=dict(zip(ids,docs));s0=[lookup[d] for d in selection['reference_doc_ids']]
            assert selection['selected_doc_ids']==ids[:5] and len(s0)==5
            base_row={'sample_id':qid,'query_index':row['query_index'],'candidate_docs':s0,'gold_docs':row['gold_docs']}
            ref=evaluate_query_metrics(base_row,samples[row['query_index']],ds,ks=(5,),corpus_docs=corpus)['at_k']['5']
            current=evaluate_query_metrics(dict(base_row,candidate_docs=docs[:5]),samples[row['query_index']],ds,ks=(5,),corpus_docs=corpus)['at_k']['5']
            saved=new_details[qid]['at_k']['5'];previous=old_details[qid]['at_k']['5']
            for k in ('Recall','FCR','NodeCov','DCC','ClosureGap'):
                assert current[k]==saved[k],(ds,qid,k)
            change=current['FCR']-previous['FCR'];counter=current['FCR']-ref['FCR']
            g+=change>0;l+=change<0;cg+=counter>0;cl+=counter<0
            same=upstream(row)==old_row['upstream_sha256'];matched+=same
            selected_changed+=s0!=docs[:5];cf.append(ref);actual.append(current)
            paired.append(dict(sample_id=qid,query_index=row['query_index'],before_Recall=previous['Recall'],after_Recall=current['Recall'],
                before_FCR=previous['FCR'],after_FCR=current['FCR'],same_run_reference_Recall=ref['Recall'],same_run_reference_FCR=ref['FCR'],
                recorded_upstream_matches=same,reference_changed=s0!=docs[:5]))
        assert len(paired)==1000
        def avg(items,k):
            v=[x[k] for x in items if x[k] is not None];return sum(v)/len(v) if v else None
        s0metrics={k+'@5':avg(cf,k) for k in ('Recall','FCR','NodeCov','DCC','ClosureGap')}
        for k in ('Recall','FCR','NodeCov','DCC','ClosureGap'):
            v=avg(actual,k);target=a['metrics'].get(k+'@5');assert v is target is None or (v is not None and target is not None and abs(v-target)<1e-10)
        value={'before_path':str(old),'after_path':str(new),'queries':1000,
            'before':{k:b['metrics'].get(k) for k in KEYS},'after':{k:a['metrics'].get(k) for k in KEYS},
            'delta':{k:None if a['metrics'].get(k) is None or b['metrics'].get(k) is None else a['metrics'][k]-b['metrics'][k] for k in KEYS},
            'paired_FCR':dict(gains=g,losses=l,net=g-l,exact_p_unadjusted=exact(g,l)),
            'same_run_reference':s0metrics,'same_run_FCR':dict(gains=cg,losses=cl,net=cg-cl,exact_p_unadjusted=exact(cg,cl)),
            'same_run_reference_changed_queries':selected_changed,'recorded_upstream_exact_matches':matched,
            'upstream_match_scope':'Exact plan, bindings, routes and branch_scores; not proof of identical full candidate state, which historical traces did not retain.',
            'efficiency':{'before':b['efficiency'],'after':a['efficiency']},'cost':{'before':b['cost'],'after':a['cost']},
            'resources':{'before':b['resources'],'after':a['resources']},'evidence_budget':{'before':b['evidence_budget'],'after':a['evidence_budget']},
            'fingerprints':{'before_report':sha(old/'report.json'),'after_report':sha(new/'report.json'),
                'before_details':sha(old/'metrics_details.json'),'after_details':sha(new/'metrics_details.json')},
            'same_run_QA':'N/A; only reference ranking is reconstructed; no counterfactual answer generation'}
        result['datasets'][ds]=value
        with (output/(ds+'_paired.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
        print('COMPARED',ds,'R/F delta',value['delta']['Recall@5'],value['delta']['FCR@5'],'same-run FCR',cg,cl,flush=True)
    count=len(result['datasets'])
    result['macro']={arm:{k:sum(x[arm][k] for x in result['datasets'].values())/count if count else None
                         for k in ('Recall@5','FCR@5')} for arm in ('before','after')}
    result['net_complete_questions_independent_runs']=sum(x['paired_FCR']['net'] for x in result['datasets'].values())
    result['net_complete_questions_same_run']=sum(x['same_run_FCR']['net'] for x in result['datasets'].values())
    write(output/'comparison.json',result)
    lines=['# 新旧Pack Full全量对比','',
        '旧版为full_no_review_20261026，本次为pack_nine_cases_20261027/new_scheme。两版均权重0.60、review_mode=off、相同Qwen3模型、原始passage Reader@5和四份复用索引。',
        '逐数据集核对相同1000题、文本/语料哈希、模型身份、oracle native hop及评测器。只读原始结果，不改运行中的算法或进程，不新增模型调用。输出仅为本分析的新文件。',
        f"已纳入正式完成结果：{len(result['datasets'])}/4；尚未完成：{', '.join(result['pending']) or '无'}。",'',
        '|数据集|旧Recall@5|新Recall@5|ΔRecall|旧FCR@5|新FCR@5|ΔFCR|完整题净增|', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for ds,x in result['datasets'].items():
        b,a,d=x['before'],x['after'],x['delta']
        lines.append('|'+ '|'.join([ds,*[fmt(v) for v in (b['Recall@5'],a['Recall@5'],d['Recall@5'],b['FCR@5'],a['FCR@5'],d['FCR@5'])],str(x['paired_FCR']['net'])])+'|')
    macro=result['macro']
    lines+=['',f"纳入数据集的等权平均：Recall@5 {fmt(macro['before']['Recall@5'])} → {fmt(macro['after']['Recall@5'])}；FCR@5 {fmt(macro['before']['FCR@5'])} → {fmt(macro['after']['FCR@5'])}。独立全量结果的完整证据题净变化为{result['net_complete_questions_independent_runs']:+d}。"]
    lines+=['','## 相同本次上游内：当前Pack参考S0与新选包','',
        '从本次trace的reference_doc_ids恢复S0五篇原文，用相同独立标注重新计算检索指标。它固定了本次上游，较独立旧/新全流程差值更接近末端选包的直接作用；未重算S0答案。',
        '|数据集|S0 Recall@5|新Recall@5|S0 FCR@5|新FCR@5|FCR改善/下降|', '|---|---:|---:|---:|---:|---:|']
    for ds,x in result['datasets'].items():
        s,a=x['same_run_reference'],x['after'];p=x['same_run_FCR']
        lines.append(f"|{ds}|{fmt(s['Recall@5'])}|{fmt(a['Recall@5'])}|{fmt(s['FCR@5'])}|{fmt(a['FCR@5'])}|{p['gains']}/{p['losses']}|")
    lines+=['','## 回答指标与依赖指标','',
        '|数据集|旧EM/Accuracy|新EM/Accuracy|旧F1/Macro-F1|新F1/Macro-F1|旧DCC@5|新DCC@5|旧ClosureGap@5|新ClosureGap@5|', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for ds,x in result['datasets'].items():
        b,a=x['before'],x['after'];q=('ClassificationAccuracy','ClassificationMacroF1') if ds=='hover' else ('ExactMatch','F1')
        lines.append('|'+ '|'.join([ds,*[fmt(v) for v in (b[q[0]],a[q[0]],b[q[1]],a[q[1]],b['DCC@5'],a['DCC@5'],b['ClosureGap@5'],a['ClosureGap@5'])]])+'|')
    lines+=['','## 时间与成本','',
        '|数据集|旧处理秒/题|新处理秒/题|旧case小时|新case小时|新选择器均值秒|新选择器P95秒|旧LLM逻辑请求|新LLM逻辑请求|', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for ds,x in result['datasets'].items():
        b,a=x['efficiency']['before'],x['efficiency']['after'];t=a['budgeted_selector_seconds'];c=x['cost']
        lines.append(f"|{ds}|{b['processing_seconds_per_query']:.3f}|{a['processing_seconds_per_query']:.3f}|{b['case_wall_seconds']/3600:.3f}|{a['case_wall_seconds']/3600:.3f}|{t['mean']:.3f}|{t['p95']:.3f}|{c['before']['llm_logical_calls']}|{c['after']['llm_logical_calls']}|")
    lines+=['','处理秒/题为窗口墙钟的吞吐均摊，不是隔离单题延迟。旧HotpotQA前781题串行、新版全程8题窗口，因此其整批时间下降主要受执行方式影响，不能归为Pack算法提速。其余数据集两版均为原生8题窗口，仍是单次实测而非重复计时结论。',
        '新选择器单独时间包括特征、搜索和Reader可见性检查，未包括参考S0构造；新增LLM/embedding请求为0，但CPU开销非0。新旧全流程调用数少量变化还涉及独立运行、缓存及失败尝试。',
        '旧2Wiki及新MuSiQue各有一次失败尝试的token usage未知，已保留；相关tokens总数仅为已知部分。原索引历史建库成本不可得时N/A，复用不记为原始建库免费。',
        '共享GPU采样不代表方法独占资源。全部成本、资源和证据预算、来源哈希及未校正配对统计见comparison.json。']
    (output/'BEFORE_AFTER.md').write_text('\n'.join(lines)+'\n')
    print('REPORT',output/'BEFORE_AFTER.md',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args();compare(a.output)

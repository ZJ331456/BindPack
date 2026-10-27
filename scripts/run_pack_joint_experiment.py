#!/usr/bin/env python3
"""Serial case matrix, fixed upstream, pilot100 then disjoint validation100."""
from pathlib import Path
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from dataclasses import asdict
import argparse, hashlib, json, os, subprocess, sys, threading, time

PROJECT=Path('/root/BindPack')
sys.path[:0]=[str(PROJECT/'src'),str(PROJECT/'scripts')]
from capture_joint_inputs import DATASETS, ARCHIVE, read, write, digest, token

ARMS=('baseline','review','closed','residual','risk','sources','joint_all','joint_free')
FEATURES={'closed':{},'residual':{'residual':True},'risk':{'risk':True},'sources':{'sources':True},
          'joint_all':{'residual':True,'risk':True,'sources':True},
          'joint_free':{'residual':True,'risk':True,'sources':True}}

def checkpoint(root,ds,phase,arm,qid):return root/'cases'/ds/(phase+'_'+arm)/'checkpoints'/token(qid)

def metrics(root,ds,phase,arm):
    m=read(root/'cases'/ds/(phase+'_'+arm)/'report.json')['metrics']
    if (root/'retrieval_priority_update.json').exists():return {k:m[k] for k in ['Recall@5','FCR@5']}
    keys=['Recall@5','FCR@5','ClassificationAccuracy' if ds=='hover' else 'ExactMatch',
          'ClassificationMacroF1' if ds=='hover' else 'F1']
    if ds=='musique':keys.append('DCC@5')
    return {k:m[k] for k in keys}

def case(root,ds,phase,arm):
    import numpy as np
    from transformers import AutoTokenizer
    from bindpack import BindPackConfig
    from bindpack.evaluation.measurement import Recorder,ResourceMonitor,distribution
    from bindpack.evaluation.measured_reader import prepare
    from bindpack.evaluation.result_export import detailed_result
    from bindpack.evaluation.metric_report import write_metric_report
    from bindpack.utils.misc_utils import QuerySolution
    from run_pack_weight_sweep import ReadOnlyPassages,make_runtime
    from run_pack_weight_diagnostic import restore_pack_input
    from eval_utils import get_gold_answers,get_gold_docs
    chosen=[c for c in read(root/'selection_locked.json')['datasets'][ds] if c['phase']==phase]
    output=root/'cases'/ds/(phase+'_'+arm);output.mkdir(parents=True,exist_ok=True)
    if (output/'completed.ok').exists():return
    retrieval_only=(root/'retrieval_priority_update.json').exists()
    resumed=any((output/'checkpoints').glob('*.json')) if (output/'checkpoints').exists() else False
    if not (output/'corpus.json').exists():(output/'corpus.json').symlink_to(root/'metadata'/ds/'corpus.json')
    # Retain prior failed/export-only attempts and their resource samples.
    if (output/'resources.json').exists():
        history=output/'resource_history';history.mkdir(exist_ok=True)
        import shutil
        shutil.copy2(output/'resources.json',history/(str(time.time_ns())+'.json'))
    cfg=BindPackConfig.from_file(**read(root/'capture'/ds/'configuration.json'))
    cfg.save_dir=str(root/'reader_cache'/ds);cfg.pack.review_mode='standard' if arm=='review' else 'off'
    cfg.pack.selector_mode='current' if arm in {'baseline','review'} else (arm if arm in {'closed','residual','risk','sources'} else 'combined')
    # A deterministic state cap controls this diagnostic. The generous emergency
    # deadline avoids making the chosen ranking depend on competing CPU load.
    cfg.pack.joint_search_seconds=5.0
    cfg.pack.joint_fixed_prefix=0 if arm=='joint_free' else 2
    if arm=='joint_free':cfg.pack.joint_state_cap=4096
    tokenizer=AutoTokenizer.from_pretrained('/root/models/Qwen3-8B',local_files_only=True)
    store=ReadOnlyPassages(next((PROJECT/'outputs/4_datasets_result/shared_indexes'/ds).rglob('vdb_chunk.parquet')))
    recorder=Recorder(output/'selection_accounting')
    rag=make_runtime(cfg,store,recorder,tokenizer)
    rag.evidence_runtime._joint_tokenizer=tokenizer
    selected_features=FEATURES.get(arm,read(root/'combination.json').get('features',{}) if (root/'combination.json').exists() else {})
    rag.evidence_runtime._joint_features=selected_features
    # Selection may tokenize/render, but must never generate or retrieve.
    def deny(*a,**kw):raise AssertionError('Pack issued a forbidden model request')
    if arm!='review':
        rag.llm_model.openai_client.chat.completions.create=deny
        rag.llm_model.infer=deny
    references=[json.loads(s)['row'] for s in (ARCHIVE/'cases'/ds/'full/progress.jsonl').open() if s.strip()]
    answers={r['reader']['budget']['prompt_sha256']:{'answer':r['answer'],'reader':r['reader'],
             'reuse_source':'completed_native_full'} for r in references}
    # Cross-arm answer reuse requires an identical ordered, rendered context.
    for p in (root/'cases'/ds).glob('*/checkpoints/*.json'):
        r=read(p)['row']
        if r.get('qa_executed',True):
            answers[r['reader']['budget']['prompt_sha256']]={'answer':r['answer'],
                'reader':r['reader'],'reuse_source':str(p)}
    benchmark=read(root/'metadata'/ds/'benchmark.json')
    state=threading.local(); readers=[]; readers_lock=threading.Lock()
    def answer(job):
        qid,solution,reading,budget,row0=job
        if not hasattr(state,'rag'):
            rec=Recorder(output/'reader_parts'/str(threading.get_ident()))
            state.rag=make_runtime(cfg,store,rec,tokenizer);state.rec=rec
            with readers_lock:readers.append(rec)
        with state.rec.section('reader',qid):
            begin=time.perf_counter();answered,raw,metadata=state.rag.qa([reading]);seconds=time.perf_counter()-begin
        row0.update(answer=answered[0].answer,reader={'raw_response':raw[0],'metadata':metadata[0],'budget':budget},
                    reader_reused=False,reader_seconds=seconds)
        write(checkpoint(root,ds,phase,arm,qid),{'row':row0,'sha256':digest(row0)})
        return qid
    started=time.perf_counter();pending=[]
    with ResourceMonitor(recorder) as monitor,ThreadPoolExecutor(max_workers=4) as pool:
        def commit(item,seconds,timing=None):
            n,c,saved,solution,ids,selection_seconds=item
            qid=c['sample_id'];cp=checkpoint(root,ds,phase,arm,qid)
            reading,budget=prepare(rag,solution,tokenizer,ds)
            sample=benchmark[c['query_index']]
            solution.gold_answers=get_gold_answers([sample],ds)[0]
            solution.gold_docs=sorted(set(get_gold_docs([sample],ds)[0]))
            row=detailed_result(solution,sample,c['query_index'],solution.gold_docs,c['benchmark_hops'],10,200)
            row.update(ranking_complete=True,actual_ranking_length=len(solution.docs),
                pack_seconds=seconds,pack_relation_weight=.6,variant=arm,phase=phase,
                pack_time_scope='serial_component' if timing is None else 'native_window_throughput_allocation',
                review_window=timing,frozen_input_sha256=saved['input_sha256'],pack_model_requests=
                solution.retrieval_trace['evidence']['improvement_support_semantic_veto'].get('extra_requests',0))
            reuse=answers.get(budget['prompt_sha256'])
            if retrieval_only:
                row.update(answer='',reader={'raw_response':None,'metadata':{},'budget':budget},
                           qa_executed=False,reader_reused=False,reader_seconds=None)
                write(cp,{'row':row,'sha256':digest(row)})
            elif reuse:
                row.update(answer=reuse['answer'],reader=deepcopy(reuse['reader']),reader_reused=True,
                    answer_reuse_source=reuse['reuse_source'],reader_seconds=None)
                row['reader']['budget']=budget
                write(cp,{'row':row,'sha256':digest(row)})
            else:pending.append(pool.submit(answer,(qid,solution,reading,budget,row)))
            if len(pending)>=8:pending.pop(0).result()
            if n%10==0 or n==100:
                write(root/'status.json',{'dataset':ds,'phase':phase,'arm':arm,'selected':n,'total':100})
                print(f'CASE {phase} {ds}/{arm} {n}/100; pack={seconds:.3f}s',flush=True)
        review_batch=[]
        def flush_review():
            if not review_batch:return
            wid=f'{ds}:{phase}:{time.time_ns()}'
            begin=time.perf_counter()
            with recorder.section('terminal_review',wid):
                rag.evidence_runtime.pack.postprocess([(it[3],it[4][:200].copy()) for it in review_batch])
            wall=time.perf_counter()-begin
            timing={'id':wid,'size':len(review_batch),'wall_seconds':wall,
                    'query_ids':[it[1]['sample_id'] for it in review_batch]}
            recorder.append('review_windows.jsonl',timing)
            for it in review_batch:commit(it,it[5]+wall/len(review_batch),timing)
            review_batch.clear()
        for n,c in enumerate(chosen,1):
            qid=c['sample_id'];cp=checkpoint(root,ds,phase,arm,qid)
            if cp.exists():continue
            saved=read(root/'frozen_inputs'/ds/token(qid));original=saved['input']
            assert digest(original)==saved['input_sha256']
            frozen=restore_pack_input(original);begin=time.perf_counter()
            ids,scores,trace=rag.evidence_runtime.pack.finalize(original['query'],np.asarray(original['ids'],int),
                    np.asarray(original['scores'],float),{},frozen)
            seconds=time.perf_counter()-begin
            if arm=='baseline':assert ids[:200].tolist()==saved['baseline_doc_ids']
            solution=QuerySolution(question=c['question'],docs=[rag.evidence_runtime._document(int(i)) for i in ids[:200]],
                    doc_scores=scores[:200],retrieval_trace={'stage_ablation':'full','evidence':trace})
            item=(n,c,saved,solution,ids,seconds)
            if arm=='review':
                review_batch.append(item)
                if len(review_batch)==8:flush_review()
            else:
                rag.evidence_runtime.pack.postprocess([(solution,ids[:200].copy())])
                commit(item,time.perf_counter()-begin)
        flush_review()
        for f in pending:f.result()
        write(output/'resources.json',monitor.summary())
    rows=[read(checkpoint(root,ds,phase,arm,c['sample_id']))['row'] for c in chosen]
    assert len(rows)==100 and len({r['sample_id'] for r in rows})==100
    manifest=read(ARCHIVE/'runs'/ds/'metadata'/ds/'sample_manifest.json')
    manifest.update(dataset=ds,data_path=str(root/'metadata'/ds/'benchmark.json'),
        selected_indices=[c['query_index'] for c in chosen],sample_ids=[c['sample_id'] for c in chosen],
        diagnostic_selection=True,selection_sha256=read(root/'selection_locked.json')['sha256'])
    write(output/'sample_manifest.json',manifest)
    protocol={'dataset':ds,'variant':arm,'phase':phase,'features':selected_features,
        'configuration':asdict(cfg),'index_rebuilt':False,'route_plan_shared':True,
        'pack_model_calls':sum(r['pack_model_requests'] for r in rows),'candidate_domain':'same captured evidence_candidates union baseline S0, within parent Top200',
        'cost_scope':'Pack CPU and necessary Reader; capture cost separately; no end-to-end speed claim',
        'reader_reuse':'only identical ordered rendered prompt hash','isolated_validation':phase=='validation',
        'primary_metrics':['Recall@5','FCR@5'] if retrieval_only else 'original gate',
        'control_review_policy':'native windows up to8; preserved serial prefix remains separately labeled' if arm=='review' else None}
    write(output/'protocol.json',protocol)
    qa_complete=all(r.get('qa_executed',True) for r in rows)
    result={'dataset':ds,'method':'bindpack','stage_ablation':'full','eval_mode':'rag_qa' if qa_complete else 'retrieve',
        'data_path':str(root/'metadata'/ds/'benchmark.json'),'corpus_path':str(root/'metadata'/ds/'corpus.json'),
        'selected_indices':manifest['selected_indices'],'sample_size_effective':100,'diagnostic_selection':True,
        'runtime_config':asdict(cfg),'results':rows}
    write(output/'result.json',result)
    result_sha=hashlib.sha256((output/'result.json').read_bytes()).hexdigest()
    write(output/'validated.ok',{'result_sha256':result_sha,'sample_count':100,'protocol_sha256':digest(protocol)})
    report=write_metric_report(output/'result.json',data_path=result['data_path'],manifest_path=output/'sample_manifest.json',ks=(3,5,10,20,200))
    calls=[]
    for p in (output/'reader_parts').glob('*/llm_calls.jsonl'):
        calls.extend(json.loads(s) for s in p.open() if s.strip())
    review_calls=[]
    p=output/'selection_accounting/llm_calls.jsonl'
    if p.exists():review_calls=[json.loads(s) for s in p.open() if s.strip()]
    windows={r['review_window']['id']:r['review_window'] for r in rows if r.get('review_window')}
    report.update(efficiency={'pack_seconds':distribution([r['pack_seconds'] for r in rows]),
        'pack_time_scope':'mixed serial prefix and window throughput allocations; not isolated query latency' if windows else 'serial component intervals',
        'serial_pack_seconds':distribution([r['pack_seconds'] for r in rows if not r.get('review_window')]),
        'review_window_wall_seconds':distribution([w['wall_seconds'] for w in windows.values()]),
        'window_query_count':sum(r.get('review_window') is not None for r in rows),
        'fresh_reader_seconds':distribution([r['reader_seconds'] for r in rows if r['reader_seconds'] is not None]),
        'case_wall_seconds':None if resumed else time.perf_counter()-started,
        'this_attempt_wall_seconds':time.perf_counter()-started,'resumed_from_checkpoints':resumed,'reader_concurrency':4},
        cost={'pack_model_calls':sum(r['pack_model_requests'] for r in rows),'pack_embedding_calls':0,
            'review_http_attempts':len(review_calls),'review_input_tokens':sum(c.get('prompt_tokens') or 0 for c in review_calls),
            'review_output_tokens':sum(c.get('completion_tokens') or 0 for c in review_calls),'new_reader_http_attempts':len(calls),
            'reader_input_tokens':sum(c.get('prompt_tokens') or 0 for c in calls),
            'reader_output_tokens':sum(c.get('completion_tokens') or 0 for c in calls),
            'answers_reused_identical_prompt':sum(r['reader_reused'] for r in rows),
            'reader_length_outputs':sum(r['reader']['metadata'].get('finish_reason')=='length' for r in rows)},
        evidence_budget={'reader_top_k':5,'mean_evidence_tokens':sum(r['reader']['budget']['evidence_tokens'] for r in rows)/100},
        resources=read(output/'resources.json'))
    write(output/'report.json',report)
    write(output/'completed.ok',{'queries':100,'result_sha256':result_sha})
    print('REPORT',phase,ds,arm,json.dumps({k:report['metrics'].get(k) for k in ['Recall@5','FCR@5','ExactMatch','F1','ClassificationAccuracy','ClassificationMacroF1','DCC@5']}),flush=True)

def assess(root,phase,arm):
    result={};improve=False;safe=True
    for ds in DATASETS:
        p=root/'cases'/ds/(phase+'_'+arm)/'report.json'
        if not p.exists():continue
        a=metrics(root,ds,phase,'baseline');b=metrics(root,ds,phase,arm)
        delta={k:b[k]-a[k] for k in a}
        ok=all(v>=-1e-10 for v in delta.values());gain=any(v>1e-10 for v in delta.values())
        result[ds]={'delta':delta,'nonregressing':ok,'improved':gain,'retain':ok and gain}
        safe &= ok;improve |= gain
    return {'datasets':result,'retain_globally':safe and improve and len(result)==4,
            'promising_any_dataset':any(v['retain'] for v in result.values())}

def summary(root):
    lines=['# Pack逐项实现：固定上游100题与独立后续100题','',
        '权重0.60，末端核验off；上游只共享执行一次。诊断数据不作为独立论文测试。','',
        '|阶段|数据集|方案|Recall@5|FCR@5|EM/Accuracy|F1/Macro-F1|DCC@5|Pack均值秒|新增Reader请求|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for phase in ['pilot','validation']:
        for ds in DATASETS:
            for arm in list(ARMS)+['combined']:
                p=root/'cases'/ds/(phase+'_'+arm)/'report.json'
                if not p.exists():continue
                r=read(p);m=r['metrics'];ks=['Recall@5','FCR@5','ClassificationAccuracy' if ds=='hover' else 'ExactMatch','ClassificationMacroF1' if ds=='hover' else 'F1','DCC@5']
                vals=['N/A' if m.get(k) is None else f'{m[k]:.6f}' for k in ks]
                lines.append('|'+ '|'.join([phase,ds,arm]+vals+[f"{r['efficiency']['pack_seconds']['mean']:.4f}",str(r['cost']['new_reader_http_attempts'])])+'|')
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')

def main(root):
    from capture_joint_inputs import prepare
    prepare(root)
    policy={'pilot':100,'validation':100,'cohorts_disjoint':True,'relation_weight':.6,'review_mode':'off',
        'pilot_gate':'No Recall/FCR/QA-primary/correctness/DCC regression within a dataset and at least one gain.',
        'validation_candidates':'Any individual arm passing the pilot gate on at least one dataset is tested on all four validation cohorts; never treated as globally retained on pilot alone.',
        'combination':'Combine only directions nonregressing on all four pilot datasets; require gain; default strategy remains current until validation.',
        'no_outcome_based_sampling':True,'production_default_unchanged':True}
    if (root/'experiment.json').exists():assert read(root/'experiment.json')==policy
    else:write(root/'experiment.json',policy)
    env=os.environ.copy()
    for ds in DATASETS:
        if not (root/'capture'/ds/'completed.ok').exists():
            subprocess.run([sys.executable,str(PROJECT/'scripts/capture_joint_inputs.py'),'--output',str(root),'--dataset',ds],check=True,env=env)
    for arm in ARMS:
        for ds in DATASETS:case(root,ds,'pilot',arm);summary(root)
    decisions={arm:assess(root,'pilot',arm) for arm in FEATURES}
    retained=[arm for arm,d in decisions.items() if arm not in {'joint_all','joint_free'} and d['retain_globally']]
    merged={k:True for arm in retained for k,v in FEATURES[arm].items() if v}
    write(root/'combination.json',{'retained_directions':retained,'features':merged})
    candidates=[arm for arm,d in decisions.items() if d['promising_any_dataset']]
    if len(retained)>1:
        for ds in DATASETS:case(root,ds,'pilot','combined')
        decisions['combined']=assess(root,'pilot','combined')
        if decisions['combined']['promising_any_dataset']:candidates.append('combined')
    write(root/'pilot_decisions.json',decisions);summary(root)
    if candidates:
        for ds in DATASETS:
            case(root,ds,'validation','baseline');case(root,ds,'validation','review')
        for arm in candidates:
            for ds in DATASETS:case(root,ds,'validation',arm);summary(root)
    validations={arm:assess(root,'validation',arm) for arm in candidates}
    promoted=[arm for arm,d in validations.items() if d['retain_globally'] and decisions[arm]['retain_globally']]
    write(root/'decision.json',{'pilot':decisions,'validation':validations,'validated_global_candidates':promoted,
        'production_default_changed':False,'note':'Keep all outcomes including regressions; no automatic production promotion.'})
    summary(root);write(root/'completed.ok',{'pilot_unique_questions':400,'validation_executed':bool(candidates),'validated_global_candidates':promoted})
    print('COMPLETE',str(root/'RESULTS.md'),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    main(a.output.resolve())

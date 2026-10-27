#!/usr/bin/env python3
"""Locked old50 per dataset: single changes, predeclared screening, combination."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime
import argparse
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PROJECT=Path('/root/BindPack')
BASE=PROJECT/'outputs/pack_weight_diagnostic_20261025'
DEFAULT=PROJECT/'outputs/component_diagnostic_r060_20261026'
DATASETS=('hotpotqa','2wikimultihopqa','musique','hover')
ARMS=('coverage_facts','plan_repair','chain_package','route_fusion','hover_reader')
FEATURES={
    'coverage_facts':'Canonical coverage facts only; unchanged relevance scores, R=.60 and redundancy.',
    'plan_repair':'After both original planning attempts fail, canonicalize existing literal dependencies and revalidate; execute recovered plan from the same pre-Plan candidates.',
    'chain_package':'Bounded complete winning source chain; exact quotes/types/branch, confidence>=.85, anchors and strict supports retained; relevance loss bounded by existing closure weight.',
    'route_fusion':'Original PathSet fusion may boost, but cannot lower a selected path document below its pre-PathSet score.',
    'hover_reader':'Atomic claim and cross-passage reasoning prompt; original two labels, passages, model, budget and parser unchanged.'}
SCRIPT=Path(__file__).resolve()


def read(path):return json.loads(Path(path).read_text())


def digest(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def load_sweep(output):
    p=output/'source_snapshot/scripts/run_pack_weight_sweep.py'
    s=importlib.util.spec_from_file_location('component_sweep_helpers',p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def prepare(output):
    lock=read(BASE/'selection_locked.ok');selection=read(BASE/'selection_manifest.json')
    assert digest(selection)==lock['sha256']
    if (output/'prepared.ok').exists():
        assert read(output/'selection_locked.ok')==lock;return
    output.mkdir(parents=True,exist_ok=True)
    write(output/'selection_manifest.json',selection);write(output/'selection_locked.ok',lock)
    write(output/'experiment.json',dict(schema_version=1,relation_weight=.6,arms=FEATURES,
        selection_sha256=lock['sha256'],questions_per_dataset=50,total_questions=200,
        diagnostic_only=True,index_rebuild=False,serial_execution=True,
        selection_rule='Equal dataset mean deltas of QA F1 (HoVer Macro-F1), EM/Accuracy and FCR@5 must all be >=0, with at least one >0. Mixed tradeoffs are recorded but not combined.',
        baseline='Fresh unchanged Full=.60 for each question; original .60 cache-only parity required first. All Pack/Route/reader variants share its frozen inputs; Plan repair shares its pre-Plan state.',
        original_results={ds:sha(BASE/'cases'/ds/'full_r060/result.json') for ds in DATASETS},
        hop_policy='Original old50 historical protocol: HotpotQA/2Wiki prior=2; MuSiQue/HoVer original oracle hints. No hop change in this test.',
        models=dict(llm='/root/models/Qwen3-8B',embedding='/root/models/Qwen3-Embedding-8B'),
        production_policy='Only a verified final candidate without any dataset-level QA/EM/FCR regression may be promoted; otherwise leave the baseline method and export the candidate.' ))
    for ds in DATASETS:
        meta=output/'metadata'/ds;meta.mkdir(parents=True,exist_ok=True)
        for name in ('benchmark.json','corpus.json'):
            p=meta/name
            if not p.exists():p.symlink_to(os.path.relpath(BASE/'metadata'/ds/name,meta))
        identity=read(BASE/'metadata'/ds/'sample_manifest.json');identity['data_path']=str(meta/'benchmark.json')
        write(meta/'sample_manifest.json',identity)
        method=read(BASE/'metadata'/ds/'full_r060_method.json');method['runtime']['save_dir']=str(output/'runtime'/ds/'index')
        assert method['pack']['relation_weight']==.6
        write(meta/'method.json',method)
    snapshot=output/'source_snapshot'
    shutil.copytree(BASE/'source_snapshot/src',snapshot/'src',dirs_exist_ok=True,
        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (snapshot/'scripts').mkdir(parents=True,exist_ok=True)
    for name in (SCRIPT.name,'diagnostic_improvements.py','run_pack_weight_sweep.py',
                 'finalize_component_diagnostic.py','check_component_production.py'):
        shutil.copy2(PROJECT/'scripts'/name,snapshot/'scripts'/name)
    write(output/'source_fingerprint.json',{str(p.relative_to(snapshot)):sha(p) for p in snapshot.rglob('*.py')})
    write(output/'production_base_hashes.json',{str(p.relative_to(PROJECT)):sha(p) for p in (PROJECT/'src/bindpack').rglob('*.py')})
    (output/'README.md').write_text('# 固定0.60的四项诊断\n\n原始每个数据集50题，共200题，样本锁不变。第4项分为Route和HoVer reader两个单项，因此总共5个单项，然后自动筛选并测试组合。\n\n'
        '索引全部复用，模型和并行参数不变。基线先对原0.60完成缓存校验，再建立统一新基线和Route/Plan输入快照。每题断点保存。\n\n'
        '筛选规则在experiment.json中预先锁定；诊断集不是正式测试结论。新100条/数据集不参与本次筛选。\n\n'
        f'日志：`tail -f {output}/run.log`\n\n结果自动写入comparison.md、selection_decision.json和各case的report.json、metrics_details.json。\n')
    write(output/'prepared.ok',dict(selection_sha256=lock['sha256'],new_requests=0))


def load_environment(output):
    sys.path.insert(0,str(output/'source_snapshot/src'));sys.path.insert(0,str(output/'source_snapshot/scripts'))
    os.environ.setdefault('OPENAI_API_KEY','EMPTY')


def row_path(output,ds,arm,qid):
    return output/'cases'/ds/arm/'checkpoints'/(hashlib.sha256(str(qid).encode()).hexdigest()+'.json')


def completed_rows(output,ds,arm):
    selected=read(output/'selection_manifest.json')['datasets'][ds]['rows'];rows=[]
    for q in selected:
        p=row_path(output,ds,arm,q['sample_id'])
        if not p.exists():break
        v=read(p);assert digest(v['row'])==v['sha256'] and v['row']['sample_id']==q['sample_id'];rows.append(v['row'])
    return rows


def export_answer(rag,solution,tokenizer,recorder,ds,chosen,sample,arm,helper,features,fingerprint):
    from bindpack.evaluation.measured_reader import prepare as prepare_reader
    from bindpack.evaluation.result_export import detailed_result
    from eval_utils import get_gold_answers,get_gold_docs
    with recorder.section('reader_'+arm,chosen['sample_id']):
        t=time.perf_counter();reading,budget=prepare_reader(rag,solution,tokenizer,ds)
        answered,raw,metas=rag.qa([reading]);solution.answer=answered[0].answer;seconds=time.perf_counter()-t
    solution.gold_answers=get_gold_answers([sample],ds)[0]
    solution.gold_docs=sorted(set(get_gold_docs([sample],ds)[0]))
    result=detailed_result(solution,sample,chosen['query_index'],solution.gold_docs,chosen['benchmark_hops'],10,200)
    result.update(ranking_complete=True,actual_ranking_length=len(solution.docs),pack_relation_weight=.6,
        experimental_variant=arm,experimental_features=list(features),frozen_pack_input_sha256=fingerprint,
        reader=dict(raw_response=raw[0],metadata=metas[0],budget=budget),
        timing=dict(reader_seconds=seconds,scope='Incremental diagnostic with warmed caches, not cold end-to-end performance'))
    return helper.jsonable(result)


def solution_from_bundle(bundle):
    import numpy as np
    from bindpack.utils.misc_utils import QuerySolution
    s=bundle['baseline_solution']
    return QuerySolution(question=bundle['input']['query'],docs=s['docs'],doc_scores=np.asarray(s['scores']),retrieval_trace=s['retrieval_trace'])


def variant_solution(rag,bundle,features,helper,recorder,qid,output,ds):
    import numpy as np
    from bindpack.utils.misc_utils import QuerySolution
    from bindpack.stages.pack.coverage_selector import CoverageSelector
    from diagnostic_improvements import FactCoverageSelector,recover_plan,positive_path_fusion,complete_chain_package
    original=bundle['input'];state=helper.restore_pack_input(original)
    ids=np.asarray(original['ids'],dtype=int);scores=np.asarray(original['scores'],dtype=float)
    diagnostic={};changed=False
    if 'plan_repair' in features and not state.get('_evidence_plan'):
        plan,repair=recover_plan(original['query'],bundle['hops'],state['evidence_trace'].get('planning_outputs',{}),rag.config.plan)
        diagnostic['plan_repair']=repair
        if plan:
            state=deepcopy(bundle['pre_plan']);state['evidence_candidates']={int(k):v for k,v in state['evidence_candidates'].items()}
            state['_evidence_plan']=plan;state['evidence_trace']['plan']=plan
            previous=original['state']['evidence_trace']
            state['evidence_trace']['planning_outputs']=dict(previous['planning_outputs'],deterministic_recovery=repair)
            state['evidence_trace']['llm_plan_calls']=previous['llm_plan_calls']
            with recorder.section('recovered_plan_execute',qid):rag.evidence_runtime.plan.executor.execute([state],None)
            rag.evidence_runtime._trim_pool(state);changed=True
    if 'route_fusion' in features:
        before=bundle.get('pre_path')
        if before:
            ids,scores,route=positive_path_fusion(before['ids'],before['scores'],ids,scores,len(rag.passage_node_keys))
            diagnostic['route_fusion']=route;changed|=bool(route['restored_penalized_passages'])
        else:diagnostic['route_fusion']=dict(noop='native_dense_fallback_has_no_pathset')
    if features=={'plan_repair'} and not changed:return None,None,diagnostic
    if features=={'hover_reader'} and ds!='hover':return None,None,dict(noop='reader_change_only_for_hover')
    runtime=rag.evidence_runtime
    runtime.pack.coverage=FactCoverageSelector(runtime,runtime.pack.scorer) if 'coverage_facts' in features else CoverageSelector(runtime,runtime.pack.scorer)
    payload=helper.freeze_pack_input(original['query'],ids,scores,state)
    with recorder.section('pack_'+('_'.join(sorted(features))),qid):
        result=runtime.pack.finalize(original['query'],ids,scores,{},state)
        if 'chain_package' in features:result=complete_chain_package(runtime,result,state,original['query'],ids,scores)
        out_ids,out_scores,trace=result
        trace=dict(trace,experimental_changes=diagnostic)
        solution=QuerySolution(question=original['query'],docs=[runtime._document(int(i)) for i in out_ids[:200]],
            doc_scores=out_scores[:200],retrieval_trace=dict(stage_ablation='full',evidence=trace))
        runtime.pack.postprocess([(solution,out_ids[:200].copy())])
    runtime.pack.coverage=CoverageSelector(runtime,runtime.pack.scorer)
    return solution,payload,diagnostic


def worker(output,ds,phase,limit=None):
    load_environment(output)
    import numpy as np
    from transformers import AutoTokenizer
    from bindpack import BindPack,BindPackConfig
    from bindpack.evaluation.measurement import Recorder,ResourceMonitor
    from diagnostic_improvements import enable_hover_reader
    from bindpack.stages.pack.coverage_selector import CoverageSelector
    sweep=load_sweep(output);helper=sweep.helpers(BASE)
    sys.path.insert(0,str(BASE/'source_snapshot/scripts'))
    selected=read(output/'selection_manifest.json')['datasets'][ds]['rows']
    assert digest(read(output/'selection_manifest.json'))==read(output/'selection_locked.ok')['sha256']
    selected_features=read(output/'selection_decision.json')['retained'] if phase=='combined' else []
    arms=('combined',) if phase=='combined' else ('baseline',)+ARMS
    if all(len(completed_rows(output,ds,arm))==50 for arm in arms):
        for arm in arms:finish(output,ds,arm)
        return
    if phase=='combined' and len(selected_features)<=1:
        source=selected_features[0] if selected_features else 'baseline'
        for row in completed_rows(output,ds,source):
            new=deepcopy(row);new.update(experimental_variant='combined',experimental_features=selected_features,
                exact_result_reused_from=source)
            write(row_path(output,ds,'combined',row['sample_id']),dict(row=new,sha256=digest(new)))
            token=hashlib.sha256(str(row['sample_id']).encode()).hexdigest()+'.json'
            write(output/'frozen_inputs'/ds/'combined'/token,read(output/'frozen_inputs'/ds/source/token))
        finish(output,ds,'combined');return
    recorder=Recorder(output/'runtime'/ds/'accounting')
    tokenizer=AutoTokenizer.from_pretrained('/root/models/Qwen3-8B',local_files_only=True)
    recorder.install_llm_factory(__import__('bindpack.llm.openai_gpt',fromlist=['CacheOpenAI']).CacheOpenAI,tokenizer)
    index=helper.private_index(output,ds)
    marker=output/'runtime'/ds/'cache_initialized.ok'
    if not marker.exists():
        for p in (BASE/'runtime'/ds/'index/llm_cache').glob('*.sqlite'):shutil.copy2(p,index/'llm_cache'/p.name)
        write(marker,dict(original_cache_copied=True))
    config=BindPackConfig.from_file(output/'metadata'/ds/'method.json',save_dir=str(index),stage_ablation='full')
    assert config.pack.relation_weight==.6
    monitor=ResourceMonitor(recorder)
    with monitor:
        with recorder.section('model_and_index_loading'):
            rag=BindPack(config=config);recorder.install_embedding(rag.embedding_model)
            rag.prepare_retrieval_objects();rag._build_index_side_artifacts()
        def no_index(*args,**kwargs):raise RuntimeError('Diagnostic must reuse the shared index')
        rag.index=no_index
        if not (output/'runtime'/ds/'parity_r060.ok').exists():sweep.parity(BASE,output,ds,rag,tokenizer,recorder,helper)
        print(f'INDEX READY {ds}; phase={phase}; reused passages={len(rag.passage_node_keys)}',flush=True)
        captured={};capture_enabled=False
        native_plan=rag.evidence_runtime.plan.process_window
        native_path=rag._apply_path_set_optimization
        native_finalize=rag.evidence_runtime.pack.finalize
        def plan_window(states,executor=None):
            if capture_enabled:
                s=states[0];captured['pre_plan']=helper.jsonable({k:deepcopy(s[k]) for k in ('query','evidence_candidates','evidence_trace','_evidence_search_cache')})
            return native_plan(states,executor)
        def pathset(*args,**kwargs):
            if capture_enabled:
                captured['pre_path']=helper.jsonable(dict(ids=kwargs['base_sorted_doc_ids'],scores=kwargs['base_sorted_doc_scores']))
            return native_path(*args,**kwargs)
        def finalize(query,ids,scores,ctx,state):
            if capture_enabled:captured['input']=helper.freeze_pack_input(query,ids,scores,state)
            return native_finalize(query,ids,scores,ctx,state)
        rag.evidence_runtime.plan.process_window=plan_window;rag._apply_path_set_optimization=pathset
        rag.evidence_runtime.pack.finalize=finalize
        original_hover=deepcopy(rag.prompt_template_manager.templates.get('rag_qa_hover'))
        benchmark=read(output/'metadata'/ds/'benchmark.json')
        for ordinal,chosen in enumerate(selected[:limit or 50]):
            qid=chosen['sample_id'];sample=benchmark[chosen['query_index']]
            path=output/'bundles'/ds/(hashlib.sha256(str(qid).encode()).hexdigest()+'.json')
            if phase=='individual' and not path.exists():
                captured.clear();capture_enabled=True
                with recorder.section('baseline_route_plan_pack',qid):
                    rag.set_query_hop_overrides([chosen['benchmark_hops']],source='historical_old50_fixed_for_component_test')
                    solution=rag.retrieve([chosen['question']],num_to_retrieve=200)[0]
                capture_enabled=False
                bundle=dict(captured,hops=chosen['benchmark_hops'],baseline_solution=helper.jsonable(dict(
                    docs=solution.docs,scores=solution.doc_scores,retrieval_trace=solution.retrieval_trace)))
                write(path,dict(payload=bundle,sha256=digest(bundle)))
            saved=read(path);bundle=saved['payload'];assert digest(bundle)==saved['sha256']
            for arm in arms:
                checkpoint=row_path(output,ds,arm,qid)
                if checkpoint.exists():continue
                t=time.perf_counter();features=set(selected_features) if arm=='combined' else ({arm} if arm!='baseline' else set())
                rag.prompt_template_manager.templates['rag_qa_hover']=deepcopy(original_hover)
                if 'hover_reader' in features:enable_hover_reader(rag)
                if arm=='baseline':solution=solution_from_bundle(bundle);payload=bundle['input'];diagnostic={}
                else:solution,payload,diagnostic=variant_solution(rag,bundle,features,helper,recorder,qid,output,ds)
                if solution is None:
                    old=read(row_path(output,ds,'baseline',qid))['row'];row=deepcopy(old)
                    row.update(experimental_variant=arm,experimental_features=sorted(features),
                        exact_result_reused_from='baseline',experimental_diagnostic=diagnostic)
                    payload=bundle['input']
                else:
                    row=export_answer(rag,solution,tokenizer,recorder,ds,chosen,sample,arm,helper,sorted(features),digest(payload))
                    row.update(experimental_diagnostic=diagnostic)
                row['timing']['incremental_arm_seconds']=time.perf_counter()-t
                write(checkpoint,dict(row=row,sha256=digest(row)))
                frozen=output/'frozen_inputs'/ds/arm/(hashlib.sha256(str(qid).encode()).hexdigest()+'.json')
                write(frozen,dict(input=payload,input_sha256=digest(payload)))
                n=len(completed_rows(output,ds,arm))
                write(output/'status.json',dict(phase=phase,dataset=ds,arm=arm,completed_arm=n,expected_arm=50,
                    updated_at=datetime.now().astimezone().isoformat()))
                print(f'QUERY {ds}/{arm} {n}/50; {time.perf_counter()-t:.2f}s; FCR5={row["all_gold_in_top5"]}; answer={row["answer"]!r}',flush=True)
            if (ordinal+1)%10==0:print(f'PROGRESS {ds} {ordinal+1}/50 questions; phase={phase}',flush=True)
        rag.prompt_template_manager.templates['rag_qa_hover']=original_hover
    write(output/'runtime'/ds/f'phase_timings_{phase}_{os.getpid()}.json',recorder.phases)
    write(output/'runtime'/ds/'resource_summary.json',monitor.summary())
    for obj,name,original in reversed(recorder.restorations):setattr(obj,name,original)
    for arm in arms:
        if len(completed_rows(output,ds,arm))==50:finish(output,ds,arm)


def finish(output,ds,arm):
    from bindpack.evaluation.metric_report import write_metric_report
    rows=completed_rows(output,ds,arm);assert len(rows)==50
    method=read(output/'metadata'/ds/'method.json');identity=read(output/'metadata'/ds/'sample_manifest.json')
    features=rows[0]['experimental_features']
    directory=output/'cases'/ds/arm
    protocol=dict(variant=arm,features=features,relation_weight=.6,reader_top_k=5,index_rebuilt=False,
        selection_sha256=read(output/'selection_locked.ok')['sha256'],diagnostic_only=True)
    write(directory/'protocol.json',protocol)
    result=dict(dataset=ds,method='bindpack',stage_ablation='full',eval_mode='rag_qa',
        data_path=str(output/'metadata'/ds/'benchmark.json'),corpus_path=str(output/'metadata'/ds/'corpus.json'),
        selected_indices=identity['selected_indices'],sample_size_effective=50,diagnostic_selection=True,
        pack_relation_weight=.6,experimental_variant=arm,experimental_features=features,
        runtime_config=dict(method['runtime'],**{k:method[k] for k in ('candidates','route','plan','pack')}),
        hop_provenance=identity.get('hop_provenance'),results=rows)
    write(directory/'result.json',result)
    write(directory/'validated.ok',dict(result_sha256=sha(directory/'result.json'),protocol_sha256=digest(protocol),sample_count=50))
    report=write_metric_report(directory/'result.json',data_path=output/'metadata'/ds/'benchmark.json',
        manifest_path=output/'metadata'/ds/'sample_manifest.json',ks=(3,5,8,10,20,200))
    print(f'REPORT {ds}/{arm}: '+json.dumps({k:report['metrics'].get(k) for k in ('Recall@5','FCR@5','ExactMatch','F1','ClassificationMacroF1','DCC@5')},ensure_ascii=False),flush=True)


def summarize(output,choose=False):
    rows=[];table=['# 固定0.60：单项改动与组合诊断','','同一批原始200题，每个数据集50题。普通问答使用EM/token F1，HoVer使用Accuracy/Macro-F1。基线为本轮未改算法Full；旧0.60仅作运行一致性参照。','',
        '| 数据集 | 版本 | Recall@5 | FCR@5 | EM / Accuracy | F1 / Macro-F1 | DCC@5 |',
        '|---|---|---:|---:|---:|---:|---:|']
    all_arms=('baseline',)+ARMS+('combined',)
    for ds in DATASETS:
        variants={}
        for arm in all_arms:
            p=output/'cases'/ds/arm/'report.json'
            if not p.exists():continue
            m=read(p)['metrics'];variants[arm]=m
            qa='ClassificationAccuracy' if ds=='hover' else 'ExactMatch';f1='ClassificationMacroF1' if ds=='hover' else 'F1'
            table.append(f'| {ds} | {arm} | '+' | '.join('N/A' if m.get(k) is None else f'{m[k]:.6f}' for k in ('Recall@5','FCR@5',qa,f1,'DCC@5'))+' |')
        rows.append(dict(dataset=ds,variants=variants))
    write(output/'comparison.json',dict(relation_weight=.6,datasets=rows))
    (output/'comparison.md').write_text('\n'.join(table)+'\n')
    if choose:
        decisions={};retained=[]
        for arm in ARMS:
            delta=[]
            for row in rows:
                ds=row['dataset'];a=row['variants']['baseline'];b=row['variants'][arm]
                qa='ClassificationAccuracy' if ds=='hover' else 'ExactMatch';f1='ClassificationMacroF1' if ds=='hover' else 'F1'
                delta.append(dict(dataset=ds,qa=b[f1]-a[f1],correctness=b[qa]-a[qa],fcr=b['FCR@5']-a['FCR@5']))
            means={k:sum(d[k] for d in delta)/4 for k in ('qa','correctness','fcr')}
            keep=all(v>=-1e-10 for v in means.values()) and any(v>1e-10 for v in means.values())
            status='retain' if keep else ('tradeoff' if any(v>1e-10 for v in means.values()) else 'discard_or_no_gain')
            decisions[arm]=dict(status=status,mean_deltas=means,per_dataset=delta)
            if keep:retained.append(arm)
        write(output/'selection_decision.json',dict(retained=retained,decisions=decisions,
            rule=read(output/'experiment.json')['selection_rule'],locked_before_outcomes=True))
        print('SCREENING '+json.dumps(decisions,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--worker',choices=DATASETS)
    p.add_argument('--phase',choices=('individual','combined'),default='individual');p.add_argument('--limit',type=int)
    p.add_argument('--summarize-only',action='store_true');args=p.parse_args();output=args.output.resolve()
    logging.basicConfig(level=logging.WARNING,format='%(asctime)s %(levelname)s %(message)s')
    if args.worker:worker(output,args.worker,args.phase,args.limit);return
    if args.summarize_only:summarize(output);return
    prepare(output)
    if args.prepare_only:return
    import fcntl
    with (output/'.run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(output/'run.pid').write_text(str(os.getpid())+'\n')
        for phase in ('individual','combined'):
            if phase=='combined':summarize(output,choose=True)
            for ds in DATASETS:
                print(f'START {phase} {ds}',flush=True)
                command=[sys.executable,'-B','-u',str(output/'source_snapshot/scripts'/SCRIPT.name),
                    '--output',str(output),'--worker',ds,'--phase',phase]
                subprocess.run(command,check=True);summarize(output)
        write(output/'experiments_completed.ok',dict(questions=200,single_arms=5,combination=True,relation_weight=.6))
        print('EXPERIMENTS COMPLETE: all single changes and selected combination.',flush=True)
        subprocess.run([sys.executable,'-B','-u',str(output/'source_snapshot/scripts/finalize_component_diagnostic.py'),
            '--output',str(output)],check=True)


if __name__=='__main__':main()

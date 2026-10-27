#!/usr/bin/env python3
"""Replay additional Full Pack weights on the locked 200-question diagnostic.

No index building, embedding inference, Route or Plan execution is permitted.
Native Pack and reader methods consume the exact original frozen inputs. A
cache-only 0.60 parity check must pass before any new-weight inference.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import hashlib
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PROJECT=Path('/root/BindPack')
DEFAULT_BASE=PROJECT/'outputs/pack_weight_diagnostic_20261025'
DATASETS=('hotpotqa','2wikimultihopqa','musique','hover')
SCRIPT=Path(__file__).resolve()


def read(path):return json.loads(Path(path).read_text())


def digest(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False)
        handle.write('\n');handle.flush();os.fsync(handle.fileno())
    temporary.replace(path)


def append(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as handle:
        handle.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
        handle.flush();os.fsync(handle.fileno())


def label(weight):
    if not math.isfinite(weight) or not 0<=weight<=1 or abs(weight*100-round(weight*100))>1e-8:
        raise ValueError('Weights must be finite values in [0,1] with at most two decimal places')
    return f'full_r{round(weight*100):03d}'


def helpers(base):
    path=base/'source_snapshot/scripts/run_pack_weight_diagnostic.py'
    spec=importlib.util.spec_from_file_location('frozen_pack_diagnostic_helpers',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class ReadOnlyPassages:
    """The two non-vector columns in the original passage store, in row order."""
    def __init__(self,parquet):
        import pyarrow.parquet as pq
        table=pq.read_table(parquet,columns=['hash_id','content'])
        self.keys=table['hash_id'].to_pylist()
        self.texts=table['content'].to_pylist()
        if len(self.keys)!=len(set(self.keys)):raise ValueError('Duplicate physical passage keys')
        self.rows={key:dict(hash_id=key,content=text) for key,text in zip(self.keys,self.texts)}

    def get_row(self,key):return self.rows[key]

    def insert_strings(self,*args,**kwargs):raise RuntimeError('Frozen Pack replay cannot update the index')


def prepare(base,output,weights):
    assert read(base/'audit.json')['status']=='passed'
    selection=read(base/'selection_manifest.json');lock=read(base/'selection_locked.ok')
    assert digest(selection)==lock['sha256']
    if len(set(weights))!=len(weights):raise ValueError('Duplicate weights')
    if any(weight in (.25,.60) for weight in weights):raise ValueError('Reuse completed 0.25/0.60 instead of re-running them')
    names=[label(weight) for weight in weights]
    identity=dict(schema_version=1,base=str(base),weights=weights,cases=names,
        selection_sha256=lock['sha256'],sample_count=200,diagnostic_only=True,
        shared_frozen_input=True,rebuild_index=False,rerun_route_plan=False,embedding_inference=False,
        baseline_results={ds:{'path':str(base/'cases'/ds/'full_r060/result.json'),
            'sha256':sha(base/'cases'/ds/'full_r060/result.json')} for ds in DATASETS},
        native_source={str(p.relative_to(base/'source_snapshot/src')):sha(p)
            for p in (base/'source_snapshot/src').rglob('*.py')})
    if (output/'experiment.json').exists():
        assert read(output/'experiment.json')==identity,'Frozen sweep protocol changed'
        return
    output.mkdir(parents=True,exist_ok=True)
    write(output/'experiment.json',identity)
    write(output/'selection_manifest.json',selection)
    write(output/'selection_locked.ok',lock)
    for ds in DATASETS:
        old=base/'metadata'/ds;meta=output/'metadata'/ds;meta.mkdir(parents=True,exist_ok=True)
        for file in ('benchmark.json','corpus.json'):
            (meta/file).symlink_to(os.path.relpath(old/file,meta))
        manifest=read(old/'sample_manifest.json');manifest['data_path']=str(meta/'benchmark.json')
        write(meta/'sample_manifest.json',manifest)
        baseline=read(old/'full_r060_method.json')
        for name,weight in zip(names,weights):
            method=deepcopy(baseline)
            method['runtime']['save_dir']=str(output/'runtime'/ds/'cache')
            method['pack']['relation_weight']=weight
            write(meta/(name+'_method.json'),method)
    snapshot=output/'source_snapshot'
    shutil.copytree(base/'source_snapshot/src',snapshot/'src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (snapshot/'scripts').mkdir(parents=True,exist_ok=True)
    shutil.copy2(SCRIPT,snapshot/'scripts'/SCRIPT.name)
    write(output/'prepared.ok',dict(selection_sha256=lock['sha256'],weights=weights,new_model_requests=0))


def load_environment(output):
    sys.path.insert(0,str(output/'source_snapshot/src'))
    os.environ.setdefault('OPENAI_API_KEY','EMPTY')


def make_runtime(config,store,recorder,tokenizer):
    from bindpack.core.base import BaseRAG
    from bindpack.stages.route.route_refinement import RouteRefinement
    from bindpack.prompts.prompt_template_manager import PromptTemplateManager
    from bindpack.llm.openai_gpt import CacheOpenAI
    from bindpack.pipeline import EvidencePipeline

    class PackReplayRAG:
        # The native reader and parser execute unchanged, without constructing
        # an embedding model, graph retriever, index writer or OpenIE pipeline.
        qa=BaseRAG.qa
        _extract_llm_text=staticmethod(RouteRefinement._extract_llm_text)

        def __init__(self):
            self.config=self.global_config=config
            self.passage_node_keys=store.keys
            self.chunk_embedding_store=store
            self.prompt_template_manager=PromptTemplateManager(role_mapping={'system':'system','user':'user','assistant':'assistant'})
            self.llm_model=CacheOpenAI.from_experiment_config(config)
            recorder.install_llm(self.llm_model,tokenizer)
            self.evidence_runtime=EvidencePipeline(self)

        def retrieve(self,*args,**kwargs):raise RuntimeError('Replay must not execute Route or Plan')
        def index(self,*args,**kwargs):raise RuntimeError('Replay must not build an index')
        def dense_passage_retrieval(self,*args,**kwargs):raise RuntimeError('Replay must not execute embedding retrieval')

    return PackReplayRAG()


def run_arm(rag,tokenizer,recorder,original,weight,qid,helper):
    import numpy as np
    from bindpack.utils.misc_utils import QuerySolution
    from bindpack.evaluation.measured_reader import prepare as prepare_reader
    config=rag.config;config.pack.relation_weight=weight
    runtime=rag.evidence_runtime;key=label(weight)[5:]
    fingerprint=digest(original)
    state=helper.restore_pack_input(original)
    start=time.perf_counter()
    with recorder.section('pack_'+key+'_selection',qid):
        ids,scores,trace=runtime.pack.finalize(original['query'],np.asarray(original['ids'],dtype=int),
            np.asarray(original['scores'],dtype=float),{},state)
    assert digest(original)==fingerprint
    solution=QuerySolution(question=original['query'],docs=[runtime._document(int(d)) for d in ids[:200]],
        doc_scores=scores[:200],retrieval_trace={'stage_ablation':'full','evidence':trace})
    with recorder.section('pack_'+key+'_review',qid):runtime.pack.postprocess([(solution,ids[:200].copy())])
    pack_seconds=time.perf_counter()-start
    with recorder.section('reader_'+key,qid):
        start=time.perf_counter()
        reading,budget=prepare_reader(rag,solution,tokenizer,config.dataset)
        answered,raw,metas=rag.qa([reading]);solution.answer=answered[0].answer
        reader_seconds=time.perf_counter()-start
    assert digest(original)==fingerprint
    return solution,dict(raw_response=raw[0],metadata=metas[0],budget=budget),dict(pack_seconds=pack_seconds,reader_seconds=reader_seconds)


def parity(base,output,ds,rag,tokenizer,recorder,helper):
    expected=read(base/'cases'/ds/'full_r060/result.json')['results']
    rows=read(base/'selection_manifest.json')['datasets'][ds]['rows']
    transport=rag.llm_model.openai_client.chat.completions
    real_create=transport.create
    def deny_http(*args,**kwargs):raise AssertionError('0.60 parity cache miss: check configuration, prompts or replay state before new inference')
    transport.create=deny_http
    checked=[]
    try:
        for row,old in zip(rows,expected):
            assert row['sample_id']==old['sample_id']
            qid=row['sample_id'];path=base/'frozen_inputs'/ds/(hashlib.sha256(str(qid).encode()).hexdigest()+'.json')
            saved=read(path);frozen=saved['input'];assert digest(frozen)==saved['input_sha256']==old['frozen_pack_input_sha256']
            solution,reader,timing=run_arm(rag,tokenizer,recorder,frozen,.60,qid,helper)
            assert solution.docs==old['candidate_docs'],f'0.60 ranking mismatch: {ds}/{qid}'
            assert solution.answer==old['answer'],f'0.60 reader mismatch: {ds}/{qid}'
            assert reader['budget']['prompt_sha256']==old['reader']['budget']['prompt_sha256']
            assert solution.retrieval_trace['evidence']['improvement_support_semantic_veto']['original_parent_doc_ids']==old['retrieval_trace']['evidence']['improvement_support_semantic_veto']['original_parent_doc_ids']
            checked.append(dict(sample_id=qid,frozen_input_sha256=saved['input_sha256']))
        assert len(checked)==50
    finally:transport.create=real_create
    write(output/'runtime'/ds/'parity_r060.ok',dict(status='passed',checked=50,new_http_requests=0,
        baseline_result_sha256=sha(base/'cases'/ds/'full_r060/result.json'),rows=checked))
    print(f'PARITY PASS {ds}: all 50 original 0.60 rankings, prompts and answers reproduced; zero HTTP requests',flush=True)


def worker(base,output,ds,limit,parity_only):
    load_environment(output)
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from bindpack import BindPackConfig
    from bindpack.evaluation.measurement import Recorder,ResourceMonitor,jsonl,usage
    from bindpack.evaluation.result_export import detailed_result
    helper=helpers(base);settings=read(output/'experiment.json');weights=settings['weights'];names=settings['cases']
    assert settings['selection_sha256']==digest(read(base/'selection_manifest.json'))
    assert sha(base/'cases'/ds/'full_r060/result.json')==settings['baseline_results'][ds]['sha256']
    selected=read(base/'selection_manifest.json')['datasets'][ds]['rows']
    log=output/'runtime'/ds/'pairs.jsonl';pairs=list(jsonl(log))
    for pair,row in zip(pairs,selected):assert pair['sample_id']==row['sample_id'] and digest(pair['payload'])==pair['sha256']
    if len(pairs)==50:
        finish(output,ds,pairs);return
    cache=output/'runtime'/ds/'cache/llm_cache';cache.mkdir(parents=True,exist_ok=True)
    for src in (base/'runtime'/ds/'index/llm_cache').glob('*.sqlite'):
        dst=cache/src.name
        if not dst.exists():shutil.copy2(src,dst)
    cfg=BindPackConfig.from_file(base/'metadata'/ds/'full_r060_method.json',
        save_dir=str(cache.parent),stage_ablation='full')
    assert cfg.llm_name=='qwen3-8b' and cfg.embedding_model_name=='/root/models/Qwen3-Embedding-8B'
    recorder=Recorder(output/'runtime'/ds/'accounting')
    tokenizer=AutoTokenizer.from_pretrained('/root/models/Qwen3-8B',local_files_only=True)
    passage_path=next((PROJECT/'outputs/4_datasets_result/shared_indexes'/ds).rglob('vdb_chunk.parquet'))
    start=time.perf_counter();store=ReadOnlyPassages(passage_path)
    rag=make_runtime(cfg,store,recorder,tokenizer)
    print(f'PASSAGES READY {ds}: {len(store.keys)} original passages in {time.perf_counter()-start:.2f}s; no embeddings, graph load or upstream execution',flush=True)
    monitor=ResourceMonitor(recorder)
    with monitor:
        marker=output/'runtime'/ds/'parity_r060.ok'
        if not marker.exists():parity(base,output,ds,rag,tokenizer,recorder,helper)
        else:
            receipt=read(marker);assert receipt['checked']==50 and receipt['baseline_result_sha256']==settings['baseline_results'][ds]['sha256']
        if not parity_only:
            benchmark=read(output/'metadata'/ds/'benchmark.json')
            old={row['sample_id']:row for row in read(base/'cases'/ds/'full_r060/result.json')['results']}
            for ordinal in range(len(pairs),min(50,limit or 50)):
                row=selected[ordinal];qid=row['sample_id'];t=time.perf_counter()
                saved=read(base/'frozen_inputs'/ds/(hashlib.sha256(str(qid).encode()).hexdigest()+'.json'))
                frozen=saved['input'];assert digest(frozen)==saved['input_sha256']==old[qid]['frozen_pack_input_sha256']
                exports={}
                for name,weight in zip(names,weights):
                    solution,reader,timing=run_arm(rag,tokenizer,recorder,frozen,weight,qid,helper)
                    # Evaluation annotations are attached only after Pack and QA.
                    gold=old[qid]['gold_docs'];solution.gold_docs=gold;solution.gold_answers=old[qid]['gold_answers']
                    export=detailed_result(solution,benchmark[row['query_index']],row['query_index'],gold,row['benchmark_hops'],10,200)
                    export.update(ranking_complete=True,actual_ranking_length=200,diagnostic_cohort=row['group'],
                        pack_relation_weight=weight,frozen_pack_input_sha256=saved['input_sha256'],reader=reader,
                        timing=dict(timing,shared_route_plan_reused=True,end_to_end_seconds=None,
                                    latency_scope='Pack + reader replay only; upstream from frozen source run'))
                    exports[name]=helper.jsonable(export)
                payload=dict(rows=exports,cohort=row['group'],frozen_pack_input_sha256=saved['input_sha256'])
                event=dict(sample_id=qid,payload=payload,sha256=digest(payload));append(log,event);pairs.append(event)
                completed=sum(sum(1 for _ in jsonl(output/'runtime'/dataset/'pairs.jsonl')) for dataset in DATASETS)
                write(output/'status.json',dict(dataset=ds,completed_dataset_pairs=len(pairs),total_completed_pairs=completed,
                    expected_total_pairs=200,phase='running',updated_at=datetime.now().astimezone().isoformat()))
                result=' '.join(f'{w:.2f}:FCR5={exports[n]["all_gold_in_top5"]}' for n,w in zip(names,weights))
                print(f'PAIR {ds} {len(pairs)}/50 ({completed}/200); {result}; {time.perf_counter()-t:.2f}s',flush=True)
    write(output/'runtime'/ds/('parity_phase_timings.json' if parity_only else 'phase_timings.json'),recorder.phases)
    for obj,name,original in reversed(recorder.restorations):setattr(obj,name,original)
    if len(pairs)==50:finish(output,ds,pairs)
    summarize(base,output)


def finish(output,ds,pairs):
    from bindpack.evaluation.metric_report import write_metric_report
    from bindpack.evaluation.measurement import jsonl,usage
    settings=read(output/'experiment.json');meta=output/'metadata'/ds;identity=read(meta/'sample_manifest.json')
    for name,weight in zip(settings['cases'],settings['weights']):
        case=output/'cases'/ds/name;method=read(meta/(name+'_method.json'))
        protocol=dict(stage_ablation='full',diagnostic=True,relation_weight=weight,
            selection_sha256=settings['selection_sha256'],shared_frozen_route_plan=True,
            replay_validated_by_r060_cache_only_parity=True,upstream_rerun=False,embedding_inference=False,index_rebuilt=False,
            cache_policy='Private copy of original 0.25/0.60 cache; identical prompts reuse identical responses; incremental costs only')
        write(case/'protocol.json',protocol)
        result=dict(dataset=ds,method='bindpack',stage_ablation='full',eval_mode='rag_qa',
            data_path=str(meta/'benchmark.json'),corpus_path=str(meta/'corpus.json'),
            selected_indices=identity['selected_indices'],sample_size_effective=50,diagnostic_selection=True,
            pack_relation_weight=weight,hop_provenance=identity.get('hop_provenance'),
            runtime_config=dict(method['runtime'],**{key:method[key] for key in ('candidates','route','plan','pack')}),
            results=[pair['payload']['rows'][name] for pair in pairs])
        write(case/'result.json',result)
        write(case/'validated.ok',dict(result_sha256=sha(case/'result.json'),protocol_sha256=digest(protocol),sample_count=50))
        report=write_metric_report(case/'result.json',data_path=meta/'benchmark.json',manifest_path=meta/'sample_manifest.json',ks=(3,5,8,10,20,200))
        print(f'REPORT {ds}/{name}: '+json.dumps({key:report['metrics'].get(key) for key in ('Recall@5','FCR@5','ExactMatch','F1','ClassificationMacroF1','DCC@5')},ensure_ascii=False),flush=True)
    calls=list(jsonl(output/'runtime'/ds/'accounting/llm_calls.jsonl'))
    assert not any('r060' in c['phase'] or 'route' in c['phase'] or 'embedding' in c['phase'] for c in calls)
    write(output/'runtime'/ds/'cost.json',dict(physical=usage(calls),embedding_calls=0,route_plan_calls=0,index_build_calls=0,
        scope='Incremental Pack/reader replay only; no fresh full-pipeline efficiency comparison'))
    write(output/'cases'/ds/'completed.ok',dict(pairs=50,selection_sha256=settings['selection_sha256']))


def summarize(base,output):
    settings=read(output/'experiment.json');all_data=[]
    new_weights=' / '.join(f'{w:.2f}' for w in settings['weights'])
    lines=['# Full Pack 权重对比：同一批 200 条诊断样本','',f'旧 0.25 / 0.60 结果直接引用；新增 {new_weights} 使用相同冻结 Pack 输入。每个数据集 50 条，分组和样本顺序不变。前三个数据集使用 EM / token F1，HoVer 使用 Accuracy / Macro-F1。','',
        '| 数据集 | R 权重 | Recall@5 | FCR@5 | EM / Accuracy | token F1 / Macro-F1 | DCC@5 | NodeCov@5 | ClosureGap@5 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for ds in DATASETS:
        variants={}
        for weight,source in [(.25,base),(.60,base),*[(w,output) for w in settings['weights']]]:
            name=label(weight);path=source/'cases'/ds/name/'report.json'
            if path.exists():variants[weight]=dict(case=name,path=str(path),metrics=read(path)['metrics'])
        baseline=variants[.60]['metrics']
        for weight in sorted(variants):
            value=variants[weight];m=value['metrics'];qa='ClassificationAccuracy' if ds=='hover' else 'ExactMatch';f1='ClassificationMacroF1' if ds=='hover' else 'F1'
            keys=('Recall@5','FCR@5',qa,f1,'DCC@5','NodeCov@5','ClosureGap@5')
            values=[m.get(key) for key in keys]
            value['delta_vs_r060']={k:None if m.get(k) is None else m[k]-baseline[k] for k in keys}
            target=os.path.relpath(value['path'],output)
            lines.append(f'| [{ds}]({target}) | {weight:.2f} | '+' | '.join('N/A' if v is None else f'{v:.6f}' for v in values)+' |')
        all_data.append(dict(dataset=ds,variants={f'{k:.2f}':v for k,v in sorted(variants.items())}))
    lines+=['','本表来自按历史结果预选的诊断样本，不是全量、随机代表性主实验。0.60 回放入口先用缓存逐题复现原排名、提示和答案；新增权重只改变 relation_weight，保留原 DAG、语义审核和 Reader@5。',
        '','没有重新建库、查询 Embedding、Route 或 Plan 推理。只读取原索引的 passage 文本，并使用已有冻结上游结果。新增请求与 token 记入 runtime/<dataset>/accounting，缓存命中不算新物理请求。',
        '','旧三数据集/HoVer 的 hop 条件保持原实验，2Wiki 仍是该历史批次的 prior=2。最终配置应在独立开发样本验证后锁定。']
    write(output/'comparison.json',dict(diagnostic_only=True,selection_sha256=settings['selection_sha256'],datasets=all_data))
    (output/'comparison.md').write_text('\n'.join(lines)+'\n')


def audit(base,output):
    from bindpack.evaluation.measurement import jsonl
    settings=read(output/'experiment.json');selected=read(base/'selection_manifest.json');totals=Counter()
    phase_prefixes=tuple(f'{phase}_{case[5:]}' for case in settings['cases'] for phase in ('pack','reader'))
    for ds in DATASETS:
        assert read(output/'runtime'/ds/'parity_r060.ok')['checked']==50
        pairs=list(jsonl(output/'runtime'/ds/'pairs.jsonl'));assert len(pairs)==50
        for pair,row in zip(pairs,selected['datasets'][ds]['rows']):
            assert pair['sample_id']==row['sample_id'] and digest(pair['payload'])==pair['sha256']
            saved=read(base/'frozen_inputs'/ds/(hashlib.sha256(str(row['sample_id']).encode()).hexdigest()+'.json'))
            assert digest(saved['input'])==saved['input_sha256']==pair['payload']['frozen_pack_input_sha256']
            for name,weight in zip(settings['cases'],settings['weights']):
                result=pair['payload']['rows'][name]
                assert result['pack_relation_weight']==weight and result['frozen_pack_input_sha256']==saved['input_sha256']
                assert result['retrieval_trace']['evidence']['stages']==dict(route=True,plan=True,pack=True)
                assert len(set(result['candidate_docs']))==len(result['candidate_docs'])==200
                totals['reader_partial']+=bool(result['reader']['budget']['partial_passage_ranks'])
                totals['reader_dropped']+=bool(result['reader']['budget']['dropped_passage_ranks'])
        for name,weight in zip(settings['cases'],settings['weights']):
            report=read(output/'cases'/ds/name/'report.json')
            assert report['sample_alignment']['verified'] and report['coverage']['sample_count']==report['coverage']['qa_n']==50
        for c in jsonl(output/'runtime'/ds/'accounting/llm_calls.jsonl'):
            assert c['phase'].startswith(phase_prefixes)
            totals['http_attempts']+=1;totals['failed_attempts']+=c['status']!='success'
            totals['unknown_usage']+=c.get('prompt_tokens') is None or c.get('completion_tokens') is None
            totals['input_tokens']+=c.get('prompt_tokens') or 0;totals['output_tokens']+=c.get('completion_tokens') or 0
    answer_count=200*len(settings['weights'])
    write(output/'audit.json',dict(status='passed',parity_r060_questions=200,parity_r060_new_http_requests=0,
        additional_questions=200,additional_answers=answer_count,upstream_reruns=0,embedding_calls=0,index_build_calls=0,
        selection_sha256=settings['selection_sha256'],incremental_cost=dict(totals)))
    write(output/'completed.ok',dict(status='complete',additional_answers=answer_count,selection_sha256=settings['selection_sha256'],audit_sha256=sha(output/'audit.json')))
    write(output/'status.json',dict(phase='complete',total_completed_pairs=200,expected_total_pairs=200))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=DEFAULT_BASE)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--weights',nargs='+',type=float,default=[.40,.80])
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--parity-only',action='store_true')
    parser.add_argument('--worker',choices=DATASETS)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--summarize-only',action='store_true')
    args=parser.parse_args();base=args.base.resolve();output=(args.output or base/'extension_r040_r080').resolve()
    logging.basicConfig(level=logging.WARNING,format='%(asctime)s %(levelname)s %(message)s')
    if args.worker:worker(base,output,args.worker,args.limit,args.parity_only);return
    if args.summarize_only:summarize(base,output);return
    prepare(base,output,args.weights)
    if args.prepare_only:return
    write(output/'run.pid',os.getpid())
    for ds in DATASETS:
        print(f'START {ds}: '+('cached 0.60 parity only' if args.parity_only else 'new Full Pack weights '+str(args.weights)),flush=True)
        command=[sys.executable,'-B','-u',str(output/'source_snapshot/scripts'/SCRIPT.name),
                 '--base',str(base),'--output',str(output),'--worker',ds]
        if args.parity_only:command.append('--parity-only')
        subprocess.run(command,check=True)
    if not args.parity_only:
        load_environment(output);summarize(base,output);audit(base,output)
        print(f'COMPLETE: {args.weights} on all 200 frozen questions; {200*len(args.weights)} additional answers',flush=True)


if __name__=='__main__':main()

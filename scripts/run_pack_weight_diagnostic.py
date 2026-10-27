#!/usr/bin/env python3
"""Outcome-stratified 200-question diagnostic of Full Pack R=0.25 vs R=0.60.

The cohorts use historical outputs only and are frozen before new inference.
Each question computes Route+Plan once, saves the actual Pack input, and runs
both native Full Pack settings on independent copies of that same input.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path('/root/BindPack')
ARCHIVE = PROJECT/'outputs/4_datasets_result'
DATASETS = ('hotpotqa', '2wikimultihopqa', 'musique', 'hover')
GROUPS = ('similar', 'expected_r025', 'expected_r060')
CASES = ('full_r025', 'full_r060')


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def append(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')
        stream.flush(); os.fsync(stream.fileno())


def jsonable(value):
    if isinstance(value, dict): return {str(k): jsonable(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)): return [jsonable(v) for v in sorted(value)]
    if hasattr(value, 'tolist'): return value.tolist()
    return value


def canonical_corpus(corpus):
    """Match the independent evaluator for legacy title/text corpus objects."""
    return [item if isinstance(item,str) else item.get('title','')+'\n'+
            (item.get('text') or item.get('paragraph_text','')) for item in corpus]


def freeze_pack_input(query, ids, scores, state):
    keys = ('evidence_candidates', 'evidence_trace', '_evidence_plan',
            '_evidence_winning_bindings', '_evidence_beams')
    return jsonable(dict(query=query, ids=ids, scores=scores,
                         state={key: deepcopy(state[key]) for key in keys if key in state}))


def restore_pack_input(payload):
    state = deepcopy(payload['state'])
    state['evidence_candidates'] = {int(k): v for k,v in state.get('evidence_candidates', {}).items()}
    return state


def choose_group(rows, size, seed=42):
    """Round-robin native hop strata, with deterministic historical-priority ties."""
    groups = defaultdict(list)
    for row in rows: groups[str(row['benchmark_hops'])].append(row)
    for values in groups.values():
        values.sort(key=lambda r: (r.get('priority', 0), digest([seed, r['sample_id']])))
    result = []
    while len(result) < size:
        moved = False
        for key in sorted(groups):
            if groups[key] and len(result) < size:
                result.append(groups[key].pop(0)); moved = True
        if not moved: raise ValueError(f'Not enough historical candidates for cohort: {size}')
    return result


def prepare(output):
    if (output/'selection_manifest.json').is_file():
        manifest = read(output/'selection_manifest.json')
        assert digest(manifest) == read(output/'selection_locked.ok')['sha256']
        return
    output.mkdir(parents=True, exist_ok=True)
    relocation = read(ARCHIVE/'relocation_map.json')
    manifest = dict(schema_version=1, purpose='outcome-stratified diagnostic, not representative evaluation',
        selection_seed=42, selection_uses_historical_outcomes=True,
        selection_uses_new_r060_outcomes=False, counts_per_dataset=dict(similar=10, expected_r025=20, expected_r060=20),
        cohort_interpretation='Predicted preferences from old Full vs w/o Pack, not measured 0.25 vs 0.60 preferences; all selected outcomes are retained.',
        datasets={})
    for dataset in DATASETS:
        folder = ARCHIVE/'cases'/dataset
        full, wo = read(folder/'full/result.json'), read(folder/'wo_pack/result.json')
        fd = read(folder/'full/metrics_details.json')['per_query']
        wd = read(folder/'wo_pack/metrics_details.json')['per_query']
        local = read(ARCHIVE/'analysis'/f'{dataset}_pairwise_objectives.json')
        predicted = {row['sample_id'] for row in local if row['restore_relation_weight_0_6_pair_reverses']}
        pools = {group: [] for group in GROUPS}
        for a,b,ad,bd in zip(full['results'], wo['results'], fd, wd):
            assert a['sample_id'] == b['sample_id'] == ad['sample_id'] == bd['sample_id']
            ac,bc = ad['at_k']['5']['FCR'],bd['at_k']['5']['FCR']
            if ac > bc:
                group, reason, priority = 'expected_r025','historical Full FCR@5 > w/o Pack FCR@5',0
            elif bc > ac:
                group = 'expected_r060'
                reason = 'historical Full lost complete evidence; local R=0.60 pair reversal' if a['sample_id'] in predicted else 'historical w/o Pack FCR@5 > Full; weaker preference proxy'
                priority = 0 if a['sample_id'] in predicted else 1
            elif ad['at_k']['5']['Recall'] > bd['at_k']['5']['Recall']:
                group,reason,priority='expected_r025','historical Full Recall@5 > w/o Pack; weaker preference proxy',2
            elif ad['at_k']['5']['Recall'] < bd['at_k']['5']['Recall']:
                group,reason,priority='expected_r060','historical w/o Pack Recall@5 > Full; weaker preference proxy',2
            elif ad['qa']['F1'] > bd['qa']['F1']:
                group,reason,priority='expected_r025','historical Full answer F1 > w/o Pack; weaker preference proxy',3
            elif ad['qa']['F1'] < bd['qa']['F1']:
                group,reason,priority='expected_r060','historical w/o Pack answer F1 > Full; weaker preference proxy',3
            elif a['candidate_docs'][:5] == b['candidate_docs'][:5]:
                group,reason,priority = 'similar','historical Full and w/o Pack have identical ordered Top5',0
            else: continue
            pools[group].append(dict(sample_id=a['sample_id'], query_index=a['query_index'],
                benchmark_hops=a['benchmark_hops'], question=a['question'], group=group, reason=reason, priority=priority,
                historical_full_fcr5=ac, historical_wo_pack_fcr5=bc,
                historical_full_em=ad['qa']['ExactMatch'], historical_wo_pack_em=bd['qa']['ExactMatch']))
        chosen = [r for group,size in zip(GROUPS,(10,20,20)) for r in choose_group(pools[group],size)]
        assert len(chosen) == len({r['sample_id'] for r in chosen}) == 50
        # Interleave the three cohorts; no inference outcome can affect order.
        chosen.sort(key=lambda r: digest([42,dataset,r['sample_id']]))
        report = read(folder/'full/report.json')
        metadata = output/'metadata'/dataset; metadata.mkdir(parents=True,exist_ok=True)
        benchmark = ARCHIVE/relocation[report['inputs']['benchmark']['path']]
        corpus = ARCHIVE/relocation[report['inputs']['corpus']['path']]
        shutil.copy2(benchmark, metadata/'benchmark.json')
        shutil.copy2(corpus, metadata/'corpus.json')
        identity = dict(dataset=dataset, data_path=str(metadata/'benchmark.json'),data_sha256=sha(metadata/'benchmark.json'),
            selected_indices=[r['query_index'] for r in chosen], sample_ids=[r['sample_id'] for r in chosen],
            corpus_docs=report['inputs']['corpus'].get('corpus_docs',len(read(corpus))),
            corpus_docs_sha256=digest(canonical_corpus(read(corpus))), development_fixture=True,
            diagnostic_selection=True, selection_sha256=digest(chosen),
            hop_provenance=full.get('hop_provenance'))
        write(metadata/'sample_manifest.json',identity)
        config = deepcopy(full['runtime_config'])
        method = {key: config.pop(key) for key in ('candidates','route','plan','pack')}
        config.pop('stage_ablation',None)
        config['save_dir'] = str(output/'runtime'/dataset/'index')
        config.update(dataset=dataset,qa_top_k=5,retrieval_top_k=200,llm_prefetch_workers=8,embedding_batch_size=4)
        method['runtime'] = config
        for weight,name in zip((.25,.60),CASES):
            params=deepcopy(method);params['pack']['relation_weight']=weight
            write(metadata/(name+'_method.json'),params)
        a,b=[read(metadata/(name+'_method.json')) for name in CASES]
        b['pack']['relation_weight']=a['pack']['relation_weight']; assert a==b
        manifest['datasets'][dataset]=dict(rows=chosen, available_cohorts={k:len(v) for k,v in pools.items()},
            source_result_sha256=sha(folder/'full/result.json'), source_wo_pack_sha256=sha(folder/'wo_pack/result.json'),
            historical_hop_provenance=full.get('hop_provenance'))
        print(f'FROZEN {dataset}: 10 similar + 20 expected r025 + 20 expected r060',flush=True)
    write(output/'selection_manifest.json',manifest)
    write(output/'selection_locked.ok',dict(sha256=digest(manifest),queries=200,locked_before_new_inference=True))
    snapshot=output/'source_snapshot'
    for name in ('src','scripts','configs'):
        shutil.copytree(PROJECT/name,snapshot/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'),dirs_exist_ok=True)
    write(output/'source_fingerprint.json',{str(p.relative_to(snapshot)):sha(p) for p in snapshot.rglob('*.py')})


def private_index(output,dataset):
    shared=ARCHIVE/'shared_indexes'/dataset
    dest=output/'runtime'/dataset/'index'
    dest.mkdir(parents=True,exist_ok=True)
    # Large vector files are strictly read-only in retrieval. Everything that
    # native loading may write has a private copy; no rag.index() is called.
    model=next(p for p in shared.iterdir() if p.is_dir() and p.name!='llm_cache')
    for file in ('graph.pickle','index_manifest.json','chunk_metadata.json'):
        if (model/file).is_file() and not (dest/model.name/file).exists():
            (dest/model.name).mkdir(exist_ok=True)
            shutil.copy2(model/file,dest/model.name/file)
    for directory in ('chunk_embeddings','entity_embeddings','fact_embeddings'):
        target=dest/model.name/directory;target.mkdir(parents=True,exist_ok=True)
        for src in (model/directory).glob('*.parquet'):
            dst=target/src.name
            if not dst.exists():dst.symlink_to(src)
    for src in shared.glob('openie_results*.json'):
        dst=dest/src.name
        if not dst.exists():dst.symlink_to(src)
    cache_source=ARCHIVE/'cases'/dataset/'full/index/llm_cache'
    cache=dest/'llm_cache';cache.mkdir(exist_ok=True)
    for src in cache_source.glob('*.sqlite'):
        if not (cache/src.name).exists():shutil.copy2(src,cache/src.name)
    return dest


def worker(output,dataset,limit=None):
    sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
    import numpy as np
    from transformers import AutoTokenizer
    from bindpack import BindPack, BindPackConfig
    from bindpack.evaluation.measurement import Recorder,ResourceMonitor,usage,jsonl
    from bindpack.evaluation.measured_reader import prepare as prepare_reader
    from bindpack.evaluation.result_export import detailed_result
    from bindpack.evaluation.metric_report import write_metric_report
    from bindpack.llm.openai_gpt import CacheOpenAI
    from bindpack.utils.misc_utils import QuerySolution
    from eval_utils import get_gold_answers,get_gold_docs
    selection=read(output/'selection_manifest.json')
    assert digest(selection)==read(output/'selection_locked.ok')['sha256']
    selected=selection['datasets'][dataset]['rows']
    pair_log=output/'runtime'/dataset/'pairs.jsonl'
    pairs=list(jsonl(pair_log))
    for pair,expected in zip(pairs,selected):
        assert pair['sample_id']==expected['sample_id'] and pair['sha256']==digest(pair['payload'])
    if len(pairs)==50:
        finish_dataset(output,dataset,pairs);return
    metadata=output/'metadata'/dataset
    data=read(metadata/'benchmark.json')
    recorder=Recorder(output/'runtime'/dataset/'accounting')
    tokenizer=AutoTokenizer.from_pretrained('/root/models/Qwen3-8B',local_files_only=True)
    recorder.install_llm_factory(CacheOpenAI,tokenizer)
    index=private_index(output,dataset)
    cfg=BindPackConfig.from_file(metadata/'full_r025_method.json',save_dir=str(index),stage_ablation='full')
    monitor=ResourceMonitor(recorder)
    start=time.perf_counter()
    with monitor:
        with recorder.section('model_and_index_loading'):
            rag=BindPack(config=cfg)
            recorder.install_embedding(rag.embedding_model)
            rag.prepare_retrieval_objects();rag._build_index_side_artifacts()
        print(f'INDEX READY {dataset}; model+index={time.perf_counter()-start:.1f}s; reuse corpus={len(rag.passage_node_keys)}',flush=True)
        runtime=rag.evidence_runtime
        native_finalize=runtime.pack.finalize
        native_review=runtime.pack.postprocess
        captured={}
        def capture(query,ids,scores,context,state):
            if cfg.pack.relation_weight != .25:raise RuntimeError('Unexpected weight in shared upstream generation')
            captured['input']=freeze_pack_input(query,ids,scores,state)
            write(captured['file'],dict(input=captured['input'],input_sha256=digest(captured['input']),upstream_seconds=None))
            begin=time.perf_counter()
            with recorder.section('pack_r025_selection',captured['sample_id']):
                answer=native_finalize(query,ids,scores,context,state)
            captured['selection_seconds']=time.perf_counter()-begin
            return answer
        def capture_review(inputs):
            begin=time.perf_counter()
            with recorder.section('pack_r025_review',captured['sample_id']):answer=native_review(inputs)
            captured['review_seconds']=time.perf_counter()-begin
            return answer
        runtime.pack.finalize=capture
        runtime.pack.postprocess=capture_review
        for ordinal in range(len(pairs),min(len(selected),limit or len(selected))):
            row=selected[ordinal];qid=row['sample_id'];query=row['question']
            filename=output/'frozen_inputs'/dataset/(hashlib.sha256(str(qid).encode()).hexdigest()+'.json')
            captured.clear();captured.update(file=filename,sample_id=qid)
            begin=time.perf_counter()
            cfg.pack.relation_weight=.25
            if filename.is_file():
                stored=read(filename);frozen=stored['input'];assert digest(frozen)==stored['input_sha256']
                shared_seconds=stored['upstream_seconds'];solutions={}
            else:
                with recorder.section('shared_route_plan',qid):
                    rag.set_query_hop_overrides([row['benchmark_hops']],source='historical_fixed_for_pack_comparison')
                    actual=rag.retrieve([query],num_to_retrieve=200)[0]
                frozen=captured['input']
                shared_seconds=time.perf_counter()-begin-captured['selection_seconds']-captured['review_seconds']
                write(filename,dict(input=frozen,input_sha256=digest(frozen),upstream_seconds=shared_seconds))
                solutions={'full_r025':(actual,captured['selection_seconds']+captured['review_seconds'])}
            invariant=digest(frozen)
            exports={}
            for name,weight in zip(CASES,(.25,.60)):
                cfg.pack.relation_weight=weight
                if name in solutions:
                    solution,pack_seconds=solutions[name]
                else:
                    state=restore_pack_input(frozen)
                    stage_start=time.perf_counter()
                    with recorder.section(f'pack_{name[5:]}_selection',qid):
                        ids,scores,trace=native_finalize(query,np.asarray(frozen['ids'],dtype=int),np.asarray(frozen['scores'],dtype=float),{},state)
                    solution=QuerySolution(question=query,docs=[runtime._document(int(d)) for d in ids[:200]],
                        doc_scores=scores[:200],retrieval_trace={'stage_ablation':'full','evidence':trace})
                    with recorder.section(f'pack_{name[5:]}_review',qid):native_review([(solution,ids[:200].copy())])
                    pack_seconds=time.perf_counter()-stage_start
                assert digest(frozen)==invariant, 'A Pack arm mutated the shared frozen input'
                with recorder.section(f'reader_{name[5:]}',qid):
                    reading_start=time.perf_counter()
                    reading,budget=prepare_reader(rag,solution,tokenizer,dataset)
                    answered,raw,metas=rag.qa([reading])
                    solution.answer=answered[0].answer
                    reading_seconds=time.perf_counter()-reading_start
                sample=data[row['query_index']]
                solution.gold_answers=get_gold_answers([sample],dataset)[0]
                solution.gold_docs=sorted(set(get_gold_docs([sample],dataset)[0]))
                export=detailed_result(solution,sample,row['query_index'],solution.gold_docs,row['benchmark_hops'],10,200)
                export.update(ranking_complete=True,actual_ranking_length=len(solution.docs),
                    diagnostic_cohort=row['group'],pack_relation_weight=weight,frozen_pack_input_sha256=invariant,
                    timing=dict(shared_route_plan_seconds=shared_seconds,pack_seconds=pack_seconds,reader_seconds=reading_seconds,
                        end_to_end_seconds=None if shared_seconds is None else shared_seconds+pack_seconds+reading_seconds),
                    reader=dict(raw_response=raw[0],metadata=metas[0],budget=budget))
                exports[name]=jsonable(export)
            cfg.pack.relation_weight=.25
            assert exports[CASES[0]]['frozen_pack_input_sha256']==exports[CASES[1]]['frozen_pack_input_sha256']
            payload=dict(rows=exports,cohort=row['group'],frozen_pack_input_sha256=invariant)
            event=dict(sample_id=qid,payload=payload,sha256=digest(payload))
            append(pair_log,event);pairs.append(event)
            write(output/'status.json',dict(dataset=dataset,completed_dataset_pairs=len(pairs),expected_dataset_pairs=50,
                total_completed_pairs=sum(sum(1 for _ in jsonl(output/'runtime'/ds/'pairs.jsonl')) for ds in DATASETS),
                expected_total_pairs=200,updated_at=datetime.now().astimezone().isoformat()))
            print(f'PAIR {dataset} [{"#"*int(24*len(pairs)/50):24}] {len(pairs)}/50; {row["group"]}; '
                f'{time.perf_counter()-begin:.2f}s; FCR5 .25={exports[CASES[0]]["all_gold_in_top5"]} .60={exports[CASES[1]]["all_gold_in_top5"]}; '
                f'answers .25={exports[CASES[0]]["answer"]!r} .60={exports[CASES[1]]["answer"]!r}',flush=True)
    write(output/'runtime'/dataset/'phase_timings.json',recorder.phases)
    for obj,name,original in reversed(recorder.restorations):setattr(obj,name,original)
    if len(pairs)==50:finish_dataset(output,dataset,pairs)
    summarize(output)


def finish_dataset(output,dataset,pairs):
    sys.path.insert(0,str(ROOT/'src'))
    from bindpack.evaluation.metric_report import write_metric_report
    from bindpack.evaluation.measurement import jsonl,usage
    metadata=output/'metadata'/dataset
    identity=read(metadata/'sample_manifest.json')
    selection=read(output/'selection_manifest.json')
    for name,weight in zip(CASES,(.25,.60)):
        directory=output/'cases'/dataset/name;directory.mkdir(parents=True,exist_ok=True)
        method=read(metadata/(name+'_method.json'))
        protocol=dict(stage_ablation='full',diagnostic=True,representative_evaluation=False,relation_weight=weight,
            selection_sha256=digest(selection),same_route_plan_input_per_pair=True,
            unchanged_other_method_parameters=True,models=dict(llm='/root/models/Qwen3-8B',embedding='/root/models/Qwen3-Embedding-8B'),
            cache_policy='private copy of historical Full cache; both Pack arms reuse one Route+Plan input; cache hits and physical requests recorded; not fresh-run cost measurement',
            hop_condition=selection['datasets'][dataset]['historical_hop_provenance'])
        write(directory/'protocol.json',protocol)
        result=dict(dataset=dataset,method='bindpack',stage_ablation='full',eval_mode='rag_qa',
            data_path=str(metadata/'benchmark.json'),corpus_path=str(metadata/'corpus.json'),
            selected_indices=identity['selected_indices'],sample_size_effective=50,
            runtime_config=dict(method['runtime'],**{k:method[k] for k in ('pack','plan','route','candidates')}),
            diagnostic_selection=True,pack_relation_weight=weight,hop_provenance=selection['datasets'][dataset]['historical_hop_provenance'],
            results=[p['payload']['rows'][name] for p in pairs])
        write(directory/'result.json',result)
        write(directory/'validated.ok',dict(result_sha256=sha(directory/'result.json'),protocol_sha256=digest(protocol),sample_count=50))
        report=write_metric_report(directory/'result.json',data_path=metadata/'benchmark.json',manifest_path=metadata/'sample_manifest.json',ks=(3,5,8,10,20,200))
        print(f'REPORT {dataset}/{name}: '+json.dumps({k:report['metrics'].get(k) for k in ('Recall@5','FCR@5','ExactMatch','F1','ClassificationMacroF1','DCC@5')},ensure_ascii=False),flush=True)
    calls=list(jsonl(output/'runtime'/dataset/'accounting/llm_calls.jsonl'))
    logical=list(jsonl(output/'runtime'/dataset/'accounting/llm_logical_calls.jsonl'))
    phases=sorted({c['phase'] for c in calls+logical})
    write(output/'runtime'/dataset/'cost.json',dict(physical=usage(calls),logical_calls=len(logical),
        cache_hits=sum(bool(x.get('cache_hit')) for x in logical),
        by_phase={phase:dict(physical=usage([c for c in calls if c['phase']==phase]),logical_calls=sum(c['phase']==phase for c in logical)) for phase in phases},
        scope='Actual incremental diagnostic costs; historical cache reused, shared Route+Plan charged once; no new index construction.'))
    write(output/'cases'/dataset/'completed.ok',dict(pairs=50,selection_sha256=digest(selection)))


def summarize(output):
    rows=[]
    for dataset in DATASETS:
        reports=[]
        for name in CASES:
            path=output/'cases'/dataset/name/'report.json'
            if path.is_file():reports.append(read(path))
        if len(reports)!=2:continue
        details=[read(output/'cases'/dataset/name/'metrics_details.json')['per_query'] for name in CASES]
        selection=read(output/'selection_manifest.json')['datasets'][dataset]['rows']
        groups=[]
        for group in GROUPS:
            inds=[i for i,r in enumerate(selection) if r['group']==group]
            arms=[]
            for arm in details:
                arms.append({key:sum((arm[i]['at_k']['5'][key] if key in ('Recall','FCR') else arm[i]['qa'][key]) for i in inds)/len(inds)
                    for key in ('Recall','FCR','ExactMatch','F1')})
            groups.append(dict(group=group,n=len(inds),r025=arms[0],r060=arms[1]))
        rows.append(dict(dataset=dataset,n=50,r025=reports[0]['metrics'],r060=reports[1]['metrics'],groups=groups,
            fcr_r060_only=sum(a['at_k']['5']['FCR']<b['at_k']['5']['FCR'] for a,b in zip(*details)),
            fcr_r025_only=sum(a['at_k']['5']['FCR']>b['at_k']['5']['FCR'] for a,b in zip(*details))))
    write(output/'comparison.json',dict(purpose='outcome-stratified diagnostic, not representative evaluation',datasets=rows))
    lines=['# Full Pack 相关性权重诊断：0.25 vs 0.60','','每个数据集固定 10 条接近、20 条预计偏向 0.25、20 条预计偏向 0.60。预期来自历史 Full/w/o Pack，不保证新两组的实际偏好。样本清单在新推理前冻结；不按新结果换题。','','每题 Route+Plan 只计算一次，两个 Full 设置共用相同输入快照，保留原 DAG 和末端语义审核，Reader@5。仅修改 relation_weight。以下分数仅适用于此诊断集。','',
        '| 数据集 | 权重 | Recall@5 | FCR@5 | EM / Accuracy | token F1 / Macro-F1 | DCC@5 |','|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        for weight,key in ((.25,'r025'),(.60,'r060')):
            m=row[key];q='ClassificationAccuracy' if row['dataset']=='hover' else 'ExactMatch';f='ClassificationMacroF1' if row['dataset']=='hover' else 'F1'
            vals=[m.get(k) for k in ('Recall@5','FCR@5',q,f,'DCC@5')]
            lines.append(f'| {row["dataset"]} | {weight:.2f} | '+' | '.join('N/A' if v is None else f'{v:.6f}' for v in vals)+' |')
    lines+=['','原始报告在 `cases/<dataset>/<full_r025|full_r060>/report.json`，逐题输入在 `frozen_inputs/`，实际请求/token账本在 `runtime/<dataset>/accounting/`。历史缓存复用，不能把本轮新增请求数当成新框架的完整线上成本。','',
        '历史 HotpotQA、2Wiki 仍保留原来的 hop prior=2，以便只比较 Pack 权重；本轮不是修正 hop 后的正式主实验。']
    (output/'comparison.md').write_text('\n'.join(lines)+'\n')
    if len(rows)==4:write(output/'completed.ok',dict(pairs=200,answer_evaluations=400,selection_sha256=read(output/'selection_locked.ok')['sha256']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=str(PROJECT/'outputs/pack_weight_diagnostic_20261025'))
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--worker',choices=DATASETS)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--summarize-only',action='store_true')
    args=parser.parse_args();output=Path(args.output).resolve()
    logging.basicConfig(level=logging.WARNING,format='%(asctime)s %(levelname)s %(message)s')
    if args.summarize_only:
        sys.path.insert(0,str(ROOT/'src'));summarize(output);return
    if args.worker:worker(output,args.worker,args.limit);return
    prepare(output)
    if args.prepare_only:return
    snapshot=output/'source_snapshot'
    for dataset in DATASETS:
        print(f'START {dataset} 50 paired questions',flush=True)
        command=[sys.executable,'-B','-u',str(snapshot/'scripts/run_pack_weight_diagnostic.py'),
                 '--output',str(output),'--worker',dataset]
        subprocess.run(command,check=True)
    sys.path.insert(0,str(snapshot/'src'));summarize(output)


if __name__=='__main__':main()

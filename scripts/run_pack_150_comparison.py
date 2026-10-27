#!/usr/bin/env python3
"""Paired before/after Pack evaluation on locked old50 + new100 per dataset."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PROJECT=Path('/root/BindPack')
OLD=PROJECT/'outputs/pack_candidates_r060_20261026'
COMPONENT=PROJECT/'outputs/component_diagnostic_r060_20261026'
EXPANSION=PROJECT/'outputs/pack_r060_expanded_20261025'
WEIGHTS=PROJECT/'outputs/pack_weight_diagnostic_20261025'
DEFAULT=PROJECT/'outputs/pack_150_before_after_20261026'
DATASETS=('hotpotqa','2wikimultihopqa','musique','hover')
ARMS=('before','after')
COHORTS=('old50','new100')
SCRIPT=Path(__file__).resolve()


def read(p):return json.loads(Path(p).read_text())
def digest(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def token(qid):return hashlib.sha256(str(qid).encode()).hexdigest()+'.json'
def cache_dirs(root,ds,cohort):
    if cohort not in ('combined150',)+COHORTS:raise ValueError('Unknown comparison cohort')
    names=COHORTS if cohort=='combined150' else (cohort,)
    return {name:str(root/'runtime'/ds/name/'index') for name in names}
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    with t.open('w') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    t.replace(p)


def payload(chosen):
    saved=read(chosen['frozen_path']);assert sha(chosen['frozen_path'])==chosen['frozen_file_sha256']
    if chosen['cohort']=='old50':
        bundle=saved['payload'];assert digest(bundle)==saved['sha256'];value=bundle['input']
    else:
        value=saved['input'];assert digest(value)==saved['input_sha256']
    assert digest(value)==chosen['frozen_pack_input_sha256'] and value['query']==chosen['question']
    return value


def prepare(root):
    if (root/'prepared.ok').exists():
        assert digest(read(root/'selection_manifest.json'))==read(root/'selection_locked.ok')['sha256']
        common=root/'source_snapshot/src'
        if not common.exists():common.symlink_to('before/src',target_is_directory=True)
        return
    assert read(OLD/'audit.json')['status']==read(EXPANSION/'audit.json')['status']=='passed'
    promotion=read(OLD/'production_promotion.json');assert promotion['status']=='applied_and_verified'
    assert all(sha(PROJECT/rel)==value for rel,value in promotion['changes'].items())
    old_selection=read(OLD/'selection_manifest.json');new_selection=read(EXPANSION/'selection_manifest.json')
    assert digest(old_selection)==read(OLD/'selection_locked.ok')['sha256']
    assert digest(new_selection)==read(EXPANSION/'selection_locked.ok')['sha256']
    root.mkdir(parents=True,exist_ok=True);selection=dict(datasets={},unique_questions=600,per_dataset=150,
        cohorts=dict(old50='Historical outcome-selected diagnostic',new100='Previous outcome-blind hash sample; not selected again'),
        original_old50_sha256=digest(old_selection),original_new100_sha256=digest(new_selection))
    input_hashes={};source_hashes={}
    for arm,source in (('before',OLD/'variants/baseline/src'),('after',PROJECT/'src')):
        target=root/'source_snapshot'/arm/'src'
        shutil.copytree(source,target,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        source_hashes[arm]={str(p.relative_to(target)):sha(p) for p in target.rglob('*') if p.is_file()}
    different={rel for rel,value in source_hashes['before'].items() if source_hashes['after'].get(rel)!=value}
    assert different=={p.removeprefix('src/') for p in promotion['changes']},different
    assert source_hashes['before']['bindpack/prompts/templates/rag_qa_hover.py']==source_hashes['after']['bindpack/prompts/templates/rag_qa_hover.py']
    (root/'source_snapshot/src').symlink_to('before/src',target_is_directory=True)
    for ds in DATASETS:
        selected=[]
        for cohort,rows0 in (('old50',old_selection['datasets'][ds]['rows']),('new100',new_selection['datasets'][ds]['rows'])):
            assert len(rows0)==(50 if cohort=='old50' else 100)
            for original in rows0:
                chosen={k:original[k] for k in ('sample_id','query_index','question','benchmark_hops')}
                chosen['cohort']=cohort
                frozen=(COMPONENT/'bundles'/ds/token(chosen['sample_id'])) if cohort=='old50' else (EXPANSION/'frozen_inputs'/ds/token(chosen['sample_id']))
                chosen.update(frozen_path=str(frozen),frozen_file_sha256=sha(frozen))
                value=read(frozen);value=value['payload']['input'] if cohort=='old50' else value['input']
                chosen['frozen_pack_input_sha256']=digest(value);payload(chosen);selected.append(chosen)
        assert len({v['sample_id'] for v in selected})==len({v['query_index'] for v in selected})==150
        selection['datasets'][ds]=dict(rows=selected)
        meta=root/'metadata'/ds;meta.mkdir(parents=True,exist_ok=True)
        for name in ('benchmark.json','corpus.json'):
            (meta/name).symlink_to(EXPANSION/'metadata'/ds/name)
        identity=read(EXPANSION/'metadata'/ds/'sample_manifest.json')
        identity.update(data_path=str(meta/'benchmark.json'),selected_indices=[q['query_index'] for q in selected],
            sample_ids=[q['sample_id'] for q in selected],diagnostic_selection=True,selection_sha256=digest(selected))
        write(meta/'sample_manifest.json',identity)
        method=read(EXPANSION/'metadata'/ds/'full_r060_method.json');assert method['pack']['relation_weight']==.6
        assert method['runtime']['qa_top_k']==5
        method['runtime']['save_dir']=None  # The worker selects its cohort-specific cache.
        write(meta/'method.json',method)
        references={}
        for cohort in COHORTS:
            references[cohort]={}
            for arm in ARMS:
                reference=OLD/'cases'/ds/('baseline' if arm=='before' else 'combined')/'result.json' if cohort=='old50' else EXPANSION/'cases'/ds/'new100/result.json'
                references[cohort][arm]=dict(path=str(reference),sha256=sha(reference))
            cache_source=(OLD/'runtime'/ds/'index/llm_cache') if cohort=='old50' else (EXPANSION/'runtime'/ds/'index/llm_cache')
            cache=root/'runtime'/ds/cohort/'index/llm_cache';cache.mkdir(parents=True,exist_ok=True)
            for p in cache_source.glob('*.sqlite'):shutil.copy2(p,cache/p.name)
        input_hashes[ds]=dict(references=references,benchmark_sha256=sha(meta/'benchmark.json'),corpus_sha256=sha(meta/'corpus.json'))
    write(root/'selection_manifest.json',selection);write(root/'selection_locked.ok',dict(sha256=digest(selection),unique_questions=600,locked_before_inference=True))
    write(root/'source_fingerprint.json',source_hashes)
    write(root/'experiment.json',dict(relation_weight=.6,reader_top_k=5,paired=True,diagnostic_only=True,
        before='Original Pack R=.60 with the current common reader',after='Goal-aware redundancy + local request short-circuit; R=.60',
        reader_control='Identical current reader in both arms. New100 HoVer baseline is answered again because its historical reader differs.',
        route_plan_frozen=True,index_rebuilt=False,embedding_inference=False,inputs=input_hashes,
        cohort_control='Same old50+new100 IDs as the previous expansion; report both strata and pooled150 separately.',
        hop_policy='Retain the historical frozen-input protocol; HotpotQA/2Wiki prior=2, existing MuSiQue/HoVer oracle hints. Not a corrected-hop main experiment.',
        cost_scope='Incremental warm-cache Pack+reader. Excludes frozen Route/Plan, model initialization and index construction; no end-to-end speed claim.',
        models=dict(llm='/root/models/Qwen3-8B',embedding='/root/models/Qwen3-Embedding-8B'),
        new_selection=False,algorithm_retuning=False,production_source_modified=False))
    scripts=root/'source_snapshot/scripts';scripts.mkdir(parents=True,exist_ok=True)
    for name in (SCRIPT.name,'run_pack_weight_sweep.py','eval_utils.py'):
        shutil.copy2(PROJECT/'scripts'/name,scripts/name)
    write(root/'prepared.ok',dict(selection_sha256=digest(selection),new_model_requests=0))


def completed(root,ds,arm):
    values=[]
    for chosen in read(root/'selection_manifest.json')['datasets'][ds]['rows']:
        p=root/'checkpoints'/ds/arm/token(chosen['sample_id'])
        if not p.exists():break
        saved=read(p);assert saved['sha256']==digest(saved['row']);assert saved['row']['sample_id']==chosen['sample_id']
        values.append(saved['row'])
    return values


def load_sweep(root):
    path=root/'source_snapshot/scripts/run_pack_weight_sweep.py'
    spec=importlib.util.spec_from_file_location('paired150_native_helpers',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def finish(root,ds,arm):
    from bindpack.evaluation.metric_report import write_metric_report
    from bindpack.evaluation.measurement import distribution
    values=completed(root,ds,arm);assert len(values)==150
    meta=root/'metadata'/ds;method=read(meta/'method.json');identity=read(meta/'sample_manifest.json')
    for cohort in ('combined150',)+COHORTS:
        rows=values if cohort=='combined150' else [v for v in values if v['comparison_cohort']==cohort]
        case=root/'cases'/ds/arm if cohort=='combined150' else root/'strata'/ds/cohort/arm
        manifest=dict(identity,selected_indices=[v['query_index'] for v in rows],sample_ids=[v['sample_id'] for v in rows],selection_sha256=digest([v['sample_id'] for v in rows]))
        caches=cache_dirs(root,ds,cohort)
        write(case/'sample_manifest.json',manifest)
        if not (case/'corpus.json').exists():(case/'corpus.json').symlink_to(meta/'corpus.json')
        protocol=dict(stage_ablation='full',arm=arm,cohort=cohort,sample_count=len(rows),relation_weight=.6,reader_top_k=5,
            selection_sha256=read(root/'selection_locked.ok')['sha256'],common_reader=True,route_plan_frozen=True,index_rebuilt=False,
            cache_directories_by_cohort=caches)
        write(case/'protocol.json',protocol)
        result=dict(dataset=ds,method='bindpack',stage_ablation='full',eval_mode='rag_qa',data_path=str(meta/'benchmark.json'),
            corpus_path=str(meta/'corpus.json'),selected_indices=manifest['selected_indices'],sample_size_effective=len(rows),
            runtime_config=dict(method['runtime'],save_dir=None if cohort=='combined150' else caches[cohort],
                **{k:method[k] for k in ('pack','plan','route','candidates')}),
            cache_directories_by_cohort=caches,
            pack_relation_weight=.6,comparison_arm=arm,comparison_cohort=cohort,diagnostic_selection=True,hop_provenance=identity.get('hop_provenance'),results=rows)
        write(case/'result.json',result);write(case/'validated.ok',dict(result_sha256=sha(case/'result.json'),protocol_sha256=digest(protocol),sample_count=len(rows)))
        report=write_metric_report(case/'result.json',data_path=meta/'benchmark.json',manifest_path=case/'sample_manifest.json',ks=(3,5,8,10,20,200))
        c=Counter()
        for row in rows:c.update(row['diagnostic_cost'])
        write(case/'cost.json',dict(committed_rows=dict(c),pack_seconds=distribution([v['timing']['pack_seconds'] for v in rows]),
            reader_seconds=distribution([v['timing']['reader_seconds'] for v in rows]),scope=read(root/'experiment.json')['cost_scope']))
        print('REPORT',ds,arm,cohort,json.dumps({k:report['metrics'].get(k) for k in ('Recall@5','FCR@5','ExactMatch','F1','ClassificationMacroF1','DCC@5')},ensure_ascii=False),flush=True)


def worker(root,ds,arm):
    sys.path.insert(0,str(root/'source_snapshot'/arm/'src'));sys.path.insert(0,str(root/'source_snapshot/scripts'))
    os.environ.setdefault('OPENAI_API_KEY','EMPTY')
    from transformers import AutoTokenizer
    from bindpack import BindPackConfig
    from bindpack.evaluation.measurement import Recorder,ResourceMonitor,jsonl
    from bindpack.evaluation.result_export import detailed_result
    from eval_utils import get_gold_answers,get_gold_docs
    if len(completed(root,ds,arm))==150:finish(root,ds,arm);return
    sweep=load_sweep(root);helper=sweep.helpers(WEIGHTS)
    tokenizer=AutoTokenizer.from_pretrained('/root/models/Qwen3-8B',local_files_only=True)
    store=sweep.ReadOnlyPassages(next((PROJECT/'outputs/4_datasets_result/shared_indexes'/ds).rglob('vdb_chunk.parquet')))
    chosen_rows=read(root/'selection_manifest.json')['datasets'][ds]['rows'];benchmark=read(root/'metadata'/ds/'benchmark.json')
    for cohort in COHORTS:
        cfg=BindPackConfig.from_file(root/'metadata'/ds/'method.json',save_dir=str(root/'runtime'/ds/cohort/'index'),stage_ablation='full')
        assert cfg.pack.relation_weight==.6 and cfg.qa_top_k==5
        recorder=Recorder(root/'accounting'/ds/arm/cohort)
        rag=sweep.make_runtime(cfg,store,recorder,tokenizer)
        ref=read(root/'experiment.json')['inputs'][ds]['references'][cohort][arm]
        assert sha(ref['path'])==ref['sha256'];expected={v['sample_id']:v for v in read(ref['path'])['results']}
        transport=rag.llm_model.openai_client.chat.completions;real=transport.create
        def guarded_create(*args,**kwargs):
            if cohort=='old50':raise AssertionError('Previously tested old50 generated a cache miss')
            if arm=='before' and not (ds=='hover' and recorder.phase.startswith('reader_')):
                raise AssertionError('Original new100 baseline ranking/reader cache parity failed')
            return real(*args,**kwargs)
        transport.create=guarded_create
        try:
            with ResourceMonitor(recorder) as monitor:
                for chosen in (v for v in chosen_rows if v['cohort']==cohort):
                    qid=chosen['sample_id'];checkpoint=root/'checkpoints'/ds/arm/token(qid)
                    if checkpoint.exists():continue
                    original=payload(chosen);t=time.perf_counter()
                    nl=len(jsonl(recorder.directory/'llm_logical_calls.jsonl'));nh=len(jsonl(recorder.directory/'llm_calls.jsonl'))
                    solution,reader,timing=sweep.run_arm(rag,tokenizer,recorder,original,.6,qid,helper)
                    reference=expected[qid]
                    if arm=='before' or cohort=='old50':
                        assert solution.docs==reference['candidate_docs'],(ds,arm,cohort,qid,'ranking parity')
                        if cohort=='old50' or ds!='hover':
                            assert solution.answer==reference['answer'],(ds,arm,cohort,qid,'answer parity')
                            assert reader['budget']['prompt_sha256']==reference['reader']['budget']['prompt_sha256'],(ds,qid,'reader prompt parity')
                    sample=benchmark[chosen['query_index']]
                    solution.gold_answers=get_gold_answers([sample],ds)[0];solution.gold_docs=sorted(set(get_gold_docs([sample],ds)[0]))
                    row=helper.jsonable(detailed_result(solution,sample,chosen['query_index'],solution.gold_docs,chosen['benchmark_hops'],10,200))
                    logical=jsonl(recorder.directory/'llm_logical_calls.jsonl')[nl:];http=jsonl(recorder.directory/'llm_calls.jsonl')[nh:]
                    costs=dict(logical_calls=len(logical),cache_hits=sum(bool(c.get('cache_hit')) for c in logical),http_attempts=len(http),
                        logical_input_tokens=sum(c.get('metadata',{}).get('prompt_tokens') or 0 for c in logical),logical_output_tokens=sum(c.get('metadata',{}).get('completion_tokens') or 0 for c in logical),
                        physical_input_tokens=sum(c.get('prompt_tokens') or 0 for c in http),physical_output_tokens=sum(c.get('completion_tokens') or 0 for c in http))
                    row.update(ranking_complete=True,actual_ranking_length=len(solution.docs),pack_relation_weight=.6,
                        comparison_arm=arm,comparison_cohort=cohort,frozen_pack_input_sha256=chosen['frozen_pack_input_sha256'],
                        reader=reader,timing=dict(timing,incremental_seconds=time.perf_counter()-t),diagnostic_cost=costs)
                    write(checkpoint,dict(row=row,sha256=digest(row)))
                    n=len(completed(root,ds,arm));write(root/'status.json',dict(dataset=ds,arm=arm,cohort=cohort,completed=n,expected=150,updated_at=datetime.now().astimezone().isoformat()))
                    print(f'QUERY {ds}/{arm} {n}/150 ({cohort}); {time.perf_counter()-t:.2f}s; calls={len(logical)}; HTTP={len(http)}; answer={solution.answer!r}',flush=True)
                write(recorder.directory/'resource_summary.json',monitor.summary())
        finally:
            transport.create=real
            for obj,name,native in reversed(recorder.restorations):setattr(obj,name,native)
    finish(root,ds,arm)


def summarize(root):
    lines=['# 固定每数据集150条：Pack优化前后对照','',
        '权重均为0.60；同一Route+Plan输入、同一当前reader、原始共享索引。old50为历史挑选诊断集，new100为此前确定的额外样本。','',
        '| 数据集 | 子集 | 版本 | Recall@5 | FCR@5 | EM / Accuracy | F1 / Macro-F1 | DCC@5 | NodeCov@5 | ClosureGap@5 | 逻辑调用 |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    summary={}
    for ds in DATASETS:
        summary[ds]={}
        for cohort in ('combined150',)+COHORTS:
            summary[ds][cohort]={}
            for arm in ARMS:
                case=root/'cases'/ds/arm if cohort=='combined150' else root/'strata'/ds/cohort/arm
                if not (case/'report.json').exists():continue
                m=read(case/'report.json')['metrics'];cost=read(case/'cost.json')['committed_rows'];summary[ds][cohort][arm]=dict(metrics=m,cost=cost)
                keys=('Recall@5','FCR@5','ClassificationAccuracy' if ds=='hover' else 'ExactMatch','ClassificationMacroF1' if ds=='hover' else 'F1','DCC@5','NodeCov@5','ClosureGap@5')
                lines.append(f'| {ds} | {cohort} | {arm} | '+' | '.join('N/A' if m.get(k) is None else f'{m[k]:.6f}' for k in keys)+f' | {cost["logical_calls"]} |')
    write(root/'comparison.json',summary);(root/'comparison.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT);p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--worker',choices=DATASETS);p.add_argument('--arm',choices=ARMS,default='before');args=p.parse_args();root=args.output.resolve()
    if args.worker:worker(root,args.worker,args.arm);return
    prepare(root)
    if args.prepare_only:return
    import fcntl
    with (root/'.run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(root/'run.pid').write_text(str(os.getpid())+'\n')
        for ds in DATASETS:
            for arm in ARMS:
                print('START',ds,arm,'fixed150 index-reuse frozen-Route/Plan',flush=True)
                subprocess.run([sys.executable,'-B','-u',str(root/'source_snapshot/scripts'/SCRIPT.name),'--output',str(root),'--worker',ds,'--arm',arm],check=True)
                summarize(root)
        write(root/'completed.ok',dict(status='complete',unique_questions=600,paired_predictions=1200,relation_weight=.6,production_source_modified=False))
        write(root/'status.json',dict(phase='complete',unique_questions=600,paired_predictions=1200))
        print('COMPLETE: four datasets x 150 paired before/after; production unchanged.',flush=True)


if __name__=='__main__':main()

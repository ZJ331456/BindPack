#!/usr/bin/env python3
"""Lock two disjoint 100-question cohorts and capture exact native Pack inputs."""
from pathlib import Path
from copy import deepcopy
from dataclasses import asdict
import argparse, hashlib, json, os, shutil, sys, time

PROJECT = Path('/root/BindPack')
sys.path[:0] = [str(PROJECT/'src'), str(PROJECT/'scripts')]
DATASETS = ('hotpotqa', '2wikimultihopqa', 'musique', 'hover')
ARCHIVE = PROJECT/'outputs/4_datasets_result/full_no_review_20261026'

def read(p): return json.loads(Path(p).read_text())
def digest(x): return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(',',':')).encode()).hexdigest()
def token(x): return hashlib.sha256(x.encode()).hexdigest()+'.json'
def write(p, x):
    p=Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp'); t.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False)+'\n');t.replace(p)

def prepare(root):
    if (root/'selection_locked.json').exists(): return
    manifest={}
    for ds in DATASETS:
        rows=[json.loads(s)['row'] for s in (ARCHIVE/'cases'/ds/'full/progress.jsonl').open() if s.strip()]
        assert len(rows)>=200
        ordered=sorted(rows, key=lambda r: digest(['pack-joint-fixed-20261027',ds,r['sample_id']]))[:200]
        manifest[ds]=[{'sample_id':r['sample_id'],'query_index':r['query_index'],
                       'question':r['question'],'benchmark_hops':r['benchmark_hops'],
                       'phase':'pilot' if i<100 else 'validation'} for i,r in enumerate(ordered)]
        meta=root/'metadata'/ds;meta.mkdir(parents=True,exist_ok=True)
        source=ARCHIVE/'runs'/ds
        for src,name in [(source/(ds+'.json'),'benchmark.json'),
                         (source/'metadata'/ds/'corpus.json','corpus.json')]:
            if not src.exists():
                src=PROJECT/'datasets'/(ds+'.json') if name=='benchmark.json' else src
            (meta/name).symlink_to(src)
    write(root/'selection_locked.json',{'datasets':manifest,'sha256':digest(manifest),
        'rule':'outcome-blind hash among durable completed rows; pilot100 then disjoint validation100',
        'development_diagnostic':True,'hover_available_completed':520})

def capture(root, ds):
    from bindpack import BindPack, BindPackConfig
    from bindpack.evaluation.measurement import Recorder,ResourceMonitor
    from utils.archived_index import prepare_archive,verify_private
    from run_pack_weight_diagnostic import freeze_pack_input
    from transformers import AutoTokenizer
    from eval_utils import build_docs_from_full_corpus
    chosen=read(root/'selection_locked.json')['datasets'][ds]
    case=root/'capture'/ds;case.mkdir(parents=True,exist_ok=True)
    cfg_values=read(ARCHIVE/'cases'/ds/'full/protocol.json')['configuration']
    cfg_values.update(save_dir=str(case/'index'),stage_ablation='full')
    cfg=BindPackConfig.from_file(**cfg_values)
    cfg.pack.review_mode='off';cfg.pack.selector_mode='current'
    docs=read(root/'metadata'/ds/'corpus.json')
    source=PROJECT/'outputs/4_datasets_result/shared_indexes'/ds
    prepare_archive(source,case/'index',case,{'dataset':ds},docs,cfg)
    cache=case/'index/llm_cache';cache.mkdir(exist_ok=True)
    for p in (ARCHIVE/'cases'/ds/'full/index/llm_cache').glob('*.sqlite'):
        target=cache/p.name
        if not target.exists():shutil.copy2(p,target)
    recorder=Recorder(case)
    tokenizer=AutoTokenizer.from_pretrained('/root/models/Qwen3-8B',local_files_only=True)
    from bindpack.llm.openai_gpt import CacheOpenAI
    recorder.install_llm_factory(CacheOpenAI,tokenizer)
    with ResourceMonitor(recorder) as monitor:
        rag=BindPack(config=cfg)
        rag.evidence_runtime.capture_source_alternatives=True
        recorder.install_embedding(rag.embedding_model)
        rag.prepare_retrieval_objects();rag._build_index_side_artifacts()
        def deny(*a,**kw):raise RuntimeError('Index build forbidden in Pack diagnostic')
        rag.index=deny
        native=rag.evidence_runtime.pack.finalize
        pending={}
        def freeze(query,ids,scores,ctx,state):
            pending[query]=freeze_pack_input(query,ids,scores,state)
            return native(query,ids,scores,ctx,state)
        rag.evidence_runtime.pack.finalize=freeze
        write(case/'configuration.json',asdict(cfg))
        left=[c for c in chosen if not (root/'frozen_inputs'/ds/token(c['sample_id'])).exists()]
        for start in range(0,len(left),8):
            batch=left[start:start+8];pending.clear()
            begin=time.perf_counter()
            rag.set_query_hop_overrides([c['benchmark_hops'] for c in batch],source='benchmark')
            with recorder.section('shared_upstream_capture',f'{ds}:{start}'):
                solutions=rag.retrieve([c['question'] for c in batch],num_to_retrieve=200)
            for c,sol in zip(batch,solutions):
                value=pending[c['question']]
                write(root/'frozen_inputs'/ds/token(c['sample_id']),{'input':value,'input_sha256':digest(value),
                    'baseline_doc_ids':sol.retrieval_trace['evidence']['improvement_support_semantic_veto']['final_doc_ids'],
                    'baseline_trace':sol.retrieval_trace,'source_alternatives_captured':True,
                    'query_index':c['query_index'],'sample_id':c['sample_id'],'batch_upstream_seconds':time.perf_counter()-begin})
            n=sum((root/'frozen_inputs'/ds/token(c['sample_id'])).exists() for c in chosen)
            write(root/'status.json',{'phase':'capture','dataset':ds,'completed':n,'total':200})
            print(f'CAPTURE {ds} {n}/200; batch={time.perf_counter()-begin:.1f}s',flush=True)
        write(case/'resource_summary.json',monitor.summary())
    verify_private(case,release=True)
    write(case/'completed.ok',{'queries':200,'new_index_build_calls':0})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--dataset',choices=DATASETS)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);prepare(a.output)
    if a.dataset:capture(a.output,a.dataset)

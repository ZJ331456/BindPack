#!/usr/bin/env python3
"""Recover native Pack inputs using frozen Full semantics and the same indexes.

No new LLM response or answer is permitted. Only missing numerical Route/dense
scores are recomputed. Every ablation is subsequently conditioned on Full parity.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time

PROJECT = Path('/root/BindPack')
SOURCE = PROJECT/'outputs/4_datasets_result/mechanism_complete_20261027'
DATASETS = ('hotpotqa','2wikimultihopqa','musique','hover')

def read(path): return json.loads(Path(path).read_text())
def digest(value): return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)
def token(qid): return hashlib.sha256(str(qid).encode()).hexdigest()+'.json.gz'

def memoized_dense(native):
    """Reuse identical within-window numerical results, returning private arrays."""
    cache={};counts={'hits':0,'misses':0}
    def cached(query):
        if query not in cache:
            ids,scores=native(query)
            cache[query]=(ids.copy(),scores.copy());counts['misses']+=1
        else:counts['hits']+=1
        return tuple(value.copy() for value in cache[query])
    return cached,cache,counts

def restore_saved_routes(runtime,state,saved_routes):
    """Recover routes omitted from outer dense-fallback traces as well as DAG routes."""
    present=state['evidence_trace']['routes']
    assert present==saved_routes[:len(present)],'Existing route prefix differs from saved Full'
    for route in saved_routes[len(present):]:
        ids=runtime._add_route(state,route['source'],route['goal'],route['query'],route['requirements'])
        assert ids==route['doc_ids'],'Saved route numerical parity failed: '+route['source']
    assert state['evidence_trace']['routes']==saved_routes

def capture(root,dataset,limit=None,wait_parity=False):
    os.environ.setdefault('OPENAI_API_KEY','EMPTY')
    # Use the current safe copy-on-write/reserve helper, while all retrieval
    # semantics below come from the immutable original Full snapshot.
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from utils.archived_index import prepare_archive,verify_private
    snapshot=SOURCE/'source_snapshot'
    sys.path[:0]=[str(snapshot/'src'),str(snapshot/'scripts')]
    from bindpack import BindPack,BindPackConfig
    from bindpack.evaluation.measurement import Recorder,ResourceMonitor
    from run_pack_weight_diagnostic import freeze_pack_input
    from run_pack_weight_sweep import sha
    import numpy as np

    source_case=SOURCE/'cases'/dataset/'wo_source_bound_eligibility'
    rows=[json.loads(line)['row'] for line in (source_case/'progress.jsonl').open() if line.strip()]
    assert len(rows)==1000 and len({r['sample_id'] for r in rows})==1000
    selected=set(read(root/'selection_locked.json')['datasets'][dataset]['pilot']+
                 read(root/'selection_locked.json')['datasets'][dataset]['validation'])
    preflight={r['sample_id'] for r in rows[:8]}
    wanted=selected|preflight
    by_query={}
    for row in rows:
        previous=by_query.setdefault(row['question'],row)
        if previous is not row:
            a,b=previous['retrieval_trace'],row['retrieval_trace']
            assert previous['benchmark_hops']==row['benchmark_hops']
            for key in ('static_sub_questions','pcqd_sub_questions'):
                assert a.get(key)==b.get(key),'Ambiguous duplicate semantic output: '+key
            for key in ('plan','bindings','routes','source_records','branch_scores'):
                assert a['evidence'].get(key)==b['evidence'].get(key),'Ambiguous duplicate Plan: '+key
            assert a['fact_filter'].get('selected_fact_indices')==b['fact_filter'].get('selected_fact_indices')
    active_rows=[]
    configuration=read(source_case/'protocol.json')['configuration']
    case=root/'capture'/dataset;case.mkdir(parents=True,exist_ok=True)
    cfg=BindPackConfig.from_file(**dict(configuration,save_dir=str(case/'index')))
    assert cfg.pack.review_mode=='off' and cfg.pack.selector_mode=='combined' and cfg.pack.relation_weight==.6
    docs=read(SOURCE/'runs'/'wo_source_bound_eligibility'/dataset/'metadata'/dataset/'corpus.json')
    recorder=Recorder(case)
    started=time.perf_counter()
    with ResourceMonitor(recorder) as monitor:
        receipt=prepare_archive(PROJECT/'outputs/4_datasets_result/shared_indexes'/dataset,
                                case/'index',case,{'dataset':dataset},docs,cfg)
        rag=BindPack(config=cfg)
        recorder.install_embedding(rag.embedding_model)
        def deny(*args,**kwargs):raise AssertionError('Forbidden: new LLM, QA or index construction')
        rag.llm_model.infer=deny
        rag.llm_model.openai_client.chat.completions.create=deny
        rag.qa=deny;rag.index=deny
        rag.prepare_retrieval_objects();rag._build_index_side_artifacts()
        rag.dense_passage_retrieval,dense_cache,dense_counts=memoized_dense(rag.dense_passage_retrieval)
        def frozen_rerank(query,scores):
            f=deepcopy(by_query[query]['retrieval_trace']['fact_filter'])
            return list(f.get('selected_fact_indices',[])),[tuple(x) for x in f.get('facts_after_rerank',[])],f
        rag.rerank_facts=frozen_rerank
        rag._decompose_query=lambda query,*args,**kwargs:deepcopy(by_query[query]['retrieval_trace'].get('static_sub_questions',[]))
        def pcqd(query,*args,**kwargs):
            t=by_query[query]['retrieval_trace']
            rag._pcqd_diagnostics_by_query=getattr(rag,'_pcqd_diagnostics_by_query',{})
            rag._pcqd_diagnostics_by_query[query]=deepcopy(t.get('pcqd_validation',{}))
            return deepcopy(t.get('pcqd_sub_questions',[]))
        rag._decompose_query_with_hints=pcqd
        runtime=rag.evidence_runtime
        def frozen_plan(states,executor=None):
            for state in states:
                original_row=active_rows[state['query_idx']]
                assert original_row['question']==state['query']
                old=original_row['retrieval_trace']['evidence']
                restore_saved_routes(runtime,state,old['routes'])
                for key in ('plan','bindings','branch_scores','source_records','planning_outputs',
                            'verification_outputs','semantic_failures','rejected_hypotheses',
                            'llm_plan_calls','llm_verification_calls'):
                    if key in old:state['evidence_trace'][key]=deepcopy(old[key])
                assert state['evidence_trace']['routes']==old['routes'],'Static/PCQD route parity failed'
                state['_evidence_plan']=deepcopy(old['plan'])
                state['_evidence_winning_bindings']=deepcopy(old['bindings'])
                state['_evidence_beams']=deepcopy(old['branch_scores'])
                winning=next((b for b in state['_evidence_beams'] if b['bindings']==old['bindings']),{'proofs':{}})
                for node,proof in winning['proofs'].items():
                    runtime._candidate(state,int(proof['doc_id']))['verified'].append(dict(proof,goal='dag:'+node))
        runtime.plan.process_window=frozen_plan
        inputs=root/'inputs'/dataset;inputs.mkdir(parents=True,exist_ok=True)
        def freeze(query,ids,scores,context,state):
            original=active_rows[state['query_idx']]
            assert original['question']==query
            if original['sample_id'] not in wanted:
                return ids,scores,state['evidence_trace']
            payload=freeze_pack_input(query,ids,scores,state)
            event={'sample_id':original['sample_id'],'query_index':original['query_index'],
                   'input_sha256':digest(payload),'input':payload,'source_progress_sha256':source_sha,
                   'recovery_version':4,'semantic_profile':'adopted_no_source_full','recovery_code_sha256':recovery_sha}
            path=inputs/token(original['sample_id'])
            if not path.exists():
                temp=path.with_suffix(path.suffix+'.tmp')
                with gzip.open(temp,'wt',encoding='utf-8',compresslevel=3) as stream:
                    json.dump(event,stream,ensure_ascii=False,separators=(',',':'),allow_nan=False)
                temp.replace(path)
            # Capture only: no Pack optimization, review, QA or benchmark scoring.
            return ids,scores,state['evidence_trace']
        runtime.pack.finalize=freeze
        runtime.pack.postprocess=lambda *args,**kwargs:{}
        write(case/'configuration.json',asdict(cfg))
        source_sha=sha(source_case/'progress.jsonl')
        recovery_sha=sha(Path(__file__))
        for start in range(0,min(1000,limit or 1000),8):
            if wait_parity and start>=8:
                gate=root/'preflight'/dataset/'passed.json'
                while not gate.exists():time.sleep(.2)
            batch=rows[start:start+8]
            dense_cache.clear()
            if all(r['sample_id'] not in wanted or (inputs/token(r['sample_id'])).exists() for r in batch):
                # Restore the original cross-window main-query embedding cache,
                # including duplicate questions, without repeating Route or Pack.
                rag.get_query_embeddings([r['question'] for r in batch])
                continue
            active_rows=batch
            begin=time.perf_counter()
            rag.set_query_hop_overrides([r['benchmark_hops'] for r in batch],source='benchmark')
            with recorder.section('shared_numeric_capture',f'{dataset}:{start}'):
                rag.retrieve([r['question'] for r in batch],num_to_retrieve=200)
            done=sum((inputs/token(sid)).exists() for sid in selected)
            write(root/'capture_status.json',{'dataset':dataset,'completed':done,'total':len(selected),'original_rows_visited':start+len(batch),'elapsed_seconds':time.perf_counter()-started})
            print(f'CAPTURE {dataset} {done}/{len(selected)}; window={time.perf_counter()-begin:.2f}s; new LLM=0; QA=0',flush=True)
        write(case/'resources.json',monitor.summary())
    verify_private(case,release=limit is None)
    write(case/'capture_audit.json',{'dataset':dataset,'inputs':sum(1 for _ in inputs.glob('*.json.gz')),
        'source_progress_sha256':source_sha,'new_llm_calls':0,'qa_calls':0,'index_build_calls':0,
        'shared_numeric_recomputation':True,'semantic_outputs':'exact saved Full fact filtering, QD, PCQD, Plan, bindings and source hypotheses',
        'duplicate_question_rows_preserved_by_sample_id':len(rows)-len(by_query),
        'recovery_code_sha256':recovery_sha,
        'within_window_dense_cache':dense_counts,
        'wall_seconds':time.perf_counter()-started,'archive':receipt['source']})
    if limit is None:
        assert all((inputs/token(sid)).exists() for sid in wanted)
        write(case/'completed.ok',{'inputs':len(wanted),'study_inputs':len(selected),'source_progress_sha256':source_sha})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--dataset',choices=DATASETS,required=True);p.add_argument('--limit',type=int);p.add_argument('--wait-parity',action='store_true')
    a=p.parse_args();capture(a.output.resolve(),a.dataset,a.limit,a.wait_parity)

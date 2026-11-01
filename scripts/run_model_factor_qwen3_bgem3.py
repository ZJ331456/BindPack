#!/usr/bin/env python3
"""Table 6: native MuSiQue Full with Qwen3-8B and fresh BGE-M3 vectors."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
from zoneinfo import ZoneInfo

PROJECT=Path('/root/BindPack')
ROOT=Path(__file__).resolve().parents[1]
DATASET='musique'
ARCHIVE=PROJECT/'outputs/4_datasets_result'
SOURCE=ARCHIVE/'shared_indexes/musique'
EMBEDDING='/root/models/bge-m3'
PYTHON='/root/anaconda3/envs/rag/bin/python'


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4194304),b''):result.update(block)
    return result.hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)
def stamp():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def freeze(root):
    protocol=root/'experiment.json'
    if not protocol.exists():
        snapshot=root/'source_snapshot'
        for folder in ('src','scripts','configs'):
            shutil.copytree(PROJECT/folder,snapshot/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        inputs={}
        for suffix in ('','_corpus'):
            source=PROJECT/'datasets'/(DATASET+suffix+'.json');target=root/'frozen_datasets'/source.name
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);inputs[source.name]=sha(target)
        files={str(p.relative_to(snapshot)):sha(p) for p in snapshot.rglob('*') if p.is_file()}
        write(protocol,dict(created_at=stamp(),dataset=DATASET,queries=1000,eval_mode='rag_qa',reader_k=5,
            llm='/root/models/Qwen3-8B',embedding=EMBEDDING,source_bound_eligibility=False,
            support_semantic_veto=False,selector_mode='combined',relevance_weight=.6,
            vllm_gpu_memory_utilization=.82,vllm_target_gib=19.68,vllm_max_num_seqs=32,
            embedding_policy='1024-dimensional normalized CLS dense vectors, raw query text, max length 2048',
            index_policy='reuse identical published Qwen3 OpenIE facts; rebuild all passage/entity/fact vectors and synonym edges using BGE-M3; no Qwen vectors or query answers reused',
            parallelism_policy='calibrate 8/16/32 retrieval and Reader workers on the same 32 queries; choose measured throughput; separate fresh 1000-query Full',
            native_hop_policy='same benchmark-native hop hints as current Full',
            datasets_sha256=inputs,source_sha256=files))
        (root/'README.md').write_text('# Qwen3-8B + BGE-M3 / MuSiQue Full\n\n'
            'Table 6 experiment: all 1000 questions, native Route/Plan/Pack and Reader@5. '
            'Source eligibility and final review are off; joint Pack and relevance weight0.60 are retained. '
            'Qwen3-8B uses BF16 at approximately20GiB. BGE-M3 uses normalized CLS dense embeddings without query instructions.\n\n'
            'The original immutable OpenIE records are reused. All vectors and embedding-dependent synonym edges are rebuilt. '
            'Calibration is separate from the final Full and its costs are retained. '
            'Within each window both native retrieval and native Reader calls are parallel; every response and cost is checkpointed.\n\n'
            'Progress: run.log/status.json. Final results: full/cases/musique/full/report.json and RESULTS.md. '
            'BGE index: shared_indexes/musique. Index cost: index_build/report.json. '
            'Model server: service/vllm.log. A successful final result fills only the matching Table6 row.\n')
    protocol=read(protocol)
    for relative,value in protocol['source_sha256'].items():
        if sha(root/'source_snapshot'/relative)!=value:raise ValueError('Frozen experiment code changed: '+relative)
    for name,value in protocol['datasets_sha256'].items():
        if sha(root/'frozen_datasets'/name)!=value:raise ValueError('Frozen dataset changed: '+name)


def settings(root,workers):
    path=root/'settings'/f'workers{workers}.json'
    value=read(ROOT/'configs/ablations_qwen3.json')
    value.update(method_config=str(ROOT/'configs/full_no_source_eligibility.json'),datasets=[DATASET],
                 datasets_dir=str(root/'frozen_datasets'))
    value['runtime'].update(embedding_model_name=EMBEDDING,embedding_batch_size=32,
                            llm_prefetch_workers=workers,openie_max_workers=8)
    value['evaluation'].update(eval_mode='rag_qa',sample_size=0)
    if path.exists():assert read(path)==value
    else:write(path,value)
    return path


def build_index(root):
    sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from bindpack import BindPack, BindPackConfig
    from bindpack.evaluation.measurement import Recorder,ResourceMonitor,jsonl,usage,digest
    from bindpack.utils.misc_utils import reformat_openie_results,extract_entity_nodes,flatten_facts
    from utils.archived_index import validate_archive
    from eval_utils import build_docs_from_full_corpus,ensure_gold_docs_in_corpus,get_gold_docs
    torch.set_num_threads(8)
    data=read(root/'frozen_datasets/musique.json');corpus=read(root/'frozen_datasets/musique_corpus.json')
    gold=[sorted(set(group)) for group in get_gold_docs(data,DATASET)]
    docs=ensure_gold_docs_in_corpus(build_docs_from_full_corpus(corpus),gold)
    source_receipt=validate_archive(SOURCE,DATASET,docs,'/root/models/Qwen3-Embedding-8B','qwen3-8b')
    write(root/'index_build/source_index.json',source_receipt)
    base=read(settings(root,32));runtime=base['runtime']
    archive=root/'shared_indexes/musique';model=archive/'qwen3-8b__root_models_bge-m3'
    model.mkdir(parents=True,exist_ok=True)
    source_model=SOURCE/'qwen3-8b__root_models_Qwen3-Embedding-8B'
    source_manifest=read(source_model/'index_manifest.json')
    manifest=dict(schema_version=1,rag_type='bindpack',text_normalization=source_manifest['text_normalization'],
        embedding=dict(provider='sentence_transformers_cls_dense',model_name=EMBEDDING,normalized=True,
            max_sequence_length=2048,dtype='auto',dimension=1024,query_instruction='',
            local_model_config_sha256=sha(Path(EMBEDDING)/'config.json')),
        openie=source_manifest['openie'],graph_construction=source_manifest['graph_construction'],
        is_directed_graph=source_manifest['is_directed_graph'],
        producer=dict(class_name='bindpack.embedding_model.BGE.BGEM3EmbeddingModel',
            source_index=str(SOURCE),source_inventory_sha256=source_receipt['archive_inventory_sha256'],
            extraction_policy='immutable published extraction reused without relabeling it as a fresh extraction'))
    write(model/'index_manifest.json',manifest)
    openie=SOURCE/'openie_results_ner_qwen3-8b.json';target=archive/openie.name
    if target.exists():assert sha(target)==sha(openie)
    else:shutil.copy2(openie,target)
    config=BindPackConfig.from_file(base['method_config'],save_dir=str(archive),dataset=DATASET,corpus_len=len(docs),
        synonymy_edge_query_batch_size=512,synonymy_edge_key_batch_size=8192,
        **{k:runtime[k] for k in ('llm_name','llm_base_url','embedding_model_name','embedding_batch_size',
            'openie_max_workers','llm_prefetch_workers','max_new_tokens','openie_strict','openie_prompt_version','openie_validation_mode')})
    recorder=Recorder(root/'index_build');monitor=ResourceMonitor(recorder)
    from transformers import AutoTokenizer
    from bindpack.llm.openai_gpt import CacheOpenAI
    tokenizer=AutoTokenizer.from_pretrained('/root/models/Qwen3-8B',local_files_only=True)
    recorder.install_llm_factory(CacheOpenAI,tokenizer)
    began=time.perf_counter()
    with monitor,recorder.section('offline_index'):
        rag=BindPack(config=config);recorder.install_embedding(rag.embedding_model);recorder.install_index(rag)
        def forbid_llm(*args,**kwargs):raise RuntimeError('Embedding replacement must not re-extract or query an LLM')
        rag.llm_model.infer=forbid_llm
        chunks=pq.read_table(source_model/'chunk_embeddings/vdb_chunk.parquet',columns=['content'])['content'].to_pylist()
        entities=pq.read_table(source_model/'entity_embeddings/vdb_entity.parquet',columns=['content'])['content'].to_pylist()
        facts=pq.read_table(source_model/'fact_embeddings/vdb_fact.parquet',columns=['content'])['content'].to_pylist()
        original_rows=read(target)['docs'];_,triples=reformat_openie_results(original_rows)
        from bindpack.utils.misc_utils import compute_mdhash_id
        chunk_ids=[compute_mdhash_id(text,'chunk-') for text in chunks]
        chunk_triples=[rag._normalize_graph_triples(triples[key].triples) for key in chunk_ids]
        entity_nodes,chunk_entities=extract_entity_nodes(chunk_triples)
        if set(entities)!=set(entity_nodes) or set(facts)!={str(fact) for fact in flatten_facts(chunk_triples)}:
            raise ValueError('Cached published facts/entities differ from the verified source index')
        print(stamp(),'ENCODE BGE-M3 passages:',len(chunks),'entities:',len(entities),'facts:',len(facts),flush=True)
        for label,texts in [('chunk',chunks),('entity',entities),('fact',facts)]:
            getattr(rag,label+'_embedding_store').insert_strings(texts)
            store=getattr(rag,label+'_embedding_store')
            matrix=store.get_embeddings(store.get_all_ids())
            if matrix.shape!=(len(texts),1024) or not np.isfinite(matrix).all():raise ValueError('Invalid BGE vector shape/data')
            if not np.allclose(np.linalg.norm(matrix,axis=1),1,atol=.002):raise ValueError('BGE vectors are not normalized')
            del matrix
            print(stamp(),'BGE vector store ready:',label,len(texts),flush=True)
        # Reconstruct native factual/passage edges, then compute fresh BGE synonym edges.
        import igraph as ig
        rag.graph=ig.Graph(directed=config.is_directed_graph);rag.ready_to_retrieve=False
        rag.node_to_node_stats={};rag.ent_node_to_chunk_ids={}
        rag.add_fact_edges(chunk_ids,chunk_triples);rag.add_passage_edges(chunk_ids,chunk_entities)
        rag.add_synonymy_edges();rag.augment_graph()
        rag.graph['openie_chunk_digests']=rag._openie_chunk_digests(chunk_ids,chunk_triples)
        rag.save_igraph();rag.prepare_retrieval_objects();rag._build_index_side_artifacts()
        seconds=time.perf_counter()-began
    resources=monitor.summary()
    assert not jsonl(root/'index_build/llm_calls.jsonl')
    for relative,entry in source_receipt['assets'].items():assert sha(SOURCE/relative)==entry['sha256']
    report=dict(seconds=seconds,corpus_sha256=digest(docs),corpus_docs=len(docs),
        passages=len(chunks),entities=len(entities),facts=len(facts),dimension=1024,
        graph_nodes=rag.graph.vcount(),graph_edges=rag.graph.ecount(),configuration=asdict(config),
        offline_index_usage=usage([]),embedding_calls=jsonl(root/'index_build/embedding_calls.jsonl'),
        resources=resources,reused_openie_sha256=sha(target),source_index_unchanged=True,
        qwen_vectors_reused=False,synonym_edges_rebuilt=True,
        scope='successful finalization interval: BGE model loading, remaining encoding, native graph/synonym rebuilding and retrieval preparation; earlier incomplete encoding attempts remain in embedding_calls/resource_samples and setup revision records; historical OpenIE construction excluded')
    write(root/'index_build/report.json',report)
    manifest['build_measurement']=dict(path='index_build/report.json',sha256=sha(root/'index_build/report.json'))
    write(model/'index_manifest.json',manifest)
    files=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p))
           for p in archive.rglob('*') if p.is_file() and 'llm_cache' not in p.parts]
    p=root/'index_build/report.json';files.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p)))
    write(root/'manifest.json',dict(files=files,embedding=EMBEDDING,dataset=DATASET))
    validate_archive(archive,DATASET,docs,EMBEDDING,'qwen3-8b',dict(normalized=True,max_sequence_length=2048,dtype='auto'))
    write(root/'index_build/completed.ok',dict(report_sha256=sha(root/'index_build/report.json'),manifest_sha256=sha(root/'manifest.json')))
    print(stamp(),'BGE INDEX COMPLETE',seconds,'seconds',flush=True)


def complete(output,n):
    case=output/'cases/musique/full';marker=case/'completed.ok'
    if not marker.exists():return False
    done=read(marker)
    assert done['query_count']==n and done['report_sha256']==sha(case/'report.json')
    assert done['result_sha256']==sha(case/'result.json') and done['llm_calls_sha256']==sha(case/'llm_calls.jsonl')
    report=read(case/'report.json');assert report['coverage']['sample_count']==report['coverage']['qa_n']==n
    assert read(case/'audit.json')['status']=='passed'
    return True


def publish(root):
    case=root/'full/cases/musique/full';assert complete(root/'full',1000)
    report=read(case/'report.json');metrics=report['metrics']
    paper=PROJECT/'docs/DASFAA_2027_latex'
    # Update exactly one row; leave other authors' table entries and figures intact.
    values=[f'{metrics[key]*100:.2f}' for key in ('Recall@5','FCR@5','DCC@5','F1')]
    line='qwen3-8B & BGE-M3 & BindPack & '+' & '.join(values)+r' \\'
    for relative in ('results/main/model-configurations.tex','results/appendix/models.tex'):
        path=paper/relative;text=path.read_text();lines=text.splitlines()
        matches=[i for i,row in enumerate(lines) if row.strip().startswith('qwen3-8B & BGE-M3 & BindPack &')]
        if len(matches)!=1:raise ValueError('Model table row missing or ambiguous: '+relative)
        lines[matches[0]]=line;path.write_text('\n'.join(lines)+'\n')
    write(paper/'results/model_factor_qwen3_bge_m3.json',dict(queries=1000,dataset=DATASET,report=str(case/'report.json'),
        report_sha256=sha(case/'report.json'),configuration=read(case/'protocol.json')['configuration'],metrics=metrics))
    for entry,target in [('main.tex',paper/'main.pdf'),('main-local.tex',paper/'local_only/main-local.pdf')]:
        build=root/'paper_build'/Path(entry).stem;source=build/'source';output=build/'output'
        shutil.copytree(paper,source,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','main.pdf','main-local.pdf'))
        output.mkdir(parents=True,exist_ok=True)
        command=['/root/.cache/tectonic-bindpack/tectonic','-C','-b','http://127.0.0.1:18876/tlextras-2022.0r0.tar',
                 '--keep-logs','--outdir',str(output),entry]
        with (build/'compile.log').open('w') as log:
            subprocess.run(command,cwd=source,env=dict(os.environ,XDG_CACHE_HOME='/root/.cache/bindpack_english_tex_cache'),stdout=log,stderr=subprocess.STDOUT,check=True)
        temp=target.with_name(target.stem+'.model-factor.tmp.pdf');shutil.copy2(output/(Path(entry).stem+'.pdf'),temp);temp.replace(target)
    print(stamp(),'TABLE 6 UPDATED',dict(zip(('Recall@5','FCR@5','DCC@5','QA F1'),values)),flush=True)


def main(args):
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    if not args.frozen:
        freeze(root)
        os.execv(sys.executable,[sys.executable,'-B','-u',str(root/'source_snapshot/scripts'/Path(__file__).name),
            '--output',str(root),'--frozen',*(['--build-index'] if args.build_index else [])])
    if args.build_index:
        build_index(root);return
    import fcntl
    lock=(root/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (root/'run.pid').write_text(str(os.getpid())+'\n')
    state=dict(status='running',phase='indexing',dataset=DATASET,pid=os.getpid(),queries=1000)
    current=[None];stop=threading.Event();ended=threading.Event()
    def interrupted(*unused):
        stop.set()
        if current[0] is not None and current[0].poll() is None:current[0].send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    def heartbeat():
        while not ended.is_set():
            status=dict(state,updated_at=stamp());where=state.get('worker_output')
            if where and (Path(where)/'status.json').exists():status['active']=read(Path(where)/'status.json')
            write(root/'status.json',status);active=status.get('active',{})
            n=active.get('completed_queries',0);total=active.get('expected_queries',state.get('expected_queries',1000))
            bar='#'*int(24*n/max(total,1))+'-'*(24-int(24*n/max(total,1)))
            print(stamp(),'PROGRESS',state['phase'],f'[{bar}] {n}/{total}',flush=True);ended.wait(30)
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    def execute(command,output):
        if stop.is_set():raise KeyboardInterrupt('Stopped at saved checkpoint')
        state['worker_output']=str(output)
        process=subprocess.Popen(command,cwd=PROJECT,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            text=True,bufsize=1);current[0]=process
        for line in process.stdout:print(line,end='',flush=True)
        code=process.wait();current[0]=None
        if stop.is_set():raise KeyboardInterrupt('Stopped at saved checkpoint')
        if code:raise RuntimeError('Worker failed with code '+str(code))
    def evaluate(output,workers,n):
        if complete(output,n):return read(output/'cases/musique/full/report.json')
        command=[sys.executable,'-B','-u',str(ROOT/'scripts/run_bindpack_measured.py'),'--frozen-worker',
            '--dataset',DATASET,'--output',str(output),'--config',str(settings(root,workers)),
            '--datasets-dir',str(root/'frozen_datasets'),'--reuse-archive-index',str(root/'shared_indexes/musique'),
            '--query-window',str(workers),'--reader-workers',str(workers),'--model-factor-profile','qwen3_bge_m3',
            '--eval-mode','rag_qa','--source-bound-eligibility','off','--candidate-output-top-k','10']
        if n!=1000:command+=['--query-limit',str(n)]
        if (output/'experiment_config.json').exists():command+=['--resume']
        write(output/'launch.json',dict(command=command,workers=workers,n=n,started_at=stamp()))
        execute(command,output)
        if not complete(output,n):raise RuntimeError('Completed worker failed its result audit')
        return read(output/'cases/musique/full/report.json')
    try:
        if not (root/'index_build/completed.ok').exists():
            execute([sys.executable,'-B','-u',str(Path(__file__).resolve()),'--output',str(root),'--frozen','--build-index'],root/'index_build')
        state.update(phase='calibration',expected_queries=32);calibration=[]
        for workers in (8,16,32):
            report=evaluate(root/'calibration'/f'window{workers}',workers,32)
            calibration.append(dict(workers=workers,seconds_per_query=report['efficiency']['processing_seconds_per_query']))
        best=min(calibration,key=lambda row:row['seconds_per_query'])['workers']
        write(root/'parallelism.json',dict(calibration=calibration,chosen_workers=best,selection_uses='measured throughput only, not quality or gold outcomes'))
        state.update(phase='full',expected_queries=1000,chosen_workers=best)
        report=evaluate(root/'full',best,1000)
        write(root/'summary.json',report)
        keys=('Recall@5','FCR@5','DCC@5','ExactMatch','F1')
        (root/'RESULTS.md').write_text('# Qwen3-8B + BGE-M3 / MuSiQue Full\n\n|N|'+'|'.join(keys)+'|\n|---:|'+'---:|'*len(keys)+'\n|1000|'+'|'.join(f"{report['metrics'][k]*100:.2f}" for k in keys)+'|\n')
        state.update(status='complete',phase='complete',completed_queries=1000)
        write(root/'completed.ok',dict(queries=1000,report=str(root/'full/cases/musique/full/report.json'),
            report_sha256=sha(root/'full/cases/musique/full/report.json')))
        try:publish(root)
        except Exception as error:
            state['publication_error']=str(error);print(stamp(),'Experiment complete; paper publication error:',error,flush=True)
    except BaseException as error:
        state.update(status='paused' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error));raise
    finally:
        ended.set();thread.join(timeout=3);write(root/'status.json',dict(state,updated_at=stamp()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=PROJECT/'outputs/model_factor_qwen3_bgem3_musique_20261030')
    parser.add_argument('--frozen',action='store_true');parser.add_argument('--build-index',action='store_true')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(name)s %(levelname)s %(message)s')
    main(parser.parse_args())

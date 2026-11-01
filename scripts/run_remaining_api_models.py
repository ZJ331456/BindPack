#!/usr/bin/env python3
"""Table 6: two fixed Dense rankings/readers, then API extraction + BGE BindPack."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from dataclasses import asdict
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
from types import SimpleNamespace
from zoneinfo import ZoneInfo

PROJECT=Path('/root/BindPack');ROOT=Path(__file__).resolve().parents[1]
API_SOURCE=PROJECT/'outputs/api_qwen38_flash_fresh_index_musique_full_20261030'
INDEX=API_SOURCE/'run/cases/musique/full/index'
OLD_MODEL=INDEX/'qwen3.8-flash__root_models_Qwen3-Embedding-8B'
CREDENTIALS=Path('/root/.config/bindpack/dashscope.json')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for part in iter(lambda:stream.read(4194304),b''):h.update(part)
    return h.hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def lines(p):return [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()] if Path(p).exists() else []
def stamp():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')

def freeze(root):
    snapshot=root/'api_source_snapshot'
    if not (root/'api_protocol.json').exists():
        for name in ('src','scripts','configs'):
            shutil.copytree(PROJECT/name,snapshot/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        marker=read(API_SOURCE/'run/cases/musique/full/completed.ok')
        assert marker['query_count']==1000 and sha(API_SOURCE/'run/cases/musique/full/report.json')==marker['report_sha256']
        write(root/'api_protocol.json',dict(created_at=stamp(),questions_per_case=1000,
            ordered_jobs=['dense_qwen','dense_bge','bindpack_bge'],llm='qwen3.8-flash',thinking=False,
            qa_max_tokens=2028,reader_top_k=5,workers=32,
            extraction_reuse='same API model, exact completed OpenIE; do not re-extract with a different model',
            cache_policy='same API original SQLite response cache; v2 keys bind all messages, generation parameters, endpoint and model',
            source_api_report=str(API_SOURCE/'run/cases/musique/full/report.json'),source_api_report_sha256=marker['report_sha256'],
            source_sha256={str(p.relative_to(snapshot)):sha(p) for p in snapshot.rglob('*') if p.is_file()}))
    for name,value in read(root/'api_protocol.json')['source_sha256'].items():
        if sha(snapshot/name)!=value:raise ValueError('Frozen model-factor source changed: '+name)
    sys.path.insert(0,str(snapshot/'scripts'))
    from utils.llm_cache_reuse import seed_response_cache
    receipt=seed_response_cache(INDEX/'llm_cache/qwen3.8-flash_cache.sqlite',root/'response_cache/qwen3.8-flash_cache.sqlite')
    if not (root/'response_cache_seed.json').exists():write(root/'response_cache_seed.json',receipt)

def dense(root,job):
    from bindpack.configuration.runtime import RuntimeConfig
    from bindpack.core.base import BaseRAG
    from bindpack.llm.openai_gpt import CacheOpenAI
    from bindpack.prompts import PromptTemplateManager
    from bindpack.utils.misc_utils import QuerySolution
    from bindpack.evaluation.measured_reader import prepare
    from bindpack.evaluation.measurement import Recorder,ResourceMonitor,digest
    from bindpack.evaluation.metric_report import _qa
    from utils.api_cost import summarize_calls
    from transformers import AutoTokenizer
    case=root/'model_cases'/job;case.mkdir(parents=True,exist_ok=True)
    if (case/'completed.ok').exists():
        assert sha(case/'report.json')==read(case/'completed.ok')['report_sha256'];return
    encoder='qwen3_emb' if job=='dense_qwen' else 'bge_m3'
    embedding='/root/models/Qwen3-Embedding-8B' if job=='dense_qwen' else '/root/models/bge-m3'
    source=Path('/root/baseline/scripts/outputs/standard_full_20261019/cases/musique')/encoder
    source_report=read(source/'report.json');marker=read(source/'validated.ok')
    assert sha(source/'result.json')==marker['result_sha256']==source_report['inputs']['result']['sha256']
    assert source_report['coverage']['sample_count']==1000
    old_result=read(source/'result.json');original=old_result['results']
    assert len(original)==1000 and old_result['runtime_config']['embedding_model_name']==embedding
    assert sha(PROJECT/'datasets/musique.json')==source_report['inputs']['benchmark']['sha256']
    credentials=read(CREDENTIALS)
    runtime=RuntimeConfig(llm_name=credentials['model'],llm_base_url=credentials['base_url'],
        embedding_model_name=embedding,dataset='musique',qa_top_k=5,max_new_tokens=2048,temperature=0,seed=None)
    protocol=dict(method='Dense',dataset='musique',queries=1000,configuration=asdict(runtime),
        retrieval_reused=True,retrieval_source=str(source/'report.json'),source_report_sha256=sha(source/'report.json'),
        source_result_sha256=marker['result_sha256'],qa_max_tokens=2028,reader_top_k=5,workers=32,
        thinking_enabled=False,response_cache=str(root/'response_cache/qwen3.8-flash_cache.sqlite'),
        cache_eligibility='exact native messages/generation/model/endpoint key; inherited costs excluded',
        no_embedding_or_retrieval_rerun=True)
    if (case/'protocol.json').exists():assert read(case/'protocol.json')==protocol
    else:write(case/'protocol.json',protocol)
    tokenizer=AutoTokenizer.from_pretrained('/root/models/Qwen3-8B',local_files_only=True)
    recorder=Recorder(case);llm=CacheOpenAI(cache_dir=str(root/'response_cache'),global_config=runtime)
    recorder.install_llm(llm,tokenizer)
    rag=SimpleNamespace(global_config=runtime,llm_model=llm,prompt_template_manager=PromptTemplateManager())
    prepared={}
    for row in original:
        solution,budget=prepare(rag,QuerySolution(question=row['question'],docs=row['docs']),tokenizer,'musique')
        assert budget['prompt_sha256']==row['reader']['budget']['prompt_sha256'],row['sample_id']
        prepared[row['sample_id']]=(solution,budget)
    completed={e['row']['sample_id']:e['row'] for e in lines(case/'progress.jsonl')}
    start=time.perf_counter();monitor=ResourceMonitor(recorder)
    def answer(row):
        sid=row['sample_id'];solution,budget=prepared[sid]
        with recorder.thread_section('online_reader',sid):
            tick=time.perf_counter();results,raw,metadata=BaseRAG.qa(rag,[solution],max_output_tokens=2028)
        return dict(query_index=row['query_index'],sample_id=sid,question=row['question'],
            answer=results[0].answer,gold_answers=row['gold_answers'],docs=solution.docs,
            reader=dict(raw_response=raw[0],metadata=metadata[0],budget=budget),reader_seconds=time.perf_counter()-tick)
    with monitor,recorder.section('online_reader'):
        with ThreadPoolExecutor(max_workers=32) as pool:
            tasks={pool.submit(answer,row):row['sample_id'] for row in original if row['sample_id'] not in completed}
            for future in as_completed(tasks):
                row=future.result();completed[row['sample_id']]=row
                recorder.append('progress.jsonl',dict(row=row,sha256=digest(row)))
                if len(completed)%50==0:print(stamp(),job,len(completed),'/1000',flush=True)
    rows=[completed[row['sample_id']] for row in original];data=read(PROJECT/'datasets/musique.json')
    qa,coverage,details=_qa(rows,[data[row['query_index']] for row in rows],dict(dataset='musique',eval_mode='rag_qa'))
    assert coverage['qa_n']==1000
    metrics={**source_report['metrics'],**qa};calls=lines(case/'llm_calls.jsonl');logical=lines(case/'llm_logical_calls.jsonl')
    # A completely cached case still has an auditable empty physical ledger.
    (case/'llm_calls.jsonl').touch(exist_ok=True)
    cost=summarize_calls(calls)
    report=dict(dataset='musique',method='Dense',metrics=metrics,
        coverage={**source_report['coverage'],**coverage},cost=cost,
        cache_hits=sum(e.get('cache_hit') is True for e in logical),
        efficiency=dict(wall_seconds=time.perf_counter()-start,reader_workers=32),resources=monitor.summary())
    write(case/'result.json',dict(dataset='musique',method='Dense',results=rows,qa_details=details,metrics=metrics))
    write(case/'report.json',report)
    assert sha(source/'result.json')==marker['result_sha256']
    write(case/'audit.json',dict(status='passed',queries=1000,qa_n=1000,retrieval_unchanged=True,
        prompts_match_dense_control=True,model='qwen3.8-flash',cache_hits=report['cache_hits'],physical_requests=len(calls)))
    write(case/'completed.ok',dict(query_count=1000,report_sha256=sha(case/'report.json'),
        result_sha256=sha(case/'result.json'),llm_calls_sha256=sha(case/'llm_calls.jsonl')))
    print(stamp(),'PASS',job,json.dumps(metrics),flush=True)

def build_bge(root):
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    import igraph as ig
    from bindpack import BindPack,BindPackConfig
    from bindpack.evaluation.measurement import Recorder,ResourceMonitor,usage,jsonl,digest
    from bindpack.utils.misc_utils import reformat_openie_results,extract_entity_nodes,flatten_facts,compute_mdhash_id
    from bindpack.llm.openai_gpt import CacheOpenAI
    from transformers import AutoTokenizer
    from utils.archived_index import validate_archive
    if (root/'index_build/completed.ok').exists():
        assert sha(root/'index_build/report.json')==read(root/'index_build/completed.ok')['report_sha256'];return
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    identity=read(API_SOURCE/'index_identity.json')
    assert identity['extraction_model']=='qwen3.8-flash' and identity['passages']==11656
    for entry in identity['inventory']:assert sha(INDEX/entry['path'])==entry['sha256']
    archive=root/'shared_indexes/musique';model=archive/'qwen3.8-flash__root_models_bge-m3'
    model.mkdir(parents=True,exist_ok=True)
    source_openie=INDEX/'openie_results_ner_qwen3.8-flash.json';target=archive/source_openie.name
    if target.exists():assert sha(target)==sha(source_openie)
    else:shutil.copy2(source_openie,target)
    base=read(ROOT/'configs/ablations_qwen3.json');credentials=read(CREDENTIALS)
    base['runtime'].update(llm_name='qwen3.8-flash',llm_base_url=credentials['base_url'],
        embedding_model_name='/root/models/bge-m3',embedding_batch_size=32,llm_prefetch_workers=32,openie_max_workers=8)
    base.update(method_config=str(ROOT/'configs/full_no_source_eligibility.json'),datasets=['musique'])
    base['evaluation'].update(eval_mode='rag_qa',sample_size=1000)
    write(root/'api_bge_config.json',base)
    runtime=base['runtime'];config=BindPackConfig.from_file(base['method_config'],save_dir=str(archive),dataset='musique',corpus_len=11656,
        synonymy_edge_query_batch_size=512,synonymy_edge_key_batch_size=8192,
        **{key:runtime[key] for key in ('llm_name','llm_base_url','embedding_model_name','embedding_batch_size',
            'openie_max_workers','llm_prefetch_workers','max_new_tokens','openie_strict','openie_prompt_version','openie_validation_mode')})
    manifest=dict(schema_version=1,rag_type='bindpack',text_normalization='ascii_alnum_lower_v1',
        embedding=dict(model_name='/root/models/bge-m3',normalized=True,max_sequence_length=2048,dtype='auto',dimension=1024,query_instruction=''),
        openie=dict(identity=dict(model_name='qwen3.8-flash'),source=str(source_openie),sha256=sha(source_openie)),
        is_directed_graph=False,producer=dict(source_api_report_sha256=read(root/'api_protocol.json')['source_api_report_sha256'],
            extraction_reused=True,old_vectors_reused=False,synonym_edges_rebuilt=True))
    write(model/'index_manifest.json',manifest)
    recorder=Recorder(root/'index_build');monitor=ResourceMonitor(recorder)
    recorder.install_llm_factory(CacheOpenAI,AutoTokenizer.from_pretrained('/root/models/Qwen3-8B',local_files_only=True))
    began=time.perf_counter()
    with monitor,recorder.section('offline_index'):
        rag=BindPack(config=config);recorder.install_embedding(rag.embedding_model);recorder.install_index(rag)
        def forbid(*a,**kw):raise RuntimeError('API OpenIE is already complete; no new model extraction permitted')
        rag.llm_model.infer=forbid
        chunks=pq.read_table(OLD_MODEL/'chunk_embeddings/vdb_chunk.parquet',columns=['content'])['content'].to_pylist()
        entities=pq.read_table(OLD_MODEL/'entity_embeddings/vdb_entity.parquet',columns=['content'])['content'].to_pylist()
        facts=pq.read_table(OLD_MODEL/'fact_embeddings/vdb_fact.parquet',columns=['content'])['content'].to_pylist()
        _,triples=reformat_openie_results(read(target)['docs'])
        chunk_ids=[compute_mdhash_id(text,'chunk-') for text in chunks]
        chunk_triples=[rag._normalize_graph_triples(triples[key].triples) for key in chunk_ids]
        nodes,chunk_entities=extract_entity_nodes(chunk_triples)
        assert set(nodes)==set(entities) and set(facts)=={str(fact) for fact in flatten_facts(chunk_triples)}
        for label,texts in [('chunk',chunks),('entity',entities),('fact',facts)]:
            print(stamp(),'BGE ENCODE',label,len(texts),flush=True)
            store=getattr(rag,label+'_embedding_store');store.insert_strings(texts)
            matrix=store.get_embeddings(store.get_all_ids());assert matrix.shape==(len(texts),1024)
            assert np.isfinite(matrix).all() and np.allclose(np.linalg.norm(matrix,axis=1),1,atol=.002)
            del matrix
        rag.graph=ig.Graph(directed=False);rag.ready_to_retrieve=False;rag.node_to_node_stats={};rag.ent_node_to_chunk_ids={}
        rag.add_fact_edges(chunk_ids,chunk_triples);rag.add_passage_edges(chunk_ids,chunk_entities)
        rag.add_synonymy_edges();rag.augment_graph();rag.graph['openie_chunk_digests']=rag._openie_chunk_digests(chunk_ids,chunk_triples)
        rag.save_igraph();rag.prepare_retrieval_objects();rag._build_index_side_artifacts()
    assert not jsonl(root/'index_build/llm_calls.jsonl')
    # Match the native full-corpus order, not embedding-store serialization order.
    from eval_utils import build_docs_from_full_corpus,ensure_gold_docs_in_corpus,get_gold_docs
    data=read(PROJECT/'datasets/musique.json');docs=ensure_gold_docs_in_corpus(
        build_docs_from_full_corpus(read(PROJECT/'datasets/musique_corpus.json')),
        [sorted(set(group)) for group in get_gold_docs(data,'musique')])
    assert set(chunks)==set(docs) and len(chunks)==11656
    report=dict(seconds=time.perf_counter()-began,corpus_sha256=digest(docs),passages=11656,
        entities=len(entities),facts=len(facts),dimension=1024,offline_index_usage=usage([]),
        reused_api_extraction_sha256=sha(source_openie),source_model='qwen3.8-flash',resources=monitor.summary(),
        graph_nodes=rag.graph.vcount(),graph_edges=rag.graph.ecount(),configuration=asdict(config))
    write(root/'index_build/report.json',report)
    manifest['build_measurement']=dict(path='index_build/report.json',sha256=sha(root/'index_build/report.json'))
    write(model/'index_manifest.json',manifest)
    assets=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p)) for p in archive.rglob('*')
            if p.is_file() and 'llm_cache' not in p.parts]
    p=root/'index_build/report.json';assets.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p)))
    write(root/'manifest.json',dict(files=assets,dataset='musique',embedding='/root/models/bge-m3'))
    validate_archive(archive,'musique',docs,'/root/models/bge-m3','qwen3.8-flash',dict(normalized=True,max_sequence_length=2048,dtype='auto'))
    write(root/'index_build/completed.ok',dict(report_sha256=sha(root/'index_build/report.json')))
    print(stamp(),'PASS API-EXTRACTION BGE INDEX',report['seconds'],'seconds',flush=True)

def run_command(command,env,log):
    Path(log).parent.mkdir(parents=True,exist_ok=True)
    with Path(log).open('a') as stream:
        process=subprocess.Popen(command,cwd=PROJECT,env=env,stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:
            stream.write(line);stream.flush();print(line,end='',flush=True)
        code=process.wait()
    if code:raise RuntimeError(f'Job failed ({code}); checkpoints retained: {log}')

def summarize_queue(root):
    table=['# Full Reader 补充与 API 模型对照','',
        '全量均为每组1000题；模型组依次运行 Dense/Qwen3-Emb-8B、Dense/BGE-M3、BindPack/BGE-M3。',
        '', '| API 组 | Recall@5 | FCR@5 | DCC@5 | QA F1 | 新HTTP请求 | 缓存命中 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for job in ('dense_qwen','dense_bge','bindpack_bge'):
        case=root/'model_cases'/job
        if job=='bindpack_bge':case=case/'run/cases/musique/full'
        if not (case/'completed.ok').exists():table.append('| '+job+' | 运行中/等待 | — | — | — | — | — |');continue
        report=read(case/'report.json');metrics=report['metrics'];cost=report['cost']
        calls=cost.get('physical_requests',cost.get('llm_http_attempts'))
        hits=report.get('cache_hits',cost.get('llm_cache_hits',0))
        table.append('| '+job+' | '+' | '.join(f'{metrics[k]*100:.2f}' for k in ('Recall@5','FCR@5','DCC@5','F1'))+f' | {calls} | {hits} |')
    table+=['','原始API抽取与响应缓存费用只作为历史来源，不重复算入新请求成本。缓存键同时匹配模型、API地址、全部消息和有效生成配置。',
        'BGE版保留API模型抽取，重新生成全部BGE向量与同义边；不复用Qwen embedding向量。',
        '各case保存report.json、result.json、完成哈希、逐题断点和请求账本；完成后自动更新论文表格与PDF。']
    (root/'RESULTS.md').write_text('\n'.join(table)+'\n')

def controller(root):
    if not (root/'main_paper_updated.ok').exists():raise ValueError('Complete main QA and publish its paper tables first')
    freeze(root)
    credentials=read(CREDENTIALS)
    original_config=read(API_SOURCE/'config.json')['runtime']
    assert credentials['model']==original_config['llm_name']=='qwen3.8-flash'
    assert credentials['base_url']==original_config['llm_base_url']
    env=dict(os.environ,OPENAI_API_KEY=credentials['api_key'],DASHSCOPE_API_KEY=credentials['api_key'],
        PYTHONPATH='/root/.cache/pathcondrag/runtime_deps_qwen3_tf4513',PYTHONUNBUFFERED='1',TQDM_DISABLE='1',
        TOKENIZERS_PARALLELISM='false',BINDPACK_LLM_MAX_IN_FLIGHT='32',BINDPACK_PROVIDER_REJECTION_POLICY='abstain',
        BINDPACK_RETRIEVAL_DEVICE='cuda',BINDPACK_KNN_DEVICE='cuda',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
    snapshot=root/'api_source_snapshot';runner=snapshot/'scripts/run_remaining_api_models.py'
    write(root/'api.pid.json',dict(pid=os.getpid(),started_at=stamp()))
    summarize_queue(root)
    for job in ('dense_qwen','dense_bge','bindpack_bge'):
        write(root/'status.json',dict(stage='api_models',job=job,started_at=stamp()))
        print(stamp(),'START MODEL CASE',job,flush=True)
        if job!='bindpack_bge':
            run_command([sys.executable,'-B','-u',str(runner),'--output',str(root),'--job',job],env,root/'model_cases'/job/'run.log')
        else:
            (root/'index_build').mkdir(exist_ok=True)
            run_command([sys.executable,'-B','-u',str(runner),'--output',str(root),'--job','build_bge'],env,root/'index_build/run.log')
            out=root/'model_cases/bindpack_bge/run'
            cmd=[sys.executable,'-B','-u',str(snapshot/'scripts/run_bindpack_measured.py'),'--frozen-worker',
                '--dataset','musique','--output',str(out),'--config',str(root/'api_bge_config.json'),
                '--datasets-dir',str(PROJECT/'datasets'),'--reuse-archive-index',str(root/'shared_indexes/musique'),
                '--query-window','32','--reader-workers','32','--model-factor-profile','dashscope_bge_m3',
                '--response-cache-seed',str(root/'response_cache/qwen3.8-flash_cache.sqlite'),
                '--source-bound-eligibility','off','--eval-mode','rag_qa','--qa-max-tokens','2028','--candidate-output-top-k','10']
            if (out/'experiment_config.json').exists():cmd+=['--resume']
            run_command(cmd,env,root/'model_cases/bindpack_bge/run.log')
        print(stamp(),'PASS MODEL CASE',job,flush=True)
        summarize_queue(root)
        # Model results are always retained even if a future paper edit breaks compilation.
        paper_env=dict(os.environ);paper_env.pop('PYTHONPATH',None)
        publish=subprocess.run(['/root/anaconda3/bin/python','-B',str(PROJECT/'scripts/publish_completed_paper_results.py'),
            '--batch-root',str(root),'--tag',job+'_complete'],cwd=PROJECT,env=paper_env)
        if publish.returncode:print('PAPER UPDATE PENDING; completed model report retained:',job,flush=True)
    write(root/'api_models.completed.ok',dict(status='complete',cases=3,finished_at=stamp()))
    write(root/'status.json',dict(stage='complete',cases=3,finished_at=stamp()))
    print(stamp(),'ALL THREE MODEL CASES COMPLETE',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--job',choices=['queue','dense_qwen','dense_bge','build_bge'],default='queue')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if CREDENTIALS.stat().st_mode&0o077:raise ValueError('Credentials must be owner-only readable')
    if args.job=='queue':controller(args.output.resolve())
    else:
        sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
        if args.job=='build_bge':build_bge(args.output.resolve())
        else:dense(args.output.resolve(),args.job)

#!/usr/bin/env python3
"""Small, genuinely fresh API index + native retrieval/QA; never starts a full run."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

PROJECT=Path('/root/BindPack')
CREDENTIALS=Path('/root/.config/bindpack/dashscope.json')

def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(value.encode()).hexdigest()

def setup(root):
    sys.path.insert(0,str(PROJECT/'scripts'))
    from run_cloud_api_pilot import select_indices
    from eval_utils import build_docs_from_full_corpus,ensure_gold_docs_in_corpus,get_gold_docs
    data=read(PROJECT/'datasets/musique.json')
    indices,hops=select_indices(data,'musique',4)
    selected=[data[index] for index in indices]
    all_gold=[sorted(set(group)) for group in get_gold_docs(data,'musique')]
    full_docs=ensure_gold_docs_in_corpus(build_docs_from_full_corpus(read(PROJECT/'datasets/musique_corpus.json')),all_gold)
    full_docs=list(dict.fromkeys(full_docs))
    uniform_positions=random.Random(20261030).sample(range(len(full_docs)),128)
    uniform=[full_docs[index] for index in uniform_positions]
    required=list(dict.fromkeys(doc for group in get_gold_docs(selected,'musique') for doc in sorted(group)))
    docs=list(dict.fromkeys(uniform+required))
    snapshot=root/'source_snapshot'
    for folder in ('src','scripts','configs'):
        shutil.copytree(PROJECT/folder,snapshot/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    credentials=read(CREDENTIALS)
    config=read(PROJECT/'configs/ablations_qwen3.json')
    config.update(method_config=str(snapshot/'configs/full_no_source_eligibility.json'),datasets=['musique'])
    config['runtime'].update(llm_name=credentials['model'],llm_base_url=credentials['base_url'],
        llm_prefetch_workers=32,openie_max_workers=8,embedding_batch_size=32)
    config['evaluation'].update(eval_mode='rag_qa',sample_size=4)
    write(root/'config.json',config)
    write(root/'frozen_datasets/musique.json',selected)
    write(root/'frozen_datasets/musique_corpus.json',[
        dict(title=doc.split('\n',1)[0],text=doc.split('\n',1)[1]) for doc in docs])
    manifest=dict(original_query_indices=indices,hop_distribution=hops,full_questions=len(data),
        full_unique_passages=len(full_docs),pilot_questions=4,pilot_passages=len(docs),
        uniform_random_passages=128,uniform_original_positions=uniform_positions,
        uniform_chunk_ids=['chunk-'+hashlib.md5(doc.encode()).hexdigest() for doc in uniform],
        # Native chunk keys use the prefix recorded in each OpenIE row; hashes
        # below also let the analysis match exact text without assuming it.
        uniform_document_sha256=[digest(doc) for doc in uniform],
        added_gold_context_passages=len(docs)-len(uniform),
        selection='128 uniform random corpus passages (fixed seed), plus gold context only for four functional QA probes',
        evaluation_scope='small-corpus functional integration; not formal full-corpus retrieval quality',
        index_reused=False,index_extraction_model=credentials['model'],
        online_model=credentials['model'],embedding='/root/models/Qwen3-Embedding-8B',
        thinking_enabled=False,qa_max_tokens=2028,index_workers=8,online_workers=32,
        gpu_embedding=True,gpu_vector_search=True,gpu_pagerank=True,gpu_synonym_knn=True,
        full_run_authorized=False,created_unix=time.time(),
        original_data_sha256=sha(PROJECT/'datasets/musique.json'),
        original_corpus_sha256=sha(PROJECT/'datasets/musique_corpus.json'),
        source_sha256={str(path.relative_to(snapshot)):sha(path) for path in snapshot.rglob('*') if path.is_file()})
    write(root/'pilot_protocol.json',manifest)
    return manifest

def main(args):
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    if CREDENTIALS.stat().st_mode&0o077:raise ValueError('Credentials must be owner-only readable')
    if not (root/'pilot_protocol.json').exists():protocol=setup(root)
    elif not args.resume:raise ValueError('Pilot exists: use --resume')
    else:protocol=read(root/'pilot_protocol.json')
    for relative,expected in protocol['source_sha256'].items():
        if sha(root/'source_snapshot'/relative)!=expected:raise ValueError('Frozen source changed: '+relative)
    if args.resume and (root/'completed.ok').exists():
        done=read(root/'run/cases/musique/full/completed.ok')
        for name,key in [('report.json','report_sha256'),('result.json','result_sha256'),('llm_calls.jsonl','llm_calls_sha256')]:
            if sha(root/'run/cases/musique/full'/name)!=done[key]:
                raise ValueError('Completed artifact changed: '+name)
        marker=read(root/'completed.ok')
        if sha(root/'COST_ESTIMATE.json')!=marker['cost_estimate_sha256']:
            raise ValueError('Completed estimate changed')
        print('Pilot already complete; artifacts verified; zero new model calls',flush=True)
        return
    credentials=read(CREDENTIALS)
    assert credentials['model']==protocol['index_extraction_model']=='qwen3.8-flash'
    env=dict(os.environ,OPENAI_API_KEY=credentials['api_key'],DASHSCOPE_API_KEY=credentials['api_key'],
        PYTHONPATH='/root/.cache/pathcondrag/runtime_deps_qwen3_tf4513',
        BINDPACK_RETRIEVAL_DEVICE='cuda',BINDPACK_KNN_DEVICE='cuda',
        PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
    command=[sys.executable,'-B','-u',str(root/'source_snapshot/scripts/run_bindpack_measured.py'),
        '--frozen-worker','--dataset','musique','--output',str(root/'run'),
        '--config',str(root/'config.json'),'--datasets-dir',str(root/'frozen_datasets'),
        '--query-window','8','--reader-workers','8',
        '--model-factor-profile','dashscope_qwen3_emb8b_fresh',
        '--source-bound-eligibility','off','--eval-mode','rag_qa',
        '--qa-max-tokens','2028','--candidate-output-top-k','10']
    if args.resume and (root/'run/experiment_config.json').exists():command+=['--resume']
    write(root/'launch.json',dict(command=command,index_reuse=False,full_run=False,
        credentials_values_recorded=False,started_unix=time.time()))
    start=time.perf_counter()
    with (root/'run.log').open('a') as log:
        process=subprocess.Popen(command,cwd=PROJECT,env=env,stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        write(root/'run.pid.json',dict(controller_pid=os.getpid(),worker_pid=process.pid))
        for line in process.stdout:
            log.write(line);log.flush();print(line,end='',flush=True)
        code=process.wait()
    write(root/'pilot_execution.json',dict(returncode=code,wall_seconds=time.perf_counter()-start,
        finished_unix=time.time(),full_run_started=False))
    if code:raise RuntimeError('Pilot failed; no full dataset run was started. Inspect '+str(root/'run.log'))
    print('FRESH INDEX + RETRIEVAL + QA PILOT COMPLETE; no full experiment started',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    main(parser.parse_args())

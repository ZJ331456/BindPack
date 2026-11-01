#!/usr/bin/env python3
"""Native stage runs with durable checkpoints and measured reader@5 costs/resources."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import threading
import time
import traceback
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_ROOT = Path("/root/BindPack")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["hotpotqa", "2wikimultihopqa", "musique", "hover"], default="hover")
    p.add_argument("--output", required=True)
    p.add_argument("--config", default=str(ORIGINAL_ROOT / "configs/ablations_qwen3.json"))
    p.add_argument("--datasets-dir", default=str(ORIGINAL_ROOT / "datasets"))
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--case", choices=['full','wo_route','wo_plan','wo_pack'], default='full')
    p.add_argument("--mechanism-ablation", choices=['full','wo_source_bound_eligibility',
        'wo_dependency_closure','wo_joint_selection'], default='full')
    p.add_argument("--source-bound-eligibility", choices=['on','off'], default=None,
                   help='Explicit baseline setting, independently combined with dependency/joint ablations')
    p.add_argument("--eval-mode", choices=['rag_qa','retrieve'], default='rag_qa')
    p.add_argument("--candidate-output-top-k",type=int,default=200,
                   help='Text export prefix; complete passage hashes and gold ranks are retained')
    p.add_argument("--reuse-index", help="completed Full case's index; independent graph/vector copy, empty LLM cache")
    p.add_argument("--reuse-archive-index", help="verified archive/shared_indexes/<dataset>; no rebuilding or historical query cache")
    p.add_argument("--query-limit", type=int, help="development check on a full archived corpus; preserve native hop/label diversity")
    p.add_argument("--query-indices", help="Frozen JSON list of original dataset indices; reuse the full archived corpus")
    p.add_argument("--prepare-only", action="store_true", help="freeze and validate inputs without loading models or sending LLM requests")
    p.add_argument("--query-window", type=int, choices=[1,8,16,32], default=8,
                   help="Native cross-query prefetch window; datasets remain serial")
    p.add_argument("--reader-workers", type=int, choices=[1,8,16,32], default=1,
                   help="Opt-in concurrent native Reader calls within each durable retrieval window")
    p.add_argument("--model-factor-profile", choices=['qwen3_bge_m3','dashscope_qwen3_emb8b',
                                                    'dashscope_qwen3_emb8b_fresh','dashscope_bge_m3'],
                   help="Explicit Table 6 model replacement; other runs retain the fixed-model guard")
    p.add_argument('--index-extraction-llm', help='Explicit original extraction model when replacing only the online LLM')
    p.add_argument('--qa-max-tokens',type=int,help='Reader-only output token cap; reaching it retains the response without retry')
    p.add_argument('--response-cache-seed',help='Explicit same-provider/model response cache; exact native cache keys only')
    p.add_argument("--frozen-worker", action="store_true", help=argparse.SUPPRESS)
    return p


def freeze_worker(args):
    output = Path(args.output).resolve()
    if not args.resume and output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output directory, or explicitly --resume the same frozen run")
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / "source_snapshot"
    if args.case != 'full':
        if not args.resume or not snapshot.exists():
            raise ValueError('Append ablations to a frozen Full batch using --resume')
        from utils.measured_cases import install_ablation_snapshot
        snapshot = install_ablation_snapshot(output, ROOT/'scripts')
    elif args.resume:
        if not (snapshot / "scripts/run_bindpack_measured.py").exists():
            raise ValueError("Missing frozen worker for resume")
    else:
        for name in ("src", "scripts", "configs"):
            shutil.copytree(ROOT / name, snapshot / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    argv = [sys.executable, "-B", "-u", str(snapshot / "scripts/run_bindpack_measured.py"),
        "--dataset", args.dataset, "--output", str(output), "--datasets-dir", str(Path(args.datasets_dir).resolve()),
        "--config", str(Path(args.config).resolve()), "--frozen-worker"]
    if args.smoke: argv.append("--smoke")
    if args.resume: argv.append("--resume")
    if args.prepare_only: argv.append("--prepare-only")
    if args.case != 'full': argv += ['--case',args.case]
    if args.reuse_index: argv += ['--reuse-index',str(Path(args.reuse_index).resolve())]
    if args.reuse_archive_index: argv += ['--reuse-archive-index',str(Path(args.reuse_archive_index).resolve())]
    if args.query_limit is not None: argv += ['--query-limit',str(args.query_limit)]
    if args.query_indices is not None: argv += ['--query-indices',str(Path(args.query_indices).resolve())]
    argv += ['--query-window',str(args.query_window)]
    argv += ['--reader-workers',str(args.reader_workers)]
    if args.model_factor_profile:
        argv += ['--model-factor-profile',args.model_factor_profile]
    if args.index_extraction_llm:
        argv += ['--index-extraction-llm',args.index_extraction_llm]
    if args.qa_max_tokens is not None:argv += ['--qa-max-tokens',str(args.qa_max_tokens)]
    if args.response_cache_seed:argv += ['--response-cache-seed',str(Path(args.response_cache_seed).resolve())]
    argv += ['--mechanism-ablation',args.mechanism_ablation,'--eval-mode',args.eval_mode]
    if args.source_bound_eligibility is not None:
        argv += ['--source-bound-eligibility',args.source_bound_eligibility]
    argv += ['--candidate-output-top-k',str(args.candidate_output_top_k)]
    os.execv(sys.executable, argv)


def run(args):
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    from bindpack import BindPack, BindPackConfig
    from bindpack.evaluation.measurement import (Recorder, ResourceMonitor, digest, distribution,
        jsonl, sha256, usage, write_json)
    from bindpack.evaluation.measured_reader import prepare
    from bindpack.evaluation.result_export import detailed_result
    from bindpack.evaluation.metric_report import write_metric_report
    from eval_utils import (build_docs_from_full_corpus, ensure_gold_docs_in_corpus,
        get_benchmark_hops, describe_benchmark_hops, get_gold_answers, get_gold_docs,
        get_query_text, get_sample_id)

    output = Path(args.output).resolve()
    stage = args.case
    if not 5<=args.candidate_output_top_k<=200:raise ValueError('candidate-output-top-k must be 5..200')
    case = output / "cases" / args.dataset / stage
    metadata = output / "metadata" / args.dataset
    case.mkdir(parents=True, exist_ok=True)
    metadata.mkdir(parents=True, exist_ok=True)
    lock = (ORIGINAL_ROOT / "outputs/.bindpack_measured.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config_file = output / "experiment_config.json"
    if not args.resume:
        shutil.copy2(args.config, config_file)
        for stem in (args.dataset, args.dataset + "_corpus"):
            shutil.copy2(Path(args.datasets_dir) / (stem + ".json"), output / (stem + ".json"))
    settings = json.loads(config_file.read_text())
    runtime = settings["runtime"]
    expected_embedding = ('/root/models/bge-m3' if args.model_factor_profile in ('qwen3_bge_m3','dashscope_bge_m3')
                          else '/root/models/Qwen3-Embedding-8B')
    if (runtime["llm_model_path"] != "/root/models/Qwen3-8B" or
            runtime["embedding_model_name"] != expected_embedding):
        raise ValueError("Measured comparison requires the fixed Qwen3 models")
    if args.index_extraction_llm and (args.model_factor_profile!='dashscope_qwen3_emb8b' or not args.reuse_archive_index):
        raise ValueError('Changing online versus extraction LLM requires explicit cloud profile and verified archived index')
    if args.model_factor_profile in ('dashscope_qwen3_emb8b','dashscope_qwen3_emb8b_fresh','dashscope_bge_m3'):
        from bindpack.llm.openai_gpt import _is_dashscope_endpoint
        if not _is_dashscope_endpoint(runtime['llm_base_url']) or runtime['llm_name']!='qwen3.8-flash':
            raise ValueError('Cloud profile expects the configured DashScope qwen3.8-flash endpoint/model')
        if args.model_factor_profile=='dashscope_qwen3_emb8b' and not args.index_extraction_llm:
            raise ValueError('Cloud profile must explicitly retain the original extraction-model identity')
        if args.model_factor_profile=='dashscope_qwen3_emb8b_fresh':
            if args.reuse_index or args.reuse_archive_index or args.index_extraction_llm:
                raise ValueError('Fresh cloud extraction profile forbids all index reuse')
            if runtime['openie_validation_mode']!='structural':
                raise ValueError('Cloud extraction currently supports the structural OpenIE contract')
    if args.response_cache_seed and not (args.model_factor_profile or '').startswith('dashscope_'):
        raise ValueError('Response-cache reuse requires an explicit cloud model profile')
    if args.reader_workers > 1 and (args.query_window == 1 or args.eval_mode != 'rag_qa'):
        raise ValueError('Concurrent Reader requires window retrieval and actual QA')
    if args.qa_max_tokens is not None and not 1<=args.qa_max_tokens<=2048:
        raise ValueError('Reader cap must fit the reserved output budget1..2048')
    data_path, corpus_source = output / (args.dataset + ".json"), output / (args.dataset + "_corpus.json")
    data, corpus = json.loads(data_path.read_text()), json.loads(corpus_source.read_text())
    hops_all = get_benchmark_hops(data, args.dataset)
    indices = list(range(len(data)))
    if args.reuse_index and args.reuse_archive_index:
        raise ValueError('Select one index reuse source')
    if args.query_limit is not None and (args.query_limit < 1 or not args.reuse_archive_index):
        raise ValueError('query-limit requires a positive limit and an archived full corpus')
    if args.query_indices is not None:
        if args.smoke or args.query_limit is not None or not args.reuse_archive_index:
            raise ValueError('query-indices requires archived corpus and cannot combine with smoke/query-limit')
        indices=json.loads(Path(args.query_indices).read_text())
        if (not isinstance(indices,list) or not indices or
                any(type(i) is not int or not 0<=i<len(data) for i in indices) or len(set(indices))!=len(indices)):
            raise ValueError('query-indices must be nonempty, unique, in-range integer indices')
    if args.smoke or args.query_limit is not None:
        # Cover distinct native hop/label combinations; this is a development check.
        chosen, seen = [], set()
        for i, sample in enumerate(data):
            key = (hops_all[i], sample.get("label"))
            if key not in seen:
                seen.add(key); chosen.append(i)
            if len(chosen) == (args.query_limit or 3): break
        if args.query_limit is not None:
            chosen += [i for i in range(len(data)) if i not in chosen][:max(0,args.query_limit-len(chosen))]
        indices = chosen
    samples = [data[i] for i in indices]
    # The legacy helper returns lists converted from Python sets. Canonicalize
    # annotation order before corpus completion/fixture construction so another
    # interpreter/hash seed produces the same frozen corpus on resume.
    gold = [sorted(set(group)) for group in get_gold_docs(samples, args.dataset)]
    all_gold = [sorted(set(group)) for group in get_gold_docs(data, args.dataset)]
    docs = build_docs_from_full_corpus(corpus)
    before = len(docs)
    docs = ensure_gold_docs_in_corpus(docs, all_gold)
    added = len(docs) - before
    if args.smoke and not args.reuse_archive_index:
        required = list(dict.fromkeys(doc for group in gold for doc in group))
        pool = [d for d in docs if d not in set(required)]
        random.Random(42).shuffle(pool)
        docs = required + pool[:max(0, 32-len(required))]
    identity = dict(dataset=args.dataset, data_path=str(data_path), data_sha256=sha256(data_path),
        corpus_source=str(corpus_source), corpus_source_sha256=sha256(corpus_source),
        source_data_path=str(Path(args.datasets_dir) / (args.dataset + ".json")),
        selected_indices=indices, sample_ids=[get_sample_id(s, args.dataset) for s in samples],
        corpus_docs=len(docs), corpus_docs_sha256=digest(docs), gold_completion_added=added,
        development_fixture=args.smoke or args.query_limit is not None or args.query_indices is not None, oracle_hops=True,
        hop_provenance=describe_benchmark_hops(data, args.dataset))
    def freeze(path, value):
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError(f"Frozen input/protocol changed: {path}")
        if not path.exists(): write_json(path, value)
    freeze(metadata / "sample_manifest.json", identity)
    freeze(metadata / "corpus.json", docs)
    model_paths = [Path(runtime[key]) for key in ("llm_model_path", "embedding_model_name")]
    model_identity = {str(p): {str(f.relative_to(p)): dict(bytes=f.stat().st_size,
        sha256=sha256(f) if f.suffix == ".json" else None)
        for f in sorted(p.rglob('*')) if f.is_file() and f.suffix in ('.json','.safetensors','.bin')}
        for p in model_paths}
    if args.model_factor_profile in ('dashscope_qwen3_emb8b','dashscope_qwen3_emb8b_fresh','dashscope_bge_m3'):
        model_identity.pop(runtime['llm_model_path'])
        model_identity['remote_llm']=dict(model=runtime['llm_name'],base_url=runtime['llm_base_url'],
            thinking_enabled=False,credentials='environment only; never included in protocol')
        model_identity['reader_budget_tokenizer']=dict(path=runtime['llm_model_path'],
            purpose='local conservative context preflight; actual API billing tokens come from server usage',
            tokenizer_config_sha256=sha256(Path(runtime['llm_model_path'])/'tokenizer_config.json'))
    sources = {str(p.relative_to(ROOT)): sha256(p) for folder in (ROOT/'src', ROOT/'scripts')
               for p in sorted(folder.rglob('*.py'))}
    source_method = Path(settings["method_config"])
    if not source_method.is_absolute(): source_method = ROOT / source_method
    method = output / "method_config.json"
    if not method.exists(): shutil.copy2(source_method, method)
    overrides = {k: runtime[k] for k in ("llm_name", "llm_base_url", "embedding_model_name", "embedding_batch_size",
        "openie_max_workers", "llm_prefetch_workers", "max_new_tokens", "openie_strict",
        "openie_prompt_version", "openie_validation_mode")}
    if args.source_bound_eligibility is not None:
        overrides['source_bound_eligibility'] = args.source_bound_eligibility == 'on'
    config = BindPackConfig.from_file(method, **overrides, save_dir=str(case/'index'), dataset=args.dataset,
        corpus_len=len(docs), stage_ablation=stage, retrieval_top_k=200, qa_top_k=5, max_qa_steps=1,
        mechanism_ablation=args.mechanism_ablation)
    protocol = dict(schema_version=1, mode="smoke" if identity['development_fixture'] else "full", method="bindpack", stage_ablation=stage,
        dataset=identity, configuration=asdict(config), models=model_identity, source=sources,
        query_concurrency=args.query_window, native_llm_concurrency=config.llm_prefetch_workers, embedding_batch_size=config.embedding_batch_size,
        cache_policy="independent empty cache on fresh run; retain cache and all real attempts on resume",
        hop_condition="oracle native benchmark hops, recorded explicitly; no gold documents/answers passed to retrieval",
        reader_policy="native prompt/parser; reader@5; baseline-compatible 8192 input+output prefix budget",
        resources="2s samples; CPU process/tree; whole shared GPU utilization/memory/power; no exclusive GPU attribution")
    protocol.update(mechanism_ablation=args.mechanism_ablation,eval_mode=args.eval_mode,
        candidate_output_top_k=args.candidate_output_top_k,
        qa_executed=args.eval_mode=='rag_qa',upstream_policy='fresh native Route and Plan; no historical model responses')
    if args.model_factor_profile:
        protocol['model_factor_profile'] = args.model_factor_profile
        protocol['http_max_in_flight'] = int(os.environ.get('BINDPACK_LLM_MAX_IN_FLIGHT','8'))
        if args.model_factor_profile.startswith('dashscope_'):
            protocol['provider_rejection_policy']=os.environ.get('BINDPACK_PROVIDER_REJECTION_POLICY','stop')
            protocol['provider_rejection_scope']='explicit data_inspection_failed only; retain failed request, original source and every query; no retry or content rewriting'
    if args.index_extraction_llm:
        protocol['index_extraction_llm']=args.index_extraction_llm
        protocol['model_replacement_scope']='online Route/Plan/Reader replacement; original extraction and embedding index retained'
    if args.model_factor_profile=='dashscope_qwen3_emb8b_fresh':
        protocol['index_extraction_llm']=runtime['llm_name']
        protocol['model_replacement_scope']='fresh API NER/OpenIE, fresh graph and all vectors, API Route/Plan/Reader'
    if args.model_factor_profile=='dashscope_bge_m3':
        protocol['index_extraction_llm']=runtime['llm_name']
        protocol['model_replacement_scope']='same API extraction reused; BGE passage/entity/fact vectors and synonym edges rebuilt; native API Route/Plan/Reader'
    if args.response_cache_seed:
        protocol['response_cache_reuse']=dict(source=str(Path(args.response_cache_seed).resolve()),
            policy='exact native v2 keys include messages, effective generation parameters and base_url; real cache hits recorded; inherited cost excluded')
        protocol['cache_policy']='explicit same-model/provider response cache seed; exact native keys; retain all new physical attempts'
    if args.reader_workers > 1:
        protocol['reader_concurrency'] = args.reader_workers
    if args.qa_max_tokens is not None:
        protocol['qa_max_tokens']=args.qa_max_tokens
        protocol['qa_length_policy']='retain and parse truncated native response; no length retry or continuation'
    if os.environ.get('BINDPACK_RETRIEVAL_DEVICE','cpu')=='cuda':
        protocol['retrieval_compute_device']='cuda'
        protocol['pagerank_compute_device']='cuda'
    if args.reuse_index:
        protocol['index_reuse'] = dict(source=str(Path(args.reuse_index).resolve()),
            policy='private graph/vector/OpenIE copy; no Full or cross-ablation LLM/query caches')
    if args.reuse_archive_index:
        protocol['index_reuse'] = dict(source=str(Path(args.reuse_archive_index).resolve()),
            policy='verified immutable archive; private copies; independent empty application cache')
    execution_protocol=None
    if args.query_window>1:
        from utils.window_execution import freeze_protocol
        protocol,execution_protocol=freeze_protocol(case,protocol,freeze)
    else:freeze(case/'protocol.json', protocol)
    if args.prepare_only:
        write_json(output/'prepared.ok',dict(queries=len(indices),corpus_docs=len(docs),
            protocol_sha256=digest(protocol),model_calls=0))
        print('Inputs prepared; no model calls',flush=True)
        return
    if (case/'completed.ok').exists():
        marker = json.loads((case/'completed.ok').read_text())
        for name in ('report','result','llm_calls'):
            suffix = '.jsonl' if name == 'llm_calls' else '.json'
            if marker[name+'_sha256'] != sha256(case/(name+suffix)):
                raise ValueError("Completed artifact changed: "+name)
        print("Already complete; no model calls", flush=True)
        return
    rows=[]
    for i,item in enumerate(jsonl(case/'progress.jsonl')):
        row=item['row']
        if item['sha256'] != digest(row) or row['query_index'] != indices[i] or row['sample_id'] != identity['sample_ids'][i]:
            raise ValueError("Invalid or mismatched durable query checkpoint")
        rows.append(row)
    recorder=Recorder(case)
    monitor=ResourceMonitor(recorder)
    started=time.perf_counter()
    prior_seconds=sum(s['wall_seconds'] for s in jsonl(case/'execution_sessions.jsonl'))
    stop=threading.Event()
    def heartbeat():
        while not stop.is_set():
            n=len(rows);bar='#'*int(24*n/len(indices))+'-'*(24-int(24*n/len(indices)))
            calls=sum(1 for _ in (case/'llm_calls.jsonl').open()) if (case/'llm_calls.jsonl').exists() else 0
            status=dict(dataset=args.dataset,method="bindpack",stage_ablation=stage,phase=recorder.phase,
                completed_queries=n,expected_queries=len(indices),llm_http_attempts=calls,
                elapsed_seconds=prior_seconds+time.perf_counter()-started,updated_unix_seconds=time.time(),pid=os.getpid())
            write_json(output/'status.json', status)
            print(f"{datetime.now():%Y-%m-%d %H:%M:%S} PROGRESS {args.dataset}/{stage} [{bar}] {n}/{len(indices)} | {recorder.phase} | LLM attempts={calls} | elapsed={status['elapsed_seconds']:.0f}s",flush=True)
            stop.wait(30)
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    session_status="failed"
    requested_stop=threading.Event()
    previous_handlers={number:signal.getsignal(number) for number in (signal.SIGTERM,signal.SIGINT)}
    def finish_current_query(number,frame):
        requested_stop.set()
    for number in previous_handlers:signal.signal(number,finish_current_query)
    try:
        request=urllib.request.Request(runtime['llm_base_url'].rstrip('/')+'/models',
            headers={'Authorization':'Bearer '+os.environ.get('OPENAI_API_KEY','EMPTY')})
        with urllib.request.urlopen(request,timeout=10) as response: service=json.load(response)
        if runtime['llm_name'] not in {v['id'] for v in service['data']}:raise ValueError("Unexpected vLLM served model")
        write_json(output/'service_identity.json',service)
        with monitor:
            copy_seconds = 0.0
            if args.reuse_archive_index:
                from utils.archived_index import prepare_archive
                with recorder.section('offline_index'):
                    copy_start=time.perf_counter()
                    prepare_archive(args.reuse_archive_index,case/'index',case,identity,docs,config,
                        extraction_llm_name=args.index_extraction_llm)
                    copy_seconds=time.perf_counter()-copy_start
            if args.reuse_index:
                from utils.measured_cases import prepare_reused_index
                with recorder.section('offline_index'):
                    copy_start = time.perf_counter()
                    prepare_reused_index(args.reuse_index,case/'index',case,identity,protocol['models'])
                    copy_seconds = time.perf_counter()-copy_start
            with recorder.section('model_loading'):
                from transformers import AutoTokenizer
                tokenizer=AutoTokenizer.from_pretrained(runtime['llm_model_path'],local_files_only=True)
                from bindpack.llm.openai_gpt import CacheOpenAI
                if args.response_cache_seed:
                    from utils.llm_cache_reuse import seed_response_cache
                    receipt=seed_response_cache(args.response_cache_seed,
                        case/'index/llm_cache'/(runtime['llm_name'].replace('/','_')+'_cache.sqlite'))
                    write_json(case/'response_cache_seed.json',receipt)
                recorder.install_llm_factory(CacheOpenAI,tokenizer)
                rag=BindPack(config=config)
                if args.qa_max_tokens is not None:
                    from functools import partial
                    rag.qa=partial(rag.qa,max_output_tokens=args.qa_max_tokens)
                if args.eval_mode=='retrieve':
                    def deny_qa(*a,**kw):raise AssertionError('Final QA generation is disabled in this experiment')
                    rag.qa=deny_qa
                recorder.install_embedding(rag.embedding_model)
                recorder.install_index(rag)
                if args.reuse_archive_index:
                    def deny_build(*a,**kw):raise RuntimeError('This run must reuse the archived index; index construction is forbidden')
                    rag.index=deny_build
            build_path=case/'index_build.json'
            with recorder.section('offline_index'):
                begin=time.perf_counter()
                if build_path.exists():
                    build=json.loads(build_path.read_text())
                    if build['corpus_sha256'] != digest(docs):raise ValueError('Index corpus changed')
                    if set(rag.chunk_embedding_store.get_all_texts()) != set(docs):raise ValueError('Index passages changed')
                    rag.prepare_retrieval_objects();rag._build_index_side_artifacts()
                else:
                    rag.index(docs=docs)
                    build=dict(seconds=time.perf_counter()-begin,corpus_sha256=digest(docs),
                        index_bytes=sum(p.stat().st_size for p in (case/'index').rglob('*') if p.is_file()),
                        passages=len(docs),graph_nodes=rag.graph.vcount(),graph_edges=rag.graph.ecount())
                    write_json(build_path,build)
                prepare_seconds=copy_seconds+time.perf_counter()-begin
            original_build='N/A' if build['seconds'] is None else f"{build['seconds']:.2f}s"
            print(f"{datetime.now():%Y-%m-%d %H:%M:%S} INDEX READY {args.dataset}; passages={len(docs)}; original_build={original_build}",flush=True)
            def serial_solutions():
                for position in range(len(rows),len(indices)):
                    if requested_stop.is_set():raise KeyboardInterrupt('Stopped at a durable query boundary; all prior costs/checkpoints retained')
                    qid=identity['sample_ids'][position]
                    with recorder.section('online_retrieval',qid):
                        begin=time.perf_counter()
                        rag.set_query_hop_overrides([hops_all[indices[position]]],source='benchmark')
                        solution=rag.retrieve([get_query_text(samples[position],args.dataset)],num_to_retrieve=200)[0]
                        seconds=time.perf_counter()-begin
                    yield position,solution,dict(execution_mode='serial',retrieval_seconds=seconds)
            if args.query_window>1:
                from utils.window_execution import iter_windows
                stream=iter_windows(rag,case,len(rows),[get_query_text(s,args.dataset) for s in samples],
                    [hops_all[j] for j in indices],identity['sample_ids'],recorder,requested_stop,args.query_window)
            else:stream=serial_solutions()
            if args.reader_workers > 1:
                from utils.parallel_reader import iter_readers
                stream=iter_readers(stream,rag,case,identity['sample_ids'],recorder,requested_stop,
                    args.reader_workers,lambda solution:prepare(rag,solution,tokenizer,args.dataset))
            for i,solution,retrieval_timing in stream:
                if requested_stop.is_set():raise KeyboardInterrupt('Stopped at a durable query boundary; all prior costs/checkpoints retained')
                sample=samples[i];qid=identity['sample_ids'][i];query=get_query_text(sample,args.dataset)
                prepared_reader=retrieval_timing.pop('_prepared_reader',None)
                if prepared_reader is not None:
                    budget=prepared_reader['budget'];raw=[prepared_reader['raw_response']]
                    metas=[prepared_reader['metadata']];reading_seconds=prepared_reader['seconds'];budget_seconds=0.0
                    solution.answer=prepared_reader['answer']
                else:
                    with recorder.section('online_reader' if args.eval_mode=='rag_qa' else 'budget_preparation',qid):
                        reading_start=time.perf_counter()
                        reader_solution,budget=prepare(rag,solution,tokenizer,args.dataset)
                        if args.eval_mode=='rag_qa':
                            answered,raw,metas=rag.qa([reader_solution])
                            solution.answer=answered[0].answer
                            reading_seconds=time.perf_counter()-reading_start
                        else:
                            solution.answer=None
                            raw,metas=[None],[{'finish_reason':None,'qa_executed':False}]
                            reading_seconds=0.0
                        budget_seconds=time.perf_counter()-reading_start if args.eval_mode=='retrieve' else 0.0
                serial=retrieval_timing['execution_mode']=='serial'
                retrieval_seconds=retrieval_timing.get('retrieval_seconds')
                elapsed=retrieval_seconds+reading_seconds+budget_seconds if serial else None
                solution.gold_answers=get_gold_answers([sample],args.dataset)[0]
                solution.gold_docs=gold[i]
                row=detailed_result(solution,sample,indices[i],gold[i],hops_all[indices[i]],10,args.candidate_output_top_k)
                row.update(ranking_complete=len(row['candidate_docs'])==len(solution.docs),actual_ranking_length=len(solution.docs),
                    timing=dict(end_to_end_seconds=elapsed,retrieval_seconds=retrieval_seconds,
                        reader_seconds=reading_seconds,**{k:v for k,v in retrieval_timing.items() if k!='retrieval_seconds'}),
                    reader=dict(raw_response=raw[0],metadata=metas[0],budget=budget))
                if args.eval_mode=='retrieve':row['timing']['budget_preparation_seconds']=budget_seconds
                if args.candidate_output_top_k<200:
                    row['complete_ranking_content_sha256']=[hashlib.sha256(d.encode()).hexdigest() for d in solution.docs]
                row=json.loads(json.dumps(row,ensure_ascii=False,allow_nan=False))
                recorder.append('progress.jsonl',dict(sha256=digest(row),row=row))
                rows.append(row)
                timing_text=f'{elapsed:.3f}s' if serial else f"window={retrieval_timing['retrieval_window_seconds']:.3f}s/{retrieval_timing['window_size']}q; reader={reading_seconds:.3f}s"
                print(f"{datetime.now():%Y-%m-%d %H:%M:%S} {args.dataset}/{stage}: {i+1}/{len(indices)}; {timing_text}; answer={solution.answer!r}",flush=True)
        profile=accounting(case,rows,recorder,monitor,build,prepare_seconds,prior_seconds+time.perf_counter()-started)
        cuda=getattr(rag,'_cuda_retrieval_index',None)
        if cuda is not None:
            profile['efficiency']['cuda_retrieval']=dict(cuda.stats,ppr_calls=cuda.ppr_calls,
                ppr_seconds=cuda.ppr_seconds,ppr_max_iterations=cuda.max_iterations)
        if args.query_window>1:
            from utils.window_execution import augment_accounting
            augment_accounting(profile,case,rows)
            profile['measurement_protocol']['execution_protocol']=execution_protocol
        if args.reader_workers > 1:
            from utils.parallel_reader import augment_reader_accounting
            augment_reader_accounting(profile,case,rows,args.reader_workers)
        reviews=[r['retrieval_trace'].get('evidence',{}).get('improvement_support_semantic_veto',{}) for r in rows]
        profile['measurement_protocol']['pack_review_mode']=config.pack.review_mode
        profile['measurement_protocol']['pack_selector_mode']=config.pack.selector_mode
        selected=[r['retrieval_trace'].get('evidence',{}).get('budgeted_selection',{}) for r in rows]
        profile['efficiency']['budgeted_selector_seconds']=distribution(d['seconds'] for d in selected if d.get('enabled'))
        profile['efficiency']['budgeted_selector_scope']='feature preparation, bounded search and reader visibility; excludes current-Pack reference S0 construction, Route/Plan and generation'
        profile['cost']['budgeted_selector']=dict(enabled=config.pack_enabled and config.pack.selector_mode!='current' and
            config.mechanism_ablation!='wo_joint_selection',
            llm_calls=sum(d.get('model_requests',0) for d in selected),embedding_calls=0)
        profile['measurement_protocol'].update(mechanism_ablation=config.mechanism_ablation,
            qa_executed=args.eval_mode=='rag_qa',upstream_reexecuted=True)
        if args.eval_mode=='retrieve':
            profile['evidence_budget']['scope']='Prepared Reader@5 context only; final QA was not executed'
            for name in ('actual_passages','evidence_tokens','prompt_tokens'):
                profile['evidence_budget']['prepared_'+name]=profile['evidence_budget'].pop(name)
            profile['evidence_budget']['reader_executed']=False
            profile['efficiency']['budget_preparation_seconds']=distribution(r['timing']['budget_preparation_seconds'] for r in rows)
            prep=sum(r['timing']['budget_preparation_seconds'] for r in rows)
            if args.query_window>1:
                profile['efficiency']['completed_query_processing_wall_seconds']+=prep
                profile['efficiency']['processing_seconds_per_query']=profile['efficiency']['completed_query_processing_wall_seconds']/len(rows)
        profile['cost']['support_semantic_veto']=dict(mode=config.pack.review_mode,
            enabled=config.pack_enabled and config.pack.review_mode!='off',
            logical_extra_requests=sum(d.get('extra_requests',0) for d in reviews),
            accepted_swaps=sum(bool(d.get('accepted')) for d in reviews))
        if config.pack.review_mode=='off':
            profile['cost']['support_semantic_veto'].update(http_attempts=0,input_tokens=0,output_tokens=0)
        if args.reuse_index:
            reused=json.loads((case/'index_reuse.json').read_text())
            profile['efficiency']['offline_index'].update(reused=True,source=reused['source'],
                original_build_cost_scope='shared Full index; one original build, not a new build for this ablation')
            profile['cost']['shared_original_index'] = reused['original_build_usage']
            profile['cost']['shared_original_index_scope'] = 'historical shared build usage; not added to this case physical request totals'
        if args.reuse_archive_index:
            reused=json.loads((case/'index_reuse.json').read_text())
            profile['efficiency']['offline_index'].update(reused=True,source=reused['source'],new_index_build_calls=0,
                original_build_cost_scope=reused['original_build_unavailable_reason'])
            profile['cost']['shared_original_index']=reused['original_build_usage']
            profile['cost']['shared_original_index_scope']=reused['original_build_unavailable_reason']
        if args.model_factor_profile and args.model_factor_profile.startswith('dashscope_'):
            from utils.api_cost import summarize_calls
            cloud_calls=jsonl(case/'llm_calls.jsonl')
            cloud_usage=summarize_calls(cloud_calls)
            profile['cost'].update(currency='CNY',currency_cost=cloud_usage['estimated_cny'],
                currency_cost_unavailable_reason=None,currency_cost_scope='known physical API usage including index and online; unknown usage stays unknown; excludes local hardware',
                api_usage=cloud_usage,api_by_phase={phase:summarize_calls([e for e in cloud_calls if e['phase']==phase])
                    for phase in sorted({e['phase'] for e in cloud_calls})},
                provider_content_rejections=sum(e.get('metadata',{}).get('provider_rejected',False)
                    for e in jsonl(case/'llm_logical_calls.jsonl')))
        result=dict(dataset=args.dataset,method='bindpack',stage_ablation=stage,eval_mode=args.eval_mode,
            mechanism_ablation=config.mechanism_ablation,qa_executed=args.eval_mode=='rag_qa',
            data_path=str(data_path),corpus_path=str(metadata/'corpus.json'),selected_indices=indices,
            sample_size_effective=len(rows),runtime_config=asdict(config),retrieval_metrics={},qa_metrics={},
            hop_provenance=identity['hop_provenance'],hop_distribution=dict(Counter(str(hops_all[i]) for i in indices)),
            accounting=profile,results=rows)
        write_json(case/'result.json',result)
        write_json(case/'validated.ok',dict(result_sha256=sha256(case/'result.json'),protocol_sha256=digest(protocol)))
        report=write_metric_report(case/'result.json',manifest_path=metadata/'sample_manifest.json',ks=[1,3,5,8,10,20,200])
        report.update(profile)
        report.update(mechanism_ablation=config.mechanism_ablation,qa_executed=args.eval_mode=='rag_qa')
        report['display_metrics']={k:'N/A' if v is None else v for k,v in report['metrics'].items()}
        write_json(case/'report.json',report)
        audit=validate(case,rows,profile)
        write_json(case/'audit.json',audit)
        write_json(case/'completed.ok',dict(query_count=len(rows),report_sha256=sha256(case/'report.json'),
            result_sha256=sha256(case/'result.json'),llm_calls_sha256=sha256(case/'llm_calls.jsonl')))
        if stage == 'full':
            write_json(output/'completed.ok',dict(cases=1,queries=len(rows),report=str(case/'report.json')))
            write_json(output/'summary.json',dict(status='complete',dataset=args.dataset,stage_ablation=stage,
                queries=len(rows),metrics=report['metrics'],**profile))
        session_status='complete'
        print('PASS '+args.dataset+'/'+stage+'; metrics='+json.dumps(report['metrics']),flush=True)
    except BaseException as exc:
        write_json(case/'error.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        raise
    finally:
        for number,handler in previous_handlers.items():signal.signal(number,handler)
        stop.set();thread.join(timeout=3)
        recorder.append('execution_sessions.jsonl',dict(wall_seconds=time.perf_counter()-started,status=session_status,
            model_load_seconds=sum(s['seconds'] for s in recorder.phases if s['phase']=='model_loading'),queries=len(rows)))
        state=json.loads((output/'status.json').read_text())
        state.update(phase=session_status,completed_queries=len(rows),updated_unix_seconds=time.time())
        write_json(output/'status.json',state)


def accounting(case,rows,recorder,monitor,build,prepare_seconds,case_seconds):
    from bindpack.evaluation.measurement import jsonl,usage,distribution
    client_workers=json.loads((case/'protocol.json').read_text())['configuration']['llm_prefetch_workers']
    events=jsonl(case/'llm_calls.jsonl');logical=jsonl(case/'llm_logical_calls.jsonl');emb=jsonl(case/'embedding_calls.jsonl')
    offline=usage([e for e in events if e['phase']=='offline_index'])
    online=usage([e for e in events if e['phase'].startswith('online_')])
    online_logical=[e for e in logical if e['phase'].startswith('online_')]
    index_stages=jsonl(case/'stage_timings.jsonl')
    return dict(efficiency=dict(query_count=len(rows),case_wall_seconds=case_seconds,
        query_pipeline_seconds=distribution(r['timing']['end_to_end_seconds'] for r in rows),
        retrieval_seconds=distribution(r['timing']['retrieval_seconds'] for r in rows),
        reader_seconds=distribution(r['timing']['reader_seconds'] for r in rows),
        model_load_seconds=sum(x['seconds'] for x in recorder.phases if x['phase']=='model_loading')+
            sum(s.get('model_load_seconds',0) for s in jsonl(case/'execution_sessions.jsonl')),
        query_concurrency=1,native_llm_concurrency_limit=client_workers,
        embedding_batch_size=json.loads((case/'protocol.json').read_text())['configuration']['embedding_batch_size'],
        native_llm_concurrency_scope='client thread-pool bound; not the physical vLLM GPU sequence count',
        latency_scope='contiguous native retrieval/planning plus budget preparation and reader; excludes model loading/indexing/checkpoint writing',
        cache_condition='independent application cache; resident index/model; first query retained; shared vLLM cache not reset',
        offline_index=dict(reused=len(jsonl(case/'execution_sessions.jsonl'))>0,original_build_seconds=build['seconds'],
            current_prepare_seconds=prepare_seconds,index_bytes=build['index_bytes'],statistics=build,
            stages={s:distribution(e['seconds'] for e in index_stages if e['stage']==s and e['phase']=='offline_index')
                    for s in sorted({e['stage'] for e in index_stages})},
            stage_scope='named method wall intervals include internal IO; embedding provider durations are separately recorded; do not add overlapping timings')),
        cost=dict(online=online,offline_index=offline,llm_http_attempts=online['http_attempts'],
            llm_logical_calls=len(online_logical),llm_cache_hits=sum(e['cache_hit'] is True for e in online_logical),
            llm_call_count_per_query=len(online_logical)/len(rows),
            llm_retries=sum(max(0,n-1) for n in Counter(e['call_id'] for e in events if e['phase'].startswith('online_')).values()),
            llm_failed_attempts=online['failed_attempts'],
            input_tokens_known_total=online['input_tokens'],output_tokens_known_total=online['output_tokens'],
            input_tokens_per_query=online['input_tokens']/len(rows),output_tokens_per_query=online['output_tokens']/len(rows),
            token_usage_complete=online['attempts_with_unknown_usage']==0,
            attempts_with_unknown_token_usage=online['attempts_with_unknown_usage'],
            offline_index_llm_calls=offline['successful_calls'],offline_index_llm_input_tokens=offline['input_tokens'],
            offline_index_llm_output_tokens=offline['output_tokens'],
            finish_length_calls_by_phase={phase:sum(e.get('finish_reason')=='length' for e in events if e['phase']==phase)
                for phase in ['offline_index','online_retrieval','online_reader']},
            currency_cost=None,currency_cost_unavailable_reason='self-hosted; no monetary tariff supplied',
            gpu_compute_seconds=None,gpu_compute_seconds_unavailable_reason='shared GPU; sampled utilization is not exclusive device time',
            embedding={p:dict(calls=len(es),texts=sum(e['texts'] for e in es),effective_tokens=sum(e['effective_tokens'] for e in es),
                truncated_texts=sum(e['truncated_texts'] for e in es),seconds=sum(e['seconds'] for e in es))
                for p in ['offline_index','online_retrieval','online_reader'] for es in [[e for e in emb if e['phase']==p]]}),
        evidence_budget=dict(reader_passages=5,evidence_unit='original passage',
            actual_passages=distribution(r['reader']['budget']['used_passages'] for r in rows),
            evidence_tokens=distribution(r['reader']['budget']['evidence_tokens'] for r in rows),
            prompt_tokens=distribution(r['reader']['budget']['prompt_tokens_local'] for r in rows),
            retrieval_lengths=distribution(len(r['candidate_docs']) for r in rows),
            partial_passage_query_count=sum(bool(r['reader']['budget']['partial_passage_ranks']) for r in rows),
            dropped_passage_query_count=sum(bool(r['reader']['budget']['dropped_passage_ranks']) for r in rows),
            reader_finish_length_count=sum(r['reader']['metadata']['finish_reason']=='length' for r in rows)),
        resources=monitor.summary(),measurement_protocol=dict(version='bindpack-measured-v1',
            resource_units='bytes for RAM; MiB for GPU memory; seconds for time; watts for sampled GPU power',
            native_algorithm_modified=False,oracle_hops=True))


def validate(case,rows,profile):
    from bindpack.evaluation.measurement import jsonl,usage,digest
    assert len({r['sample_id'] for r in rows})==len(rows)
    saved=jsonl(case/'progress.jsonl')
    assert all(e['sha256']==digest(e['row']) for e in saved) and [e['row'] for e in saved]==rows
    events=jsonl(case/'llm_calls.jsonl');logical=jsonl(case/'llm_logical_calls.jsonl')
    assert all(e['call_id'] in {v['call_id'] for v in logical} for e in events)
    assert profile['cost']['online']==usage([e for e in events if e['phase'].startswith('online_')])
    assert profile['cost']['offline_index']==usage([e for e in events if e['phase']=='offline_index'])
    readers=[e for e in logical if e['phase']=='online_reader' and e['status']=='success']
    protocol=json.loads((case/'protocol.json').read_text())
    if protocol.get('qa_executed',True):
        assert set(r['sample_id'] for r in rows)==set(e['query_id'] for e in readers)
        for row in rows:
            last = next(e for e in reversed(readers) if e['query_id']==row['sample_id'])
            assert last['prompt_sha256']==row['reader']['budget']['prompt_sha256']
            assert last['metadata']==row['reader']['metadata']
    else:
        assert not readers and not any(e['phase']=='online_reader' for e in events)
        assert all(r['reader']['metadata'].get('qa_executed') is False for r in rows)
    assert all(r['reader']['budget']['prompt_tokens_local']+2048<=8192 for r in rows)
    assert all(r['reader']['budget']['used_passages']<=5 for r in rows)
    assert profile['resources']['samples']>0
    review_mode=protocol['configuration']['pack'].get('review_mode','standard')
    reviews=[r['retrieval_trace'].get('evidence',{}).get('improvement_support_semantic_veto',{}) for r in rows]
    pack=protocol['configuration']['pack'];selector=pack.get('selector_mode','current')
    budgeted=[r['retrieval_trace'].get('evidence',{}).get('budgeted_selection',{}) for r in rows]
    if protocol['stage_ablation']=='wo_pack' or protocol.get('mechanism_ablation')=='wo_joint_selection':
        assert all(not d for d in budgeted), 'w/o Pack unexpectedly ran the experimental selector'
    elif selector!='current':
        assert all(d.get('enabled') and d.get('model_requests')==0 and d.get('gold_labels_used') is False
                   and d.get('top200_set_preserved') and d.get('fixed_prefix')==pack['joint_fixed_prefix'] for d in budgeted)
        for row,d,review in zip(rows,budgeted,reviews):
            assert d['selected_doc_ids']==review['final_doc_ids'][:5]
            assert len(d['candidate_domain'])<=len(d['reference_doc_ids'])+80
            assert row['reader']['budget']['used_passages']<=5
    if review_mode=='off' and protocol['stage_ablation']!='wo_pack':
        assert all(d.get('enabled') is False and d.get('review_mode')=='off'
                   and d.get('extra_requests')==0 and not d.get('accepted')
                   and d.get('original_parent_doc_ids')==d.get('final_doc_ids') for d in reviews)
    from bindpack.evaluation.mechanism_audit import audit_mechanism_rows
    mechanism_audit=audit_mechanism_rows(rows,protocol.get('mechanism_ablation','full'),
        protocol['configuration'].get('source_bound_eligibility',True))
    return dict(status='passed',queries=len(rows),physical_requests=len(events),logical_calls=len(logical),
        pack_review_mode=review_mode,review_extra_requests=sum(d.get('extra_requests',0) for d in reviews),
        review_accepted_swaps=sum(bool(d.get('accepted')) for d in reviews),
        online_unknown_usage=profile['cost']['online']['attempts_with_unknown_usage'],
        offline_unknown_usage=profile['cost']['offline_index']['attempts_with_unknown_usage'],
        mechanism_audit=mechanism_audit)


if __name__=='__main__':
    args=parser().parse_args()
    if not args.frozen_worker:freeze_worker(args)
    if args.model_factor_profile:
        # CacheOpenAI captures its process-wide HTTP bound on import. Match the
        # calibrated worker count before importing BindPack rather than silently
        # throttling a 16/32-worker model-factor experiment to the default eight.
        settings_path=Path(args.output)/'experiment_config.json' if args.resume else Path(args.config)
        workers=json.loads(settings_path.read_text())['runtime']['llm_prefetch_workers']
        os.environ['BINDPACK_LLM_MAX_IN_FLIGHT']=str(workers)
        print(f'MODEL FACTOR: HTTP in-flight limit={workers}; Reader workers={args.reader_workers}',flush=True)
    os.environ.setdefault('OPENAI_API_KEY','sk-local-dummy')
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(name)s %(levelname)s %(message)s')
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Interrupted by signal {signum}; completed query checkpoints retained')
    signal.signal(signal.SIGTERM, interrupted)
    run(args)

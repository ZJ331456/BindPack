#!/usr/bin/env python3
"""Six controlled retrieval + Reader@5 runs on four immutable shared indexes."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import fcntl
import gzip
import hashlib
import json
import logging
import os
from pathlib import Path
import pickle
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from types import SimpleNamespace

CODE = Path(__file__).resolve().parents[1]
PROJECT = Path('/root/BindPack')
DATASETS = ('hotpotqa', '2wikimultihopqa', 'musique', 'hover')
ARMS = ('full', 'wo_source_binding', 'wo_joint_selection', 'flat_dependency_support',
        'singleton_search', 'wo_parent_answer_propagation')
LABELS = ('Full', 'w/o Source Binding', 'Reference (S₀)', 'Flat Support',
          'Singleton Search', 'w/o Parent Answer Propagation')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4194304), b''): h.update(block)
    return h.hexdigest()


def save_window(path, value):
    """Only internally generated, trusted checkpoints are unpickled."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with gzip.open(temporary, 'wb', compresslevel=2) as stream:
        pickle.dump(value, stream, protocol=5)
    temporary.replace(path)
    write(path.with_suffix('.sha256.json'), {'sha256': sha(path)})


def load_window(path):
    if sha(path) != read(path.with_suffix('.sha256.json'))['sha256']:
        raise ValueError('Corrupt upstream checkpoint: ' + str(path))
    with gzip.open(path, 'rb') as stream:
        return pickle.load(stream)


def arm_config(base, arm):
    from bindpack import BindPackConfig
    values = asdict(base)
    values.update(mechanism_ablation=arm if arm in ('wo_source_binding', 'wo_joint_selection',
                  'wo_parent_answer_propagation') else 'full',
                  pack_mechanism_ablation=arm if arm in ('flat_dependency_support', 'singleton_search') else 'none')
    return BindPackConfig.from_file(**values)


def selected_indices(data, dataset, limit):
    if not limit: return list(range(len(data)))
    from eval_utils import get_benchmark_hops
    hops = get_benchmark_hops(data, dataset)
    chosen, seen = [], set()
    for i, sample in enumerate(data):
        key = (hops[i], sample.get('label'))
        if key not in seen:
            seen.add(key); chosen.append(i)
        if len(chosen) >= limit: break
    return (chosen + [i for i in range(len(data)) if i not in chosen])[:limit]


def serial_case_order(arms, is_complete):
    """Do not release another case until the preceding case is validated."""
    for arm in arms:
        yield arm
        if not is_complete(arm):
            raise RuntimeError('Previous complete case has not been validated: '+arm)


def case_parallelism(root, window, reader_workers):
    path = Path(root)/'concurrency_profile.json'
    profile = read(path)['cases'] if path.exists() else {}
    result = {}
    for arm in ARMS:
        value = profile.get(arm, dict(llm_workers=window,reader_workers=reader_workers))
        if set(value) != {'llm_workers','reader_workers'} or any(
                type(v) is not int or not 1<=v<=32 for v in value.values()):
            raise ValueError('Invalid concurrency profile: '+arm)
        result[arm]=dict(value)
    return result


def case_timing_totals(windows,arm):
    return dict(full_upstream_seconds=sum(w['full_seconds'] for w in windows),
        pack_replay_seconds=sum(w['pack_seconds'].get(arm,0.) for w in windows),
        parent_reexecution_seconds=sum(w['parent_seconds'] for w in windows) if arm==ARMS[-1] else 0.,
        reader_window_seconds=sum(w['qa_seconds'].get(arm,0.) for w in windows))


def summarize(root):
    results = []
    for ds in DATASETS:
        reports = {}
        for arm in ARMS:
            p = root/'cases'/ds/arm/'report.json'
            if p.exists(): reports[arm] = read(p)
        for arm, report in reports.items():
            metrics = report['metrics']; full = reports.get('full', {}).get('metrics', {})
            results.append(dict(dataset=ds, arm=arm, label=LABELS[ARMS.index(arm)],
                n=report['sample_alignment']['sample_count'], metrics=metrics,
                delta_from_full_pp={key: 100*(v-full[key]) for key, v in metrics.items()
                                    if v is not None and full.get(key) is not None},
                timing=report.get('timing', {})))
    write(root/'summary.json', dict(results=results, qa_executed=True, index_rebuilt=False))
    lines = ['# 新机制消融：检索与 QA', '', '全部使用原共享索引；K=5；差值单位为 pp。', '',
             '|Dataset|Variant|N|R@5|FCR@5|EM / Accuracy|F1 / Macro-F1|DCC@5|',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in results:
        m = row['metrics']; ds = row['dataset']
        values = [m.get(k) for k in ('Recall@5', 'FCR@5',
            'ClassificationAccuracy' if ds == 'hover' else 'ExactMatch',
            'ClassificationMacroF1' if ds == 'hover' else 'F1', 'DCC@5')]
        lines.append('|'+ '|'.join([ds, row['label'], str(row['n'])] +
                                  ['N/A' if v is None else f'{100*v:.2f}' for v in values])+'|')
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return results


def worker(args):
    os.environ['BINDPACK_LLM_MAX_IN_FLIGHT'] = str(max(args.window, args.reader_workers))
    sys.path[:0] = [str(CODE/'src'), str(CODE/'scripts')]
    import numpy as np
    import torch
    from bindpack import BindPack, BindPackConfig
    from bindpack.pipeline import EvidencePipeline
    from bindpack.stages.pack.stage import PackStage
    from bindpack.evaluation.measurement import Recorder, ResourceMonitor, digest, jsonl, usage
    from bindpack.evaluation.measured_reader import prepare
    from bindpack.evaluation.result_export import detailed_result
    from bindpack.evaluation.metric_report import write_metric_report
    from bindpack.utils.misc_utils import QuerySolution
    from transformers import AutoTokenizer
    from eval_utils import (build_docs_from_full_corpus, ensure_gold_docs_in_corpus,
        get_benchmark_hops, describe_benchmark_hops, get_gold_answers, get_gold_docs,
        get_query_text, get_sample_id)
    from utils.redesigned_index import prepare_shared_index, verify_shared_index

    torch.set_num_threads(4)
    root, ds = args.output.resolve(), args.dataset
    meta = root/'metadata'/ds; meta.mkdir(parents=True, exist_ok=True)
    source_data = PROJECT/'datasets'/f'{ds}.json'
    source_corpus = PROJECT/'datasets'/f'{ds}_corpus.json'
    for source, target in ((source_data, meta/'benchmark.json'), (source_corpus, meta/'original_corpus.json')):
        if target.exists():
            if sha(source) != sha(target): raise ValueError('Dataset changed after freeze')
        else: shutil.copy2(source, target)
    data = read(meta/'benchmark.json'); corpus = read(meta/'original_corpus.json')
    gold_all = [sorted(set(group)) for group in get_gold_docs(data, ds)]
    original = build_docs_from_full_corpus(corpus)
    docs = ensure_gold_docs_in_corpus(original, gold_all)
    indices = selected_indices(data, ds, args.limit)
    identity = dict(dataset=ds, data_path=str(meta/'benchmark.json'), data_sha256=sha(meta/'benchmark.json'),
        corpus_docs=len(docs), corpus_docs_sha256=digest(docs), selected_indices=indices,
        sample_ids=[get_sample_id(data[i], ds) for i in indices],
        development_fixture=bool(args.limit), gold_completion_added=len(docs)-len(original),
        oracle_hops=True, hop_provenance=describe_benchmark_hops(data, ds))
    for path, value in ((meta/'sample_manifest.json', identity), (meta/'corpus.json', docs)):
        if path.exists() and read(path) != value: raise ValueError('Frozen inputs changed')
        if not path.exists(): write(path, value)
    settings = read(root/'experiment_config.json'); rt = settings['runtime']
    runtime_overrides = {k: rt[k] for k in (
        'llm_name', 'llm_base_url', 'embedding_model_name', 'embedding_batch_size', 'openie_max_workers',
        'llm_prefetch_workers', 'max_new_tokens', 'openie_strict', 'openie_prompt_version', 'openie_validation_mode')}
    runtime_overrides['llm_prefetch_workers'] = args.window
    cfg = BindPackConfig.from_file(root/'method_config.json', **runtime_overrides,
        save_dir=str(root/'runtime'/ds/'index'), dataset=ds, corpus_len=len(docs), stage_ablation='full',
        mechanism_ablation='full', pack_mechanism_ablation='none', retrieval_top_k=200, qa_top_k=5, max_qa_steps=1)
    parallelism = case_parallelism(root,args.window,args.reader_workers)
    case_configs = {}
    for arm in ARMS:
        values=asdict(arm_config(cfg,arm))
        values['llm_prefetch_workers']=parallelism[arm]['llm_workers']
        case_configs[arm]=BindPackConfig.from_file(**values)
    cfg=case_configs['full']
    if cfg.pack.budget != 5 or cfg.pack.review_mode != 'off': raise ValueError('Expected fixed K=5 and review=off')
    for arm in ARMS:
        case = root/'cases'/ds/arm; case.mkdir(parents=True, exist_ok=True)
        protocol = dict(dataset=identity, configuration=asdict(case_configs[arm]), arms=list(ARMS),
            variant_name=LABELS[ARMS.index(arm)],
            eval_mode='rag_qa', reader_top_k=5, context_window=8192, output_tokens=2048,
            upstream_policy='four Pack controls share exact Full Route/Plan/Pack inputs and S0; parent control shares Route and DAG, reexecutes Plan without parent answers',
            index_rebuilt=False, query_window=args.window, reader_workers=parallelism[arm]['reader_workers'],
            execution_schedule='complete_cases_serial_v1',
            http_max_in_flight=max(args.window,args.reader_workers))
        if (case/'protocol.json').exists() and read(case/'protocol.json') != protocol:
            raise ValueError('Frozen case protocol changed')
        write(case/'protocol.json', protocol)
    receipt_path = meta/'index_receipt.json'
    prepare_shared_index(PROJECT/'outputs/shared_indexes'/ds, Path(cfg.save_dir), receipt_path, ds, docs, cfg)
    rows = {arm: [] for arm in ARMS}
    for arm in ARMS:
        for event in jsonl(root/'cases'/ds/arm/'progress.jsonl'):
            row = event['row']; position = len(rows[arm])
            if (digest(row) != event['sha256'] or row['query_index'] != indices[position] or
                    row['sample_id'] != identity['sample_ids'][position]): raise ValueError('Invalid query checkpoint')
            rows[arm].append(row)
    if all(len(rows[a]) == len(indices) and (root/'cases'/ds/a/'completed.ok').exists() for a in ARMS):
        for arm in ARMS:
            case = root/'cases'/ds/arm; marker = read(case/'completed.ok')
            if sha(case/'result.json') != marker['result_sha256'] or sha(case/'report.json') != marker['report_sha256']:
                raise ValueError('Completed result/report changed: '+arm)
        verify_shared_index(receipt_path); print(f'ALREADY COMPLETE {ds}', flush=True); return
    recorder = Recorder(root/'measurements'/ds)
    tokenizer = AutoTokenizer.from_pretrained(rt['llm_model_path'], local_files_only=True)
    from bindpack.llm.openai_gpt import CacheOpenAI
    recorder.install_llm_factory(CacheOpenAI, tokenizer)
    monitor = ResourceMonitor(recorder)
    stop = threading.Event()
    for number in (signal.SIGTERM, signal.SIGINT): signal.signal(number, lambda *_: stop.set())
    started = time.perf_counter(); session_status = 'failed'
    current = {'phase': 'model_loading', 'window_start': 0}
    def heartbeat():
        while not stop.wait(30):
            state = dict(status='running', pid=os.getpid(), dataset=ds, **current,
                completed={a: len(r) for a, r in rows.items()}, elapsed_seconds=time.perf_counter()-started)
            write(root/'worker_status.json', state)
            print(f'HEARTBEAT {ds} {current} completed={state["completed"]}', flush=True)
    thread = threading.Thread(target=heartbeat, daemon=True); thread.start()
    windows = root/'upstream'/ds; timings = jsonl(root/'measurements'/ds/'windows.jsonl')
    def finalize_case(arm):
        resources = monitor.summary()
        events = jsonl(root/'measurements'/ds/'llm_calls.jsonl')
        by_phase = {phase: usage([e for e in events if e['phase'] == phase])
                    for phase in sorted({e['phase'] for e in events})}
        case = root/'cases'/ds/arm
        assert len(rows[arm]) == len(indices)
        assert all(r['answer'] is not None and r['reader']['budget']['used_passages'] <= 5 and
            r['reader']['budget']['prompt_tokens_local']+2048 <= 8192 for r in rows[arm])
        result = dict(dataset=ds, method='bindpack', stage_ablation='full', mechanism_ablation=arm,
            variant_name=LABELS[ARMS.index(arm)],
            eval_mode='rag_qa', qa_executed=True, data_path=str(meta/'benchmark.json'),
            corpus_path=str(meta/'corpus.json'), selected_indices=indices, sample_size_effective=len(indices),
            runtime_config=asdict(case_configs[arm]), results=rows[arm],
            hop_provenance=identity['hop_provenance'], retrieval_metrics={}, qa_metrics={})
        write(case/'result.json', result)
        write(case/'validated.ok', dict(result_sha256=sha(case/'result.json'),
            protocol_sha256=digest(read(case/'protocol.json'))))
        report = write_metric_report(case/'result.json', manifest_path=meta/'sample_manifest.json', ks=[5,10,20,200])
        report.update(timing=dict(**case_timing_totals(timings,arm),
            all_execution_sessions_seconds=sum(s['wall_seconds'] for s in
                jsonl(root/'measurements'/ds/'execution_sessions.jsonl'))+time.perf_counter()-started,
            scope='Full upstream shared once across first five arms; parent adds Plan execution; all arms run QA'),
            resources=resources, index_rebuilt=False, qa_executed=True,
            variant_name=LABELS[ARMS.index(arm)],
            cost=dict(reader=by_phase.get('reader:'+arm, usage([])),
                shared_full_upstream=by_phase.get('full_upstream', usage([])),
                additional_parent_plan=by_phase.get('parent_plan', usage([])) if arm == ARMS[-1] else usage([]),
                attribution='Full upstream incurred once; shared by first five arms and Route/DAG of parent control; physical attempts include retries, native cache hits have no physical token charge'))
        write(case/'report.json', report)
        write(case/'completed.ok', dict(queries=len(indices), result_sha256=sha(case/'result.json'),
            report_sha256=sha(case/'report.json')))
        recorder.append('case_events.jsonl', dict(event='complete', arm=arm,
            variant_name=LABELS[ARMS.index(arm)], queries=len(rows[arm]), unix=time.time()))
        summarize(root)
        print(f'CASE COMPLETE {ds}/{LABELS[ARMS.index(arm)]}: {len(rows[arm])}/{len(indices)}',flush=True)

    try:
        with monitor:
            rag = BindPack(config=cfg)
            def forbidden(*unused, **kw): raise RuntimeError('Index construction is forbidden in this experiment')
            rag.index = forbidden
            recorder.install_embedding(rag.embedding_model)
            for store in (rag.chunk_embedding_store, rag.entity_embedding_store, rag.fact_embedding_store):
                store.insert_strings = forbidden
            rag.save_igraph = forbidden
            node_names = set(rag.graph.vs['name'])
            needed = set(rag.chunk_embedding_store.get_all_ids()) | set(rag.entity_embedding_store.get_all_ids())
            if not needed <= node_names: raise ValueError('Shared graph is missing passage/entity nodes')
            rag.prepare_retrieval_objects(); rag._build_index_side_artifacts()
            if set(rag.chunk_embedding_store.get_all_texts()) != set(docs): raise ValueError('Index corpus changed')
            hops = get_benchmark_hops(data, ds)
            for arm in serial_case_order(ARMS, lambda a: (root/'cases'/ds/a/'completed.ok').exists()):
                case = root/'cases'/ds/arm
                if (case/'completed.ok').exists():
                    marker = read(case/'completed.ok')
                    assert len(rows[arm]) == len(indices)
                    assert sha(case/'result.json') == marker['result_sha256']
                    assert sha(case/'report.json') == marker['report_sha256']
                    continue
                current.update(arm=arm, phase='case_start', window_start=0, **parallelism[arm])
                recorder.append('case_events.jsonl',dict(event='start',arm=arm,
                    variant_name=LABELS[ARMS.index(arm)],unix=time.time()))
                print(f'CASE START {ds}/{LABELS[ARMS.index(arm)]}: one complete case at a time',flush=True)
                for lo in range(0,len(indices),args.window):
                    hi=min(lo+args.window,len(indices));current.update(window_start=lo)
                    timing_path=windows/f'{lo:04d}_{hi:04d}_{arm}_timing.json'
                    phase_path=windows/f'{lo:04d}_{hi:04d}_reader_timing.json'
                    window_path=windows/f'{lo:04d}_{hi:04d}.pkl.gz'
                    if len(rows[arm])>=hi:
                        if not any(w.get('arm')==arm and w['start']==lo for w in timings):
                            if timing_path.exists(): timing=read(timing_path)
                            else:
                                first=rows[arm][lo]['timing']
                                timing=dict(arm=arm,start=lo,end=hi,queries=hi-lo,
                                    full_seconds=first['full_upstream_window_seconds'] if arm=='full' else 0.,
                                    parent_seconds=first['parent_window_seconds'] if arm==ARMS[-1] else 0.,
                                    pack_seconds={a:first['pack_replay_window_seconds'] if a==arm else 0. for a in ARMS},
                                    qa_seconds={arm:read(phase_path)[arm]})
                                write(timing_path,timing)
                            recorder.append('windows.jsonl',timing);timings.append(timing)
                        continue
                    if stop.is_set():raise KeyboardInterrupt('Durable case checkpoints retained')
                    selected=indices[lo:hi];questions=[get_query_text(data[i],ds) for i in selected]
                    if len(set(questions))!=len(questions):raise ValueError('Duplicate query text within one window')
                    if arm!='full' and not window_path.exists():raise ValueError('Missing completed Full upstream')
                    if window_path.exists(): bundle = load_window(window_path)
                    else:
                        current['phase'] = 'full_upstream'
                        full_runtime = EvidencePipeline(rag); rag.evidence_runtime = full_runtime; rag.config = cfg
                        preplan, pack_inputs, references = {}, {}, {}
                        original_process, original_finalize = full_runtime.process_window, full_runtime.pack.finalize
                        def capture_process(states, executor=None):
                            for s in states: preplan[s['query']] = deepcopy(s)
                            return original_process(states, executor)
                        def capture_finalize(query, ids, scores, context, state):
                            pack_inputs[query] = dict(query=query, ids=deepcopy(ids), scores=deepcopy(scores),
                                                     context=deepcopy(context), state=deepcopy(state))
                            full_runtime._capture_pack_reference = lambda reference: references.__setitem__(query, deepcopy(reference))
                            return original_finalize(query, ids, scores, context, state)
                        full_runtime.process_window = capture_process; full_runtime.pack.finalize = capture_finalize
                        rag.set_query_hop_overrides([hops[i] for i in selected], source='benchmark')
                        tick = time.perf_counter()
                        with recorder.section('full_upstream', f'{lo}:{hi}'):
                            full = rag.retrieve(questions, num_to_retrieve=200)
                        full_seconds = time.perf_counter()-tick
                        bundle = dict(indices=selected, questions=questions, preplan=preplan, pack_inputs=pack_inputs,
                            references=references, full=[s.to_dict(top_k=len(s.docs)) for s in full],
                            full_seconds=full_seconds)
                        save_window(window_path, bundle)
                    if bundle['indices']!=selected or bundle['questions']!=questions:raise ValueError('Window changed')
                    solutions={'full':[QuerySolution(**raw) for raw in bundle['full']]}
                    pack_seconds={a:0. for a in ARMS};audits=[];parent={'seconds':0.}
                    if arm=='full':
                        if args.limit:
                            parity_runtime = SimpleNamespace(cfg=cfg, rag=rag, _document=rag.evidence_runtime._document,
                                _trace=lambda: {}, _joint_tokenizer=tokenizer)
                            for offset, query in enumerate(questions):
                                payload = bundle['pack_inputs'][query]
                                parity_runtime._pack_reference = deepcopy(bundle['references'][query])
                                ids, scores, trace = PackStage(parity_runtime).finalize(query, deepcopy(payload['ids']),
                                    deepcopy(payload['scores']), deepcopy(payload['context']), deepcopy(payload['state']))
                                assert [parity_runtime._document(int(d)) for d in ids[:200]] == solutions['full'][offset].docs
                                original_joint = solutions['full'][offset].retrieval_trace['evidence']['budgeted_selection']
                                assert trace['budgeted_selection']['reference_doc_ids'] == original_joint['reference_doc_ids']
                                assert abs(trace['budgeted_selection']['objective']-original_joint['objective']) < 1e-12
                            audits.append(dict(arm='full', exact_replay_parity=True, queries=len(questions)))
                    elif arm in ARMS[1:5]:
                        current['phase']='pack_replay:'+arm
                        solutions[arm] = []
                        runtime = SimpleNamespace(cfg=case_configs[arm], rag=rag,
                            _document=rag.evidence_runtime._document, _trace=lambda: {},
                            _joint_tokenizer=tokenizer)
                        for offset, query in enumerate(questions):
                            payload = bundle['pack_inputs'][query]
                            runtime._pack_reference = deepcopy(bundle['references'][query])
                            tick = time.perf_counter()
                            with recorder.section('pack:'+arm, identity['sample_ids'][lo+offset]):
                                ids, scores, trace = PackStage(runtime).finalize(query, deepcopy(payload['ids']),
                                    deepcopy(payload['scores']), deepcopy(payload['context']), deepcopy(payload['state']))
                            pack_seconds[arm] += time.perf_counter()-tick
                            full_trace = solutions['full'][offset].retrieval_trace['evidence']
                            joint = full_trace.get('budgeted_selection', {})
                            reference = bundle['references'][query][0]
                            if arm == 'wo_joint_selection':
                                assert list(ids[:5]) == list(reference[:5]) and not trace.get('budgeted_selection')
                            else:
                                diag = trace.get('budgeted_selection', {})
                                assert diag['reference_doc_ids'] == list(reference[:5])
                                if arm in ('flat_dependency_support', 'singleton_search'):
                                    assert diag['candidate_domain'] == joint['candidate_domain']
                                    assert diag['full_package_bank'] == joint['full_package_bank']
                                    assert diag['soft_weight_effective'] == joint['soft_weight_effective']
                                if arm == 'singleton_search': assert diag['package_action_count'] == 0
                                if arm == 'flat_dependency_support': assert diag['closure_reward_mode'] == 'flat'
                            assert set(map(int, ids[:200])) == set(map(int, reference[:200]))
                            doc_ids = [int(d) for d in ids[:200]]
                            solutions[arm].append(QuerySolution(question=query,
                                docs=[runtime._document(d) for d in doc_ids], doc_scores=np.asarray(scores)[:200],
                                retrieval_trace=dict(evidence=trace, shared_full_upstream=True,
                                    selected_doc_ids=doc_ids[:5], upstream_checkpoint_sha256=sha(window_path))))
                            audits.append(dict(arm=arm, query=query, shared_full_upstream=True,
                                               reference_top5=list(map(int, reference[:5]))))
                    else:
                        parent_path = windows/f'{lo:04d}_{hi:04d}_parent.pkl.gz'
                        if parent_path.exists(): parent = load_window(parent_path)
                        else:
                            current['phase'] = 'parent_plan'
                            parent_cfg = case_configs[ARMS[-1]]
                            # EvidencePipeline takes rag.config at construction.
                            rag.config = parent_cfg; parent_runtime = EvidencePipeline(rag); rag.evidence_runtime = parent_runtime
                            plans = {q: deepcopy(bundle['pack_inputs'][q]['state']['_evidence_plan']) for q in questions}
                            def frozen_plan(question, hops):
                                parent_runtime._plan_diagnostics[question] = dict(attempts=0, shared_full_dag=True)
                                return deepcopy(plans[question]), None
                            parent_runtime.plan.planner.plan = frozen_plan
                            original_parent_finalize = parent_runtime.pack.finalize
                            def parent_finalize(query, ids, scores, context, state):
                                payload = bundle['pack_inputs'][query]
                                assert np.array_equal(ids, payload['ids'])
                                assert np.array_equal(scores, payload['scores'])
                                static_routes = lambda trace: [r for r in trace['routes'] if not r['goal'].startswith('dag:')]
                                assert static_routes(state['evidence_trace']) == static_routes(payload['state']['evidence_trace'])
                                return original_parent_finalize(query, ids, scores, context, state)
                            parent_runtime.pack.finalize = parent_finalize
                            original_iter = rag._iter_retrieval_states
                            def parent_states(queries):
                                states = [deepcopy(bundle['preplan'][q]) for q in queries]
                                with ThreadPoolExecutor(max_workers=cfg.llm_prefetch_workers) as pool:
                                    parent_runtime.process_window(states, pool)
                                yield from states
                            rag._iter_retrieval_states = parent_states
                            rag.set_query_hop_overrides([hops[i] for i in selected], source='benchmark')
                            tick = time.perf_counter()
                            try:
                                with recorder.section('parent_plan', f'{lo}:{hi}'):
                                    parent_solutions = rag.retrieve(questions, num_to_retrieve=200)
                            finally:
                                rag._iter_retrieval_states = original_iter; rag.config = cfg
                            for q, s in zip(questions, parent_solutions):
                                assert s.retrieval_trace['evidence']['plan'] == plans[q]
                            parent = dict(solutions=[s.to_dict(top_k=len(s.docs)) for s in parent_solutions],
                                          seconds=time.perf_counter()-tick, shared_full_route=True, shared_full_dag=True)
                            save_window(parent_path, parent)
                        solutions[ARMS[-1]] = [QuerySolution(**raw) for raw in parent['solutions']]
                    qa_seconds=read(phase_path) if phase_path.exists() else {}
                    current['phase'] = 'reader:'+arm
                    case = root/'cases'/ds/arm
                    def answer(offset):
                        position = lo+offset; index = indices[position]; solution = solutions[arm][offset]
                        qid = identity['sample_ids'][position]
                        with recorder.thread_section('reader:'+arm, qid):
                            tick = time.perf_counter()
                            reader, budget = prepare(rag, solution, tokenizer, ds)
                            answered, raw, metadata = rag.qa([reader])
                            solution.answer = answered[0].answer
                            seconds = time.perf_counter()-tick
                        solution.gold_answers = get_gold_answers([data[index]], ds)[0]
                        solution.gold_docs = gold_all[index]
                        row = detailed_result(solution, data[index], index, gold_all[index], hops[index], 10, 200)
                        row.update(ranking_complete=True, actual_ranking_length=len(solution.docs),
                            reader=dict(raw_response=raw[0], metadata=metadata[0], budget=budget),
                            timing=dict(reader_seconds=seconds, full_upstream_window_seconds=bundle['full_seconds'],
                                pack_replay_window_seconds=pack_seconds[arm], parent_window_seconds=parent['seconds'],
                                window_size=hi-lo, timing_scope='window throughput; shared upstream charged once'))
                        return position, json.loads(json.dumps(row, ensure_ascii=False, allow_nan=False))
                    missing = list(range(max(lo, len(rows[arm]))-lo, hi-lo))
                    ready = {}; tick = time.perf_counter(); prior_reader_seconds = qa_seconds.get(arm, 0.)
                    with ThreadPoolExecutor(max_workers=parallelism[arm]['reader_workers']) as pool:
                        futures = [pool.submit(answer, offset) for offset in missing]
                        for future in as_completed(futures):
                            position, row = future.result(); ready[position] = row
                            qa_seconds[arm] = prior_reader_seconds + time.perf_counter()-tick
                            write(phase_path, qa_seconds)
                            while len(rows[arm]) in ready:
                                row = ready.pop(len(rows[arm])); event = dict(sha256=digest(row), row=row)
                                with (case/'progress.jsonl').open('a') as stream:
                                    stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False)+'\n')
                                    stream.flush(); os.fsync(stream.fileno())
                                rows[arm].append(row)
                    qa_seconds[arm] = prior_reader_seconds + (time.perf_counter()-tick if missing else 0.)
                    write(phase_path, qa_seconds)
                    print(f'PROGRESS {ds}/{LABELS[ARMS.index(arm)]} {len(rows[arm])}/{len(indices)} QA_window={qa_seconds[arm]:.2f}s', flush=True)
                    timing=dict(arm=arm,start=lo,end=hi,queries=hi-lo,
                        full_seconds=bundle['full_seconds'] if arm=='full' else 0.,
                        parent_seconds=parent['seconds'] if arm==ARMS[-1] else 0.,
                        pack_seconds=pack_seconds,qa_seconds={arm:qa_seconds[arm]})
                    write(timing_path,timing)
                    recorder.append('windows.jsonl',timing);timings.append(timing)
                    write(windows/f'{lo:04d}_{hi:04d}_{arm}_audit.json',dict(status='passed',arm=arm,checks=audits,
                        parent_shared_route=arm==ARMS[-1],parent_shared_dag=arm==ARMS[-1]))
                    print('WINDOW COMPLETE '+ds+'/'+LABELS[ARMS.index(arm)]+' '+json.dumps(timing),flush=True)
                    write(meta/'live_progress.json',dict(dataset=ds,active_arm=arm,execution_schedule='complete_cases_serial_v1',
                        completed={a:len(rs) for a,rs in rows.items()},
                        retrieval={a:{'Recall@5':sum(r['retrieval_metrics']['Recall@5'] for r in rs)/len(rs),
                            'FCR@5':sum(r['all_gold_in_top5'] for r in rs)/len(rs)} for a,rs in rows.items() if rs},
                        full_upstream_seconds=sum(w['full_seconds'] for w in timings)))
                if stop.is_set():raise KeyboardInterrupt('Stopped before starting another complete case')
                finalize_case(arm)
        verify_shared_index(receipt_path)
        events = jsonl(root/'measurements'/ds/'llm_calls.jsonl')
        write(meta/'completed.ok', dict(queries=len(indices), arms=list(ARMS), status='passed',
            new_index_build_calls=0, physical_llm_usage=usage(events),
            physical_usage_by_phase={phase: usage([e for e in events if e['phase'] == phase])
                                     for phase in sorted({e['phase'] for e in events})}))
        session_status = 'complete'
        print(f'DATASET PASS {ds}: {len(indices)} queries × 6 retrieval+QA', flush=True)
    finally:
        recorder.append('execution_sessions.jsonl', dict(status=session_status,
            wall_seconds=time.perf_counter()-started, completed={a: len(r) for a, r in rows.items()}))
        write(root/'worker_status.json', dict(status=session_status, pid=os.getpid(), dataset=ds,
            completed={a: len(r) for a, r in rows.items()}, elapsed_seconds=time.perf_counter()-started))
        stop.set(); thread.join(timeout=2)
        for obj, name, original in reversed(recorder.restorations): setattr(obj, name, original)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=PROJECT/'outputs/ablations_redesigned_20261108')
    p.add_argument('--dataset', choices=DATASETS)
    p.add_argument('--window', type=int)
    p.add_argument('--reader-workers', type=int)
    p.add_argument('--concurrency-profile',type=Path,help='Frozen per-case calibrated LLM/Reader worker counts')
    p.add_argument('--limit', type=int, default=0, help='Development check only; 0 runs all 1000 queries')
    p.add_argument('--worker', action='store_true')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    saved=read(args.output/'protocol.json') if args.resume and (args.output/'protocol.json').exists() else {}
    if args.window is None:args.window=saved.get('window',8)
    if args.reader_workers is None:args.reader_workers=saved.get('reader_workers',8)
    active_datasets=[args.dataset] if args.dataset else list(DATASETS)
    if not 1 <= args.window <= 32 or not 1 <= args.reader_workers <= 32: p.error('Invalid concurrency')
    if args.limit < 0: p.error('limit must be nonnegative')
    if args.worker: worker(args); return
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=True)
    lock = (PROJECT/'outputs/.redesigned_ablations.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    measured_lock = (PROJECT/'outputs/.bindpack_measured.lock').open('a')
    fcntl.flock(measured_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not (root/'source_snapshot').exists():
        if args.concurrency_profile:
            target=root/'concurrency_profile.json'
            if args.concurrency_profile.resolve()!=target.resolve():shutil.copy2(args.concurrency_profile,target)
        for folder in ('src', 'scripts', 'configs'):
            shutil.copytree(PROJECT/folder, root/'source_snapshot'/folder,
                ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        shutil.copy2(PROJECT/'configs/redesigned_ablations_qwen3.json', root/'experiment_config.json')
        method = Path(read(root/'experiment_config.json')['method_config'])
        if not method.is_absolute(): method = PROJECT/method
        shutil.copy2(method, root/'method_config.json')
        source_hashes = {str(path.relative_to(root/'source_snapshot')): sha(path)
            for path in (root/'source_snapshot').rglob('*') if path.is_file()}
        write(root/'protocol.json', dict(datasets=active_datasets, arms=list(ARMS), labels=list(LABELS),
            source_sha256=source_hashes, qa_executed=True, index_rebuilt=False, limit=args.limit,
            execution_schedule='complete_cases_serial_v1',
            concurrency_profile_sha256=sha(root/'concurrency_profile.json') if (root/'concurrency_profile.json').exists() else None,
            window=args.window, reader_workers=args.reader_workers))
    else:
        if not args.resume: raise ValueError('Output exists; use --resume')
        protocol = read(root/'protocol.json')
        if protocol['datasets']!=active_datasets:raise ValueError('Frozen dataset selection changed')
        if protocol.get('execution_schedule') != 'complete_cases_serial_v1':
            raise ValueError('Cannot resume an interleaved run as a serial complete-case experiment')
        active_profile=sha(root/'concurrency_profile.json') if (root/'concurrency_profile.json').exists() else None
        if protocol.get('concurrency_profile_sha256')!=active_profile:raise ValueError('Frozen concurrency profile changed')
        if (protocol['limit'], protocol['window'], protocol['reader_workers']) != (args.limit,args.window,args.reader_workers):
            raise ValueError('Resume settings differ')
        for relative, value in protocol['source_sha256'].items():
            if sha(root/'source_snapshot'/relative) != value: raise ValueError('Frozen source changed')
    (root/'run.pid').write_text(str(os.getpid())+'\n')
    state = dict(status='running', pid=os.getpid(), started_unix=time.time(), datasets=active_datasets, arms=list(ARMS),
        execution_schedule='complete_cases_serial_v1')
    child = None
    def interrupted(*unused):
        if child is not None and child.poll() is None: child.send_signal(signal.SIGTERM)
        raise KeyboardInterrupt('Stopped; resume the same frozen output')
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    try:
        for ds in active_datasets:
            state.update(dataset=ds); write(root/'status.json', state)
            command = [sys.executable, '-B', '-u', str(root/'source_snapshot/scripts/run_redesigned_ablations.py'),
                '--worker', '--output', str(root), '--dataset', ds, '--limit', str(args.limit),
                '--window', str(args.window), '--reader-workers', str(args.reader_workers)]
            child = subprocess.Popen(command, cwd=PROJECT)
            if child.wait() != 0: raise RuntimeError('Dataset worker failed: '+ds)
            child = None; summarize(root)
        state.update(status='complete', finished_unix=time.time())
        write(root/'completed.ok', state)
    except BaseException as error:
        state.update(status='stopped' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        raise
    finally:
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM); child.wait()
        write(root/'status.json', state); summarize(root)


if __name__ == '__main__':
    os.environ.setdefault('OPENAI_API_KEY', 'EMPTY')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('BINDPACK_LLM_MAX_IN_FLIGHT', '8')
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(name)s %(levelname)s %(message)s')
    main()

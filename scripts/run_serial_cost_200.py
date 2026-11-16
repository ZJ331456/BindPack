#!/usr/bin/env python3
"""Stratified 4 x 50 native BindPack latency/call measurements, strictly serial."""
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

PROJECT = Path('/root/BindPack')
ADOPTED = PROJECT / 'outputs/ablations_redesigned_20261108'
DATASETS = ('hotpotqa', '2wikimultihopqa', 'musique', 'hover')
QUOTAS = {'hotpotqa': {2: 50}, '2wikimultihopqa': {2: 38, 4: 12},
          'musique': {2: 26, 3: 16, 4: 8}, 'hover': {3: 28, 4: 22}}
SEED = 'BindPack-serial-cost-50-per-dataset-20261109-v1'


def stamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4194304), b''):
            value.update(block)
    return value.hexdigest()


def jsonl(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []


def stratum(sample, dataset):
    if dataset in ('hotpotqa', '2wikimultihopqa'):
        return len({fact[0] for fact in sample['supporting_facts']})
    if dataset == 'musique':
        return len(sample['question_decomposition'])
    return sample['num_hops']


def choose_samples(data, dataset, sample_ids):
    if len(data) != len(set(sample_ids)):
        raise ValueError('Sample IDs must be unique')
    groups = defaultdict(list)
    for index, sample in enumerate(data):
        groups[stratum(sample, dataset)].append(index)

    def score(index, purpose):
        text = f'{SEED}|{dataset}|{purpose}|{sample_ids[index]}'
        return hashlib.sha256(text.encode()).digest()

    selected = []
    for group, quota in QUOTAS[dataset].items():
        if len(groups[group]) < quota:
            raise ValueError(f'Not enough {dataset} examples in stratum {group}')
        selected.extend(sorted(groups[group], key=lambda i: score(i, 'choose'))[:quota])
    selected.sort(key=lambda i: score(i, 'order'))
    if len(selected) != 50 or dict(Counter(stratum(data[i], dataset) for i in selected)) != QUOTAS[dataset]:
        raise RuntimeError('Sample quotas differ')
    return selected


def prepare(root):
    protocol_path = root / 'protocol.json'
    if protocol_path.exists():
        protocol = read(protocol_path)
        for relative, expected in protocol['source_sha256'].items():
            if sha(root / 'source_snapshot' / relative) != expected:
                raise RuntimeError('Frozen source changed: ' + relative)
        for relative, expected in protocol['input_sha256'].items():
            if sha(root / relative) != expected:
                raise RuntimeError('Frozen input changed: ' + relative)
        return protocol
    snapshot = root / 'source_snapshot'
    # Use the exact native source of the previous four-dataset Full experiment.
    for name in ('src', 'scripts', 'configs'):
        shutil.copytree(ADOPTED / 'source_snapshot' / name, snapshot / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(Path(__file__), snapshot / 'scripts/run_serial_cost_200.py')
    sys.path.insert(0, str(snapshot / 'scripts'))
    from eval_utils import get_sample_id, get_benchmark_hops
    inputs = {}
    for dataset in DATASETS:
        for suffix in ('', '_corpus'):
            name = dataset + suffix + '.json'
            src = PROJECT / 'datasets' / name
            original = ADOPTED / 'metadata' / dataset / ('benchmark.json' if not suffix else 'original_corpus.json')
            if sha(src) != sha(original):
                raise RuntimeError('Dataset differs from the adopted Full: ' + name)
            dst = root / 'frozen_datasets' / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            inputs[str(dst.relative_to(root))] = sha(dst)
        data = read(root / 'frozen_datasets' / (dataset + '.json'))
        ids = [get_sample_id(sample, dataset) for sample in data]
        indices = choose_samples(data, dataset, ids)
        native_hops = get_benchmark_hops(data, dataset)
        selection = dict(dataset=dataset, seed=SEED, n=50, population=len(data),
                         indices=indices, sample_ids=[ids[i] for i in indices],
                         requested_distribution=QUOTAS[dataset],
                         actual_distribution=dict(Counter(stratum(data[i], dataset) for i in indices)),
                         native_hints=[native_hops[i] for i in indices],
                         selection='SHA256 random selection within fixed annotation strata and SHA256 order; no outputs or times used')
        write(root / 'samples' / (dataset + '.json'), selection)
        write(root / 'indices' / (dataset + '.json'), indices)
        for folder in ('samples', 'indices'):
            path = root / folder / (dataset + '.json')
            inputs[str(path.relative_to(root))] = sha(path)
        original = read(ADOPTED / 'cases' / dataset / 'full/protocol.json')['configuration']
        original.update(save_dir=str(root / 'runs' / dataset / 'index'),
                        llm_prefetch_workers=1, openie_max_workers=1, embedding_batch_size=4)
        write(root / 'configs' / (dataset + '.json'), original)
        inputs['configs/' + dataset + '.json'] = sha(root / 'configs' / (dataset + '.json'))
    protocol = dict(created_at=stamp(), method='BindPack Full', llm='qwen3-8b',
                    embedding='/root/models/Qwen3-Embedding-8B', questions=200,
                    queries_per_dataset=50, quotas=QUOTAS, seed=SEED, reader_top_k=5,
                    query_concurrency=1, llm_workers=1, http_max_in_flight=1, reader_workers=1,
                    native_source_origin=str(ADOPTED / 'source_snapshot'),
                    wall_time='one continuous perf_counter_ns interval: native retrieval/Route/Plan/Pack, Reader context preparation, native QA and answer parsing; resident model/index; includes native caching and retry cost; excludes initial model/index loading, evaluation and durable measurement/checkpoint writes',
                    calls='all logical online infer calls including native within-run hits; all physical SDK attempts including retries also recorded',
                    cache_policy='fresh independent application and query caches per dataset; no historical LLM or answer reuse; native within-run caching retained; vLLM server/prefix caches resident across datasets',
                    hop_policy='native benchmark annotation hints unchanged from adopted Full; sampling source/node/hop strata follow user counts',
                    first_query='included; model and full corpus index prepared before timed queries',
                    execution='datasets sequential; each query completes retrieval and QA before the next query; all client requests serial; server max-num-seqs1',
                    model_service=read(root / 'service/launch.json'),
                    input_sha256=inputs,
                    source_sha256={str(p.relative_to(snapshot)): sha(p) for p in snapshot.rglob('*') if p.is_file()})
    write(protocol_path, protocol)
    return protocol


def worker(root, dataset):
    snapshot = root / 'source_snapshot'
    sys.path.insert(0, str(snapshot / 'src'))
    sys.path.insert(0, str(snapshot / 'scripts'))
    import torch
    from bindpack import BindPack, BindPackConfig
    from bindpack.evaluation.measurement import Recorder, digest
    from bindpack.evaluation.measured_reader import prepare as prepare_reader
    from bindpack.evaluation.result_export import detailed_result
    from bindpack.llm.openai_gpt import CacheOpenAI
    from transformers import AutoTokenizer
    from eval_utils import (build_docs_from_full_corpus, ensure_gold_docs_in_corpus,
                            get_gold_docs, get_gold_answers, get_benchmark_hops,
                            get_query_text, get_sample_id)
    from utils.redesigned_index import prepare_shared_index, verify_shared_index

    class BufferedRecorder(Recorder):
        # Avoid charging fsync for observational traces to user-facing latency.
        def __init__(self, directory):
            super().__init__(directory)
            self.pending = []

        def append(self, name, event):
            with self.lock:
                self.pending.append((name, event))

        def flush(self):
            with self.lock:
                pending, self.pending = self.pending, []
            for name, event in pending:
                super().append(name, event)

    torch.set_num_threads(4)
    stop_requested = [False]
    def request_stop(*unused):
        stop_requested[0] = True
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    case = root / 'runs' / dataset
    case.mkdir(parents=True, exist_ok=True)
    selection = read(root / 'samples' / (dataset + '.json'))
    config = BindPackConfig.from_file(**read(root / 'configs' / (dataset + '.json')))
    data = read(root / 'frozen_datasets' / (dataset + '.json'))
    corpus = read(root / 'frozen_datasets' / (dataset + '_corpus.json'))
    gold_all = [sorted(set(group)) for group in get_gold_docs(data, dataset)]
    docs = ensure_gold_docs_in_corpus(build_docs_from_full_corpus(corpus), gold_all)
    preparation_start = time.perf_counter()
    receipt_path = case / 'index_receipt.json'
    prepare_shared_index(PROJECT / 'outputs/shared_indexes' / dataset, Path(config.save_dir),
                         receipt_path, dataset, docs, config)
    recorder = BufferedRecorder(case)
    tokenizer = AutoTokenizer.from_pretrained('/root/models/Qwen3-8B', local_files_only=True)
    recorder.install_llm_factory(CacheOpenAI, tokenizer)
    rag = None
    rows = [event['row'] for event in jsonl(case / 'progress.jsonl')]
    for i, row in enumerate(rows):
        if row['sample_id'] != selection['sample_ids'][i]:
            raise RuntimeError('Checkpoint query order changed')
    try:
        rag = BindPack(config=config)
        if set(rag.chunk_embedding_store.get_all_texts()) != set(docs):
            raise RuntimeError('Full corpus/index mismatch')
        rag.prepare_retrieval_objects()
        rag._build_index_side_artifacts()
        def deny_index(*args, **kwargs):
            raise RuntimeError('Index building is forbidden for this latency benchmark')
        rag.index = deny_index
        preparation_seconds = time.perf_counter() - preparation_start
        write(case / 'runtime_config.json', asdict(config))
        write(case / 'preparation.json', dict(seconds=preparation_seconds, corpus_passages=len(docs),
                                             index_build_calls=0, excluded_from_query_latency=True))
        print(stamp(), dataset, 'INDEX/MODEL READY', len(docs), 'passages', flush=True)
        native_hops = get_benchmark_hops(data, dataset)
        for position in range(len(rows), 50):
            if stop_requested[0]:
                raise KeyboardInterrupt('Stopped at a completed query boundary')
            index = selection['indices'][position]
            sample = data[index]
            qid = get_sample_id(sample, dataset)
            torch.cuda.reset_peak_memory_stats()
            start_unix = time.time()
            start = time.perf_counter_ns()
            with recorder.section('online_retrieval', qid):
                rag.set_query_hop_overrides([native_hops[index]], source='benchmark')
                solution = rag.retrieve([get_query_text(sample, dataset)], num_to_retrieve=200)[0]
            retrieval_done = time.perf_counter_ns()
            with recorder.section('online_reader', qid):
                reader_solution, budget = prepare_reader(rag, solution, tokenizer, dataset)
                answered, responses, metadata = rag.qa([reader_solution])
                solution.answer = answered[0].answer
            end = time.perf_counter_ns()
            completed_unix = time.time()
            events = list(recorder.pending)
            logical = [event for name, event in events if name == 'llm_logical_calls.jsonl']
            physical = [event for name, event in events if name == 'llm_calls.jsonl']
            recorder.flush()
            answers = get_gold_answers([sample], dataset)[0]
            solution.gold_answers, solution.gold_docs = answers, gold_all[index]
            row = detailed_result(solution, sample, index, gold_all[index], native_hops[index], 10, 10)
            row.update(timing=dict(execution_mode='serial', started_unix=start_unix,
                                   completed_unix=completed_unix, end_to_end_seconds=(end-start)/1e9,
                                   retrieval_seconds=(retrieval_done-start)/1e9,
                                   reader_seconds=(end-retrieval_done)/1e9),
                       calls=dict(logical=len(logical), http_attempts=len(physical),
                                  cache_hits=sum(e.get('cache_hit') is True for e in logical),
                                  failed_attempts=sum(e['status'] != 'success' for e in physical)),
                       reader=dict(raw_response=responses[0], metadata=metadata[0], budget=budget),
                       sample_stratum=stratum(sample, dataset))
            row['gpu_memory'] = dict(embedding_peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                                     embedding_peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
            row = json.loads(json.dumps(row, ensure_ascii=False, allow_nan=False))
            Recorder.append(recorder, 'progress.jsonl', dict(sha256=digest(row), row=row))
            rows.append(row)
            write(case / 'status.json', dict(status='running', dataset=dataset, completed_queries=len(rows),
                  expected_queries=50, updated_at=stamp(), last_wall_seconds=row['timing']['end_to_end_seconds']))
            print(stamp(), dataset, f'{len(rows)}/50', f"wall={row['timing']['end_to_end_seconds']:.3f}s",
                  f"calls={row['calls']['logical']}, HTTP={row['calls']['http_attempts']}", flush=True)
        recorder.flush()
        verification = verify_shared_index(receipt_path)
        write(case / 'index_verified.json', verification)
        write(case / 'result.json', dict(dataset=dataset, n=50, method='bindpack', results=rows))
        marker = dict(questions=50, completed_at=stamp(), result_sha256=sha(case / 'result.json'),
                      calls_sha256=sha(case / 'llm_calls.jsonl'), logical_sha256=sha(case / 'llm_logical_calls.jsonl'),
                      progress_sha256=sha(case / 'progress.jsonl'))
        write(case / 'completed.ok', marker)
        write(case / 'status.json', dict(status='complete', completed_queries=50, expected_queries=50))
    finally:
        recorder.flush()
        for obj, name, original in reversed(recorder.restorations):
            setattr(obj, name, original)


def aggregate(root):
    snapshot = root / 'source_snapshot'
    sys.path.insert(0, str(snapshot / 'src'))
    # Stdlib-only distribution implementation; no model initialization.
    all_wall, per_dataset, intervals, query_intervals = [], [], [], []
    total_logical = total_http = total_hits = total_failures = 0
    import statistics
    for dataset in DATASETS:
        case = root / 'runs' / dataset
        marker = read(case / 'completed.ok')
        for name, key in [('result.json', 'result_sha256'), ('llm_calls.jsonl', 'calls_sha256'),
                          ('llm_logical_calls.jsonl', 'logical_sha256'), ('progress.jsonl', 'progress_sha256')]:
            if sha(case / name) != marker[key]:
                raise RuntimeError('Changed completed measurement: ' + name)
        rows = read(case / 'result.json')['results']
        selection = read(root / 'samples' / (dataset + '.json'))
        if len(rows) != 50 or [r['sample_id'] for r in rows] != selection['sample_ids']:
            raise RuntimeError('Incomplete or reordered sample results')
        if dict(Counter(r['sample_stratum'] for r in rows)) != QUOTAS[dataset]:
            raise RuntimeError('Measured sample distribution differs')
        calls, logical = jsonl(case / 'llm_calls.jsonl'), jsonl(case / 'llm_logical_calls.jsonl')
        if any(not event['phase'].startswith('online_') for event in calls + logical):
            raise RuntimeError('Unexpected non-online LLM calls')
        if sum(r['calls']['logical'] for r in rows) != len(logical) or sum(r['calls']['http_attempts'] for r in rows) != len(calls):
            raise RuntimeError('Per-question and event call counts differ')
        wall = [r['timing']['end_to_end_seconds'] for r in rows]
        for row in rows:
            t = row['timing']
            if abs(t['end_to_end_seconds'] - t['retrieval_seconds'] - t['reader_seconds']) > 1e-7:
                raise RuntimeError('Timing boundaries differ')
            query_intervals.append((t['started_unix'], t['completed_unix']))
        for event in calls:
            intervals.append((event['started_unix'], event['started_unix'] + event['seconds']))
        hits = sum(event.get('cache_hit') is True for event in logical)
        failures = sum(event['status'] != 'success' for event in calls)
        item = dict(dataset=dataset, questions=50, distribution=QUOTAS[dataset],
                    wall_seconds_total=sum(wall), wall_seconds_per_query=statistics.mean(wall),
                    wall_median_seconds=statistics.median(wall),
                    llm_logical_calls=len(logical), llm_calls_per_query=len(logical)/50,
                    http_attempts=len(calls), http_calls_per_query=len(calls)/50,
                    cache_hits=hits, failed_attempts=failures,
                    result=str(case / 'result.json'))
        per_dataset.append(item)
        total_logical += len(logical)
        total_http += len(calls)
        total_hits += hits
        total_failures += failures
        all_wall.extend(wall)
    def assert_serial(values, kind):
        values = sorted(values)
        if any(end > next_start + .001 for (_, end), (next_start, _) in zip(values, values[1:])):
            raise RuntimeError(kind + ' overlapped')
    assert_serial(intervals, 'HTTP requests')
    assert_serial(query_intervals, 'Queries')
    result = dict(status='complete', completed_at=stamp(), method='BindPack Full', llm='Qwen3-8B',
                  embedding='Qwen3-Embedding-8B', questions=200, query_concurrency=1,
                  llm_concurrency=1, per_dataset=per_dataset,
                  overall=dict(questions=200, wall_seconds_total=sum(all_wall),
                               wall_seconds_per_query=statistics.mean(all_wall),
                               llm_logical_calls=total_logical, llm_calls_per_query=total_logical/200,
                               http_attempts=total_http, http_calls_per_query=total_http/200,
                               cache_hits=total_hits, failed_attempts=total_failures),
                  verification=dict(query_overlap=False, http_overlap=False, exact_quotas=True,
                                    result_hashes_verified=True), protocol=str(root / 'protocol.json'))
    write(root / 'aggregate.json', result)
    lines = ['# BindPack / Qwen3-8B + Qwen3-Embedding-8B / serial 200 queries', '',
             '| Dataset | Questions | Wall-time (s/question) | LLM Calls/question | HTTP attempts/question |',
             '| --- | ---: | ---: | ---: | ---: |']
    for item in per_dataset + [dict(dataset='Overall', **result['overall'])]:
        lines.append(f"| {item['dataset']} | {item['questions']} | {item['wall_seconds_per_query']:.3f} | {item['llm_calls_per_query']:.3f} | {item['http_calls_per_query']:.3f} |")
    lines.extend(['', 'Wall-time is measured continuously from native retrieval start through native QA and answer parsing. Model/index preparation and measurement/checkpoint disk writes are excluded. First query is included. All 200 sampled questions remain in the denominator.', '',
                  'LLM Calls counts logical native calls, including within-run application cache hits. Physical HTTP attempts, retries and failed requests are separately recorded. No historical application/query cache or answers are reused. Native vLLM server caches remain resident.', '',
                  'Both query and physical HTTP intervals were checked for overlap; maximum concurrency is 1. Native benchmark hop hints, Full configuration and Reader@5 match the adopted four-dataset experiment. Corpus/index covers the full benchmark for each dataset.', '',
                  'Detailed per-query times and call counts: runs/<dataset>/result.json. Frozen sample IDs, quotas and indices: samples/. Provenance and timing boundaries: protocol.json. Raw SDK attempts: runs/<dataset>/llm_calls.jsonl; logical calls: llm_logical_calls.jsonl.'])
    (root / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    write(root / 'completed.ok', dict(questions=200, aggregate_sha256=sha(root / 'aggregate.json'), completed_at=stamp()))
    return result


def main(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    prepare(root)
    if args.prepare_only:
        print('Prepared exact 200-question sample; no LLM calls', flush=True)
        return
    if args.dataset:
        worker(root, args.dataset)
        return
    start = time.perf_counter()
    try:
        for position, dataset in enumerate(DATASETS):
            write(root / 'status.json', dict(status='running', dataset=dataset,
                  completed_datasets=position, total_datasets=4, query_concurrency=1, updated_at=stamp()))
            case = root / 'runs' / dataset
            if not (case / 'completed.ok').exists():
                command = [sys.executable, '-B', '-u', str(root / 'source_snapshot/scripts/run_serial_cost_200.py'),
                           '--output', str(root), '--dataset', dataset]
                print(stamp(), 'START', dataset, '50 queries, serial retrieval + QA', flush=True)
                process = subprocess.Popen(command, cwd=PROJECT, stdin=subprocess.DEVNULL)
                write(root / 'worker.json', dict(dataset=dataset, pid=process.pid, command=command))
                if process.wait():
                    raise RuntimeError('Dataset worker failed: ' + dataset)
        result = aggregate(root)
        result['controller_wall_seconds'] = time.perf_counter() - start
        write(root / 'controller_timing.json', dict(seconds=result['controller_wall_seconds']))
        write(root / 'status.json', dict(status='complete', completed_datasets=4, completed_queries=200,
              completed_at=stamp(), overall=result['overall']))
        pointer = PROJECT / 'outputs/latest_serial_cost_200.json'
        write(pointer, dict(status='complete', output_dir=str(root), aggregate=str(root / 'aggregate.json')))
        print(stamp(), 'COMPLETE', result['overall'], flush=True)
    except BaseException as exc:
        write(root / 'status.json', dict(status='failed', error=f'{type(exc).__name__}: {exc}', updated_at=stamp()))
        traceback.print_exc()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', choices=DATASETS)
    parser.add_argument('--prepare-only', action='store_true')
    main(parser.parse_args())

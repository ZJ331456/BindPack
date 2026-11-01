#!/usr/bin/env python3
"""Reproduce BindPack Full on the four bundled 1,000-question benchmarks."""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('hotpotqa', '2wikimultihopqa', 'musique', 'hover')
os.environ.setdefault('LITELLM_LOCAL_MODEL_COST_MAP', 'True')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/bindpack_full')
    parser.add_argument('--datasets', nargs='+', choices=DATASETS)
    parser.add_argument('--datasets-dir', type=Path)
    parser.add_argument('--models-dir', type=Path)
    parser.add_argument('--llm-base-url')
    parser.add_argument('--query-window', type=int, choices=(1, 8, 16, 32))
    parser.add_argument('--reader-workers', type=int, choices=(1, 8, 16, 32))
    reuse = parser.add_mutually_exclusive_group()
    reuse.add_argument('--index-root', type=Path, help='Verified archive shared_indexes directory')
    reuse.add_argument('--rebuild-index', action='store_true', help='Build all selected indexes from scratch')
    parser.add_argument('--resume', action='store_true', help='Use the original snapshot and durable checkpoints')
    parser.add_argument('--prepare-only', action='store_true', help='Validate all inputs/indexes without model calls')
    return parser.parse_args()


def prepare(args):
    output = args.output.resolve()
    protocol_file = output / 'run_protocol.json'
    if args.resume:
        protocol = read(protocol_file)
        if protocol['schema_version'] != 1 or protocol['method'] != 'bindpack_full':
            raise ValueError('This output is not a frozen BindPack Full run')
        checks = {'datasets': args.datasets, 'datasets_dir': args.datasets_dir,
                  'models_dir': args.models_dir, 'llm_base_url': args.llm_base_url,
                  'query_window': args.query_window, 'reader_workers': args.reader_workers,
                  'index_root': args.index_root}
        for key, actual in checks.items():
            if actual is not None:
                actual = str(actual.resolve()) if isinstance(actual, Path) else actual
                if protocol[key] != actual:
                    raise ValueError('Resume must preserve the original ' + key)
        if args.rebuild_index and protocol['index_root'] is not None:
            raise ValueError('Resume cannot switch from archived indexes to rebuilding')
        for relative, expected in protocol['source_sha256'].items():
            if sha(output / 'source_snapshot' / relative) != expected:
                raise ValueError('Frozen source changed: ' + relative)
        for relative, expected in protocol['input_sha256'].items():
            if sha(output / relative) != expected:
                raise ValueError('Frozen input changed: ' + relative)
        return protocol
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose an empty --output, or --resume this same run')
    datasets = args.datasets or list(DATASETS)
    if len(set(datasets)) != len(datasets):
        raise ValueError('Select each dataset only once')
    data_dir = (args.datasets_dir or ROOT / 'datasets').resolve()
    models = (args.models_dir or Path('/root/models')).resolve()
    for dataset in datasets:
        questions = read(data_dir / (dataset + '.json'))
        if not isinstance(questions, list) or len(questions) != 1000:
            raise ValueError(dataset + ' must contain all 1,000 questions')
        if not (data_dir / (dataset + '_corpus.json')).is_file():
            raise ValueError('Missing corpus: ' + dataset)
    for model in ('Qwen3-8B', 'Qwen3-Embedding-8B'):
        if not (models / model / 'config.json').is_file():
            raise ValueError('Missing local model: ' + str(models / model))
    window = args.query_window or 8
    readers = args.reader_workers or (1 if window == 1 else 8)
    if window == 1 and readers != 1:
        raise ValueError('Use --reader-workers 1 with a serial query window')
    default_index = ROOT / 'outputs/4_datasets_result/shared_indexes'
    index = None if args.rebuild_index else (args.index_root or (default_index if default_index.is_dir() else None))
    if index is not None:
        index = index.resolve()
        for dataset in datasets:
            if not (index / dataset).is_dir():
                raise ValueError('Missing archived index: ' + str(index / dataset))
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / 'source_snapshot'
    for folder in ('src', 'scripts'):
        shutil.copytree(ROOT / folder, snapshot / folder,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.egg-info'))
    frozen_data = output / 'frozen_datasets'
    frozen_data.mkdir()
    for dataset in datasets:
        for suffix in ('.json', '_corpus.json'):
            shutil.copy2(data_dir / (dataset + suffix), frozen_data / (dataset + suffix))
    defaults = read(snapshot / 'src/bindpack/configuration/defaults.json')
    if (defaults['runtime']['source_bound_eligibility'] is not False or
            defaults['runtime']['mechanism_ablation'] != 'full' or
            defaults['runtime']['pack_mechanism_ablation'] != 'none' or
            defaults['pack']['selector_mode'] != 'combined' or
            defaults['pack']['review_mode'] != 'off' or
            defaults['pack']['relation_weight'] != .6):
        raise ValueError('Source defaults differ from the final paper Full')
    runtime = dict(defaults['runtime'], llm_model_path=str(models / 'Qwen3-8B'),
                   embedding_model_name=str(models / 'Qwen3-Embedding-8B'),
                   llm_base_url=args.llm_base_url or defaults['runtime']['llm_base_url'],
                   llm_prefetch_workers=window, openie_max_workers=8)
    config = dict(method_config='src/bindpack/configuration/defaults.json', runtime=runtime)
    write(output / 'run_config.json', config)
    protocol = dict(schema_version=1, method='bindpack_full',
        created_at=datetime.now().astimezone().isoformat(), datasets=datasets,
        queries_per_dataset=1000, expected_queries=1000 * len(datasets),
        datasets_dir=str(data_dir), models_dir=str(models), llm_base_url=runtime['llm_base_url'],
        index_root=str(index) if index else None, query_window=window, reader_workers=readers,
        source_sha256={str(p.relative_to(snapshot)): sha(p) for p in sorted(snapshot.rglob('*'))
                       if p.is_file() and p.suffix in ('.py', '.json', '.sh')},
        input_sha256={str(p.relative_to(output)): sha(p) for p in sorted(frozen_data.glob('*.json'))},
        stage='full', source_bound_eligibility=False, selector_mode='combined', review_mode='off',
        qa_top_k=5, retrieval_top_k=200, relation_weight=.6,
        hop_policy='native benchmark hop annotations, oracle-assisted as in the paper',
        qa_policy='native Reader@5; no thinking; 2048 output cap; all questions retained',
        cache_policy='fresh per-dataset application cache; retain real attempts on resume')
    protocol['input_sha256']['run_config.json'] = sha(output / 'run_config.json')
    commit = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                            text=True, capture_output=True)
    protocol['git_commit'] = commit.stdout.strip() if commit.returncode == 0 else None
    write(protocol_file, protocol)
    return protocol


def completed(case):
    if not (case / 'completed.ok').is_file():
        return False
    marker = read(case / 'completed.ok')
    for name, suffix in (('report', '.json'), ('result', '.json'), ('llm_calls', '.jsonl')):
        if sha(case / (name + suffix)) != marker[name + '_sha256']:
            raise ValueError('Completed artifact changed: ' + str(case / name))
    report = read(case / 'report.json')
    if marker['query_count'] != 1000 or report['coverage']['sample_count'] != 1000 or report['coverage']['qa_n'] != 1000:
        raise ValueError('Full evaluation is not complete: ' + str(case))
    if read(case / 'audit.json')['status'] != 'passed':
        raise ValueError('Full evaluation did not pass its audit: ' + str(case))
    return True


def summarize(output, protocol, status):
    results = []
    for dataset in protocol['datasets']:
        case = output / 'runs' / dataset / 'cases' / dataset / 'full'
        if completed(case):
            report = read(case / 'report.json')
            results.append(dict(dataset=dataset, queries=1000, metrics=report['metrics'],
                                report=str(case / 'report.json'), report_sha256=sha(case / 'report.json')))
    write(output / 'summary.json', dict(status=status, completed_datasets=len(results),
          expected_datasets=len(protocol['datasets']), results=results))
    lines = ['# BindPack Full', '', '| Dataset | Recall@5 | FCR@5 | DCC@5 | EM / Accuracy | F1 / Macro-F1 |',
             '|---|---:|---:|---:|---:|---:|']
    def percent(value):
        return 'N/A' if value is None else f'{round(float(value) * 100, 10):.2f}'
    for result in results:
        m = result['metrics']
        classification = result['dataset'] == 'hover'
        values = [m.get('Recall@5'), m.get('FCR@5'), m.get('DCC@5'),
                  m.get('ClassificationAccuracy' if classification else 'ExactMatch'),
                  m.get('ClassificationMacroF1' if classification else 'F1')]
        lines.append('| ' + result['dataset'] + ' | ' + ' | '.join(map(percent, values)) + ' |')
    (output / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    return results


def ensure_service(protocol):
    url = protocol['llm_base_url'].rstrip('/') + '/models'
    def probe():
        request = urllib.request.Request(url, headers={
            'Authorization': 'Bearer ' + os.environ.get('OPENAI_API_KEY', 'EMPTY')})
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                rows = json.load(response)['data']
        except urllib.error.URLError:
            return False
        selected = next((r for r in rows if r['id'] == 'qwen3-8b'), None)
        if selected is None:
            raise ValueError('The configured endpoint serves a different model')
        if selected.get('root') and Path(selected['root']).resolve() != Path(protocol['models_dir']) / 'Qwen3-8B':
            raise ValueError('Existing server model path differs from this Full protocol')
        return True
    if probe():
        return
    if protocol['llm_base_url'] != 'http://127.0.0.1:8035/v1':
        raise RuntimeError('Start the configured Qwen3-8B endpoint before evaluating')
    service = ROOT / 'outputs/services'
    service.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MODELS_DIR=protocol['models_dir'], MAX_NUM_SEQS=str(protocol['query_window']))
    with (service / 'qwen3-8b.log').open('a') as log:
        process = subprocess.Popen(['bash', str(ROOT / 'scripts/_start_llm.sh')], env=env,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (service / 'qwen3-8b.pid').write_text(str(process.pid) + '\n')
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError('Qwen3 server exited; inspect ' + str(service / 'qwen3-8b.log'))
        if probe():
            return
        time.sleep(2)
    raise RuntimeError('Qwen3 server startup timed out; inspect its service log')


def main(args):
    protocol = prepare(args)
    output = args.output.resolve()
    lock = (output / '.queue.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    current = [None]
    stopping = [False]
    def stop(number, frame):
        stopping[0] = True
        if current[0] is not None and current[0].poll() is None:
            current[0].send_signal(signal.SIGTERM)
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, stop)
    def emit(message):
        line = f'{datetime.now():%Y-%m-%d %H:%M:%S} {message}'
        print(line, flush=True)
        with (output / 'run.log').open('a') as log:
            log.write(line + '\n')
    (output / 'run.pid').write_text(str(os.getpid()) + '\n')
    state = dict(status='running', pid=os.getpid(), dataset=None, completed_datasets=0)
    try:
        for dataset in protocol['datasets']:
            case = output / 'runs' / dataset / 'cases' / dataset / 'full'
            if completed(case):
                emit('SKIP completed Full: ' + dataset)
                state['completed_datasets'] += 1
                continue
            if stopping[0]:
                break
            if not args.prepare_only:
                ensure_service(protocol)
            if stopping[0]:
                break
            worker = output / 'runs' / dataset
            worker.mkdir(parents=True, exist_ok=True)
            links = output / 'cases'
            links.mkdir(exist_ok=True)
            link = links / dataset
            if link.is_symlink():
                if link.resolve() != (worker / 'cases' / dataset).resolve():
                    raise ValueError('Unexpected case link: ' + str(link))
            elif link.exists():
                raise ValueError('Case link is occupied: ' + str(link))
            else:
                link.symlink_to(Path('../runs') / dataset / 'cases' / dataset, target_is_directory=True)
            command = [sys.executable, '-B', '-u', str(output / 'source_snapshot/scripts/_evaluate_full.py'),
                '--dataset', dataset, '--output', str(worker), '--config', str(output / 'run_config.json'),
                '--datasets-dir', str(output / 'frozen_datasets'), '--query-window', str(protocol['query_window']),
                '--reader-workers', str(protocol['reader_workers'])]
            if (worker / 'experiment_config.json').exists():
                command.append('--resume')
            if protocol['index_root']:
                command += ['--reuse-archive-index', str(Path(protocol['index_root']) / dataset)]
            if args.prepare_only:
                command.append('--prepare-only')
            state['dataset'] = dataset
            write(output / 'status.json', state)
            emit('START Full: ' + dataset)
            env = dict(os.environ, BINDPACK_PROJECT_ROOT=str(ROOT),
                       PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false')
            process = subprocess.Popen(command, env=env, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            current[0] = process
            with (output / 'run.log').open('a') as combined, (worker / 'run.log').open('a') as log:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    combined.write(line); combined.flush()
                    log.write(line); log.flush()
            code = process.wait()
            current[0] = None
            if stopping[0]:
                break
            if code or not args.prepare_only and not completed(case):
                raise RuntimeError('Dataset failed; checkpoints retained: ' + str(worker / 'run.log'))
            if not args.prepare_only:
                if protocol['index_root']:
                    from utils.archived_index import verify_private
                    verify_private(case, release=True)
                state['completed_datasets'] += 1
                summarize(output, protocol, 'running')
            emit(('PREPARED: ' if args.prepare_only else 'PASS Full: ') + dataset)
        state['status'] = 'paused' if stopping[0] else ('prepared' if args.prepare_only else 'complete')
        results = summarize(output, protocol, state['status'])
        if state['status'] == 'complete':
            write(output / 'completed.ok', dict(datasets=len(results), queries=1000 * len(results),
                  summary_sha256=sha(output / 'summary.json')))
        emit(state['status'].upper())
    except BaseException as error:
        state.update(status='failed', error_type=type(error).__name__, error=str(error))
        summarize(output, protocol, 'failed')
        raise
    finally:
        state['dataset'] = None
        state['updated_at'] = datetime.now().astimezone().isoformat()
        write(output / 'status.json', state)


if __name__ == '__main__':
    main(options())

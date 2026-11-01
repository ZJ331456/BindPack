#!/usr/bin/env python3
"""Run current Full on 4 x 400 fixed queries, serially, then publish paper costs."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
from zoneinfo import ZoneInfo

from utils.serial_cost import DATASETS, SAMPLE_SIZE, SEED, aggregate, read, sha, stratified_sample, write

PROJECT = Path('/root/BindPack')
ARCHIVE = PROJECT / 'outputs/4_datasets_result'
FULL = ARCHIVE / 'final_full_no_source_20261028'
DEFAULT = PROJECT / 'outputs/serial_cost_1600_20261029'
PAPER = PROJECT / 'docs/DASFAA_2027_latex'


def stamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def sources(snapshot):
    return {str(p.relative_to(snapshot)): sha(p) for p in snapshot.rglob('*')
            if p.is_file() and p.suffix in ('.py', '.json')}


def prepare(root):
    if (root / 'protocol.json').exists():
        protocol = read(root / 'protocol.json')
        assert sources(root / 'source_snapshot') == protocol['source_sha256']
        assert sha(root / 'config.json') == protocol['config_sha256']
        for ds, files in protocol['dataset_sha256'].items():
            for name, digest in files.items():
                assert sha(root / 'frozen_datasets' / name) == digest
            assert sha(root / 'samples' / (ds + '.json')) == protocol['sample_sha256'][ds]
        return protocol
    snapshot = root / 'source_snapshot'
    for directory in ('src', 'scripts', 'configs'):
        shutil.copytree(PROJECT / directory, snapshot / directory,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    defaults = read(snapshot / 'src/bindpack/configuration/defaults.json')
    contract = read(FULL / 'manifest.json')['full_default_configuration']
    for key in ('route', 'plan', 'pack', 'candidates'):
        assert defaults[key] == contract[key], 'Current algorithm changed from adopted Full: ' + key
    assert defaults['runtime']['source_bound_eligibility'] is False
    from eval_utils import get_benchmark_hops, get_sample_id, get_classification_label
    dataset_hashes = {}; sample_hashes = {}
    for ds in DATASETS:
        dataset_hashes[ds] = {}
        for suffix in ('', '_corpus'):
            name = ds + suffix + '.json'
            source = FULL / 'input_provenance' / ds / name
            assert sha(source) == sha(PROJECT / 'datasets' / name), 'Dataset changed from adopted Full: ' + name
            target = root / 'frozen_datasets' / name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target); dataset_hashes[ds][name] = sha(target)
        data = read(root / 'frozen_datasets' / (ds + '.json')); hops = get_benchmark_hops(data, ds)
        records = [dict(sample_id=get_sample_id(sample, ds), hop=hops[i],
                        label=get_classification_label(sample) if ds == 'hover' else '')
                   for i, sample in enumerate(data)]
        chosen, strata = stratified_sample(ds, records)
        write(root / 'indices' / (ds + '.json'), chosen)
        write(root / 'samples' / (ds + '.json'), dict(dataset=ds, seed=SEED,
            population=len(data), n=SAMPLE_SIZE, indices=chosen,
            sample_ids=[records[i]['sample_id'] for i in chosen], strata=strata,
            selection='proportional hop/label strata; deterministic SHA selection and order; no performance or timing inputs'))
        sample_hashes[ds] = sha(root / 'samples' / (ds + '.json'))
        # Independent preflight questions, excluded from the 1,600-question accounting.
        if ds in ('hotpotqa', 'hover'):
            other = [i for i in range(len(data)) if i not in set(chosen)]
            selected = []; labels = set()
            for i in other:
                label = records[i]['label']
                if ds == 'hotpotqa' or label not in labels:
                    selected.append(i); labels.add(label)
                if len(selected) == 2:
                    break
            write(root / 'preflight_indices' / (ds + '.json'), selected)
    settings = read(PROJECT / 'configs/ablations_qwen3.json')
    settings['method_config'] = str(snapshot / 'configs/full_no_source_eligibility.json')
    settings['datasets'] = list(DATASETS); settings['datasets_dir'] = str(root / 'frozen_datasets')
    settings['runtime'].update(llm_prefetch_workers=1, openie_max_workers=1)
    settings['evaluation'].update(eval_mode='rag_qa', sample_size=SAMPLE_SIZE, qa_top_k=5)
    write(root / 'config.json', settings)
    protocol = dict(schema_version=1, created_at=stamp(), datasets=list(DATASETS), n=1600,
        queries_per_dataset=SAMPLE_SIZE, query_window=1, client_llm_workers=1,
        qa_executed=True, reader_k=5, method='current Full', source_bound_eligibility=False,
        support_semantic_veto=False, relevance_weight=.6, seed=SEED,
        dataset_sha256=dataset_hashes, sample_sha256=sample_hashes, source_sha256=sources(snapshot),
        config_sha256=sha(root / 'config.json'),
        execution='one evaluation worker at a time; datasets serial; each query and all client LLM requests serial',
        inputs='full verified corpus/index per dataset; selected 400 queries only; no historical query/LLM cache reused',
        hop_policy='annotation-derived native hop hints, unchanged from the adopted Full',
        service='resident vLLM unchanged; physical max-num-seqs8, GPU fraction0.55; shared prefix cache not reset',
        aggregate='sum all online totals over 1600; completed native retrieval + prepared Reader@5 duration; loading/index preparation separate',
        failure_policy='preserve every physical attempt and checkpoint; do not publish incomplete cases',
        preflight='2 HotpotQA + 2 HoVer outside official selection; separate accounting',
        publication='after all four completed/audited reports, fill Appendix cost/time/resource tables and compile; do not replace 1000-question quality with 400-question scores')
    write(root / 'protocol.json', protocol)
    (root / 'README.md').write_text('# BindPack串行开销：4×400题\n\n'
        '最新版Full，来源资格与末端核验关闭，相关性权重0.60，原生Reader@5。'
        'HotpotQA→2Wiki→MuSiQue→HoVer；一个评测工作进程、查询窗口1、客户端LLM线程1。'
        '复用完整公共索引，重跑检索与QA，不复用历史答案或应用缓存。\n\n'
        'samples/冻结分层抽样和题目ID；runs/保存逐题结果、请求/tokens、耗时和资源。'
        '预检题不计入1600题。样本标记development_fixture仅表示使用原1000题的子集，语料与索引均为完整版本。\n\n'
        '四组完成后才生成aggregate.json，按总量/1600计算均值，并更新论文附录的开销、时间与资源。'
        '基线仍是各4000题，表注会区分样本量。400题QA不会覆盖原1000题主表。\n\n'
        '查看：tail -f /root/BindPack/outputs/serial_cost_1600_20261029/run.log\n'
        '恢复：bash /root/BindPack/scripts/run_serial_cost_benchmark.sh\n')
    return protocol


def complete(output, ds, n):
    case = output / 'cases' / ds / 'full'
    if not (case / 'completed.ok').exists():
        return False
    done = read(case / 'completed.ok'); assert done['query_count'] == n
    for name, suffix in [('report', '.json'), ('result', '.json'), ('llm_calls', '.jsonl')]:
        assert sha(case / (name + suffix)) == done[name + '_sha256']
    assert read(case / 'audit.json')['status'] == 'passed'
    assert read(case / 'report.json')['coverage']['qa_n'] == n
    return True


def compile_paper(root):
    """Compile the updated manuscript with existing cached TeX dependencies."""
    tool = Path('/root/.cache/tectonic-bindpack/tectonic')
    if not tool.is_file():
        raise RuntimeError('Paper sources updated; cached TeX compiler unavailable. Resume with --publish-only after restoring it.')
    build = root / 'paper_build'; source = build / 'source'; output = build / 'output'
    shutil.copytree(PAPER, source, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__', 'main.pdf'))
    output.mkdir(parents=True, exist_ok=True)
    command = [str(tool), '-C', '-b', 'http://127.0.0.1:18876/tlextras-2022.0r0.tar',
               '--keep-logs', '--outdir', str(output), 'main.tex']
    with (root / 'paper_compile.log').open('w') as log:
        code = subprocess.run(command, cwd=source,
            env=dict(os.environ, XDG_CACHE_HOME='/tmp/bindpack_english_tex_cache'),
            stdout=log, stderr=subprocess.STDOUT).returncode
    if code:
        raise RuntimeError('Measurements and paper sources saved; PDF compilation failed: ' + str(root / 'paper_compile.log'))
    temp = PAPER / 'main.serial-cost.tmp.pdf'; shutil.copy2(output / 'main.pdf', temp)
    temp.replace(PAPER / 'main.pdf')
    validation_path = PAPER / 'results/update_validation.json'
    validation = read(validation_path) if validation_path.exists() else {}
    log_text = (output / 'main.log').read_text(errors='replace')
    validation.update(updated_at=stamp(), pdf_sha256=sha(PAPER / 'main.pdf'), compile_exit=0,
        overfull_boxes=log_text.count('Overfull'),
        unresolved_reference_warning='There were undefined references' in log_text,
        matched_serial_measurement_exists=True, serial_cost_queries=1600,
        serial_cost_queries_per_dataset=400, query_concurrency=1, client_llm_workers=1,
        serial_cost_timing_resource_fields_NA=False,
        original_1000_question_quality_tables_unchanged=True)
    write(validation_path, validation)
    write(root / 'paper_compiled.ok', dict(compiled_at=stamp(), pdf=str(PAPER / 'main.pdf'),
                                          pdf_sha256=sha(PAPER / 'main.pdf')))


def publish(root):
    data = aggregate(root); write(root / 'aggregate.json', data)
    # The paper synchronizer verifies all source hashes again before using these measurements.
    write(PAPER / 'results/serial_cost.json', data)
    before = sha(PAPER / 'figures/fig02_framework.pdf')
    result = subprocess.run(['python', str(PAPER / 'sync_results.py')], cwd=PAPER,
                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), capture_output=True, text=True)
    (root / 'paper_update.log').write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError('Measurements saved; paper synchronization failed: ' + str(root / 'paper_update.log'))
    assert sha(PAPER / 'figures/fig02_framework.pdf') == before
    compile_paper(root)
    write(root / 'paper_updated.ok', dict(updated_at=stamp(), queries=1600,
        serial_cost_sha256=sha(PAPER / 'results/serial_cost.json'), framework_unchanged=True))
    print(stamp(), 'PAPER UPDATED:', data['means'], flush=True)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--publish-only', action='store_true')
    parser.add_argument('--frozen', action='store_true')
    args = parser.parse_args(); root = args.output.resolve(); root.mkdir(parents=True, exist_ok=True)
    prepare(root)
    if not args.frozen:
        os.execv(sys.executable, [sys.executable, '-B', '-u', str(root / 'source_snapshot/scripts/run_serial_cost_benchmark.py'),
            '--frozen', '--output', str(root), *(['--prepare-only'] if args.prepare_only else []),
            *(['--publish-only'] if args.publish_only else [])])
    if args.prepare_only:
        print('Prepared all 1600 samples; no model calls'); return
    if args.publish_only:
        publish(root); return
    lock = (PROJECT / 'outputs/.bindpack_serial_cost.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (root / 'run.pid').write_text(str(os.getpid()) + '\n')
    stop = threading.Event(); end = threading.Event(); child = [None]
    state = dict(status='running', phase='preflight', pid=os.getpid(), completed_datasets=0, expected_datasets=4,
                 official_completed_queries=0, expected_queries=1600, query_concurrency=1, client_llm_workers=1)
    def signal_stop(*unused):
        stop.set()
        if child[0] and child[0].poll() is None:
            child[0].send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, signal_stop); signal.signal(signal.SIGINT, signal_stop)
    def heartbeat():
        while not end.is_set():
            value = dict(state, updated_at=stamp()); output = state.get('worker_output')
            if output and (Path(output) / 'status.json').exists():
                value['active'] = read(Path(output) / 'status.json')
            write(root / 'status.json', value)
            active = value.get('active', {})
            print(stamp(), 'PROGRESS', value['phase'], value.get('dataset', ''),
                  f"{active.get('completed_queries',0)}/{active.get('expected_queries','?')}",
                  f"datasets={value['completed_datasets']}/4", flush=True)
            end.wait(30)
    watcher = threading.Thread(target=heartbeat, daemon=True); watcher.start()
    def execute(ds, output, indices, n):
        if stop.is_set():
            raise KeyboardInterrupt('Requested stop')
        state.update(dataset=ds, worker_output=str(output))
        case = output / 'cases' / ds / 'full'
        if not complete(output, ds, n):
            command = [sys.executable, '-B', '-u', str(root / 'source_snapshot/scripts/run_bindpack_measured.py'),
                '--frozen-worker', '--dataset', ds, '--output', str(output), '--config', str(root / 'config.json'),
                '--datasets-dir', str(root / 'frozen_datasets'), '--reuse-archive-index', str(ARCHIVE / 'shared_indexes' / ds),
                '--query-indices', str(indices), '--query-window', '1', '--source-bound-eligibility', 'off',
                '--mechanism-ablation', 'full', '--eval-mode', 'rag_qa', '--candidate-output-top-k', '10']
            if (output / 'experiment_config.json').exists():
                command.append('--resume')
            case.mkdir(parents=True, exist_ok=True)
            write(output / 'launch.json', dict(command=command, dataset=ds, n=n))
            print(stamp(), 'START', state['phase'], ds, n, 'queries; serial retrieval + QA', flush=True)
            process = subprocess.Popen(command, cwd=PROJECT, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            child[0] = process
            with (case / 'run.log').open('a') as log:
                for line in process.stdout:
                    log.write(line); log.flush(); print(line, end='', flush=True)
            code = process.wait(); child[0] = None
            if stop.is_set():
                raise KeyboardInterrupt('Stopped; durable query checkpoints and costs retained')
            if code or not complete(output, ds, n):
                raise RuntimeError('Case failed; preserved checkpoints: ' + str(case / 'run.log'))
        if not (case / 'private_assets_released.ok').exists():
            from utils.archived_index import verify_private
            verify_private(case, release=True)
        print(stamp(), 'PASS', ds, n, flush=True)
    try:
        for ds in ('hotpotqa', 'hover'):
            execute(ds, root / 'preflight' / ds, root / 'preflight_indices' / (ds + '.json'), 2)
        write(root / 'preflight/passed.json', dict(status='passed', queries=4, official_cost_denominator=False))
        state['phase'] = 'official'
        for ds in DATASETS:
            execute(ds, root / 'runs' / ds, root / 'indices' / (ds + '.json'), SAMPLE_SIZE)
            state['completed_datasets'] += 1; state['official_completed_queries'] += SAMPLE_SIZE
        state['phase'] = 'publishing'
        data = publish(root)
        state.update(status='complete', phase='complete', means=data['means'])
        write(root / 'completed.ok', dict(queries=1600, datasets=4, finished_at=stamp(), paper_updated=True))
    except BaseException as exc:
        state.update(status='stopped' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc))
        raise
    finally:
        end.set(); watcher.join(timeout=3); write(root / 'status.json', dict(state, updated_at=stamp()))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Evaluate 100 new questions per dataset at R=.60, retain the old 50 separately."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PROJECT = Path('/root/BindPack')
BASE = PROJECT / 'outputs/pack_weight_diagnostic_20261025'
ARCHIVE = PROJECT / 'outputs/4_datasets_result'
DEFAULT = PROJECT / 'outputs/pack_r060_expanded_20261025'
DATASETS = ('hotpotqa', '2wikimultihopqa', 'musique', 'hover')
SCRIPT = Path(__file__).resolve()


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def helpers():
    path = BASE / 'source_snapshot/scripts/run_pack_weight_diagnostic.py'
    spec = importlib.util.spec_from_file_location('original_pack_diagnostic', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def choose_new(rows, excluded, dataset, count=100, seed=20261025):
    """Outcome-blind deterministic draw without replacement, excluding old IDs."""
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate source sample IDs')
    pool = [{k: row[k] for k in ('sample_id', 'query_index', 'question', 'benchmark_hops')}
            for row in rows if row['sample_id'] not in excluded]
    if len(pool) < count:
        raise ValueError('Insufficient unseen questions')
    pool.sort(key=lambda row: digest([seed, dataset, row['sample_id']]))
    return pool[:count]


def prepare(output):
    old_lock = read(BASE / 'selection_locked.ok')
    old_selection = read(BASE / 'selection_manifest.json')
    assert digest(old_selection) == old_lock['sha256']
    if (output / 'selection_locked.ok').exists():
        assert digest(read(output / 'selection_manifest.json')) == read(output / 'selection_locked.ok')['sha256']
        return
    assert read(BASE / 'audit.json')['status'] == 'passed'
    output.mkdir(parents=True, exist_ok=True)
    selection = dict(schema_version=1, relation_weight=.6, seed=20261025,
        new_per_dataset=100, old_per_dataset=50,
        selection_uses_outcomes=False, excludes_original_diagnostic=True,
        sampling='Uniform hash-order draw from the remaining 950 questions per dataset.',
        combined_interpretation='Old outcome-stratified 50 + new outcome-blind 100; report strata separately.',
        original_selection_sha256=old_lock['sha256'], datasets={})
    inputs = {}
    for ds in DATASETS:
        old = read(BASE / 'cases' / ds / 'full_r060/result.json')
        source_path = ARCHIVE / 'cases' / ds / 'full/result.json'
        source = read(source_path)
        excluded = {r['sample_id'] for r in old['results']}
        chosen = choose_new(source['results'], excluded, ds)
        assert len(excluded) == 50 and not excluded & {r['sample_id'] for r in chosen}
        selection['datasets'][ds] = dict(rows=chosen,
            excluded_ids=sorted(excluded), source_population=len(source['results']),
            hop_condition=old['hop_provenance'],
            hop_distribution=dict(Counter(str(r['benchmark_hops']) for r in chosen)))
        meta = output / 'metadata' / ds
        meta.mkdir(parents=True, exist_ok=True)
        for file in ('benchmark.json', 'corpus.json'):
            target = meta / file
            if not target.exists():
                target.symlink_to(os.path.relpath(BASE / 'metadata' / ds / file, meta))
        identity = read(BASE / 'metadata' / ds / 'sample_manifest.json')
        identity.update(data_path=str(meta / 'benchmark.json'),
            selected_indices=[r['query_index'] for r in chosen],
            sample_ids=[r['sample_id'] for r in chosen],
            selection_sha256=digest(chosen), diagnostic_selection=False,
            development_fixture=True, outcome_blind_expansion=True)
        write(meta / 'sample_manifest.json', identity)
        method = read(BASE / 'metadata' / ds / 'full_r060_method.json')
        assert method['pack']['relation_weight'] == .6
        method['runtime']['save_dir'] = str(output / 'runtime' / ds / 'index')
        write(meta / 'full_r060_method.json', method)
        passage = next((ARCHIVE / 'shared_indexes' / ds).rglob('vdb_chunk.parquet'))
        inputs[ds] = dict(old_result_sha256=sha(BASE / 'cases' / ds / 'full_r060/result.json'),
            historical_result_sha256=sha(source_path),
            passages=dict(path=str(passage), sha256=sha(passage)))
        print(f'FROZEN {ds}: 100 new / 50 excluded; hops={selection["datasets"][ds]["hop_distribution"]}', flush=True)
    write(output / 'selection_manifest.json', selection)
    write(output / 'selection_locked.ok', dict(sha256=digest(selection),
        new_questions=400, combined_questions=600, locked_before_inference=True))
    write(output / 'experiment.json', dict(base=str(BASE), inputs=inputs, relation_weight=.6,
        native_source='Exact original diagnostic snapshot; explicit method R=.60.',
        index_policy='Reuse existing complete shared_indexes; no indexing or corpus embedding.',
        cache_policy='Private copies of historical Full LLM cache; incremental costs, not cold-cache costs.',
        hop_policy='Same historical protocol as the original 50; 2Wiki prior=2 retained for comparability.',
        serial_dataset_order=list(DATASETS)))
    snapshot = output / 'source_snapshot'
    shutil.copytree(BASE / 'source_snapshot/src', snapshot / 'src',
        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'), dirs_exist_ok=True)
    (snapshot / 'scripts').mkdir(parents=True, exist_ok=True)
    shutil.copy2(SCRIPT, snapshot / 'scripts' / SCRIPT.name)
    write(output / 'source_fingerprint.json', {str(p.relative_to(snapshot)): sha(p)
        for p in snapshot.rglob('*.py')})
    (output / 'README.md').write_text(
        '# Full Pack R=0.60 扩展评测\n\n'
        '每个数据集新增100题，与原50题不重复；固定种子20261025，按sample ID散列抽样，不使用历史分数挑题。'
        '原50题按历史结果分层，合并150题仍含诊断集选择偏差。\n\n'
        '仅运行 Full，相关性权重0.60；模型、Reader@5及其余算法参数沿用原诊断快照。'
        '为与原50题合并，保留历史hop协议，2Wiki仍是prior=2，不能作为修复hop后的主实验。\n\n'
        '复用4_datasets_result/shared_indexes，不建库；可写graph/元数据/LLM缓存使用私有副本。'
        '物理请求、token、缓存命中、查询Embedding、时间、资源及Reader预算在runtime下独立记录，属于缓存复用条件下的新增开销。\n\n'
        '每题保存checkpoint，中断后从已完成题后继续；原50题直接引用，不重新推理。'
        '完成后cases/<dataset>/new100和combined150各有result.json、report.json、metrics_details.json；'
        'comparison.md同时列旧50、新100、合并150。HoVer合并Macro-F1由全部预测重新计算。\n\n'
        '日志：`tail -f ' + str(output / 'run.log') + '`\n')


def checkpoint_path(output, ds, qid):
    return output / 'checkpoints' / ds / (hashlib.sha256(str(qid).encode()).hexdigest() + '.json')


def completed(output, ds):
    selected = read(output / 'selection_manifest.json')['datasets'][ds]['rows']
    result = []
    for row in selected:
        file = checkpoint_path(output, ds, row['sample_id'])
        if not file.exists():
            break
        saved = read(file)
        assert saved['sha256'] == digest(saved['row'])
        assert saved['row']['sample_id'] == row['sample_id']
        result.append(saved['row'])
    assert len(list((output / 'checkpoints' / ds).glob('*.json'))) == len(result)
    return result


def status(output, ds, phase='running'):
    counts = {d: len(completed(output, d)) for d in DATASETS}
    write(output / 'status.json', dict(dataset=ds, phase=phase,
        counts=counts, completed=sum(counts.values()), expected=400,
        updated_at=datetime.now().astimezone().isoformat()))


def worker(output, ds, limit=None):
    sys.path.insert(0, str(output / 'source_snapshot/src'))
    sys.path.insert(0, str(BASE / 'source_snapshot/scripts'))
    import numpy as np
    from transformers import AutoTokenizer
    from bindpack import BindPack, BindPackConfig
    from bindpack.evaluation.measurement import Recorder, ResourceMonitor
    from bindpack.evaluation.measured_reader import prepare as prepare_reader
    from bindpack.evaluation.result_export import detailed_result
    from bindpack.llm.openai_gpt import CacheOpenAI
    from bindpack.utils.misc_utils import QuerySolution
    from eval_utils import get_gold_answers, get_gold_docs
    helper = helpers()
    selected = read(output / 'selection_manifest.json')['datasets'][ds]['rows']
    assert digest(read(output / 'selection_manifest.json')) == read(output / 'selection_locked.ok')['sha256']
    rows = completed(output, ds)
    if len(rows) == 100:
        finish(output, ds, rows)
        return
    meta = output / 'metadata' / ds
    data = read(meta / 'benchmark.json')
    recorder = Recorder(output / 'runtime' / ds / 'accounting')
    tokenizer = AutoTokenizer.from_pretrained('/root/models/Qwen3-8B', local_files_only=True)
    recorder.install_llm_factory(CacheOpenAI, tokenizer)
    index = helper.private_index(output, ds)
    cfg = BindPackConfig.from_file(meta / 'full_r060_method.json', save_dir=str(index), stage_ablation='full')
    assert cfg.pack.relation_weight == .6
    monitor = ResourceMonitor(recorder)
    current = {}
    with monitor:
        with recorder.section('model_and_index_loading'):
            rag = BindPack(config=cfg)
            recorder.install_embedding(rag.embedding_model)
            rag.prepare_retrieval_objects()
            rag._build_index_side_artifacts()
        # Index construction is forbidden, including accidental future calls.
        def deny_index(*args, **kwargs):
            raise RuntimeError('Expansion must reuse the existing index')
        rag.index = deny_index
        runtime = rag.evidence_runtime
        native_finalize = runtime.pack.finalize
        def capture(query, ids, scores, context, state):
            frozen = helper.freeze_pack_input(query, ids, scores, state)
            write(current['frozen_file'], dict(input=frozen, input_sha256=digest(frozen)))
            return native_finalize(query, ids, scores, context, state)
        runtime.pack.finalize = capture
        print(f'INDEX READY {ds}: {len(rag.passage_node_keys)} passages reused; resume={len(rows)}/100', flush=True)
        for ordinal in range(len(rows), min(100, limit or 100)):
            chosen = selected[ordinal]
            qid, query = chosen['sample_id'], chosen['question']
            start = time.perf_counter()
            token = hashlib.sha256(str(qid).encode()).hexdigest()
            retrieved_file = output / 'retrieved' / ds / (token + '.json')
            current['frozen_file'] = output / 'frozen_inputs' / ds / (token + '.json')
            if retrieved_file.exists():
                saved = read(retrieved_file)
                assert saved['sample_id'] == qid
                solution = QuerySolution(question=query, docs=saved['docs'], doc_scores=np.asarray(saved['doc_scores']),
                    retrieval_trace=saved['retrieval_trace'])
                retrieval_seconds = saved['retrieval_seconds']
            else:
                with recorder.section('online_retrieval', qid):
                    t = time.perf_counter()
                    rag.set_query_hop_overrides([chosen['benchmark_hops']], source='historical_fixed_for_pack_expansion')
                    solution = rag.retrieve([query], num_to_retrieve=200)[0]
                    retrieval_seconds = time.perf_counter() - t
                write(retrieved_file, helper.jsonable(dict(sample_id=qid, docs=solution.docs,
                    doc_scores=solution.doc_scores, retrieval_trace=solution.retrieval_trace,
                    retrieval_seconds=retrieval_seconds)))
            frozen = read(current['frozen_file'])
            assert frozen['input_sha256'] == digest(frozen['input'])
            with recorder.section('online_reader', qid):
                t = time.perf_counter()
                reading, budget = prepare_reader(rag, solution, tokenizer, ds)
                answered, raw, metas = rag.qa([reading])
                solution.answer = answered[0].answer
                reader_seconds = time.perf_counter() - t
            sample = data[chosen['query_index']]
            solution.gold_answers = get_gold_answers([sample], ds)[0]
            solution.gold_docs = sorted(set(get_gold_docs([sample], ds)[0]))
            export = detailed_result(solution, sample, chosen['query_index'], solution.gold_docs,
                chosen['benchmark_hops'], 10, 200)
            export.update(ranking_complete=True, actual_ranking_length=len(solution.docs),
                pack_relation_weight=.6, expansion_cohort='new100_outcome_blind',
                frozen_pack_input_sha256=frozen['input_sha256'],
                timing=dict(retrieval_seconds=retrieval_seconds, reader_seconds=reader_seconds,
                    end_to_end_seconds=retrieval_seconds + reader_seconds,
                    scope='Observed with reused index and historical LLM cache, excluding initialization.'),
                reader=dict(raw_response=raw[0], metadata=metas[0], budget=budget))
            export = helper.jsonable(export)
            write(checkpoint_path(output, ds, qid), dict(row=export, sha256=digest(export)))
            rows.append(export)
            status(output, ds)
            print(f'QUERY {ds} [{"#" * int(24 * len(rows) / 100):24}] {len(rows)}/100; '
                f'{sum(len(completed(output, d)) for d in DATASETS)}/400; '
                f'{time.perf_counter()-start:.2f}s; FCR5={export["all_gold_in_top5"]}; answer={export["answer"]!r}', flush=True)
    write(output / 'runtime' / ds / f'phase_timings_{os.getpid()}.json', recorder.phases)
    for obj, name, original in reversed(recorder.restorations):
        setattr(obj, name, original)
    if len(rows) == 100:
        finish(output, ds, rows)
    summarize(output)


def finish(output, ds, rows):
    from bindpack.evaluation.metric_report import write_metric_report
    from bindpack.evaluation.measurement import distribution, jsonl, usage
    meta = output / 'metadata' / ds
    identity = read(meta / 'sample_manifest.json')
    method = read(meta / 'full_r060_method.json')
    old = read(BASE / 'cases' / ds / 'full_r060/result.json')
    assert sha(BASE / 'cases' / ds / 'full_r060/result.json') == read(output / 'experiment.json')['inputs'][ds]['old_result_sha256']
    assert len(rows) == 100 and len(old['results']) == 50
    assert not {r['sample_id'] for r in rows} & {r['sample_id'] for r in old['results']}
    for cohort, samples in (('new100', rows), ('combined150', old['results'] + rows)):
        directory = output / 'cases' / ds / cohort
        manifest = deepcopy(identity)
        manifest.update(selected_indices=[r['query_index'] for r in samples],
            sample_ids=[r['sample_id'] for r in samples],
            diagnostic_selection=cohort == 'combined150',
            selection_sha256=digest([r['sample_id'] for r in samples]))
        write(directory / 'sample_manifest.json', manifest)
        # The evaluator resolves the corpus beside its explicit sample manifest.
        corpus_link = directory / 'corpus.json'
        if not corpus_link.exists():
            corpus_link.symlink_to(os.path.relpath(meta / 'corpus.json', directory))
        protocol = dict(stage_ablation='full', relation_weight=.6, cohort=cohort,
            sample_count=len(samples), new_inference_count=100,
            old_results_reused=50 if cohort == 'combined150' else 0,
            selection_uses_outcomes=cohort == 'combined150',
            hop_condition=old['hop_provenance'],
            no_index_build=True, serial_execution=True, reader_top_k=5)
        write(directory / 'protocol.json', protocol)
        result = dict(dataset=ds, method='bindpack', stage_ablation='full', eval_mode='rag_qa',
            data_path=str(meta / 'benchmark.json'), corpus_path=str(meta / 'corpus.json'),
            selected_indices=manifest['selected_indices'], sample_size_effective=len(samples),
            runtime_config=dict(method['runtime'], **{k: method[k] for k in ('pack', 'plan', 'route', 'candidates')}),
            diagnostic_selection=cohort == 'combined150', pack_relation_weight=.6,
            hop_provenance=old['hop_provenance'], results=samples)
        write(directory / 'result.json', result)
        write(directory / 'validated.ok', dict(result_sha256=sha(directory / 'result.json'),
            protocol_sha256=digest(protocol), sample_count=len(samples)))
        report = write_metric_report(directory / 'result.json', data_path=meta / 'benchmark.json',
            manifest_path=directory / 'sample_manifest.json', ks=(3, 5, 8, 10, 20, 200))
        print(f'REPORT {ds}/{cohort}: ' + json.dumps(report['metrics'], ensure_ascii=False), flush=True)
    accounting = output / 'runtime' / ds / 'accounting'
    calls = jsonl(accounting / 'llm_calls.jsonl')
    logical = jsonl(accounting / 'llm_logical_calls.jsonl')
    embeddings = jsonl(accounting / 'embedding_calls.jsonl')
    budgets = [r['reader']['budget'] for r in rows]
    efficiency = dict(scope='New 100 only; reused index and warm historical LLM cache; not cold-run cost.',
        physical=usage(calls), logical_calls=len(logical), cache_hits=sum(bool(c.get('cache_hit')) for c in logical),
        by_phase={p: usage([c for c in calls if c['phase'] == p]) for p in sorted({c['phase'] for c in calls})},
        end_to_end_seconds=distribution([r['timing']['end_to_end_seconds'] for r in rows]),
        retrieval_seconds=distribution([r['timing']['retrieval_seconds'] for r in rows]),
        reader_seconds=distribution([r['timing']['reader_seconds'] for r in rows]),
        evidence_tokens=distribution([b['evidence_tokens'] for b in budgets]),
        reader_prompt_tokens=distribution([b['prompt_tokens_local'] for b in budgets]),
        partial_passages=sum(len(b['partial_passage_ranks']) for b in budgets),
        dropped_passages=sum(len(b['dropped_passage_ranks']) for b in budgets),
        query_embedding_calls=len(embeddings),
        query_embedding_tokens=sum(e['effective_tokens'] for e in embeddings),
        new_index_build_calls=0, new_index_build_seconds=0,
        original_offline_index_cost='N/A: reused existing index; see original provenance',
        resource_samples_path=str(accounting / 'resource_samples.jsonl'))
    write(output / 'runtime' / ds / 'efficiency.json', efficiency)
    write(output / 'cases' / ds / 'completed.ok', dict(new_questions=100, combined_questions=150))


def summarize(output):
    result = []
    lines = ['# Full Pack R=0.60：旧50、新100与合并150', '',
        '每个数据集新增100条，排除原诊断50条，不按结果筛选。旧50条含历史成绩分层，合并150条仍是混合样本；新100条单独列出。分数为0–1小数。', '',
        '| 数据集 | 子集 | n | Recall@5 | FCR@5 | EM / Accuracy | token F1 / Macro-F1 | DCC@5 | NodeCov@5 | ClosureGap@5 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for ds in DATASETS:
        variants = {}
        for cohort, path in (('old50', BASE / 'cases' / ds / 'full_r060/report.json'),
                            ('new100', output / 'cases' / ds / 'new100/report.json'),
                            ('combined150', output / 'cases' / ds / 'combined150/report.json')):
            if not path.exists():
                continue
            m = read(path)['metrics']
            variants[cohort] = dict(path=str(path), metrics=m)
            qa = 'ClassificationAccuracy' if ds == 'hover' else 'ExactMatch'
            f1 = 'ClassificationMacroF1' if ds == 'hover' else 'F1'
            keys = ('Recall@5', 'FCR@5', qa, f1, 'DCC@5', 'NodeCov@5', 'ClosureGap@5')
            n = dict(old50=50, new100=100, combined150=150)[cohort]
            values = ['N/A' if m.get(k) is None else f'{m[k]:.6f}' for k in keys]
            lines.append(f'| {ds} | [{cohort}]({os.path.relpath(path, output)}) | {n} | ' + ' | '.join(values) + ' |')
        result.append(dict(dataset=ds, variants=variants))
    lines += ['', 'HoVer合并Macro-F1按150条原始预测重新计算，不能平均两份Macro-F1。MuSiQue提供可信gold DAG指标，其余无gold DAG的项目为N/A。', '',
        '仅统计新增100条的实际请求、缓存命中、token、查询Embedding、时间、资源及Reader预算，详见runtime/<dataset>/efficiency.json和accounting/。旧50条的重放成本不与新100条拼接为全新端到端成本。', '',
        '索引复用；保留原50条的模型、Reader与hop协议，2Wiki仍为历史prior=2，不能作为修复hop后的正式全量结果。']
    write(output / 'comparison.json', dict(relation_weight=.6, datasets=result))
    (output / 'comparison.md').write_text('\n'.join(lines) + '\n')


def audit(output):
    protocol = read(output / 'experiment.json')
    for ds in DATASETS:
        spec = protocol['inputs'][ds]
        assert sha(spec['passages']['path']) == spec['passages']['sha256']
        assert sha(BASE / 'cases' / ds / 'full_r060/result.json') == spec['old_result_sha256']
        assert len(completed(output, ds)) == 100
        rows = read(output / 'cases' / ds / 'combined150/result.json')['results']
        assert len(rows) == len({r['sample_id'] for r in rows}) == 150
        assert all(row['pack_relation_weight'] == .6 for row in rows)
    for relative, checksum in read(output / 'source_fingerprint.json').items():
        assert sha(output / 'source_snapshot' / relative) == checksum
    write(output / 'audit.json', dict(status='passed', new_questions=400,
        old_questions_reused=200, combined_unique_questions=600,
        source_passage_stores_unchanged=4, original_r060_results_unchanged=4,
        new_index_build_calls=0, source_snapshot_unchanged=True))
    write(output / 'completed.ok', dict(status='complete', new_questions=400,
        combined_questions=600, selection_sha256=read(output / 'selection_locked.ok')['sha256']))
    status(output, None, 'complete')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--worker', choices=DATASETS)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(levelname)s %(message)s')
    if args.worker:
        worker(output, args.worker, args.limit)
        return
    if args.summarize_only:
        summarize(output)
        return
    prepare(output)
    summarize(output)
    if args.prepare_only:
        return
    import fcntl
    with (output / '.run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(output / 'run.pid.json', dict(pid=os.getpid()))
        snapshot_script = output / 'source_snapshot/scripts' / SCRIPT.name
        for ds in DATASETS:
            print(f'START {ds}; preserved={len(completed(output, ds))}/100', flush=True)
            status(output, ds)
            try:
                subprocess.run([sys.executable, '-B', '-u', str(snapshot_script),
                    '--output', str(output), '--worker', ds], check=True)
            except subprocess.CalledProcessError:
                status(output, ds, 'failed_checkpoint_preserved')
                raise
        summarize(output)
        audit(output)
        print('COMPLETE: 400 new predictions; 200 old predictions retained; 600 combined.', flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Read-only four-cell Pack diagnostics and one new Closure-aware Packaging QA arm.

Full, Flat and Singleton are diagnostic replays, never new baseline experiments.
Every replay must match the completed historical ranking and objective before
the new arm can reach the reader. All four cells retain the structured Full S0.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict
import fcntl
import gc
import gzip
import hashlib
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import shutil
import signal
import subprocess
import sys
import threading
import time
from types import MethodType, SimpleNamespace

CODE = Path(__file__).resolve().parents[1]
PROJECT = Path('/root/BindPack')
DEFAULT_SOURCE = PROJECT/'outputs/ablations_redesigned_20261108'
DATASETS = ('hotpotqa', '2wikimultihopqa', 'musique', 'hover')
CELLS = ('full', 'flat_dependency_support', 'singleton_search', 'wo_closure_aware_packaging')
NEW_ARM = CELLS[-1]
LABELS = ('Full', 'Flat Support', 'Singleton Search', 'w/o Closure-aware Packaging')
ENV = {}
TIMING_PROBE_VERSION = 'balanced_observer_off_kernel_v1'


def read(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    if isinstance(value, dict): return {str(k): canonical(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [canonical(v) for v in value]
    if isinstance(value, (set, frozenset)): return [canonical(v) for v in sorted(value)]
    if hasattr(value, 'tolist'): return canonical(value.tolist())
    return value


def digest(value):
    return hashlib.sha256(json.dumps(canonical(value), ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w') as stream:
        json.dump(canonical(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def append(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(canonical(value), ensure_ascii=False, allow_nan=False)+'\n')
        stream.flush(); os.fsync(stream.fileno())


def jsonlines(path):
    path = Path(path)
    if not path.exists(): return
    with path.open() as stream:
        for line in stream:
            if line.strip(): yield json.loads(line)


def write_gzip(path, value):
    """Stream JSON into a separate per-question compressed artifact."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    with gzip.open(temporary, 'wt', encoding='utf-8', compresslevel=3) as stream:
        json.dump(canonical(value), stream, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    temporary.replace(path)
    checksum = sha(path)
    write(path.with_name(path.name+'.sha256.json'), dict(sha256=checksum))
    return checksum


def read_gzip(path):
    path = Path(path)
    if sha(path) != read(path.with_name(path.name+'.sha256.json'))['sha256']:
        raise ValueError('Compressed checkpoint changed: '+str(path))
    with gzip.open(path, 'rt', encoding='utf-8') as stream: return json.load(stream)


def load_upstream(path):
    """Only the user's completed, hash-verified internal checkpoint is loaded."""
    path = Path(path)
    if sha(path) != read(path.with_suffix('.sha256.json'))['sha256']:
        raise ValueError('Historical upstream checkpoint changed: '+str(path))
    with gzip.open(path, 'rb') as stream: return pickle.load(stream)


def parity(ids, trace, historical):
    """Exact old rankings and S0; tolerate only float representation roundoff."""
    diag = trace.get('budgeted_selection', {})
    if list(map(int, ids[:200])) != historical['ids']:
        raise AssertionError('Historical Top200 parity failed: '+historical['arm'])
    for key in ('reference_doc_ids', 'candidate_domain', 'full_package_bank', 'expanded_states'):
        if canonical(diag.get(key)) != historical[key]:
            raise AssertionError('Historical parity failed: '+historical['arm']+'/'+key)
    for key in ('objective', 'reference_objective', 'soft_weight_effective'):
        if abs(float(diag[key])-historical[key]) > 1e-12:
            raise AssertionError('Historical objective parity failed: '+historical['arm']+'/'+key)
    if diag.get('search_deadline_reached'):
        raise AssertionError('Diagnostic replay reached its wall-time limit')


def same_controls(cells):
    common = cells['full']['trace']['budgeted_selection']
    for arm in CELLS[1:]:
        diag = cells[arm]['trace']['budgeted_selection']
        for key in ('reference_doc_ids', 'candidate_domain', 'full_package_bank',
                    'soft_weight_effective', 'reward_loss_weight', 'residual_reward_enabled'):
            if canonical(diag[key]) != canonical(common[key]):
                raise AssertionError('Four-cell controls differ: '+arm+'/'+key)
        if arm in ('singleton_search', NEW_ARM) and diag['package_action_count'] != 0:
            raise AssertionError('Package action leaked into '+arm)
        mode = 'flat' if arm in ('flat_dependency_support', NEW_ARM) else 'ancestor_closed'
        if diag['closure_reward_mode'] != mode: raise AssertionError('Incorrect support mode')


def observer_off_cohort(indices, sample_ids, count=32):
    """Choose an outcome-blind, stable subset using sample IDs alone."""
    pairs = list(zip(indices, sample_ids))
    ordered = sorted(pairs, key=lambda item: (
        hashlib.sha256(str(item[1]).encode()).hexdigest(), str(item[1]), item[0]))[:min(count,len(pairs))]
    return dict(schema=TIMING_PROBE_VERSION, maximum_queries=32,
        selected_indices=[item[0] for item in ordered], sample_ids=[item[1] for item in ordered],
        selection='ascending SHA256(str(sample_id)); no outcomes, support scores, or runtimes used',
        measurement_scope='diagnostic warm selector kernel only; no Route, Plan, index, QA, cold loading, or deployment latency')


def observer_off_probe(prepare_select, historical, sample_id, cache_info):
    """Counterbalance Full/Singleton, with observer allocation outside the probe.

    Each callback is prepared before GC and timing. Two unmeasured warmups cover
    both arms' reader-visible finalists. Every measured result passes the same
    historical parity checks as the instrumented diagnostic cells.
    """
    reverse = int(hashlib.sha256(str(sample_id).encode()).hexdigest()[-1],16)%2
    order = (('singleton_search','full','full','singleton_search') if reverse else
             ('full','singleton_search','singleton_search','full'))
    warmups = []
    for arm in ('full','singleton_search'):
        select = prepare_select(arm); gc.collect()
        ids, _, trace = select(); parity(ids,trace,historical[arm])
        if trace['budgeted_selection'].get('pack_diagnostics'):
            raise AssertionError('Timing warmup unexpectedly enabled instrumentation')
        warmups.append(dict(arm=arm,parity_passed=True))
    measured = []
    for position,arm in enumerate(order):
        select = prepare_select(arm)
        gc.collect()
        before = cache_info()
        wall_started,cpu_started = time.perf_counter(),time.process_time()
        ids,_,trace = select()
        cpu_seconds,wall_seconds = time.process_time()-cpu_started,time.perf_counter()-wall_started
        after = cache_info()
        parity(ids,trace,historical[arm])
        diag = trace['budgeted_selection']
        if diag.get('pack_diagnostics'): raise AssertionError('Timing probe enabled instrumentation')
        misses = {key:after[key]-before[key] for key in before}
        measured.append(dict(position=position,arm=arm,wall_seconds=wall_seconds,
            process_cpu_seconds=cpu_seconds,native_selector_seconds=diag['seconds'],
            parity_passed=True,expanded_states=diag['expanded_states'],
            shared_cache_misses=misses,fully_warm_shared_caches=not any(misses.values())))
    return dict(schema=TIMING_PROBE_VERSION,version=TIMING_PROBE_VERSION,
        observer_enabled=False,order=list(order),
        warmups=warmups,measurements=measured,parity_passed=True,
        fully_warm_shared_caches=all(row['fully_warm_shared_caches'] for row in measured),
        scope='matched warm CPU selector kernel, counterbalanced F-S-S-F / S-F-F-S; not cold or end-to-end latency',
        exclusions='state/runtime preparation, gc.collect, parity validation and recording are outside measured intervals; '
                   'numeric residual and reader-visible prompt caches are shared and warmed for both arms')


def init_cpu(config, parquet, root, dataset):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    sys.path[:0] = [str(CODE/'src'), str(CODE/'scripts')]
    import torch
    torch.set_num_threads(1)
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from bindpack import BindPackConfig
    from bindpack.prompts.prompt_template_manager import PromptTemplateManager
    table = pq.read_table(parquet, columns=['content'])
    ENV.update(config=config, texts=table['content'].to_pylist(), root=Path(root), dataset=dataset,
        tokenizer=AutoTokenizer.from_pretrained('/root/models/Qwen3-8B', local_files_only=True),
        reader=SimpleNamespace(prompt_template_manager=PromptTemplateManager()), configs={})
    for arm in CELLS:
        values = deepcopy(config); values.update(mechanism_ablation='full',
            pack_mechanism_ablation='none' if arm == 'full' else arm)
        ENV['configs'][arm] = BindPackConfig.from_file(**values)


def replay_query(job):
    """Sequential cells share numerical residual features and reader preparation."""
    import numpy as np
    from bindpack.stages.pack.stage import PackStage
    from bindpack.stages.pack import budgeted_selector
    from bindpack.evaluation.measured_reader import prepare
    from bindpack.utils.misc_utils import QuerySolution
    identity = job['identity']; index = identity['query_index']; ds = ENV['dataset']
    cached = ENV['root']/'replays'/ds/f'{index:04d}.json.gz'
    if cached.exists() and cached.with_name(cached.name+'.sha256.json').exists():
        record = read_gzip(cached)
        if record['identity'] != identity: raise ValueError('Replay checkpoint identity changed')
        diagnostic = ENV['root']/record['diagnostic']['path']
        if sha(diagnostic) != record['diagnostic']['sha256']:
            raise ValueError('Diagnostic checkpoint changed')
        return record
    payload, reference = job['payload'], job['reference']
    for arm in CELLS[1:3]:
        if job['historical'][arm].get('upstream_checkpoint_sha256') != identity['upstream_sha256']:
            raise AssertionError('Historical Pack cell used a different Full upstream checkpoint: '+arm)
    prepared, residual_cache = {}, {}
    cache_misses = dict(residual=0,reader_visible=0)
    original_residual = budgeted_selector.residual_features
    def residual(question, texts):
        key = (question, tuple(texts))
        if key not in residual_cache:
            residual_cache[key] = original_residual(question, texts)
            cache_misses['residual'] += 1
        return residual_cache[key]
    def visible(question, ordered):
        key = (question, tuple(ordered))
        if key not in prepared:
            cache_misses['reader_visible'] += 1
            solution = QuerySolution(question=question, docs=[ENV['texts'][int(d)] for d in ordered],
                doc_scores=np.ones(len(ordered)))
            reading, budget = prepare(ENV['reader'], solution, ENV['tokenizer'], ds,
                k=5, context_window=8192, output_tokens=2048)
            prepared[key] = (dict(zip(ordered, reading.docs)), budget)
        return prepared[key]
    budgeted_selector.residual_features = residual
    cells, diagnostics = {}, {}
    observer_off_timing = None
    started = time.perf_counter()
    try:
        if job.get('timing_probe',False):
            def prepare_select(arm):
                runtime=SimpleNamespace(cfg=ENV['configs'][arm],rag=ENV['reader'],
                    _document=lambda doc_id:ENV['texts'][int(doc_id)],_trace=lambda:{},
                    _pack_reference=deepcopy(reference),_joint_visible=visible)
                selector=PackStage(runtime)
                ids,scores,context,state=(deepcopy(payload[key]) for key in ('ids','scores','context','state'))
                return lambda:selector.finalize(payload['query'],ids,scores,context,state)
            observer_off_timing=observer_off_probe(prepare_select,job['historical'],
                identity['sample_id'],lambda:dict(cache_misses))
            observer_off_timing.update(cohort_sha256=identity['timing_cohort_sha256'],
                sample_id=identity['sample_id'],dataset=ds,query_index=index)
        for arm in CELLS:
            cfg = ENV['configs'][arm]
            runtime = SimpleNamespace(cfg=cfg, rag=ENV['reader'],
                _document=lambda doc_id: ENV['texts'][int(doc_id)], _trace=lambda: {},
                _pack_reference=deepcopy(reference), _joint_visible=visible,
                _pack_diagnostics={'capture_score_rows': True, 'representative_limit': 12})
            tick = time.perf_counter()
            ids, scores, trace = PackStage(runtime).finalize(payload['query'],
                deepcopy(payload['ids']), deepcopy(payload['scores']), deepcopy(payload['context']),
                deepcopy(payload['state']))
            seconds = time.perf_counter()-tick
            if arm != NEW_ARM: parity(ids, trace, job['historical'][arm])
            if set(map(int, ids[:200])) != set(map(int, reference[0][:200])):
                raise AssertionError('Candidate ranking domain changed')
            diagnostics[arm] = canonical(runtime._pack_diagnostics)
            if diagnostics[arm].get('schema_version') != 1:
                raise AssertionError('Core diagnostic interface is unavailable')
            cells[arm] = dict(ids=list(map(int, ids[:200])), scores=list(map(float, scores[:200])),
                trace=canonical(trace), seconds=seconds, historical_parity_passed=arm != NEW_ARM,
                historical_native_selector_seconds=job['historical'][arm].get('native_selector_seconds')
                    if arm != NEW_ARM else None)
        same_controls(cells)
    finally:
        budgeted_selector.residual_features = original_residual
    diagnostic_path = ENV['root']/'diagnostics'/ds/f'{index:04d}.json.gz'
    checksum = write_gzip(diagnostic_path, dict(identity=identity, cells=diagnostics,
        historical_parity_passed=True, shared_structured_s0=True,
        observer_off_timing=observer_off_timing,
        numerical_residual_computations=len(residual_cache), shared_reader_preparations=len(prepared)))
    # Full score rows live only in the compressed diagnostic, never result.json.
    summaries = {a: {k: v for k, v in d.items() if k != 'score_rows'} for a, d in diagnostics.items()}
    for arm in CELLS[:3]:
        trace = cells[arm]['trace']
        cells[arm]['trace'] = dict(budgeted_selection={key: trace['budgeted_selection'][key] for key in (
            'objective','reference_objective','reference_doc_ids','candidate_domain','full_package_bank',
            'soft_weight_effective','expanded_states','package_action_count','closure_reward_mode')})
    record = canonical(dict(identity=identity, cells=cells, diagnostics_summary=summaries,
        observer_off_timing=observer_off_timing,
        seconds=time.perf_counter()-started, diagnostic=dict(path=str(diagnostic_path.relative_to(ENV['root'])),
            sha256=checksum), shared_structured_s0=True))
    write_gzip(cached, record)
    return record


def generation_identity(config):
    return {k: config.get(k) for k in ('llm_name', 'llm_base_url', 'temperature', 'seed',
        'num_gen_choices', 'max_new_tokens', 'response_format', 'azure_endpoint')}


def reusable_full_response(full_row, budget, original_config, new_config):
    return (generation_identity(original_config) == generation_identity(new_config)
        and budget['prompt_sha256'] == full_row['reader']['budget']['prompt_sha256']
        and budget['context_window'] == full_row['reader']['budget']['context_window']
        and budget['reserved_output_tokens'] == full_row['reader']['budget']['reserved_output_tokens'])


def interaction_metrics(metrics):
    keys = set.intersection(*(set(metrics[a]) for a in CELLS))
    return {key: (100*(metrics['full'][key]-metrics['flat_dependency_support'][key]
        -metrics['singleton_search'][key]+metrics[NEW_ARM][key]))
        if all(metrics[a][key] is not None for a in CELLS) else None for key in sorted(keys)}


def parse_inclusions(value):
    result = {ds: set() for ds in DATASETS}
    for token in filter(None, value.split(',')):
        dataset, separator, index = token.strip().partition(':')
        if not separator or dataset not in result or not index.isdigit():
            raise ValueError('Use --include-indices musique:567,hover:317')
        result[dataset].add(int(index))
    return result


def source_rows(source, ds, selected, text_to_id):
    historical, full_rows, subset_rows = {}, {}, {}
    canonical_texts = {text: text for text in text_to_id}
    for arm in CELLS[:3]:
        case = source/'cases'/ds/arm; done = read(case/'completed.ok')
        for name in ('result', 'report'):
            if sha(case/(name+'.json')) != done[name+'_sha256']:
                raise ValueError('Historical completed artifact changed: '+str(case/name))
        selected_rows, seen = [], set()
        for event in jsonlines(case/'progress.jsonl'):
            row = event['row']; index = row['query_index']
            if digest(row) != event['sha256']: raise ValueError('Historical row checksum differs')
            if index not in selected: continue
            if index in seen: raise ValueError('Duplicate historical row')
            seen.add(index)
            diag = row['retrieval_trace']['evidence']['budgeted_selection']
            if diag.get('search_deadline_reached'): raise ValueError('Historical search deadline affected parity')
            slim = dict(arm=arm, ids=[text_to_id[d] for d in row['candidate_docs']],
                native_selector_seconds=float(diag.get('seconds',0.)),
                upstream_checkpoint_sha256=row['retrieval_trace'].get('upstream_checkpoint_sha256'),
                **{key: canonical(diag[key]) for key in ('reference_doc_ids', 'candidate_domain',
                    'full_package_bank', 'expanded_states', 'objective', 'reference_objective', 'soft_weight_effective')})
            historical.setdefault(index, {})[arm] = slim
            metric_row = {key: row[key] for key in ('question','query_index','sample_id','answer',
                'gold_answers','gold_docs','gold_document_ranks','ranking_complete','actual_ranking_length')}
            metric_row['candidate_docs'] = [canonical_texts[d] for d in row['candidate_docs']]
            metric_row['docs'] = metric_row['candidate_docs'][:10]
            selected_rows.append(metric_row)
            if arm == 'full':
                full_rows[index] = {key: row[key] for key in ('question','query_index','sample_id','answer',
                    'gold_answers','gold_docs','benchmark_hops','reader')}
                full_rows[index]['_source_row_sha256'] = event['sha256']
        if {r['query_index'] for r in selected_rows} != selected: raise ValueError('Missing historical samples')
        subset_rows[arm] = selected_rows
    return historical, full_rows, subset_rows


def verify_receipt(receipt, parquet):
    source = Path(receipt['source'])
    if source.parent.name != 'shared_indexes': raise ValueError('Unexpected shared index source')
    relative = str(parquet.relative_to(source)); expected = receipt['assets'][relative]
    if parquet.is_symlink() or parquet.stat().st_size != expected['bytes'] or sha(parquet) != expected['sha256']:
        raise ValueError('Shared passage index changed')


def diagnostic_aggregate(records):
    records=list(records)
    out = {arm: dict(queries=0, score_comparison={}, package_totals={}, action_totals={},
        search_totals={},chosen_path_counts={},support={},weights={},timing={},
        same_top5_as_full=0, changed_objective_from_full=0) for arm in CELLS}
    values_by_arm = {arm: {} for arm in CELLS}
    for record in records:
        for arm in CELLS:
            dest = out[arm]; dest['queries'] += 1
            cell = record['cells'][arm]; full = record['cells']['full']
            dest['same_top5_as_full'] += cell['ids'][:5] == full['ids'][:5]
            dest['changed_objective_from_full'] += abs(cell['trace']['budgeted_selection']['objective']
                -full['trace']['budgeted_selection']['objective']) > 1e-12
            detail = record['diagnostics_summary'][arm]
            series = values_by_arm[arm]
            for key,value in detail.get('weights',{}).items():
                if type(value) in (int,float): series.setdefault('weights:'+key,[]).append(value)
            for key,value in detail.get('timings',{}).items():
                if type(value) in (int,float): series.setdefault('timing:'+key,[]).append(value)
            native = cell.get('historical_native_selector_seconds')
            if native is not None: series.setdefault('timing:historical_native_selector_seconds',[]).append(native)
            for phase,key in (('reference','reference_scores'),('chosen','chosen_scores')):
                scores = detail.get(key)
                if scores:
                    support=dest['support'].setdefault(phase,dict(queries=0,closed_one=0,flat_one=0,
                        equal_support=0,different_objectives=0,representatives=[]))
                    support['queries']+=1
                    support['closed_one']+=abs(scores['C_closed']-1.)<1e-12
                    support['flat_one']+=abs(scores['C_flat']-1.)<1e-12
                    support['equal_support']+=abs(scores['C_closed']-scores['C_flat'])<1e-12
                    support['different_objectives']+=abs(scores['J_full']-scores['J_flat'])>1e-12
                    for field in ('C_closed','C_flat','J_full','J_flat','used_objective'):
                        series.setdefault('support:'+phase+':'+field,[]).append(scores[field])
                    example=dict(query_index=record.get('identity',{}).get('query_index'),
                        sample_id=record.get('identity',{}).get('sample_id'),**scores)
                    support['representatives']=sorted(support['representatives']+[example],
                        key=lambda r:(-abs(r['J_flat']-r['J_full']),-abs(r['C_flat']-r['C_closed'])))[:12]
            for stage, values in detail.get('score_comparison', {}).items():
                stage_out = dest['score_comparison'].setdefault(stage, dict(count=0,
                    different_support_count=0,different_objective_count=0,
                    max_abs_delta_support=0.,max_abs_delta_objective=0.,representatives=[],by_cardinality={}))
                for key in ('count','different_support_count','different_objective_count'):
                    stage_out[key] += values.get(key,0)
                for key in ('max_abs_delta_support','max_abs_delta_objective'):
                    stage_out[key] = max(stage_out[key],values.get(key,0.))
                for size, group in values.get('by_cardinality',{}).items():
                    target = stage_out['by_cardinality'].setdefault(size, dict(count=0,
                        different_support_count=0,different_objective_count=0,max_abs_delta_objective=0.))
                    for key in ('count','different_support_count','different_objective_count'):
                        target[key] += group.get(key,0)
                    target['max_abs_delta_objective'] = max(target['max_abs_delta_objective'],
                        group.get('max_abs_delta_objective',0.))
                examples=[dict(query_index=record.get('identity',{}).get('query_index'),
                    sample_id=record.get('identity',{}).get('sample_id'),**r)
                    for r in values.get('representatives',[])]
                stage_out['representatives'] = sorted(stage_out['representatives']+examples,
                    key=lambda r:-abs(r.get('delta_J_flat_minus_full',r.get('J_flat',0)-r.get('J_full',0))))[:12]
            search = detail.get('search', {})
            for key in ('frontier_update_count','unique_scored_states','expanded_states','finalist_count',
                        'reader_visible_evaluation_count','package_first_discovery_finalists',
                        'observed_package_path_finalists','chosen_complete_bank_packages',
                        'chosen_complete_multi_passage_bank_packages','reference_complete_multi_passage_bank_packages'):
                dest['search_totals'][key]=dest['search_totals'].get(key,0)+search.get(key,0)
            path=search.get('chosen_path',{})
            for key,value in dict(package_first_discovery=path.get('package_steps',0)>0,
                    multi_passage_package_first_discovery=path.get('multi_passage_package_steps',0)>0,
                    observed_package_path=path.get('observed_package_path',False),
                    observed_multi_passage_package_path=path.get('observed_multi_passage_package_path',False),
                    observed_multi_passage_growth_path=path.get('observed_multi_passage_growth_path',False),
                    chosen_equals_reference=search.get('chosen_equals_reference',False),
                    contains_multi_passage_bank_package=search.get('chosen_complete_multi_passage_bank_packages',0)>0).items():
                dest['chosen_path_counts'][key]=dest['chosen_path_counts'].get(key,0)+int(value)
            for name, value in search.get('packages', {}).items():
                if type(value) in (int, float): dest['package_totals'][name] = dest['package_totals'].get(name, 0)+value
                elif isinstance(value,dict):
                    target = dest['package_totals'].setdefault(name,{})
                    for kind, count in value.items():
                        if type(count) in (int,float): target[kind] = target.get(kind,0)+count
            for kind, counts in search.get('actions', {}).items():
                target = dest['action_totals'].setdefault(kind, {})
                for name, value in counts.items():
                    if type(value) in (int, float): target[name] = target.get(name, 0)+value
    for arm, series in values_by_arm.items():
        for key,values in series.items():
            group,name=key.split(':',1)
            if group=='support': out[arm]['support'][name.split(':')[0]][name.split(':')[1]+'_distribution']=distribution(values)
            else: out[arm][group][name]=distribution(values)
        out[arm]['timing_scope']=(
            'bounded_search_seconds and package_generation/package_bank_ranking exclude observer work; '
            'feature_setup_seconds and algorithm_seconds are affected by shared residual/prompt caches '
            'with Full evaluated first and cannot alone establish an engineering speedup. '
            'historical_native_selector_seconds is the preserved original total, measured separately.')
        out[arm]['score_stage_scope']=(
            'package_bank_control_full deliberately ranks the common frozen bank with Full scoring; '
            'initial/frontier/expansion/reader_visible use the arm-specific objective. '
            'partial and full candidate cardinalities are reported separately.')
    timing_probe=aggregate_observer_off_timing(records)
    for arm in CELLS:
        out[arm]['observer_off_timing']=timing_probe.get(arm,dict(sample_count=0,measurement_count=0,
            scope='Only Full and Singleton have the independent observer-off timing probe'))
    return out


def distribution(values):
    values=sorted(float(v) for v in values)
    if not values: return dict(n=0,mean=None,p50=None,p95=None,min=None,max=None,total=0.)
    def quantile(p):
        at=(len(values)-1)*p;lo=int(at);hi=min(lo+1,len(values)-1)
        return values[lo]+(values[hi]-values[lo])*(at-lo)
    return dict(n=len(values),mean=sum(values)/len(values),p50=quantile(.5),p95=quantile(.95),
        min=values[0],max=values[-1],total=sum(values))


def aggregate_observer_off_timing(records):
    arms=('full','singleton_search')
    out={arm:dict(sample_count=0,measurement_count=0,parity_pass_count=0,
        fully_warm_measurement_count=0,shared_cache_miss_measurement_count=0,order_counts={}) for arm in arms}
    series={arm:{} for arm in arms}
    for record in records:
        probe=record.get('observer_off_timing')
        if not probe: continue
        if not probe['parity_passed'] or probe['observer_enabled']:
            raise AssertionError('Invalid independent timing-probe checkpoint')
        measurements=probe['measurements']
        if len(measurements)!=4 or any(not row['parity_passed'] for row in measurements):
            raise AssertionError('Incomplete historical parity in timing probe')
        by_arm={arm:[row for row in measurements if row['arm']==arm] for arm in arms}
        if any(len(group)!=2 for group in by_arm.values()): raise AssertionError('Timing order is not balanced')
        order='-'.join('F' if arm=='full' else 'S' for arm in probe['order'])
        all_warm=all(row['fully_warm_shared_caches'] for row in measurements)
        for arm in arms:
            dest=out[arm];values=series[arm];dest['sample_count']+=1
            dest['order_counts'][order]=dest['order_counts'].get(order,0)+1
            for row in by_arm[arm]:
                dest['measurement_count']+=1;dest['parity_pass_count']+=1
                warm=row['fully_warm_shared_caches']
                dest['fully_warm_measurement_count']+=int(warm)
                dest['shared_cache_miss_measurement_count']+=int(not warm)
                for clock in ('wall_seconds','process_cpu_seconds','native_selector_seconds'):
                    values.setdefault(clock,[]).append(row[clock])
                    if warm: values.setdefault('fully_warm_'+clock,[]).append(row[clock])
            if all_warm:
                for clock in ('wall_seconds','process_cpu_seconds','native_selector_seconds'):
                    full=sum(row[clock] for row in by_arm['full'])/2
                    single=sum(row[clock] for row in by_arm['singleton_search'])/2
                    values.setdefault('paired_full_minus_singleton_'+clock,[]).append(full-single)
        for arm in arms:
            out[arm]['fully_warm_sample_count']=out[arm].get('fully_warm_sample_count',0)+int(all_warm)
    for arm in arms:
        for clock in ('wall_seconds','process_cpu_seconds','native_selector_seconds'):
            for name in (clock,'fully_warm_'+clock,'paired_full_minus_singleton_'+clock):
                out[arm][name]=distribution(series[arm].get(name,[]))
        out[arm]['scope']=(
            'Counterbalanced matched warm selector kernel with instrumentation disabled. '
            'Full and Singleton each receive an unmeasured warmup; state/runtime preparation, '
            'gc.collect and parity checks are outside clocks. Residual/prompt caches are shared. '
            'Measurements with any shared-cache miss remain visible and are excluded from fully_warm_* '
            'and fully-warm paired contrasts. These are not cold, deployed, or end-to-end RAG latencies.')
    return out


def worker(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['BINDPACK_LLM_MAX_IN_FLIGHT'] = str(args.reader_workers)
    sys.path[:0] = [str(CODE/'src'), str(CODE/'scripts')]
    import numpy as np
    import pyarrow.parquet as pq
    from bindpack import BindPackConfig
    from bindpack.core import base as base_module
    from bindpack.evaluation.measurement import Recorder, ResourceMonitor, usage
    from bindpack.evaluation.measured_reader import prepare
    from bindpack.evaluation.result_export import detailed_result
    from bindpack.evaluation.metric_report import write_metric_report, _qa
    from bindpack.evaluation.structural_metrics import evaluate_structural_metrics
    from bindpack.llm.openai_gpt import CacheOpenAI
    from bindpack.prompts.prompt_template_manager import PromptTemplateManager
    from bindpack.utils.misc_utils import QuerySolution
    from transformers import AutoTokenizer
    root, source, ds = args.output.resolve(), args.source.resolve(), args.dataset
    meta = root/'metadata'/ds; meta.mkdir(parents=True, exist_ok=True)
    old_meta = source/'metadata'/ds
    identity = read(old_meta/'sample_manifest.json')
    original_indices = identity['selected_indices'];original_ids=dict(zip(original_indices,identity['sample_ids']))
    included = parse_inclusions(args.include_indices)[ds]
    if not included <= set(original_indices): raise ValueError('Included development sample is out of range')
    wanted = set(original_indices[:args.limit or None]) | included
    indices = [i for i in original_indices if i in wanted]
    identity = dict(identity, selected_indices=indices, sample_ids=[original_ids[i] for i in indices],
        development_fixture=bool(args.limit or included), source_manifest_sha256=sha(old_meta/'sample_manifest.json'))
    for name in ('benchmark.json', 'corpus.json'):
        target = meta/name
        if target.exists() and sha(target) != sha(old_meta/name): raise ValueError('Frozen benchmark changed')
        if not target.exists(): shutil.copy2(old_meta/name, target)
    identity['data_path'] = str(meta/'benchmark.json')
    if (meta/'sample_manifest.json').exists() and read(meta/'sample_manifest.json') != identity:
        raise ValueError('Frozen selected samples changed')
    write(meta/'sample_manifest.json', identity)
    cohort=observer_off_cohort(indices,identity['sample_ids'])
    cohort_path=meta/'observer_off_timing_cohort.json'
    if cohort_path.exists() and read(cohort_path)!=cohort:
        raise ValueError('Frozen observer-off timing cohort changed')
    write(cohort_path,cohort);timing_indices=set(cohort['selected_indices'])
    data, corpus = read(meta/'benchmark.json'), read(meta/'corpus.json')
    old_cfg = read(source/'cases'/ds/'full/protocol.json')['configuration']
    values = dict(old_cfg, save_dir=str(root/'runtime'/ds), mechanism_ablation='full',
        pack_mechanism_ablation=NEW_ARM)
    cfg = BindPackConfig.from_file(**values)
    if cfg.qa_top_k != 5 or cfg.max_new_tokens != 2048 or cfg.pack.review_mode != 'off':
        raise ValueError('Reader protocol differs from completed Full')
    receipt = read(old_meta/'index_receipt.json')
    index_source = Path(receipt['source']); model_dir = receipt['model_directory']
    parquet = index_source/model_dir/'chunk_embeddings/vdb_chunk.parquet'
    verify_receipt(receipt, parquet)
    texts = pq.read_table(parquet, columns=['content'])['content'].to_pylist()
    if texts != corpus: raise ValueError('Shared passage order differs from frozen corpus')
    historical, full_rows, baseline_rows = source_rows(source, ds, set(indices), dict(zip(texts, range(len(texts)))))
    case = root/'cases'/ds/NEW_ARM; case.mkdir(parents=True, exist_ok=True)
    protocol = dict(configuration=asdict(cfg), dataset=identity, eval_mode='rag_qa', reader_top_k=5,
        output_tokens=2048, context_window=8192, upstream_reexecuted=False, index_rebuilt=False,
        historical_cases='read-only Full/Flat/Singleton; diagnostic replays only', source=str(source),
        shared_reference='same structured Full S0 retains the original DAG-aware reference pipeline',
        reader_workers=args.reader_workers, cpu_workers=args.workers,
        observer_off_timing_probe=dict(schema=TIMING_PROBE_VERSION,maximum_queries=32,
            cohort_sha256=digest(cohort),observer_enabled=False,
            scope='extra diagnostic kernel measurements only; original formal results are reused'),
        qa_reuse='completed Full only; exact prepared prompt SHA and generation settings; original raw/meta preserved')
    if (case/'protocol.json').exists() and read(case/'protocol.json') != protocol: raise ValueError('Case protocol changed')
    write(case/'protocol.json', protocol)
    rows = []
    for event in jsonlines(case/'progress.jsonl'):
        row = event['row']; position = len(rows)
        if digest(row) != event['sha256'] or position >= len(indices) or row['query_index'] != indices[position]:
            raise ValueError('Invalid new-arm row checkpoint')
        if row['sample_id'] != identity['sample_ids'][position]: raise ValueError('Sample identity changed')
        rows.append(row)
    if (case/'completed.ok').exists():
        done = read(case/'completed.ok')
        if len(rows) != len(indices) or any(sha(case/(n+'.json')) != done[n+'_sha256'] for n in ('result','report')):
            raise ValueError('Completed new-arm artifacts changed')
        print('ALREADY COMPLETE '+ds, flush=True); return
    recorder = Recorder(root/'measurements'/ds)
    tokenizer = AutoTokenizer.from_pretrained('/root/models/Qwen3-8B', local_files_only=True)
    reader = SimpleNamespace(global_config=cfg, prompt_template_manager=PromptTemplateManager())
    llm = CacheOpenAI(cache_dir=str(root/'runtime'/ds/'llm_cache'), global_config=cfg)
    recorder.install_llm(llm, tokenizer)
    reader.llm_model = llm; reader.qa = MethodType(base_module.BaseRAG.qa, reader)
    # Progress is logged once per durable window; native parsing/prompt code stays intact.
    base_module.tqdm = lambda items, *unused, **kw: items
    stop = threading.Event(); started = time.perf_counter(); status = 'failed'
    current = {'phase': 'pack_diagnostics', 'completed': len(rows)}
    for number in (signal.SIGTERM, signal.SIGINT): signal.signal(number, lambda *_: stop.set())
    def heartbeat():
        while not stop.wait(30):
            write(root/'worker_status.json', dict(dataset=ds, pid=os.getpid(), **current,
                elapsed_seconds=time.perf_counter()-started))
            print('HEARTBEAT '+ds+' '+json.dumps(current), flush=True)
    thread = threading.Thread(target=heartbeat, daemon=True); thread.start()
    monitor = ResourceMonitor(recorder)
    all_records = []; window_times = list(jsonlines(root/'measurements'/ds/'windows.jsonl'))
    selected = set(indices)
    try:
        with monitor, ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
                initializer=init_cpu, initargs=(old_cfg, str(parquet), str(root), ds)) as pool:
            paths = sorted((source/'upstream'/ds).glob('[0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9].pkl.gz'))
            for path in paths:
                lo, hi = map(int, path.name.split('.')[0].split('_'))
                wanted = [i for i in indices if i in set(read(source/'metadata'/ds/'sample_manifest.json')['selected_indices'][lo:hi])]
                if not wanted: continue
                if stop.is_set(): raise KeyboardInterrupt('Durable checkpoints retained')
                bundle = load_upstream(path); source_sha = sha(path)
                jobs = []
                for index, question in zip(bundle['indices'], bundle['questions']):
                    if index not in selected: continue
                    if question != full_rows[index]['question']: raise ValueError('Historical question differs')
                    jobs.append(dict(identity=dict(dataset=ds, query_index=index,
                        sample_id=full_rows[index]['sample_id'], upstream_path=str(path), upstream_sha256=source_sha,
                        timing_probe=index in timing_indices,timing_probe_schema=TIMING_PROBE_VERSION,
                        timing_cohort_sha256=digest(cohort),
                        historical_sha256=digest(historical[index])), payload=bundle['pack_inputs'][question],
                        reference=bundle['references'][question], historical=historical[index],
                        timing_probe=index in timing_indices))
                current.update(phase='pack_diagnostics', window_start=lo)
                tick = time.perf_counter()
                records = list(pool.map(replay_query, jobs, chunksize=1)); all_records.extend(records)
                pack_wall = time.perf_counter()-tick
                current['phase'] = 'reader'
                ready = {}; tick = time.perf_counter()
                def answer(record):
                    index = record['identity']['query_index']; old = full_rows[index]
                    cell = record['cells'][NEW_ARM]
                    solution = QuerySolution(question=old['question'], docs=[texts[d] for d in cell['ids']],
                        doc_scores=np.asarray(cell['scores']), retrieval_trace=dict(evidence=cell['trace'],
                            reused_frozen_full_upstream=True, diagnostic=record['diagnostic']))
                    with recorder.thread_section('online_reader', old['sample_id']):
                        began = time.perf_counter()
                        reading, budget = prepare(reader, solution, tokenizer, ds)
                        reuse = reusable_full_response(old, budget, old_cfg, asdict(cfg))
                        if reuse:
                            response = deepcopy(old['reader']); solution.answer = old['answer']
                            provenance = dict(kind='exact_completed_full_prompt', source=str(source/'cases'/ds/'full/progress.jsonl'),
                                source_row_sha256=old['_source_row_sha256'], prompt_sha256=budget['prompt_sha256'],
                                generation_identity=generation_identity(old_cfg), new_llm_calls=0)
                            append(root/'measurements'/ds/'qa_reuse.jsonl', dict(sample_id=old['sample_id'], **provenance))
                        else:
                            answered, raw, metadata = reader.qa([reading]); solution.answer = answered[0].answer
                            response = dict(raw_response=raw[0], metadata=metadata[0])
                            provenance = dict(kind='native_reader', prompt_sha256=budget['prompt_sha256'],
                                generation_identity=generation_identity(asdict(cfg)),
                                physical_usage='see llm_calls.jsonl; exact native response cache hits remain separately recorded')
                        seconds = time.perf_counter()-began
                    solution.gold_answers = old['gold_answers']; solution.gold_docs = old['gold_docs']
                    response.update(budget=budget, provenance=provenance)
                    row = detailed_result(solution, data[index], index, old['gold_docs'], old['benchmark_hops'], 10, 200)
                    row.update(reader=response, ranking_complete=True, actual_ranking_length=len(solution.docs),
                        timing=dict(reader_seconds=seconds, cpu_diagnostic_seconds=record['seconds'],
                            scope='four-cell diagnostic replay is analysis; only the combined arm executes new QA'))
                    return canonical(row)
                missing = [record for record in records if record['identity']['query_index'] not in {r['query_index'] for r in rows}]
                with ThreadPoolExecutor(max_workers=args.reader_workers) as readers:
                    futures = [readers.submit(answer, record) for record in missing]
                    for future in as_completed(futures):
                        row = future.result(); ready[row['query_index']] = row
                        while len(rows) < len(indices) and indices[len(rows)] in ready:
                            row = ready.pop(indices[len(rows)])
                            append(case/'progress.jsonl', dict(sha256=digest(row), row=row)); rows.append(row)
                            current['completed'] = len(rows)
                timing = dict(window_start=lo, window_end=hi, queries=len(records), cpu_diagnostic_wall_seconds=pack_wall,
                    new_reader_window_seconds=time.perf_counter()-tick, completed=len(rows))
                append(root/'measurements'/ds/'windows.jsonl', timing); window_times.append(timing)
                write(root/'diagnostics'/ds/'aggregate.json', diagnostic_aggregate(all_records))
                from report_closure_interaction import refresh
                refresh(root)
                print('WINDOW COMPLETE '+ds+' '+json.dumps(timing), flush=True)
            if len(rows) != len(indices): raise ValueError('Some selected samples lack frozen upstream windows')
            resources = monitor.summary()
        verify_receipt(receipt, parquet)
        result = dict(dataset=ds, method='bindpack', stage_ablation='full', mechanism_ablation='full',
            pack_mechanism_ablation=NEW_ARM, variant_name=LABELS[-1], eval_mode='rag_qa', qa_executed=True,
            data_path=str(meta/'benchmark.json'), corpus_path=str(meta/'corpus.json'),
            selected_indices=indices, sample_size_effective=len(rows), runtime_config=asdict(cfg), results=rows,
            hop_provenance=identity.get('hop_provenance'), upstream_reexecuted=False, index_rebuilt=False,
            shared_reference='original structured S0; this ablation does not remove all DAG mechanisms')
        write(case/'result.json', result)
        write(case/'validated.ok', dict(result_sha256=sha(case/'result.json'), protocol_sha256=digest(protocol)))
        report = write_metric_report(case/'result.json', manifest_path=meta/'sample_manifest.json', ks=[5,10,20,200])
        attempts = list(jsonlines(root/'measurements'/ds/'llm_calls.jsonl'))
        report.update(variant_name=LABELS[-1], index_rebuilt=False, upstream_reexecuted=False,
            cost=dict(physical_new_reader=usage(attempts), exact_full_qa_reused=sum(
                r['reader']['provenance']['kind']=='exact_completed_full_prompt' for r in rows),
                historical_upstream_calls_charged=0), resources=resources,
            timing=dict(successful_diagnostic_window_seconds=sum(w['cpu_diagnostic_wall_seconds'] for w in window_times),
                new_reader_window_seconds=sum(w['new_reader_window_seconds'] for w in window_times),
                current_execution_seconds=time.perf_counter()-started),
            shared_reference=result['shared_reference'])
        write(case/'report.json', report)
        metrics = {NEW_ARM: report['metrics']}
        source_metrics = {}
        for arm in CELLS[:3]:
            old_rows = sorted(baseline_rows[arm], key=lambda r: indices.index(r['query_index']))
            samples = [data[i] for i in indices]
            structural = evaluate_structural_metrics(old_rows, samples, ds, ks=[5,10,20,200], corpus_docs=corpus)
            qa, _, _ = _qa(old_rows, samples, dict(dataset=ds, eval_mode='rag_qa', qa_executed=True))
            metrics[arm] = dict(**structural['metrics'], **qa)
            source_metrics[arm] = dict(case=str(source/'cases'/ds/arm), report_sha256=sha(source/'cases'/ds/arm/'report.json'),
                evaluation='recomputed CPU metrics on the selected existing rows; no new baseline retrieval or QA')
        write(root/'analysis'/ds/'interaction.json', dict(dataset=ds, queries=len(rows), cells=metrics,
            interaction_contrast_pp=interaction_metrics(metrics),
            contrast='M_full - M_flat - M_singleton + M_combined; percentage points',
            source_cells=source_metrics, shared_structured_s0=True,
            diagnostic_aggregate=diagnostic_aggregate(all_records)))
        write(case/'completed.ok', dict(queries=len(rows), result_sha256=sha(case/'result.json'),
            report_sha256=sha(case/'report.json'), status='passed', old_three_cell_parity=True))
        status = 'complete'; print('DATASET PASS '+ds+': '+str(len(rows))+' new-arm retrieval+QA', flush=True)
    finally:
        append(root/'measurements'/ds/'execution_sessions.jsonl', dict(status=status,
            wall_seconds=time.perf_counter()-started, completed=len(rows), unix=time.time()))
        stop.set(); thread.join(timeout=2)
        for obj, name, original in reversed(recorder.restorations): setattr(obj, name, original)


def summarize(root):
    datasets = []
    for ds in DATASETS:
        path = root/'analysis'/ds/'interaction.json'
        if path.exists(): datasets.append(read(path))
    write(root/'summary.json', dict(datasets=datasets, new_variant=LABELS[-1], old_cases_rerun=False,
        new_route_plan_calls=0, index_rebuilt=False, shared_structured_s0=True))
    lines = ['# Closure-aware Packaging interaction', '',
        'Old Full / Flat / Singleton rows are reused. Diagnostic CPU replays pass historical parity; only the new combined arm adds QA.',
        'All cells retain the same structured S₀. The combined arm removes the closure reward and package actions, while retaining other DAG mechanisms.', '',
        '|Dataset|Cell|N|R@5|FCR@5|EM / Accuracy|F1 / Macro-F1|DCC@5|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for item in datasets:
        ds = item['dataset']; keys = ('Recall@5','FCR@5','ClassificationAccuracy' if ds=='hover' else 'ExactMatch',
            'ClassificationMacroF1' if ds=='hover' else 'F1','DCC@5')
        for arm in CELLS:
            values = [item['cells'][arm].get(k) for k in keys]
            lines.append('|'+ '|'.join([ds,LABELS[CELLS.index(arm)],str(item['queries'])]+
                ['N/A' if v is None else f'{100*v:.3f}' for v in values])+'|')
        values = [item['interaction_contrast_pp'].get(k) for k in keys]
        lines.append('|'+ '|'.join([ds,'Interaction (pp)',str(item['queries'])]+
            ['N/A' if v is None else f'{v:.3f}' for v in values])+'|')
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    from report_closure_interaction import refresh
    refresh(root)


def freeze(args):
    root, source = args.output.resolve(), args.source.resolve()
    source_done = source/'completed.ok'
    if read(source_done).get('status') != 'complete': raise ValueError('Original six-arm run must be complete')
    if source == root or source in root.parents: raise ValueError('New output must be separate from old results')
    if (root/'protocol.json').exists():
        if not args.resume: raise ValueError('Output exists; use --resume')
        protocol = read(root/'protocol.json')
        for name, expected in dict(source=str(source),limit=args.limit,workers=args.workers,
                reader_workers=args.reader_workers,include_indices=args.include_indices).items():
            if protocol[name] != expected: raise ValueError('Resume setting changed: '+name)
        for path, expected in protocol['source_hashes'].items():
            if sha(Path(path)) != expected: raise ValueError('Historical source changed: '+path)
        for path, expected in protocol['frozen_code'].items():
            if sha(root/'source_snapshot'/path) != expected: raise ValueError('Frozen code changed: '+path)
        return
    root.mkdir(parents=True, exist_ok=True)
    for folder in ('src','scripts','configs'):
        shutil.copytree(PROJECT/folder, root/'source_snapshot'/folder,
            ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    hashes = {}
    for path in [source/'protocol.json',source_done,source/'experiment_config.json',source/'method_config.json']:
        hashes[str(path)] = sha(path)
    for ds in ([args.dataset] if args.dataset else DATASETS):
        for name in ('benchmark.json','corpus.json','sample_manifest.json','index_receipt.json'):
            path = source/'metadata'/ds/name; hashes[str(path)] = sha(path)
        for arm in CELLS[:3]:
            for name in ('protocol.json','progress.jsonl','completed.ok','report.json','result.json'):
                path = source/'cases'/ds/arm/name; hashes[str(path)] = sha(path)
        for path in (source/'upstream'/ds).glob('*.sha256.json'):
            hashes[str(path)] = sha(path)
    code = {str(p.relative_to(root/'source_snapshot')): sha(p)
        for p in (root/'source_snapshot').rglob('*') if p.is_file()}
    write(root/'protocol.json', dict(schema_version=1,source=str(source),source_hashes=hashes,frozen_code=code,
        limit=args.limit,workers=args.workers,reader_workers=args.reader_workers,
        include_indices=args.include_indices,cells=list(CELLS),
        observer_off_timing_probe=dict(schema=TIMING_PROBE_VERSION,maximum_queries_per_dataset=32,
            warmups=['full','singleton_search'],orders=['F-S-S-F','S-F-F-S'],
            formal_results_changed=False,observer_enabled=False),
        datasets=[args.dataset] if args.dataset else list(DATASETS), new_route_plan_calls=0,index_rebuilt=False,
        scope='four diagnostic CPU cells, one added formal retrieval+QA arm; all retain structured Full S0'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    parser.add_argument('--output',type=Path,default=PROJECT/'outputs/closure_packaging_interaction_20261108')
    parser.add_argument('--dataset',choices=DATASETS)
    parser.add_argument('--limit',type=int,default=0,help='First N existing samples for preflight; 0 uses all 1000')
    parser.add_argument('--include-indices',default='',help='Additional development examples: musique:567,hover:317')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--reader-workers',type=int,default=32)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1<=args.workers<=8 or not 1<=args.reader_workers<=32 or args.limit<0: parser.error('Invalid run size/concurrency')
    try: parse_inclusions(args.include_indices)
    except ValueError as error: parser.error(str(error))
    if args.worker: worker(args); return
    lock = (PROJECT/'outputs/.closure_interaction.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    freeze(args); root=args.output.resolve(); (root/'run.pid').write_text(str(os.getpid())+'\n')
    state=dict(status='running',pid=os.getpid(),started_unix=time.time(),new_variant=NEW_ARM)
    child=None
    def stop(*unused):
        if child is not None and child.poll() is None: child.send_signal(signal.SIGTERM)
        raise KeyboardInterrupt('Stopped; resume this frozen output')
    for number in (signal.SIGTERM,signal.SIGINT): signal.signal(number,stop)
    try:
        for ds in ([args.dataset] if args.dataset else DATASETS):
            state['dataset']=ds;write(root/'status.json',state)
            command=[sys.executable,'-B','-u',str(root/'source_snapshot/scripts/run_closure_interaction.py'),
                '--worker','--source',str(args.source.resolve()),'--output',str(root),'--dataset',ds,
                '--limit',str(args.limit),'--workers',str(args.workers),'--reader-workers',str(args.reader_workers),
                '--include-indices',args.include_indices]
            child=subprocess.Popen(command,cwd=PROJECT)
            if child.wait()!=0: raise RuntimeError('Closure interaction worker failed: '+ds)
            child=None;summarize(root)
        state.update(status='complete',finished_unix=time.time());write(root/'completed.ok',state)
    except BaseException as error:
        state.update(status='stopped' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error));raise
    finally:
        if child is not None and child.poll() is None: child.send_signal(signal.SIGTERM);child.wait()
        write(root/'status.json',state);summarize(root)


if __name__=='__main__':
    os.environ.setdefault('OPENAI_API_KEY','EMPTY')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    os.environ['BINDPACK_LLM_MAX_IN_FLIGHT']='32'
    logging.basicConfig(level=logging.WARNING,format='%(asctime)s %(name)s %(levelname)s %(message)s')
    main()

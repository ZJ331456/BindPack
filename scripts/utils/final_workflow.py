"""Run the retained BindPack method against a frozen shared index.

Only planning, plan_prune, dag_package and support_semantic_veto are supported.
Index integrity, private request caches and exact metric accounting are kept
separate from historical parameter screening and component search.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil

from . import workflow_artifacts as experiments
from .common import http_status_counts, read_json, write_json
from .workflow_runtime import (
    DEFAULT_OUT, DEFAULT_RUNTIME, PC_ARGUMENTS, ROOT, code_hashes, environment, execute,
)

DATASETS = experiments.DATASETS
FLAGS = ('planning', 'plan_prune', 'dag_package', 'support_semantic_veto')
CASE_NAME = experiments.CASE_NAME


@dataclass(frozen=True)
class Variant:
    name: str
    flags: tuple[str, ...] = FLAGS
    plan_validation: str = 'canonical_refs'
    plan_routing: str = 'question_structure'

    def __post_init__(self):
        if len(set(self.flags)) != len(self.flags) or not set(self.flags) <= set(FLAGS):
            raise ValueError('Only retained BindPack component flags are supported')
        if self.plan_validation != 'canonical_refs' or self.plan_routing != 'question_structure':
            raise ValueError('The formal method uses canonical_refs and question_structure')

def subset_manifest(manifest, data, hops, indices):
    result = deepcopy(manifest)
    result.update(
        selected_indices=list(indices), sample_size_requested=len(indices),
        selected_sample_ids=[experiments.sample_identity(data[i], i) for i in indices],
        benchmark_hops=[hops[i] for i in indices],
        hop_distribution=dict(Counter(str(hops[i]) for i in indices)),
        gold_support_count_distribution=dict(Counter(
            str(len(experiments.gold_docs(data[i], manifest['dataset']))) for i in indices)),
    )
    return result


def aggregate_measurements(rows):
    keys = tuple(rows[0]['metrics'])
    return {
        'n_samples': len(rows),
        'retrieval_metrics': {key: sum(row['metrics'][key] for row in rows) / len(rows) for key in keys},
        'all_gold_top5': sum(row['all_gold_top5'] for row in rows) / len(rows),
        'all_gold_top10': sum(row['all_gold_top10'] for row in rows) / len(rows),
        'per_question': rows,
    }


def compact(report):
    return {key: value for key, value in report.items() if key != 'per_question'}


def initialize_case(context, case):
    if case.exists():
        raise ValueError(f'Unvalidated case already exists; inspect before retrying: {case}')
    case.mkdir(parents=True)
    source, manifest = context['clone_source'], context['manifest']
    if experiments.asset_hashes(source, manifest['model_dir']) != manifest['source_asset_sha256']:
        raise ValueError('Clone assets differ from the public frozen index')
    linked = {source / manifest['model_dir'] / name for name in experiments.ASSETS}

    def copy_file(original, target):
        if Path(original) in linked:
            os.link(original, target)
            return target
        return shutil.copy2(original, target)

    shutil.copytree(source, case / 'index', copy_function=copy_file,
                    ignore=shutil.ignore_patterns('llm_cache', '*.lock', '*.sqlite*',
                                                 'metrics*.json', 'eval_results*', 'openie_progress*'))
    shutil.copytree(context['cache'], case / 'index/llm_cache')
    actual = experiments.asset_hashes(case / 'index', manifest['model_dir'])
    if actual != manifest['source_asset_sha256']:
        raise ValueError('Cloned graph/vector assets differ')
    if experiments.cache_hashes(case / 'index/llm_cache') != context['cache_hashes']:
        raise ValueError('Private starting cache differs')
    write_json(case / 'before.json', {'asset_sha256': actual,
                                     'initial_cache_sha256': context['cache_hashes']})


def command(args, context, case, indices, flags):
    manifest = context['manifest']
    runtime = manifest['runtime']
    return [args.python, '-B', '-u', str(ROOT / 'scripts/eval_dataset.py'),
        '--dataset', context['dataset'], '--sample_size', str(len(indices)),
        '--sample_seed', '42', '--sample_indices_file', str(case / 'selected_indices.json'),
        '--llm_name', 'qwen3-8b', '--llm_base_url', args.llm_base_url,
        '--embedding_batch_size', str(runtime['embedding_batch_size']),
        '--openie_max_workers', '8', '--llm_prefetch_workers', '8',
        '--eval_mode', 'retrieve', '--retrieval_top_k', '200', '--result_top_k', '10',
        '--candidate_output_top_k', '200', '--reuse_index',
        *PC_ARGUMENTS,
        '--data_path', manifest['data_path'], '--corpus_path', manifest['corpus_path'],
        '--corpus_mode', 'full', '--qa_top_k', '5', '--max_qa_steps', '1',
        '--max_new_tokens', '2048',
        '--embedding_model_name', runtime['embedding_model_name'],
        '--openie_strict', 'false', '--openie_prompt_version', 'optimized',
        '--openie_validation_mode', 'structural', '--improvement_stage', '4',
        '--evidence_improvements', ','.join(flag for flag in FLAGS if flag in flags),
        '--evidence_plan_node_budget', '6', '--evidence_plan_depth_budget', '4',
        '--stratified_eval', '--stratified_output', str(case / 'stratified.json'),
        '--save_dir', str(case / 'index'), '--output', str(case / 'result.json')]


def validate_case(args, context, case, manifest, flags, elapsed, start, end):
    result = read_json(case / 'result.json')
    measurements, metrics = experiments.validate_result(
        result, manifest, CASE_NAME, context['data'], context['corpus'], expected_stage=4)
    config = result.get('runtime_config', {})
    value = config.get('evidence_improvements', '')
    actual_flags = set(value.split(',')) - {''} if isinstance(value, str) else set(value)
    if actual_flags != set(flags):
        raise ValueError(f'Configured improvements differ: {actual_flags} != {flags}')
    stats = result.get('llm_request_stats') or {}
    if stats.get('failures') != 0 or stats.get('max_in_flight') != 8:
        raise ValueError(f'LLM error accounting or concurrency differs: {stats}')
    if start < 0 or end < start:
        raise ValueError('vLLM request log rotated or truncated')
    status = http_status_counts(args.vllm_log, start, end)
    if any(code != '200' for code in status):
        raise ValueError(f'Non-200 API response: {status}')
    if stats.get('http_attempts') and not status:
        raise ValueError('HTTP attempts lack server request accounting')
    before = read_json(case / 'before.json')
    after = experiments.asset_hashes(case / 'index', manifest['model_dir'])
    source_after = experiments.asset_hashes(context['source'], manifest['model_dir'])
    if after != before['asset_sha256'] or source_after != manifest['source_asset_sha256']:
        raise ValueError('Private or public frozen index changed')
    report = aggregate_measurements(measurements)
    report.update(name=case.name, dataset=context['dataset'], improvement_stage=4,
        improvements=list(flag for flag in FLAGS if flag in flags), validated=True,
        seconds=elapsed, retrieval_seconds=result.get('retrieval_seconds'),
        llm_request_stats=stats, http_status_in_log=status,
        hop_distribution=manifest['hop_distribution'],
        vllm_log_slice={'path': args.vllm_log, 'start': start, 'end': end},
        asset_sha256_before=before['asset_sha256'], asset_sha256_after=after,
        initial_cache_sha256=before['initial_cache_sha256'],
        cache_policy=context['cache_policy'],
        benchmark_hop_policy=manifest['benchmark_hop_policy'])
    write_json(case / 'report.json', report)
    write_json(case / 'validated.ok', {
        'result_sha256': experiments.sha256(case / 'result.json'),
        'report_sha256': experiments.sha256(case / 'report.json'),
        'manifest_sha256': experiments.sha256(case / 'manifest.json')})
    print(f"[validated] {context['dataset']}/{case.name} n={len(measurements)} "
          f"R@5={metrics['Recall@5']:.4f} R@10={metrics['Recall@10']:.4f} sec={elapsed}", flush=True)
    return report, result


def verify_artifacts(case):
    marker = read_json(case / 'validated.ok')
    for field, filename in [('result_sha256', 'result.json'), ('report_sha256', 'report.json'),
                            ('manifest_sha256', 'manifest.json')]:
        if marker.get(field) != experiments.sha256(case / filename):
            raise ValueError(f'Validated artifact changed before cleanup: {case / filename}')
    report, manifest = read_json(case / 'report.json'), read_json(case / 'manifest.json')
    rows = report.get('per_question') or []
    if (report.get('validated') is not True or len(rows) != report.get('n_samples') or
            [row['query_index'] for row in rows] != manifest['selected_indices']):
        raise ValueError(f'Incomplete validated measurements: {case}')
    return report


def context_for(out, dataset, support):
    """Read frozen receipts; a previous retrieval case is never required."""
    manifest = read_json(out / 'metadata' / dataset / 'manifest.json')
    source = out / 'shared_indexes' / dataset
    if Path(manifest['source_index']).resolve() != source.resolve():
        raise ValueError(f'{dataset}: source index is not the declared public index')
    data, corpus, hops = experiments.validated_dataset(
        manifest['data_path'], manifest['corpus_path'], dataset_name=dataset)
    if (experiments.sha256(manifest['data_path']) != manifest['data_sha256'] or
            experiments.sha256(manifest['corpus_path']) != manifest['corpus_sha256']):
        raise ValueError(f'{dataset}: data or corpus changed')
    if experiments.asset_hashes(source, manifest['model_dir']) != manifest['source_asset_sha256']:
        raise ValueError(f'{dataset}: frozen source index changed')
    cache_source = out / 'metadata' / dataset / 'initial_llm_cache'
    if not cache_source.is_dir():
        raise ValueError(f'{dataset}: frozen initial LLM cache is missing')
    expected_cache = manifest['initial_cache_sha256']
    if experiments.cache_hashes(cache_source) != expected_cache:
        raise ValueError(f'{dataset}: frozen initial LLM cache changed')
    cache = support / 'initial_cache' / dataset
    if not cache.is_dir():
        # This receipt snapshot is immutable and has no writers. Copy exact
        # bytes so the frozen hash remains valid; SQLite backup can rewrite
        # the database header even when its logical contents are identical.
        cache.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(cache_source, cache, ignore=shutil.ignore_patterns('*.lock'))
    if experiments.cache_hashes(cache) != expected_cache:
        raise ValueError(f'{dataset}: private initial cache snapshot differs')
    return {'dataset': dataset, 'manifest': manifest, 'source': source,
            'clone_source': source, 'data': data, 'corpus': corpus, 'hops': hops,
            'cache': cache, 'cache_hashes': expected_cache,
            'cache_policy': 'Private frozen post-index cache snapshot; no prior retrieval run required.'}


def evaluate(samples, row, candidates):
    """Measure a ranking offline; gold labels are never sent to a selector."""
    gold = experiments.gold_docs(samples[row['query_index']], row['dataset'])
    metrics = {f'Recall@{k}': len(gold & set(candidates[:k])) / len(gold)
               for k in (1, 2, 5, 10, 20, 200)}
    return {'query_index': row['query_index'], 'metrics': metrics,
            'all_gold_top5': gold <= set(candidates[:5]), 'all_gold_top10': gold <= set(candidates[:10])}


def profile_command(args, context, case, indices, variant):
    return command(args, context, case, indices, frozenset(variant.flags)) + [
        '--evidence_plan_validation', variant.plan_validation,
        '--evidence_plan_routing', variant.plan_routing,
    ]


def verify_saved_case(context, case, manifest, variant):
    marker = read_json(case / 'validated.ok')
    report = verify_artifacts(case)
    if read_json(case / 'manifest.json') != manifest:
        raise ValueError(f'Question subset changed: {case}')
    profile = case / 'method_config.json'
    if (read_json(profile) != json.loads(json.dumps(asdict(variant))) or
            marker.get('method_config_sha256') != experiments.sha256(profile)):
        raise ValueError(f'Method configuration changed: {case}')
    expected = context['manifest']['source_asset_sha256']
    before = read_json(case / 'before.json')
    if (experiments.asset_hashes(context['source'], manifest['model_dir']) != expected or
            before['asset_sha256'] != expected or before['initial_cache_sha256'] != context['cache_hashes']):
        raise ValueError(f'Public index or initial cache changed: {case}')
    if marker.get('cleaned_private_index') or (marker.get('private_index_cleanup_pending') and
                                              not (case / 'index').exists()):
        proof = case / 'private_index_cleanup.json'
        saved = read_json(proof)
        if (marker.get('cleanup_proof_sha256') != experiments.sha256(proof) or
                saved['asset_sha256_at_cleanup'] != expected or
                saved['initial_cache_sha256'] != context['cache_hashes'] or (case / 'index').exists()):
            raise ValueError(f'Private index cleanup proof changed: {case}')
    elif experiments.asset_hashes(case / 'index', manifest['model_dir']) != expected:
        raise ValueError(f'Private index changed: {case}')
    return report

def cleanup_private_index(context, case):
    """Keep exports and signed cleanup evidence; unlink only this private index."""
    marker = read_json(case / 'validated.ok')
    if marker.get('cleaned_private_index'):
        return
    index = case / 'index'
    expected = context['manifest']['source_asset_sha256']
    if (index.is_symlink() or not index.is_dir() or
            experiments.asset_hashes(index, context['manifest']['model_dir']) != expected):
        raise ValueError(f'Refusing cleanup of an unverified private index: {index}')
    before = read_json(case / 'before.json')
    proof = case / 'private_index_cleanup.json'
    write_json(proof, {'asset_sha256_at_cleanup': expected,
                       'initial_cache_sha256': before['initial_cache_sha256'],
                       'scope': 'only private case/index; exported Top200/Top10, traces and reports retained'})
    marker.update(private_index_cleanup_pending=True,
                  cleanup_proof_sha256=experiments.sha256(proof))
    write_json(case / 'validated.ok', marker)
    shutil.rmtree(index)
    marker.update(cleaned_private_index=True, private_index_cleanup_pending=False)
    write_json(case / 'validated.ok', marker)


def run_case(args, context, case, indices, variant, env, *, cleanup=True):
    manifest = subset_manifest(context['manifest'], context['data'], context['hops'], indices)
    if (case / 'validated.ok').is_file():
        report = verify_saved_case(context, case, manifest, variant)
        if cleanup:
            cleanup_private_index(context, case)
        return report
    initialize_case(context, case)
    write_json(case / 'selected_indices.json', list(indices))
    write_json(case / 'manifest.json', manifest)
    write_json(case / 'method_config.json', asdict(variant))
    start = Path(args.vllm_log).stat().st_size
    elapsed = execute(profile_command(args, context, case, indices, variant), case / 'run.log', env)
    end = Path(args.vllm_log).stat().st_size
    report, result = validate_case(args, context, case, manifest, frozenset(variant.flags), elapsed, start, end)
    try:
        runtime = result.get('runtime_config', {})
        for key, expected in [('evidence_plan_validation', variant.plan_validation),
                              ('evidence_plan_routing', variant.plan_routing)]:
            if runtime.get(key) != expected:
                raise ValueError(f'{case}: {key}={runtime.get(key)} != {expected}')
        report.update(method_config=asdict(variant), improvements=list(variant.flags))
        write_json(case / 'report.json', report)
        marker = read_json(case / 'validated.ok')
        marker.update(report_sha256=experiments.sha256(case / 'report.json'),
                      method_config_sha256=experiments.sha256(case / 'method_config.json'))
        write_json(case / 'validated.ok', marker)
        if cleanup:
            cleanup_private_index(context, case)
    except Exception:
        (case / 'validated.ok').unlink(missing_ok=True)
        raise
    return report

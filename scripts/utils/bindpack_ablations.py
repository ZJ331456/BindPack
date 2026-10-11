"""Ablation orchestration with immutable indexes and explicit input receipts."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
from pathlib import Path
import shutil
import urllib.request
from tqdm import tqdm

from . import final_workflow as wf
from . import workflow_artifacts as assets
from .common import read_json, write_json
from .workflow_runtime import ROOT, DEFAULT_RUNTIME, environment, execute
from .http_audit import locate

LOG = logging.getLogger('bindpack.ablations')
DATASETS = ('hotpotqa', '2wikimultihopqa', 'musique')
SMOKE_INDICES = {'hotpotqa': [145, 339], '2wikimultihopqa': [161, 532], 'musique': [716, 889]}


def snapshot_at(directory, index):
    with gzip.open(Path(directory) / f'{index:05d}.json.gz', 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def publish(context, case, rows, rankings, details, *, scope, input_receipt):
    from bindpack.experiments.reporting import aggregate_rows
    case.mkdir(parents=True, exist_ok=True)
    report = aggregate_rows(rows)
    report.update(name=case.name, dataset=context['dataset'], validated=True,
                  evaluation_scope='development_queries_previously_used_for_method_selection',
                  experiment_scope=scope, gold_labels_used_by_selector=False,
                  shared_index=str(context['source']), input_receipt=input_receipt)
    exported = []
    for index, ids, document, diagnostic in rankings:
        sample = context['data'][index]
        exported.append({'query_index': index, 'question': sample['question'],
            'sample_id': assets.sample_identity(sample, index),
            'benchmark_hops': context['hops'][index],
            'docs': [document(d) for d in ids[:10]], 'candidate_doc_ids': ids[:200],
            'retrieval_trace': diagnostic})
    write_json(case / 'result.json', {'dataset': context['dataset'], 'eval_mode': 'retrieve',
        'sample_size_effective': len(rows),
        'selected_indices': [r['query_index'] for r in rows], 'result_top_k': 10,
        'retrieval_metrics': report['retrieval_metrics'], 'results': exported,
        'experiment_scope': scope, 'input_receipt': input_receipt,
        'candidate_text_storage': 'shared frozen snapshot/source receipt; IDs exported here'})
    write_json(case / 'report.json', report)
    write_json(case / 'diagnostics.json', details)
    write_json(case / 'validated.ok', {name + '_sha256': assets.sha256(case / (name + '.json'))
                                     for name in ('result', 'report', 'diagnostics')})
    LOG.info('[validated-offline] %s/%s n=%d R@5=%.4f R@10=%.4f', context['dataset'],
             case.name, len(rows), report['retrieval_metrics']['Recall@5'], report['retrieval_metrics']['Recall@10'])
    return report


def review_replay(args, context, indices, output):
    from bindpack.experiments.review_ablation import run_review_ablations
    from bindpack.experiments.reporting import score_ranking, paired_bootstrap, interaction_bootstrap
    source = context['source'].parents[1] / 'cases' / context['dataset'] / 'ours/result.json'
    payload = read_json(source)
    by_index = {row['query_index']: row for row in payload['results']}
    groups = {}
    for index in tqdm(indices, desc=f'{context["dataset"]} frozen review', unit='query'):
        row = by_index[index]
        trace = row['retrieval_trace']['evidence']['improvement_support_semantic_veto']
        documents = dict(zip(trace['final_doc_ids'], row['candidate_docs']))
        replay = run_review_ablations(row, documents)
        for name, variant in replay['conditions'].items():
            if variant['status'] != 'resolved':
                raise ValueError(f'{context["dataset"]}/{index}: missing frozen review response')
            ids = variant['final_doc_ids']
            group = groups.setdefault(name, {'rows': [], 'rankings': [], 'details': []})
            group['rows'].append(score_ranking(index, ids, documents.__getitem__,
                                    list(assets.gold_docs(context['data'][index], context['dataset']))))
            group['rankings'].append((index, ids, documents.__getitem__, variant))
            group['details'].append({'query_index': index, 'frozen_inputs': replay.get('frozen_input_sha256'),
                                     'accepted': variant['accepted']})
    reports = {}
    for name, group in groups.items():
        reports[name] = publish(context, output / context['dataset'] / f'abla_review_{name}',
            group['rows'], group['rankings'], group['details'],
            scope='same_proposal_and_reviewer_response_guard_veto_factorial',
            input_receipt={'source': str(source), 'source_sha256': assets.sha256(source),
                           'model_requests': 0, 'cache_policy': 'frozen-response CPU replay'})
    comparisons = {name: paired_bootstrap(groups[name]['rows'], groups['both']['rows'])
                   for name in ('none', 'guard_only', 'veto_only')}
    comparisons['interaction'] = interaction_bootstrap(
        groups['none']['rows'], groups['guard_only']['rows'],
        groups['veto_only']['rows'], groups['both']['rows'])
    write_json(args.metadata / context['dataset'] / 'review_comparisons.json', comparisons)
    return reports


def fixed_replay(args, context, indices, source_case, output):
    from bindpack.experiments.fixed_selection import run_fixed_selection
    from bindpack.experiments.reporting import score_ranking, paired_bootstrap, interaction_bootstrap
    groups = {}
    snapshots = source_case / 'snapshots'
    receipt = read_json(snapshots / 'manifest.json')
    expected_hashes = {r['query_index']: r['input_sha256'] for r in receipt['snapshots']}
    for index in tqdm(indices, desc=f'{context["dataset"]} fixed selectors', unit='query'):
        with gzip.open(snapshots / f'{index:05d}.json.gz', 'rb') as stream:
            blob = stream.read()
        if hashlib.sha256(blob).hexdigest() != expected_hashes[index]:
            raise ValueError('Frozen snapshot hash changed')
        snapshot = json.loads(blob)
        variants = run_fixed_selection(snapshot)['variants']
        if variants['current_closed']['ids'] != snapshot.get('expected_dag_ids'):
            raise ValueError(f'{context["dataset"]}/{index}: real DAG replay differs')
        documents = {int(k): v for k, v in snapshot['documents'].items()}
        for name, variant in variants.items():
            ids = variant['ids']
            group = groups.setdefault(name, {'rows': [], 'rankings': [], 'details': []})
            group['rows'].append(score_ranking(index, ids, documents.__getitem__,
                                    list(assets.gold_docs(context['data'][index], context['dataset']))))
            group['rankings'].append((index, ids, documents.__getitem__, variant['diagnostics']))
            group['details'].append({'query_index': index, 'diagnostics': variant['diagnostics']})
    reports = {}
    for name, group in groups.items():
        scope = ('existing_proof_qualification_and_coverage_reward_on_shared_closed_package_space'
                 if name.endswith(('_flat', '_closed')) else
                 'ordinary_node_coverage_greedy_reference' if name == 'ordinary_node_coverage_greedy'
                 else 'score_ranking_reference_without_proof_filter')
        reports[name] = publish(context, output / context['dataset'] / f'abla_fixed_{name}',
            group['rows'], group['rankings'], group['details'],
            scope=scope,
            input_receipt={'snapshots': str(snapshots), 'manifest_sha256': assets.sha256(snapshots / 'manifest.json'),
                           'model_requests': 0, 'top2_fixed': True, 'semantic_review_disabled': True})
    comparisons = {name: paired_bootstrap(groups[name]['rows'], groups['current_closed']['rows'])
                   for name in groups if name != 'current_closed'}
    comparisons['interaction'] = interaction_bootstrap(
        groups['weak_flat']['rows'], groups['weak_closed']['rows'],
        groups['current_flat']['rows'], groups['current_closed']['rows'])
    write_json(args.metadata / context['dataset'] / 'fixed_comparisons.json', comparisons)
    return reports


def online_case(args, context, indices, output, name, mode, source=None, fraction=1.0,
                *, prepare_only=False, execution_receipt=None):
    case = output / context['dataset'] / name
    manifest = wf.subset_manifest(context['manifest'], context['data'], context['hops'], indices)
    if (case / 'validated.ok').exists():
        old = read_json(case / 'experiment_protocol.json')
        if (read_json(case / 'manifest.json') != manifest or old['mode'] != mode
                or old['budget_fraction'] != fraction or old['snapshot_source'] != (str(source) if source else None)):
            raise ValueError('Existing case configuration differs; refusing automatic reuse')
        marker = read_json(case / 'validated.ok')
        if (marker.get('ledger_sha256') != assets.sha256(case / 'request_ledger.json') or
                read_json(case / 'request_ledger.json')['summary'].get('complete') is not True):
            raise ValueError('Validated request ledger changed or is incomplete')
        return wf.verify_artifacts(case)
    if execution_receipt is None:
        wf.initialize_case(context, case)
        write_json(case / 'manifest.json', manifest)
        write_json(case / 'selected_indices.json', indices)
    flags = tuple(f for f in wf.FLAGS if mode != 'no_plan_prune' or f != 'plan_prune')
    command = wf.command(args, context, case, indices, flags)
    command[command.index(str(ROOT / 'scripts/eval_dataset.py'))] = str(ROOT / 'scripts/ablation_worker.py')
    command += ['--ablation-mode', mode, '--snapshot-dir', str((source or case) / 'snapshots'),
                '--ledger-path', str(case / 'request_ledger.json'), '--budget-fraction', str(fraction)]
    protocol = {'mode': mode, 'flags': flags,
        'shared_index': str(context['source']), 'budget_fraction': fraction,
        'generation_budget': 2048, 'thinking': False, 'llm_concurrency': 8,
        'embedding_batch_size': 4, 'hop_source': manifest['benchmark_hop_policy'],
        'snapshot_source': str(source) if source else None}
    if execution_receipt is None:
        write_json(case / 'experiment_protocol.json', protocol)
    elif read_json(case / 'experiment_protocol.json') != json.loads(json.dumps(protocol)):
        raise ValueError('Prepared case configuration changed')
    if prepare_only:
        return {'argv': command[3:], 'log_path': str(case / 'run.log'),
                'vllm_log': args.vllm_log, 'receipt_path': str(case / 'execution_receipt.json')}
    if execution_receipt is None:
        start = Path(args.vllm_log).stat().st_size
        elapsed = execute(command, case / 'run.log', environment(args))
        end = Path(args.vllm_log).stat().st_size
    else:
        if execution_receipt.get('completed') is not True:
            raise ValueError('Suite job did not complete')
        start, end = execution_receipt['log_start'], execution_receipt['log_end']
        elapsed = execution_receipt['seconds']
    ledger = read_json(case / 'request_ledger.json')
    if ledger['summary'].get('complete') is not True:
        raise ValueError('Request cost ledger is incomplete')
    report, _ = wf.validate_case(args, context, case, manifest, flags, elapsed, start, end)
    if mode in ('qd', 'iterative'):
        report['wall_time_scope'] = 'incremental_control_after_replayed_common_base_plus_model_loading'
    report['request_cost'] = ledger['summary']
    report['evaluation_scope'] = 'development_queries_previously_used_for_method_selection'
    write_json(case / 'report.json', report)
    marker = read_json(case / 'validated.ok')
    marker['report_sha256'] = assets.sha256(case / 'report.json')
    marker['ledger_sha256'] = assets.sha256(case / 'request_ledger.json')
    write_json(case / 'validated.ok', marker)
    return report


def online_conditions(capture, fractions):
    conditions = [('abla_capture_full', 'capture', None, 1.0),
                  ('abla_no_plan_prune', 'no_plan_prune', None, 1.0),
                  ('abla_binding_string', 'binding_string', None, 1.0),
                  ('abla_binding_relation', 'binding_relation', None, 1.0)]
    conditions.extend((f'abla_budget_{mode}_{round(fraction * 100)}', mode, capture, fraction)
                      for fraction in fractions for mode in ('qd', 'iterative'))
    return conditions


def smoke_online_cases(args, context, indices, output, conditions):
    """Sequential real-model smoke jobs share weights, never request caches."""
    jobs = []
    for name, mode, source, fraction in conditions:
        case = output / context['dataset'] / name
        if (case / 'validated.ok').exists():
            continue
        if (case / 'execution_receipt.json').is_file():
            receipt = read_json(case / 'execution_receipt.json')
            if receipt.get('completed') is True:
                online_case(args, context, indices, output, name, mode, source, fraction,
                            execution_receipt=receipt)
                continue
        jobs.append(online_case(args, context, indices, output, name, mode, source, fraction,
                                prepare_only=True))
    if jobs:
        job_file = args.metadata / context['dataset'] / 'smoke_jobs.json'
        write_json(job_file, jobs)
        execute([args.python, '-B', '-u', str(ROOT / 'scripts/ablation_suite_worker.py'),
                 '--job-file', str(job_file)],
                args.metadata / context['dataset'] / 'smoke_suite.log', environment(args))
    reports = {}
    for name, mode, source, fraction in conditions:
        case = output / context['dataset'] / name
        receipt = None if (case / 'validated.ok').exists() else read_json(case / 'execution_receipt.json')
        reports[name] = online_case(args, context, indices, output, name, mode, source, fraction,
                                    execution_receipt=receipt)
    return reports


def online_comparisons(args, context, capture, reports):
    from bindpack.experiments.reporting import paired_bootstrap, annotation_items
    reference = reports['abla_capture_full']
    comparisons = {name: paired_bootstrap(report['per_question'], reference['per_question'])
                   for name, report in reports.items() if name != 'abla_capture_full'}
    annotations = []
    snapshot_indices = read_json(capture / 'snapshots/manifest.json')['snapshots']
    for entry in tqdm(snapshot_indices, desc=f'{context["dataset"]} annotation export', unit='query'):
        index = entry['query_index']
        path = capture / 'snapshots' / f'{index:05d}.json.gz'
        if path.is_file():
            annotations.extend(annotation_items(snapshot_at(capture / 'snapshots', index)))
    write_json(args.metadata / context['dataset'] / 'online_comparisons.json', comparisons)
    write_json(args.metadata / context['dataset'] / 'source_check_annotation.json', {
        'status': 'pending_human_annotation', 'labels_used_by_selector': False, 'items': annotations})


def write_summary(args, reports):
    lines = ['# NV2 BindPack 检索机制消融结果', '',
             '同一冻结索引；本地开发查询全量。QA 与新测试集验证尚未执行。', '',
             '| 数据集 | 条件 | R@5 | R@10 | All gold@5 | 逻辑 tokens |',
             '|---|---|---:|---:|---:|---:|']
    for dataset, groups in reports.items():
        flat = dict(groups.get('online', {}))
        for group in ('fixed', 'review'):
            flat.update({f'{group}/{key}': value for key, value in groups.get(group, {}).items()})
        for name, report in flat.items():
            metrics = report['retrieval_metrics']
            tokens = (report.get('request_cost') or {}).get('total_tokens', 0)
            lines.append(f'| {dataset} | {name} | {metrics["Recall@5"]:.4f} | '
                         f'{metrics["Recall@10"]:.4f} | {report["all_gold_top5"]:.4f} | {tokens} |')
    lines.extend(['', '预算对照表中的 tokens 是增量调用；完整预算（含复用共同基础检索）见逐题 budget_control。',
                  '固定输入与保护／否决回放没有新增模型请求；其上游输入构造成本单独保留。',
                  '配对 bootstrap 与交互效应见各数据集 comparisons.json。'])
    (args.metadata / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def start_full_after_smoke(args, root, reports):
    """Publish a compact test receipt before deleting owned smoke artifacts."""
    smoke = root / '_bindpack_ablation_smoke'
    expected = (4 + 2 * len(args.budget_fractions) + 9 + 4) * len(args.datasets)
    validated = [path for dataset in args.datasets
                 for path in (smoke / 'cases' / dataset).glob('abla_*/validated.ok')]
    if len(validated) != expected:
        raise ValueError('Smoke condition count differs; full run was not admitted')
    compact = {'validated': True, 'datasets': args.datasets, 'conditions': expected,
               'samples_per_condition': 2, 'qa': False, 'reports': {
                   dataset: {group: {name: wf.compact(report) for name, report in conditions.items()}
                             for group, conditions in groups.items()}
                   for dataset, groups in reports.items()},
               'code_snapshot': read_json(args.metadata / 'code_snapshot.json')}
    destination = root / 'metadata/bindpack_ablations'
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / 'smoke_validation.json', compact)
    shutil.rmtree(smoke)
    shutil.rmtree(args.metadata)
    LOG.info('[smoke-passed] removed temporary results; starting full development-query evaluation')
    execute([args.python, '-B', '-u', str(ROOT / 'scripts/run_bindpack_ablations.py'),
             '--out-root', str(root), '--python', args.python, '--llm-base-url', args.llm_base_url,
             '--vllm-log', args.vllm_log, '--datasets', *args.datasets,
             '--budget-fractions', *[str(f) for f in args.budget_fractions]],
            root / 'logs/run_nv2_ablations_full.log', environment(args))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-root', default=str(ROOT / 'outputs/3multi_hop_datasets_results_with_nv2_10_8'))
    parser.add_argument('--python', default='/root/anaconda3/envs/rag/bin/python')
    parser.add_argument('--llm-base-url', default='http://127.0.0.1:8035/v1')
    parser.add_argument('--vllm-log', default='')
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--full-after-smoke', action='store_true')
    parser.add_argument('--review-only', action='store_true')
    parser.add_argument('--budget-fractions', nargs='+', type=float, default=[0.5, 1.0])
    args = parser.parse_args()
    if args.full_after_smoke and (not args.smoke or args.review_only):
        parser.error('--full-after-smoke requires --smoke and all experiment families')
    if any(not 0 < f <= 1 for f in args.budget_fractions):
        parser.error('budget fractions must be in (0,1]')
    if any(abs(f * 100 - round(f * 100)) > 1e-8 for f in args.budget_fractions):
        parser.error('budget fractions must use integer percentages to avoid case-name collisions')
    root = Path(args.out_root).resolve()
    run_name = 'bindpack_ablation_smoke' if args.smoke else 'bindpack_ablations'
    args.metadata = root / 'metadata' / run_name
    args.metadata.mkdir(parents=True, exist_ok=True)
    code_receipt = {str(path.relative_to(ROOT)): assets.sha256(path)
                    for directory in (ROOT / 'src/bindpack', ROOT / 'scripts')
                    for path in sorted(directory.rglob('*.py'))}
    code_receipt['scripts/run_bindpack_ablations_nv2.sh'] = assets.sha256(ROOT / 'scripts/run_bindpack_ablations_nv2.sh')
    receipt_path = args.metadata / 'code_snapshot.json'
    if receipt_path.exists() and read_json(receipt_path) != code_receipt:
        raise ValueError('Ablation code changed since this batch started; use a separate metadata batch')
    write_json(receipt_path, code_receipt)
    output = root / '_bindpack_ablation_smoke/cases' if args.smoke else root / 'cases'
    args.embedding_model = '/root/models/NV-Embed-v2'
    args.embedding_provider = 'nvembed'
    args.runtime_deps = str(DEFAULT_RUNTIME)
    args.hippo_root = '/root/baseline/HippoRAG'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if not args.review_only:
        with urllib.request.urlopen(args.llm_base_url.rstrip('/') + '/models', timeout=10) as response:
            if response.status != 200:
                raise ValueError('vLLM is not ready')
        args.vllm_log = args.vllm_log or locate(args.llm_base_url)
    protocol = {'datasets': args.datasets, 'sample_size': 2 if args.smoke else 'all_local_1000',
        'shared_indexes': str(root / 'shared_indexes'), 'qa': False,
        'binding_factor': 'online_gate_controls_and_separate_existing_proof_qualification',
        'closure_factor': 'coverage_reward_in_common_closed_package_proposal_space',
        'budget_policy': 'per_query_observed_capture_tokens_including_common_base',
        'budget_fractions': args.budget_fractions, 'index_rebuild': False,
        'production_algorithm_modified': False}
    write_json(args.metadata / 'protocol.json', protocol)
    all_reports = {}
    for dataset in args.datasets:
        (args.metadata / dataset).mkdir(parents=True, exist_ok=True)
        context = wf.context_for(root, dataset, args.metadata / 'support')
        indices = SMOKE_INDICES[dataset] if args.smoke else context['manifest']['selected_indices']
        LOG.info('[dataset-start] %s n=%d', dataset, len(indices))
        all_reports[dataset] = {'review': review_replay(args, context, indices, output)}
        if args.review_only:
            continue
        capture = output / dataset / 'abla_capture_full'
        conditions = online_conditions(capture, args.budget_fractions)
        if args.smoke:
            reports = smoke_online_cases(args, context, indices, output, conditions)
        else:
            reports = {name: online_case(args, context, indices, output, name, mode, source, fraction)
                       for name, mode, source, fraction in conditions}
        all_reports[dataset]['online'] = reports
        all_reports[dataset]['fixed'] = fixed_replay(args, context, indices, capture, output)
        online_comparisons(args, context, capture, reports)
        if assets.asset_hashes(context['source'], context['manifest']['model_dir']) != context['manifest']['source_asset_sha256']:
            raise ValueError('Public index failed final integrity check')
        write_json(args.metadata / 'progress.json', {'completed_dataset': dataset, 'reports': all_reports})
    write_json(args.metadata / 'summary.json', all_reports)
    write_summary(args, all_reports)
    write_json(args.metadata / 'completed.ok', {'datasets': args.datasets, 'qa': False, 'protocol': protocol})
    LOG.info('[done] all requested ablations validated: %s', args.metadata)
    if args.full_after_smoke:
        start_full_after_smoke(args, root, all_reports)

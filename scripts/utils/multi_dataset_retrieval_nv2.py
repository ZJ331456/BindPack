"""Reuse three shared NV-Embed-v2 indexes and evaluate BindPack as ours.

--build-indexes explicitly enables index construction. --build-indexes --smoke
uses two intact questions and twenty intact passages per dataset in a disposable
output directory. Retrieval keeps Top10/Top200 and omits QA.
"""
from __future__ import annotations

import argparse
import logging
import os
from copy import copy
from pathlib import Path

from bindpack.index.publication_policy import parse_bool

from . import multi_dataset_retrieval as base
from . import multi_dataset_retrieval_support_semantic_veto as semantic
from .common import read_json, write_json
from .http_audit import locate


DEFAULT_OUT = base.ROOT / 'outputs/3multi_hop_datasets_results_with_nv2_10_8'
METHODS = (semantic.CASE_NAME,)
LOG = logging.getLogger('multi_dataset_nv2')


def case_path(out, dataset, method, smoke):
    work = out
    if smoke:
        work = out / semantic.SMOKE_NAME
    return work / 'cases' / dataset / method


def publish_summary(out, stages, smoke, case_name=semantic.CASE_NAME):
    """Aggregate the three independently validated reports without reranking."""
    reports, completed = {}, 0
    lines = ['| Dataset | Method | R@1 | R@2 | R@5 | R@10 | All gold@5 | Seconds | HTTP attempts |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for dataset in base.DATASETS:
        reports[dataset] = {}
        for method in (case_name,):
            case = case_path(out, dataset, method, smoke)
            if not (case / 'validated.ok').is_file():
                continue
            report = read_json(case / 'report.json')
            if report.get('validated') is not True:
                raise ValueError(f'Case report is not validated: {case}')
            reports[dataset][method] = {
                **{key: value for key, value in report.items() if key != 'per_question'},
                'case_path': str(case),
            }
            completed += 1
            values = [f"{report['retrieval_metrics'][f'Recall@{k}']:.4f}" for k in (1, 2, 5, 10)]
            lines.append(f'| {dataset} | {method} | ' + ' | '.join(values)
                         + f" | {report['all_gold_top5']:.4f} | {report['seconds']}"
                         + f" | {report['llm_request_stats'].get('http_attempts', 0)} |")
    summary = {'datasets': reports, 'stages': stages, 'smoke': smoke,
               'expected_case_count': len(base.DATASETS) * len(METHODS),
               'validated_case_count': completed}
    summary_name = 'comparison' if case_name == semantic.CASE_NAME else f'{case_name}_comparison'
    write_json(out / f'{summary_name}.json', summary)
    (out / f'{summary_name}.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)
    return completed


def retain_request_logs(out, smoke, case_name=semantic.CASE_NAME):
    """Retain actual service-log slices before its process or file disappears."""
    for dataset in base.DATASETS:
        metadata = out / 'metadata' / dataset
        report_path = metadata / 'index_build_report.json'
        if report_path.is_file():
            semantic.retain_http_audit(metadata, read_json(report_path))
        for method in (case_name,):
            case = case_path(out, dataset, method, smoke)
            if (case / 'validated.ok').is_file():
                semantic.retain_http_audit(case, read_json(case / 'report.json'))


def run(args):
    case_name = getattr(args, 'case_name', semantic.CASE_NAME)
    reuse_indexes = getattr(args, 'reuse_indexes', True)
    if not case_name or case_name in ('.', '..') or Path(case_name).name != case_name:
        raise ValueError('Case name must be a single directory name')
    out = Path(args.out_root).resolve()
    if not out.is_relative_to(base.ROOT / 'outputs') or out == base.ROOT / 'outputs':
        raise ValueError('Output must be a dedicated directory under BindPack/outputs')
    if args.smoke and 'smoke' not in out.name:
        raise ValueError('Smoke requires a separate output directory with "smoke" in its name')
    if args.smoke and reuse_indexes:
        raise ValueError('--smoke builds separate small indexes; use --build-indexes')
    if reuse_indexes:
        for dataset in base.DATASETS:
            index = out / 'shared_indexes' / dataset
            receipt = out / 'metadata' / dataset / 'manifest.json'
            if not index.is_dir() or not receipt.is_file():
                raise ValueError(f'Missing frozen shared index or receipt: {dataset}')
            manifest = read_json(receipt)
            if manifest.get('index_build_status') != 'ready':
                raise ValueError(f'Shared index is not ready: {dataset}')
            runtime = manifest['runtime']
            if (runtime.get('embedding_model_name') != args.embedding_model
                    or runtime.get('embedding_provider') != args.embedding_provider
                    or runtime.get('embedding_batch_size') != 4):
                raise ValueError(f'{dataset}: frozen embedding configuration differs')
    if args.embedding_provider != 'nvembed' or 'NV-Embed-v2' not in args.embedding_model:
        raise ValueError('This runner requires the NV-Embed-v2 model and nvembed provider')
    if args.embedding_batch_size != 4:
        raise ValueError('The formal workflow keeps embedding batch=4')
    args.vllm_log = args.vllm_log or locate(args.llm_base_url)
    base.log_offset(Path(args.vllm_log))
    out.mkdir(parents=True, exist_ok=True)
    protocol = {
        'datasets': list(base.DATASETS), 'methods': [case_name], 'smoke': args.smoke,
        'reuse_indexes': reuse_indexes,
        'embedding_model': args.embedding_model, 'embedding_provider': args.embedding_provider,
        'embedding_batch_size': args.embedding_batch_size, 'llm_name': 'qwen3-8b',
        'nv_embedding_oom_split': parse_bool(os.environ.get('PATHCONDRAG_NVEMBED_OOM_SPLIT', 'false')),
        'llm_base_url': args.llm_base_url, 'llm_workers': 8, 'index_workers': 8,
        'max_new_tokens': 2048, 'thinking': False, 'hop_source': 'benchmark',
        'openie_strict': args.openie_strict, 'openie_prompt_version': args.openie_prompt_version,
        'openie_validation_mode': args.openie_validation_mode,
        'eval_mode': 'retrieve', 'retrieval_top_k': 200,
        'result_top_k': 10, 'candidate_output_top_k': 200,
        'stages': (['reuse_all_shared_indexes'] if reuse_indexes else ['build_all_shared_indexes'])
                  + ['retrieve_support_semantic_veto'],
    }
    prefix = 'nv2_final' if case_name == semantic.CASE_NAME else f'nv2_{case_name}'
    completed_file = out / ('completed.ok' if case_name == semantic.CASE_NAME else f'{case_name}_completed.ok')
    protocol_file = out / 'metadata' / f'{prefix}_workflow_protocol.json'
    protocol_file.parent.mkdir(exist_ok=True)
    if protocol_file.is_file() and read_json(protocol_file) != protocol:
        raise ValueError('Existing NV2 workflow protocol differs; inspect before retrying')
    write_json(protocol_file, protocol)
    completed_file.unlink(missing_ok=True)
    stages = {}
    frozen = {}

    def record(name, state, **details):
        stages[name] = {'state': state, **details}
        write_json(out / 'metadata' / f'{prefix}_workflow_stage_status.json', stages)
        LOG.info('[workflow-stage] %s %s %s', name, state, details)

    index_args = copy(args)
    index_args.indexes_only = True
    secondary_args = copy(args)
    secondary_args.smoke_out_root = None
    if reuse_indexes:
        frozen = {dataset: base.frozen_hashes(out / 'shared_indexes' / dataset)
                  for dataset in base.DATASETS}
        write_json(out / 'metadata' / f'{prefix}_frozen_index_sha256.json', frozen)
        record('reuse_all_shared_indexes', 'ready', rebuild=False)
    functions = ([] if reuse_indexes else [('build_all_shared_indexes', base.run, index_args)])
    functions.append(('retrieve_support_semantic_veto', semantic.run, secondary_args))
    for name, function, options in functions:
        record(name, 'running')
        try:
            status = function(options)
            retain_request_logs(out, args.smoke, case_name)
            if status:
                record(name, 'failed', returncode=status)
                completed_file.unlink(missing_ok=True)
                publish_summary(out, stages, args.smoke, case_name)
                return status
            if name == 'build_all_shared_indexes':
                frozen = {dataset: base.frozen_hashes(out / 'shared_indexes' / dataset)
                          for dataset in base.DATASETS}
                write_json(out / 'metadata' / f'{prefix}_frozen_index_sha256.json', frozen)
            elif any(base.frozen_hashes(out / 'shared_indexes' / dataset) != frozen[dataset]
                     for dataset in base.DATASETS):
                raise ValueError('A retrieval stage changed a frozen graph/vector/OpenIE/manifest asset')
            # Publish completion only after all three final cases.
            completed_file.unlink(missing_ok=True)
            record(name, 'ready')
        except Exception as error:
            record(name, 'failed', error=f'{type(error).__name__}: {error}')
            completed_file.unlink(missing_ok=True)
            LOG.exception('[workflow-failed] %s', name)
            publish_summary(out, stages, args.smoke, case_name)
            return 1
    count = publish_summary(out, stages, args.smoke, case_name)
    if count != len(base.DATASETS) * len(METHODS):
        raise ValueError(f'Expected three validated retrieval cases, found {count}')
    write_json(completed_file, {
        'datasets': list(base.DATASETS), 'methods': [case_name],
        'validated_case_count': count, 'smoke': args.smoke,
    })
    LOG.info('[done] three frozen NV2 indexes and three %s retrieval cases validated: %s', case_name, out)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-root')
    parser.add_argument('--case-name', default='ours')
    index_mode = parser.add_mutually_exclusive_group()
    index_mode.add_argument('--reuse-indexes', dest='reuse_indexes', action='store_true')
    index_mode.add_argument('--build-indexes', dest='reuse_indexes', action='store_false')
    parser.set_defaults(reuse_indexes=True)
    parser.add_argument('--datasets-dir', default='/root/datasets')
    parser.add_argument('--hippo-root', default='/root/baseline/HippoRAG')
    parser.add_argument('--runtime-deps', default=str(base.DEFAULT_RUNTIME))
    parser.add_argument('--python', default='/root/anaconda3/envs/rag/bin/python')
    parser.add_argument('--llm-base-url', default='http://127.0.0.1:8035/v1')
    parser.add_argument('--vllm-log', help='Actual server request log; default: local vLLM stdout descriptor')
    parser.add_argument('--embedding-model', default='/root/models/NV-Embed-v2')
    parser.add_argument('--embedding-provider', default='nvembed')
    parser.add_argument('--embedding-batch-size', type=int, choices=[4], default=4)
    parser.add_argument('--openie-strict', '--openie_strict', type=parse_bool, default=False)
    parser.add_argument('--openie-prompt-version', '--openie_prompt_version',
                        choices=('origin', 'optimized'), default='optimized')
    parser.add_argument('--openie-validation-mode', '--openie_validation_mode',
                        choices=('structural', 'source_verified'), default='structural')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(argv)
    args.out_root = args.out_root or str(DEFAULT_OUT.with_name(DEFAULT_OUT.name + '_smoke')
                                        if args.smoke else DEFAULT_OUT)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    return run(args)

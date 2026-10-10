"""Build and validate three reusable indexes for the retained BindPack method.

Index extraction failures follow openie_strict. Artifact identity failures are
recorded and block use of the affected index. No historical method is run.
"""
import argparse
import logging
import os
import shutil
import urllib.request
from pathlib import Path
from types import SimpleNamespace

from bindpack.index.publication_policy import parse_bool
from . import workflow_artifacts as experiments
from .common import read_json, write_json
from .workflow_runtime import (
    ALL_ASSETS, DEFAULT_OUT, DEFAULT_RUNTIME, PC_ARGUMENTS, ROOT, code_hashes, environment, execute,
)

DATASETS = experiments.DATASETS
CASE_NAME = experiments.CASE_NAME
CASES = (CASE_NAME,)
SMOKE_POSITIONS = {'hotpotqa': (140, 901), '2wikimultihopqa': (866, 966), 'musique': (808, 930)}
LOG = logging.getLogger('bindpack.shared_indexes')

def sample_id(sample, position):
    return sample.get('id', sample.get('_id', position))


def smoke_dataset(out, source, name):
    """Keep original questions/gold passages plus 20 intact source passages."""
    from eval_utils import get_gold_docs

    data, _, hops = experiments.validated_dataset(
        source / f'{name}.json', source / f'{name}_corpus.json', dataset_name=name)
    preferred = SMOKE_POSITIONS.get(name)
    positions = (list(preferred) if preferred and max(preferred) < len(data) else
                 [hops.index(2), hops.index(4)] if 2 in hops and 4 in hops else
                 list(range(min(2, len(data)))))
    samples = [data[position] for position in positions]
    corpus = read_json(source / f'{name}_corpus.json')
    by_text = {experiments.passage_text(row): row for row in corpus}
    destination = out / 'smoke_datasets'
    dataset_file = destination / f'{name}.json'
    corpus_file = destination / f'{name}_corpus.json'
    provenance_file = out / 'metadata' / name / 'smoke_provenance.json'
    fixtures = (dataset_file, corpus_file, provenance_file)
    if any(path.exists() for path in fixtures):
        if not all(path.is_file() for path in fixtures):
            raise ValueError(f'{name}: incomplete smoke fixtures; inspect before retrying')
        provenance = read_json(provenance_file)
        expected = {
            'original_indices': positions,
            'original_sample_ids': [sample_id(data[i], i) for i in positions],
            'source_dataset_sha256': experiments.sha256(source / f'{name}.json'),
            'source_corpus_sha256': experiments.sha256(source / f'{name}_corpus.json'),
        }
        if any(provenance.get(key) != value for key, value in expected.items()):
            raise ValueError(f'{name}: smoke source provenance changed')
        saved = read_json(corpus_file)
        texts = [experiments.passage_text(row) for row in saved]
        if (read_json(dataset_file) != samples or len(saved) < 20
                or provenance.get('corpus_docs') != len(saved)
                or len(set(texts)) != len(texts)
                or any(row != by_text.get(text) for row, text in zip(saved, texts))):
            raise ValueError(f'{name}: existing smoke fixtures differ from their original sources')
        experiments.validated_dataset(dataset_file, corpus_file, dataset_name=name)
        manifest_file = out / 'metadata' / name / 'manifest.json'
        if manifest_file.is_file():
            manifest = read_json(manifest_file)
            if (experiments.sha256(dataset_file) != manifest['data_sha256']
                    or experiments.sha256(corpus_file) != manifest['corpus_sha256']):
                raise ValueError(f'{name}: existing smoke fixture SHA differs from the frozen manifest')
        # Preserve the exact corpus order and serialized bytes on resume.
        # A changed process hash seed must never rewrite a frozen fixture.
        return destination
    selected = {}
    for documents in get_gold_docs(samples, name):
        for document in sorted(documents):
            selected[document] = by_text[document]
    # Include the historical NV2 OOM passages and a longest intact passage.
    # Short-only fixtures cannot exercise full-length encoder memory peaks.
    if 'NV-Embed' in os.environ.get('HIPPO_EMBEDDING_MODEL_NAME', ''):
        historical = {'hotpotqa': 'Nashville Terminal Subdivision',
                      '2wikimultihopqa': 'Pattom A. Thanu Pillai'}
        title = historical.get(name)
        for row in corpus:
            text = experiments.passage_text(row)
            if title and text.partition('\n')[0].strip() == title:
                selected[text] = row
        longest = max(corpus, key=lambda row: len(experiments.passage_text(row)))
        selected[experiments.passage_text(longest)] = longest
    # A bounded smoke still uses complete original passages and original gold.
    for row in sorted(corpus, key=lambda item: (len(experiments.passage_text(item)),
                                               experiments.passage_text(item))):
        text = experiments.passage_text(row)
        if len(text.partition('\n')[2]) >= 80:
            selected[text] = row
        if len(selected) >= 20:
            break
    if len(selected) < 20:
        raise ValueError(f'{name}: smoke needs 20 unique source passages')
    destination.mkdir(parents=True, exist_ok=True)
    write_json(dataset_file, samples)
    write_json(corpus_file, list(selected.values()))
    write_json(provenance_file, {
        'original_indices': positions,
        'original_sample_ids': [sample_id(data[i], i) for i in positions],
        'source_dataset_sha256': experiments.sha256(source / f'{name}.json'),
        'source_corpus_sha256': experiments.sha256(source / f'{name}_corpus.json'),
        'corpus_docs': len(selected),
    })
    return destination


def layout(out, name):
    metadata = out / 'metadata' / name
    metadata.mkdir(parents=True, exist_ok=True)
    cases = out / 'cases' / name
    cases.mkdir(parents=True, exist_ok=True)
    alias = metadata / 'cases'
    if not alias.exists():
        alias.symlink_to(cases, target_is_directory=True)
    elif not alias.is_symlink() or alias.resolve() != cases.resolve():
        raise ValueError(f'Case directory alias differs: {alias}')
    (metadata / 'logs').mkdir(exist_ok=True)
    return metadata, out / 'shared_indexes' / name


def log_offset(path):
    if not path.is_file():
        raise ValueError(f'vLLM request log is missing: {path}')
    return path.stat().st_size


def frozen_hashes(index):
    candidates = list(index.glob('*/index_manifest.json'))
    if len(candidates) != 1:
        raise ValueError(f'Expected one model-specific index manifest under {index}')
    directory = candidates[0].parent
    hashes = {name: experiments.sha256(directory / name) for name in ALL_ASSETS}
    hashes['openie_results_ner_qwen3-8b.json'] = experiments.sha256(index / 'openie_results_ner_qwen3-8b.json')
    return hashes


def commands(args, metadata, index, datasets, name):
    """Construct the one index build and final-method retrieval commands."""
    options = ['--openie_strict', str(args.openie_strict).lower(),
               '--openie_prompt_version', args.openie_prompt_version,
               '--openie_validation_mode', getattr(args, 'openie_validation_mode', 'structural')]
    model = getattr(args, 'embedding_model', experiments.EMBEDDING_MODEL)
    provider = getattr(args, 'embedding_provider', experiments.DEFAULT_EMBEDDING_PROVIDER)
    batch = str(getattr(args, 'embedding_batch_size', 4))
    common = ['--dataset', name, '--sample_size', '2' if args.smoke else '0',
              '--sample_seed', '42', '--sample_indices_file', str(metadata / 'selected_indices.json'),
              '--llm_name', 'qwen3-8b', '--llm_base_url', args.llm_base_url,
              '--embedding_batch_size', batch, '--openie_max_workers', '8',
              '--llm_prefetch_workers', '8']
    builder = [args.python, '-B', '-u', str(ROOT / 'scripts/build_shared_index.py'),
        *common, *options, '--datasets_dir', str(datasets), '--rag_type', 'hipporag',
        '--embedding_name', model, '--embedding_provider', provider,
        '--openie_mode', 'online', '--eval_mode', 'index_only',
        '--force_index_from_scratch', 'true', '--force_openie_from_scratch', 'true',
        '--save_dir_exact', '--save_dir', str(index),
        '--output', str(metadata / 'index_build_result.json')]
    case = metadata / 'cases' / CASE_NAME
    retrieval = [args.python, '-B', '-u', str(ROOT / 'scripts/eval_dataset.py'),
        *common, *options, *PC_ARGUMENTS, '--eval_mode', 'retrieve',
        '--retrieval_top_k', '200', '--result_top_k', '10', '--candidate_output_top_k', '200',
        '--reuse_index', '--data_path', str(datasets / f'{name}.json'),
        '--corpus_path', str(datasets / f'{name}_corpus.json'), '--corpus_mode', 'full',
        '--qa_top_k', '5', '--max_qa_steps', '1', '--max_new_tokens', '2048', '--embedding_model_name', model,
        '--improvement_stage', '4',
        '--evidence_improvements', 'planning,plan_prune,dag_package,support_semantic_veto',
        '--evidence_plan_validation', 'canonical_refs', '--evidence_plan_routing', 'question_structure',
        '--stratified_eval', '--stratified_output', str(case / 'stratified.json'),
        '--save_dir', str(case / 'index'), '--output', str(case / 'result.json')]
    return builder, {CASE_NAME: retrieval}


def run(args):
    args.embedding_model = getattr(args, 'embedding_model', experiments.EMBEDDING_MODEL)
    args.embedding_provider = getattr(args, 'embedding_provider', experiments.DEFAULT_EMBEDDING_PROVIDER)
    args.embedding_batch_size = getattr(args, 'embedding_batch_size', 4)
    if args.embedding_batch_size != 4:
        raise ValueError('The formal workflow keeps embedding batch=4')
    out = Path(args.out_root).resolve()
    if not out.is_relative_to(ROOT / 'outputs') or out == ROOT / 'outputs':
        raise ValueError('Output must be a dedicated directory under BindPack/outputs')
    if args.smoke and 'smoke' not in out.name:
        raise ValueError('Smoke output must have "smoke" in its name, separate from full indexes')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'logs').mkdir(exist_ok=True)
    (out / 'shared_indexes').mkdir(exist_ok=True)
    if shutil.disk_usage(out).free < 5 * 1024 ** 3:
        raise ValueError('At least 5 GiB of free disk is needed')
    with urllib.request.urlopen(args.llm_base_url.rstrip('/') + '/models', timeout=10) as response:
        if response.status != 200:
            raise ValueError('vLLM service is not ready')
    server_log = Path(args.vllm_log)
    log_offset(server_log)
    baseline_code = code_hashes(Path(args.hippo_root))
    env = environment(args)
    env.update(HIPPO_ALLOW_INDEX_RESUME='1', HIPPO_EMBEDDING_MODEL_NAME=args.embedding_model,
               HIPPO_EMBEDDING_BASE_URL='')
    statuses = {}
    protocol = {'datasets': list(DATASETS), 'method': CASE_NAME, 'smoke': args.smoke,
                'openie_strict': args.openie_strict, 'openie_prompt_version': args.openie_prompt_version,
                'openie_validation_mode': getattr(args, 'openie_validation_mode', 'structural'),
                'embedding_model': args.embedding_model, 'embedding_provider': args.embedding_provider,
                'embedding_batch_size': 4, 'llm_name': 'qwen3-8b', 'llm_workers': 8,
                'ner_max_tokens': 512, 'triple_max_tokens': 2048, 'enable_thinking': False}
    protocol_file = out / 'metadata' / 'final_index_protocol.json'
    protocol_file.parent.mkdir(exist_ok=True)
    if protocol_file.exists() and read_json(protocol_file) != protocol:
        raise ValueError('Existing index protocol differs; use a different output directory')
    write_json(protocol_file, protocol)

    def record(stage, state, **details):
        statuses[stage] = {'state': state, **details}
        write_json(out / 'metadata' / 'shared_index_stage_status.json', statuses)
        LOG.info('[index-stage] %s %s %s', stage, state, details)

    for name in DATASETS:
        stage = f'{name}/index'
        try:
            metadata, index = layout(out, name)
            datasets = smoke_dataset(out, Path(args.datasets_dir), name) if args.smoke else Path(args.datasets_dir)
            options = SimpleNamespace(out_root=str(metadata), source_index=str(index),
                shared_output_root=str(out), dataset=name,
                data_path=str(datasets / f'{name}.json'), corpus_path=str(datasets / f'{name}_corpus.json'),
                sample_size=2 if args.smoke else 0, sample_seed=42, sample_indices_file=None,
                cases=CASE_NAME, build_shared_index=True,
                embedding_model=args.embedding_model, embedding_provider=args.embedding_provider,
                embedding_batch_size=4)
            experiments.prepare(options)
            builder, _ = commands(args, metadata, index, datasets, name)
            _, documents, _ = experiments.validated_dataset(options.data_path, options.corpus_path, dataset_name=name)
            from .fresh_index_validation import validate_fresh_index
            if experiments.index_ready(options) != 0:
                start = log_offset(server_log)
                record(stage, 'running', source_index=str(index))
                elapsed = execute(builder, out / 'logs' / f'{name}_index_build.log', env)
                validate_fresh_index(metadata, index, documents, strict=args.openie_strict)
                experiments.freeze_index(SimpleNamespace(out_root=str(metadata), elapsed=elapsed,
                    vllm_log=str(server_log), log_start=start, log_end=log_offset(server_log),
                    openie_strict=args.openie_strict))
            else:
                validate_fresh_index(metadata, index, documents, strict=args.openie_strict)
            report = read_json(metadata / 'index_build_report.json')
            record(stage, 'ready', source_index=str(index),
                   openie_failure_count=report.get('openie_failure_count', 0), openie_strict=args.openie_strict)
        except Exception as error:
            LOG.exception('[failed] %s', stage)
            record(stage, 'failed', error=f'{type(error).__name__}: {error}')
    if code_hashes(Path(args.hippo_root)) != baseline_code:
        raise ValueError('Original HippoRAG source changed during index building')
    failed = {stage: value for stage, value in statuses.items() if value['state'] != 'ready'}
    failure_path = out / 'metadata' / 'shared_index_failed_stages.json'
    if failed:
        write_json(failure_path, failed)
        return 1
    failure_path.unlink(missing_ok=True)
    write_json(out / 'indexes_completed.ok', {'datasets': list(DATASETS), 'smoke': args.smoke,
                                             'embedding_model': args.embedding_model, 'embedding_batch_size': 4})
    if getattr(args, 'indexes_only', False):
        return 0
    from .multi_dataset_retrieval_support_semantic_veto import run as retrieve
    args.smoke_out_root = None
    return retrieve(args)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-root')
    parser.add_argument('--datasets-dir', default='/root/datasets')
    parser.add_argument('--hippo-root', default='/root/baseline/HippoRAG')
    parser.add_argument('--runtime-deps', default=str(DEFAULT_RUNTIME))
    parser.add_argument('--python', default='/root/anaconda3/envs/rag/bin/python')
    parser.add_argument('--llm-base-url', default='http://127.0.0.1:8035/v1')
    parser.add_argument('--vllm-log')
    parser.add_argument('--embedding-model', default=experiments.EMBEDDING_MODEL)
    parser.add_argument('--embedding-provider', default=experiments.DEFAULT_EMBEDDING_PROVIDER)
    parser.add_argument('--embedding-batch-size', type=int, choices=[4], default=4)
    parser.add_argument('--openie-strict', type=parse_bool, default=False)
    parser.add_argument('--openie-prompt-version', choices=('origin', 'optimized'), default='optimized')
    parser.add_argument('--openie-validation-mode', choices=('structural', 'source_verified'), default='structural')
    parser.add_argument('--indexes-only', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(argv)
    args.out_root = args.out_root or str(DEFAULT_OUT.with_name(DEFAULT_OUT.name + '_smoke')
                                        if args.smoke else DEFAULT_OUT)
    from .http_audit import locate
    args.vllm_log = args.vllm_log or locate(args.llm_base_url)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    return run(args)

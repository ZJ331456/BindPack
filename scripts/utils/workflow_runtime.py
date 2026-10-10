"""Runtime preparation and bounded subprocess execution for formal runs."""
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from . import workflow_artifacts as experiments

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / 'outputs/3multi_hop_datasets_results_10_5'
_runtime_candidates = (
    Path('/root/.cache/bindpack/runtime_deps_qwen3_tf4513'),
    Path('/root/.cache/pathcondrag/runtime_deps_qwen3_tf4513'),
)
DEFAULT_RUNTIME = next((path for path in _runtime_candidates if (path / 'transformers').is_dir()),
                       _runtime_candidates[0])
LOG = logging.getLogger('bindpack.workflow')
ALL_ASSETS = experiments.ASSETS + ('openie_state.json', 'index_manifest.json', 'chunk_metadata.json')
PC_ARGUMENTS = [
    '--use_iterative_retrieval', '--iterative_round1_top_docs', '1',
    '--iterative_round2_seed_top_k', '5',
    '--iterative_merge_alpha', '0.45', '--iterative_min_seed_entities', '1',
    '--use_query_decomposition', '--use_path_conditioned_qd', '--qd_min_hops', '2',
    '--qd_sub_retrieval_top_k', '3', '--pcqd_ground_top_docs', '5', '--pcqd_entity_top_k', '3',
    '--pcqd_path_score_threshold', '0.60', '--pcqd_weight_base', '0.40',
    '--pcqd_weight_static', '0.20', '--pcqd_weight_path', '0.40',
    '--hop_source', 'benchmark', '--qd_max_sub_questions', '4',
]

def code_hashes(hippo_root):
    listing = subprocess.run(['git', '-C', str(hippo_root), 'ls-files', '-z', '--', '*.py'],
                             check=True, capture_output=True).stdout
    return {name.decode(): experiments.sha256(hippo_root / name.decode())
            for name in listing.split(b'\0') if name and (hippo_root / name.decode()).is_file()}


def qwen_environment(args):
    env = os.environ.copy()
    runtime = Path(args.runtime_deps).resolve()
    if not (runtime / 'transformers').is_dir():
        raise ValueError(f'Qwen3-compatible isolated runtime is missing: {runtime}')
    env['PYTHONPATH'] = str(runtime) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false',
               OPENAI_API_KEY=env.get('OPENAI_API_KEY', 'EMPTY'),
               PATHCONDRAG_LLM_MAX_IN_FLIGHT='8', HIPPORAG_LLM_MAX_IN_FLIGHT='8',
               HIPPO_OPENIE_MAX_WORKERS='8', HIPPO_OPENIE_NER_WORKERS='8',
               HIPPO_OPENIE_TRIPLE_WORKERS='8', HIPPO_OPENIE_NER_MAX_TOKENS='512',
               HIPPO_OPENIE_TRIPLE_MAX_TOKENS='2048', HIPPO_OPENIE_QUALITY_MAX_RETRIES='2',
               PATHCONDRAG_SHARED_KNN_DEVICE='cuda', HIPPORAG_KNN_DEVICE='cuda')
    if (Path(args.out_root) / 'shared_hipporag2_index').exists():
        env['HIPPO_ALLOW_INDEX_RESUME'] = '1'
    return env


def execute(command, logfile, env):
    LOG.info('[command] %s', ' '.join(command))
    started = time.monotonic()
    if logfile.is_file() and logfile.stat().st_size:
        previous = logfile.with_name(f'{logfile.stem}.attempt_{time.time_ns()}.log')
        logfile.rename(previous)
        LOG.info('[previous-attempt-log] %s', previous)
    with logfile.open('w', encoding='utf-8') as stream:
        with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                              errors='replace', bufsize=1) as process:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
            if process.wait() != 0:
                raise subprocess.CalledProcessError(process.returncode, command)
    return int(time.monotonic() - started)


def environment(args):
    """Keep NV-Embed in its own runtime; Qwen embeddings need isolated deps."""
    model = getattr(args, 'embedding_model', experiments.EMBEDDING_MODEL)
    provider = getattr(args, 'embedding_provider', experiments.DEFAULT_EMBEDDING_PROVIDER)
    if provider != 'nvembed' and 'NV-Embed' not in model:
        env = qwen_environment(args)
        env.update(HIPPO_EMBEDDING_MODEL_NAME=model, HIPPO_EMBEDDING_BASE_URL='',
                   HIPPO_ROOT=getattr(args, 'hippo_root', '/root/baseline/HippoRAG'))
        return env
    env = os.environ.copy()
    # Never put Qwen3's transformers/accelerate bundle before the NV runtime.
    parts = [part for part in env.get('PYTHONPATH', '').split(os.pathsep)
             if part and 'runtime_deps_qwen3' not in part
             and Path(part).resolve() != DEFAULT_RUNTIME.resolve()]
    if parts:
        env['PYTHONPATH'] = os.pathsep.join(parts)
    else:
        env.pop('PYTHONPATH', None)
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', PYTHONHASHSEED='42',
               TOKENIZERS_PARALLELISM='false',
               OPENAI_API_KEY=env.get('OPENAI_API_KEY', 'EMPTY'),
               PATHCONDRAG_LLM_MAX_IN_FLIGHT='8', HIPPORAG_LLM_MAX_IN_FLIGHT='8',
               HIPPO_OPENIE_MAX_WORKERS='8', HIPPO_OPENIE_NER_WORKERS='8',
               HIPPO_OPENIE_TRIPLE_WORKERS='8', HIPPO_OPENIE_NER_MAX_TOKENS='512',
               HIPPO_OPENIE_TRIPLE_MAX_TOKENS='2048', HIPPO_OPENIE_QUALITY_MAX_RETRIES='2',
               PATHCONDRAG_SHARED_KNN_DEVICE='cpu', HIPPORAG_KNN_DEVICE='cpu',
               HIPPO_EMBEDDING_MODEL_NAME=model, HIPPO_EMBEDDING_BASE_URL='',
               HIPPO_ROOT=getattr(args, 'hippo_root', '/root/baseline/HippoRAG'))
    env['PATHCONDRAG_NVEMBED_OOM_SPLIT'] = env.get('PATHCONDRAG_NVEMBED_OOM_SPLIT', 'false')
    return env

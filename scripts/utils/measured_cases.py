"""Frozen measured-case orchestration and independent copies of index assets."""
from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import time

CASES = ('full', 'wo_route', 'wo_plan', 'wo_pack')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def index_assets(index):
    """No LLM/query caches, OpenIE checkpoint DBs or rebuildable Route caches."""
    index = Path(index).resolve()
    result = {}
    for path in sorted(index.rglob('*')):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(index)
        if 'llm_cache' in relative.parts:
            continue
        keep = (path.suffix == '.parquet' or path.name == 'graph.pickle'
                or path.name.startswith('openie_results_ner_') and path.suffix == '.json'
                or path.name in ('openie_publication_report.json', 'index_manifest.json'))
        if keep:
            result[str(relative)] = dict(bytes=path.stat().st_size, sha256=sha(path))
    if (not any(name.endswith('graph.pickle') for name in result)
            or not any(name.endswith('vdb_chunk.parquet') for name in result)):
        raise ValueError(f'Incomplete native index assets: {index}')
    return result


def copy_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name('.' + target.name + '.copy.tmp')
    try:
        try:
            with source.open('rb') as src, tmp.open('wb') as dst:
                fcntl.ioctl(dst.fileno(), 0x40049409, src.fileno())
            shutil.copystat(source, tmp)
        except OSError:
            shutil.copy2(source, tmp)
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)


def prepare_reused_index(source, destination, case, identity, models=None):
    source, destination, case = Path(source).resolve(), Path(destination).resolve(), Path(case)
    if source == destination or source in destination.parents:
        raise ValueError('Private index must be outside the source index')
    source_case = source.parent
    marker = read(source_case / 'completed.ok')
    if marker['report_sha256'] != sha(source_case / 'report.json'):
        raise ValueError('Source case has no matching completed report')
    source_protocol = read(source_case / 'protocol.json')
    if models is not None and source_protocol['models'] != models:
        raise ValueError('Shared index model identity mismatch')
    source_identity = source_protocol['dataset']
    for key in ('dataset', 'data_sha256', 'corpus_docs_sha256', 'selected_indices', 'sample_ids'):
        if source_identity[key] != identity[key]:
            raise ValueError('Source index input identity mismatch: ' + key)
    assets = index_assets(source)
    receipt_path = case / 'index_reuse.json'
    if receipt_path.exists():
        old = read(receipt_path)
        if old['source'] != str(source) or old['assets'] != assets:
            raise ValueError('Shared original index changed')
    started = time.perf_counter()
    for relative, metadata in assets.items():
        target = destination / relative
        if target.is_symlink():
            raise ValueError('Refusing symlinked writable assets')
        if target.exists():
            if sha(target) != metadata['sha256']:
                raise ValueError('Existing private asset changed: ' + str(target))
        else:
            copy_file(source / relative, target)
        if sha(target) != metadata['sha256']:
            raise ValueError('Private copy failed validation: ' + str(target))
        if target.stat().st_ino == (source / relative).stat().st_ino:
            raise ValueError('Private asset must not be a writable hard link')
    original = read(source_case / 'index_build.json')
    report = read(source_case / 'report.json')
    receipt = dict(source=str(source), assets=assets, source_build=original,
        original_build_usage=report['cost']['offline_index'],
        current_copy_and_verify_seconds=time.perf_counter()-started,
        policy='independent private copies; no LLM/query/lookup cache inherited; original build charged once to the shared index')
    write(receipt_path, receipt)
    build_path = case / 'index_build.json'
    if not build_path.exists():
        write(build_path, dict(original, reused_from=str(source), original_build_usage=report['cost']['offline_index']))
    return receipt


def install_ablation_snapshot(root, control_scripts):
    root, control_scripts = Path(root), Path(control_scripts)
    original = root / 'source_snapshot'
    overlay = root / 'ablation_source_snapshot'
    allowed = {'scripts/run_bindpack_measured.py', 'scripts/utils/measured_cases.py'}
    if not overlay.exists():
        shutil.copytree(original, overlay, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for name in allowed:
            target = overlay / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(control_scripts / Path(name).relative_to('scripts'), target)
    old_files = {str(p.relative_to(original)): sha(p) for folder in ('src', 'scripts')
                 for p in (original / folder).rglob('*.py')}
    new_files = {str(p.relative_to(overlay)): sha(p) for folder in ('src', 'scripts')
                 for p in (overlay / folder).rglob('*.py')}
    changed = {name for name in old_files.keys() | new_files.keys()
               if old_files.get(name) != new_files.get(name)}
    if not changed <= allowed:
        raise ValueError('Ablation overlay changed native algorithm sources: ' + str(changed))
    for name in allowed:
        if sha(overlay / name) != sha(control_scripts / Path(name).relative_to('scripts')):
            raise ValueError('Frozen ablation control code differs from the queue')
    write(root / 'serial_extension/ablation_overlay.json', dict(
        base_snapshot=str(original), ablation_snapshot=str(overlay),
        changed_files=sorted(changed), native_sources_byte_identical=True,
        full_snapshot_untouched=True, overlay_sources=new_files))
    return overlay


def verify_prefixes(root):
    root = Path(root)
    path = root / 'serial_extension/preservation.json'
    if not path.exists():
        return None
    proof = read(path)
    case = root / 'cases/hover/full'
    for name in ('progress.jsonl', 'llm_calls.jsonl', 'llm_logical_calls.jsonl', 'embedding_calls.jsonl',
                 'resource_samples.jsonl', 'stage_timings.jsonl', 'execution_sessions.jsonl'):
        item = proof['files'][name]
        with (case / name).open('rb') as stream:
            prefix = stream.read(item['bytes'])
        if hashlib.sha256(prefix).hexdigest() != item['sha256']:
            raise ValueError('Preserved Full prefix changed: ' + name)
    for name in ('protocol.json', 'index_build.json'):
        if sha(case / name) != proof['files'][name]['sha256']:
            raise ValueError('Preserved Full receipt changed: ' + name)
    for name, metadata in proof['index_assets'].items():
        if sha(case / 'index' / name) != metadata['sha256']:
            raise ValueError('Preserved Full index changed: ' + name)
    return proof['completed_queries']

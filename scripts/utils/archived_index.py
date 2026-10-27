"""Verify archived corpus indexes and make private query-only working copies."""
from __future__ import annotations

from pathlib import Path
import shutil
import time

from .measured_cases import copy_file, read, sha, write


def validate_archive(source, dataset, documents, embedding_model, llm_name, embedding_settings=None):
    """Index identity depends on exact passage contents/models, not query hops."""
    import pyarrow.parquet as pq
    source=Path(source).resolve();archive=source.parent.parent
    if source.parent.name!='shared_indexes' or source.name!=dataset:
        raise ValueError('Expected archive/shared_indexes/<dataset>')
    inventory_path=archive/'manifest.json';inventory=read(inventory_path)
    known={r['path']:r for r in inventory['files']}
    def recorded(path):
        entry=known.get(str(path.relative_to(archive)))
        if entry is None or not path.is_file() or sha(path)!=entry['sha256']:
            raise ValueError('Unverified archived provenance: '+str(path))
        return read(path)
    model_name=llm_name.replace('/','_')+'_'+embedding_model.replace('/','_')
    model=source/model_name
    manifest_path=model/'index_manifest.json'
    original_build={};original_usage=None;model_proof=[]
    if manifest_path.is_file():
        manifest=read(manifest_path)
    else:
        if not model.is_dir() or str(manifest_path.relative_to(archive)) in known:
            raise ValueError('Archived index model mismatch or missing model manifest')
        # The original measured HoVer index predates index_manifest.json.
        # Its immutable completed native Full protocol is an independent proof.
        full=archive/'cases'/dataset/'full'
        proto=recorded(full/'protocol.json');done=recorded(full/'completed.ok')
        report=recorded(full/'report.json');original_build=recorded(full/'index_build.json')
        if done['report_sha256']!=sha(full/'report.json') or proto.get('method')!='bindpack' or proto.get('stage_ablation')!='full':
            raise ValueError('Invalid measured Full provenance')
        cfg=proto['configuration']
        if proto['dataset']['dataset']!=dataset:raise ValueError('Measured Full dataset mismatch')
        manifest={'embedding':{'model_name':cfg['embedding_model_name'],
            'normalized':cfg['embedding_return_as_normalized'],'max_sequence_length':cfg['embedding_max_seq_len'],
            'dtype':cfg['embedding_model_dtype']},'openie':{'identity':{'model_name':cfg['llm_name']}}}
        original_usage=report.get('cost',{}).get('offline_index')
        model_proof=[dict(path=str(full/name),sha256=sha(full/name)) for name in ('protocol.json','completed.ok','report.json','index_build.json')]
    if manifest['embedding']['model_name']!=embedding_model or manifest['openie']['identity']['model_name']!=llm_name:
        raise ValueError('Archived index model mismatch')
    if embedding_settings is not None:
        for key,value in embedding_settings.items():
            if manifest['embedding'].get(key)!=value:raise ValueError('Archived embedding configuration mismatch: '+key)
    required=[model/'graph.pickle',
        model/'chunk_embeddings/vdb_chunk.parquet',model/'entity_embeddings/vdb_entity.parquet',model/'fact_embeddings/vdb_fact.parquet']
    if manifest_path.exists():required.append(manifest_path)
    for p in required:
        if not p.is_file() or p.is_symlink():raise ValueError('Missing independent archived asset: '+str(p))
    assets={}
    selected=required+[p for p in source.glob('openie_results_ner_*.json')]
    selected += [p for p in (model/'chunk_metadata.json',model/'openie_publication_report.json') if p.is_file()]
    if not any(p.name.startswith('openie_results_ner_') for p in selected):raise ValueError('Missing archived OpenIE documents')
    for p in selected:
        relative=str(p.relative_to(source));entry=known.get(str(p.relative_to(archive)))
        actual=sha(p)
        if p.is_symlink() or entry is None or actual!=entry['sha256'] or p.stat().st_size!=entry['bytes']:
            raise ValueError('Archived asset differs from the immutable inventory: '+str(p))
        assets[relative]=dict(bytes=p.stat().st_size,sha256=actual)
    chunks=pq.read_table(model/'chunk_embeddings/vdb_chunk.parquet',columns=['content'])['content'].to_pylist()
    if len(chunks)!=len(set(chunks)) or len(documents)!=len(set(documents)) or set(chunks)!=set(documents):
        raise ValueError('Archived index passages differ from the full current corpus')
    from bindpack.evaluation.measurement import digest
    if original_build and original_build['corpus_sha256']!=digest(documents):
        raise ValueError('Measured source corpus fingerprint differs')
    return dict(source=str(source),dataset=dataset,assets=assets,archive_inventory_sha256=sha(inventory_path),
        corpus_docs=len(documents),corpus_sha256=digest(documents),indexed_passage_order_sha256=digest(chunks),
        models=dict(llm=llm_name,embedding=embedding_model),
        identity_policy='exact full passage texts and model identity; query annotations/hop budgets are separately frozen',
        archived_model_provenance=model_proof,
        original_build_usage=original_usage,original_build_seconds=original_build.get('seconds'),
        original_build_unavailable_reason=('historical measured Full index build; reference only, not charged again'
            if original_build else 'archived graph/vectors are verified; a measured original build ledger is unavailable'))


def prepare_archive(source,destination,case,identity,documents,config):
    source,destination,case=Path(source).resolve(),Path(destination).resolve(),Path(case)
    if source==destination or source in destination.parents:raise ValueError('Private index overlaps the archived index')
    start=time.perf_counter()
    settings=dict(normalized=config.embedding_return_as_normalized,max_sequence_length=config.embedding_max_seq_len,dtype=config.embedding_model_dtype)
    receipt=validate_archive(source,identity['dataset'],documents,config.embedding_model_name,config.llm_name,settings)
    marker=case/'index_reuse.json'
    if marker.exists():
        previous=read(marker)
        for key in ('source','assets','corpus_sha256','models'):
            if previous[key]!=receipt[key]:raise ValueError('Frozen archived index identity changed: '+key)
    destination.mkdir(parents=True,exist_ok=True)
    needed=sum(item['bytes'] for relative,item in receipt['assets'].items() if not (destination/relative).exists())
    if shutil.disk_usage(destination).free < needed + 1024**3:
        raise OSError('Insufficient free space for private index copies and the result/checkpoint reserve')
    for relative,expected in receipt['assets'].items():
        src=source/relative;target=destination/relative
        if target.exists():
            if target.is_symlink() or sha(target)!=expected['sha256']:raise ValueError('Private index asset changed: '+str(target))
        else:copy_file(src,target)
        if sha(target)!=expected['sha256'] or target.stat().st_ino==src.stat().st_ino:
            raise ValueError('Invalid private copy of archived asset: '+str(target))
    receipt.update(current_copy_and_verify_seconds=time.perf_counter()-start,
        policy='independent graph/vector/OpenIE copies; no historical LLM, query-vector or lookup cache inherited')
    write(marker,receipt)
    build=dict(seconds=receipt['original_build_seconds'],corpus_sha256=receipt['corpus_sha256'],index_bytes=sum(v['bytes'] for v in receipt['assets'].values()),
        passages=len(documents),graph_nodes=None,graph_edges=None,reused_from=str(source),original_build_usage=receipt['original_build_usage'],
        new_index_build_calls=0,original_build_unavailable_reason=receipt['original_build_unavailable_reason'])
    if (case/'index_build.json').exists():
        if read(case/'index_build.json')!=build:raise ValueError('Archived index build receipt changed')
    else:write(case/'index_build.json',build)
    return receipt


def verify_private(case, *, release=False):
    """Verify source and private inputs before releasing only verified copies."""
    case=Path(case);receipt=read(case/'index_reuse.json');source=Path(receipt['source']);private=case/'index'
    for relative,expected in receipt['assets'].items():
        if sha(source/relative)!=expected['sha256']:raise ValueError('Archived source asset changed: '+relative)
        path=private/relative
        if not path.is_file() or path.is_symlink() or sha(path)!=expected['sha256']:
            raise ValueError('Private query-only index asset changed: '+relative)
    write(case/'index_reuse_verified.ok',dict(source_and_private_unchanged=True,assets=len(receipt['assets'])))
    if release:
        for relative in receipt['assets']:(private/relative).unlink()
        write(case/'private_assets_released.ok',dict(released_assets=len(receipt['assets']),source_unchanged=True,
            policy='release only verified graph/vector/OpenIE copies; retain checkpoints, costs and independent LLM/query caches'))

"""Freeze current shared indexes and make independent query-only working copies.

These assets were unpacked without the old archive inventory.  This module
records their present bytes and checks exact ordered passage identity; it does
not claim to authenticate the original build, and never builds an index.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

from .measured_cases import copy_file, read, sha, write


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _paths(source, model_name, llm_name):
    model = source / model_name
    if not model.is_dir() or model.is_symlink():
        raise ValueError("Missing independent shared model directory: " + str(model))
    required = [model / "graph.pickle",
        model / "chunk_embeddings/vdb_chunk.parquet",
        model / "entity_embeddings/vdb_entity.parquet",
        model / "fact_embeddings/vdb_fact.parquet",
        source / ("openie_results_ner_" + llm_name.replace("/", "_") + ".json")]
    optional = [model / name for name in ("index_manifest.json",
        "chunk_metadata.json", "openie_publication_report.json")]
    selected = required + [path for path in optional if path.exists()]
    for path in selected:
        if (not path.is_file() or path.is_symlink()
                or any(parent.is_symlink() for parent in path.parents
                       if parent != source and source in parent.parents)):
            raise ValueError("Missing independent shared asset: " + str(path))
    return sorted(selected)


def prepare_shared_index(source, destination, receipt_path, dataset, documents, config):
    """Verify/freeze source bytes and copy assets without inheriting query caches.

    ``documents`` must be the full corpus after the existing gold-completion
    policy.  In particular HoVer has 9441 indexed passages, including its one
    previously completed gold passage.  Resume permits a private runtime
    ``index_manifest.json`` to change, while graph/vector/OpenIE bytes remain
    identical to the frozen source.
    """
    import pyarrow.parquet as pq

    started = time.perf_counter()
    raw_source, raw_destination = Path(source), Path(destination)
    if raw_source.is_symlink() or raw_destination.is_symlink():
        raise ValueError("Shared/private index roots must be independent directories")
    source, destination = raw_source.resolve(), raw_destination.resolve()
    receipt_path = Path(receipt_path).resolve()
    if (source == destination or source in destination.parents
            or destination in source.parents):
        raise ValueError("Private index overlaps the shared source")
    if receipt_path == source or source in receipt_path.parents:
        raise ValueError("Shared source cannot contain the run's writable receipt")
    if source.name != dataset or source.parent.name != "shared_indexes":
        raise ValueError("Expected shared_indexes/<dataset>")
    llm_name, embedding_model = config.llm_name, config.embedding_model_name
    model_name = llm_name.replace("/", "_") + "_" + embedding_model.replace("/", "_")
    selected = _paths(source, model_name, llm_name)
    model = source / model_name
    manifest_path = model / "index_manifest.json"
    manifest_present = manifest_path.is_file()
    embedding_settings = dict(normalized=config.embedding_return_as_normalized,
        max_sequence_length=config.embedding_max_seq_len, dtype=config.embedding_model_dtype)
    if manifest_present:
        manifest = read(manifest_path)
        if (manifest.get("embedding", {}).get("model_name") != embedding_model
                or manifest.get("openie", {}).get("identity", {}).get("model_name") != llm_name):
            raise ValueError("Shared index model identity mismatch")
        for name, expected in embedding_settings.items():
            if manifest["embedding"].get(name) != expected:
                raise ValueError("Shared index embedding setting mismatch: " + name)
        identity_scope = "original index manifest model names and embedding settings"
    else:
        if dataset != "hover":
            raise ValueError("Missing shared index model manifest: " + str(manifest_path))
        identity_scope = ("legacy HoVer directory namespace and OpenIE filename only; "
            "original model/embedding-settings manifest is unavailable")

    chunks = pq.read_table(model / "chunk_embeddings/vdb_chunk.parquet",
        columns=["content"])["content"].to_pylist()
    documents = list(documents)
    if (not documents or len(documents) != len(set(documents))
            or len(chunks) != len(set(chunks)) or chunks != documents):
        raise ValueError("Shared index passages differ from the exact ordered full corpus")
    publication = model / "openie_publication_report.json"
    if publication.is_file() and read(publication).get("document_count") != len(documents):
        raise ValueError("Shared OpenIE publication passage count mismatch")
    assets = {str(path.relative_to(source)):
        dict(bytes=path.stat().st_size, sha256=sha(path)) for path in selected}
    receipt = dict(schema="bindpack_current_shared_index_v1", source=str(source),
        destination=str(destination), dataset=dataset, model_directory=model_name,
        models=dict(llm=llm_name, embedding=embedding_model),
        requested_embedding_settings=embedding_settings,
        source_index_manifest_present=manifest_present,
        original_archive_inventory_present=(source.parent.parent / "manifest.json").is_file(),
        model_identity_scope=identity_scope,
        provenance_limitation="Current bytes are frozen at this run; no historical archive inventory was supplied or authenticated.",
        corpus_docs=len(documents), corpus_sha256=_digest(documents),
        indexed_passage_order_sha256=_digest(chunks), assets=assets,
        mutable_private_assets=[str(Path(model_name) / "index_manifest.json")],
        policy="Independent CoW clones or physical copies; no hard links, index construction, historical LLM cache, or query cache reuse.",
        new_index_build_calls=0)
    old = read(receipt_path) if receipt_path.exists() else None
    if old is not None:
        for key, expected in receipt.items():
            if old.get(key) != expected:
                raise ValueError("Frozen shared index receipt changed: " + key)
    destination.mkdir(parents=True, exist_ok=True)
    for relative, expected in assets.items():
        src, target = source / relative, destination / relative
        if target.is_symlink() or any(parent.is_symlink() for parent in target.parents
                if parent != destination and destination in parent.parents):
            raise ValueError("Refusing a symlinked private asset: " + str(target))
        if target.exists():
            if not target.is_file() or target.samefile(src):
                raise ValueError("Private shared-index asset is not independent: " + str(target))
            if relative not in receipt["mutable_private_assets"] and (
                    target.stat().st_size != expected["bytes"] or sha(target) != expected["sha256"]):
                raise ValueError("Private graph/vector/OpenIE asset changed: " + relative)
        else:
            copy_file(src, target, reserve_bytes=1024 ** 3)
            if (target.samefile(src)
                    or target.stat().st_size != expected["bytes"] or sha(target) != expected["sha256"]):
                raise ValueError("Invalid private shared-index copy: " + relative)
    if old is None:
        receipt["initial_prepare_seconds"] = time.perf_counter() - started
        write(receipt_path, receipt)
    else:
        receipt = old
    return receipt


def verify_shared_index(receipt_path):
    """Verify frozen source assets and private immutable data after execution."""
    receipt = read(receipt_path)
    source, destination = Path(receipt["source"]), Path(receipt["destination"])
    if source.is_symlink() or destination.is_symlink():
        raise ValueError("Shared/private index root became a symlink")
    selected = _paths(source, receipt["model_directory"], receipt["models"]["llm"])
    if {str(path.relative_to(source)) for path in selected} != set(receipt["assets"]):
        raise ValueError("Shared source asset inventory changed")
    for relative, expected in receipt["assets"].items():
        src, target = source / relative, destination / relative
        if src.stat().st_size != expected["bytes"] or sha(src) != expected["sha256"]:
            raise ValueError("Shared source asset changed: " + relative)
        if relative in receipt["mutable_private_assets"]:
            continue
        if (not target.is_file() or target.is_symlink()
                or target.samefile(src)
                or target.stat().st_size != expected["bytes"] or sha(target) != expected["sha256"]):
            raise ValueError("Private graph/vector/OpenIE asset changed: " + relative)
    return dict(source_unchanged=True, private_immutable_assets_unchanged=True,
        source_assets=len(receipt["assets"]), new_index_build_calls=0)

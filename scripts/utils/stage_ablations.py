"""Four whole-stage controls on one frozen Qwen3 index per dataset.

Algorithm values live in configuration/defaults.json. This runner controls resources,
sample identity, artifact isolation, and resumable experiment bookkeeping.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request

from eval_utils import (build_docs_from_full_corpus, build_docs_from_samples,
                        get_benchmark_hops, get_gold_docs, describe_benchmark_hops,
                        get_query_text, get_sample_id, get_gold_answers, get_dataset_task)
from report_metrics import write_metric_report
from utils.reporting_protocol import persist_reporting_receipt, resolve_existing_protocol

ROOT = Path(__file__).resolve().parents[2]
CASES = ("full", "wo_route", "wo_plan", "wo_pack")
HIPPO_CASE = "hippporag2_reproduce"
BASELINE_MAIN = ROOT.parent / "baseline" / "HippoRAG" / "main.py"
DATASETS = ("hotpotqa", "2wikimultihopqa", "musique", "hover", "nq", "popqa")
LOG = logging.getLogger("bindpack.stage_ablations")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def merge_method_parameters(base, overrides):
    """Resolve partial method files before freezing subprocess inputs."""
    if not isinstance(base, dict) or not isinstance(overrides, dict):
        raise ValueError("Method configuration must be an object")
    result = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_method_parameters(result[key], value)
        else:
            result[key] = value
    return result


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for blob in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(blob)
    return value.hexdigest()


def model_dir(runtime):
    return runtime["llm_name"].replace("/", "_") + "_" + runtime["embedding_model_name"].replace("/", "_")


def index_hashes(directory, runtime):
    """Freeze graph/vector/OpenIE inputs; derived caches may change."""
    directory = Path(directory)
    model = directory / model_dir(runtime)
    required = [model / "graph.pickle", model / "chunk_embeddings/vdb_chunk.parquet"]
    if any(not path.is_file() or path.is_symlink() for path in required):
        raise ValueError(f"Missing Qwen3 graph/vector assets: {model}")
    paths = required + list(model.glob("*_embeddings/*.parquet"))
    paths += list(model.glob("index_manifest.json")) + list(model.glob("chunk_metadata.json"))
    paths += list(model.glob("openie_state.json"))
    paths += list(directory.glob("openie_results_ner_*.json"))
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError(f"Index assets must be regular independent files: {directory}")
    return {str(path.relative_to(directory)): sha256(path) for path in sorted(set(paths))}


def safe_output(path):
    path = Path(path).resolve()
    parent = (ROOT / "outputs").resolve()
    if path == parent or parent not in path.parents:
        raise ValueError("Use a dedicated output directory under BindPack/outputs")
    return path


def selected_indices(total, size, seed):
    if size < 0:
        raise ValueError("sample_size must be >= 0")
    return sorted(random.Random(seed).sample(range(total), size)) if 0 < size < total else list(range(total))


def prepare_context(config, dataset, args, output, *, batch=None):
    stem = "nq_rear" if dataset == "nq" else dataset
    data_path = Path(args.data_path or Path(config["datasets_dir"]) / f"{stem}.json").resolve()
    corpus_path = Path(args.corpus_path or Path(config["datasets_dir"]) / f"{stem}_corpus.json").resolve()
    samples = read_json(data_path)
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"Empty/invalid benchmark: {data_path}")
    hops = get_benchmark_hops(samples, dataset)
    for sample in samples:
        get_query_text(sample, dataset)
    if dataset == "hover":
        for sample in samples:
            get_sample_id(sample, dataset)
        get_gold_answers(samples, dataset)  # validate native targets before model calls
    evaluation = config["evaluation"]
    size = args.sample_size if args.sample_size is not None else evaluation[
        "smoke_sample_size" if args.smoke else "sample_size"]
    indices = selected_indices(len(samples), size, evaluation["sample_seed"])
    subset = [samples[index] for index in indices]
    # Explicit corpora are respected in smoke mode, useful for tiny fixtures.
    reuse = args.reuse_shared_index_root is not None
    small_corpus = args.smoke and not args.corpus_path and not reuse
    docs = (build_docs_from_samples(subset, dataset) if small_corpus
            else build_docs_from_full_corpus(read_json(corpus_path)))
    docs = list(dict.fromkeys(docs))
    gold = {doc for group in get_gold_docs(samples if reuse else subset, dataset) for doc in group}
    missing = sorted(gold - set(docs))
    docs.extend(missing)  # eval_dataset's gold-completion policy, once for all cases.
    if not docs:
        raise ValueError(f"Empty corpus: {dataset}")
    rows = [{"title": doc.partition("\n")[0], "text": doc.partition("\n")[2]} for doc in docs]
    metadata = (batch / dataset) if reuse else output / "metadata" / dataset
    identity = {
        "dataset": dataset, "data_path": str(data_path), "data_sha256": sha256(data_path),
        "corpus_source": "selected_sample_contexts" if small_corpus else str(corpus_path),
        "corpus_source_sha256": None if small_corpus else sha256(corpus_path),
        "corpus_docs_sha256": digest(docs), "corpus_docs": len(docs),
        "gold_completion_added": len(missing), "selected_indices": indices,
        "sample_ids": [
            get_sample_id(sample, dataset) if dataset == "hover"
            else sample.get("id", sample.get("_id", index))
            for index, sample in zip(indices, subset)
        ],
        "benchmark_hops": [hops[index] for index in indices],
        "hop_policy": describe_benchmark_hops(samples, dataset),
        "task": get_dataset_task(dataset),
        "corpus_mode": "selected_sample_contexts" if small_corpus else "full",
    }
    context = {"identity": identity, "corpus": rows, "metadata": metadata,
               "index": output / "shared_indexes" / dataset}
    if reuse:
        context["index"] = Path(args.reuse_shared_index_root).expanduser().resolve() / dataset
        context["source_receipt"] = validate_shared_index(context, config)
    return context


def _json_serialization_equivalence(manifest, identity):
    """Allow formatting changes only when both complete JSON values are exact.

    Historical byte fingerprints remain authoritative: the recorded original
    files must still exist and match them.  No strings, passage identities,
    sample fields or array order are normalized for this comparison.
    """
    receipt = {"kind": "exact_json_value_equivalence"}
    for name, old_path_key, old_sha_key, new_path_key, new_sha_key in (
        ("data", "data_path", "data_sha256", "data_path", "data_sha256"),
        ("corpus", "corpus_path", "corpus_sha256", "corpus_source", "corpus_source_sha256"),
    ):
        pairs = {}
        values = {}
        for version, source, path_key, sha_key in (
            ("original", manifest, old_path_key, old_sha_key),
            ("current", identity, new_path_key, new_sha_key),
        ):
            path_text = source.get(path_key)
            if not isinstance(path_text, str) or not path_text or not Path(path_text).is_absolute():
                raise ValueError(f"{name} {version} source path is unavailable")
            raw_path = Path(path_text)
            path = raw_path.resolve()
            if raw_path.is_symlink() or not path.is_file():
                raise ValueError(f"{name} {version} source is not a regular independent file: {raw_path}")
            expected = source.get(sha_key)
            actual = sha256(path)
            if actual != expected:
                raise ValueError(f"{name} {version} source byte fingerprint changed: {path}")
            value = read_json(path)
            semantic = digest(value)
            pairs[version] = {"path": str(path), "sha256": actual, "json_sha256": semantic}
            values[version] = semantic
        if values["original"] != values["current"]:
            raise ValueError(f"{name} complete JSON values differ; rebuild or use the matching index")
        receipt[name] = pairs
    return receipt


def validate_shared_index(context, config):
    """Read historical provenance and assets without changing an old index."""
    index, identity, runtime = context["index"], context["identity"], config["runtime"]
    archive_manifest=index.parent.parent/'manifest.json'
    if archive_manifest.is_file() and isinstance(read_json(archive_manifest).get('files'),list):
        # Merged four-dataset archives carry a full immutable asset inventory,
        # rather than the original runner's metadata/<dataset>/manifest.json.
        from utils.archived_index import validate_archive
        documents=[row['title']+'\n'+row['text'] for row in context['corpus']]
        if digest(documents)!=identity['corpus_docs_sha256']:
            raise ValueError('Canonical archive passage serialization differs from the current corpus')
        proof=validate_archive(index,identity['dataset'],documents,
                               runtime['embedding_model_name'],runtime['llm_name'])
        return {'source_index':str(index),'manifest_sha256':proof['archive_inventory_sha256'],
                'assets':{name:info['sha256'] for name,info in proof['assets'].items()},
                'provenance':'verified_merged_archive','archive_validation':proof}
    source_metadata = index.parent.parent / "metadata" / identity["dataset"]
    manifest_path = source_metadata / "manifest.json"
    manifest = read_json(manifest_path)
    model = index / model_dir(runtime)
    index_manifest_path = model / "index_manifest.json"
    index_manifest = read_json(index_manifest_path)
    expected_models = {key: runtime[key] for key in ("embedding_model_name", "llm_name")}
    if (manifest.get("dataset") != identity["dataset"]
            or manifest.get("index_build_status") != "ready"
            or manifest.get("model_dir") != model.name
            or any(manifest.get("runtime", {}).get(key) != value for key, value in expected_models.items())
            or index_manifest.get("embedding", {}).get("model_name") != runtime["embedding_model_name"]
            or index_manifest.get("openie", {}).get("identity", {}).get("model_name") != runtime["llm_name"]):
        raise ValueError(f"Shared index model/dataset identity mismatch: {index}")
    if manifest.get("indexed_docs") != identity["corpus_docs"]:
        raise ValueError(f"Shared index corpus/benchmark identity mismatch: {index}")
    equivalence = None
    if (manifest.get("data_sha256") != identity["data_sha256"]
            or manifest.get("corpus_sha256") != identity["corpus_source_sha256"]):
        try:
            equivalence = _json_serialization_equivalence(manifest, identity)
        except (OSError, ValueError, TypeError) as error:
            raise ValueError(f"Shared index corpus/benchmark identity mismatch: {index}; {error}") from error
    selected_path = source_metadata / "selected_indices.json"
    old_indices = read_json(selected_path)
    old_ids = manifest.get("selected_sample_ids", [])
    if old_indices != manifest.get("selected_indices") or len(old_indices) != len(old_ids):
        raise ValueError(f"Shared index historical sample identity mismatch: {index}")
    old_samples = dict(zip(old_indices, old_ids))
    if any(old_samples.get(number) != sample_id for number, sample_id in
           zip(identity["selected_indices"], identity["sample_ids"])):
        raise ValueError(f"Selected samples differ from the shared index benchmark: {index}")
    assets = index_hashes(index, runtime)
    required = ("graph.pickle", "chunk_embeddings/vdb_chunk.parquet",
                "entity_embeddings/vdb_entity.parquet", "fact_embeddings/vdb_fact.parquet")
    original = manifest.get("source_asset_sha256", {})
    if any(original.get(relative) != assets.get(str(Path(model.name) / relative)) for relative in required):
        raise ValueError(f"Shared index assets differ from the historical manifest: {index}")
    receipt = {"source_index": str(index), "manifest_sha256": sha256(manifest_path),
               "selected_indices_sha256": sha256(selected_path),
               "index_manifest_sha256": sha256(index_manifest_path), "assets": assets}
    if equivalence is not None:
        receipt["serialization_equivalence"] = equivalence
    return receipt


def command(config, context, directory, stage="full", *, index_only=False):
    runtime, evaluation = config["runtime"], config["evaluation"]
    args = [runtime["python"], "-B", "-u", str(ROOT / "scripts/eval_dataset.py")]
    values = {"dataset": context["identity"]["dataset"],
              "data_path": context["identity"]["data_path"],
              "corpus_path": str(context["metadata"] / "corpus.json"),
              "sample_indices_file": str(context["metadata"] / "selected_indices.json"),
              "sample_seed": evaluation["sample_seed"], "stage_ablation": stage,
              "method_config": config["method_config"],
              "save_dir": str(directory),
              "output": str(directory.parent / "result.json") if not index_only else str(directory / "build_report.json"),
              "llm_name": runtime["llm_name"], "llm_base_url": runtime["llm_base_url"],
              "embedding_model_name": runtime["embedding_model_name"],
              **{key: runtime[key] for key in ("embedding_batch_size", "openie_max_workers", "llm_prefetch_workers",
                  "max_new_tokens", "openie_strict", "openie_prompt_version", "openie_validation_mode")},
              **{key: evaluation[key] for key in ("eval_mode", "retrieval_top_k", "result_top_k",
                  "candidate_output_top_k", "qa_top_k", "max_qa_steps")}}
    for key, value in values.items():
        args.extend(["--" + key, str(value).lower() if isinstance(value, bool) else str(value)])
    args += ["--no_ensure_gold_in_corpus", "--index_only" if index_only else "--reuse_index"]
    return args


def hippo_command(config, context, directory):
    """Original HippoRAG 2 retrieval/QA on a private copy of the shared index."""
    runtime, evaluation = config["runtime"], config["evaluation"]
    if not BASELINE_MAIN.is_file():
        raise ValueError(f"HippoRAG baseline entry is missing: {BASELINE_MAIN}")
    return [runtime["python"], "-B", "-u", str(BASELINE_MAIN),
            "--dataset", context["identity"]["dataset"], "--rag_type", "hipporag",
            "--llm_base_url", runtime["llm_base_url"], "--llm_name", runtime["llm_name"],
            "--embedding_name", runtime["embedding_model_name"], "--embedding_provider", "transformers",
            "--embedding_batch_size", str(runtime["embedding_batch_size"]),
            "--openie_mode", "online", "--save_dir", str(directory), "--save_dir_exact",
            "--datasets_dir", str(config["datasets_dir"]),
            "--sample_size", "0", "--sample_seed", str(evaluation["sample_seed"]),
            "--sample_indices_file", str(context["metadata"] / "selected_indices.json"),
            "--eval_mode", evaluation["eval_mode"],
            "--output", str(directory.parent / "result.json"),
            "--retrieval_top_k", str(evaluation["retrieval_top_k"]),
            "--result_top_k", str(evaluation["result_top_k"]),
            "--candidate_output_top_k", str(evaluation["candidate_output_top_k"]),
            "--llm_prefetch_workers", str(runtime["llm_prefetch_workers"]),
            "--openie_max_workers", str(runtime["openie_max_workers"]),
            "--reuse_index"]


def environment(config):
    env = os.environ.copy()
    runtime = config["runtime"]
    if runtime.get("runtime_deps"):
        deps = Path(runtime["runtime_deps"])
        if not (deps / "transformers").is_dir():
            raise ValueError(f"Qwen3 runtime_deps missing: {deps}; use a compatible environment or empty runtime_deps")
        env["PYTHONPATH"] = str(deps) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", PYTHONHASHSEED="42",
               TOKENIZERS_PARALLELISM="false", OPENAI_API_KEY=env.get("OPENAI_API_KEY", "EMPTY"),
               BINDPACK_LLM_MAX_IN_FLIGHT="8",
               BINDPACK_KNN_DEVICE=env.get("BINDPACK_KNN_DEVICE", "cpu"))
    return env


def check_service(runtime):
    request = urllib.request.Request(runtime["llm_base_url"].rstrip("/") + "/models",
                                     headers={"Authorization": "Bearer " + os.environ.get("OPENAI_API_KEY", "EMPTY")})
    with urllib.request.urlopen(request, timeout=10) as response:
        models = {row["id"] for row in json.load(response).get("data", [])}
    if runtime["llm_name"] not in models:
        raise ValueError(f"LLM service does not expose {runtime['llm_name']!r}; available: {sorted(models)}")


def execute(argv, log, env):
    LOG.info("%s", shlex.join(argv))
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log.open("a", encoding="utf-8") as stream:
        stream.write("\nCOMMAND " + shlex.join(argv) + "\n")
        stream.flush()
        # Mirror unbuffered child output into both the case log and the shell's
        # main log, so tail -f also shows long-running retrieval progress.
        with subprocess.Popen(argv, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
    if process.returncode:
        raise RuntimeError(f"Evaluation failed ({process.returncode}); inspect {log}")
    return round(time.monotonic() - started, 3)


def clone_index(source, destination):
    # Independent copies keep accidental writes from altering another case.
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(
        "llm_cache", "*.lock", "*.sqlite*", "build_report.json", "frozen_index.json", "build.log"))


def copy_asset(source, destination):
    """Use independent copy-on-write extents when supported; never hard-link."""
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name("." + destination.name + ".copy.tmp")
    try:
        try:
            with source.open("rb") as source_stream, temporary.open("wb") as destination_stream:
                fcntl.ioctl(destination_stream.fileno(), 0x40049409, source_stream.fileno())  # FICLONE
            shutil.copystat(source, temporary)
        except OSError:
            shutil.copy2(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def clone_shared_assets(source, destination, assets):
    """Copy only verified inputs, leaving historical and cross-case caches out."""
    for relative, expected in assets.items():
        target = destination / relative
        if target.exists():
            if target.is_symlink() or sha256(target) != expected:
                raise ValueError(f"Private index changed: {target}")
        else:
            copy_asset(source / relative, target)


def release_shared_assets(directory, assets):
    """Release this case's temporary copies after successful validation."""
    for relative in assets:
        path = directory / relative
        path.unlink()
        while path.parent != directory:
            path = path.parent
            try:
                path.rmdir()
            except OSError:
                break


def validate_result(path, config, identity, stage):
    result = read_json(path)
    runtime = result.get("runtime_config", {})
    if (result.get("stage_ablation") != stage or runtime.get("stage_ablation") != stage
            or result.get("selected_indices") != identity["selected_indices"]
            or result.get("sample_size_effective") != len(identity["selected_indices"])
            or result.get("dataset") != identity["dataset"]
            or result.get("eval_mode") != config["evaluation"]["eval_mode"]
            or runtime.get("embedding_model_name") != config["runtime"]["embedding_model_name"]
            or runtime.get("llm_name") != config["runtime"]["llm_name"]
            or [row.get("query_index") for row in result.get("results", [])] != identity["selected_indices"]
            or [row.get("sample_id") for row in result.get("results", [])] != identity["sample_ids"]):
        raise ValueError(f"Result stage/model/sample identity mismatch: {path}")
    if not result.get("retrieval_metrics"):
        raise ValueError(f"Missing retrieval metrics: {path}")
    if config["evaluation"]["eval_mode"] == "rag_qa" and (
            result.get("qa_metrics", {}).get("ExactMatch") is None
            or result.get("qa_metrics", {}).get("F1") is None):
        raise ValueError(f"Missing QA metrics: {path}")
    method = result.get("method_config", {})
    if (method.get("sha256") != config["method_source"]["sha256"]
            or method.get("parameters") != config["method_source"]["parameters"]):
        raise ValueError(f"Result method configuration mismatch: {path}")
    return result


def validate_hippo_result(path, config, identity):
    result = read_json(path)
    if (result.get("method") != "hipporag2" or result.get("dataset") != identity["dataset"]
            or result.get("selected_indices") != identity["selected_indices"]
            or result.get("sample_size_effective") != len(identity["selected_indices"])
            or result.get("eval_mode") != config["evaluation"]["eval_mode"]
            or result.get("reuse_index") is not True
            or result.get("llm_name") != config["runtime"]["llm_name"]
            or result.get("embedding_name") != config["runtime"]["embedding_model_name"]
            or [row.get("query_index") for row in result.get("results", [])] != identity["selected_indices"]
            or [row.get("sample_id") for row in result.get("results", [])] != identity["sample_ids"]):
        raise ValueError(f"HippoRAG result identity mismatch: {path}")
    if not result.get("retrieval_metrics"):
        raise ValueError(f"Missing retrieval metrics: {path}")
    if config["evaluation"]["eval_mode"] == "rag_qa" and (
            result.get("qa_metrics", {}).get("ExactMatch") is None
            or result.get("qa_metrics", {}).get("F1") is None):
        raise ValueError(f"Missing QA metrics: {path}")
    return result


def selected_cases(config):
    return list(CASES) + ([HIPPO_CASE] if config.get("include_hipporag") else [])


def save_metrics_report(case, identity, metadata):
    """Postprocess validated outputs without rerunning retrieval on failure."""
    failure = case / "report_error.json"
    try:
        report = write_metric_report(case / "result.json", data_path=identity["data_path"],
                                     manifest_path=metadata / "sample_manifest.json")
        failure.unlink(missing_ok=True)
        return report
    except Exception as exc:
        LOG.exception("Validated result retained; metric reporting failed for %s", case)
        try:
            write_json(failure, {"status": "error", "postprocessing_only": True,
                                "result": str(case / "result.json"), "error": str(exc)})
        except OSError:
            LOG.exception("Could not persist the metric-report error receipt for %s", case)
        return None


def run_dataset(config, context, output, env):
    identity, index, runtime = context["identity"], context["index"], config["runtime"]
    metadata = context["metadata"]
    for name, value in (("corpus.json", context["corpus"]), ("selected_indices.json", identity["selected_indices"]),
                        ("sample_manifest.json", identity)):
        path = metadata / name
        if path.exists() and read_json(path) != value:
            raise ValueError(f"Frozen dataset metadata changed: {path}")
        write_json(path, value)
    receipt_path = metadata / "frozen_index.json"
    reused = "source_receipt" in context
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt["identity"] != identity or receipt["assets"] != index_hashes(index, runtime):
            raise ValueError(f"Frozen index or corpus changed: {index}")
    elif reused:
        receipt = {"identity": identity, "embedding_model_name": runtime["embedding_model_name"],
                   "llm_name": runtime["llm_name"], **context["source_receipt"]}
        write_json(receipt_path, receipt)
        LOG.info("Reusing verified shared index %s (%s documents)", index, identity["corpus_docs"])
    else:
        index.mkdir(parents=True, exist_ok=True)
        execute(command(config, context, index, index_only=True), metadata / "index_build.log", env)
        report = read_json(index / "build_report.json")
        if (report.get("index_build_complete") is not True
                or report.get("indexed_docs") != identity["corpus_docs"]
                or report["runtime_config"]["embedding_model_name"] != runtime["embedding_model_name"]):
            raise ValueError(f"Index build identity mismatch: {index}")
        receipt = {"identity": identity, "embedding_model_name": runtime["embedding_model_name"],
                   "llm_name": runtime["llm_name"], "assets": index_hashes(index, runtime)}
        write_json(receipt_path, receipt)
    reports = []
    for stage in selected_cases(config):
        case = output / "cases" / identity["dataset"] / stage
        case.mkdir(parents=True, exist_ok=True)
        private = case / "index"
        hippo = stage == HIPPO_CASE
        protocol = {"stage_ablation": stage, "method": "hipporag2" if hippo else "bindpack",
                    "input_sha256": digest(identity),
                    "frozen_index_sha256": digest(receipt), "config_sha256": digest(config),
                    "cache_policy": "empty independent LLM cache for every case; resume keeps that case cache"}
        if reused:
            protocol["index_policy"] = "private asset copy; release graph/vector/OpenIE copies after validation"
        protocol_path, marker = case / "protocol.json", case / "validated.ok"
        if protocol_path.exists() and read_json(protocol_path) != protocol:
            raise ValueError(f"Case protocol changed: {case}")
        if not protocol_path.exists() and any(case.iterdir()):
            raise ValueError(f"Refusing existing untracked case: {case}")
        write_json(protocol_path, protocol)
        if marker.exists():
            completed = read_json(marker)
            if completed.get("protocol_sha256") != digest(protocol):
                raise ValueError(f"Validated case protocol changed: {case}")
            if completed["result_sha256"] != sha256(case / "result.json"):
                raise ValueError(f"Validated result changed: {case}")
            result = (validate_hippo_result(case / "result.json", config, identity) if hippo
                      else validate_result(case / "result.json", config, identity, stage))
            if not completed.get("private_assets_released") and index_hashes(private, runtime) != receipt["assets"]:
                raise ValueError(f"Private index changed: {private}")
            LOG.info("Reusing validated %s/%s", identity["dataset"], stage)
        else:
            LOG.info("Starting %s/%s (%s samples)", identity["dataset"], stage, len(identity["selected_indices"]))
            if reused:
                clone_shared_assets(index, private, receipt["assets"])
            elif not private.exists():
                clone_index(index, private)
            if index_hashes(private, runtime) != receipt["assets"]:
                raise ValueError(f"Private index changed: {private}")
            argv = hippo_command(config, context, private) if hippo else command(config, context, private, stage)
            seconds = execute(argv, case / "run.log", env)
            result = (validate_hippo_result(case / "result.json", config, identity) if hippo
                      else validate_result(case / "result.json", config, identity, stage))
            if index_hashes(private, runtime) != receipt["assets"]:
                raise ValueError(f"Evaluation altered graph/vector inputs: {case}")
            if reused:
                release_shared_assets(private, receipt["assets"])
            write_json(marker, {"result_sha256": sha256(case / "result.json"), "seconds": seconds,
                                "protocol_sha256": digest(protocol), "private_assets_released": reused})
            LOG.info("Validated %s/%s in %.3fs; temporary inputs released=%s",
                     identity["dataset"], stage, seconds, reused)
        if index_hashes(index, runtime) != receipt["assets"]:
            raise ValueError(f"Evaluation altered frozen shared index: {index}")
        metric_report = save_metrics_report(case, identity, metadata)
        reports.append({"dataset": identity["dataset"], "stage_ablation": stage,
                        "samples": len(identity["selected_indices"]), "result": str(case / "result.json"),
                        "retrieval_metrics": result["retrieval_metrics"], "qa_metrics": result["qa_metrics"],
                        "metrics_report": str(case / "report.json") if metric_report else None,
                        "evidence_metrics": metric_report["metrics"] if metric_report else None,
                        "llm_request_stats": result.get("llm_request_stats", {}),
                        "seconds": read_json(marker)["seconds"]})
    return reports


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/ablations_qwen3.json"))
    parser.add_argument("--method-config", help="Override fixed method parameters; paths are relative to the repository")
    parser.add_argument("--out-root")
    parser.add_argument("--reuse-shared-index-root", help="Read verified historical shared indexes; never rebuild or write them")
    parser.add_argument("--batch-name", default="stage_ablations_qwen3_reuse",
                        help="Metadata namespace under out-root/metadata in shared-index reuse mode")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS)
    parser.add_argument("--datasets-dir")
    parser.add_argument("--data-path", help="Override benchmark path when selecting one dataset")
    parser.add_argument("--corpus-path", help="Override corpus path when selecting one dataset, including smoke")
    parser.add_argument("--python")
    parser.add_argument("--runtime-deps")
    parser.add_argument("--llm-name")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--embedding-model", dest="embedding_model_name")
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--eval-mode", choices=("retrieve", "rag_qa"))
    parser.add_argument("--include-hipporag", action="store_true",
                        help="Also run original HippoRAG 2 on the same frozen index as hippporag2_reproduce")
    parser.add_argument("--smoke", action="store_true", help="Two questions per dataset; reuse mode keeps the frozen full corpus")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print commands without writes or model calls")
    parser.add_argument("--resume", action="store_true", help="Resume only when inputs, configuration and code match")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = read_json(args.config)
    if tuple(config.get("cases", [])) != CASES:
        raise ValueError(f"Exactly four ordered cases are required: {CASES}")
    method_path = Path(args.method_config or config["method_config"]).expanduser()
    if not method_path.is_absolute():
        method_path = ROOT / method_path
    method_path = method_path.resolve()
    defaults_path = ROOT / "src/bindpack/configuration/defaults.json"
    parameters = merge_method_parameters(read_json(defaults_path), read_json(method_path))
    method_bytes = (json.dumps(parameters, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    config["method_source"] = {
        "path": str(method_path), "source_sha256": sha256(method_path),
        "defaults_sha256": sha256(defaults_path),
        "sha256": hashlib.sha256(method_bytes).hexdigest(), "parameters": parameters,
    }
    for name in ("python", "runtime_deps", "llm_name", "llm_base_url", "embedding_model_name"):
        if getattr(args, name) is not None:
            config["runtime"][name] = getattr(args, name)
    if args.datasets_dir:
        config["datasets_dir"] = args.datasets_dir
    if args.eval_mode:
        config["evaluation"]["eval_mode"] = args.eval_mode
    config["include_hipporag"] = bool(args.include_hipporag)
    if config["include_hipporag"] and not BASELINE_MAIN.is_file():
        raise ValueError(f"HippoRAG baseline entry is missing: {BASELINE_MAIN}")
    datasets = args.datasets or config["datasets"]
    if len(set(datasets)) != len(datasets) or not datasets or any(d not in DATASETS for d in datasets):
        raise ValueError("Select unique supported datasets")
    if (args.data_path or args.corpus_path) and len(datasets) != 1:
        raise ValueError("--data-path/--corpus-path require one dataset")
    if config["include_hipporag"] and "hover" in datasets:
        raise ValueError("The external HippoRAG reproduction does not support HoVer fact verification; "
                         "run HoVer with run_bindpack_ablations_qwen3.sh without --include-hipporag")
    runtime = config["runtime"]
    if "Qwen3-Embedding-8B" not in runtime["embedding_model_name"]:
        raise ValueError("This experiment uses Qwen3-Embedding-8B; another embedding requires a separate workflow")
    model_path = Path(runtime["embedding_model_name"].removeprefix("Transformers/"))
    if not model_path.is_dir() or not Path(runtime["llm_model_path"]).is_dir():
        raise ValueError("Local Qwen3-8B and Qwen3-Embedding-8B model directories must exist")
    reuse = args.reuse_shared_index_root is not None
    if reuse:
        shared_root = Path(args.reuse_shared_index_root).expanduser().resolve()
        if not shared_root.is_dir():
            raise ValueError(f"Shared index directory does not exist: {shared_root}")
        args.reuse_shared_index_root = str(shared_root)
        if not args.batch_name or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in args.batch_name):
            raise ValueError("--batch-name must contain only letters, digits, underscores or hyphens")
        default_output = (ROOT / "outputs/stage_ablations_qwen3_reuse_smoke") if args.smoke else shared_root.parent
    else:
        default_output = ROOT / "outputs" / ("stage_ablations_qwen3_smoke" if args.smoke else "stage_ablations_qwen3")
    output = safe_output(args.out_root or default_output)
    if reuse and (output == shared_root or shared_root in output.parents):
        raise ValueError("Output must be outside the read-only shared index root")
    batch = output / "metadata" / args.batch_name if reuse else output
    # Every subprocess reads this byte-for-byte snapshot, even if the source
    # configuration is edited after the experiment starts.
    config["method_config"] = str(batch / "method_config.json")
    if args.smoke and "smoke" not in output.name.lower():
        raise ValueError("Smoke output directory must include 'smoke' in its name")
    contexts = [prepare_context(config, dataset, args, output, batch=batch) for dataset in datasets]
    sources = [*sorted((ROOT / "src").rglob("*.py")), *sorted((ROOT / "scripts").rglob("*.py")),
               *sorted((ROOT / "scripts").rglob("*.sh"))]
    protocol = {"schema_version": 2, "config": config, "datasets": datasets, "smoke": args.smoke,
                "identities": [context["identity"] for context in contexts],
                "model_config_sha256": {str(path): sha256(path) for path in
                    (model_path / "config.json", Path(runtime["llm_model_path"]) / "config.json")},
                "code_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
                "safeguard_policy": "veto enabled in full/wo_route/wo_plan; disabled with all Pack in wo_pack",
                "include_hipporag": bool(config.get("include_hipporag")),
                "cases": selected_cases(config)}
    if reuse:
        protocol["shared_indexes"] = [context["source_receipt"] for context in contexts]
        protocol["index_policy"] = "source read only; private copy per case, released after validation; independent LLM caches"
    protocol_path = batch / "protocol.json"
    reporting_receipt = None
    if protocol_path.exists():
        try:
            protocol, reporting_receipt = resolve_existing_protocol(protocol_path, protocol, root=ROOT)
        except ValueError as exc:
            raise ValueError(f"Existing inputs/configuration/code differ; choose a new --out-root: {exc}") from exc
        if reporting_receipt is not None:
            LOG.info("Verified reporting-only update; original retrieval protocol retained")
        if not args.resume and not args.dry_run:
            raise ValueError("Output already exists; use --resume to validate and continue")
    elif batch.exists() and any(batch.iterdir()):
        raise ValueError("Output contains untracked files; choose a new --out-root or --batch-name")
    for context in contexts:
        for stage in selected_cases(config):
            case = output / "cases" / context["identity"]["dataset"] / stage
            if case.exists() and any(case.iterdir()) and not protocol_path.exists():
                raise ValueError(f"Refusing existing untracked case: {case}; choose a new --out-root")
    if args.dry_run:
        for context in contexts:
            print(json.dumps(context["identity"], ensure_ascii=False))
            if reuse:
                print("REUSE VERIFIED SHARED INDEX " + str(context["index"]))
            else:
                print(shlex.join(command(config, context, context["index"], index_only=True)))
            for stage in selected_cases(config):
                private = output / "cases" / context["identity"]["dataset"] / stage / "index"
                print(shlex.join(hippo_command(config, context, private) if stage == HIPPO_CASE
                                 else command(config, context, private, stage)))
        return 0
    env = environment(config)
    check_service(runtime)
    batch.mkdir(parents=True, exist_ok=True)
    lock_path = ROOT / "outputs/.recall_top5_improvements.lock"
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another retrieval experiment holds the shared experiment lock") from error
        snapshot = Path(config["method_config"])
        if snapshot.exists() and snapshot.read_bytes() != method_bytes:
            raise ValueError(f"Frozen method configuration changed: {snapshot}")
        snapshot.write_bytes(method_bytes)
        if not protocol_path.exists():
            write_json(protocol_path, protocol)
        if reporting_receipt is not None:
            persist_reporting_receipt(protocol_path, reporting_receipt)
        reports = []
        for context in contexts:
            reports.extend(run_dataset(config, context, output, env))
        write_json(batch / "comparison.json", {"smoke": args.smoke, "cases": selected_cases(config),
                                               "eval_mode": config["evaluation"]["eval_mode"], "results": reports})
        lines = ["| Dataset | Case | Samples | R@1 | R@5 | R@10 | QA EM | QA F1 | Seconds |",
                 "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for report in reports:
            metrics, qa = report["retrieval_metrics"], report["qa_metrics"]
            numbers = [metrics.get(f"Recall@{k}") for k in (1, 5, 10)] + [qa.get("ExactMatch"), qa.get("F1")]
            lines.append(f"| {report['dataset']} | {report['stage_ablation']} | {report['samples']} | "
                         + " | ".join("—" if number is None else f"{number:.4f}" for number in numbers)
                         + f" | {report['seconds']} |")
        (batch / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        write_json(batch / "completed.ok", {"protocol_sha256": digest(protocol), "validated_case_count": len(reports),
                                              "smoke": args.smoke, "comparison_sha256": sha256(batch / "comparison.json")})
        print("\n".join(lines))
    return 0

#!/usr/bin/env python3
"""BindPack dataset evaluation entry."""

import argparse
import hashlib
import json
import logging
import os
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BindPack_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(BindPack_ROOT)

# bindpack package
sys.path.insert(0, os.path.join(BindPack_ROOT, "src"))

from bindpack import BindPack, BindPackConfig
from bindpack.evaluation.result_export import detailed_result
from bindpack.index.publication_policy import parse_bool
from eval_utils import (
    build_docs_from_full_corpus,
    build_docs_from_samples,
    ensure_gold_docs_in_corpus,
    get_gold_answers,
    get_gold_docs,
    get_benchmark_hops,
    load_json,
    sample_indices,
)
from stratified_eval import evaluate_stratified, print_stratified_results, save_stratified_results

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", "sk-local-dummy"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("bindpack.eval")


def _select_samples(samples: List[Dict[str, Any]], sample_size: int, sample_seed: int, sample_index: int,
                    sample_indices_file: str = ""):
    total = len(samples)
    if sample_indices_file:
        selected_indices = load_json(sample_indices_file)
        if isinstance(selected_indices, dict):
            selected_indices = selected_indices.get("selected_indices")
        if (not isinstance(selected_indices, list) or not selected_indices
                or any(type(i) is not int or not 0 <= i < total for i in selected_indices)
                or len(set(selected_indices)) != len(selected_indices)):
            raise ValueError("sample_indices_file must contain unique valid integer indices")
        if sample_size > 0 and len(selected_indices) != sample_size:
            raise ValueError("sample_size and sample_indices_file count differ")
    elif sample_index >= 0:
        if sample_index >= total:
            raise ValueError(f"sample_index={sample_index} out of range (total={total})")
        selected_indices = [sample_index]
    else:
        idxs = sample_indices(total=total, sample_size=sample_size, seed=sample_seed)
        selected_indices = idxs if idxs is not None else list(range(total))
    selected_samples = [samples[i] for i in selected_indices]
    return selected_samples, selected_indices


def run_eval(dataset: str, args: argparse.Namespace) -> str:
    dataset_file_stem = "nq_rear" if dataset == "nq" else dataset
    data_path = args.data_path or os.path.join(PROJECT_ROOT, "datasets", f"{dataset_file_stem}.json")
    corpus_path = args.corpus_path or os.path.join(PROJECT_ROOT, "datasets", f"{dataset_file_stem}_corpus.json")
    hop_source = getattr(args, "hop_source", "benchmark")
    if hop_source != "benchmark":
        raise ValueError(f"Unknown hop_source={hop_source!r}")

    method_path = Path(args.method_config).expanduser().resolve()
    method_bytes = method_path.read_bytes()
    method_metadata = {
        "path": str(method_path),
        "sha256": hashlib.sha256(method_bytes).hexdigest(),
        "parameters": json.loads(method_bytes),
    }
    save_dir = args.save_dir or os.path.join(BindPack_ROOT, "outputs", "evaluation", dataset)

    logger.info("[1/4] Loading dataset: %s", dataset)
    samples_all = load_json(data_path)
    corpus = load_json(corpus_path)

    # Validate the entire benchmark split before sampling. A subset must not
    # conceal missing MuSiQue decomposition labels elsewhere in the file.
    benchmark_hops_all = get_benchmark_hops(samples_all, dataset)
    if (not isinstance(benchmark_hops_all, list) or len(benchmark_hops_all) != len(samples_all)
            or any(type(hop) is not int or not 1 <= hop <= 4 for hop in benchmark_hops_all)):
        raise ValueError(f"{dataset}: benchmark hop labels/priors are missing or invalid")

    samples, selected_indices = _select_samples(
        samples=samples_all,
        sample_size=max(0, int(args.sample_size)),
        sample_seed=int(args.sample_seed),
        sample_index=int(args.sample_index),
        sample_indices_file=getattr(args, "sample_indices_file", ""),
    )

    queries = [s["question"] for s in samples]
    benchmark_hops = [benchmark_hops_all[i] for i in selected_indices]
    if dataset == "musique":
        hop_provenance = {
            "kind": "per_question_oracle",
            "field": "question_decomposition length",
            "oracle": True,
            "validated_samples": len(samples_all),
        }
    elif dataset in ("hotpotqa", "2wikimultihopqa"):
        hop_provenance = {
            "kind": "dataset_level_prior",
            "value": 2,
            "oracle": False,
            "note": "Question type and support count are not serial hop labels.",
        }
    else:
        hop_provenance = {
            "kind": "dataset_level_prior",
            "value": 1,
            "oracle": False,
            "note": "Single-hop benchmark prior; support count is not a hop label.",
        }
    hop_type_distribution = (
        dict(Counter(str(s["type"]) for s in samples))
        if dataset in ("hotpotqa", "2wikimultihopqa")
        else {}
    )
    gold_answers = get_gold_answers(samples)
    gold_docs = get_gold_docs(samples, dataset)

    if args.corpus_mode == "sample_only":
        docs = build_docs_from_samples(samples=samples, dataset_name=dataset)
    else:
        docs = build_docs_from_full_corpus(corpus)
        if args.max_corpus_docs > 0:
            docs = docs[: args.max_corpus_docs]

    if args.ensure_gold_in_corpus:
        docs = ensure_gold_docs_in_corpus(docs=docs, gold_docs=gold_docs)

    logger.info(
        "Samples=%d (selected=%s), Corpus docs for indexing=%d (mode=%s)",
        len(samples),
        selected_indices,
        len(docs),
        args.corpus_mode,
    )

    runtime_overrides = {
        "save_dir": save_dir,
        "dataset": dataset,
        "corpus_len": len(docs),
        "force_index_from_scratch": args.force_index_from_scratch,
        "force_openie_from_scratch": args.force_openie_from_scratch,
        "stage_ablation": args.stage_ablation,
    }
    # Fixed algorithm parameters come from one method file. Only resource and
    # evaluation controls may be overridden at this command-line boundary.
    for name in (
        "llm_name", "llm_base_url", "embedding_model_name", "embedding_base_url",
        "retrieval_top_k", "qa_top_k", "max_new_tokens", "max_qa_steps",
        "embedding_batch_size", "openie_max_workers", "openie_strict",
        "openie_prompt_version", "openie_validation_mode", "llm_prefetch_workers",
    ):
        value = getattr(args, name)
        if value is not None:
            runtime_overrides[name] = value
    config = BindPackConfig.from_file(method_path, **runtime_overrides)

    logger.info("[2/4] %s", "Loading shared index" if args.reuse_index else "Building index")
    rag = BindPack(config=config)
    if getattr(args, "reuse_index", False):
        if set(rag.chunk_embedding_store.get_all_texts()) != set(docs):
            raise ValueError("Shared index passages differ from the evaluation corpus")
        if not os.path.isfile(os.path.join(rag.working_dir, "graph.pickle")):
            raise ValueError("reuse_index requires an existing graph")
        rag.prepare_retrieval_objects()
        rag._build_index_side_artifacts()
        logger.info("Reused existing graph/vectors; built only BindPack side artifacts")
    else:
        rag.index(docs=docs)
    if getattr(args, "index_only", False):
        if set(rag.chunk_embedding_store.get_all_texts()) != set(docs):
            raise ValueError("Constructed passage index does not cover the evaluation corpus")
        if not os.path.isfile(os.path.join(rag.working_dir, "graph.pickle")):
            raise ValueError("Index construction did not produce a graph")
        out_path = args.output or os.path.join(save_dir, "index_build_report.json")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as stream:
            json.dump({"dataset": dataset, "eval_mode": "index_only", "index_build_complete": True,
                       "indexed_docs": len(set(docs)), "selected_indices": selected_indices,
                       "runtime_config": asdict(config), "method_config": method_metadata},
                      stream, ensure_ascii=False, indent=2)
        logger.info("Index ready: %s", out_path)
        return out_path
    rag.set_query_hop_overrides(hops=benchmark_hops, source="benchmark")
    logger.info("Benchmark hop policy: %s; selected distribution=%s", hop_provenance,
                dict(Counter(str(h) for h in benchmark_hops)))

    logger.info("[3/4] Running evaluation mode=%s", args.eval_mode)
    retrieval_metrics: Dict[str, Any] = {}
    qa_metrics: Dict[str, Any] = {}
    retrieval_diagnostics: Dict[str, Any] = {}
    output_rows: List[Dict[str, Any]] = []

    has_answers = any(len(a) > 0 for a in gold_answers)
    predicted_answers_for_stratified = None

    if args.eval_mode == "rag_qa" and has_answers:
        query_solutions, _, _, retrieval_metrics, qa_metrics = rag.rag_qa(
            queries=queries,
            gold_docs=gold_docs,
            gold_answers=gold_answers,
        )
        solutions = query_solutions
        predicted_answers_for_stratified = [q.answer if q.answer is not None else "" for q in query_solutions]
    else:
        retrieval_results, retrieval_metrics = rag.retrieve(
            queries=queries,
            num_to_retrieve=config.retrieval_top_k,
            gold_docs=gold_docs,
        )
        solutions = retrieval_results

    output_rows = [
        detailed_result(solution, samples[i], selected_indices[i], gold_docs[i],
                        benchmark_hops[i] if benchmark_hops is not None else None,
                        getattr(args, "result_top_k", 10), getattr(args, "candidate_output_top_k", 200))
        for i, solution in enumerate(solutions)
    ]

    out_path = args.output or os.path.join(save_dir, "evaluation_results.json")

    # Stratified evaluation if enabled
    if args.stratified_eval:
        logger.info("Running stratified evaluation by query hop complexity...")
        if args.eval_mode == "rag_qa" and has_answers:
            retrieval_results_for_stratified = query_solutions
        else:
            retrieval_results_for_stratified = retrieval_results

        stratified_results = evaluate_stratified(
            queries=queries,
            retrieval_results=retrieval_results_for_stratified,
            gold_docs=gold_docs,
            gold_answers=gold_answers,
            global_config=config,
            predicted_answers=predicted_answers_for_stratified,
            query_hops=benchmark_hops,
            hop_provenance=hop_provenance,
        )
        print_stratified_results(stratified_results)

        if args.stratified_output:
            stratified_path = args.stratified_output
        else:
            stratified_path = os.path.splitext(out_path)[0] + "_stratified.json"
        save_stratified_results(stratified_results, stratified_path)

    if hasattr(rag, "get_retrieval_diagnostics"):
        try:
            retrieval_diagnostics = rag.get_retrieval_diagnostics()
        except Exception as exc:
            logger.exception("Failed to collect retrieval diagnostics: %s", exc)
            retrieval_diagnostics = {}

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": dataset,
                "stage_ablation": config.stage_ablation,
                "hop_source": hop_source,
                "hop_provenance": hop_provenance,
                "hop_distribution": (
                    dict(Counter(str(h) for h in benchmark_hops))
                    if benchmark_hops is not None
                    else retrieval_diagnostics.get("hop_counter", {})
                ),
                "hop_type_distribution": hop_type_distribution,
                "data_path": data_path,
                "corpus_path": corpus_path,
                "sample_size_requested": int(args.sample_size),
                "sample_seed": int(args.sample_seed),
                "sample_index": int(args.sample_index),
                "sample_size_effective": len(samples),
                "selected_indices": selected_indices,
                "result_top_k": getattr(args, "result_top_k", 10),
                "candidate_output_top_k": getattr(args, "candidate_output_top_k", 200),
                "corpus_mode": args.corpus_mode,
                "indexed_docs": len(docs),
                "eval_mode": args.eval_mode,
                "retrieval_metrics": retrieval_metrics,
                "qa_metrics": qa_metrics,
                "retrieval_diagnostics": retrieval_diagnostics,
                "retrieval_seconds": float(getattr(rag, "all_retrieval_time", 0.0)),
                "llm_request_stats": (
                    rag.llm_model.get_request_stats()
                    if hasattr(rag.llm_model, "get_request_stats") else {}
                ),
                "runtime_config": asdict(config),
                "method_config": method_metadata,
                "results": output_rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Measurement runs only after the model output has been persisted. A report
    # failure can be retried offline and must not discard an expensive run.
    report_error = Path(out_path).with_name("report_error.json")
    try:
        from bindpack.evaluation.metric_report import write_metric_report
        write_metric_report(out_path, data_path=data_path, require_validated=False)
        report_error.unlink(missing_ok=True)
    except Exception as exc:
        logger.exception("Results saved; metric reporting failed. Retry with scripts/report_metrics.py")
        try:
            report_error.write_text(json.dumps({"status": "error", "postprocessing_only": True,
                "result": str(out_path), "error": str(exc)}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
        except OSError:
            logger.exception("Could not persist the metric-report error receipt")

    logger.info("[4/4] Done. Results saved to: %s", out_path)
    return out_path


def _openie_worker_count(value: str) -> int:
    try:
        workers = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("OpenIE workers must be an integer between 1 and 8.") from exc
    if not 1 <= workers <= 8:
        raise argparse.ArgumentTypeError("OpenIE workers must be between 1 and 8.")
    return workers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate BindPack on supported datasets")
    parser.add_argument("--dataset", required=True, choices=["musique", "hotpotqa", "2wikimultihopqa", "nq", "popqa"])
    parser.add_argument("--data_path", default="")
    parser.add_argument("--corpus_path", default="")
    parser.add_argument("--hop_source", choices=["benchmark"], default="benchmark",
                        help="Use documented dataset labels/priors; MuSiQue per-question hops are oracle labels.")

    parser.add_argument("--sample_size", type=int, default=0, help="0 means all")
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--sample_index", type=int, default=-1)
    parser.add_argument("--sample_indices_file", default="")
    parser.add_argument("--reuse_index", action="store_true")
    parser.add_argument("--index_only", action="store_true", help="Build/check the shared index and exit before retrieval.")
    parser.add_argument("--stage_ablation", choices=["full", "wo_route", "wo_plan", "wo_pack"],
                        default="full", help="Whole-stage controls: complete method, fixed graph/DPR Route baseline, no dependency Plan, or no Pack.")
    parser.add_argument("--method_config", default=str(Path(BindPack_ROOT) / "src/bindpack/configuration/defaults.json"),
                        help="Fixed Route, Plan, Pack and shared-candidate parameters as JSON.")

    parser.add_argument("--corpus_mode", choices=["full", "sample_only"], default="full")
    parser.add_argument("--max_corpus_docs", type=int, default=0)
    parser.add_argument("--ensure_gold_in_corpus", action="store_true", default=True)
    parser.add_argument("--no_ensure_gold_in_corpus", dest="ensure_gold_in_corpus", action="store_false")

    parser.add_argument("--eval_mode", choices=["rag_qa", "retrieve"], default="retrieve")
    parser.add_argument("--retrieval_top_k", type=int)
    parser.add_argument("--result_top_k", type=int, default=10)
    parser.add_argument("--candidate_output_top_k", type=int, default=200)
    parser.add_argument("--qa_top_k", type=int, default=None)
    parser.add_argument("--max_qa_steps", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--embedding_batch_size", type=int, default=None)
    parser.add_argument("--openie_strict", type=parse_bool, default=None,
                        help="true aborts on final extraction failure; false excludes failed relations and continues.")
    parser.add_argument("--openie_prompt_version", choices=["origin", "optimized"], default=None)
    parser.add_argument("--openie_validation_mode", choices=["structural", "source_verified"], default=None,
                        help="Index extraction: local structure checks (default) or extra LLM source evidence audits.")
    parser.add_argument("--openie_max_workers", type=_openie_worker_count, default=None,
                        help="Online indexing NER/triple concurrency (1..8); omitted uses method configuration.")
    parser.add_argument("--llm_prefetch_workers", type=int, default=None,
                        help="Bounded QD/PCQD LLM prefetch across queries; omitted uses method configuration.")

    parser.add_argument("--save_dir", default="")
    parser.add_argument("--output", default="")

    parser.add_argument("--llm_name", default=None)
    parser.add_argument("--llm_base_url", default=None)
    parser.add_argument("--embedding_model_name", default=None)
    parser.add_argument("--embedding_base_url", default=None)

    parser.add_argument("--force_index_from_scratch", action="store_true", default=False)
    parser.add_argument("--force_openie_from_scratch", action="store_true", default=False)

    # Evaluation options
    parser.add_argument("--stratified_eval", action="store_true", default=False,
                        help="Enable stratified evaluation by query hop complexity")
    parser.add_argument("--stratified_output", default="",
                        help="Output path for stratified metrics JSON (defaults to <output>_stratified.json)")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_eval(dataset=args.dataset, args=args)


if __name__ == "__main__":
    main()

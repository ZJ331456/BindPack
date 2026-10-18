from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from numbers import Integral
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm

from ..core.base import BaseRAG
from ..index.artifacts import IndexArtifactsMixin
from ..evaluation.retrieval_eval import RetrievalRecall
from ..utils.misc_utils import QuerySolution, compute_mdhash_id, min_max_normalize

from ..config import BindPackConfig
from ..stages.route import RouteStage

logger = logging.getLogger(__name__)


class BindPack(IndexArtifactsMixin, RouteStage, BaseRAG):
    """Orchestrate Route discovery, Plan execution and Pack selection."""

    def __init__(self, config: Optional[BindPackConfig] = None):
        self.config = config if config is not None else BindPackConfig.from_file()
        super().__init__(global_config=self.config)
        from ..pipeline import EvidencePipeline

        self.evidence_runtime = EvidencePipeline(self)
        self._query_hop_overrides: Optional[List[int]] = None
        self._query_hop_override_source: Optional[str] = None

        # Index-side artifacts
        self.entity_idf: Dict[str, float] = {}
        self.chunk_to_entities: Dict[str, Set[str]] = {}
        self.bridge_neighbor_cache: Dict[str, List[str]] = {}
        self.entity_key_to_local_idx: Dict[str, int] = {}
        self.passage_key_to_local_idx: Dict[str, int] = {}

        self._route_artifacts_path = os.path.join(
            self.working_dir, "route_index_artifacts.json"
        )

        self.reset_retrieval_diagnostics()

        logger.info("BindPack initialized: stage_ablation=%s", self.config.stage_ablation)

    # ---------------------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------------------
    def set_query_hop_overrides(self, hops: List[int], source: str) -> None:
        """Use one externally supplied hop count for each query in the next retrieve.

        The list is aligned by position, never by question text. Use benchmark
        structure labels or an explicit dataset prior (for example, one for NQ).
        ``retrieve`` requires and validates these labels before starting work.
        """
        if not isinstance(hops, list) or any(
            isinstance(hop, bool) or not isinstance(hop, Integral) or not 1 <= hop <= 4
            for hop in hops
        ):
            raise ValueError("hops must be a list of integers from 1 through 4")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("hop override source must be a nonempty string")
        self._query_hop_overrides = [int(hop) for hop in hops]
        self._query_hop_override_source = source.strip()

    def clear_query_hop_overrides(self) -> None:
        """Clear the labels; subsequent retrieval requires new explicit labels."""
        self._query_hop_overrides = None
        self._query_hop_override_source = None

    def _qd_subquestion_limit(self, hops: int) -> int:
        """Preserve the old three-question prompt except for labeled four-hop items."""
        configured = self.config.route.refinement.max_sub_questions
        if self._query_hop_overrides is None:
            return configured
        return min(configured, max(3, hops))

    def reset_retrieval_diagnostics(self):
        super().reset_retrieval_diagnostics()
        cfg = getattr(self, "config", None)

        route_enabled = bool(cfg is not None and cfg.route_enabled)
        self.retrieval_diagnostics.update(
            {
                "hop_counter": {"1": 0, "2": 0, "3": 0, "4": 0},
                "hop_override_source": getattr(
                    self, "_query_hop_override_source", None
                ),
                "avg_bridge_entities": 0.0,
                "avg_path_candidates": 0.0,
                "avg_selected_paths": 0.0,
                "avg_iterative_round2_seeds": 0.0,
                "iterative_used_count": 0,
                "qd_used_count": 0,
                "avg_qd_sub_questions": 0.0,
                "avg_qd_seq_rewrites": 0.0,
                "qd_query_ppr_used_count": 0,
                "qd_query_ppr_failed_count": 0,
                "pcqd_used_count": 0,
                "avg_pcqd_sub_questions": 0.0,
                "avg_path_hints": 0.0,
                "pcqd_entity_replacement_rate": 0.0,
                "pcqd_fallback_rate": 0.0,
                "pcqd_conflict_rate": 0.0,
                "_pcqd_entity_replacement_num": 0.0,
                "_pcqd_entity_replacement_den": 0.0,
                "_pcqd_fallback_count": 0,
                "_pcqd_conflict_sum": 0.0,
                "fallback_query_ppr_count": 0,
                "fallback_strategy_counter": {},
                "module_usage": {
                    "qcappr": int(route_enabled),
                    "eba": int(route_enabled),
                    "path_set_opt": int(route_enabled),
                    "iterative_retrieval": int(route_enabled),
                    "query_decomposition": int(route_enabled),
                    "path_conditioned_qd": int(route_enabled),
                    "entity_idf_index": int(route_enabled),
                    "bridge_cache_index": int(route_enabled),
                },
                "per_query": [],
                "stage_ablation": getattr(cfg, "stage_ablation", "full"),
            }
        )

    def get_retrieval_diagnostics(self) -> Dict[str, Any]:
        diag = json.loads(json.dumps(self.retrieval_diagnostics))
        total = int(diag.get("total_queries", 0))
        safe_total = max(1, total)
        fallback = int(diag.get("fallback_to_dpr_count", 0))
        query_ppr_fallback = int(diag.get("fallback_query_ppr_count", 0))
        no_facts = int(diag.get("no_facts_count", 0))
        diag["fallback_rate"] = float(fallback / safe_total)
        diag["no_facts_rate"] = float(no_facts / safe_total)
        diag["fallback_to_dpr_rate"] = float(fallback / safe_total)
        diag["fallback_query_ppr_rate"] = float(query_ppr_fallback / safe_total)
        # hide internal accumulators
        diag.pop("_pcqd_entity_replacement_num", None)
        diag.pop("_pcqd_entity_replacement_den", None)
        diag.pop("_pcqd_fallback_count", None)
        diag.pop("_pcqd_conflict_sum", None)
        return diag

    # ---------------------------------------------------------------------
    # Entity labels used by grounding and index-side artifacts
    # ---------------------------------------------------------------------

    def _entity_label(self, ent_key: str) -> str:
        try:
            row = self.entity_embedding_store.get_row(ent_key)
            if isinstance(row, dict):
                content = str(row.get("content", "")).strip()
                if content:
                    return content
        except Exception:
            pass
        return self._entity_surface(ent_key)

    def _iter_retrieval_states(self, queries: List[str]):
        """Bound outstanding LLM jobs to one small, ordered query window."""
        workers = self.config.llm_prefetch_workers
        evidence_runtime = getattr(self, "evidence_runtime", None)
        if (
            workers <= 1 or not self.config.route_enabled
        ) and evidence_runtime is None:
            for q_idx, query in enumerate(queries):
                yield {"query_idx": q_idx, "query": query}
            return

        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="bindpack-route"
        ) as executor:
            for start in range(0, len(queries), workers):
                states: List[Dict[str, Any]] = []
                # Embed/read scores on the caller thread. Only independent LLM
                # fact-filter calls enter the bounded worker pool.
                for q_idx in range(start, min(start + workers, len(queries))):
                    query = queries[q_idx]
                    query_fact_scores = self.get_fact_scores(query)
                    state: Dict[str, Any] = {
                        "query_idx": q_idx,
                        "query": query,
                        "fact_scores": query_fact_scores,
                        "rerank_future": executor.submit(
                            self.rerank_facts, query, query_fact_scores
                        ),
                    }
                    states.append(state)
                for state in states:
                    q_idx, query = state["query_idx"], state["query"]
                    query_fact_scores = state.pop("fact_scores")
                    top_k_fact_indices, top_k_facts, rerank_log = state.pop(
                        "rerank_future"
                    ).result()
                    state["rerank"] = (
                        query_fact_scores,
                        top_k_fact_indices,
                        top_k_facts,
                        rerank_log,
                    )
                    if top_k_facts:
                        hop_override = (
                            self._query_hop_overrides[q_idx]
                            if self._query_hop_overrides is not None
                            else None
                        )
                        base = self._path_graph_search(
                            query=query,
                            query_fact_scores=query_fact_scores,
                            top_k_facts=top_k_facts,
                            top_k_fact_indices=top_k_fact_indices,
                            hop_override=hop_override,
                            defer_qd=True,
                        )
                        state["base"] = base
                        _base_ids, _base_scores, ctx = base
                        if (
                            ctx.get("_qd_deferred")
                            and int(ctx["hops"]) >= self.config.route.refinement.min_hops
                        ):
                            state["qd_plan"] = True
                    elif evidence_runtime is not None:
                        ids, scores = self.dense_passage_retrieval(query)
                        hops = self._query_hop_overrides[q_idx]
                        state["base"] = (
                            ids,
                            scores,
                            {
                                "hops": hops,
                                "seed_entities": [],
                                "seed_distribution": {},
                                "bridges_by_seed": {},
                            },
                        )
                        if (
                            self.config.route_enabled
                            and hops >= self.config.route.refinement.min_hops
                        ):
                            state["qd_plan"] = True

                # Phase 1: parallelize independent static decomposition calls.
                for state in states:
                    if "qd_plan" in state:
                        state["static_future"] = executor.submit(
                            self._decompose_query,
                            state["query"],
                            self._qd_subquestion_limit(int(state["base"][2]["hops"])),
                        )
                for state in states:
                    if "static_future" in state:
                        state["static_sub_questions"] = state.pop(
                            "static_future"
                        ).result()

                # Phase 2: build static tracks and path hints serially.  The
                # original path only asks PCQD after a usable static track.
                for state in states:
                    if "qd_plan" not in state:
                        continue
                    static_sub_questions = state["static_sub_questions"]
                    if not static_sub_questions:
                        continue
                    base_ids, base_scores, ctx = state["base"]
                    static_track = self._build_qd_track_ranking(
                        query=state["query"],
                        hops=int(ctx["hops"]),
                        sub_questions=static_sub_questions,
                        base_sorted_doc_ids=base_ids,
                        original_seed_entities=set(ctx["seed_entities"]),
                    )
                    state["static_track"] = static_track
                    static_ids, static_scores, static_stats = static_track
                    static_used = bool(
                        static_stats.get("used", False)
                        and static_ids is not None
                        and static_scores is not None
                    )
                    if static_used:
                        state["path_hints"] = self._build_pcqd_path_hints(
                            base_sorted_doc_ids=base_ids,
                            base_sorted_doc_scores=base_scores,
                            seed_distribution=ctx["seed_distribution"],
                            bridges_by_seed=ctx["bridges_by_seed"],
                        )
                        state["hint_diagnostics"] = dict(
                            getattr(self, "_last_pcqd_hint_stats", {})
                        )

                # Phase 3: parallelize PCQD calls only where the old serial
                # path would make one, then finalize every query in input order.
                for state in states:
                    if "path_hints" in state and state["path_hints"][0]:
                        state["pcqd_future"] = executor.submit(
                            self._decompose_query_with_hints,
                            state["query"],
                            state["path_hints"][0],
                            self._qd_subquestion_limit(int(state["base"][2]["hops"])),
                        )
                for state in states:
                    if "pcqd_future" in state:
                        state["pcqd_sub_questions"] = state.pop("pcqd_future").result()
                if evidence_runtime is not None:
                    evidence_runtime.process_window(states, executor)
                for state in states:
                    yield state

    def retrieve(
        self,
        queries: List[str],
        num_to_retrieve: Optional[int] = None,
        gold_docs: Optional[List[List[str]]] = None,
    ):
        if self._query_hop_overrides is None:
            raise ValueError(
                "BindPack requires one explicit hop count per query. Call "
                "set_query_hop_overrides(hops, source) with benchmark labels "
                "or a declared dataset prior before retrieve()."
            )
        if len(self._query_hop_overrides) != len(queries):
            raise ValueError(
                "query hop override count must match the number of queries: "
                f"{len(self._query_hop_overrides)} != {len(queries)}"
            )
        retrieve_start_time = time.time()
        self.reset_retrieval_diagnostics()

        if num_to_retrieve is None:
            num_to_retrieve = self.config.retrieval_top_k

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        if not self.chunk_to_entities:
            # lazy recovery for runs using existing index not built by current process
            self._build_index_side_artifacts()

        self.entity_key_to_local_idx = {
            k: i for i, k in enumerate(self.entity_node_keys)
        }
        self.passage_key_to_local_idx = {
            k: i for i, k in enumerate(self.passage_node_keys)
        }

        self.get_query_embeddings(queries)

        retrieval_results: List[QuerySolution] = []
        evidence_runtime = getattr(self, "evidence_runtime", None)
        review_enabled = self.config.pack_enabled
        review_inputs = []

        for state in tqdm(
            self._iter_retrieval_states(queries),
            total=len(queries),
            desc="BindPack Retrieving",
        ):
            q_idx, query = state["query_idx"], state["query"]
            if "rerank" in state:
                query_fact_scores, top_k_fact_indices, top_k_facts, rerank_log = state[
                    "rerank"
                ]
            else:
                query_fact_scores = self.get_fact_scores(query)
                top_k_fact_indices, top_k_facts, rerank_log = self.rerank_facts(
                    query, query_fact_scores
                )
            no_facts_reason = (
                rerank_log.get("no_facts_reason", "none")
                if isinstance(rerank_log, dict)
                else "none"
            )

            if len(top_k_facts) == 0:
                fallback_strategy = "dpr"
                if isinstance(rerank_log, dict):
                    rerank_log["fallback_strategy"] = fallback_strategy
                sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)
                fallback_to_dpr = True
                self._record_retrieval_diagnostic(
                    query_idx=q_idx,
                    query=query,
                    fallback_to_dpr=True,
                    rerank_log=rerank_log,
                )
                self._increment_counter(
                    self.retrieval_diagnostics["fallback_strategy_counter"], "dpr"
                )
                trace = {
                    "stage_ablation": self.config.stage_ablation,
                    "fact_filter": rerank_log,
                    "dense_fallback": fallback_to_dpr,
                    "assigned_hops": self._query_hop_overrides[q_idx]
                    if self._query_hop_overrides is not None
                    else None,
                }
                if self.config.stage_ablation != "full":
                    trace["stage_ablation"] = self.config.stage_ablation
                evidence_runtime = getattr(self, "evidence_runtime", None)
                if evidence_runtime is not None:
                    (
                        sorted_doc_ids,
                        sorted_doc_scores,
                        evidence_trace,
                    ) = evidence_runtime.finalize(
                        query,
                        sorted_doc_ids,
                        sorted_doc_scores,
                        state.get("base", (None, None, {}))[2],
                        state,
                    )
                    trace["evidence"] = evidence_trace
                retrieval_results.append(
                    QuerySolution(
                        question=query,
                        docs=[
                            self.chunk_embedding_store.get_row(
                                self.passage_node_keys[idx]
                            )["content"]
                            for idx in sorted_doc_ids[:num_to_retrieve]
                        ],
                        doc_scores=sorted_doc_scores[:num_to_retrieve],
                        retrieval_trace=trace,
                    )
                )
                self.retrieval_diagnostics["per_query"].append(
                    {
                        "query_idx": q_idx,
                        "no_facts": True,
                        "no_facts_reason": no_facts_reason,
                        "fallback_strategy": "dpr",
                        "hops": None,
                        "hop_source": self._query_hop_override_source,
                        "assigned_hops": self._query_hop_overrides[q_idx]
                        if self._query_hop_overrides is not None
                        else None,
                        "bridge_entities": 0,
                        "path_candidates": 0,
                        "selected_paths": 0,
                        "iterative_round2_seeds": 0,
                        "pcqd_used": False,
                        "pcqd_sub_questions": 0,
                        "qd_sub_query_ppr_used": 0,
                        "qd_sub_query_ppr_failed": 0,
                        "qd_seq_rewrites": 0,
                        "pcqd_path_hints": 0,
                        "pcqd_entity_replacement_rate": 0.0,
                        "pcqd_fallback": False,
                        "pcqd_conflict_rate": 0.0,
                    }
                )
                if review_enabled:
                    review_inputs.append(
                        (retrieval_results[-1], sorted_doc_ids[:num_to_retrieve].copy())
                    )
                continue

            if "base" in state:
                sorted_doc_ids, sorted_doc_scores, ctx = state["base"]
                if "qd_plan" in state:
                    # Propagate terminal request errors: silently dropping QD or
                    # PCQD changes retrieval results without alerting the user.
                    static_sub_questions = state["static_sub_questions"]
                    pcqd_sub_questions = state.get("pcqd_sub_questions")
                    path_hints, conflict_rate = state.get("path_hints", ([], 0.0))
                    (
                        sorted_doc_ids,
                        sorted_doc_scores,
                        qd_stats,
                    ) = self._query_decomposition_retrieval(
                        query=query,
                        hops=int(ctx["hops"]),
                        base_sorted_doc_ids=sorted_doc_ids,
                        base_sorted_doc_scores=sorted_doc_scores,
                        seed_entities=ctx["seed_entities"],
                        seed_distribution=ctx["seed_distribution"],
                        bridges_by_seed=ctx["bridges_by_seed"],
                        prefetched_qd=(
                            static_sub_questions,
                            pcqd_sub_questions,
                            path_hints,
                            conflict_rate,
                        ),
                        prefetched_static_track=state.get("static_track"),
                    )
                    ctx.update(qd_stats)
                ctx.pop("_qd_deferred", None)
            else:
                sorted_doc_ids, sorted_doc_scores, ctx = self._path_graph_search(
                    query=query,
                    query_fact_scores=query_fact_scores,
                    top_k_facts=top_k_facts,
                    top_k_fact_indices=top_k_fact_indices,
                    hop_override=(
                        self._query_hop_overrides[q_idx]
                        if self._query_hop_overrides is not None
                        else None
                    ),
                )
            self._record_retrieval_diagnostic(
                query_idx=q_idx,
                query=query,
                fallback_to_dpr=False,
                rerank_log=rerank_log,
            )

            if self.config.route_enabled:
                (
                    sorted_doc_ids,
                    sorted_doc_scores,
                    path_stats,
                ) = self._apply_path_set_optimization(
                    query=query,
                    base_sorted_doc_ids=sorted_doc_ids,
                    base_sorted_doc_scores=sorted_doc_scores,
                    ctx=ctx,
                )
            else:
                path_stats = {
                    "path_candidates": 0,
                    "selected_paths": 0,
                    "selected_path_candidates": [],
                }
            ctx.update(path_stats)

            trace = {
                "stage_ablation": self.config.stage_ablation,
                "fact_filter": rerank_log,
                "dense_fallback": False,
                "hops": int(ctx["hops"]),
                "seed_entities": ctx["seed_entities"],
                "fact_seed_distribution": ctx.get("fact_seed_distribution", {}),
                "static_sub_questions": state.get("static_sub_questions", []),
                "pcqd_sub_questions": state.get("pcqd_sub_questions", []),
                "pcqd_validation": getattr(self, "_pcqd_diagnostics_by_query", {}).get(
                    query, {}
                ),
                "path_hints": state.get("path_hints", ([], 0.0))[0],
                "hint_diagnostics": state.get("hint_diagnostics", {}),
                "selected_path_candidates": path_stats.get(
                    "selected_path_candidates", []
                ),
            }
            if self.config.stage_ablation != "full":
                trace["stage_ablation"] = self.config.stage_ablation
                trace["route_adapter"] = ctx.get("route_adapter", "enhanced_route")
            evidence_runtime = getattr(self, "evidence_runtime", None)
            if evidence_runtime is not None:
                (
                    sorted_doc_ids,
                    sorted_doc_scores,
                    evidence_trace,
                ) = evidence_runtime.finalize(
                    query, sorted_doc_ids, sorted_doc_scores, ctx, state
                )
                trace["evidence"] = evidence_trace
            top_docs = [
                self.chunk_embedding_store.get_row(self.passage_node_keys[idx])[
                    "content"
                ]
                for idx in sorted_doc_ids[:num_to_retrieve]
            ]
            top_scores = sorted_doc_scores[:num_to_retrieve]
            retrieval_results.append(
                QuerySolution(
                    question=query,
                    docs=top_docs,
                    doc_scores=top_scores,
                    retrieval_trace=trace,
                )
            )
            if review_enabled:
                review_inputs.append(
                    (retrieval_results[-1], sorted_doc_ids[:num_to_retrieve].copy())
                )

            hops = int(ctx["hops"])
            self.retrieval_diagnostics["hop_counter"][str(hops)] += 1
            self.retrieval_diagnostics["avg_bridge_entities"] += float(
                ctx.get("bridge_count", 0)
            )
            self.retrieval_diagnostics["avg_path_candidates"] += float(
                path_stats.get("path_candidates", 0)
            )
            self.retrieval_diagnostics["avg_selected_paths"] += float(
                path_stats.get("selected_paths", 0)
            )
            round2_seed_count = int(ctx.get("iterative_round2_seed_count", 0))
            self.retrieval_diagnostics["avg_iterative_round2_seeds"] += float(
                round2_seed_count
            )
            if ctx.get("iterative_used", False):
                self.retrieval_diagnostics["iterative_used_count"] += 1
            if ctx.get("qd_used", False):
                self.retrieval_diagnostics["qd_used_count"] += 1
            self.retrieval_diagnostics["avg_qd_sub_questions"] += float(
                ctx.get("qd_sub_questions", 0)
            )
            self.retrieval_diagnostics["avg_qd_seq_rewrites"] += float(
                ctx.get("qd_seq_rewrites", 0)
            )
            self.retrieval_diagnostics["qd_query_ppr_used_count"] += int(
                ctx.get("qd_sub_query_ppr_used", 0)
            )
            self.retrieval_diagnostics["qd_query_ppr_failed_count"] += int(
                ctx.get("qd_sub_query_ppr_failed", 0)
            )
            if ctx.get("pcqd_used", False):
                self.retrieval_diagnostics["pcqd_used_count"] += 1
            self.retrieval_diagnostics["avg_pcqd_sub_questions"] += float(
                ctx.get("pcqd_sub_questions", 0)
            )
            self.retrieval_diagnostics["avg_path_hints"] += float(
                ctx.get("pcqd_path_hints", 0)
            )
            self.retrieval_diagnostics["_pcqd_entity_replacement_num"] += float(
                ctx.get("pcqd_entity_replacement_num", 0.0)
            )
            self.retrieval_diagnostics["_pcqd_entity_replacement_den"] += float(
                ctx.get("pcqd_entity_replacement_den", 0.0)
            )
            if ctx.get("pcqd_fallback", False):
                self.retrieval_diagnostics["_pcqd_fallback_count"] += 1
            self.retrieval_diagnostics["_pcqd_conflict_sum"] += float(
                ctx.get("pcqd_conflict_rate", 0.0)
            )

            self.retrieval_diagnostics["per_query"].append(
                {
                    "query_idx": q_idx,
                    "no_facts": False,
                    "no_facts_reason": no_facts_reason,
                    "hops": hops,
                    "hop_source": self._query_hop_override_source,
                    "bridge_entities": int(ctx.get("bridge_count", 0)),
                    "path_candidates": int(path_stats.get("path_candidates", 0)),
                    "selected_paths": int(path_stats.get("selected_paths", 0)),
                    "iterative_round2_seeds": round2_seed_count,
                    "iterative_used": bool(ctx.get("iterative_used", False)),
                    "qd_used": bool(ctx.get("qd_used", False)),
                    "qd_sub_questions": int(ctx.get("qd_sub_questions", 0)),
                    "qd_sub_query_ppr_used": int(ctx.get("qd_sub_query_ppr_used", 0)),
                    "qd_sub_query_ppr_failed": int(
                        ctx.get("qd_sub_query_ppr_failed", 0)
                    ),
                    "qd_seq_rewrites": int(ctx.get("qd_seq_rewrites", 0)),
                    "pcqd_used": bool(ctx.get("pcqd_used", False)),
                    "pcqd_sub_questions": int(ctx.get("pcqd_sub_questions", 0)),
                    "pcqd_path_hints": int(ctx.get("pcqd_path_hints", 0)),
                    "pcqd_entity_replacement_rate": float(
                        ctx.get("pcqd_entity_replacement_rate", 0.0)
                    ),
                    "pcqd_fallback": bool(ctx.get("pcqd_fallback", False)),
                    "pcqd_conflict_rate": float(ctx.get("pcqd_conflict_rate", 0.0)),
                }
            )

        total = max(1, self.retrieval_diagnostics["total_queries"])
        self.retrieval_diagnostics["avg_bridge_entities"] = float(
            self.retrieval_diagnostics["avg_bridge_entities"] / total
        )
        self.retrieval_diagnostics["avg_path_candidates"] = float(
            self.retrieval_diagnostics["avg_path_candidates"] / total
        )
        self.retrieval_diagnostics["avg_selected_paths"] = float(
            self.retrieval_diagnostics["avg_selected_paths"] / total
        )
        self.retrieval_diagnostics["avg_iterative_round2_seeds"] = float(
            self.retrieval_diagnostics["avg_iterative_round2_seeds"] / total
        )
        self.retrieval_diagnostics["avg_qd_sub_questions"] = float(
            self.retrieval_diagnostics["avg_qd_sub_questions"] / total
        )
        self.retrieval_diagnostics["avg_qd_seq_rewrites"] = float(
            self.retrieval_diagnostics["avg_qd_seq_rewrites"] / total
        )
        self.retrieval_diagnostics["avg_pcqd_sub_questions"] = float(
            self.retrieval_diagnostics["avg_pcqd_sub_questions"] / total
        )
        self.retrieval_diagnostics["avg_path_hints"] = float(
            self.retrieval_diagnostics["avg_path_hints"] / total
        )
        repl_den = max(
            1e-9,
            float(self.retrieval_diagnostics.get("_pcqd_entity_replacement_den", 0.0)),
        )
        self.retrieval_diagnostics["pcqd_entity_replacement_rate"] = float(
            float(self.retrieval_diagnostics.get("_pcqd_entity_replacement_num", 0.0))
            / repl_den
        )
        self.retrieval_diagnostics["pcqd_fallback_rate"] = float(
            float(self.retrieval_diagnostics.get("_pcqd_fallback_count", 0.0)) / total
        )
        self.retrieval_diagnostics["pcqd_conflict_rate"] = float(
            float(self.retrieval_diagnostics.get("_pcqd_conflict_sum", 0.0)) / total
        )

        if review_enabled:
            self.retrieval_diagnostics["support_semantic_veto"] = evidence_runtime.pack.postprocess(review_inputs)

        self.all_retrieval_time += time.time() - retrieve_start_time
        logger.info("[BindPack] Total Retrieval Time %.2fs", self.all_retrieval_time)

        if gold_docs is not None:
            evaluator = RetrievalRecall(global_config=self.config)
            k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
            overall, _ = evaluator.calculate_metric_scores(
                gold_docs=gold_docs,
                retrieved_docs=[x.docs for x in retrieval_results],
                k_list=k_list,
            )
            logger.info("[BindPack] Retrieval evaluation: %s", overall)
            return retrieval_results, overall

        return retrieval_results

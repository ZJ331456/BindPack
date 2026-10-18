from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from ...utils.misc_utils import min_max_normalize

logger = logging.getLogger("bindpack.retrieval.engine")


class RouteRefinement:
    """QDRouteRefinement implementation for candidate discovery."""

    _QD_SYSTEM_PROMPT = (
        "You are a multi-hop question decomposer. "
        "Given a complex question, break it into sequential sub-questions "
        "that can each be answered by a single passage. "
        'Output ONLY a JSON object with key "sub_questions" containing a list of strings. '
        "Keep sub-questions concise and self-contained. "
        "Do NOT include the original question in the list."
    )

    _PCQD_SYSTEM_PROMPT = (
        "You are decomposing a multi-hop question with retrieved reasoning hints. "
        "Produce grounded sub-questions using evidence-backed entities only. "
        "Do not hallucinate entities. "
        'Return JSON with key "sub_questions"; each item should be an object with '
        "fields: question (string), grounded_entities (list), supporting_hint_ids (list), confidence (0-1)."
    )

    @staticmethod
    def _extract_llm_text(response_obj: Any) -> str:
        """Best-effort extraction of plain text from heterogeneous LLM return formats."""
        if response_obj is None:
            return ""
        if isinstance(response_obj, str):
            return response_obj
        if isinstance(response_obj, dict):
            for key in ("content", "text", "message"):
                val = response_obj.get(key)
                if isinstance(val, str):
                    return val
                if isinstance(val, dict):
                    nested = RouteRefinement._extract_llm_text(val)
                    if nested:
                        return nested
            return str(response_obj)
        if isinstance(response_obj, list):
            if not response_obj:
                return ""
            # OpenAI-style list of messages / choices.
            first = response_obj[0]
            txt = RouteRefinement._extract_llm_text(first)
            if txt:
                return txt
            return "".join(
                RouteRefinement._extract_llm_text(x) for x in response_obj if x is not None
            )
        return str(response_obj)

    @staticmethod
    def _parse_sub_questions_from_text(
        raw_text: str, max_sub_questions: int
    ) -> List[str]:
        """
        Parse sub-questions from JSON-ish LLM output.
        Accepts strict JSON, fenced JSON, and lightly noisy wrappers.
        """
        text = (raw_text or "").strip()
        if not text:
            return []

        candidates: List[str] = [text]
        # Try extracting the largest JSON object span as a backup.
        left = text.find("{")
        right = text.rfind("}")
        if 0 <= left < right:
            candidates.append(text[left : right + 1])

        for cand in candidates:
            try:
                parsed = json.loads(cand)
            except Exception:
                continue
            sub_qs = parsed.get("sub_questions", []) if isinstance(parsed, dict) else []
            if not isinstance(sub_qs, list):
                continue
            cleaned = [str(q).strip() for q in sub_qs if str(q).strip()]
            if cleaned:
                return cleaned[:max_sub_questions]

        # Final fallback: treat lines as potential sub-questions.
        line_like: List[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            # strip bullet/enumeration prefix
            line = line.lstrip("-*0123456789. )(").strip()
            if line:
                line_like.append(line)
        return line_like[:max_sub_questions]

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = (text or "").lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _entity_surface(ent_key: str) -> str:
        if not isinstance(ent_key, str):
            return ""
        return ent_key[7:] if ent_key.startswith("entity-") else ent_key

    def _grounded_entity_surface(self, ent_key: str) -> str:
        label = self._entity_label(ent_key)
        if re.fullmatch("(?:entity-)?[0-9a-fA-F]{32}", label or ""):
            return ""
        return label

    def _entity_supported_in_evidence(self, ent_key: str, evidence_text: str) -> bool:
        ent = self._normalize_text(self._grounded_entity_surface(ent_key))
        evidence = self._normalize_text(evidence_text)
        if not ent or not evidence:
            return False
        return ent in evidence

    def _decompose_query(self, query: str, max_sub_questions: int) -> List[str]:
        """
        Use the LLM to decompose a multi-hop query into ordered sub-questions.
        Results are cached in ``_qd_cache`` to avoid redundant LLM calls.
        """
        if not hasattr(self, "_qd_cache"):
            self._qd_cache: Dict[Tuple[str, int], List[str]] = {}

        cache_key = (query, max_sub_questions)
        if self.config.route.refinement.cache_decompositions and cache_key in self._qd_cache:
            return self._qd_cache[cache_key]

        messages = [
            {"role": "system", "content": self._QD_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Decompose into at most {max_sub_questions} sub-questions:\n{query}"
                ),
            },
        ]

        try:
            # Do NOT pass response_format=json_object: vLLM guided decoding
            # (xgrammar) can crash the whole engine on this stack. Prompt + parser
            # already expect JSON text.
            infer_ret = self.llm_model.infer(
                messages=messages,
                temperature=self.config.route.refinement.temperature,
            )

            # Compatible with backends returning:
            # (message, metadata), (message, metadata, cache_hit), or bare message
            response_obj = infer_ret
            if isinstance(infer_ret, tuple):
                if len(infer_ret) >= 1:
                    response_obj = infer_ret[0]
                else:
                    response_obj = ""

            raw = self._extract_llm_text(response_obj)
            sub_qs = self._parse_sub_questions_from_text(
                raw, max_sub_questions=max_sub_questions
            )
        except Exception as e:
            logger.exception("[QD] decompose_query failed for %r: %s", query[:60], e)
            raise

        if self.config.route.refinement.cache_decompositions:
            self._qd_cache[cache_key] = sub_qs
        return sub_qs

    def _decompose_query_with_hints(
        self, query: str, path_hints: List[Dict[str, Any]], max_sub_questions: int
    ) -> List[Dict[str, Any]]:
        if not path_hints:
            return []
        if not hasattr(self, "_pcqd_cache"):
            self._pcqd_cache: Dict[str, List[Dict[str, Any]]] = {}
        cache_key = f"{query}::{json.dumps(path_hints, ensure_ascii=False, sort_keys=True)}::{max_sub_questions}"
        if self.config.route.refinement.cache_decompositions and cache_key in self._pcqd_cache:
            return self._pcqd_cache[cache_key]
        hint_lines: List[str] = []
        for i, h in enumerate(path_hints):
            src = str(h.get("source_entity", ""))
            bridge = str(h.get("candidate_answer_or_bridge", ""))
            evidence = str(h.get("evidence", ""))
            conf = float(h.get("confidence", 0.0))
            hint_lines.append(
                f"[{i}] {src} -> {bridge}; confidence={conf:.3f}; evidence={evidence}"
            )
        grounding_instruction = (
            "Replace ambiguous mentions using supported entities from hints."
        )
        user_prompt = (
            f"Original question:\n{query}\n\nRetrieved reasoning hints:\n"
            + "\n".join(hint_lines)
            + f"\n\nTask: generate at most {max_sub_questions} grounded sub-questions.\nRules:\n1. Use entities from hints only when evidence supports them.\n2. {grounding_instruction}\n3. Do not invent unsupported entities.\n4. Return JSON with key sub_questions."
        )
        messages = [
            {"role": "system", "content": self._PCQD_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        structured_sub_qs: List[Dict[str, Any]] = []
        raw = ""
        validation_rejections = []
        try:
            infer_ret = self.llm_model.infer(
                messages=messages, temperature=self.config.route.refinement.temperature
            )
            response_obj = (
                infer_ret[0]
                if isinstance(infer_ret, tuple) and len(infer_ret) >= 1
                else infer_ret
            )
            raw = self._extract_llm_text(response_obj)
            candidates = [raw]
            left, right = (raw.find("{"), raw.rfind("}"))
            if 0 <= left < right:
                candidates.append(raw[left : right + 1])
            for cand in candidates:
                try:
                    parsed = json.loads(cand)
                except Exception:
                    continue
                sq = parsed.get("sub_questions", []) if isinstance(parsed, dict) else []
                if not isinstance(sq, list):
                    continue
                for item in sq:
                    if isinstance(item, str):
                        q = item.strip()
                        if q:
                            structured_sub_qs.append(
                                {
                                    "question": q,
                                    "grounded_entities": [],
                                    "supporting_hint_ids": [],
                                    "confidence": 0.5,
                                }
                            )
                    elif isinstance(item, dict):
                        q = str(item.get("question", "")).strip()
                        if not q:
                            continue
                        ge = item.get("grounded_entities", [])
                        if not isinstance(ge, list):
                            ge = []
                        sh = item.get("supporting_hint_ids", [])
                        if not isinstance(sh, list):
                            sh = []
                        invalid_hint_reference = any(
                            (
                                isinstance(x, bool)
                                or not str(x).strip().isdigit()
                                or int(x) >= len(path_hints)
                                for x in sh
                            )
                        )
                        conf = item.get("confidence", 0.5)
                        try:
                            conf = float(conf)
                        except Exception:
                            conf = 0.5
                        structured_sub_qs.append(
                            {
                                "question": q,
                                "grounded_entities": [
                                    str(x) for x in ge if str(x).strip()
                                ],
                                "supporting_hint_ids": [
                                    int(x) for x in sh if str(x).strip().isdigit()
                                ],
                                "invalid_hint_reference": invalid_hint_reference,
                                "confidence": min(1.0, max(0.0, conf)),
                            }
                        )
                if structured_sub_qs:
                    break
        except Exception as e:
            logger.exception(
                "[PCQD] decompose with hints failed for %r: %s", query[:60], e
            )
            raise
        if not structured_sub_qs:
            fallback_qs = self._decompose_query(
                query, max_sub_questions=max_sub_questions
            )
            structured_sub_qs = [
                {
                    "question": q,
                    "grounded_entities": [],
                    "supporting_hint_ids": [],
                    "confidence": 0.5,
                }
                for q in fallback_qs
            ]
        validated = []
        for item in structured_sub_qs:
            ids = item["supporting_hint_ids"]
            if item.get("invalid_hint_reference") or any(
                (type(i) is not int or i < 0 or i >= len(path_hints) for i in ids)
            ):
                validation_rejections.append(
                    {"question": item["question"], "reason": "invalid_hint_reference"}
                )
                continue
            entities = item["grounded_entities"]
            if entities and (not ids):
                ids = [
                    i
                    for i, h in enumerate(path_hints)
                    if any(
                        (
                            self._normalize_text(ent)
                            in self._normalize_text(str(h.get("evidence", "")))
                            for ent in entities
                        )
                    )
                ]
            referenced = " ".join((str(path_hints[i].get("evidence", "")) for i in ids))
            if any(
                (
                    not self._normalize_text(ent)
                    or self._normalize_text(ent) not in self._normalize_text(referenced)
                    or re.fullmatch("(?:entity-)?[0-9a-fA-F]{32}", ent)
                    for ent in entities
                )
            ):
                validation_rejections.append(
                    {
                        "question": item["question"],
                        "reason": "unsupported_entity_binding",
                    }
                )
                continue
            item["supporting_hint_ids"] = ids
            item["evidence_validated"] = True
            validated.append(item)
        structured_sub_qs = validated
        structured_sub_qs = structured_sub_qs[:max_sub_questions]
        if not hasattr(self, "_pcqd_diagnostics_by_query"):
            self._pcqd_diagnostics_by_query = {}
        self._pcqd_diagnostics_by_query[query] = {
            "raw_response": raw,
            "validation_rejections": validation_rejections,
            "validated_sub_questions": structured_sub_qs,
        }
        if self.config.route.refinement.cache_decompositions:
            self._pcqd_cache[cache_key] = structured_sub_qs
        return structured_sub_qs

    def _build_pcqd_path_hints(
        self,
        base_sorted_doc_ids: np.ndarray,
        base_sorted_doc_scores: np.ndarray,
        seed_distribution: Dict[str, float],
        bridges_by_seed: Dict[str, List[str]],
    ) -> Tuple[List[Dict[str, Any]], float]:
        self._last_pcqd_hint_stats = {
            "strict_accepted": 0,
            "relaxed_accepted": 0,
            "unsupported_rejected": 0,
            "score_rejected": 0,
        }
        if not seed_distribution:
            return ([], 0.0)
        top_n = min(self.config.route.refinement.pcqd.ground_top_docs, len(base_sorted_doc_ids))
        if top_n <= 0:
            return ([], 0.0)
        top_doc_ids = base_sorted_doc_ids[:top_n].tolist()
        top_doc_scores = np.asarray(
            min_max_normalize(base_sorted_doc_scores[:top_n]), dtype=np.float32
        )
        grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
        top_doc_context: List[Tuple[int, str, Set[str], str, float]] = []
        for rank, doc_id in enumerate(top_doc_ids):
            if doc_id >= len(self.passage_node_keys):
                continue
            passage_key = self.passage_node_keys[doc_id]
            doc_entities = self.chunk_to_entities.get(passage_key, set())
            row = self.chunk_embedding_store.get_row(passage_key)
            evidence_text = str(row.get("content", "")) if isinstance(row, dict) else ""
            path_score = (
                float(top_doc_scores[rank]) if rank < len(top_doc_scores) else 0.0
            )
            top_doc_context.append(
                (doc_id, passage_key, doc_entities, evidence_text, path_score)
            )

        def _add_hint(
            seed: str,
            bridge: str,
            passage_key: str,
            evidence_text: str,
            path_score: float,
            strict: bool,
            hint_type: str = "adhoc",
        ):
            if strict:
                if path_score < self.config.route.refinement.pcqd.path_score_threshold:
                    self._last_pcqd_hint_stats["score_rejected"] += 1
                    return
                if not self._entity_supported_in_evidence(bridge, evidence_text):
                    self._last_pcqd_hint_stats["unsupported_rejected"] += 1
                    return
            source_label = self._grounded_entity_surface(seed)
            bridge_label = self._grounded_entity_surface(bridge)
            if (
                not source_label
                or not bridge_label
                or (not self._entity_supported_in_evidence(bridge, evidence_text))
            ):
                self._last_pcqd_hint_stats["unsupported_rejected"] += 1
                return
            self._last_pcqd_hint_stats[
                "strict_accepted" if strict else "relaxed_accepted"
            ] += 1
            snippet = evidence_text[:400]
            location = evidence_text.casefold().find(bridge_label.casefold())
            if location >= 0:
                snippet = evidence_text[
                    max(0, location - 160) : location + len(bridge_label) + 240
                ]
            key = (seed, bridge)
            if key not in grouped:
                grouped[key] = {
                    "_seed_key": seed,
                    "_bridge_key": bridge,
                    "source_entity": source_label,
                    "candidate_answer_or_bridge": bridge_label,
                    "evidence": snippet,
                    "confidence": 0.0,
                    "votes": 0,
                    "max_path_score": 0.0,
                    "support_passages": [],
                    "hint_type": hint_type if strict else "relaxed",
                }
            grouped[key]["votes"] += 1
            grouped[key]["max_path_score"] = max(
                grouped[key]["max_path_score"], path_score
            )
            grouped[key]["confidence"] = max(
                grouped[key]["confidence"],
                0.7 * path_score + 0.3 * float(seed_distribution.get(seed, 0.0)),
            )
            grouped[key]["support_passages"].append(passage_key)

        for _, passage_key, doc_entities, evidence_text, path_score in top_doc_context:
            for seed, bridge_entities in sorted(bridges_by_seed.items()):
                for bridge in bridge_entities:
                    if bridge not in doc_entities:
                        continue
                    _add_hint(
                        seed,
                        bridge,
                        passage_key,
                        evidence_text,
                        path_score,
                        strict=True,
                    )
        if not grouped:
            seed_keys = set(seed_distribution.keys())
            for (
                _,
                passage_key,
                doc_entities,
                evidence_text,
                path_score,
            ) in top_doc_context:
                doc_seed_entities = sorted(seed_keys & doc_entities)
                if not doc_seed_entities:
                    continue
                bridge_pool = sorted(
                    doc_entities - seed_keys,
                    key=lambda e: (-float(self.entity_idf.get(e, 1.0)), e),
                )
                bridge_pool = bridge_pool[: max(1, self.config.route.refinement.pcqd.entity_top_k)]
                for seed in doc_seed_entities:
                    for bridge in bridge_pool:
                        _add_hint(
                            seed,
                            bridge,
                            passage_key,
                            evidence_text,
                            path_score,
                            strict=True,
                        )
        if not grouped:
            relaxed_docs = top_doc_context[:1]
            seed_keys = set(seed_distribution.keys())
            for _, passage_key, doc_entities, evidence_text, path_score in relaxed_docs:
                doc_seed_entities = sorted(seed_keys & doc_entities)
                if not doc_seed_entities:
                    doc_seed_entities = sorted(seed_keys)[:1]
                bridge_pool = sorted(
                    doc_entities - seed_keys,
                    key=lambda e: (-float(self.entity_idf.get(e, 1.0)), e),
                )
                bridge_pool = bridge_pool[: max(1, self.config.route.refinement.pcqd.entity_top_k)]
                for seed in doc_seed_entities:
                    for bridge in bridge_pool:
                        _add_hint(
                            seed,
                            bridge,
                            passage_key,
                            evidence_text,
                            path_score,
                            strict=False,
                        )
        hints = list(grouped.values())
        pre_vote_bridges = defaultdict(set)
        for hint in hints:
            pre_vote_bridges[hint["_seed_key"]].add(hint["_bridge_key"])
        pre_vote_conflict = sum((len(b) > 1 for b in pre_vote_bridges.values())) / max(
            1, len(pre_vote_bridges)
        )
        if hints and self.config.route.refinement.pcqd.bridge_voting:
            per_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for h in hints:
                per_source[str(h.get("_seed_key", ""))].append(h)
            voted_hints: List[Dict[str, Any]] = []
            for src in sorted(per_source):
                cands = per_source[src]
                cands.sort(
                    key=lambda x: (
                        -int(x.get("votes", 0)),
                        -float(x.get("confidence", 0.0)),
                        -float(x.get("max_path_score", 0.0)),
                        -float(self.entity_idf.get(str(x.get("_bridge_key", "")), 1.0)),
                        str(x.get("_bridge_key", "")),
                    )
                )
                keep = cands[: self.config.route.refinement.pcqd.max_bridges_per_source]
                voted_hints.extend(keep)
            hints = voted_hints
        hints.sort(
            key=lambda x: (
                -float(x["confidence"]),
                -int(x["votes"]),
                str(x.get("_seed_key", "")),
                str(x.get("_bridge_key", "")),
            )
        )
        if self.config.route.refinement.pcqd.entity_top_k > 0:
            hints = hints[: self.config.route.refinement.pcqd.entity_top_k]
        source_to_bridges_selected: Dict[str, Set[str]] = defaultdict(set)
        for h in hints:
            src = str(h.get("_seed_key", ""))
            br = str(h.get("_bridge_key", ""))
            if src and br:
                source_to_bridges_selected[src].add(br)
        for i, h in enumerate(hints):
            h["hint_id"] = i
            h.pop("_seed_key", None)
            h.pop("_bridge_key", None)
        total_sources = sum(
            (
                1
                for _, bridges in source_to_bridges_selected.items()
                if len(bridges) >= 1
            )
        )
        conflict_sources = sum(
            (1 for _, bridges in source_to_bridges_selected.items() if len(bridges) > 1)
        )
        conflict_rate = float(conflict_sources / max(1, total_sources))
        conflict_rate = float(pre_vote_conflict)
        self._last_pcqd_hint_stats["conflict_before_voting"] = float(pre_vote_conflict)
        self._last_pcqd_hint_stats["selected_hints"] = len(hints)
        return (hints, conflict_rate)

    def _build_qd_track_ranking(
        self,
        query: str,
        hops: int,
        sub_questions: List[str],
        base_sorted_doc_ids: np.ndarray,
        original_seed_entities: Set[str],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
        sub_qs = [q.strip() for q in sub_questions if isinstance(q, str) and q.strip()]
        if not sub_qs:
            return (
                None,
                None,
                {
                    "used": False,
                    "seed_count": 0,
                    "sub_doc_count": 0,
                    "sub_retrieval_mode": "dpr",
                    "sub_query_ppr_used": 0,
                    "sub_query_ppr_failed": 0,
                    "sub_seq_rewrites": 0,
                },
            )

        sub_doc_ids_all: List[int] = []
        local_doc_weights = defaultdict(float)
        ppr_used_count = 0
        ppr_failed_count = 0
        seq_rewrite_count = 0
        retrieval_mode = "dpr"
        for sub_q in sub_qs:
            try:
                sub_ids, _ = self.dense_passage_retrieval(sub_q)
            except Exception as e:
                logger.warning("[QD] DPR failed for sub-question %r: %s", sub_q[:60], e)
                continue
            local_ids = sub_ids[: self.config.route.refinement.sub_retrieval_top_k].tolist()
            sub_doc_ids_all.extend(local_ids)
            for rank, doc_id in enumerate(local_ids):
                local_doc_weights[doc_id] += 1.0 / (1.0 + rank) / len(sub_qs)

        if not sub_doc_ids_all:
            return (
                None,
                None,
                {
                    "used": False,
                    "seed_count": 0,
                    "sub_doc_count": 0,
                    "sub_retrieval_mode": retrieval_mode,
                    "sub_query_ppr_used": ppr_used_count,
                    "sub_query_ppr_failed": ppr_failed_count,
                    "sub_seq_rewrites": seq_rewrite_count,
                },
            )

        sub_doc_ids_unique = list(dict.fromkeys(sub_doc_ids_all))
        sub_entities = self._collect_entities_from_top_docs(sub_doc_ids_unique)
        scored_seeds = self._select_iterative_bridge_seeds(
            query=query,
            candidate_entities=sub_entities,
            original_seed_entities=original_seed_entities,
        )
        if not scored_seeds:
            return (
                None,
                None,
                {
                    "used": False,
                    "seed_count": 0,
                    "sub_doc_count": len(sub_doc_ids_unique),
                    "sub_retrieval_mode": retrieval_mode,
                    "sub_query_ppr_used": ppr_used_count,
                    "sub_query_ppr_failed": ppr_failed_count,
                    "sub_seq_rewrites": seq_rewrite_count,
                },
            )

        seed_distribution = self._softmax(dict(scored_seeds), temperature=0.7)
        num_nodes = len(self.graph.vs["name"])
        phrase_weights = np.zeros(num_nodes, dtype=np.float32)
        passage_weights = np.zeros(num_nodes, dtype=np.float32)

        for ent_key, prob in seed_distribution.items():
            v_idx = self.node_name_to_vertex_idx.get(ent_key)
            if v_idx is not None:
                phrase_weights[v_idx] += float(prob)

        sub_scores = np.zeros(len(self.passage_node_keys), dtype=np.float32)
        for rank, doc_id in enumerate(sub_doc_ids_unique):
            if doc_id >= len(self.passage_node_keys):
                continue
            sub_scores[doc_id] = local_doc_weights[doc_id]
        sub_scores_norm = np.asarray(min_max_normalize(sub_scores), dtype=np.float32)

        for doc_id in sub_doc_ids_unique:
            if doc_id >= len(self.passage_node_keys):
                continue
            pkey = self.passage_node_keys[doc_id]
            pidx = self.node_name_to_vertex_idx.get(pkey)
            if pidx is None:
                continue
            passage_weights[pidx] = (
                float(sub_scores_norm[doc_id]) * self.config.passage_node_weight
            )

        node_weights = phrase_weights + passage_weights
        if float(np.sum(node_weights)) <= 0:
            return (
                None,
                None,
                {
                    "used": False,
                    "seed_count": len(seed_distribution),
                    "sub_doc_count": len(sub_doc_ids_unique),
                    "sub_retrieval_mode": retrieval_mode,
                    "sub_query_ppr_used": ppr_used_count,
                    "sub_query_ppr_failed": ppr_failed_count,
                    "sub_seq_rewrites": seq_rewrite_count,
                },
            )

        damping, _ = self._qcappr_params(hops)
        track_ids, track_scores = self.run_ppr(node_weights, damping=damping)
        return (
            track_ids,
            track_scores,
            {
                "used": True,
                "seed_count": len(seed_distribution),
                "sub_doc_count": len(sub_doc_ids_unique),
                "sub_retrieval_mode": retrieval_mode,
                "sub_query_ppr_used": ppr_used_count,
                "sub_query_ppr_failed": ppr_failed_count,
                "sub_seq_rewrites": seq_rewrite_count,
            },
        )

    @staticmethod
    def _track_scores_to_full(
        num_passages: int, sorted_ids: np.ndarray, sorted_scores: np.ndarray
    ) -> np.ndarray:
        full = np.zeros(num_passages, dtype=np.float32)
        norm_scores = np.asarray(min_max_normalize(sorted_scores), dtype=np.float32)
        full[sorted_ids] = norm_scores
        return full

    def _query_decomposition_retrieval(
        self,
        query: str,
        hops: int,
        base_sorted_doc_ids: np.ndarray,
        base_sorted_doc_scores: np.ndarray,
        seed_entities: Set[str],
        seed_distribution: Dict[str, float],
        bridges_by_seed: Dict[str, List[str]],
        prefetched_qd: Optional[
            Tuple[
                List[str], Optional[List[Dict[str, Any]]], List[Dict[str, Any]], float
            ]
        ] = None,
        prefetched_static_track: Optional[
            Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]
        ] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        if hops < self.config.route.refinement.min_hops:
            return (
                base_sorted_doc_ids,
                base_sorted_doc_scores,
                {
                    "qd_used": False,
                    "qd_sub_questions": 0,
                    "qd_sub_query_ppr_used": 0,
                    "qd_sub_query_ppr_failed": 0,
                    "qd_seq_rewrites": 0,
                    "pcqd_used": False,
                    "pcqd_sub_questions": 0,
                    "pcqd_path_hints": 0,
                    "pcqd_entity_replacement_num": 0.0,
                    "pcqd_entity_replacement_den": 0.0,
                    "pcqd_entity_replacement_rate": 0.0,
                    "pcqd_fallback": False,
                    "pcqd_conflict_rate": 0.0,
                },
            )

        max_sub_questions = self._qd_subquestion_limit(hops)
        static_sub_questions = (
            prefetched_qd[0]
            if prefetched_qd is not None
            else self._decompose_query(query, max_sub_questions)
        )
        if not static_sub_questions:
            return (
                base_sorted_doc_ids,
                base_sorted_doc_scores,
                {
                    "qd_used": False,
                    "qd_sub_questions": 0,
                    "qd_sub_query_ppr_used": 0,
                    "qd_sub_query_ppr_failed": 0,
                    "qd_seq_rewrites": 0,
                    "pcqd_used": False,
                    "pcqd_sub_questions": 0,
                    "pcqd_path_hints": 0,
                    "pcqd_entity_replacement_num": 0.0,
                    "pcqd_entity_replacement_den": 0.0,
                    "pcqd_entity_replacement_rate": 0.0,
                    "pcqd_fallback": False,
                    "pcqd_conflict_rate": 0.0,
                },
            )

        static_ids, static_scores, static_stats = (
            prefetched_static_track
            if prefetched_static_track is not None
            else self._build_qd_track_ranking(
                query=query,
                hops=hops,
                sub_questions=static_sub_questions,
                base_sorted_doc_ids=base_sorted_doc_ids,
                original_seed_entities=set(seed_entities),
            )
        )
        static_used = bool(
            static_stats.get("used", False)
            and static_ids is not None
            and static_scores is not None
        )
        if not static_used:
            return (
                base_sorted_doc_ids,
                base_sorted_doc_scores,
                {
                    "qd_used": False,
                    "qd_sub_questions": len(static_sub_questions),
                    "qd_sub_query_ppr_used": int(
                        static_stats.get("sub_query_ppr_used", 0)
                    ),
                    "qd_sub_query_ppr_failed": int(
                        static_stats.get("sub_query_ppr_failed", 0)
                    ),
                    "qd_seq_rewrites": int(static_stats.get("sub_seq_rewrites", 0)),
                    "pcqd_used": False,
                    "pcqd_sub_questions": 0,
                    "pcqd_path_hints": 0,
                    "pcqd_entity_replacement_num": 0.0,
                    "pcqd_entity_replacement_den": 0.0,
                    "pcqd_entity_replacement_rate": 0.0,
                    "pcqd_fallback": False,
                    "pcqd_conflict_rate": 0.0,
                },
            )

        # Old behavior: base + static QD merge

        # PC-QD path hints
        if prefetched_qd is not None:
            path_hints, conflict_rate = prefetched_qd[2], prefetched_qd[3]
        else:
            path_hints, conflict_rate = self._build_pcqd_path_hints(
                base_sorted_doc_ids=base_sorted_doc_ids,
                base_sorted_doc_scores=base_sorted_doc_scores,
                seed_distribution=seed_distribution,
                bridges_by_seed=bridges_by_seed,
            )
        pcqd_fallback = len(path_hints) == 0
        pcqd_structured_sub_qs: List[Dict[str, Any]] = []
        if path_hints:
            pcqd_structured_sub_qs = (
                prefetched_qd[1]
                if prefetched_qd is not None and prefetched_qd[1] is not None
                else self._decompose_query_with_hints(
                    query=query,
                    path_hints=path_hints,
                    max_sub_questions=max_sub_questions,
                )
            )

        pcqd_questions: List[str] = []
        replacement_num = 0.0
        replacement_den = 0.0
        for item in pcqd_structured_sub_qs:
            q = str(item.get("question", "")).strip()
            if not q:
                continue
            pcqd_questions.append(q)
            replacement_den += 1.0
            grounded_entities = item.get("grounded_entities", [])
            if not isinstance(grounded_entities, list):
                grounded_entities = []
            q_norm = self._normalize_text(q)
            replaced = any(
                self._normalize_text(str(ent)) in q_norm
                for ent in grounded_entities
                if str(ent).strip()
            )
            if replaced:
                replacement_num += 1.0

        pcqd_ids = None
        pcqd_scores = None
        pcqd_used = False
        pcqd_stats: Dict[str, Any] = {}
        if pcqd_questions:
            pcqd_ids, pcqd_scores, pcqd_stats = self._build_qd_track_ranking(
                query=query,
                hops=hops,
                sub_questions=pcqd_questions,
                base_sorted_doc_ids=base_sorted_doc_ids,
                original_seed_entities=set(seed_entities),
            )
            pcqd_used = bool(
                pcqd_stats.get("used", False)
                and pcqd_ids is not None
                and pcqd_scores is not None
            )
            if not pcqd_used:
                pcqd_fallback = True
        else:
            pcqd_fallback = True

        # If PC-QD failed and static fallback is disabled, keep base ranking.
        if (
            pcqd_fallback
            and not self.config.route.refinement.pcqd.fallback_to_static
            and not self.config.route.refinement.pcqd.include_static
        ):
            return (
                base_sorted_doc_ids,
                base_sorted_doc_scores,
                {
                    "qd_used": True,
                    "qd_sub_questions": len(static_sub_questions),
                    "qd_sub_query_ppr_used": int(
                        static_stats.get("sub_query_ppr_used", 0)
                    ),
                    "qd_sub_query_ppr_failed": int(
                        static_stats.get("sub_query_ppr_failed", 0)
                    ),
                    "qd_seq_rewrites": int(static_stats.get("sub_seq_rewrites", 0)),
                    "pcqd_used": False,
                    "pcqd_sub_questions": 0,
                    "pcqd_path_hints": len(path_hints),
                    "pcqd_entity_replacement_num": replacement_num,
                    "pcqd_entity_replacement_den": replacement_den,
                    "pcqd_entity_replacement_rate": float(
                        replacement_num / max(1.0, replacement_den)
                    ),
                    "pcqd_fallback": True,
                    "pcqd_conflict_rate": conflict_rate,
                },
            )

        # 3-track fusion: base + static + path-conditioned
        num_passages = len(self.passage_node_keys)
        base_full = self._track_scores_to_full(
            num_passages, base_sorted_doc_ids, base_sorted_doc_scores
        )
        static_full = (
            self._track_scores_to_full(num_passages, static_ids, static_scores)
            if static_used
            else np.zeros(num_passages, dtype=np.float32)
        )
        pcqd_full = (
            self._track_scores_to_full(num_passages, pcqd_ids, pcqd_scores)
            if pcqd_used and pcqd_ids is not None and pcqd_scores is not None
            else np.zeros(num_passages, dtype=np.float32)
        )

        w_base = float(self.config.route.refinement.pcqd.weight_base)
        w_static = (
            float(self.config.route.refinement.pcqd.weight_static)
            if self.config.route.refinement.pcqd.include_static
            else 0.0
        )
        w_path = float(self.config.route.refinement.pcqd.weight_path) if pcqd_used else 0.0
        if not static_used:
            w_static = 0.0
        if pcqd_fallback and self.config.route.refinement.pcqd.fallback_to_static:
            w_path = 0.0

        w_sum = w_base + w_static + w_path
        if w_sum <= 0:
            w_base, w_static, w_path, w_sum = 1.0, 0.0, 0.0, 1.0
        w_base, w_static, w_path = w_base / w_sum, w_static / w_sum, w_path / w_sum

        fused = w_base * base_full + w_static * static_full + w_path * pcqd_full
        fused_ids = np.argsort(fused)[::-1]
        fused_scores = fused[fused_ids]

        return (
            fused_ids,
            fused_scores,
            {
                "qd_used": True,
                "qd_sub_questions": len(static_sub_questions),
                "qd_sub_query_ppr_used": int(static_stats.get("sub_query_ppr_used", 0))
                + int(pcqd_stats.get("sub_query_ppr_used", 0)),
                "qd_sub_query_ppr_failed": int(
                    static_stats.get("sub_query_ppr_failed", 0)
                )
                + int(pcqd_stats.get("sub_query_ppr_failed", 0)),
                "qd_seq_rewrites": int(static_stats.get("sub_seq_rewrites", 0))
                + int(pcqd_stats.get("sub_seq_rewrites", 0)),
                "pcqd_used": bool(pcqd_used),
                "pcqd_sub_questions": len(pcqd_questions),
                "pcqd_path_hints": len(path_hints),
                "pcqd_entity_replacement_num": replacement_num,
                "pcqd_entity_replacement_den": replacement_den,
                "pcqd_entity_replacement_rate": float(
                    replacement_num / max(1.0, replacement_den)
                ),
                "pcqd_fallback": bool(pcqd_fallback),
                "pcqd_conflict_rate": float(conflict_rate),
            },
        )


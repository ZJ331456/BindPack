"""Dataset-native evaluation labels, independent of retrieval and predicted plans.

Document identity deliberately matches ``scripts/eval_utils.py``: exact
``title + '\\n' + text`` strings, including the dataset-specific sentence join.
Only MuSiQue currently provides a supported native dependency annotation.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable, Mapping


GOLD_GRAPH_VERSION = "musique-native-position-dag-v1"
DOCUMENT_MAPPING_VERSION = "canonical-passage-exact-v1"


def _invalid_graph(reason: str, detail: str | None = None) -> dict[str, Any]:
    result = {
        "valid": False, "reason": reason, "version": GOLD_GRAPH_VERSION,
        "mapping_version": DOCUMENT_MAPPING_VERSION, "nodes": [], "edges": [],
        "node_count": 0,
    }
    if detail is not None:
        result["detail"] = detail
    return result


def _passage(item: Mapping[str, Any], *, paragraph_fallback: bool = True) -> str:
    title = item.get("title")
    body = item.get("text")
    if paragraph_fallback:
        body = body or item.get("paragraph_text", "")
    if not isinstance(title, str) or not isinstance(body, str):
        raise ValueError("invalid_passage_text")
    return title + "\n" + body


def gold_documents_from_sample(sample: Mapping[str, Any], dataset_name: str) -> dict[str, Any]:
    """Extract document labels without silently dropping unresolved annotations.

    The two documented permissive policies for NQ-like contexts and paragraph
    flags are retained and named explicitly in the returned label metadata.
    """
    try:
        if "supporting_facts" in sample:
            facts = sample["supporting_facts"]
            if not isinstance(facts, list) or any(
                not isinstance(fact, (list, tuple)) or not fact
                or not isinstance(fact[0], str) for fact in facts
            ):
                raise ValueError("invalid_supporting_facts")
            titles = {fact[0] for fact in facts}
            contexts = sample.get("context")
            if not isinstance(contexts, list):
                raise ValueError("missing_context")
            docs, resolved = [], set()
            for context in contexts:
                if not isinstance(context, (list, tuple)) or len(context) != 2:
                    raise ValueError("invalid_context")
                title, sentences = context
                if title not in titles:
                    continue
                if not isinstance(sentences, list) or any(not isinstance(s, str) for s in sentences):
                    raise ValueError("invalid_context_sentences")
                separator = "" if dataset_name.startswith("hotpotqa") or dataset_name == "hover" else " "
                docs.append(title + "\n" + separator.join(sentences))
                resolved.add(title)
            if resolved != titles:
                raise ValueError("supporting_title_missing_from_context")
            policy = "supporting_facts_title_to_full_context"
        elif "contexts" in sample:
            contexts = sample["contexts"]
            if not isinstance(contexts, list) or any(not isinstance(c, Mapping) for c in contexts):
                raise ValueError("invalid_contexts")
            supports = [c for c in contexts if c.get("is_supporting", False)]
            policy = "contexts_explicit_support" if supports else "contexts_all_fallback"
            docs = [_passage(c, paragraph_fallback=False) for c in (supports or contexts)]
        elif "paragraphs" in sample:
            paragraphs = sample["paragraphs"]
            if not isinstance(paragraphs, list) or any(not isinstance(p, Mapping) for p in paragraphs):
                raise ValueError("invalid_paragraphs")
            docs = [_passage(p) for p in paragraphs if p.get("is_supporting") is not False]
            policy = "paragraphs_exclude_explicit_false"
        else:
            raise ValueError("missing_document_labels")
        docs = list(dict.fromkeys(docs))
        if not docs:
            raise ValueError("empty_gold_documents")
        return {
            "valid": True, "reason": None, "count": len(docs), "docs": docs,
            "policy": policy, "mapping_version": DOCUMENT_MAPPING_VERSION,
        }
    except (TypeError, KeyError, ValueError) as exc:
        return {
            "valid": False, "reason": str(exc), "count": 0, "docs": [],
            "mapping_version": DOCUMENT_MAPPING_VERSION,
        }


def validate_gold_graph(nodes: list[Mapping[str, Any]], *, source: str = "independent_gold") -> dict[str, Any]:
    """Validate an AND-dependency graph and explicit OR-of-AND evidence bundles.

    An evidence bundle is sufficient only when *every* document in it appears.
    A node can have several independently annotated sufficient bundles. The
    native MuSiQue builder below always emits a single one-document bundle.
    """
    if not isinstance(nodes, list) or not nodes:
        return _invalid_graph("empty_gold_graph")
    normalized = []
    ids = set()
    for node in nodes:
        if not isinstance(node, Mapping) or not isinstance(node.get("id"), str) or not node["id"]:
            return _invalid_graph("invalid_node_id")
        node_id = node["id"]
        if node_id in ids:
            return _invalid_graph("duplicate_node_id", node_id)
        ids.add(node_id)
        parents = node.get("parents", [])
        if not isinstance(parents, list) or any(not isinstance(p, str) for p in parents):
            return _invalid_graph("invalid_parent_list", node_id)
        bundles = node.get("evidence_bundles")
        if not isinstance(bundles, list) or not bundles:
            return _invalid_graph("missing_evidence_mapping", node_id)
        clean_bundles = []
        for bundle in bundles:
            if not isinstance(bundle, (list, tuple)) or not bundle or any(
                not isinstance(doc, str) or not doc for doc in bundle
            ):
                return _invalid_graph("invalid_evidence_bundle", node_id)
            clean_bundles.append(list(dict.fromkeys(bundle)))
        normalized.append(dict(node, parents=list(dict.fromkeys(parents)), evidence_bundles=clean_bundles))
    by_id = {node["id"]: node for node in normalized}
    for node in normalized:
        if any(parent not in ids for parent in node["parents"]):
            return _invalid_graph("missing_parent_node", node["id"])
    ancestors: dict[str, set[str]] = {}
    active = set()

    def visit(node_id: str) -> set[str]:
        if node_id in active:
            raise ValueError("cyclic_gold_graph")
        if node_id in ancestors:
            return ancestors[node_id]
        active.add(node_id)
        result = set()
        for parent in by_id[node_id]["parents"]:
            result.add(parent)
            result.update(visit(parent))
        active.remove(node_id)
        ancestors[node_id] = result
        return result

    try:
        for node in normalized:
            node["ancestors"] = sorted(visit(node["id"]))
    except (ValueError, RecursionError) as exc:
        return _invalid_graph("cyclic_gold_graph" if isinstance(exc, ValueError) else "graph_too_deep")
    edges = [[parent, node["id"]] for node in normalized for parent in node["parents"]]
    fingerprint = hashlib.sha256(json.dumps(normalized, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return {
        "valid": True, "reason": None, "version": GOLD_GRAPH_VERSION,
        "mapping_version": DOCUMENT_MAPPING_VERSION, "source": source,
        "nodes": normalized, "edges": edges, "node_count": len(normalized),
        "label_sha256": fingerprint, "dependency_semantics": "all_parents_required",
    }


def build_gold_graph(sample: Mapping[str, Any], dataset_name: str,
                     corpus_docs: Iterable[str] | None = None) -> dict[str, Any]:
    """Build MuSiQue gold solely from native decomposition and paragraph labels.

    ``#i`` refers to the one-based position in ``question_decomposition``;
    globally assigned ``step.id`` values are metadata and never reference IDs.
    Missing support/compute annotations are rejected, never reconstructed from
    answers or a method's predicted plan.
    """
    if dataset_name != "musique":
        reason = ("dataset_has_no_verified_gold_dependency_graph"
                  if dataset_name in {"hotpotqa", "2wikimultihopqa", "hover", "nq", "popqa"}
                  else "unsupported_gold_graph_dataset")
        return _invalid_graph(reason, dataset_name)
    steps = sample.get("question_decomposition")
    if not isinstance(steps, list) or not steps:
        return _invalid_graph("missing_question_decomposition")
    paragraphs = sample.get("paragraphs")
    if not isinstance(paragraphs, list):
        return _invalid_graph("missing_paragraphs")
    by_idx = {}
    for paragraph in paragraphs:
        if not isinstance(paragraph, Mapping):
            return _invalid_graph("invalid_paragraph")
        idx = paragraph.get("idx")
        if not isinstance(idx, int) or isinstance(idx, bool) or idx < 0:
            return _invalid_graph("invalid_paragraph_idx")
        if idx in by_idx:
            return _invalid_graph("duplicate_paragraph_idx", str(idx))
        by_idx[idx] = paragraph
    corpus = None if corpus_docs is None else (corpus_docs if isinstance(corpus_docs, (set, frozenset)) else set(corpus_docs))
    nodes, support_idxs = [], set()
    for position, step in enumerate(steps, 1):
        if not isinstance(step, Mapping):
            return _invalid_graph("invalid_decomposition_step", str(position))
        question = step.get("question")
        if not isinstance(question, str) or not question.strip():
            return _invalid_graph("invalid_decomposition_question", str(position))
        references = [int(match.group(1)) for match in re.finditer(r"#([0-9]+)\b", question)]
        if "#" in re.sub(r"#([0-9]+)\b", "", question):
            return _invalid_graph("invalid_dependency_reference", str(position))
        if any(ref < 1 or ref >= position for ref in references):
            return _invalid_graph("dependency_reference_not_earlier_position", str(position))
        idx = step.get("paragraph_support_idx")
        if not isinstance(idx, int) or isinstance(idx, bool) or idx < 0:
            return _invalid_graph("unsupported_missing_support_or_compute_annotation", str(position))
        if idx not in by_idx:
            return _invalid_graph("missing_supporting_paragraph_idx", str(idx))
        paragraph = by_idx[idx]
        if paragraph.get("is_supporting") is not True:
            return _invalid_graph("paragraph_not_explicitly_supporting", str(idx))
        try:
            doc = _passage(paragraph)
        except ValueError:
            return _invalid_graph("invalid_supporting_paragraph_text", str(idx))
        if corpus is not None and doc not in corpus:
            return _invalid_graph("supporting_document_missing_from_corpus", str(idx))
        nodes.append({
            "id": f"s{position}", "parents": [f"s{ref}" for ref in dict.fromkeys(references)],
            "evidence_bundles": [[doc]], "paragraph_idx": idx,
            "source_step_id": step.get("id"), "decomposition_position": position,
        })
        support_idxs.add(idx)
    marked_support_idxs = {idx for idx, paragraph in by_idx.items() if paragraph.get("is_supporting") is True}
    if support_idxs != marked_support_idxs:
        return _invalid_graph("unmapped_supporting_paragraph")
    result = validate_gold_graph(nodes, source="musique.question_decomposition+paragraphs.idx")
    result["corpus_validation"] = "exact_match_passed" if corpus is not None else "not_requested"
    return result

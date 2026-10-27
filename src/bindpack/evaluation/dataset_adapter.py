"""Annotation-aware benchmark adapters, usable before loading any model.

Hop labels describe each benchmark's documented annotation convention. They
are reference annotations, not predictions or measurements of serial DAG
depth. In particular, 2Wiki's support-count buckets are not graph depths.

This module intentionally imports only the standard library so command-line
validation and offline reporting do not initialize the retrieval stack.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "datasets"
MULTIHOP_DATASETS = ("hotpotqa", "2wikimultihopqa", "musique", "hover")
SUPPORTED_DATASETS = MULTIHOP_DATASETS + ("nq", "popqa")
HOVER_LABELS = ("SUPPORTED", "NOT_SUPPORTED")
HOP_POLICY_VERSION = 1


def _dataset_name(dataset_name: str) -> str:
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"No benchmark adapter for dataset={dataset_name!r}")
    return dataset_name


def _sample_mapping(sample: Any, dataset: str, index: int | None = None) -> Mapping:
    if not isinstance(sample, Mapping):
        position = "" if index is None else f" sample index {index}"
        raise ValueError(f"{dataset}{position} must be an object")
    return sample


def _supporting_facts(sample: Mapping, dataset: str, index: int) -> list:
    facts = sample.get("supporting_facts")
    if not isinstance(facts, list) or not facts:
        raise ValueError(f"{dataset} sample index {index} lacks nonempty supporting_facts")
    for fact in facts:
        if (
            not isinstance(fact, (list, tuple))
            or len(fact) != 2
            or not isinstance(fact[0], str)
            or not fact[0].strip()
            or type(fact[1]) is not int
            or fact[1] < 0
        ):
            raise ValueError(
                f"{dataset} sample index {index} has invalid supporting_facts; "
                "expected [nonempty title, nonnegative sentence index] pairs"
            )
    return facts


def get_benchmark_hops(samples: Sequence[Mapping], dataset_name: str) -> list[int]:
    """Read documented hop labels in sample order; never guess missing labels.

    HotpotQA counts distinct supporting titles (two, irrespective of sentence
    count). 2Wiki uses support-fact counts: exactly two maps to 2, four or more
    maps to 4; three is not a documented bucket and is rejected. MuSiQue uses
    decomposition length. HoVer uses native num_hops and checks it against
    distinct supporting titles; its native two-hop examples are also valid.
    """
    dataset = _dataset_name(dataset_name)
    hops: list[int] = []
    for index, value in enumerate(samples):
        sample = _sample_mapping(value, dataset, index)
        if dataset in ("nq", "popqa"):
            hops.append(1)
            continue
        if dataset == "musique":
            decomposition = sample.get("question_decomposition")
            if (
                not isinstance(decomposition, list)
                or len(decomposition) not in (2, 3, 4)
                or any(
                    not isinstance(step, Mapping)
                    or not isinstance(step.get("question"), str)
                    or not step["question"].strip()
                    for step in decomposition
                )
            ):
                raise ValueError(
                    f"MuSiQue sample index {index} lacks a valid "
                    "2/3/4-step question_decomposition"
                )
            hops.append(len(decomposition))
            continue
        facts = _supporting_facts(sample, dataset, index)
        if dataset == "hotpotqa":
            hop = len({fact[0] for fact in facts})
            if hop != 2:
                raise ValueError(
                    f"HotpotQA sample index {index} has {hop} distinct supporting "
                    "titles; expected exactly 2"
                )
        elif dataset == "2wikimultihopqa":
            count = len(facts)
            if count != 2 and count < 4:
                raise ValueError(
                    f"2Wiki sample index {index} has {count} supporting_facts; "
                    "expected exactly 2 or at least 4 (the 4-hop bucket)"
                )
            hop = 2 if count == 2 else 4
        else:  # HoVer
            hop = sample.get("num_hops")
            if type(hop) is not int or hop not in (2, 3, 4):
                raise ValueError(
                    f"HoVer sample index {index} lacks a valid integer num_hops in 2/3/4"
                )
            distinct = len({fact[0] for fact in facts})
            if hop != distinct:
                raise ValueError(
                    f"HoVer sample index {index}: num_hops={hop} differs from "
                    f"{distinct} distinct supporting titles"
                )
        hops.append(hop)
    return hops


def get_hop_provenance(dataset_name: str) -> dict[str, Any]:
    """Explain annotation source and the distinction from serial reasoning depth."""
    dataset = _dataset_name(dataset_name)
    if dataset in ("nq", "popqa"):
        return {
            "policy_version": HOP_POLICY_VERSION,
            "kind": "dataset_level_prior",
            "value": 1,
            "oracle": False,
            "note": "Single-hop benchmark prior; support count is not a hop label.",
        }
    policies = {
        "hotpotqa": (
            "supporting_facts distinct titles",
            "Count distinct supporting document titles; require exactly 2.",
        ),
        "2wikimultihopqa": (
            "supporting_facts length",
            "Exactly 2 support facts -> 2; >=4 -> 4; reject all other counts.",
        ),
        "musique": (
            "question_decomposition length",
            "Require a valid 2/3/4-step native question decomposition.",
        ),
        "hover": (
            "num_hops",
            "Read native integer 2/3/4; cross-check distinct supporting titles.",
        ),
    }
    field, policy = policies[dataset]
    return {
        "policy_version": HOP_POLICY_VERSION,
        "kind": "per_question_oracle",
        "field": field,
        "oracle": True,
        "policy": policy,
        "serial_dag_depth": False,
        "note": (
            "Uses reference annotations as benchmark hop/effort labels; "
            "these labels do not establish serial dependency DAG depth."
        ),
    }


def describe_benchmark_hops(samples: Sequence[Mapping], dataset_name: str) -> dict[str, Any]:
    """Return an auditable summary, including 2Wiki's raw and capped counts."""
    hops = get_benchmark_hops(samples, dataset_name)
    result = get_hop_provenance(dataset_name)
    result["validated_samples"] = len(samples)
    result["distribution"] = {
        str(hop): count for hop, count in sorted(Counter(hops).items())
    }
    if dataset_name in ("hotpotqa", "2wikimultihopqa", "hover"):
        result["raw_supporting_facts_distribution"] = {
            str(count): frequency
            for count, frequency in sorted(
                Counter(len(sample["supporting_facts"]) for sample in samples).items()
            )
        }
    if dataset_name == "2wikimultihopqa":
        result["support_count_capped_to_four"] = sum(
            len(sample["supporting_facts"]) > 4 for sample in samples
        )
    return result


def get_dataset_task(dataset_name: str) -> str:
    """HoVer claims require fact verification rather than an answer-span task."""
    return "fact_verification" if _dataset_name(dataset_name) == "hover" else "question_answering"


def get_query_text(sample: Mapping, dataset_name: str) -> str:
    dataset = _dataset_name(dataset_name)
    sample = _sample_mapping(sample, dataset)
    field = "claim" if dataset == "hover" else "question"
    query = sample.get(field)
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"{dataset} sample lacks a nonempty {field}")
    return query


def get_sample_id(sample: Mapping, dataset_name: str) -> str:
    dataset = _dataset_name(dataset_name)
    sample = _sample_mapping(sample, dataset)
    if dataset == "hover":
        identifier = sample.get("hover_id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("hover sample lacks a valid nonempty string hover_id")
        return identifier
    else:
        identifier = sample.get("id", sample.get("_id"))
    if not isinstance(identifier, (str, int)) or isinstance(identifier, bool) or not str(identifier).strip():
        raise ValueError(f"{dataset} sample lacks a valid id/_id")
    return str(identifier)


def get_classification_label(sample: Mapping) -> str:
    """Read HoVer's native target; never substitute its source Hotpot answer."""
    sample = _sample_mapping(sample, "hover")
    label = sample.get("label")
    if not isinstance(label, str) or label not in HOVER_LABELS:
        raise ValueError("HoVer label must be exactly SUPPORTED or NOT_SUPPORTED")
    return label


def render_context_document(title: str, sentences: Sequence[str], dataset_name: str) -> str:
    """Render context exactly as the benchmark's corpus serialization does.

    HoVer inherits Hotpot's sentence-leading spaces; inserting extra spaces
    changes passage identity and invalidates exact retrieval metrics.
    """
    dataset = _dataset_name(dataset_name)
    if not isinstance(title, str) or not title.strip():
        raise ValueError(f"{dataset} context title must be nonempty")
    if (
        not isinstance(sentences, (list, tuple))
        or not sentences
        or any(not isinstance(sentence, str) for sentence in sentences)
    ):
        raise ValueError(f"{dataset} context sentences must be a nonempty list of strings")
    separator = "" if dataset in ("hotpotqa", "hover") else " "
    return title + "\n" + separator.join(sentences)

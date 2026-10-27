"""Independent, CPU-only reports for completed retrieval/QA artifacts.

Neither result.json nor its validation receipt is changed. Gold labels are read
from the frozen benchmark identity, never from a method's predicted plan.
"""
from __future__ import annotations

from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import string
import tempfile

from .structural_metrics import evaluate_structural_metrics

REPORT_VERSION = "bindpack-metric-report-v1"
DEFAULT_KS = (5, 10, 20, 200)
CURRENT_CASES = ("full", "wo_route", "wo_plan", "wo_pack", "hippporag2_reproduce")
HOVER_LABELS = ("SUPPORTED", "NOT_SUPPORTED")


def _read(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()


def _sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _state(path):
    stat = Path(path).stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "inode": stat.st_ino, "device": stat.st_dev}


def _atomic_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def evaluator_fingerprint():
    """Reporting code has its own identity, independent of retrieval protocols."""
    directory = Path(__file__).resolve().parent
    paths = [Path(__file__).resolve(), directory / "structural_metrics.py"] + sorted(directory.glob("gold_*.py"))
    return {"version": REPORT_VERSION, "sha256": _digest(
        {path.name: _sha256(path) for path in paths})}


def _manifest_for(result_path, result, explicit):
    if explicit is not None:
        path = Path(explicit).resolve()
        if not path.is_file():
            raise ValueError(f"Missing frozen sample manifest: {path}")
        return path
    candidates = []
    if result.get("corpus_path"):
        candidates.append(Path(result["corpus_path"]).parent / "sample_manifest.json")
    case = result_path.parent
    if case.parent.parent.name == "cases":
        output = case.parents[2]
        candidates += sorted(output.glob(f"metadata/*/{result['dataset']}/sample_manifest.json"))
        candidates.append(output / "metadata" / result["dataset"] / "sample_manifest.json")
    matches = []
    for path in dict.fromkeys(candidates):
        if not path.is_file():
            continue
        identity = _read(path)
        if (identity.get("dataset") == result.get("dataset") and
                identity.get("selected_indices") == result.get("selected_indices")):
            matches.append((path.resolve(), identity))
    if not matches:
        raise ValueError("No matching frozen sample_manifest.json; provide manifest_path explicitly")
    if len({_digest(identity) for _, identity in matches}) > 1:
        raise ValueError("Multiple incompatible sample manifests; provide manifest_path explicitly")
    return matches[0][0]


def _answers(sample):
    if "answer" in sample or "gold_ans" in sample:
        value = sample.get("answer") or sample.get("gold_ans")
    elif "reference" in sample:
        value = sample["reference"]
    elif "obj" in sample:
        value = [sample["obj"]]
        for key in ("possible_answers", "o_aliases", "o_wiki_title"):
            extra = sample.get(key, [])
            if isinstance(extra, str) and extra.startswith("["):
                try:
                    extra = json.loads(extra)
                except ValueError:
                    pass
            value.extend(extra if isinstance(extra, list) else [extra])
    else:
        value = sample.get("answers", [])
    values = list(value) if isinstance(value, list) else [value]
    aliases = sample.get("answer_aliases", [])
    values += aliases if isinstance(aliases, list) else [aliases]
    return sorted({str(answer) for answer in values if answer is not None and str(answer)})


def _normalize(answer):
    value = "".join(char for char in answer.lower() if char not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", value).split())


def normalize_hover_prediction(prediction):
    """Accept one complete native label, never infer a label from prose."""
    if prediction is None:
        return None
    if not isinstance(prediction, str):
        raise ValueError("HoVer prediction must be a string")
    label = prediction.strip().upper()
    return label if label in HOVER_LABELS else None


def parse_hover_reader_response(response):
    """Decode one native label or the reader's exact JSON transport schema.

    Transport decoding is separate from label scoring: exported predictions
    remain atomic labels. Prose, extra object keys, lists, and invented label
    aliases are not searched or rescued into a classification.
    """
    text = response if isinstance(response, str) else str(response)
    text = text.strip()
    label = normalize_hover_prediction(text)
    if label is not None:
        return label
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return text
    if isinstance(value, dict) and set(value) == {"answer"} and isinstance(value["answer"], str):
        label = normalize_hover_prediction(value["answer"])
        if label is not None:
            return label
    return text


def score_hover_classification(predictions, gold_labels):
    """Score native HoVer labels, counting invalid outputs as failures.

    ``ExactMatch`` and ``F1`` are legacy answer-metric fields for atomic label
    agreement, both equal to accuracy. ``ClassificationMacroF1`` is the actual
    unweighted F1 over the two native classes, including false negatives caused
    by invalid outputs. Gold labels must retain their native exact spelling.
    This CPU-only helper is shared by online exports and offline reports.
    """
    if len(predictions) != len(gold_labels):
        raise ValueError("HoVer predictions and labels must have equal length")
    confusion = {label: {predicted: 0 for predicted in (*HOVER_LABELS, "INVALID")}
                 for label in HOVER_LABELS}
    details = []
    for prediction, gold in zip(predictions, gold_labels):
        if not isinstance(gold, str) or gold not in HOVER_LABELS:
            raise ValueError(f"Invalid native HoVer label: {gold!r}")
        predicted = normalize_hover_prediction(prediction)
        confusion[gold][predicted or "INVALID"] += 1
        correct = float(predicted == gold)
        details.append({"valid": True, "reason": None, "ExactMatch": correct, "F1": correct,
                        "gold_label": gold, "predicted_label": predicted,
                        "prediction_valid": predicted is not None,
                        "empty_prediction": prediction is None or not prediction.strip()})
    count = len(details)
    accuracy = sum(item["ExactMatch"] for item in details) / count if count else None
    per_label = {}
    for label in HOVER_LABELS:
        true_positive = confusion[label][label]
        false_positive = sum(confusion[gold][label] for gold in HOVER_LABELS if gold != label)
        false_negative = sum(value for predicted, value in confusion[label].items() if predicted != label)
        denominator = 2 * true_positive + false_positive + false_negative
        per_label[label] = {
            "support": sum(confusion[label].values()), "true_positive": true_positive,
            "false_positive": false_positive, "false_negative": false_negative,
            "precision": (true_positive / (true_positive + false_positive)
                          if true_positive + false_positive else (0.0 if count else None)),
            "recall": (true_positive / (true_positive + false_negative)
                       if true_positive + false_negative else (0.0 if count else None)),
            "F1": 2 * true_positive / denominator if denominator else (0.0 if count else None),
        }
    metrics = {"ExactMatch": accuracy, "F1": accuracy, "ClassificationAccuracy": accuracy,
               "ClassificationMacroF1": (sum(item["F1"] for item in per_label.values()) / len(HOVER_LABELS)
                                         if count else None)}
    coverage = {
        "qa_n": count, "qa_unavailable_reasons": {},
        "empty_prediction_count": sum(item["empty_prediction"] for item in details),
        "classification": {"task": "binary_claim_verification", "labels": list(HOVER_LABELS),
            "n": count, "invalid_prediction_count": sum(not item["prediction_valid"] for item in details),
            "confusion_matrix": confusion, "confusion_orientation": "gold_rows_predicted_columns",
            "per_label": per_label, "zero_division": 0},
    }
    return metrics, coverage, details


def _hover_qa(rows, samples, result):
    labels = [sample.get("label") for sample in samples]
    for row, label in zip(rows, labels):
        if not isinstance(label, str) or label not in HOVER_LABELS:
            raise ValueError(f"Invalid native HoVer label: {label!r}")
        exported = row.get("gold_answers")
        if exported not in (None, []) and exported != [label]:
            raise ValueError(f"Exported answer labels differ from benchmark: {row.get('sample_id')}")
    if result.get("eval_mode") == "rag_qa":
        return score_hover_classification([row.get("answer") for row in rows], labels)
    metrics, coverage, _ = score_hover_classification([], [])
    coverage["qa_unavailable_reasons"] = {"answer_generation_not_run": len(rows)}
    details = [{"valid": False, "reason": "answer_generation_not_run", "ExactMatch": None, "F1": None}
               for _ in rows]
    return metrics, coverage, details


def _sample_identity(sample, index, dataset_name):
    if dataset_name == "hover":
        sample_id = sample.get("hover_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError("HoVer sample lacks a valid native hover_id")
        return sample_id
    return sample.get("id", sample.get("_id", index))


def _sample_question(sample, dataset_name):
    if dataset_name == "hover":
        claim = sample.get("claim")
        if not isinstance(claim, str) or not claim.strip():
            raise ValueError("HoVer sample lacks a valid claim")
        return claim
    return sample.get("question", sample.get("query"))


def _qa(rows, samples, result):
    if result.get("dataset") == "hover":
        return _hover_qa(rows, samples, result)
    details, scores, reasons = [], [], Counter()
    qa_run = result.get("eval_mode") == "rag_qa"
    for row, sample in zip(rows, samples):
        aliases = _answers(sample)
        reason = None if aliases else "missing_answer_labels"
        if not qa_run:
            reason = "answer_generation_not_run"
        if reason:
            reasons[reason] += 1
            details.append({"valid": False, "reason": reason, "ExactMatch": None, "F1": None})
            continue
        exported = row.get("gold_answers")
        if exported and set(map(str, exported)) != set(aliases):
            raise ValueError(f"Exported answer labels differ from benchmark: {row.get('sample_id')}")
        prediction = row.get("answer") or ""
        if not isinstance(prediction, str):
            raise ValueError(f"Prediction must be a string: {row.get('sample_id')}")
        normalized = _normalize(prediction)
        em = max(float(normalized == _normalize(alias)) for alias in aliases)
        f1 = 0.0
        for alias in aliases:
            gold_tokens, predicted_tokens = _normalize(alias).split(), normalized.split()
            common = sum((Counter(gold_tokens) & Counter(predicted_tokens)).values())
            if common:
                f1 = max(f1, 2.0 * common / (len(gold_tokens) + len(predicted_tokens)))
        score = {"valid": True, "reason": None, "ExactMatch": em, "F1": f1,
                 "empty_prediction": not bool(prediction.strip())}
        details.append(score)
        scores.append(score)
    n = len(scores)
    metrics = {key: sum(score[key] for score in scores) / n if n else None
               for key in ("ExactMatch", "F1")}
    return metrics, {"qa_n": n, "qa_unavailable_reasons": dict(reasons),
        "empty_prediction_count": sum(score["empty_prediction"] for score in scores)}, details


def _standalone_inputs(result, data_path):
    """Validate standalone exports, disclosing that no prior frozen SHA exists."""
    data_path = Path(data_path or result.get("data_path", "")).resolve()
    if not data_path.is_file():
        raise ValueError("Standalone reporting requires an available data_path")
    data, rows = _read(data_path), result.get("results")
    if not isinstance(data, list) or not data:
        raise ValueError("Standalone benchmark must be a non-empty sample list")
    indices = result.get("selected_indices")
    if (not isinstance(rows, list) or not rows or not isinstance(indices, list)
            or len(indices) != len(rows) or len(set(indices)) != len(indices)
            or any(type(index) is not int or not 0 <= index < len(data) for index in indices)):
        raise ValueError("Standalone export has an invalid selected_indices sequence")
    samples = [data[index] for index in indices]
    for row, sample, index in zip(rows, samples, indices):
        if (row.get("query_index") != index or
                row.get("sample_id") != _sample_identity(sample, index, result.get("dataset")) or
                row.get("question") != _sample_question(sample, result.get("dataset"))):
            raise ValueError("Standalone result IDs/order/questions differ from benchmark")
    paths, docs = {"benchmark": data_path}, None
    if result.get("corpus_path") and Path(result["corpus_path"]).is_file():
        path = Path(result["corpus_path"]).resolve()
        corpus = _read(path)
        docs = [item if isinstance(item, str) else item.get("title", "") + "\n" +
                (item.get("text") or item.get("paragraph_text", "")) for item in corpus]
        paths["corpus"] = path
    inputs = {name: {"path": str(path), "sha256": _sha256(path), "state": _state(path)}
              for name, path in paths.items()}
    return rows, samples, docs, inputs


def _load_frozen_inputs(result_path, result, data_path, manifest_path, allow_standalone=False):
    try:
        manifest_path = _manifest_for(result_path, result, manifest_path)
    except ValueError as exc:
        if allow_standalone and manifest_path is None and str(exc).startswith("No matching frozen"):
            return _standalone_inputs(result, data_path)
        raise
    identity = _read(manifest_path)
    if identity.get("dataset") != result.get("dataset"):
        raise ValueError("Result dataset differs from sample manifest")
    data_path = Path(data_path or identity["data_path"]).resolve()
    data_hash = _sha256(data_path)
    if data_hash != identity.get("data_sha256"):
        raise ValueError(f"Benchmark SHA-256 differs from frozen manifest: {data_path}")
    data = _read(data_path)
    if not isinstance(data, list) or not data:
        raise ValueError("Frozen benchmark must be a non-empty sample list")
    rows, indices = result.get("results"), identity.get("selected_indices")
    if (not isinstance(rows, list) or not rows or not isinstance(indices, list)
            or len(indices) != len(rows) or result.get("selected_indices") != indices
            or len(set(indices)) != len(indices)
            or any(type(index) is not int or not 0 <= index < len(data) for index in indices)):
        raise ValueError("Invalid or mismatched result/sample-manifest index sequence")
    samples = [data[index] for index in indices]
    sample_ids = [_sample_identity(sample, index, result.get("dataset"))
                  for sample, index in zip(samples, indices)]
    if (identity.get("sample_ids") != sample_ids or
            [row.get("sample_id") for row in rows] != sample_ids or
            [row.get("query_index") for row in rows] != indices):
        raise ValueError("Result sample IDs/order differ from frozen benchmark")
    if any(row.get("question") != _sample_question(sample, result.get("dataset"))
           for row, sample in zip(rows, samples)):
        raise ValueError("Result question text differs from frozen benchmark")
    corpus_path = manifest_path.parent / "corpus.json"
    if not corpus_path.is_file():
        raise ValueError(f"Missing frozen canonical corpus: {corpus_path}")
    corpus = _read(corpus_path)
    docs = [item if isinstance(item, str) else item.get("title", "") + "\n" +
            (item.get("text") or item.get("paragraph_text", "")) for item in corpus]
    if (_digest(docs) != identity.get("corpus_docs_sha256")
            or len(docs) != identity.get("corpus_docs")):
        raise ValueError("Frozen canonical corpus differs from sample manifest")
    paths = {"sample_manifest": manifest_path, "benchmark": data_path, "corpus": corpus_path}
    source = identity.get("corpus_source")
    if source and identity.get("corpus_source_sha256"):
        source_path = Path(source).resolve()
        if _sha256(source_path) != identity.get("corpus_source_sha256"):
            raise ValueError(f"Source corpus SHA-256 differs from frozen manifest: {source_path}")
        paths["source_corpus"] = source_path
    inputs = {name: {"path": str(path), "sha256": _sha256(path), "state": _state(path)}
              for name, path in paths.items()}
    return rows, samples, docs, inputs


def _cached_report(report_path, fingerprint, ks, result_path, require_validated):
    if not report_path.is_file():
        return None
    try:
        report = _read(report_path)
        if (report.get("evaluator") != fingerprint or report.get("ks") != list(ks)
                or (require_validated and not report.get("validated_result"))):
            return None
        paths = report.get("inputs", {})
        if not paths or any(_state(item["path"]) != item["state"] for item in paths.values()):
            return None
        if paths["result"]["path"] != str(result_path):
            return None
        details = report_path.with_name("metrics_details.json")
        if _sha256(details) != report["details"]["sha256"]:
            return None
        return report
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _write_metric_report_unlocked(result_path, *, data_path=None, manifest_path=None,
                                 ks=DEFAULT_KS, require_validated=True):
    """Write report.json and metrics_details.json, returning the aggregate report.

    ``require_validated=False`` is reserved for an evaluator's post-save hook.
    The standalone CLI always waits for a matching validated.ok receipt.
    """
    result_path = Path(result_path).resolve()
    if result_path.is_dir():
        result_path /= "result.json"
    ks = tuple(sorted(set(ks)))
    if not ks or any(type(k) is not int or k <= 0 for k in ks):
        raise ValueError("K values must be positive integers")
    marker_path = result_path.with_name("validated.ok")
    protocol_path = result_path.with_name("protocol.json")
    if require_validated and not marker_path.is_file():
        raise ValueError(f"Result is not validated yet: {result_path}")
    fingerprint = evaluator_fingerprint()
    report_path = result_path.with_name("report.json")
    # Explicit overrides must be evaluated rather than hidden by the cache.
    cached = None if data_path or manifest_path else _cached_report(
        report_path, fingerprint, ks, result_path, require_validated)
    if cached is not None:
        return cached
    initial_state, result_hash = _state(result_path), _sha256(result_path)
    marker_state = None
    if marker_path.is_file():
        marker_state = _state(marker_path)
        marker = _read(marker_path)
        if marker.get("result_sha256") != result_hash:
            raise ValueError(f"Result SHA-256 differs from validated.ok: {result_path}")
        if protocol_path.is_file() and marker.get("protocol_sha256") != _digest(_read(protocol_path)):
            raise ValueError(f"Case protocol differs from validated.ok: {protocol_path}")
    result = _read(result_path)
    rows, samples, corpus, inputs = _load_frozen_inputs(
        result_path, result, data_path, manifest_path, allow_standalone=not require_validated)
    structural = evaluate_structural_metrics(rows, samples, result["dataset"], ks=ks,
                                            corpus_docs=corpus)
    qa_metrics, qa_coverage, qa_details = _qa(rows, samples, result)
    for query, qa in zip(structural["per_query"], qa_details):
        query["qa"] = qa
    metrics = {**qa_metrics, **structural["metrics"]}
    inputs["result"] = {"path": str(result_path), "sha256": result_hash, "state": initial_state}
    for name, path in (("validation", marker_path), ("case_protocol", protocol_path)):
        if path.is_file():
            inputs[name] = {"path": str(path), "sha256": _sha256(path), "state": _state(path)}
    details_path = result_path.with_name("metrics_details.json")
    details = {"schema_version": 1, "evaluator": fingerprint, "dataset": result["dataset"],
        "case": result_path.parent.name, "result_sha256": result_hash,
        "sample_manifest_sha256": inputs.get("sample_manifest", {}).get("sha256"),
        "per_query": structural["per_query"]}
    if (_state(result_path) != initial_state or
            (marker_state is not None and _state(marker_path) != marker_state)):
        raise ValueError("Result/validation marker changed while computing metrics; retry after completion")
    _atomic_write(details_path, details)
    report = {"schema_version": 1, "evaluator": fingerprint,
        "dataset": result["dataset"], "case": result_path.parent.name,
        "method": result.get("method", "bindpack"), "ks": list(ks),
        "validated_result": marker_state is not None,
        "inputs": inputs,
        "sample_alignment": {"verified": True, "sample_count": len(rows),
            "frozen_manifest_verified": "sample_manifest" in inputs,
            "provenance_note": ("benchmark/corpus match previously frozen manifest" if "sample_manifest" in inputs
                                else "standalone: current benchmark SHA recorded; no prior frozen manifest available"),
            "sample_ids_sha256": _digest([row["sample_id"] for row in rows])},
        "metrics": metrics,
        "metrics_percent": {key: None if value is None else 100 * value
                            for key, value in metrics.items()},
        "units": {"metrics": "fraction in [0,1]", "metrics_percent": "%"},
        "coverage": {**structural["coverage"], **qa_coverage},
        "dag_subset_document_metrics": structural.get("dag_subset_document_metrics", {}),
        "protocol": {**structural["protocol"],
            "aggregation": "per-query macro average over each metric's valid annotations",
            "qa": "MRQA lowercase/ASCII-punctuation/articles/whitespace normalization; max over aliases",
            "reader_top_k": result.get("runtime_config", {}).get("qa_top_k"),
            "joint_success": "not reported: one reader context cannot stand for all K"},
        "original_exported_metrics": {"retrieval": result.get("retrieval_metrics", {}),
                                      "qa": result.get("qa_metrics", {})},
        "details": {"path": details_path.name, "sha256": _sha256(details_path)}}
    if result["dataset"] == "hover":
        report["protocol"].update({
            "answer_task": "binary_claim_verification",
            "qa": "native SUPPORTED/NOT_SUPPORTED labels; trim/case-normalized whole-label predictions only; "
                  "invalid outputs are failures; legacy ExactMatch/F1 mean atomic-label agreement",
            "aggregation": "structural metrics and label accuracy: query macro average; "
                           "ClassificationMacroF1: unweighted mean of both native class F1 scores",
            "classification_gold": "native label only; hotpotqa_answer is not a label or alias",
            "classification_label_semantics": "NOT_SUPPORTED means not supported; it does not assert counterevidence",
        })
    _atomic_write(report_path, report)
    return report


def write_metric_report(result_path, *, data_path=None, manifest_path=None,
                        ks=DEFAULT_KS, require_validated=True):
    """Serialize concurrent hooks/watchers; never modify retrieval artifacts."""
    path = Path(result_path).resolve()
    if path.is_dir():
        path /= "result.json"
    if not path.is_file():
        raise ValueError(f"Missing completed result: {path}")
    if require_validated and not path.with_name("validated.ok").is_file():
        raise ValueError(f"Result is not validated yet: {path}")
    with path.with_name(".metrics_report.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _write_metric_report_unlocked(path, data_path=data_path, manifest_path=manifest_path,
                                            ks=ks, require_validated=require_validated)


def discover_cases(cases_root, case_names=CURRENT_CASES):
    """Find current experiment directories, leaving historical outputs alone."""
    root, names = Path(cases_root).resolve(), set(case_names)
    if not root.is_dir():
        raise ValueError(f"Missing cases root: {root}")
    return sorted(path for path in root.glob("*/*") if path.is_dir() and path.name in names)

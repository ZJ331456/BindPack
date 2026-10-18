"""Allow one explicitly fingerprinted reporting update on the frozen 2026-10-18 run.

The experiment protocol remains immutable.  A separate receipt records the exact
evaluation-only source update, so retrieval/configuration drift still fails closed.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


ROOT = Path(__file__).resolve().parents[2]
BASE_COMMIT = "789ecaa4e8e1fcb69bdd9aca8dde0b6e3b9fc87e"
BASE_SOURCE_COUNT = 135
BASE_CODE_SHA256 = "22d90568a97e181bb612dd8ca7032d1f727da3c092562ada5a0266a2d46be5f4"
CHANGED_FILES = frozenset(("scripts/eval_dataset.py", "scripts/utils/stage_ablations.py"))
ADDED_FILES = frozenset((
    "src/bindpack/evaluation/gold_graph.py",
    "src/bindpack/evaluation/structural_metrics.py",
    "src/bindpack/evaluation/metric_report.py",
    "scripts/report_metrics.py",
    "scripts/utils/reporting_protocol.py",
))
KIND = "evaluation_reporting_only"
RECEIPT_NAME = "reporting_protocol.json"


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _object(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _source_map(value):
    if not isinstance(value, dict) or not value:
        raise ValueError("Missing source fingerprints")
    for name, sha in value.items():
        path = Path(name) if isinstance(name, str) else None
        if (path is None or path.is_absolute() or ".." in path.parts
                or str(path) != name or not isinstance(sha, str)
                or not re.fullmatch(r"[0-9a-f]{64}", sha)):
            raise ValueError("Invalid source fingerprint path or SHA256")
    return value


def resolve_existing_protocol(protocol_path, current, *, root=ROOT, manifest_path=None):
    """Return ``(selected_protocol, reporting_receipt_or_None)`` without writes.

    Ordinary identical protocols need no migration manifest.  For the one allowed
    reporting update, every non-code field and every protected source is exact.
    Call :func:`persist_reporting_receipt` only after acquiring the experiment lock;
    never rewrite an existing experiment protocol with ``current``.
    """
    protocol_path, root = Path(protocol_path), Path(root)
    receipt_path = protocol_path.with_name(RECEIPT_NAME)
    if not protocol_path.exists():
        if receipt_path.exists():
            raise ValueError("Reporting receipt has no original experiment protocol")
        return current, None
    original = _object(protocol_path)
    if original == current:
        if receipt_path.exists():
            raise ValueError("Reporting receipt exists without its reporting source update")
        return original, None
    original_payload = {key: value for key, value in original.items() if key != "code_sha256"}
    current_payload = {key: value for key, value in current.items() if key != "code_sha256"}
    if original_payload != current_payload:
        raise ValueError("Existing inputs/configuration differ from the frozen experiment")
    before = _source_map(original.get("code_sha256"))
    after = _source_map(current.get("code_sha256"))
    if len(before) != BASE_SOURCE_COUNT or _digest(before) != BASE_CODE_SHA256:
        raise ValueError("Reporting migration is only valid for the exact frozen base source")
    manifest_path = Path(manifest_path or root / "configs/metrics_reporting_update.json")
    manifest = _object(manifest_path)
    if (set(manifest) != {"schema_version", "kind", "base_commit", "base_code_sha256", "changes", "added"}
            or manifest.get("schema_version") != 1 or manifest.get("kind") != KIND
            or manifest.get("base_commit") != BASE_COMMIT
            or manifest.get("base_code_sha256") != BASE_CODE_SHA256):
        raise ValueError("Unknown reporting migration or frozen base")
    changes, added = manifest.get("changes"), manifest.get("added")
    if (not isinstance(changes, dict) or set(changes) != CHANGED_FILES
            or not isinstance(added, dict) or set(added) != ADDED_FILES):
        raise ValueError("Reporting migration must name exactly the reviewed reporting files")
    if not CHANGED_FILES.issubset(before) or ADDED_FILES.intersection(before):
        raise ValueError("Reporting migration base source layout differs")
    if set(after) != set(before) | ADDED_FILES:
        raise ValueError("Source additions or removals exceed the reporting migration")
    for name in before:
        if name in changes:
            change = changes[name]
            if (not isinstance(change, dict) or set(change) != {"before", "after"}
                    or change.get("before") != before[name] or change.get("after") != after[name]
                    or before[name] == after[name]):
                raise ValueError(f"Reporting source migration fingerprint mismatch: {name}")
        elif before[name] != after[name]:
            raise ValueError(f"Protected experiment source changed: {name}")
    for name in ADDED_FILES:
        if added[name] != after[name]:
            raise ValueError(f"Added reporting source fingerprint mismatch: {name}")
    for name, sha in after.items():
        source = root / name
        if source.is_symlink() or not source.is_file() or _sha256(source) != sha:
            raise ValueError(f"Current source does not match reporting protocol: {name}")
    receipt = {
        "schema_version": 1, "kind": KIND, "base_commit": BASE_COMMIT,
        "original_protocol_file_sha256": _sha256(protocol_path),
        "original_protocol_sha256": _digest(original),
        "migration_manifest_sha256": _sha256(manifest_path),
        "current_code_sha256": after,
    }
    if receipt_path.exists() and _object(receipt_path) != receipt:
        raise ValueError("Existing reporting receipt or migration sources changed")
    return original, receipt


def persist_reporting_receipt(protocol_path, receipt):
    """Create the separate receipt after locking; preserve existing bytes exactly.

    ``protocol_path`` names the original experiment protocol, not the receipt.
    This function never modifies that protocol and never replaces another receipt.
    """
    if receipt is None:
        return
    protocol_path = Path(protocol_path)
    if (not protocol_path.is_file()
            or _sha256(protocol_path) != receipt.get("original_protocol_file_sha256")):
        raise ValueError("Original experiment protocol changed before recording reporting receipt")
    receipt_path = protocol_path.with_name(RECEIPT_NAME)
    if receipt_path.exists():
        if _object(receipt_path) != receipt:
            raise ValueError("Refusing to replace a different reporting receipt")
        return
    # Atomic creation without overwriting a concurrent writer's receipt.
    payload = (json.dumps(receipt, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=receipt_path.parent, prefix=".reporting_protocol_", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, receipt_path)
        except FileExistsError:
            if _object(receipt_path) != receipt:
                raise ValueError("Refusing to replace a different reporting receipt")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

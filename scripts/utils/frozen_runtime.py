"""Pin already running evaluations to their registered, immutable source tree.

The launcher only applies to exact invocations recorded before a source change.
New output directories always use the current implementation.  This does not
migrate experiment protocols or allow a changed method to resume an old batch.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sys


_INPUT_OPTIONS = frozenset(("data_path", "corpus_path", "sample_indices_file", "method_config"))
_REQUIRED_SOURCES = frozenset(("scripts/eval_dataset.py", "scripts/eval_utils.py",
                               "src/bindpack/__init__.py"))
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _read_object(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Frozen runtime requires a JSON object: {path}")
    return value


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _options(argv):
    """Parse explicit long options, rejecting ambiguous or repeated arguments."""
    result = {}
    number = 0
    while number < len(argv):
        token = argv[number]
        if not isinstance(token, str) or not token.startswith("--") or token == "--":
            raise ValueError("Frozen evaluation requires explicit long options")
        key, separator, value = token[2:].partition("=")
        if not key or key in result:
            raise ValueError(f"Repeated or invalid frozen evaluation option: {key}")
        if not separator:
            if number + 1 < len(argv) and not argv[number + 1].startswith("--"):
                number += 1
                value = argv[number]
            else:
                value = True
        result[key] = value
        number += 1
    return result


def _within(path, root):
    return path == root or root in path.parents


def _path(value, label):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f"Frozen runtime {label} must be an absolute path")
    return Path(value).resolve()


def _fingerprint(value, label):
    if (not isinstance(value, dict) or set(value) != {"path", "sha256"}
            or not isinstance(value.get("sha256"), str) or not _SHA256.fullmatch(value["sha256"])):
        raise ValueError(f"Invalid frozen runtime fingerprint: {label}")
    raw_path = Path(value["path"])
    path = _path(value["path"], label)
    if raw_path.is_symlink() or not path.is_file() or _sha256(path) != value["sha256"]:
        raise ValueError(f"Frozen runtime input changed: {label}")
    return path


def _verify_manifest(batch, project_root):
    runtime = _path(batch.get("runtime_root"), "runtime_root")
    snapshots = (project_root / "outputs/runtime_snapshots").resolve()
    if runtime == snapshots or not _within(runtime, snapshots):
        raise ValueError("Frozen runtime must be inside the runtime snapshot directory")
    manifest_path = _fingerprint(batch.get("manifest"), "manifest")
    if not _within(manifest_path, runtime):
        raise ValueError("Frozen runtime manifest is outside its snapshot")
    manifest = _read_object(manifest_path)
    files = manifest.get("files")
    if manifest.get("schema_version") != 1 or not isinstance(files, dict) or not files:
        raise ValueError("Invalid frozen runtime manifest")
    if not _REQUIRED_SOURCES.issubset(files):
        raise ValueError("Frozen runtime manifest omits evaluation sources")
    for relative, sha in files.items():
        name = Path(relative) if isinstance(relative, str) else None
        if (name is None or name.is_absolute() or ".." in name.parts or str(name) != relative
                or not isinstance(sha, str) or not _SHA256.fullmatch(sha)):
            raise ValueError("Invalid frozen runtime source fingerprint")
        source = runtime / name
        if (any((runtime / Path(*name.parts[:index])).is_symlink()
                for index in range(1, len(name.parts) + 1))
                or not source.is_file() or _sha256(source) != sha):
            raise ValueError(f"Frozen runtime source changed: {relative}")
    return runtime


def frozen_evaluation_command(argv, *, project_root, executable=None, registry_path=None):
    """Return an exact registered evaluation command, or ``None`` for a new run.

    Any invocation targeting a registered output root must match a frozen command
    and all source/input fingerprints.  Failure raises before retrieval imports.
    """
    project_root = Path(project_root).resolve()
    registry_path = Path(registry_path or project_root / "outputs/runtime_snapshots/registry.json")
    if not registry_path.exists():
        return None
    # Inspect only --output here: new CLI options need not be understood by the
    # old-run guard, and malformed unrelated commands belong to their parser.
    output_values = []
    for number, token in enumerate(argv):
        if token == "--output" and number + 1 < len(argv):
            output_values.append(argv[number + 1])
        elif token.startswith("--output="):
            output_values.append(token.partition("=")[2])
    if not output_values:
        return None
    registry = _read_object(registry_path)
    batches = registry.get("batches")
    if registry.get("schema_version") != 1 or not isinstance(batches, list):
        raise ValueError("Invalid frozen runtime registry")
    matching = []
    for batch in batches:
        if not isinstance(batch, dict):
            raise ValueError("Invalid frozen runtime batch registration")
        output_root = _path(batch.get("output_root"), "output_root")
        if any(_within(Path(value).resolve(), output_root) for value in output_values):
            matching.append(batch)
    if not matching:
        return None
    if len(matching) != 1:
        raise ValueError("Ambiguous frozen runtime batch registration")
    batch = matching[0]
    options = _options(list(argv))
    commands = batch.get("invocations")
    if not isinstance(commands, list) or not commands:
        raise ValueError("Frozen runtime has no registered evaluation invocations")
    selected = []
    for invocation in commands:
        if not isinstance(invocation, dict) or not isinstance(invocation.get("argv"), list):
            raise ValueError("Invalid frozen runtime invocation")
        if _options(invocation["argv"]) == options:
            selected.append(invocation)
    if len(selected) != 1:
        raise ValueError("Evaluation differs from the registered frozen invocation; use a new output root")
    invocation = selected[0]
    inputs = invocation.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != _INPUT_OPTIONS:
        raise ValueError("Frozen evaluation input fingerprints are incomplete")
    output_root = _path(batch["output_root"], "output_root")
    dataset, stage = options.get("dataset"), options.get("stage_ablation")
    if (not isinstance(dataset, str) or not dataset or Path(dataset).name != dataset
            or stage not in ("full", "wo_route", "wo_plan", "wo_pack")
            or options.get("reuse_index") is not True):
        raise ValueError("Invalid frozen evaluation dataset, stage or index policy")
    case = output_root / "cases" / dataset / stage
    if (Path(str(options.get("output"))).resolve() != case / "result.json"
            or Path(str(options.get("save_dir"))).resolve() != case / "index"):
        raise ValueError("Frozen evaluation case output paths differ")
    protocol = _fingerprint(batch.get("protocol"), "protocol")
    if not _within(protocol, output_root / "metadata"):
        raise ValueError("Frozen runtime protocol is outside its batch metadata")
    for name in sorted(_INPUT_OPTIONS):
        path = _fingerprint(inputs[name], name)
        if not isinstance(options.get(name), str) or Path(options[name]).resolve() != path:
            raise ValueError(f"Frozen evaluation argument differs from its input: {name}")
    runtime = _verify_manifest(batch, project_root)
    return [str(executable or sys.executable), "-B", "-u", str(runtime / "scripts/eval_dataset.py"), *argv]


def maybe_exec_frozen_evaluation(argv, project_root):
    """Replace only a registered old evaluation process with its frozen entry."""
    command = frozen_evaluation_command(argv, project_root=project_root)
    if command is None:
        return False
    env = os.environ.copy()
    root = Path(project_root).resolve()
    live_paths = {root / "src", root / "scripts", root}
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in env["PYTHONPATH"].split(os.pathsep)
            if part and Path(part).resolve() not in live_paths)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    print(f"Frozen evaluation runtime: {command[3]}", file=sys.stderr, flush=True)
    os.execve(command[0], command, env)
    return True  # os.execve returns only when mocked in tests.

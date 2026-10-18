#!/usr/bin/env python3
"""Backfill/watch CPU-only reports without interrupting retrieval experiments."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]


def _load_report_module():
    # Loading bindpack.__init__ would also load the GPU retrieval stack. Create
    # private namespaces instead; regular package users use the normal import.
    namespace = "_bindpack_offline_metrics"
    package = types.ModuleType(namespace)
    package.__path__ = [str(ROOT / "src/bindpack/evaluation")]
    sys.modules.setdefault(namespace, package)
    name = namespace + ".metric_report"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name,
            ROOT / "src/bindpack/evaluation/metric_report.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def write_metric_report(*args, **kwargs):
    """Lightweight integration entry point; does not import the retrieval stack."""
    return _load_report_module().write_metric_report(*args, **kwargs)


def _alive(pid):
    try:
        os.kill(pid, 0)
        # An unreaped child has exited even though kill(pid, 0) still succeeds.
        stat = Path(f"/proc/{pid}/stat")
        return not stat.exists() or stat.read_text().split(")", 1)[1].strip().split()[0] != "Z"
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--case", action="append", help="completed case directory (repeatable)")
    source.add_argument("--cases-root", help="cases root containing dataset/case directories")
    parser.add_argument("--case-names", nargs="+", default=None,
                        help="case names to include; default is the current four modes plus HippoRAG")
    parser.add_argument("--data-path", help="benchmark override (single --case only; SHA must match)")
    parser.add_argument("--manifest", help="frozen sample manifest (single --case only)")
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20, 200])
    parser.add_argument("--watch", action="store_true", help="report each case after validated.ok appears")
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--runner-pid", type=int, help="stop after this runner exits and one final scan")
    parser.add_argument("--batch-dir", help="stop after this batch's completed.ok appears and final scan")
    args = parser.parse_args(argv)
    if args.interval <= 0 or (args.runner_pid is not None and args.runner_pid <= 0):
        parser.error("interval and runner PID must be positive")
    if (args.data_path or args.manifest) and (not args.case or len(args.case) != 1):
        parser.error("--data-path/--manifest require exactly one --case")
    module = _load_report_module()
    seen, errors = {}, {}
    while True:
        finished = ((args.runner_pid is not None and not _alive(args.runner_pid)) or
                    (args.batch_dir and (Path(args.batch_dir) / "completed.ok").is_file()))
        cases = ([Path(path).resolve() for path in args.case] if args.case else
                 module.discover_cases(args.cases_root, args.case_names or module.CURRENT_CASES))
        for case in cases:
            if not (case / "validated.ok").is_file():
                if not args.watch:
                    print(f"SKIP unvalidated {case}", flush=True)
                continue
            try:
                report = module.write_metric_report(case / "result.json", data_path=args.data_path,
                    manifest_path=args.manifest, ks=args.ks)
                error_path = case / "report_error.json"
                if error_path.is_file():
                    try:
                        previous_error = module._read(error_path)
                        if (isinstance(previous_error, dict) and previous_error.get("postprocessing_only") is True and
                                Path(previous_error.get("result", "")).resolve() == case / "result.json"):
                            error_path.unlink(missing_ok=True)
                    except (OSError, ValueError, TypeError) as exc:
                        print(f"WARNING could not clear report error receipt {error_path}: {exc}",
                              file=sys.stderr, flush=True)
                signature = (report["inputs"]["result"]["sha256"], report["evaluator"]["sha256"])
                if seen.get(case) != signature:
                    print(json.dumps({"case": str(case), "report": str(case / "report.json"),
                        "metrics_percent": report["metrics_percent"], "coverage": report["coverage"]},
                        ensure_ascii=False), flush=True)
                seen[case] = signature
                errors.pop(case, None)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                message = f"ERROR {case}: {exc}"
                if errors.get(case) != message:
                    print(message, file=sys.stderr, flush=True)
                errors[case] = message
                try:
                    module._atomic_write(case / "report_error.json", {"status": "error",
                        "postprocessing_only": True, "result": str(case / "result.json"), "error": str(exc)})
                except OSError as receipt_error:
                    print(f"ERROR could not persist report error receipt {case}: {receipt_error}",
                          file=sys.stderr, flush=True)
        if not args.watch or finished:
            return 1 if errors else 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())

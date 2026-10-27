import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


HELPER = Path(__file__).resolve().parents[1] / "scripts/utils/frozen_runtime.py"
SPEC = importlib.util.spec_from_file_location("bindpack_test_frozen_runtime", HELPER)
frozen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(frozen)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class FrozenRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "project"
        self.output = self.root / "outputs/old_batch"
        self.runtime = self.root / "outputs/runtime_snapshots/frozen_commit"
        self.registry_path = self.root / "outputs/runtime_snapshots/registry.json"
        self.protocol = self.output / "metadata/batch/protocol.json"
        write(self.protocol, {"original": True})
        self.sources = {
            "scripts/eval_dataset.py": (
                "import json, sys\nfrom pathlib import Path\n"
                "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
                "import bindpack\nfrom eval_utils import HOP\n"
                "print(json.dumps({'hop': HOP, 'package': bindpack.VALUE, "
                "'package_file': bindpack.__file__}))\n"),
            "scripts/eval_utils.py": "HOP = 2\n",
            "src/bindpack/__init__.py": "VALUE = 'frozen'\n",
        }
        for relative, text in self.sources.items():
            path = self.runtime / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        self.manifest = self.runtime / "snapshot_manifest.json"
        write(self.manifest, {"schema_version": 1, "commit": "frozen_commit", "files": {
            relative: sha(self.runtime / relative) for relative in self.sources}})
        inputs = {}
        for name in ("data_path", "corpus_path", "sample_indices_file", "method_config"):
            path = self.output / "metadata/batch" / (name + ".json")
            write(path, {"immutable_input": name})
            inputs[name] = {"path": str(path), "sha256": sha(path)}
        case = self.output / "cases/musique/wo_plan"
        self.argv = ["--dataset", "musique", "--stage_ablation", "wo_plan"]
        for name, value in inputs.items():
            self.argv.extend(["--" + name, value["path"]])
        self.argv.extend(["--sample_seed", "42", "--save_dir", str(case / "index"),
                          "--output", str(case / "result.json"), "--no_ensure_gold_in_corpus", "--reuse_index"])
        self.batch = {
            "output_root": str(self.output),
            "protocol": {"path": str(self.protocol), "sha256": sha(self.protocol)},
            "runtime_root": str(self.runtime),
            "manifest": {"path": str(self.manifest), "sha256": sha(self.manifest)},
            "invocations": [{"argv": self.argv.copy(), "inputs": inputs}],
        }
        self.registry = {"schema_version": 1, "batches": [self.batch]}
        self._save_registry()

    def _save_registry(self):
        write(self.registry_path, self.registry)

    def _command(self, argv=None):
        return frozen.frozen_evaluation_command(self.argv if argv is None else argv, project_root=self.root)

    def _change_manifest(self, edit):
        value = json.loads(self.manifest.read_text())
        edit(value)
        write(self.manifest, value)
        self.batch["manifest"]["sha256"] = sha(self.manifest)
        self._save_registry()

    def test_exact_registered_invocation_uses_frozen_entry(self):
        command = self._command()
        self.assertEqual(command[:4], [sys.executable, "-B", "-u", str(self.runtime / "scripts/eval_dataset.py")])
        self.assertEqual(command[4:], self.argv)

    def test_new_outputs_use_current_code_even_when_old_input_is_missing(self):
        Path(self.batch["invocations"][0]["inputs"]["data_path"]["path"]).unlink()
        argv = self.argv.copy()
        argv[argv.index("--output") + 1] = str(self.root / "outputs/new_batch/result.json")
        argv.extend(["--new_rule", "new_value"])
        self.assertIsNone(self._command(argv))

    def test_missing_registry_or_missing_output_uses_current_code(self):
        self.assertIsNone(self._command(["--dataset", "hover"]))
        self.registry_path.unlink()
        self.assertIsNone(self._command())

    def test_equal_option_syntax_and_order_are_accepted(self):
        argv = ["--reuse_index", "--no_ensure_gold_in_corpus"]
        options = frozen._options(self.argv)
        for key, value in reversed(list(options.items())):
            if value is not True:
                argv.append(f"--{key}={value}")
        self.assertIsNotNone(self._command(argv))

    def test_changed_stage_or_additional_option_fails_closed(self):
        for argv in (self.argv + ["--new_rule", "enabled"],
                     ["full" if token == "wo_plan" else token for token in self.argv]):
            with self.subTest(argv=argv):
                with self.assertRaisesRegex(ValueError, "registered frozen invocation"):
                    self._command(argv)

    def test_duplicate_output_cannot_bypass_guard(self):
        argv = self.argv + ["--output", str(self.root / "outputs/new_batch/result.json")]
        with self.assertRaisesRegex(ValueError, "Repeated"):
            self._command(argv)

    def test_changed_data_and_method_and_protocol_are_rejected(self):
        for label, fingerprint in [("protocol", self.batch["protocol"]), *self.batch["invocations"][0]["inputs"].items()]:
            path = Path(fingerprint["path"])
            before = path.read_bytes()
            path.write_text("changed", encoding="utf-8")
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, "input changed"):
                    self._command()
            path.write_bytes(before)

    def test_changed_snapshot_source_is_rejected(self):
        (self.runtime / "scripts/eval_utils.py").write_text("HOP = 4\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source changed"):
            self._command()

    def test_changed_manifest_is_rejected(self):
        self.manifest.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "input changed: manifest"):
            self._command()

    def test_missing_required_snapshot_source_is_rejected(self):
        self._change_manifest(lambda value: value["files"].pop("scripts/eval_utils.py"))
        with self.assertRaisesRegex(ValueError, "omits evaluation sources"):
            self._command()

    def test_manifest_traversal_is_rejected(self):
        self._change_manifest(lambda value: value["files"].update({"../outside.py": "0" * 64}))
        with self.assertRaisesRegex(ValueError, "source fingerprint"):
            self._command()

    def test_source_symlinks_are_rejected(self):
        source = self.runtime / "scripts/eval_utils.py"
        target = self.runtime / "old_utils.py"
        source.rename(target)
        source.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "source changed"):
            self._command()

    def test_partial_input_fingerprints_are_rejected(self):
        self.batch["invocations"][0]["inputs"].pop("method_config")
        self._save_registry()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self._command()

    def test_case_path_policy_is_enforced_even_for_registered_arguments(self):
        number = self.argv.index("--save_dir") + 1
        self.argv[number] = str(self.output / "unexpected_index")
        self.batch["invocations"][0]["argv"] = self.argv.copy()
        self._save_registry()
        with self.assertRaisesRegex(ValueError, "case output paths differ"):
            self._command()

    def test_overlapping_batches_are_rejected(self):
        self.registry["batches"].append(self.batch)
        self._save_registry()
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            self._command()

    def test_exec_removes_live_paths_and_keeps_dependency_overlay(self):
        overlay = self.root / "runtime_dependencies"
        pythonpath = os.pathsep.join(map(str, (overlay, self.root / "src", self.root / "scripts", self.root)))
        with patch.dict(os.environ, {"PYTHONPATH": pythonpath}), patch.object(frozen.os, "execve") as execute:
            self.assertTrue(frozen.maybe_exec_frozen_evaluation(self.argv, self.root))
        executable, command, env = execute.call_args.args
        self.assertEqual(executable, sys.executable)
        self.assertEqual(command[3], str(self.runtime / "scripts/eval_dataset.py"))
        self.assertEqual(env["PYTHONPATH"], str(overlay))
        self.assertEqual(env["PYTHONDONTWRITEBYTECODE"], "1")

    def test_subprocess_imports_frozen_hop_rule_and_package(self):
        live_package = self.root / "src/bindpack/__init__.py"
        live_package.parent.mkdir(parents=True, exist_ok=True)
        live_package.write_text("VALUE = 'new'\n", encoding="utf-8")
        live_utils = self.root / "scripts/eval_utils.py"
        live_utils.parent.mkdir(parents=True, exist_ok=True)
        live_utils.write_text("HOP = 4\n", encoding="utf-8")
        launcher = (
            "import importlib.util\n"
            f"s = importlib.util.spec_from_file_location('runtime', {str(HELPER)!r})\n"
            "m = importlib.util.module_from_spec(s); s.loader.exec_module(m)\n"
            f"m.maybe_exec_frozen_evaluation({self.argv!r}, {str(self.root)!r})\n")
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join((str(self.root / "src"), str(self.root / "scripts")))
        result = subprocess.run([sys.executable, "-B", "-c", launcher], env=env, check=True,
                                capture_output=True, text=True)
        exported = json.loads(result.stdout)
        self.assertEqual(exported["hop"], 2)
        self.assertEqual(exported["package"], "frozen")
        self.assertEqual(Path(exported["package_file"]), self.runtime / "src/bindpack/__init__.py")


if __name__ == "__main__":
    unittest.main()

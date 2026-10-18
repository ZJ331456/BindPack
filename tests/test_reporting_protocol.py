"""An offline metric update must preserve the original experiment identity."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from utils import reporting_protocol as reporting


class ReportingProtocolTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.protocol_path = self.root / "run/protocol.json"
        self.manifest_path = self.root / "configs/metrics_reporting_update.json"
        before = {name: hashlib.sha256(("old " + name).encode()).hexdigest()
                  for name in reporting.CHANGED_FILES | {"src/bindpack/core/base.py"}}
        self.original = {"schema_version": 2, "config": {"model": "qwen3-8b"},
                         "identities": [{"dataset": "hotpotqa", "data_sha256": "frozen"}],
                         "code_sha256": before}
        self.current = copy.deepcopy(self.original)
        after = {}
        for name in set(before) | reporting.ADDED_FILES:
            data = (("old " if name not in reporting.CHANGED_FILES | reporting.ADDED_FILES else "new ") + name).encode()
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            after[name] = hashlib.sha256(data).hexdigest()
        self.current["code_sha256"] = after
        self.manifest = {
            "schema_version": 1, "kind": reporting.KIND, "base_commit": reporting.BASE_COMMIT,
            "base_code_sha256": reporting._digest(before),
            "changes": {name: {"before": before[name], "after": after[name]}
                        for name in reporting.CHANGED_FILES},
            "added": {name: after[name] for name in reporting.ADDED_FILES},
        }
        self.write(self.protocol_path, self.original)
        self.write(self.manifest_path, self.manifest)
        self.addCleanup(patch.stopall)
        patch.object(reporting, "BASE_CODE_SHA256", reporting._digest(before)).start()
        patch.object(reporting, "BASE_SOURCE_COUNT", len(before)).start()

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    def resolve(self, current=None):
        return reporting.resolve_existing_protocol(self.protocol_path, current or self.current, root=self.root)

    def test_ordinary_protocol_needs_no_manifest_and_does_not_write(self):
        self.manifest_path.unlink()
        selected, receipt = self.resolve(self.original)
        self.assertEqual(selected, self.original)
        self.assertIsNone(receipt)
        self.assertFalse(self.protocol_path.with_name(reporting.RECEIPT_NAME).exists())

    def test_new_experiment_needs_no_manifest(self):
        self.protocol_path.unlink()
        self.manifest_path.unlink()
        self.assertEqual(self.resolve(), (self.current, None))

    def test_exact_update_is_read_only_and_preserves_original_protocol(self):
        original_bytes = self.protocol_path.read_bytes()
        selected, receipt = self.resolve()
        self.assertEqual(selected, self.original)
        self.assertEqual(receipt["current_code_sha256"], self.current["code_sha256"])
        self.assertFalse(self.protocol_path.with_name(reporting.RECEIPT_NAME).exists())
        reporting.persist_reporting_receipt(self.protocol_path, receipt)
        self.assertEqual(self.protocol_path.read_bytes(), original_bytes)
        receipt_path = self.protocol_path.with_name(reporting.RECEIPT_NAME)
        receipt_bytes = receipt_path.read_bytes()
        self.assertEqual(self.resolve(), (selected, receipt))
        reporting.persist_reporting_receipt(self.protocol_path, receipt)
        self.assertEqual(receipt_path.read_bytes(), receipt_bytes)

    def test_unlisted_source_changes_are_rejected(self):
        current = copy.deepcopy(self.current)
        current["code_sha256"]["src/bindpack/core/base.py"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "Protected experiment source"):
            self.resolve(current)

    def test_arbitrary_added_source_is_rejected_even_when_declared(self):
        current = copy.deepcopy(self.current)
        current["code_sha256"]["src/bindpack/retrieval/new.py"] = "a" * 64
        self.manifest["added"]["src/bindpack/retrieval/new.py"] = "a" * 64
        self.write(self.manifest_path, self.manifest)
        with self.assertRaisesRegex(ValueError, "exactly the reviewed"):
            self.resolve(current)

    def test_missing_source_and_unknown_source_are_rejected(self):
        for name in ("scripts/eval_dataset.py", "src/bindpack/core/base.py"):
            current = copy.deepcopy(self.current)
            del current["code_sha256"][name]
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "additions or removals"):
                self.resolve(current)

    def test_configuration_and_data_drift_are_rejected(self):
        for key, value in (("config", {"model": "other"}),
                           ("identities", [{"dataset": "hotpotqa", "data_sha256": "changed"}])):
            current = copy.deepcopy(self.current)
            current[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "inputs/configuration"):
                self.resolve(current)

    def test_unknown_base_and_wrong_base_commit_are_rejected(self):
        self.manifest["base_commit"] = "0" * 40
        self.write(self.manifest_path, self.manifest)
        with self.assertRaisesRegex(ValueError, "Unknown reporting"):
            self.resolve()
        self.original["code_sha256"]["src/bindpack/core/base.py"] = "a" * 64
        self.write(self.protocol_path, self.original)
        with self.assertRaisesRegex(ValueError, "exact frozen base"):
            self.resolve()

    def test_wrong_source_hash_or_disk_contents_are_rejected(self):
        self.manifest["changes"]["scripts/eval_dataset.py"]["after"] = "a" * 64
        self.write(self.manifest_path, self.manifest)
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            self.resolve()
        self.manifest["changes"]["scripts/eval_dataset.py"]["after"] = self.current["code_sha256"]["scripts/eval_dataset.py"]
        self.write(self.manifest_path, self.manifest)
        (self.root / "scripts/eval_dataset.py").write_text("changed outside protocol")
        with self.assertRaisesRegex(ValueError, "Current source does not match"):
            self.resolve()

    def test_receipt_and_manifest_drift_are_rejected(self):
        _, receipt = self.resolve()
        reporting.persist_reporting_receipt(self.protocol_path, receipt)
        self.manifest_path.write_text(self.manifest_path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "receipt or migration sources changed"):
            self.resolve()
        self.write(self.manifest_path, self.manifest)
        changed = copy.deepcopy(receipt)
        changed["base_commit"] = "changed"
        self.write(self.protocol_path.with_name(reporting.RECEIPT_NAME), changed)
        with self.assertRaisesRegex(ValueError, "receipt or migration sources changed"):
            self.resolve()
        with self.assertRaisesRegex(ValueError, "Refusing to replace"):
            reporting.persist_reporting_receipt(self.protocol_path, receipt)

    def test_protocol_change_after_resolution_rejects_receipt_write(self):
        _, receipt = self.resolve()
        self.protocol_path.write_text(self.protocol_path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "protocol changed before recording"):
            reporting.persist_reporting_receipt(self.protocol_path, receipt)

    def test_symlink_and_unsafe_source_paths_are_rejected(self):
        path = self.root / "scripts/eval_dataset.py"
        data = path.read_bytes()
        target = self.root / "same-content"
        target.write_bytes(data)
        path.unlink()
        path.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "Current source does not match"):
            self.resolve()
        current = copy.deepcopy(self.current)
        current["code_sha256"]["../outside.py"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "Invalid source fingerprint"):
            self.resolve(current)


if __name__ == "__main__":
    unittest.main()

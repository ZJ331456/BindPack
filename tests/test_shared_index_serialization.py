"""Formatting changes can reuse an index; changed JSON values cannot."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("bindpack_serialization_runner", ROOT / "scripts/utils/stage_ablations.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class SharedIndexSerializationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.old_data = self.base / "original/musique.json"
        self.old_corpus = self.base / "original/musique_corpus.json"
        self.new_data = self.base / "new/musique.json"
        self.new_corpus = self.base / "new/musique_corpus.json"
        self.samples = [{"id": "2hop__1", "question": "Which café?", "metadata": {"a": 1, "b": 2}},
                        {"id": "4hop__2", "question": "Where?", "supports": ["A", "B"]}]
        self.corpus = [{"title": "A", "text": "First sentence. Second sentence."},
                       {"title": "B", "text": "More evidence."}]
        for path, value in ((self.old_data, self.samples), (self.old_corpus, self.corpus)):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, ensure_ascii=True, separators=(",", ":")), encoding="utf-8")
        for path, value in ((self.new_data, self.samples), (self.new_corpus, self.corpus)):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.config = {"runtime": {"llm_name": "qwen3-8b", "embedding_model_name": "Qwen3-Embedding-8B"}}
        self.index = self.base / "shared_indexes/musique"
        self.metadata = self.base / "metadata/musique"
        self.model_name = runner.model_dir(self.config["runtime"])
        self.assets = {f"{self.model_name}/{name}": str(number) * 64 for number, name in enumerate((
            "graph.pickle", "chunk_embeddings/vdb_chunk.parquet", "entity_embeddings/vdb_entity.parquet",
            "fact_embeddings/vdb_fact.parquet"), 1)}
        self.manifest = {"dataset": "musique", "index_build_status": "ready", "model_dir": self.model_name,
            "runtime": self.config["runtime"], "data_path": str(self.old_data), "corpus_path": str(self.old_corpus),
            "data_sha256": runner.sha256(self.old_data), "corpus_sha256": runner.sha256(self.old_corpus),
            "indexed_docs": 2, "selected_indices": [0, 1], "selected_sample_ids": ["2hop__1", "4hop__2"],
            "source_asset_sha256": {str(Path(name).relative_to(self.model_name)): sha for name, sha in self.assets.items()}}
        self.identity = {"dataset": "musique", "data_path": str(self.new_data),
            "data_sha256": runner.sha256(self.new_data), "corpus_source": str(self.new_corpus),
            "corpus_source_sha256": runner.sha256(self.new_corpus), "corpus_docs": 2,
            "selected_indices": [0, 1], "sample_ids": ["2hop__1", "4hop__2"]}
        runner.write_json(self.metadata / "manifest.json", self.manifest)
        runner.write_json(self.metadata / "selected_indices.json", [0, 1])
        runner.write_json(self.index / self.model_name / "index_manifest.json", {
            "embedding": {"model_name": self.config["runtime"]["embedding_model_name"]},
            "openie": {"identity": {"model_name": self.config["runtime"]["llm_name"]}}})
        self.hashes = patch.object(runner, "index_hashes", return_value=self.assets)
        self.hashes.start()
        self.addCleanup(self.hashes.stop)

    def validate(self):
        return runner.validate_shared_index({"index": self.index, "identity": self.identity}, self.config)

    def save_current(self, *, samples=None, corpus=None):
        if samples is not None:
            runner.write_json(self.new_data, samples)
            self.identity["data_sha256"] = runner.sha256(self.new_data)
        if corpus is not None:
            runner.write_json(self.new_corpus, corpus)
            self.identity["corpus_source_sha256"] = runner.sha256(self.new_corpus)

    def test_full_json_formatting_equivalence_produces_receipt_without_source_mutation(self):
        paths = [self.old_data, self.old_corpus, self.metadata / "manifest.json",
                 self.metadata / "selected_indices.json", self.index / self.model_name / "index_manifest.json"]
        before = {path: path.read_bytes() for path in paths}
        receipt = self.validate()
        equivalence = receipt["serialization_equivalence"]
        self.assertEqual(equivalence["kind"], "exact_json_value_equivalence")
        for name in ("data", "corpus"):
            sources = equivalence[name]
            self.assertNotEqual(sources["original"]["sha256"], sources["current"]["sha256"])
            self.assertEqual(sources["original"]["json_sha256"], sources["current"]["json_sha256"])
            self.assertNotEqual(sources["original"]["path"], sources["current"]["path"])
        self.assertEqual(receipt["assets"], self.assets)
        self.assertEqual(before, {path: path.read_bytes() for path in paths})

    def test_unchanged_bytes_keep_existing_policy_without_requiring_old_source_files(self):
        self.new_data.write_bytes(self.old_data.read_bytes())
        self.new_corpus.write_bytes(self.old_corpus.read_bytes())
        self.identity["data_sha256"] = runner.sha256(self.new_data)
        self.identity["corpus_source_sha256"] = runner.sha256(self.new_corpus)
        self.old_data.unlink()
        self.old_corpus.unlink()
        self.assertNotIn("serialization_equivalence", self.validate())

    def test_actual_question_change_is_rejected(self):
        samples = copy.deepcopy(self.samples)
        samples[0]["question"] += " "
        self.save_current(samples=samples)
        with self.assertRaisesRegex(ValueError, "data complete JSON values differ"):
            self.validate()

    def test_unretrieved_annotation_change_is_also_rejected(self):
        samples = copy.deepcopy(self.samples)
        samples[0]["metadata"]["a"] = 9
        self.save_current(samples=samples)
        with self.assertRaisesRegex(ValueError, "data complete JSON values differ"):
            self.validate()

    def test_query_and_corpus_array_order_changes_are_rejected(self):
        self.save_current(samples=list(reversed(self.samples)))
        with self.assertRaisesRegex(ValueError, "data complete JSON values differ"):
            self.validate()
        self.save_current(samples=self.samples, corpus=list(reversed(self.corpus)))
        with self.assertRaisesRegex(ValueError, "corpus complete JSON values differ"):
            self.validate()

    def test_passage_text_is_not_whitespace_normalized(self):
        corpus = copy.deepcopy(self.corpus)
        corpus[0]["text"] = corpus[0]["text"].replace(". ", ".  ")
        self.save_current(corpus=corpus)
        with self.assertRaisesRegex(ValueError, "corpus complete JSON values differ"):
            self.validate()

    def test_original_missing_or_drifted_files_cannot_establish_equivalence(self):
        self.old_data.unlink()
        with self.assertRaisesRegex(ValueError, "original source is not a regular"):
            self.validate()
        runner.write_json(self.old_data, self.samples)
        with self.assertRaisesRegex(ValueError, "original source byte fingerprint changed"):
            self.validate()

    def test_current_input_must_still_match_preflight_byte_sha(self):
        self.new_data.write_text(self.new_data.read_text() + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "current source byte fingerprint changed"):
            self.validate()

    def test_missing_original_path_or_symlink_is_rejected(self):
        manifest = copy.deepcopy(self.manifest)
        manifest.pop("corpus_path")
        runner.write_json(self.metadata / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "corpus original source path is unavailable"):
            self.validate()
        runner.write_json(self.metadata / "manifest.json", self.manifest)
        target = self.old_data.with_name("original_data.json")
        self.old_data.rename(target)
        self.old_data.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "original source is not a regular"):
            self.validate()

    def test_equivalence_does_not_bypass_ordered_sample_id_or_asset_checks(self):
        self.identity["sample_ids"] = list(reversed(self.identity["sample_ids"]))
        with self.assertRaisesRegex(ValueError, "Selected samples differ"):
            self.validate()
        self.identity["sample_ids"] = ["2hop__1", "4hop__2"]
        self.assets[f"{self.model_name}/graph.pickle"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "Shared index assets differ"):
            self.validate()


if __name__ == "__main__":
    unittest.main()

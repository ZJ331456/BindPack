"""Dataset annotation policies must not silently turn malformed labels into hops."""

import copy
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "bindpack_dataset_adapter_test", ROOT / "src/bindpack/evaluation/dataset_adapter.py"
)
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


def support_sample(*titles):
    return {"supporting_facts": [[title, index] for index, title in enumerate(titles)]}


def musique_sample(hops):
    return {"question_decomposition": [{"question": f"Question {i}?"} for i in range(hops)]}


class DatasetAdapterTests(unittest.TestCase):
    def test_default_data_directory_is_local_repository(self):
        self.assertEqual(adapter.DEFAULT_DATA_DIR, ROOT / "datasets")

    def test_hotpot_uses_distinct_documents_despite_extra_support_sentences(self):
        self.assertEqual(adapter.get_benchmark_hops(
            [support_sample("A", "A", "B", "B", "B")], "hotpotqa"), [2])

    def test_hotpot_rejects_missing_or_inconsistent_annotations(self):
        for sample in ({}, support_sample("A"), support_sample("A", "B", "C")):
            with self.subTest(sample=sample), self.assertRaises(ValueError):
                adapter.get_benchmark_hops([sample], "hotpotqa")

    def test_2wiki_preserves_four_hop_questions_and_caps_five_supports(self):
        samples = [support_sample("A", "B"), support_sample("A", "B", "C", "D"),
                   support_sample("A", "B", "C", "D", "D")]
        self.assertEqual(adapter.get_benchmark_hops(samples, "2wikimultihopqa"), [2, 4, 4])
        summary = adapter.describe_benchmark_hops(samples, "2wikimultihopqa")
        self.assertEqual(summary["distribution"], {"2": 1, "4": 2})
        self.assertEqual(summary["raw_supporting_facts_distribution"], {"2": 1, "4": 1, "5": 1})
        self.assertEqual(summary["support_count_capped_to_four"], 1)

    def test_hop_metadata_survives_protocol_json_round_trip(self):
        samples = {
            "hotpotqa": [support_sample("A", "B")],
            "2wikimultihopqa": [support_sample("A", "B", "C", "D", "D")],
            "musique": [musique_sample(3)],
            "hover": [dict(support_sample("A", "B", "C"), num_hops=3)],
        }
        for dataset, rows in samples.items():
            with self.subTest(dataset=dataset):
                metadata = adapter.describe_benchmark_hops(rows, dataset)
                self.assertEqual(json.loads(json.dumps(metadata)), metadata)

    def test_2wiki_uses_requested_support_count_not_evidence_or_type(self):
        sample = support_sample("A", "A", "B", "B")
        sample.update(type="comparison", evidences=[["subject", "relation", "answer"]])
        self.assertEqual(adapter.get_benchmark_hops([sample], "2wikimultihopqa"), [4])

    def test_2wiki_rejects_three_instead_of_inventing_a_four_hop_bucket(self):
        with self.assertRaisesRegex(ValueError, "3 supporting_facts"):
            adapter.get_benchmark_hops([support_sample("A", "B", "C")], "2wikimultihopqa")

    def test_support_annotations_validate_title_and_sentence_index(self):
        bad_facts = [[], [["A"]], [["", 0]], [["A", True]], [["A", -1]],
                     [["A", "1"]], [["A", 1, "extra"]], "A"]
        for dataset in ("hotpotqa", "2wikimultihopqa", "hover"):
            for facts in bad_facts:
                with self.subTest(dataset=dataset, facts=facts), self.assertRaises(ValueError):
                    adapter.get_benchmark_hops([{"supporting_facts": facts, "num_hops": 2}], dataset)

    def test_musique_retains_native_decomposition_length(self):
        self.assertEqual(adapter.get_benchmark_hops(
            [musique_sample(hop) for hop in (2, 3, 4)], "musique"), [2, 3, 4])

    def test_musique_validates_all_samples_and_rejects_missing_steps(self):
        for bad in ({}, musique_sample(1), musique_sample(5),
                    {"question_decomposition": [{"question": " "}, {"question": "Q?"}]}):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "sample index 1"):
                adapter.get_benchmark_hops([musique_sample(2), bad], "musique")

    def test_hover_native_hop_not_sentence_count_or_hotpot_origin(self):
        sample = support_sample("A", "B", "C", "C")
        sample.update(num_hops=3, hotpotqa_id="source", hotpotqa_answer="unrelated")
        self.assertEqual(adapter.get_benchmark_hops([sample], "hover"), [3])

    def test_hover_supports_native_two_three_four_hop_examples(self):
        samples = []
        for hop in (2, 3, 4):
            sample = support_sample(*[str(i) for i in range(hop)])
            sample["num_hops"] = hop
            samples.append(sample)
        self.assertEqual(adapter.get_benchmark_hops(samples, "hover"), [2, 3, 4])

    def test_hover_requires_integer_hops_matching_distinct_documents(self):
        for value in (None, True, "3", 1, 5, 2):
            sample = support_sample("A", "B", "C")
            sample["num_hops"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                adapter.get_benchmark_hops([sample], "hover")

    def test_hop_provenance_discloses_all_multihop_gold_annotation_use(self):
        for dataset in adapter.MULTIHOP_DATASETS:
            provenance = adapter.get_hop_provenance(dataset)
            self.assertTrue(provenance["oracle"])
            self.assertEqual(provenance["kind"], "per_question_oracle")
            self.assertFalse(provenance["serial_dag_depth"])
        for dataset in ("nq", "popqa"):
            self.assertEqual(adapter.get_benchmark_hops([{}, {}], dataset), [1, 1])
            self.assertFalse(adapter.get_hop_provenance(dataset)["oracle"])

    def test_unknown_dataset_and_invalid_samples_fail_without_fallback(self):
        with self.assertRaises(ValueError):
            adapter.get_benchmark_hops([], "unknown")
        with self.assertRaises(ValueError):
            adapter.get_benchmark_hops([None], "hotpotqa")

    def test_hover_query_and_gold_label_do_not_use_source_hotpot_qa(self):
        sample = {"hover_id": "hover-42", "claim": "A claim.", "question": "Wrong question?",
                  "label": "NOT_SUPPORTED", "hotpotqa_answer": "Wrong answer", "_id": "wrong-id"}
        before = copy.deepcopy(sample)
        self.assertEqual(adapter.get_query_text(sample, "hover"), "A claim.")
        self.assertEqual(adapter.get_sample_id(sample, "hover"), "hover-42")
        self.assertEqual(adapter.get_classification_label(sample), "NOT_SUPPORTED")
        self.assertEqual(adapter.get_dataset_task("hover"), "fact_verification")
        self.assertEqual(sample, before)

    def test_hover_rejects_missing_claim_id_and_noncanonical_gold_labels(self):
        for sample in ({}, {"claim": " "}, {"question": "Q?"}):
            with self.assertRaises(ValueError):
                adapter.get_query_text(sample, "hover")
        for sample in ({"hotpotqa_id": "source-id"}, {"hover_id": 42},
                       {"hover_id": True}, {"hover_id": " "}):
            with self.subTest(sample=sample), self.assertRaises(ValueError):
                adapter.get_sample_id(sample, "hover")
        for label in (None, True, "supported", " SUPPORTED ", "REFUTED", "yes", []):
            with self.subTest(label=label), self.assertRaises(ValueError):
                adapter.get_classification_label({"label": label})

    def test_other_queries_and_ids_keep_native_values(self):
        self.assertEqual(adapter.get_query_text({"question": "Q?"}, "2wikimultihopqa"), "Q?")
        self.assertEqual(adapter.get_sample_id({"_id": "wiki-1"}, "2wikimultihopqa"), "wiki-1")
        self.assertEqual(adapter.get_sample_id({"id": "2hop__1"}, "musique"), "2hop__1")
        self.assertEqual(adapter.get_dataset_task("hotpotqa"), "question_answering")

    def test_context_rendering_preserves_exact_corpus_whitespace(self):
        sentences = ["First.", " Second."]
        expected = "Title\nFirst. Second."
        self.assertEqual(adapter.render_context_document("Title", sentences, "hover"), expected)
        self.assertEqual(adapter.render_context_document("Title", sentences, "hotpotqa"), expected)
        self.assertEqual(adapter.render_context_document("Title", ["First.", "Second."],
                                                        "2wikimultihopqa"), expected)
        for invalid in ("text", [], [None]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                adapter.render_context_document("Title", invalid, "hover")


if __name__ == "__main__":
    unittest.main()

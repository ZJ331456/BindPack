"""Independent gold graph and offline metric contracts, without model calls."""

import copy
import unittest

from bindpack.evaluation.gold_graph import (
    build_gold_graph, gold_documents_from_sample, validate_gold_graph,
)
from bindpack.evaluation.structural_metrics import (
    evaluate_graph_coverage, evaluate_query_metrics, evaluate_structural_metrics,
)


def musique_sample(parents=((), (1,), (2,))):
    return {
        "id": "sample", "question": "final question",
        "paragraphs": [
            {"idx": 10 + index, "title": f"D{index + 1}", "paragraph_text": f"text {index + 1}", "is_supporting": True}
            for index in range(len(parents))
        ],
        "question_decomposition": [
            {"id": 900 + index * 19, "question": "step " + " ".join(f"#{p}" for p in refs),
             "paragraph_support_idx": 10 + index}
            for index, refs in enumerate(parents)
        ],
    }


def node(name, parents=(), bundles=None):
    return {"id": name, "parents": list(parents), "evidence_bundles": bundles or [[name]]}


class GoldGraphTests(unittest.TestCase):
    def test_musique_uses_positions_not_global_ids_and_paragraph_idx_not_list_offset(self):
        sample = musique_sample()
        sample["paragraphs"].reverse()
        corpus = {"D1\ntext 1", "D2\ntext 2", "D3\ntext 3"}
        graph = build_gold_graph(sample, "musique", corpus)
        self.assertTrue(graph["valid"], graph)
        self.assertEqual(graph["edges"], [["s1", "s2"], ["s2", "s3"]])
        self.assertEqual(graph["nodes"][2]["ancestors"], ["s1", "s2"])
        self.assertEqual(graph["nodes"][0]["source_step_id"], 900)
        self.assertEqual(graph["nodes"][0]["evidence_bundles"], [["D1\ntext 1"]])
        self.assertEqual(graph["corpus_validation"], "exact_match_passed")

    def test_native_graph_rejects_bad_references_support_flags_and_unmapped_labels(self):
        for mutation, expected in (
            (lambda s: s["question_decomposition"][1].update(question="use #900"), "dependency_reference_not_earlier_position"),
            (lambda s: s["question_decomposition"][1].update(question="use #2"), "dependency_reference_not_earlier_position"),
            (lambda s: s["question_decomposition"][1].update(question="use #0"), "dependency_reference_not_earlier_position"),
            (lambda s: s["question_decomposition"][1].update(question="use #oops"), "invalid_dependency_reference"),
            (lambda s: s["question_decomposition"][1].update(paragraph_support_idx=999), "missing_supporting_paragraph_idx"),
            (lambda s: s["question_decomposition"][1].update(paragraph_support_idx=-1), "unsupported_missing_support_or_compute_annotation"),
            (lambda s: s["paragraphs"][1].update(is_supporting=False), "paragraph_not_explicitly_supporting"),
            (lambda s: s["paragraphs"][1].update(idx=10), "duplicate_paragraph_idx"),
            (lambda s: s["paragraphs"].append({"idx": 99, "title": "extra", "text": "extra", "is_supporting": True}), "unmapped_supporting_paragraph"),
        ):
            with self.subTest(expected=expected):
                sample = musique_sample()
                mutation(sample)
                self.assertEqual(build_gold_graph(sample, "musique")["reason"], expected)
        self.assertEqual(build_gold_graph(musique_sample(), "musique", set())["reason"], "supporting_document_missing_from_corpus")

    def test_no_graph_is_fabricated_for_hotpot_or_2wiki(self):
        for dataset in ("hotpotqa", "2wikimultihopqa"):
            result = build_gold_graph({"type": "bridge", "predicted_plan": [1, 2]}, dataset)
            self.assertFalse(result["valid"])
            self.assertEqual(result["reason"], "dataset_has_no_verified_gold_dependency_graph")

    def test_invalid_graphs_are_unevaluable(self):
        invalid = [
            ([], "empty_gold_graph"),
            ([node("a"), node("a")], "duplicate_node_id"),
            ([node("a", ["missing"])], "missing_parent_node"),
            ([node("a", ["b"]), node("b", ["a"])], "cyclic_gold_graph"),
            ([{"id": "a", "parents": [], "evidence_bundles": []}], "missing_evidence_mapping"),
        ]
        for nodes, reason in invalid:
            with self.subTest(reason=reason):
                graph = validate_gold_graph(nodes)
                self.assertEqual(graph["reason"], reason)
                self.assertIsNone(evaluate_graph_coverage(graph, {"a"})["DCC"])

    def test_canonical_sentence_join_and_missing_title_validation(self):
        sample = {"supporting_facts": [["A", 0]], "context": [["A", ["x", "y"]]]}
        self.assertEqual(gold_documents_from_sample(sample, "hotpotqa")["docs"], ["A\nxy"])
        self.assertEqual(gold_documents_from_sample(sample, "2wikimultihopqa")["docs"], ["A\nx y"])
        sample["supporting_facts"].append(["absent", 0])
        self.assertFalse(gold_documents_from_sample(sample, "hotpotqa")["valid"])
        context = {"title": "A", "text": "", "paragraph_text": "fallback must not replace contexts.text"}
        self.assertEqual(gold_documents_from_sample({"contexts": [context]}, "nq")["docs"], ["A\n"])


class StructuralCoverageTests(unittest.TestCase):
    def test_chain_and_diamond_use_all_ancestors(self):
        chain = validate_gold_graph([node("a"), node("b", ["a"]), node("c", ["b"])])
        expected = [({"a", "b", "c"}, 1), ({"a", "b"}, 2 / 3), ({"a", "c"}, 1 / 3), ({"b", "c"}, 0)]
        for docs, dcc in expected:
            result = evaluate_graph_coverage(chain, docs)
            self.assertEqual(result["DCC"], dcc)
            self.assertLessEqual(result["DCC"], result["NodeCov"])
        result = evaluate_graph_coverage(chain, {"b", "c"})
        self.assertEqual(result["missing_ancestors"]["c"], ["a"])
        diamond = validate_gold_graph([node("a"), node("b", ["a"]), node("c", ["a"]), node("d", ["b", "c"])])
        result = evaluate_graph_coverage(diamond, {"a", "b", "d"})
        self.assertEqual(result["DCC"], .5)
        self.assertEqual(result["NodeCov"], .75)
        self.assertEqual(result["missing_ancestors"]["d"], ["c"])

    def test_independent_roots_document_reuse_and_or_of_and_bundles(self):
        roots = validate_gold_graph([node("a"), node("b")])
        result = evaluate_graph_coverage(roots, {"b"})
        self.assertEqual(result["DCC"], result["NodeCov"])
        graph = validate_gold_graph([
            node("a", bundles=[["shared"]]), node("b", ["a"], [["shared"]]),
            node("c", ["b"], [["x", "y"], ["z"]]),
        ])
        self.assertEqual(evaluate_graph_coverage(graph, {"shared", "x"})["DCC"], 2 / 3)
        self.assertEqual(evaluate_graph_coverage(graph, {"shared", "z"})["DCC"], 1)
        self.assertEqual(evaluate_graph_coverage(graph, {"shared", "x", "y"})["DCC"], 1)

    def test_k_counts_original_positions_duplicates_are_not_backfilled(self):
        sample = musique_sample(((), (1,)))
        row = {"candidate_docs": ["D1\ntext 1", "D1\ntext 1", "D2\ntext 2"]}
        result = evaluate_query_metrics(row, sample, "musique", ks=(1, 2, 3))
        self.assertEqual(result["at_k"]["2"]["Recall"], .5)
        self.assertEqual(result["at_k"]["2"]["FCR"], 0)
        self.assertEqual(result["at_k"]["3"]["FCR"], 1)
        self.assertEqual(result["ranking"]["duplicate_positions"], 1)

    def test_ranks_allow_offline_large_k_without_fabricating_exported_tail(self):
        sample = musique_sample(((), (1,)))
        row = {"docs": ["D1\ntext 1"], "gold_docs": ["D1\ntext 1", "D2\ntext 2"],
               "gold_document_ranks": [{"doc": "D1\ntext 1", "rank": 1}, {"doc": "D2\ntext 2", "rank": 17}]}
        result = evaluate_query_metrics(row, sample, "musique")
        self.assertEqual(result["at_k"]["10"]["FCR"], 0)
        self.assertEqual(result["at_k"]["20"]["DCC"], 1)
        self.assertTrue(result["at_k"]["200"]["exported_prefix_below_k"])
        self.assertIsNone(result["at_k"]["200"]["actual_ranking_below_k"])
        row.pop("gold_document_ranks")
        result = evaluate_query_metrics(row, sample, "musique")
        self.assertIsNone(result["at_k"]["5"]["FCR"])
        self.assertEqual(result["at_k"]["5"]["document_unavailable_reason"], "insufficient_exported_ranking_coverage")
        result = evaluate_query_metrics(row, sample, "musique", ranking_complete=True)
        self.assertEqual(result["at_k"]["200"]["Recall"], .5)

    def test_rank_null_is_absent_and_stale_rank_gold_mapping_is_rejected(self):
        sample = musique_sample(((), (1,)))
        row = {"candidate_docs": ["D1\ntext 1"],
               "gold_document_ranks": [{"doc": "D1\ntext 1", "rank": 1}, {"doc": "D2\ntext 2", "rank": None}]}
        result = evaluate_query_metrics(row, sample, "musique")["at_k"]["200"]
        self.assertEqual(result["Recall"], .5)
        self.assertEqual(result["DCC"], .5)
        for bad in (
            [{"doc": "different passage", "rank": 1}],
            [{"doc": "D1\ntext 1", "rank": 1}],
        ):
            changed = dict(row, gold_document_ranks=bad)
            result = evaluate_query_metrics(changed, sample, "musique")["at_k"]["5"]
            self.assertIsNone(result["DCC"])
            self.assertEqual(result["graph_unavailable_reason"], "gold_rank_document_set_mismatch")

    def test_prefix_rank_disagreement_and_invalid_rank_fail_closed(self):
        sample = musique_sample(((), (1,)))
        for rank in (2, True, -1, "1"):
            row = {"candidate_docs": ["D1\ntext 1", "D2\ntext 2"],
                   "gold_document_ranks": [{"doc": "D1\ntext 1", "rank": rank}, {"doc": "D2\ntext 2", "rank": 2}]}
            result = evaluate_query_metrics(row, sample, "musique")["at_k"]["5"]
            self.assertIsNone(result["FCR"])
            self.assertIsNone(result["DCC"])

    def test_empty_gold_is_excluded_and_macro_weights_queries_not_nodes(self):
        first, second = musique_sample(((), (1,))), musique_sample()
        first["id"], second["id"] = "first", "second"
        rows = [
            {"sample_id": "first", "candidate_docs": ["D1\ntext 1"] * 200},
            {"sample_id": "second", "candidate_docs": ["D1\ntext 1", "D2\ntext 2", "D3\ntext 3"] * 67},
            {"sample_id": "empty", "candidate_docs": []},
        ]
        empty = {"id": "empty", "paragraphs": [], "question_decomposition": []}
        result = evaluate_structural_metrics(rows, [first, second, empty], "musique")
        self.assertEqual(result["metrics"]["DCC@5"], .75)
        self.assertEqual(result["metrics"]["FCR@5"], .5)
        self.assertEqual(result["coverage"]["per_k"]["5"]["document_n"], 2)
        self.assertEqual(result["coverage"]["per_k"]["5"]["graph_n"], 2)
        self.assertIsNone(result["per_query"][2]["at_k"]["5"]["FCR"])

    def test_hotpot_has_valid_fcr_but_null_dcc_with_reason(self):
        sample = {"id": "x", "supporting_facts": [["A", 0]], "context": [["A", ["text"]]]}
        row = {"sample_id": "x", "gold_document_ranks": [{"doc": "A\ntext", "rank": 4}]}
        result = evaluate_structural_metrics([row], [sample], "hotpotqa")
        self.assertEqual(result["metrics"]["FCR@5"], 1)
        self.assertIsNone(result["metrics"]["DCC@5"])
        self.assertEqual(result["coverage"]["per_k"]["5"]["graph_unavailable_reasons"],
                         {"dataset_has_no_verified_gold_dependency_graph": 1})

    def test_alignment_and_k_validation(self):
        sample = musique_sample()
        with self.assertRaises(ValueError):
            evaluate_structural_metrics([], [sample], "musique")
        with self.assertRaises(ValueError):
            evaluate_structural_metrics([{"sample_id": "other"}], [sample], "musique")
        with self.assertRaises(ValueError):
            evaluate_query_metrics({}, sample, "musique", ks=(0,))
        before = copy.deepcopy(sample)
        evaluate_query_metrics({}, sample, "musique")
        self.assertEqual(sample, before)


if __name__ == "__main__":
    unittest.main()

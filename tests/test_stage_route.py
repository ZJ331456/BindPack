"""Stage replacement checks for the fixed candidate adapter."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import numpy as np

from bindpack import BindPack, BindPackConfig
from bindpack.stages.route.baseline import baseline_graph_search
from bindpack.utils.misc_utils import compute_mdhash_id


class RouteAblationTests(unittest.TestCase):
    def rag(self):
        alice = compute_mdhash_id("alice", "entity-")
        bob = compute_mdhash_id("bob", "entity-")
        nodes = [alice, bob, "unused", "p0", "p1"]
        return SimpleNamespace(
            graph=SimpleNamespace(vs={"name": nodes}),
            node_name_to_vertex_idx=dict(zip(nodes, range(len(nodes)))),
            ent_node_to_chunk_ids={alice: {"p0", "p1"}, bob: {"p0"}},
            passage_node_keys=["p0", "p1"],
            config=BindPackConfig(linking_top_k=5, passage_node_weight=.05, damping=.5),
            dense_passage_retrieval=Mock(return_value=(np.array([0, 1]), np.array([.9, .1]))),
            run_ppr=Mock(return_value=(np.array([1, 0]), np.array([.8, .2]))),
        )

    def test_baseline_uses_frequency_fact_seeds_and_fixed_ppr_without_nan(self):
        rag = self.rag()
        ids, scores, ctx = baseline_graph_search(
            rag, "q", np.array([.8]), [("Alice", "knows", "Bob")], [0], 2)
        weights = rag.run_ppr.call_args.args[0]
        np.testing.assert_allclose(weights, [.4, .8, 0., .05, 0.])
        self.assertTrue(np.isfinite(weights).all())
        self.assertEqual(rag.run_ppr.call_args.kwargs, {"damping": .5})
        self.assertEqual(ids.tolist(), [1, 0])
        self.assertEqual(ctx["bridges_by_seed"], {})
        self.assertFalse(ctx["qd_used"])
        self.assertFalse(ctx["pcqd_used"])
        self.assertFalse(ctx["iterative_used"])

    def test_wo_route_dispatch_never_enters_enhanced_seed_or_bridge_methods(self):
        rag = self.rag()
        rag.config.stage_ablation = "wo_route"
        rag._build_seed_distribution = Mock(side_effect=AssertionError("enhanced seeds"))
        rag._detect_bridge_entities = Mock(side_effect=AssertionError("enhanced bridges"))
        ids, _, ctx = BindPack._path_graph_search(
            rag, "q", np.array([.8]), [("Alice", "knows", "Bob")], [0], 2, True)
        self.assertEqual(ids.tolist(), [1, 0])
        self.assertNotIn("_qd_deferred", ctx)
        rag._build_seed_distribution.assert_not_called()
        rag._detect_bridge_entities.assert_not_called()

    def test_all_zero_restarts_return_dense_candidates(self):
        rag = self.rag()
        rag.config.passage_node_weight = 0.
        ids, _, ctx = baseline_graph_search(rag, "q", np.array([0.]), [], [], 2)
        self.assertEqual(ids.tolist(), [0, 1])
        rag.run_ppr.assert_not_called()
        self.assertEqual(ctx["seed_distribution"], {})

    def test_stage_config_rejects_misspelled_experiment(self):
        for mode in ("full", "wo_route", "wo_plan", "wo_pack"):
            cfg = BindPackConfig(stage_ablation=mode)
            self.assertEqual(cfg.stage_ablation, mode)
            self.assertEqual(cfg.embedding_model_name, "/root/models/Qwen3-Embedding-8B")
            self.assertEqual(cfg.llm_name, "qwen3-8b")
        with self.assertRaises(ValueError):
            BindPackConfig(stage_ablation="no_plna")


if __name__ == "__main__":
    unittest.main()

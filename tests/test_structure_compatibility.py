"""CPU checks for public imports and the refactored index/runtime boundaries."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bindpack import BindPack, BindPackConfig, PathCondRAG, PCRAG
from bindpack.BaseRAG import BaseRAG as OldBaseRAG
from bindpack.bindpack import BindPack as OldBindPack
from bindpack.core.base import BaseRAG
from bindpack.embedding_store import EmbeddingStore as OldEmbeddingStore
from bindpack.index.artifacts import IndexArtifactsMixin
from bindpack.index.graph_index import GraphIndexMixin
from bindpack.retrieval.engine import BindPack as Engine
from bindpack.storage.embedding_store import EmbeddingStore
from bindpack.evidence.evidence_improvements import ImprovedEvidenceRetrieval


class StructureCompatibilityTests(unittest.TestCase):
    def test_public_entries_resolve_to_one_implementation(self):
        for entry in (OldBindPack, Engine, PathCondRAG, PCRAG):
            self.assertIs(entry, BindPack)
        self.assertIs(OldBaseRAG, BaseRAG)
        self.assertIs(OldEmbeddingStore, EmbeddingStore)
        self.assertIs(BaseRAG.index, GraphIndexMixin.index)
        self.assertIs(BindPack.index, IndexArtifactsMixin.index)

    @staticmethod
    def _base_init(instance, global_config, **kwargs):
        instance.global_config = global_config
        instance.working_dir = '/tmp/bindpack_structure_cpu'

    def test_default_runtime_is_the_final_method(self):
        with patch.object(BaseRAG, '__init__', self._base_init):
            original = BindPack()
            config = BindPackConfig(
                improvement_stage=4,
                evidence_improvements='planning,plan_prune,dag_package,support_semantic_veto',
                evidence_plan_validation='canonical_refs',
                evidence_plan_routing='question_structure',
            )
            full = BindPack(global_config=config)
        self.assertIsInstance(original.evidence_runtime, ImprovedEvidenceRetrieval)
        self.assertIsInstance(full.evidence_runtime, ImprovedEvidenceRetrieval)
        self.assertEqual(full.evidence_runtime.improvements,
                         {'planning', 'plan_prune', 'dag_package', 'support_semantic_veto'})
        self.assertTrue(full._pcrag_artifacts_path.endswith('pcrag_index_artifacts.json'))

    def test_final_configuration_preserves_budget_and_rejects_removed_modes(self):
        config = BindPackConfig()
        self.assertEqual(config.max_new_tokens, 2048)
        self.assertEqual(config.llm_prefetch_workers, 8)
        self.assertTrue(config.use_query_decomposition)
        self.assertTrue(config.use_path_conditioned_qd)
        self.assertTrue(config.use_iterative_retrieval)
        for stage in (0, 2, 3, 5):
            with self.subTest(stage=stage), self.assertRaises(ValueError):
                BindPackConfig(improvement_stage=stage)
        with self.assertRaises(ValueError):
            BindPackConfig(evidence_improvements='planning,adaptive')
        with self.assertRaises(TypeError):
            BindPackConfig(use_mpce=True)

    def test_hops_must_be_explicit_and_aligned_before_retrieval(self):
        rag = object.__new__(BindPack)
        rag.clear_query_hop_overrides()
        with self.assertRaisesRegex(ValueError, 'explicit hop count'):
            rag.retrieve(['one question'])
        for hops in ([True], [0], [5]):
            with self.subTest(hops=hops), self.assertRaises(ValueError):
                rag.set_query_hop_overrides(hops, 'benchmark')
        rag.set_query_hop_overrides([2, 4], 'benchmark')
        with self.assertRaisesRegex(ValueError, 'must match'):
            rag.retrieve(['one question'])

    def test_index_override_preserves_graph_then_artifact_order(self):
        events = []
        rag = object.__new__(BindPack)
        rag.pcrag_config = SimpleNamespace(
            use_entity_idf_index=True, use_bridge_cache_index=False,
        )
        rag.ready_to_retrieve = False
        rag.prepare_retrieval_objects = lambda: events.append('prepare')
        rag._build_index_side_artifacts = lambda: events.append('artifacts')
        with patch.object(GraphIndexMixin, 'index', autospec=True) as graph_index:
            graph_index.side_effect = lambda *args, **kwargs: events.append('graph')
            rag.index(['one intact passage'], rebuild_graph=True, retry_failed_openie=False)
        graph_index.assert_called_once_with(
            rag, ['one intact passage'], rebuild_graph=True, retry_failed_openie=False,
        )
        self.assertEqual(events, ['graph', 'prepare', 'artifacts'])

    def test_disabled_index_artifacts_only_call_graph(self):
        rag = object.__new__(BindPack)
        rag.pcrag_config = SimpleNamespace(
            use_entity_idf_index=False, use_bridge_cache_index=False,
        )
        with patch.object(GraphIndexMixin, 'index', autospec=True) as graph_index:
            rag.index(['one intact passage'])
        graph_index.assert_called_once_with(
            rag, ['one intact passage'], rebuild_graph=False, retry_failed_openie=True,
        )


if __name__ == '__main__':
    unittest.main()

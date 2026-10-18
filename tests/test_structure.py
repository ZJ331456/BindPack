"""Public stage composition, configuration loading and index/runtime boundaries."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import bindpack
from bindpack import BindPack, BindPackConfig
from bindpack.core.base import BaseRAG
from bindpack.index.artifacts import IndexArtifactsMixin
from bindpack.index.graph_index import GraphIndexMixin
from bindpack.pipeline import EvidencePipeline
from bindpack.stages.plan import PlanStage
from bindpack.stages.pack import PackStage


ROOT = Path(__file__).resolve().parents[1]


class StructureTests(unittest.TestCase):
    def test_public_entries_and_stage_composition(self):
        self.assertEqual(bindpack.__all__, ['BindPack', 'BindPackConfig'])
        self.assertIs(BaseRAG.index, GraphIndexMixin.index)
        self.assertIs(BindPack.index, IndexArtifactsMixin.index)
        for removed in ('PathCondRAG', 'PCRAG', 'PathCondRAGConfig', 'PCRAGConfig'):
            self.assertFalse(hasattr(bindpack, removed))

    @staticmethod
    def _base_init(instance, **kwargs):
        instance.working_dir = '/tmp/bindpack_structure_cpu'

    def test_runtime_explicitly_composes_the_three_stages(self):
        cfg = BindPackConfig()
        with patch.object(BaseRAG, '__init__', self._base_init):
            rag = BindPack(config=cfg)
        self.assertIs(rag.config, cfg)
        self.assertIsInstance(rag.evidence_runtime, EvidencePipeline)
        self.assertIsInstance(rag.evidence_runtime.plan, PlanStage)
        self.assertIsInstance(rag.evidence_runtime.pack, PackStage)
        self.assertTrue(rag._route_artifacts_path.endswith('route_index_artifacts.json'))
        self.assertFalse(hasattr(rag, 'pcrag_config'))

    def test_nested_defaults_match_the_single_parameter_source(self):
        params = json.loads((ROOT / 'src/bindpack/configuration/defaults.json').read_text())
        config = BindPackConfig()
        for group in ('candidates', 'route', 'plan', 'pack'):
            self.assertEqual(asdict(getattr(config, group)), params[group])
        for name, value in params['runtime'].items():
            if name == 'save_dir' and value is None:
                self.assertEqual(config.save_dir, 'outputs')  # Derived runtime destination.
            else:
                self.assertEqual(getattr(config, name), value)
        self.assertEqual(config.max_new_tokens, 2048)
        self.assertEqual(config.llm_prefetch_workers, 8)
        self.assertEqual(config.pack.budget, 5)
        self.assertEqual(config.plan.beam_width, 1)

    def test_parameter_file_merges_nested_overrides_without_sharing_mutable_defaults(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'method.json'
            path.write_text(json.dumps({'pack': {'coverage_weight': .7},
                                        'route': {'refinement': {'pcqd': {'weight_path': .5}}}}))
            config = BindPackConfig.from_file(path, stage_ablation='wo_pack', save_dir=temporary)
        self.assertEqual(config.pack.coverage_weight, .7)
        self.assertEqual(config.pack.redundancy_weight, .15)
        self.assertEqual(config.route.refinement.pcqd.weight_path, .5)
        self.assertEqual(config.stage_ablation, 'wo_pack')
        config.pack.coverage_weight = 99.
        self.assertEqual(BindPackConfig().pack.coverage_weight, .45)

    def test_parameter_file_rejects_unknown_legacy_fields_and_invalid_types(self):
        invalid = [{'evidence_improvements': 'planning'}, {'route': {'typo': True}},
                   {'pack': {'budget': True}}, {'pack': {'coverage_weight': '0.4'}},
                   {'pack': {'budget': 6}}, {'plan': {'beam_width': 2}}]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'method.json'
            for payload in invalid:
                path.write_text(json.dumps(payload))
                with self.subTest(payload=payload), self.assertRaises((ValueError, TypeError)):
                    BindPackConfig.from_file(path)
        for removed in ('improvement_stage', 'evidence_improvements', 'use_query_decomposition'):
            with self.subTest(removed=removed), self.assertRaises(TypeError):
                BindPackConfig(**{removed: 4})

    def test_invalid_runtime_types_and_route_budgets_fail_before_model_loading(self):
        invalid = [
            {"llm_prefetch_workers": 2.5}, {"max_new_tokens": "oops"},
            {"temperature": float("nan")}, {"embedding_batch_size": 0},
            {"route": {"paths": {"set_size": 0}}},
            {"route": {"paths": {"candidate_docs": -1}}},
            {"route": {"paths": {"diversity_lambda": 1.1}}},
            {"route": {"iterative": {"merge_alpha": -.1}}},
        ]
        for overrides in invalid:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                BindPackConfig.from_file(**overrides)

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

    def test_index_preserves_graph_then_artifact_order(self):
        events = []
        rag = object.__new__(BindPack)
        rag.config = BindPackConfig()
        rag.config.route.index.bridge_cache = False
        rag.ready_to_retrieve = False
        rag.prepare_retrieval_objects = lambda: events.append('prepare')
        rag._build_index_side_artifacts = lambda: events.append('artifacts')
        with patch.object(GraphIndexMixin, 'index', autospec=True) as graph_index:
            graph_index.side_effect = lambda *args, **kwargs: events.append('graph')
            rag.index(['one intact passage'], rebuild_graph=True, retry_failed_openie=False)
        graph_index.assert_called_once_with(rag, ['one intact passage'],
                                            rebuild_graph=True, retry_failed_openie=False)
        self.assertEqual(events, ['graph', 'prepare', 'artifacts'])

    def test_disabled_index_artifacts_only_call_graph(self):
        rag = object.__new__(BindPack)
        rag.config = BindPackConfig()
        rag.config.route.index.entity_idf = False
        rag.config.route.index.bridge_cache = False
        with patch.object(GraphIndexMixin, 'index', autospec=True) as graph_index:
            rag.index(['one intact passage'])
        graph_index.assert_called_once_with(rag, ['one intact passage'],
                                            rebuild_graph=False, retry_failed_openie=True)


if __name__ == '__main__':
    unittest.main()

"""Disabling only final review preserves the selected Pack evidence and scores."""
from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

import numpy as np

from bindpack import BindPackConfig
from bindpack.stages.pack.stage import PackStage
from bindpack.stages.pack.verification import support_semantic_veto as review
from test_evidence_support import fixture


class ReviewDisabledTests(unittest.TestCase):
    def test_default_full_keeps_pack_but_disables_review(self):
        cfg = BindPackConfig()
        self.assertEqual(cfg.stage_ablation, 'full')
        self.assertTrue(cfg.pack_enabled)
        self.assertEqual(cfg.pack.review_mode, 'off')
        self.assertEqual(cfg.pack.relation_weight, .6)
        self.assertEqual(BindPackConfig.from_file(pack={'selector_mode':'current','review_mode': 'standard'}).pack.review_mode, 'standard')

    def test_disabled_stage_uses_no_model_tokenizer_executor_or_source_reads(self):
        runtime, solution, llm, _ = fixture()
        runtime.cfg.pack.review_mode = 'off'
        runtime._document = Mock(side_effect=AssertionError('No review source read'))
        runtime._support_semantic_tokenizer = None
        documents, scores = solution.docs, solution.doc_scores
        before_trace = deepcopy(solution.retrieval_trace['evidence'])
        with patch.object(review, 'ThreadPoolExecutor', side_effect=AssertionError('No review pool')):
            result = review.postprocess_support_semantic_veto(runtime, [(solution, list(range(20)))])
        self.assertIs(solution.docs, documents)
        self.assertIs(solution.doc_scores, scores)
        self.assertFalse(llm.calls)
        self.assertEqual(result['logical_extra_requests'], 0)
        trace = solution.retrieval_trace['evidence']
        self.assertEqual({k: trace[k] for k in before_trace}, before_trace)
        self.assertFalse(trace['improvement_support_semantic_veto']['enabled'])
        self.assertEqual(trace['improvement_support_semantic_veto']['final_doc_ids'], list(range(20)))

    def test_full_still_executes_coverage_and_dag_before_disabled_review(self):
        runtime, _, _, _ = fixture()
        runtime.cfg.pack.review_mode = 'off'
        runtime.scorer = None
        pack = PackStage(runtime)
        selected = (np.array([3, 1, 2]), np.array([.9, .8, .7]), {'selected_prefix': []})
        pack.coverage = Mock()
        pack.coverage.select.return_value = selected
        pack.dag_selector = Mock()
        pack.dag_selector.select.return_value = selected
        state = {'plan': []}
        self.assertIs(pack.finalize('question', [1, 2, 3], [.3, .2, .1], {}, state), selected)
        pack.coverage.select.assert_called_once()
        pack.dag_selector.select.assert_called_once_with(selected, state)


if __name__ == '__main__':
    unittest.main()

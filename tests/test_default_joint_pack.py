"""The adopted default runs joint packing, keeps review off and supports w/o Pack."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import numpy as np
from bindpack import BindPackConfig
from bindpack.stages.pack.stage import PackStage
from bindpack.stages.pack.budgeted_selector import BudgetedSelector
from bindpack.stages.pack.verification import support_semantic_veto
from test_evidence_dag_package import Engine,chain


class DefaultJointPackTests(unittest.TestCase):
    def runtime(self,stage='full'):
        engine=Engine();state=chain(engine)
        for c in state['evidence_candidates'].values():
            c.update(base_score=0,sources=[])
            for proof in c['verified']:proof['quality']=.95
        llm=Mock(side_effect=AssertionError('Default Pack must not call an LLM'))
        runtime=SimpleNamespace(cfg=BindPackConfig.from_file(stage_ablation=stage),
            _document=engine._document,_trace=lambda:{},rag=SimpleNamespace(llm_model=SimpleNamespace(infer=llm)),
            _joint_visible=lambda q,ids:({d:engine._document(d) for d in ids},{}))
        return runtime,state,llm

    def test_defaults_match_adopted_joint_free_parameters(self):
        default=BindPackConfig();root=Path(__file__).resolve().parents[1]
        explicit=BindPackConfig.from_file(root/'configs/pack_joint_free.json')
        self.assertEqual(asdict(default.pack),asdict(explicit.pack))
        self.assertEqual(default.pack.selector_mode,'combined')
        self.assertEqual(default.pack.review_mode,'off')
        self.assertEqual(default.pack.relation_weight,.6)
        self.assertEqual(default.pack.joint_fixed_prefix,0)
        self.assertEqual(default.pack.joint_state_cap,4096)
        self.assertEqual(default.pack.joint_search_seconds,5.0)
        self.assertEqual(default.pack.budget,5)

    def test_default_full_executes_joint_selection_with_zero_review_requests(self):
        runtime,state,llm=self.runtime();pack=PackStage(runtime)
        ids,scores,trace=pack.finalize('Where is the county containing Sample Town?',np.arange(220),np.linspace(1,0,220),{},state)
        self.assertTrue(trace['budgeted_selection']['enabled'])
        self.assertEqual(trace['budgeted_selection']['fixed_prefix'],0)
        self.assertEqual(trace['budgeted_selection']['model_requests'],0)
        self.assertEqual(set(ids[:200]),set(range(200)))
        solution=SimpleNamespace(docs=[runtime._document(int(d)) for d in ids[:200]],doc_scores=scores[:200],
                                 retrieval_trace={'evidence':trace})
        with patch.object(support_semantic_veto,'ThreadPoolExecutor',side_effect=AssertionError('Review must stay off')):
            stats=pack.postprocess([(solution,ids[:200])])
        self.assertEqual(stats['logical_extra_requests'],0)
        self.assertFalse(solution.retrieval_trace['evidence']['improvement_support_semantic_veto']['enabled'])
        llm.assert_not_called()

    def test_wo_pack_bypasses_joint_dag_and_review_under_new_defaults(self):
        runtime,state,llm=self.runtime('wo_pack');pack=PackStage(runtime)
        self.assertFalse(runtime.cfg.pack_enabled)
        with patch.object(BudgetedSelector,'select',side_effect=AssertionError('Joint selector must be bypassed')):
            with patch.object(pack.dag_selector,'select',side_effect=AssertionError('DAG selector must be bypassed')):
                ids,scores,trace=pack.finalize('A question',np.arange(220),np.linspace(1,0,220),{},state)
        self.assertEqual(trace['pack_mode'],'pointwise_base_goal')
        self.assertNotIn('budgeted_selection',trace)
        self.assertEqual(pack.postprocess([]),{})
        llm.assert_not_called()


if __name__=='__main__':unittest.main()

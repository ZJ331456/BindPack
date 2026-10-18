"""Frozen pre-refactor Full replay, recorded before moving the stage code.

The digests cover original prompts and the complete Plan/Pack trace as well as
the exact returned ranking and scores. No model or old checkout is required.
"""
import copy
import hashlib
import json
from types import ModuleType, SimpleNamespace
import unittest

import numpy as np


from bindpack import BindPackConfig
from bindpack.pipeline import EvidencePipeline as Engine


class FrozenRAG:
    def __init__(self):
        self.config = BindPackConfig()
        self.config.pack.budget = 2
        self.passage_node_keys = ['d0', 'd1', 'd2', 'd3']
        self.documents = {
            'd0': 'Biography\nAlpha wrote Work X in 1990.',
            'd1': 'Biography\nAlpha was born in Rome in 1960.',
            'd2': 'Biography\nBeta wrote Work X in 1989.',
            'd3': 'Biography\nBeta was born in Paris in 1958.',
        }
        self.chunk_embedding_store = SimpleNamespace(
            get_row=lambda key: {'content': self.documents[key]})
        self.calls, self.searches = [], []
        self.responses = iter([
            {'nodes': [
                {'id': 's1', 'question': 'Who wrote Work X?',
                 'depends_on': [], 'answer_type': 'person'},
                {'id': 's2', 'question': 'Where was ${s1.answer} born?',
                 'depends_on': ['s1'], 'answer_type': 'place'},
            ]},
            {'hypotheses': [{'answer': 'Alpha', 'doc_id': 'D0',
                             'evidence': 'Alpha wrote Work X in 1990.', 'confidence': .9}]},
            {'hypotheses': [{'answer': 'Rome', 'doc_id': 'D1',
                             'evidence': 'Alpha was born in Rome in 1960.', 'confidence': .9}]},
        ])
        self.llm_model = SimpleNamespace(infer=self.infer)

    def infer(self, messages, temperature):
        self.calls.append(copy.deepcopy(messages))
        return json.dumps(next(self.responses)), {'finish_reason': 'stop'}

    def dense_passage_retrieval(self, question):
        self.searches.append(question)
        ids = [1, 0, 2, 3] if 'Alpha' in question else [0, 2, 1, 3]
        return np.array(ids), np.array([.9, .8, .6, .4])

    @staticmethod
    def _extract_llm_text(value):
        return str(value)


def replay():
    rag = FrozenRAG()
    runtime = Engine(rag)
    state = {
        'query_idx': 0, 'query': 'Where was the writer of Work X born?',
        'base': (np.arange(4), np.array([.9, .8, .7, .6]), {'hops': 2}),
        'static_sub_questions': [], 'pcqd_sub_questions': [],
    }
    runtime.process_window([state])
    ids, scores, trace = runtime.finalize(
        state['query'], np.arange(4), np.array([.9, .8, .7, .6]), {}, state)
    return rag, ids, scores, trace


def digest(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(encoded.encode()).hexdigest()


class FullStageRegressionTests(unittest.TestCase):
    def test_full_preserves_original_rank_scores_and_bound_searches(self):
        rag, ids, scores, trace = replay()
        np.testing.assert_array_equal(ids, [0, 1, 2, 3])
        np.testing.assert_array_equal(scores, [
            2.959090909090909, 2.459090909090909,
            0.6283333333333332, 0.3214285714285714,
        ])
        self.assertEqual(rag.searches, ['Who wrote Work X?', 'Where was Alpha born?'])
        self.assertEqual(trace['bindings'], {'s1': 'Alpha', 's2': 'Rome'})

    def test_full_preserves_complete_original_trace_and_prompts(self):
        rag, _ids, _scores, trace = replay()
        normalized_trace = dict(trace)
        self.assertEqual(normalized_trace.pop("stages"), {"route": True, "plan": True, "pack": True})
        normalized_trace["stage"] = 4
        self.assertEqual(digest(normalized_trace),
                         '266dd718656537aec920e057c8d70a12c5b90f0740ac5d4801524aeff19b43b0')
        self.assertEqual(digest(rag.calls),
                         '8d7756b19084052e4a54f5f5ea6ad5f90ec4ea03028d504e2f3eb78488acd18d')


if __name__ == '__main__':
    unittest.main()

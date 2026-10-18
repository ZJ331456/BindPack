"""Final support guard and independent semantic veto without model services."""
import copy
import json
from types import ModuleType, SimpleNamespace
import unittest

import numpy as np

from bindpack import BindPackConfig
from bindpack.stages.pack.verification import support_semantic_veto as module
from bindpack.stages.pack.verification import review_prompt as review
from bindpack.stages.pack import typed_joint_selector as selector
QUERY = 'Where was the director of Aster Film born?'
ROOT_QUOTE = 'Aster Film was directed by Alice Example.'
CANDIDATE_QUOTE = 'Alice Example was born in Sample Town in 1960.'


class FakeLLM:
    def __init__(self, semantic=True, error=None):
        self.calls = []
        self.semantic, self.error = semantic, error

    def infer(self, messages, response_format):
        self.calls.append(copy.deepcopy(messages))
        if self.error:
            raise self.error
        payload = json.loads(messages[1]['content'])
        if 'PROPOSAL' in payload:
            result = {'root': 'D0', 'candidate': 'D5', 'victim': 'D4',
                      'same_entity_event': True, 'relation_grounded': True,
                      'adds_missing_constraint': True, 'removal_safe': self.semantic,
                      'old_missing_fact': 'The director birthplace is absent.',
                      'candidate_supplied_fact': 'Alice Example was born in Sample Town.',
                      'victim_reason': 'D0 retains the director and the other topic facts remain.',
                      'reason': 'The literal sources concern the same director and bridge.'}
        else:
            result = {'swap': {'promote': 'D5', 'replace': 'D4', 'confidence': .99,
                              'root': 'D0', 'missing_condition': 'Birthplace of the film director',
                              'candidate_quote': CANDIDATE_QUOTE, 'root_quote': ROOT_QUOTE,
                              'reason': 'The candidate adds the birthplace while the film director remains.'}}
        return json.dumps(result), {'finish_reason': 'stop'}


def fixture(semantic=True, protect_victim=False, context_overflow=False, error=None, question=QUERY):
    documents = {i: f'Other {i}\nA different topic is described in this unrelated paragraph.' for i in range(20)}
    documents[0] = 'Aster Film\n' + ROOT_QUOTE
    documents[5] = 'Alice Example\n' + CANDIDATE_QUOTE
    plan = [{'id': 's1', 'question': 'Who directed Aster Film?', 'depends_on': [], 'answer_type': 'person'},
            {'id': 's2', 'question': 'Where was ${s1.answer} born?', 'depends_on': ['s1'], 'answer_type': 'place'}]
    bindings = {'s1': 'Alice Example'}
    proofs = {'s1': {'doc_id': 0, 'answer': 'Alice Example', 'evidence': ROOT_QUOTE}}
    if protect_victim:
        documents[4] = 'Existing source\nAlice Example was born in Another Town.'
        bindings['s2'] = 'Another Town'
        proofs['s2'] = {'doc_id': 4, 'answer': 'Another Town', 'evidence': documents[4].partition('\n')[2]}
    trace = {'plan': plan, 'bindings': bindings, 'branch_scores': [{'bindings': bindings, 'proofs': proofs}]}
    cfg = BindPackConfig(llm_prefetch_workers=2, max_new_tokens=2048)
    cfg.pack.review_context_length = 8192
    cfg.pack.review_tokenizer = '/unused/local/model'
    llm = FakeLLM(semantic, error)
    rag = SimpleNamespace(llm_model=llm, _extract_llm_text=str)
    runtime = SimpleNamespace(cfg=cfg, rag=rag, _document=documents.__getitem__,
                              _support_semantic_tokenizer=SimpleNamespace(
                                  apply_chat_template=lambda *args, **kwargs: [1] * (10000 if context_overflow else 8)))
    solution = SimpleNamespace(question=question, docs=[documents[i] for i in range(20)],
                               doc_scores=np.linspace(1.0, .1, 20), retrieval_trace={'evidence': trace})
    return runtime, solution, llm, documents


class FinalSupportSemanticTests(unittest.TestCase):
    def test_safe_approved_swap_preserves_top_two_top_ten_and_suffix(self):
        runtime, solution, llm, documents = fixture()
        before = list(solution.docs)
        stats = module.postprocess_support_semantic_veto(runtime, [(solution, list(range(20)))])
        trace = solution.retrieval_trace['evidence']['improvement_support_semantic_veto']
        self.assertTrue(trace['accepted'])
        self.assertEqual(solution.docs[:2], before[:2])
        self.assertEqual(set(solution.docs[:10]), set(before[:10]))
        self.assertEqual(solution.docs[10:], before[10:])
        self.assertEqual(solution.docs[4], documents[5])
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(stats['applied'], 1)
        self.assertTrue(np.all(solution.doc_scores[:-1] > solution.doc_scores[1:]))

    def test_independent_semantic_rejection_keeps_original_objects(self):
        runtime, solution, llm, _ = fixture(semantic=False)
        old_docs, old_scores = solution.docs, solution.doc_scores
        stats = module.postprocess_support_semantic_veto(runtime, [(solution, list(range(20)))])
        self.assertIs(solution.docs, old_docs)
        self.assertIs(solution.doc_scores, old_scores)
        self.assertEqual(stats['semantic_vetoes'], 1)
        self.assertEqual(len(llm.calls), 2)

    def test_existing_winning_source_is_protected_even_without_dag_eligibility(self):
        runtime, solution, llm, _ = fixture(protect_victim=True)
        old_docs = solution.docs
        stats = module.postprocess_support_semantic_veto(runtime, [(solution, list(range(20)))])
        diag = solution.retrieval_trace['evidence']['improvement_support_semantic_veto']
        self.assertIn(4, diag['support_guard']['protected_source_doc_ids'])
        self.assertIs(solution.docs, old_docs)
        self.assertEqual(stats['support_vetoes'], 1)
        self.assertEqual(len(llm.calls), 2)

    def test_terminal_http_error_propagates(self):
        runtime, solution, _, _ = fixture(error=ConnectionError('HTTP unavailable'))
        with self.assertRaisesRegex(ConnectionError, 'HTTP unavailable'):
            module.postprocess_support_semantic_veto(runtime, [(solution, list(range(20)))])

    def test_context_overflow_abstains_without_issuing_a_request(self):
        runtime, solution, llm, _ = fixture(context_overflow=True)
        old_docs = solution.docs
        stats = module.postprocess_support_semantic_veto(runtime, [(solution, list(range(20)))])
        self.assertIs(solution.docs, old_docs)
        self.assertEqual(llm.calls, [])
        self.assertEqual(stats['context_abstentions'], 1)

    def test_unsupported_non_atomic_target_does_not_issue_a_request(self):
        runtime, solution, llm, _ = fixture(question='How many directors worked on Aster Film?')
        stats = module.postprocess_support_semantic_veto(runtime, [(solution, list(range(20)))])
        self.assertEqual(llm.calls, [])
        self.assertEqual(stats['applied'], 0)

    def test_physical_source_mismatch_is_rejected_before_requests(self):
        runtime, solution, llm, _ = fixture()
        solution.docs[3] = 'A different physical source'
        with self.assertRaisesRegex(ValueError, 'physical ID'):
            module.postprocess_support_semantic_veto(runtime, [(solution, list(range(20)))])
        self.assertEqual(llm.calls, [])

    def test_merged_reviewer_preserves_evaluated_visible_source_bound(self):
        runtime, solution, _, documents = fixture()
        state = module._state(solution.retrieval_trace['evidence'])
        proposal = {'promotions': [{'doc_id': 5, 'victim': 4, 'root_doc_id': 0}]}
        documents[5] += ' Extra source material.' * 200
        prompt = review.build_verifier_prompt(QUERY, list(range(20)), state, documents.__getitem__, proposal)
        payload = json.loads(prompt[1]['content'])
        self.assertEqual(review.MAX_VISIBLE_CHARS, 2000)
        self.assertEqual(len(payload['UNSELECTED_CANDIDATE']['text']), 2000)
        self.assertEqual(payload['CURRENT_TOP_FIVE'][0]['id'], 'D0')
        self.assertNotIn('D5', [p['id'] for p in payload['CURRENT_TOP_FIVE']])

    def test_review_rejects_negative_check_mismatched_id_and_truncation(self):
        runtime, solution, llm, documents = fixture()
        order = list(range(20))
        state = module._state(solution.retrieval_trace['evidence'])
        proposal_text, _ = llm.infer(
            selector.build_joint_prompt(QUERY, order, state, documents.__getitem__),
            response_format={'type': 'json_object'})
        proposed, proposal = selector.apply_joint_selection(
            QUERY, order, state, documents.__getitem__, proposal_text)
        self.assertNotEqual(proposed, order)
        review_text, _ = llm.infer(
            review.build_verifier_prompt(QUERY, order, state, documents.__getitem__, proposal),
            response_format={'type': 'json_object'})
        for change, finish, reason in (
            ({'relation_grounded': False}, 'stop', 'review_conditions_not_met'),
            ({'candidate': 'D6'}, 'stop', 'review_document_id_missing_or_mismatch'),
            ({}, 'length', 'review_response_not_completed'),
        ):
            payload = dict(json.loads(review_text), **change)
            with self.subTest(change=change, finish=finish):
                approved, diag = module.validate_verification(QUERY, order, state,
                    documents.__getitem__, proposal, json.dumps(payload), finish)
                self.assertFalse(approved)
                self.assertTrue(diag['rejection_reason'].startswith(reason))


if __name__ == '__main__':
    unittest.main()

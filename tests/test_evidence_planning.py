"""CPU tests for the final text-routed planner, canonical refs and plan pruning."""
import copy
import json
from types import ModuleType, SimpleNamespace
import unittest

from bindpack import BindPackConfig
from bindpack.pipeline import EvidencePipeline as Engine
from bindpack.stages.plan import question_structure as structure
from bindpack.stages.plan.validation import validate_plan
from bindpack.stages.plan.plan_canonicalizer import canonicalize_plan_dependencies
from bindpack.stages.plan.plan_validator import validate_retrieval_plan, prune_derived_comparison_nodes
QUERY = 'Which film has the director born earlier, Film A or Film B?'


class FakeRAG:
    def __init__(self, responses):
        self.config = BindPackConfig()
        self.responses = iter(responses)
        self.calls = []
        self.llm_model = SimpleNamespace(infer=self.infer)

    def infer(self, messages, temperature):
        self.calls.append(copy.deepcopy(messages))
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        return json.dumps(result) if isinstance(result, dict) else result, {'finish_reason': 'stop'}

    @staticmethod
    def _extract_llm_text(value):
        return str(value)


def comparison_plan():
    return {'nodes': [
        {'id': 'a1', 'question': 'Who directed Film A?', 'depends_on': [], 'answer_type': 'person'},
        {'id': 'a2', 'question': 'When was ${a1.answer} born?', 'depends_on': ['a1'], 'answer_type': 'date'},
        {'id': 'b1', 'question': 'Who directed Film B?', 'depends_on': [], 'answer_type': 'person'},
        {'id': 'b2', 'question': 'When was ${b1.answer} born?', 'depends_on': ['b1'], 'answer_type': 'date'}]}


def identity_plan():
    return {'nodes': [
        {'id': 's1', 'question': 'Who composed Opera A?', 'depends_on': [], 'answer_type': 'person'},
        {'id': 's2', 'question': 'Who composed Opera B?', 'depends_on': [], 'answer_type': 'person'},
        {'id': 's3', 'question': 'Are ${s1.answer} and ${s2.answer} the same person?',
         'depends_on': ['s1', 's2'], 'answer_type': 'boolean'}]}


class FinalPlanningTests(unittest.TestCase):
    def test_parallel_branches_use_four_nodes_under_two_hop_hint(self):
        rag = FakeRAG([comparison_plan()]);engine = Engine(rag)
        nodes, error = engine.plan.planner.plan(QUERY, 2)
        self.assertIsNone(error)
        self.assertEqual(len(nodes), 4)
        self.assertEqual(engine._plan_diagnostics[QUERY]['actual_depth'], 2)
        self.assertEqual(engine.cfg.plan.beam_width, 1)
        self.assertIn('NOT the total number', rag.calls[0][0]['content'])


    def test_ambiguous_conjunction_does_not_expand_plan(self):
        query = 'When did the country where Britain and France fought become independent?'
        self.assertFalse(structure.route_question_structure(query)['expand'])

    def test_existing_placeholders_canonicalize_dependencies_without_changing_questions(self):
        payload = comparison_plan()
        payload['nodes'][1]['depends_on'] = []
        payload['nodes'][3]['depends_on'] = ['a1', 'b1', 'b1']
        frozen = copy.deepcopy(payload)
        rag = FakeRAG([payload]);nodes, error = Engine(rag).plan.planner.plan(QUERY, 2)
        self.assertIsNone(error)
        by_id = {node['id']: node for node in nodes}
        self.assertEqual(by_id['a2']['depends_on'], ['a1'])
        self.assertEqual(by_id['b2']['depends_on'], ['b1'])
        self.assertEqual({n['id']: n['question'] for n in nodes},
                         {n['id']: n['question'] for n in payload['nodes']})
        self.assertEqual(payload, frozen)
        self.assertEqual(len(rag.calls), 1)

    def test_missing_placeholder_requires_one_bounded_model_repair(self):
        bad, corrected = comparison_plan(), comparison_plan()
        bad['nodes'][1]['question'] = 'When was the director of Film A born?'
        rag = FakeRAG([bad, corrected]);engine = Engine(rag)
        nodes, error = engine.plan.planner.plan(QUERY, 2)
        self.assertIsNone(error)
        self.assertIn('${a1.answer}', next(n for n in nodes if n['id'] == 'a2')['question'])
        self.assertEqual(engine._plan_diagnostics[QUERY]['validation_errors'], ['declared_dependency_without_placeholder'])
        self.assertEqual(len(rag.calls), 2)

    def test_wrong_birth_relation_for_death_question_requires_repair(self):
        bad, corrected = comparison_plan(), comparison_plan()
        for node in corrected['nodes']:
            if node['depends_on']:
                node['question'] = node['question'].replace('When was', 'When did').replace(' born?', ' die?')
        query = "Which film's director died earlier, Film A or Film B?"
        rag = FakeRAG([bad, corrected]);engine = Engine(rag)
        nodes, error = engine.plan.planner.plan(query, 2)
        self.assertIsNone(error)
        self.assertTrue(all('die' in n['question'] for n in nodes if n['depends_on']))
        self.assertEqual(len(rag.calls), 2)

    def test_unknown_and_self_references_cannot_become_roots(self):
        for reference in ['missing', 'a2']:
            payload = comparison_plan();payload['nodes'][1]['question'] = f'When was ${{{reference}.answer}} born?'
            self.assertEqual(canonicalize_plan_dependencies(payload)[2], 'unknown_or_self_dependency')

    def test_incomplete_placeholder_is_rejected(self):
        payload = comparison_plan();payload['nodes'][1]['question'] = 'When was ${a1.answer born?'
        self.assertEqual(canonicalize_plan_dependencies(payload)[2], 'invalid_dependency_placeholder')

    def test_cycles_and_depth_budget_are_checked_independently_of_capacity(self):
        cycle = {'nodes': [{'id': 'a', 'question': '${b.answer}', 'depends_on': ['b']},
                           {'id': 'b', 'question': '${a.answer}', 'depends_on': ['a']}]}
        self.assertEqual(validate_retrieval_plan(cycle, 6, 4)[1], 'cyclic_dependencies')
        payload = {'nodes': [{'id': f's{i}', 'question': 'Who wrote Work X?' if i == 0 else f'Where is ${{s{i-1}.answer}}?',
                              'depends_on': [] if i == 0 else [f's{i-1}']} for i in range(5)]}
        self.assertEqual(validate_retrieval_plan(payload, 6, 4)[1], 'dependency_depth_exceeded')
        self.assertIsNone(validate_retrieval_plan(payload, 6, 5)[1])

    def test_only_terminal_derived_identity_operation_is_pruned(self):
        query = 'Are Opera A and Opera B composed by the same people?'
        payload = identity_plan();frozen = copy.deepcopy(payload)
        rag = FakeRAG([payload]);nodes, error = Engine(rag).plan.planner.plan(query, 2)
        self.assertIsNone(error)
        self.assertEqual([n['id'] for n in nodes], ['s1', 's2'])
        self.assertEqual(payload, frozen)
        self.assertEqual(len(rag.calls), 1)


    def test_external_relation_and_missing_attribute_are_not_computations(self):
        for question in ['Did ${s1.answer} collaborate with ${s2.answer}?',
                         'Are ${s1.answer} and ${s2.answer} the same nationality?']:
            payload = identity_plan();payload['nodes'][-1]['question'] = question
            nodes, error = validate_plan(payload, 6)
            self.assertIsNone(error)
            self.assertEqual(prune_derived_comparison_nodes(nodes), (nodes, []))

    def test_derived_operation_with_downstream_consumer_is_retained(self):
        payload = identity_plan()
        payload['nodes'].append({'id': 's4', 'question': 'Who reported ${s3.answer}?',
                                 'depends_on': ['s3'], 'answer_type': 'person'})
        nodes, error = validate_plan(payload, 6)
        self.assertIsNone(error)
        self.assertEqual(prune_derived_comparison_nodes(nodes), (nodes, []))

    def test_ordering_needs_previously_retrieved_numeric_values(self):
        payload = identity_plan();payload['nodes'][-1]['question'] = 'Is ${s1.answer} earlier than ${s2.answer}?'
        nodes, _ = validate_plan(payload, 6)
        self.assertEqual(prune_derived_comparison_nodes(nodes)[1], [])
        for node in payload['nodes'][:2]:
            node['answer_type'] = 'date'
        nodes, _ = validate_plan(payload, 6)
        self.assertEqual([n['id'] for n in prune_derived_comparison_nodes(nodes)[0]], ['s1', 's2'])

    def test_capacity_is_enforced_after_pruning_computation(self):
        query = 'Are Opera A and Opera B composed by the same people?'
        rag = FakeRAG([identity_plan()]);rag.config.plan.node_budget = 2
        self.assertEqual(len(Engine(rag).plan.planner.plan(query, 2)[0]), 2)
        payload = identity_plan();payload['nodes'][-1]['question'] = 'Did ${s1.answer} collaborate with ${s2.answer}?'
        rag = FakeRAG([payload, payload]);rag.config.plan.node_budget = 2
        self.assertEqual(Engine(rag).plan.planner.plan(query, 2)[1], 'invalid_node_count')
        self.assertEqual(len(rag.calls), 2)

    def test_plan_failures_and_http_errors_keep_the_original_bound(self):
        rag = FakeRAG(['bad JSON', 'still bad']);engine = Engine(rag)
        self.assertEqual(engine.plan.planner.plan(QUERY, 2)[1], 'invalid_json_object')
        self.assertEqual(len(rag.calls), 2)
        rag = FakeRAG([ConnectionError('HTTP unavailable')])
        with self.assertRaisesRegex(ConnectionError, 'HTTP unavailable'):
            Engine(rag).plan.planner.plan(QUERY, 2)
        self.assertEqual(len(rag.calls), 1)


if __name__ == '__main__':
    unittest.main()

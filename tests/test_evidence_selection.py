"""Dependency closure validation shared by the final DAG selector and veto."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ModuleType('_final_closure_test_package')
PACKAGE.__path__ = [str(ROOT / 'src/bindpack')]
sys.modules[PACKAGE.__name__] = PACKAGE
SPEC = importlib.util.spec_from_file_location(
    PACKAGE.__name__ + '.evidence.selection.evidence_selection',
    ROOT / 'src/bindpack/evidence/selection/evidence_selection.py')
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


class AncestorValidationTests(unittest.TestCase):
    def test_unordered_input_preserves_transitive_ancestor_closure(self):
        plan = [{'id': 'child', 'depends_on': ['parent']},
                {'id': 'root', 'depends_on': []},
                {'id': 'parent', 'depends_on': ['root']}]
        ancestors, order, error = module._ancestors(plan)
        self.assertIsNone(error)
        self.assertEqual(order, ['root', 'parent', 'child'])
        self.assertEqual(ancestors['child'], {'root', 'parent'})

    def test_parallel_join_contains_both_source_branches(self):
        plan = [{'id': 'a'}, {'id': 'b'}, {'id': 'join', 'depends_on': ['a', 'b']}]
        ancestors, order, error = module._ancestors(plan)
        self.assertIsNone(error)
        self.assertEqual(ancestors['join'], {'a', 'b'})
        self.assertEqual(order, ['a', 'b', 'join'])

    def test_cycles_and_missing_inputs_never_return_partial_closure(self):
        for plan in [[{'id': 'a', 'depends_on': ['b']}, {'id': 'b', 'depends_on': ['a']}],
                     [{'id': 'a', 'depends_on': ['missing']}],
                     [{'id': 'a', 'depends_on': ['a']}]]:
            with self.subTest(plan=plan):
                ancestors, order, error = module._ancestors(plan)
                self.assertTrue(error)
                self.assertEqual((ancestors, order), ({}, []))

    def test_invalid_nodes_and_dependencies_are_rejected(self):
        for plan in [[None], [{'id': 1}], [{'id': 'a'}, {'id': 'a'}],
                     [{'id': 'a', 'depends_on': 'b'}], [{'id': 'a', 'depends_on': [1]}]]:
            with self.subTest(plan=plan):
                self.assertIsNotNone(module._ancestors(plan)[2])

    def test_empty_plan_has_no_implicit_ancestors(self):
        self.assertEqual(module._ancestors([]), ({}, [], None))


if __name__ == '__main__':
    unittest.main()

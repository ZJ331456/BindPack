import unittest
from adaptive_support_review import choose_protocol


class AdaptiveProtocolTests(unittest.TestCase):
    def test_missing_plan_and_partial_bindings_use_tight_source_protocol(self):
        self.assertEqual(choose_protocol({}),'tight_citations')
        self.assertEqual(choose_protocol({'plan':[{'id':'s1'},{'id':'s2'}],'bindings':{'s1':'X'}}),'tight_citations')

    def test_complete_plan_preserves_native_task_prefix(self):
        value={'plan':[{'id':'s1'},{'id':'s2'}],'bindings':{'s1':'X','s2':'Y'}}
        self.assertEqual(choose_protocol(value),'native_prefix')

    def test_gold_correctness_and_dataset_fields_cannot_change_route(self):
        value={'plan':[{'id':'s1'}],'bindings':{'s1':'X'}}
        self.assertEqual(choose_protocol(value),choose_protocol(dict(value,dataset='anything',gold_docs=['Y'],qa_correct=False)))


if __name__=='__main__':unittest.main()

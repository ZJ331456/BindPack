"""Source identity and personal attribute helpers used by final DAG checks."""
import unittest

from bindpack.stages.plan import answer_validation as module


class SourceIdentityTests(unittest.TestCase):
    def test_full_name_is_licensed_by_this_document_identity(self):
        document = 'Michael Lehmann\nMichael Stephen Lehmann (born March 30, 1957) is an American director.'
        quote = document.partition('\n')[2]
        self.assertEqual(module.supported_entity_surface('Michael Lehmann', quote, document),
                         ('Michael Stephen Lehmann', 'introductory_full_name'))

    def test_different_given_name_sharing_surname_is_not_an_alias(self):
        document = 'Michael Lehmann\nMichael Stephen Lehmann is a director. Richard Lehmann was born in 1950.'
        self.assertIsNone(module.supported_entity_surface('Michael Lehmann',
                          'Richard Lehmann was born in 1950.', document)[0])
        self.assertFalse(module._same_entity('Michael Lehmann', 'Richard Lehmann', document))

    def test_surname_requires_document_topic_identity(self):
        quote = 'Lehmann was born in Sample Town.'
        self.assertIsNone(module.supported_entity_surface('Michael Lehmann', quote,
                          'Other Person\n'+quote)[0])
        self.assertEqual(module.supported_entity_surface('Michael Lehmann', quote,
                         'Michael Lehmann\nMichael Lehmann is a director. '+quote)[1],
                         'document_anchored_surname')

    def test_time_types_cannot_accept_non_numeric_years(self):
        self.assertEqual(module._typed_answer_reason('Sample Town', 'year'), 'answer_type_mismatch')
        self.assertIsNone(module._typed_answer_reason('1957', 'year'))

    def test_life_range_preserves_birth_and_death_sides(self):
        quote = 'Alice Example (12 March 1950 – 7 June 2001) was a writer.'
        document = 'Alice Example\n'+quote
        for relation, answer in [('birth_date', '12 March 1950'), ('death_date', '7 June 2001')]:
            self.assertIsNone(module._attribute_reason(relation, answer, 'Alice Example', quote, quote, document))
        self.assertIsNotNone(module._attribute_reason('birth_date', '7 June 2001',
                                                     'Alice Example', quote, quote, document))

    def test_relatives_life_date_does_not_transfer_to_topic(self):
        quote = "Alice Example's father was born on 12 March 1950."
        document = 'Alice Example\n'+quote
        self.assertIsNotNone(module._attribute_reason('birth_date', '12 March 1950',
                                                     'Alice Example', quote, quote, document))

    def test_film_origin_cannot_establish_director_nationality(self):
        quote = 'Film A is an Italian film directed by Alice Example.'
        document = 'Film A\n'+quote
        self.assertIsNotNone(module._attribute_reason('nationality', 'Italian',
                                                     'Alice Example', quote, quote, document))

    def test_birthplace_does_not_automatically_establish_citizenship(self):
        quote = 'Alice Example was born in Italy.'
        document = 'Alice Example\n'+quote
        self.assertEqual(module._attribute_reason('nationality', 'Italy',
                                                 'Alice Example', quote, quote, document),
                         'birthplace_does_not_establish_nationality')

    def test_nationality_owned_by_named_director_is_supported(self):
        quote = 'Alice Example is an Italian film director.'
        document = 'Alice Example\n'+quote
        self.assertIsNone(module._attribute_reason('nationality', 'Italian',
                                                  'Alice Example', quote, quote, document))


if __name__ == '__main__':
    unittest.main()

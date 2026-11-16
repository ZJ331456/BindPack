import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('serial_200', Path(__file__).resolve().parents[1] / 'scripts/run_serial_cost_200.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class FixedQuotaSamplingTest(unittest.TestCase):
    def records(self, dataset):
        rows = []
        for group, count in driver.QUOTAS[dataset].items():
            for index in range(count + 8):
                row = {'id': f'{group}-{index}'}
                if dataset in ('hotpotqa', '2wikimultihopqa'):
                    row['supporting_facts'] = [[str(i), 0] for i in range(group)]
                elif dataset == 'musique':
                    row['question_decomposition'] = [{}] * group
                else:
                    row['num_hops'] = group
                rows.append(row)
        return rows

    def test_all_four_exact_quotas(self):
        from collections import Counter
        for dataset in driver.DATASETS:
            data = self.records(dataset)
            selected = driver.choose_samples(data, dataset, [row['id'] for row in data])
            self.assertEqual(len(set(selected)), 50)
            self.assertEqual(dict(Counter(driver.stratum(data[i], dataset) for i in selected)), driver.QUOTAS[dataset])

    def test_selection_reproducible_and_independent_of_answers_or_timing(self):
        data = self.records('musique')
        ids = [row['id'] for row in data]
        expected = driver.choose_samples(data, 'musique', ids)
        changed = [dict(row, answer='different', elapsed_seconds=9999) for row in data]
        self.assertEqual(expected, driver.choose_samples(changed, 'musique', ids))

    def test_rejects_insufficient_group(self):
        data = [{'num_hops': 3}] * 50
        with self.assertRaisesRegex(ValueError, 'Not enough'):
            driver.choose_samples(data, 'hover', [str(i) for i in range(50)])


if __name__ == '__main__':
    unittest.main()

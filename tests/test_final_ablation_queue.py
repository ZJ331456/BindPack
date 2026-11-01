"""The authorized order is dataset-first, with Soft Coverage excluded."""
import unittest
from run_final_ablation_queue import ordered_cases


class QueueOrderTests(unittest.TestCase):
    def test_twenty_ordered_native_cases_exclude_soft_coverage(self):
        rows=ordered_cases();self.assertEqual(len(rows),20)
        ds=['hotpotqa','2wikimultihopqa','musique','hover']
        expected=[('mechanisms',d,a) for d in ds for a in ('wo_dependency','wo_joint_selection')]
        expected += [('stages',d,a) for d in ds for a in ('wo_route','wo_plan','wo_pack')]
        self.assertEqual([(x['group'],x['dataset'],x['label']) for x in rows],expected)
        self.assertTrue(all('soft' not in x['label'] for x in rows))

    def test_stage_and_mechanism_switches_are_never_combined(self):
        for row in ordered_cases():
            if row['group']=='mechanisms':
                self.assertEqual(row['stage'],'full')
                self.assertNotEqual(row['mechanism'],'full')
            else:
                self.assertEqual(row['mechanism'],'full')
                self.assertEqual(row['stage'],row['label'])


if __name__=='__main__':unittest.main()

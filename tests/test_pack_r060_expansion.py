"""Sampling and resume integrity for the expanded R=.60 evaluation."""
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/run_pack_r060_expansion.py'
SPEC = importlib.util.spec_from_file_location('pack_expansion', SCRIPT)
run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run)


class ExpansionTests(unittest.TestCase):
    def test_draw_is_outcome_blind_reproducible_and_disjoint(self):
        rows = [dict(sample_id=str(i), query_index=i, question=f'Q{i}',
                     benchmark_hops=2, answer='old', all_gold_in_top5=i % 2)
                for i in range(200)]
        excluded = {str(i) for i in range(50)}
        first = run.choose_new(rows, excluded, 'test')
        altered = [dict(row, answer='changed', all_gold_in_top5=1-row['all_gold_in_top5'])
                   for row in reversed(rows)]
        self.assertEqual(first, run.choose_new(altered, excluded, 'test'))
        self.assertEqual(len(first), len({row['sample_id'] for row in first}))
        self.assertEqual(len(first), 100)
        self.assertFalse(excluded & {row['sample_id'] for row in first})
        with self.assertRaises(ValueError):
            run.choose_new(rows + [rows[0]], excluded, 'test')

    def test_checkpoint_resume_verifies_hashes_and_rejects_gaps(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            chosen = [dict(sample_id='a'), dict(sample_id='b')]
            run.write(root / 'selection_manifest.json', dict(datasets={'test': dict(rows=chosen)}))
            self.assertEqual(run.completed(root, 'test'), [])
            row = dict(sample_id='a', answer='A')
            path = run.checkpoint_path(root, 'test', 'a')
            run.write(path, dict(row=row, sha256=run.digest(row)))
            self.assertEqual(run.completed(root, 'test'), [row])
            run.write(path, dict(row=dict(row, answer='tampered'), sha256=run.digest(row)))
            with self.assertRaises(AssertionError):
                run.completed(root, 'test')
            path.unlink()
            run.write(run.checkpoint_path(root, 'test', 'b'),
                      dict(row=dict(sample_id='b'), sha256=run.digest(dict(sample_id='b'))))
            with self.assertRaises(AssertionError):
                run.completed(root, 'test')


if __name__ == '__main__':
    unittest.main()

"""Guard frozen inputs and restart checkpoints without importing model clients."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_pack_150_comparison as run
from analyze_pack_150_comparison import exact_p


class Paired150Tests(unittest.TestCase):
    def test_resume_returns_only_the_complete_selected_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            run.write(root/'selection_manifest.json',{'datasets':{'hotpotqa':{'rows':[{'sample_id':'a'},{'sample_id':'b'}]}}})
            row={'sample_id':'a','answer':'saved answer'}
            run.write(root/'checkpoints/hotpotqa/before'/run.token('a'),{'row':row,'sha256':run.digest(row)})
            self.assertEqual(run.completed(root,'hotpotqa','before'),[row])

    def test_resume_rejects_a_valid_digest_for_the_wrong_question(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            run.write(root/'selection_manifest.json',{'datasets':{'hotpotqa':{'rows':[{'sample_id':'a'}]}}})
            row={'sample_id':'different'}
            run.write(root/'checkpoints/hotpotqa/before'/run.token('a'),{'row':row,'sha256':run.digest(row)})
            with self.assertRaises(AssertionError):run.completed(root,'hotpotqa','before')

    def test_frozen_input_tampering_is_rejected_before_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'input.json';value={'query':'Original question','ids':[1,2],'scores':[1.,0.]}
            run.write(path,{'input':value,'input_sha256':run.digest(value)})
            chosen={'cohort':'new100','question':'Original question','frozen_path':str(path),
                    'frozen_file_sha256':run.sha(path),'frozen_pack_input_sha256':run.digest(value)}
            self.assertEqual(run.payload(chosen),value)
            value['query']='Changed question';run.write(path,{'input':value,'input_sha256':run.digest(value)})
            with self.assertRaises(AssertionError):run.payload(chosen)

    def test_pair_input_hash_does_not_depend_on_dict_insertion_order(self):
        self.assertEqual(run.digest({'ids':[1,2],'query':'Q'}),run.digest({'query':'Q','ids':[1,2]}))

    def test_exact_p_handles_ties_and_discordant_direction(self):
        self.assertEqual(exact_p(0,0),1.)
        self.assertEqual(exact_p(4,0),.125)
        self.assertEqual(exact_p(0,4),.125)
        self.assertEqual(exact_p(1,1),1.)

    def test_combined_report_records_both_actual_cache_locations(self):
        root=Path('/a/comparison')
        self.assertEqual(run.cache_dirs(root,'hover','combined150'),{
            'old50':'/a/comparison/runtime/hover/old50/index',
            'new100':'/a/comparison/runtime/hover/new100/index'})
        self.assertEqual(run.cache_dirs(root,'hover','old50'),{
            'old50':'/a/comparison/runtime/hover/old50/index'})


if __name__=='__main__':unittest.main()

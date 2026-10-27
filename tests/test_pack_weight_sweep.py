"""Read-only replay controls; no GPU or model inference."""
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/run_pack_weight_sweep.py'
SPEC=importlib.util.spec_from_file_location('pack_sweep',SCRIPT)
sweep=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(sweep)


class PackSweepTests(unittest.TestCase):
    def test_document_lookup_preserves_original_physical_row_order(self):
        with TemporaryDirectory() as folder:
            p=Path(folder)/'documents.parquet'
            pq.write_table(pa.table({'hash_id':['chunk-z','chunk-a'],'content':['Z\nBody Z','A\nBody A'],
                'embedding':[[1.,0.],[0.,1.]]}),p)
            store=sweep.ReadOnlyPassages(p)
            self.assertEqual(store.keys,['chunk-z','chunk-a'])
            self.assertEqual(store.get_row(store.keys[0])['content'],'Z\nBody Z')
            self.assertEqual(store.get_row(store.keys[1])['content'],'A\nBody A')
            with self.assertRaises(RuntimeError):store.insert_strings(['forbidden'])

    def test_weight_labels_are_distinct_and_invalid_values_rejected(self):
        self.assertEqual([sweep.label(w) for w in [.25,.4,.5,.6,.7,.8]],
                         ['full_r025','full_r040','full_r050','full_r060','full_r070','full_r080'])
        for value in [float('nan'),float('inf'),-.1,1.1,.333]:
            with self.assertRaises(ValueError):sweep.label(value)

    def test_extension_preserves_selection_and_changes_only_weight_and_cache_path(self):
        base=sweep.DEFAULT_BASE;out=base/'extension_r040_r080'
        if not (out/'experiment.json').exists():self.skipTest('sweep not prepared')
        self.assertEqual(sweep.read(out/'selection_manifest.json'),sweep.read(base/'selection_manifest.json'))
        self.assertEqual(sweep.read(out/'selection_locked.ok'),sweep.read(base/'selection_locked.ok'))
        for ds in sweep.DATASETS:
            baseline=sweep.read(base/'metadata'/ds/'full_r060_method.json')
            for weight in [.4,.8]:
                params=sweep.read(out/'metadata'/ds/(sweep.label(weight)+'_method.json'))
                self.assertEqual(params['pack']['relation_weight'],weight)
                params['pack']['relation_weight']=.6
                params['runtime']['save_dir']=baseline['runtime']['save_dir']
                self.assertEqual(params,baseline)
            self.assertEqual(sweep.read(out/'metadata'/ds/'corpus.json'),sweep.read(base/'metadata'/ds/'corpus.json'))


if __name__=='__main__':unittest.main()

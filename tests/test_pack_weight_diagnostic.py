"""Scientific controls: frozen cohorts and independent shared upstream inputs."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/run_pack_weight_diagnostic.py'
SPEC = importlib.util.spec_from_file_location('pack_diagnostic', SCRIPT)
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


class PackDiagnosticTests(unittest.TestCase):
    def test_pair_inputs_keep_physical_ids_and_cannot_share_mutable_state(self):
        state={'evidence_candidates':{7:{'doc_id':7,'verified':[],'sources':[]}},
               'evidence_trace':{'plan':[{'id':'s1'}]},'_evidence_plan':[{'id':'s1'}],
               '_evidence_winning_bindings':{'s1':'An entity'},'_evidence_beams':[]}
        frozen=diagnostic.freeze_pack_input('question',np.array([7,9]),np.array([.8,.3]),state)
        before=diagnostic.digest(frozen)
        a,b=diagnostic.restore_pack_input(frozen),diagnostic.restore_pack_input(frozen)
        a['evidence_candidates'][7]['verified'].append({'answer':'changed'})
        a['evidence_trace']['plan'][0]['id']='changed'
        self.assertEqual(b['evidence_candidates'][7]['verified'],[])
        self.assertEqual(state['evidence_trace']['plan'][0]['id'],'s1')
        self.assertEqual(diagnostic.digest(frozen),before)
        self.assertEqual(frozen['ids'],[7,9])

    def test_historical_cohort_selection_is_deterministic_and_does_not_modify_pool(self):
        rows=[dict(sample_id=str(i),benchmark_hops=2+i%3,priority=0) for i in range(90)]
        before=diagnostic.digest(rows)
        a=diagnostic.choose_group(rows,20)
        b=diagnostic.choose_group(list(reversed(rows)),20)
        self.assertEqual(a,b)
        self.assertEqual(len({r['sample_id'] for r in a}),20)
        self.assertEqual({r['benchmark_hops'] for r in a},{2,3,4})
        self.assertEqual(diagnostic.digest(rows),before)

    def test_locked_selection_contains_exact_approved_cohorts(self):
        output=Path('/root/BindPack/outputs/pack_weight_diagnostic_20261025')
        if not (output/'selection_locked.ok').exists():self.skipTest('selection not prepared')
        manifest=diagnostic.read(output/'selection_manifest.json')
        self.assertEqual(diagnostic.digest(manifest),diagnostic.read(output/'selection_locked.ok')['sha256'])
        for dataset in diagnostic.DATASETS:
            rows=manifest['datasets'][dataset]['rows']
            self.assertEqual(len(rows),50)
            self.assertEqual(len({r['sample_id'] for r in rows}),50)
            self.assertEqual(diagnostic.Counter(r['group'] for r in rows),dict(similar=10,expected_r025=20,expected_r060=20))
            methods=[diagnostic.read(output/'metadata'/dataset/(name+'_method.json')) for name in diagnostic.CASES]
            self.assertEqual(methods[0]['pack']['relation_weight'],.25)
            self.assertEqual(methods[1]['pack']['relation_weight'],.60)
            methods[1]['pack']['relation_weight']=.25
            self.assertEqual(*methods)

    def test_legacy_and_string_corpora_have_the_same_evaluation_identity(self):
        raw=[{'title':'A title','text':'A passage.'},{'title':'Other','paragraph_text':'More.'}]
        text=['A title\nA passage.','Other\nMore.']
        self.assertEqual(diagnostic.canonical_corpus(raw),text)
        self.assertEqual(diagnostic.digest(diagnostic.canonical_corpus(raw)),diagnostic.digest(text))
        output=Path('/root/BindPack/outputs/pack_weight_diagnostic_20261025')
        if not (output/'selection_locked.ok').exists():return
        for dataset in diagnostic.DATASETS:
            corpus=diagnostic.read(output/'metadata'/dataset/'corpus.json')
            identity=diagnostic.read(output/'metadata'/dataset/'sample_manifest.json')
            self.assertEqual(diagnostic.digest(diagnostic.canonical_corpus(corpus)),identity['corpus_docs_sha256'])


if __name__=='__main__':unittest.main()

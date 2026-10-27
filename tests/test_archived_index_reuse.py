"""Archive identity, private assets and source-preserving cache isolation."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
from utils import archived_index as A


class ArchivedIndexTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.archive=self.root/'archive';self.source=self.archive/'shared_indexes/hover'
        model=self.source/'llm_embedding';model.mkdir(parents=True)
        self.docs=['A\nFirst fact.','B\nSecond fact.']
        (model/'graph.pickle').write_bytes(b'graph')
        (model/'index_manifest.json').write_text(json.dumps({'embedding':{'model_name':'embedding','normalized':True,'max_sequence_length':2048,'dtype':'auto'},'openie':{'identity':{'model_name':'llm'}}}))
        for kind in ('chunk','entity','fact'):
            p=model/(kind+'_embeddings')/('vdb_'+kind+'.parquet');p.parent.mkdir()
            pq.write_table(pa.table({'hash_id':['a','b'],'content':self.docs,'embedding':[[1.,0.],[0.,1.]]}),p)
        (self.source/'openie_results_ner_llm.json').write_text('[]')
        (self.source/'llm_cache').mkdir();(self.source/'llm_cache/private.sqlite').write_bytes(b'old query cache')
        (model/'query_vectors.npy').write_bytes(b'old vectors')
        inventory=[dict(path=str(p.relative_to(self.archive)),bytes=p.stat().st_size,sha256=A.sha(p)) for p in self.source.rglob('*') if p.is_file()]
        A.write(self.archive/'manifest.json',dict(files=inventory))
        self.config=SimpleNamespace(embedding_model_name='embedding',llm_name='llm',embedding_return_as_normalized=True,embedding_max_seq_len=2048,embedding_model_dtype='auto')
        self.case=self.root/'run/cases/hover/full';self.case.mkdir(parents=True)

    def validate(self,docs=None):return A.validate_archive(self.source,'hover',self.docs if docs is None else docs,'embedding','llm')

    def test_exact_corpus_models_and_all_asset_hashes_are_required(self):
        self.assertEqual(self.validate()['corpus_docs'],2)
        with self.assertRaisesRegex(ValueError,'passages differ'):self.validate([self.docs[0],self.docs[1]+' '])
        with self.assertRaisesRegex(ValueError,'model mismatch'):A.validate_archive(self.source,'hover',self.docs,'different','llm')

    def test_changed_source_is_rejected_before_copy_or_model_load(self):
        (self.source/'llm_embedding/graph.pickle').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'immutable inventory'):self.validate()

    def test_private_copy_excludes_historical_caches_and_does_not_fabricate_build_cost(self):
        result=A.prepare_archive(self.source,self.case/'index',self.case,{'dataset':'hover'},self.docs,self.config)
        self.assertIsNone(result['original_build_seconds']);self.assertIsNone(result['original_build_usage'])
        self.assertFalse((self.case/'index/llm_cache').exists())
        self.assertFalse((self.case/'index/llm_embedding/query_vectors.npy').exists())
        for rel in result['assets']:
            self.assertNotEqual((self.source/rel).stat().st_ino,(self.case/'index'/rel).stat().st_ino)

    def test_release_preserves_results_and_independent_llm_cache(self):
        result=A.prepare_archive(self.source,self.case/'index',self.case,{'dataset':'hover'},self.docs,self.config)
        cache=self.case/'index/llm_cache/current.sqlite';cache.parent.mkdir();cache.write_bytes(b'new cache')
        (self.case/'progress.jsonl').write_text('saved checkpoint\n')
        A.verify_private(self.case,release=True)
        self.assertEqual(cache.read_bytes(),b'new cache');self.assertEqual((self.case/'progress.jsonl').read_text(),'saved checkpoint\n')
        self.assertTrue(all((self.source/rel).is_file() for rel in result['assets']))
        self.assertTrue(all(not (self.case/'index'/rel).exists() for rel in result['assets']))

    def test_embedding_strategy_change_is_rejected(self):
        self.config.embedding_max_seq_len=4096
        with self.assertRaisesRegex(ValueError,'configuration mismatch'):
            A.prepare_archive(self.source,self.case/'index',self.case,{'dataset':'hover'},self.docs,self.config)

    def test_measured_protocol_can_certify_an_index_that_never_had_a_manifest(self):
        from bindpack.evaluation.measurement import digest
        manifest=self.source/'llm_embedding/index_manifest.json';manifest.unlink()
        full=self.archive/'cases/hover/full';full.mkdir(parents=True)
        A.write(full/'protocol.json',dict(method='bindpack',stage_ablation='full',dataset={'dataset':'hover'},
            configuration=vars(self.config)))
        A.write(full/'report.json',dict(cost={'offline_index':{'input_tokens':101}}))
        A.write(full/'completed.ok',dict(report_sha256=A.sha(full/'report.json')))
        A.write(full/'index_build.json',dict(seconds=7.,corpus_sha256=digest(self.docs)))
        entries=[dict(path=str(p.relative_to(self.archive)),bytes=p.stat().st_size,sha256=A.sha(p))
            for p in self.archive.rglob('*') if p.is_file() and p!=self.archive/'manifest.json']
        A.write(self.archive/'manifest.json',dict(files=entries))
        receipt=self.validate()
        self.assertEqual(receipt['original_build_seconds'],7.)
        self.assertEqual(receipt['original_build_usage']['input_tokens'],101)
        self.assertEqual(len(receipt['archived_model_provenance']),4)
        self.assertFalse(manifest.exists())


if __name__=='__main__':unittest.main()

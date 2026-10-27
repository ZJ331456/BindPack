"""Ablation isolation, frozen native source, and append-only Full recovery."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from utils import measured_cases as C
SPEC=importlib.util.spec_from_file_location('_measured_queue_tests',ROOT/'scripts/run_bindpack_measured_cases.py')
QUEUE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(QUEUE)


class MeasuredCaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.identity=dict(dataset='hover',data_sha256='benchmark',corpus_docs_sha256='corpus',selected_indices=[0],sample_ids=['q'])
        self.models={'llm':'Qwen3-8B','embedding':'Qwen3-Embedding-8B'}
        self.source=self.root/'cases/hover/full/index'
        self.source.mkdir(parents=True)
        for name,value in [('model/graph.pickle',b'graph'),('model/chunk_embeddings/vdb_chunk.parquet',b'vectors'),
            ('openie_results_ner_qwen3-8b.json',b'{"triples":[]}'),('llm_cache/old.sqlite',b'cache'),
            ('model/openie_progress.sqlite',b'checkpoint'),('model/route_index_artifacts.json',b'lookup')]:
            p=self.source/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(value)
        C.write(self.source.parent/'report.json',dict(cost={'offline_index':{'input_tokens':100}}))
        C.write(self.source.parent/'completed.ok',dict(report_sha256=C.sha(self.source.parent/'report.json')))
        C.write(self.source.parent/'protocol.json',dict(dataset=self.identity,models=self.models))
        C.write(self.source.parent/'index_build.json',dict(seconds=10,corpus_sha256='corpus',index_bytes=8))
        self.target=self.root/'cases/hover/wo_route';self.target.mkdir()

    def test_private_assets_exclude_llm_and_query_caches_and_are_not_hardlinks(self):
        result=C.prepare_reused_index(self.source,self.target/'index',self.target,self.identity,self.models)
        self.assertEqual(len(result['assets']),3)
        self.assertFalse((self.target/'index/llm_cache').exists())
        self.assertFalse((self.target/'index/model/openie_progress.sqlite').exists())
        self.assertFalse((self.target/'index/model/route_index_artifacts.json').exists())
        for name in result['assets']:
            self.assertNotEqual((self.target/'index'/name).stat().st_ino,(self.source/name).stat().st_ino)
        (self.target/'index/model/graph.pickle').write_bytes(b'private mutation')
        self.assertEqual((self.source/'model/graph.pickle').read_bytes(),b'graph')
        with self.assertRaisesRegex(ValueError,'private asset changed'):
            C.prepare_reused_index(self.source,self.target/'index',self.target,self.identity,self.models)

    def test_wrong_model_or_sample_identity_cannot_reuse_the_index(self):
        with self.assertRaisesRegex(ValueError,'model identity'):
            C.prepare_reused_index(self.source,self.target/'index',self.target,self.identity,{'embedding':'other'})
        with self.assertRaisesRegex(ValueError,'input identity'):
            C.prepare_reused_index(self.source,self.target/'index',self.target,dict(self.identity,sample_ids=['other']),self.models)

    def test_source_asset_change_is_rejected_on_resume(self):
        C.prepare_reused_index(self.source,self.target/'index',self.target,self.identity,self.models)
        (self.source/'model/graph.pickle').write_bytes(b'changed source')
        with self.assertRaisesRegex(ValueError,'original index changed'):
            C.prepare_reused_index(self.source,self.target/'index',self.target,self.identity,self.models)

    def test_overlay_changes_control_only_and_preserves_full_native_snapshot(self):
        base=self.root/'source_snapshot'
        (base/'src').mkdir(parents=True);(base/'scripts/utils').mkdir(parents=True)
        (base/'src/native.py').write_text('unchanged native algorithm')
        (base/'scripts/run_bindpack_measured.py').write_text('old Full controller')
        scripts=self.root/'controls';(scripts/'utils').mkdir(parents=True)
        (scripts/'run_bindpack_measured.py').write_text('new ablation controller')
        (scripts/'utils/measured_cases.py').write_text('new isolation helper')
        overlay=C.install_ablation_snapshot(self.root,scripts)
        self.assertEqual((base/'scripts/run_bindpack_measured.py').read_text(),'old Full controller')
        self.assertEqual((overlay/'src/native.py').read_text(),'unchanged native algorithm')
        (overlay/'src/native.py').write_text('modified algorithm')
        with self.assertRaisesRegex(ValueError,'native algorithm'):
            C.install_ablation_snapshot(self.root,scripts)

    def test_full_uses_original_worker_without_new_cli_flags_and_ablation_uses_overlay(self):
        C.write(self.source.parent/'protocol.json',dict(dataset=dict(self.identity,source_data_path='/data/hover.json')))
        full=QUEUE.worker_command(self.root,'hover','full')
        ablated=QUEUE.worker_command(self.root,'hover','wo_plan')
        self.assertIn(str(self.root/'source_snapshot/scripts/run_bindpack_measured.py'),full)
        self.assertNotIn('--case',full);self.assertNotIn('--reuse-index',full)
        self.assertIn(str(self.root/'ablation_source_snapshot/scripts/run_bindpack_measured.py'),ablated)
        self.assertEqual(ablated[ablated.index('--case')+1],'wo_plan')
        self.assertEqual(C.CASES,('full','wo_route','wo_plan','wo_pack'))

    def test_preserved_checkpoint_and_cost_prefixes_can_only_be_appended(self):
        case=self.source.parent;files={}
        names=['progress.jsonl','llm_calls.jsonl','llm_logical_calls.jsonl','embedding_calls.jsonl',
               'resource_samples.jsonl','stage_timings.jsonl','execution_sessions.jsonl','protocol.json','index_build.json']
        for name in names:
            p=case/name
            if not p.exists():p.write_text('original\n')
            files[name]=dict(bytes=p.stat().st_size,sha256=C.sha(p))
        C.write(self.root/'serial_extension/preservation.json',dict(completed_queries=488,files=files,index_assets=C.index_assets(self.source)))
        with (case/'llm_calls.jsonl').open('a') as f:f.write('new request\n')
        self.assertEqual(C.verify_prefixes(self.root),488)
        (case/'progress.jsonl').write_text('changed\n')
        with self.assertRaisesRegex(ValueError,'prefix changed'):
            C.verify_prefixes(self.root)


if __name__=='__main__':unittest.main()

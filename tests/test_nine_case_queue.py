"""Expensive full runs must retain checkpoints, configuration and category order."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PROJECT=Path(__file__).resolve().parents[1]
def load(name,file):
    spec=importlib.util.spec_from_file_location(name,PROJECT/'scripts'/file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
F=load('native_queue_order','run_bindpack_latest_full.py')
N=load('nine_case_control','run_bindpack_nine_cases.py')


class NineCaseTests(unittest.TestCase):
    def test_full_category_finishes_before_any_wo_pack(self):
        order=F.case_order(F.DATASETS,['full','wo_pack'],True)
        self.assertEqual(order,[(ds,'full') for ds in F.DATASETS]+[(ds,'wo_pack') for ds in F.DATASETS])

    def test_nine_jobs_begin_with_original_hover_then_two_categories(self):
        jobs=N.jobs(Path('/old'),Path('/new'))
        self.assertEqual(len(jobs),9)
        self.assertEqual(jobs[0]['path'],'/old/cases/hover/full')
        self.assertEqual([j['category'] for j in jobs],['previous_hover']+['joint_full']*4+['wo_pack']*4)
        self.assertEqual([j['dataset'] for j in jobs[1:5]],list(N.DATASETS))

    def fixture(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);base=Path(temp.name)
        project=base/'project';archive=base/'archive';archive.mkdir();F.write(archive/'manifest.json',{'files':[]})
        for part in ['src/bindpack/configuration','configs','scripts','datasets']:(project/part).mkdir(parents=True)
        F.write(project/'src/bindpack/configuration/defaults.json',{'pack':{'relation_weight':.6,'review_mode':'off','selector_mode':'current'}})
        F.write(project/'configs/ablations_qwen3.json',{'evaluation':{},'runtime':{},'method_config':'src/bindpack/configuration/defaults.json'})
        (project/'scripts/dummy.py').write_text('# frozen\n')
        for ds in F.DATASETS:
            F.write(project/'datasets'/(ds+'.json'),[{'id':i} for i in range(1000)])
            F.write(project/'datasets'/(ds+'_corpus.json'),[])
        method=base/'method.json';F.write(method,{'pack':{'selector_mode':'combined','joint_fixed_prefix':0}})
        return project,archive,base/'run',method

    def test_override_is_frozen_without_changing_defaults(self):
        project,archive,root,method=self.fixture()
        with patch.object(F,'PROJECT',project),patch.object(F,'ARCHIVE',archive):
            p=F.prepare(root,['full','wo_pack'],method,True)
            self.assertEqual(p['effective_pack']['selector_mode'],'combined')
            self.assertEqual(p['effective_pack']['joint_fixed_prefix'],0)
            self.assertEqual(F.read(project/'src/bindpack/configuration/defaults.json')['pack']['selector_mode'],'current')
            self.assertEqual(F.read(root/'run_config.json')['method_config'],str(root/'method_overrides.json'))
            self.assertEqual([x['case'] for x in p['execution_order']],['full']*4+['wo_pack']*4)
            self.assertEqual(F.prepare(root,['full','wo_pack']),p)

    def test_resume_rejects_changed_method_or_order(self):
        project,archive,root,method=self.fixture()
        with patch.object(F,'PROJECT',project),patch.object(F,'ARCHIVE',archive):
            F.prepare(root,['full','wo_pack'],method,True)
            with self.assertRaises(AssertionError):F.prepare(root,['full','wo_pack'],case_first=False)
            F.write(method,{'pack':{'selector_mode':'current'}})
            with self.assertRaises(AssertionError):F.prepare(root,['full','wo_pack'],method,True)

    def test_legacy_queue_without_new_fields_still_resumes(self):
        project,archive,root,method=self.fixture()
        with patch.object(F,'PROJECT',project),patch.object(F,'ARCHIVE',archive):
            p=F.prepare(root,['full'])
            for name in ['method_configuration','effective_pack','execution_order','case_first']:p.pop(name)
            F.write(root/'queue_protocol.json',p)
            self.assertEqual(F.prepare(root,['full']),p)

    def test_original_prefix_accepts_append_but_rejects_rewrite(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);path=Path(temp.name)/'progress'
        path.write_bytes(b'preserved\n');item=dict(path=str(path),bytes=path.stat().st_size,sha256=N.sha(path))
        protocol={'preserved_prefixes':[item],'immutable_previous_files':[]}
        with path.open('ab') as stream:stream.write(b'resumed\n')
        N.verify_preserved(protocol)
        path.write_bytes(b'rewritten\n')
        with self.assertRaises(AssertionError):N.verify_preserved(protocol)


if __name__=='__main__':unittest.main()

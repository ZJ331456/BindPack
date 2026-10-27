"""Transactional source promotion safeguards, using temporary source trees."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

PROJECT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('component_promotion',PROJECT/'scripts/finalize_component_diagnostic.py')
promotion=importlib.util.module_from_spec(spec);spec.loader.exec_module(promotion)


class PromotionTests(unittest.TestCase):
    def fixture(self,folder):
        project=Path(folder)/'project';output=Path(folder)/'output'
        files=['src/bindpack/stages/pack/stage.py','src/bindpack/stages/plan/dependency_planner.py',
               'src/bindpack/stages/route/path_set_consolidation.py','src/bindpack/prompts/templates/rag_qa_hover.py',
               'src/bindpack/configuration/defaults.json']
        for name in files:
            target=project/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(PROJECT/name,target)
        scripts=output/'source_snapshot/scripts';scripts.mkdir(parents=True)
        shutil.copy2(PROJECT/'scripts/diagnostic_improvements.py',scripts/'diagnostic_improvements.py')
        hashes={name:promotion.sha(project/name) for name in files if name.endswith('.py')}
        promotion.write(output/'production_base_hashes.json',hashes)
        return project,output,{name:(project/name).read_text() for name in files}

    def test_selected_functions_are_copied_without_failed_candidates(self):
        with TemporaryDirectory() as folder:
            project,out,before=self.fixture(folder)
            with patch.object(promotion,'PROJECT',project),patch.object(promotion.subprocess,'run'):
                r=promotion.promote(out,dict(features=['coverage_facts','plan_repair','route_fusion','hover_reader']))
            self.assertEqual(r['status'],'applied_and_verified')
            text=(project/'src/bindpack/stages/pack/selection_improvements.py').read_text()
            self.assertIn('class FactCoverageSelector',text)
            self.assertNotIn('def complete_chain_package',text)
            planner=(project/'src/bindpack/stages/plan/dependency_planner.py').read_text()
            self.assertIn('nodes, recovery = recover_plan',planner)
            self.assertEqual((project/'src/bindpack/configuration/defaults.json').read_text(),before['src/bindpack/configuration/defaults.json'])

    def test_concurrent_edit_is_never_overwritten(self):
        with TemporaryDirectory() as folder:
            project,out,before=self.fixture(folder)
            path=project/'src/bindpack/stages/pack/stage.py';path.write_text(path.read_text()+'\n# User edit\n')
            changed=path.read_text()
            with patch.object(promotion,'PROJECT',project):
                r=promotion.promote(out,dict(features=['coverage_facts']))
            self.assertEqual(r['status'],'not_applied_concurrent_source_changes')
            self.assertEqual(path.read_text(),changed)
            self.assertFalse((project/'src/bindpack/stages/pack/selection_improvements.py').exists())

    def test_parity_failure_restores_old_files_and_removes_only_new_helpers(self):
        with TemporaryDirectory() as folder:
            project,out,before=self.fixture(folder)
            with patch.object(promotion,'PROJECT',project),patch.object(promotion.subprocess,'run',
                    side_effect=subprocess.CalledProcessError(1,['parity'])):
                r=promotion.promote(out,dict(features=['coverage_facts','chain_package','plan_repair']))
            self.assertEqual(r['status'],'rolled_back_parity_failure')
            for name,text in before.items():self.assertEqual((project/name).read_text(),text)
            self.assertFalse((project/'src/bindpack/stages/pack/selection_improvements.py').exists())
            self.assertFalse((project/'src/bindpack/stages/plan/plan_recovery.py').exists())


if __name__=='__main__':unittest.main()

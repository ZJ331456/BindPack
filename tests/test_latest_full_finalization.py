"""A completed summary must match all four independently validated full cases."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/finalize_bindpack_latest_full.py'
spec=importlib.util.spec_from_file_location('latest_full_summary',SCRIPT)
F=importlib.util.module_from_spec(spec);spec.loader.exec_module(F)


class FinalizationTests(unittest.TestCase):
    def fixture(self,count=4):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);root=Path(tmp.name)
        datasets=['hotpotqa','2wikimultihopqa','musique','hover']
        (root/'source_snapshot').mkdir();(root/'source_snapshot/native.py').write_text('frozen source')
        (root/'data.json').write_text('[]');(root/'archive/shared_indexes').mkdir(parents=True)
        (root/'archive/manifest.json').write_text('{}')
        F.write(root/'queue_protocol.json',dict(datasets=datasets,cases=['full'],expected_queries=4000,
            source_sha256={'native.py':F.sha(root/'source_snapshot/native.py')},
            frozen_inputs={'data':{'path':str(root/'data.json'),'sha256':F.sha(root/'data.json')}},
            index_source=str(root/'archive/shared_indexes'),archive_manifest_sha256=F.sha(root/'archive/manifest.json')))
        F.write(root/'queue_status.json',{'status':'complete'});F.write(root/'completed.ok',{'queries':4000})
        for ds in datasets[:count]:
            c=root/'cases'/ds/'full';c.mkdir(parents=True)
            metrics={k:1. for k in ('Recall@5','FCR@5','ExactMatch','F1','ClassificationAccuracy','ClassificationMacroF1')}
            metrics.update({'DCC@5':None,'NodeCov@5':None,'ClosureGap@5':None})
            F.write(c/'report.json',dict(coverage={'sample_count':1000},metrics=metrics,
                efficiency={'query_pipeline_seconds':{'mean':1.}},
                cost={'llm_logical_calls':1000,'input_tokens_known_total':2000,'output_tokens_known_total':3000}))
            (c/'result.json').write_text('{}');(c/'llm_calls.jsonl').write_text('')
            F.write(c/'completed.ok',dict(query_count=1000,report_sha256=F.sha(c/'report.json'),
                result_sha256=F.sha(c/'result.json'),llm_calls_sha256=F.sha(c/'llm_calls.jsonl')))
            F.write(c/'audit.json',{'status':'passed'});F.write(c/'index_reuse_verified.ok',{'source_and_private_unchanged':True})
            F.write(c/'private_assets_released.ok',{'source_unchanged':True})
        return root

    def test_completed_summary_cannot_omit_a_dataset(self):
        with self.assertRaises(AssertionError):F.finalize(self.fixture(count=3))

    def test_complete_status_and_all_cases_are_published(self):
        root=self.fixture();F.write(root/'summary.json',{'status':'running'})
        F.finalize(root)
        summary=F.read(root/'summary.json')
        self.assertEqual(summary['status'],'complete');self.assertEqual(summary['completed_queries'],4000)
        self.assertTrue((root/'results_at5.csv').exists())

    def test_changed_report_is_rejected(self):
        root=self.fixture();(root/'cases/hotpotqa/full/report.json').write_text('{}')
        with self.assertRaises((AssertionError,KeyError)):F.finalize(root)


if __name__=='__main__':unittest.main()

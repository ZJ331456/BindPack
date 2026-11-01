"""Larger execution windows keep checkpoints, order and accounting honest."""
import tempfile
import unittest
from pathlib import Path
from threading import Event

from bindpack import BindPackConfig
from bindpack.evaluation.measurement import Recorder
from bindpack.utils.misc_utils import QuerySolution
from utils.window_execution import iter_windows, augment_accounting


class Rag:
    def __init__(self): self.calls=[]
    def set_query_hop_overrides(self,hops,source): pass
    def retrieve(self,queries,num_to_retrieve):
        self.calls.append(list(queries))
        return [QuerySolution(question=q,docs=[q],doc_scores=[1.]) for q in queries]


class ParallelExecutionTests(unittest.TestCase):
    def test_default_stays_eight_and_explicit_larger_bounds_are_validated(self):
        self.assertEqual(BindPackConfig().llm_prefetch_workers,8)
        for n in (16,32):
            self.assertEqual(BindPackConfig(llm_prefetch_workers=n).llm_prefetch_workers,n)
        for n in (0,33):
            with self.assertRaises(ValueError):BindPackConfig(llm_prefetch_workers=n)

    def test_sixteen_query_window_can_resume_its_pending_tail_without_rereading_it(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);rec=Recorder(root);stop=Event();rag=Rag()
            q=[f'q{i}' for i in range(19)]
            stream=iter_windows(rag,root,0,q,[2]*19,q,rec,stop,16)
            first=next(stream);self.assertEqual(first[2]['execution_mode'],'native_window16')
            stop.set()
            with self.assertRaises(KeyboardInterrupt):next(stream)
            stop.clear();resumed=Rag()
            remaining=list(iter_windows(resumed,root,1,q,[2]*19,q,rec,stop,16))
            self.assertEqual(resumed.calls,[q[16:]])
            self.assertEqual([row[0] for row in remaining],list(range(1,19)))
            self.assertTrue(all(row[2]['execution_mode']=='native_window16' for row in remaining))

    def test_large_window_accounting_is_throughput_not_isolated_latency(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'retrieval_windows.jsonl').write_text('{"window_id":"w","retrieval_seconds":32}\n')
            rows=[{'timing':{'execution_mode':'native_window16','reader_seconds':0}} for _ in range(16)]
            profile={'efficiency':{'query_pipeline_seconds':{'mean':None},'retrieval_seconds':{'mean':None}},'measurement_protocol':{}}
            augment_accounting(profile,root,rows)
            self.assertEqual(profile['efficiency']['query_concurrency'],16)
            self.assertEqual(profile['efficiency']['processing_seconds_per_query'],2.)
            self.assertIsNone(profile['efficiency']['query_pipeline_seconds']['mean'])


if __name__=='__main__':unittest.main()

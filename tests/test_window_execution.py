"""Native windows preserve order, completed prefixes, pending work and cost scopes."""
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace
import tempfile
import unittest

from bindpack.evaluation.measurement import Recorder,write_json,jsonl
from bindpack.utils.misc_utils import QuerySolution
from utils.window_execution import iter_windows,freeze_protocol,augment_accounting


class Rag:
    def __init__(self):self.calls=[];self.hops=[]
    def set_query_hop_overrides(self,hops,source):self.hops.append(hops)
    def retrieve(self,queries,num_to_retrieve):
        self.calls.append(list(queries))
        return [QuerySolution(question=q,docs=[q+' passage'],doc_scores=[.7],retrieval_trace={'evidence':{'query':q}}) for q in queries]


class WindowTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.rec=Recorder(self.root);self.stop=Event()
        self.q=[f'q{i}' for i in range(11)];self.ids=[f'id{i}' for i in range(11)]

    def stream(self,rag,start=0):
        return iter_windows(rag,self.root,start,self.q,[2]*11,self.ids,self.rec,self.stop,8)

    def test_native_windows_are_eight_then_three_and_keep_physical_outputs(self):
        rag=Rag();rows=list(self.stream(rag))
        self.assertEqual([len(q) for q in rag.calls],[8,3])
        self.assertEqual([i for i,_,_ in rows],list(range(11)))
        self.assertEqual([s.docs for _,s,_ in rows],[[q+' passage'] for q in self.q])
        self.assertEqual(rag.hops,[[2]*8,[2]*3])
        self.assertFalse((self.root/'pending_retrieval_window.json').exists())
        self.assertEqual(len(jsonl(self.root/'retrieval_windows.jsonl')),2)

    def test_resume_never_retrieves_completed_queries(self):
        rag=Rag();rows=list(self.stream(rag,9))
        self.assertEqual(rag.calls,[self.q[9:]])
        self.assertEqual([i for i,_,_ in rows],[9,10])

    def test_pending_window_survives_stop_and_avoids_duplicate_retrieval(self):
        rag=Rag();stream=self.stream(rag);first=next(stream)
        self.assertEqual(first[0],0);self.stop.set()
        with self.assertRaises(KeyboardInterrupt):next(stream)
        self.stop.clear();resumed=Rag();rows=list(self.stream(resumed,1))
        self.assertEqual(resumed.calls,[self.q[8:]])
        self.assertEqual([i for i,_,_ in rows],list(range(1,11)))
        self.assertEqual(len(jsonl(self.root/'retrieval_windows.jsonl')),2)

    def test_protocol_keeps_immutable_original_and_rejects_model_changes(self):
        old={'source':{'old':'hash'},'query_concurrency':1,'configuration':{'model':'same'}}
        write_json(self.root/'protocol.json',old)
        new=dict(old,source={'new':'hash'},query_concurrency=8)
        base,active=freeze_protocol(self.root,new,write_json)
        self.assertEqual(json.loads((self.root/'protocol.json').read_text()),old)
        self.assertEqual(base,old);self.assertEqual(active['query_concurrency'],8)
        with self.assertRaises(ValueError):freeze_protocol(self.root,dict(new,configuration={'model':'changed'}),write_json)

    def test_window_is_counted_once_and_amortization_is_not_single_query_latency(self):
        write_json(self.root/'retrieval_windows.jsonl',{})
        (self.root/'retrieval_windows.jsonl').write_text(json.dumps({'window_id':'w','retrieval_seconds':16})+'\n')
        rows=[{'timing':{'retrieval_seconds':10,'end_to_end_seconds':11,'reader_seconds':1}}]
        rows += [{'timing':{'execution_mode':'native_window8','retrieval_seconds':None,'end_to_end_seconds':None,'reader_seconds':1}} for _ in range(8)]
        profile={'efficiency':{'query_pipeline_seconds':{'mean':11},'retrieval_seconds':{'mean':10}},'measurement_protocol':{}}
        augment_accounting(profile,self.root,rows)
        self.assertEqual(profile['efficiency']['completed_retrieval_wall_seconds'],26)
        self.assertIsNone(profile['efficiency']['query_pipeline_seconds']['mean'])
        self.assertEqual(profile['efficiency']['serial_query_pipeline_seconds']['mean'],11)
        self.assertEqual(profile['efficiency']['execution_segments']['native_window_queries'],8)


if __name__=='__main__':unittest.main()

"""Model-specific query protocol and concurrent Reader checkpoint guarantees."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from bindpack.embedding_model import _get_embedding_model_class
from bindpack.embedding_model.BGE import BGEM3EmbeddingModel
from bindpack.embedding_model.Transformers import TransformersEmbeddingModel
from bindpack.evaluation.measurement import Recorder,jsonl,write_json
from utils.parallel_reader import iter_readers,augment_reader_accounting


class ModelFactorTests(unittest.TestCase):
    def test_bge_factory_and_raw_query_protocol(self):
        self.assertIs(_get_embedding_model_class('/root/models/bge-m3'),BGEM3EmbeddingModel)
        self.assertIs(_get_embedding_model_class('/root/models/Qwen3-Embedding-8B'),TransformersEmbeddingModel)
        embedding=BGEM3EmbeddingModel.__new__(BGEM3EmbeddingModel)
        embedding.normalize=True
        with patch.object(TransformersEmbeddingModel,'_encode',return_value=np.array([[3.,4.]],dtype=np.float32)) as encode:
            np.testing.assert_allclose(embedding._encode(['Who wrote X?'],instruction='Find a passage'),[[.6,.8]])
            encode.assert_called_once_with(['Who wrote X?'],instruction='')

    def test_thread_tags_do_not_leak_between_readers_or_retrieval(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder=Recorder(directory);barrier=threading.Barrier(4)
            recorder.phase='online_retrieval';recorder.query_id='window'
            def work(i):
                with recorder.thread_section('online_reader',str(i)):
                    barrier.wait(timeout=5)
                    self.assertEqual(recorder.tags(),{'phase':'online_reader','query_id':str(i)})
                return recorder.tags()
            with ThreadPoolExecutor(max_workers=4) as pool:
                values=list(pool.map(work,range(4)))
            self.assertTrue(all(value=={'phase':'online_retrieval','query_id':'window'} for value in values))

    def test_concurrent_reader_orders_results_and_resumes_without_model_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            case=Path(directory);recorder=Recorder(case);ids=['a','b','c','d'];calls=[]
            barrier=threading.Barrier(4)
            write_json(case/'pending_retrieval_window.json',dict(start=0,end=4,window_id='w'))
            def qa(solutions):
                query=solutions[0];calls.append(query.question);barrier.wait(timeout=5)
                self.assertEqual(recorder.tags(),{'phase':'online_reader','query_id':query.question})
                query.answer='answer:'+query.question
                return [query],['raw:'+query.question],[{'finish_reason':'stop'}]
            rag=SimpleNamespace(qa=qa)
            def stream(start=0):
                for i in range(start,4):
                    yield i,SimpleNamespace(question=ids[i]),dict(window_id='w',execution_mode='native_window8')
            def prepare(solution):return solution,{'prompt_sha256':solution.question}
            first=iter_readers(stream(),rag,case,ids,recorder,threading.Event(),4,prepare)
            self.assertEqual(next(first)[0],0);first.close()
            self.assertEqual(sorted(calls),ids)
            resumed=list(iter_readers(stream(1),rag,case,ids,recorder,threading.Event(),4,prepare))
            self.assertEqual([r[0] for r in resumed],[1,2,3]);self.assertEqual(len(calls),4)
            self.assertEqual(resumed[0][2]['_prepared_reader']['answer'],'answer:b')
            self.assertFalse((case/'pending_reader_window.json').exists())
            windows=jsonl(case/'reader_windows.jsonl');self.assertEqual(len(windows),1)
            profile={'efficiency':{'reader_seconds':{},'completed_retrieval_wall_seconds':10},'measurement_protocol':{}}
            rows=[{'timing':{'reader_window_id':'w'}} for _ in ids]
            augment_reader_accounting(profile,case,rows,4)
            self.assertAlmostEqual(profile['efficiency']['completed_reader_wall_seconds'],windows[0]['seconds'])
            self.assertAlmostEqual(profile['efficiency']['processing_seconds_per_query'],(10+windows[0]['seconds'])/4)


if __name__=='__main__':unittest.main()

"""CUDA scoring/PageRank preserves the native weighted-walk definition."""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import igraph as ig
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from bindpack.retrieval.cuda_index import CudaRetrievalIndex
from bindpack.core.base import BaseRAG
from bindpack.utils.misc_utils import QuerySolution


@unittest.skipUnless(torch.cuda.is_available(),'CUDA device required')
class CudaRetrievalTests(unittest.TestCase):
    def graph(self,directed=False):
        graph=ig.Graph(n=6,edges=[(0,0),(0,1),(0,1),(1,2),(2,3),(3,0),(3,4)],directed=directed)
        graph.es['weight']=[.3,1.,.4,2.,.7,.1,.9]
        return graph

    def test_weighted_loops_parallel_edges_and_dangling_match_igraph(self):
        reset=np.array([.2,0,0,0,0,.8])
        for directed in (False,True):
            graph=self.graph(directed);index=CudaRetrievalIndex(graph,{'fact':np.eye(6,dtype=np.float32)})
            for damping in (0.,.45,.85):
                cpu=graph.personalized_pagerank(reset=reset,damping=damping,directed=False,weights='weight',implementation='prpack')
                gpu=index.pagerank(reset,damping)
                np.testing.assert_allclose(gpu,cpu,rtol=1e-10,atol=1e-12)
                self.assertAlmostEqual(float(gpu.sum()),1,places=11)

    def test_full_float_scoring_matches_numpy_for_vector_and_batch_column(self):
        rng=np.random.default_rng(831);matrix=rng.normal(size=(256,128)).astype(np.float32)
        query=rng.normal(size=128).astype(np.float32)
        index=CudaRetrievalIndex(self.graph(),{'fact':matrix})
        for q in [query,query[None,:]]:
            expected=matrix@q.T;actual=index.scores('fact',q)
            np.testing.assert_allclose(actual,expected,rtol=3e-5,atol=1e-5)
            np.testing.assert_array_equal(np.argsort(actual.ravel()),np.argsort(expected.ravel()))

    def test_invalid_reset_and_weights_are_not_silently_replaced(self):
        graph=self.graph();index=CudaRetrievalIndex(graph,{})
        with self.assertRaises(ValueError):index.pagerank(np.zeros(6))
        graph.es['weight'][0]=-1
        graph.es[0]['weight']=-1
        with self.assertRaises(ValueError):CudaRetrievalIndex(graph,{})


class ReaderCapTests(unittest.TestCase):
    def test_length_response_is_parsed_once_with_explicit_reader_cap(self):
        infer=Mock(return_value=('Answer: partial answer',{'finish_reason':'length'},False))
        manager=SimpleNamespace(is_template_name_valid=lambda name:True,
            render=lambda **kwargs:[{'role':'user','content':kwargs['prompt_user']}])
        rag=SimpleNamespace(global_config=SimpleNamespace(dataset='musique',qa_top_k=5),
            llm_model=SimpleNamespace(infer=infer),prompt_template_manager=manager)
        answered,raw,metadata=BaseRAG.qa(rag,[QuerySolution(question='Question?',docs=['Title\nFact.'])],max_output_tokens=2028)
        self.assertEqual(infer.call_count,1)
        self.assertEqual(infer.call_args.kwargs,{'max_completion_tokens':2028})
        self.assertEqual(answered[0].answer,'partial answer')
        self.assertEqual(raw,['Answer: partial answer'])
        self.assertEqual(metadata[0]['finish_reason'],'length')


if __name__=='__main__':unittest.main()

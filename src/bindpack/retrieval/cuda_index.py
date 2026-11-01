"""Opt-in CUDA vector scoring and weighted undirected personalized PageRank."""
from __future__ import annotations
import time
import numpy as np


class CudaRetrievalIndex:
    def __init__(self,graph,matrices):
        import torch
        if not torch.cuda.is_available():raise RuntimeError('CUDA retrieval requested but CUDA is unavailable')
        torch.backends.cuda.matmul.allow_tf32=False
        self.torch=torch;self.device=torch.device('cuda')
        self.matrices={name:torch.as_tensor(matrix,dtype=torch.float32,device=self.device)
                       for name,matrix in matrices.items()}
        self.nodes=graph.vcount();self.edges=graph.ecount()
        self._build_graph(graph)
        self.ppr_calls=0;self.ppr_seconds=0.;self.max_iterations=0

    def _build_graph(self,graph):
        import scipy.sparse as sp
        torch=self.torch;n=graph.vcount()
        edges=np.asarray(graph.get_edgelist(),dtype=np.int64).reshape(-1,2)
        weights=np.asarray(graph.es['weight'],dtype=np.float64)
        if not np.isfinite(weights).all() or (weights<0).any():raise ValueError('Invalid PageRank edge weights')
        # directed=False in the native igraph call treats every edge as an
        # undirected connection. A self-loop contributes twice to strength.
        src=np.concatenate([edges[:,0],edges[:,1]])
        dst=np.concatenate([edges[:,1],edges[:,0]])
        values=np.concatenate([weights,weights])
        strength=np.bincount(src,weights=values,minlength=n)
        positive=strength[src]>0
        transition=sp.coo_matrix((values[positive]/strength[src[positive]],
            (dst[positive],src[positive])),shape=(n,n),dtype=np.float64).tocsr()
        transition.sum_duplicates();transition.sort_indices()
        self.transition=torch.sparse_csr_tensor(torch.as_tensor(transition.indptr,dtype=torch.int64,device=self.device),
            torch.as_tensor(transition.indices,dtype=torch.int64,device=self.device),
            torch.as_tensor(transition.data,dtype=torch.float64,device=self.device),
            size=(n,n),device=self.device)
        self.dangling=torch.as_tensor(strength==0,device=self.device)
        self.stats=dict(device='cuda',vector_dtype='float32',graph_dtype='float64',tf32=False,
            nodes=n,original_edges=graph.ecount(),transition_nonzeros=transition.nnz,
            vector_bytes=sum(t.numel()*t.element_size() for t in self.matrices.values()),
            graph_bytes=sum(t.numel()*t.element_size() for t in
                (self.transition.crow_indices(),self.transition.col_indices(),self.transition.values())),
            graph_policy='same weighted directed=False walk as igraph PRPACK; reset-distributed dangling mass; L1 tolerance1e-13')

    def scores(self,name,query):
        torch=self.torch
        with torch.inference_mode():
            vector=torch.as_tensor(np.asarray(query,dtype=np.float32),device=self.device)
            operand=vector if vector.ndim==1 else vector.T
            return (self.matrices[name]@operand).cpu().numpy()

    def pagerank(self,reset,damping=.5):
        torch=self.torch;began=time.perf_counter()
        value=np.asarray(reset,dtype=np.float64)
        value=np.where(np.isnan(value)|(value<0),0,value)
        if value.shape!=(self.nodes,) or not np.isfinite(value).all() or value.sum()<=0:
            raise ValueError('Invalid personalized PageRank reset vector')
        if not 0<=damping<1:raise ValueError('PageRank damping must lie in [0,1)')
        with torch.inference_mode():
            reset_gpu=torch.as_tensor(value/value.sum(),dtype=torch.float64,device=self.device)
            rank=reset_gpu.clone()
            for iteration in range(1,10001):
                moved=torch.sparse.mm(self.transition,rank[:,None]).ravel()
                updated=damping*moved+(1-damping+damping*rank[self.dangling].sum())*reset_gpu
                change=torch.sum(torch.abs(updated-rank)).item()
                rank=updated
                if change<1e-13:break
            else:raise RuntimeError('GPU PageRank failed its convergence requirement')
            result=rank.cpu().numpy()
        self.ppr_calls+=1;self.ppr_seconds+=time.perf_counter()-began
        self.max_iterations=max(self.max_iterations,iteration)
        return result

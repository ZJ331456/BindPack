"""Local BGE-M3 dense embeddings with the model's CLS pooling configuration."""
import numpy as np
from .Transformers import TransformersEmbeddingModel


class BGEM3EmbeddingModel(TransformersEmbeddingModel):
    # BGE-M3 takes raw queries; Hippo/Qwen task instructions are not BGE input.
    uses_query_instruction = False

    def _encode(self, texts, instruction=""):
        vectors=np.asarray(super()._encode(texts, instruction=""),dtype=np.float32)
        if self.normalize:
            # BF16 division leaves norms up to ~0.4% away from one. Normalize
            # the stored/query vectors in FP32 after the dense forward pass.
            norms=np.linalg.norm(vectors,axis=1,keepdims=True)
            if not np.isfinite(vectors).all() or np.any(norms==0):
                raise ValueError('BGE-M3 produced invalid dense embeddings')
            vectors=vectors/norms
        return vectors

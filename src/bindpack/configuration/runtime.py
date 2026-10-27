"""Shared model, index, storage and evaluation runtime settings."""
import os
import json
from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal, Union, Optional
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)
_RUNTIME_DEFAULTS = json.loads(Path(__file__).with_name("defaults.json").read_text(encoding="utf-8"))["runtime"]


@dataclass
class RuntimeConfig:
    mechanism_ablation: str = _RUNTIME_DEFAULTS["mechanism_ablation"]
    llm_name: str = _RUNTIME_DEFAULTS["llm_name"]
    llm_base_url: Optional[str] = _RUNTIME_DEFAULTS["llm_base_url"]
    embedding_base_url: Optional[str] = _RUNTIME_DEFAULTS["embedding_base_url"]
    azure_endpoint: Optional[str] = _RUNTIME_DEFAULTS["azure_endpoint"]
    azure_embedding_endpoint: Optional[str] = _RUNTIME_DEFAULTS["azure_embedding_endpoint"]
    max_new_tokens: Union[None, int] = _RUNTIME_DEFAULTS["max_new_tokens"]
    num_gen_choices: int = _RUNTIME_DEFAULTS["num_gen_choices"]
    seed: Union[None, int] = _RUNTIME_DEFAULTS["seed"]
    temperature: float = _RUNTIME_DEFAULTS["temperature"]
    response_format: Union[dict, None] = field(default_factory=lambda: deepcopy(_RUNTIME_DEFAULTS["response_format"]))
    max_retry_attempts: int = _RUNTIME_DEFAULTS["max_retry_attempts"]
    force_openie_from_scratch: bool = _RUNTIME_DEFAULTS["force_openie_from_scratch"]
    force_index_from_scratch: bool = _RUNTIME_DEFAULTS["force_index_from_scratch"]
    rerank_dspy_file_path: Optional[str] = _RUNTIME_DEFAULTS["rerank_dspy_file_path"]
    passage_node_weight: float = _RUNTIME_DEFAULTS["passage_node_weight"]
    save_openie: bool = _RUNTIME_DEFAULTS["save_openie"]
    openie_mode: Literal['offline', 'online'] = _RUNTIME_DEFAULTS["openie_mode"]
    openie_max_workers: Optional[int] = _RUNTIME_DEFAULTS["openie_max_workers"]
    openie_strict: bool = _RUNTIME_DEFAULTS["openie_strict"]
    openie_prompt_version: Literal['origin', 'optimized'] = _RUNTIME_DEFAULTS["openie_prompt_version"]
    openie_validation_mode: Literal['structural', 'source_verified'] = _RUNTIME_DEFAULTS["openie_validation_mode"]
    embedding_model_name: str = _RUNTIME_DEFAULTS["embedding_model_name"]
    embedding_batch_size: int = _RUNTIME_DEFAULTS["embedding_batch_size"]
    embedding_return_as_normalized: bool = _RUNTIME_DEFAULTS["embedding_return_as_normalized"]
    embedding_max_seq_len: int = _RUNTIME_DEFAULTS["embedding_max_seq_len"]
    embedding_model_dtype: Literal['float16', 'float32', 'bfloat16', 'auto'] = _RUNTIME_DEFAULTS["embedding_model_dtype"]
    synonymy_edge_topk: int = _RUNTIME_DEFAULTS["synonymy_edge_topk"]
    synonymy_edge_query_batch_size: int = _RUNTIME_DEFAULTS["synonymy_edge_query_batch_size"]
    synonymy_edge_key_batch_size: int = _RUNTIME_DEFAULTS["synonymy_edge_key_batch_size"]
    synonymy_edge_sim_threshold: float = _RUNTIME_DEFAULTS["synonymy_edge_sim_threshold"]
    is_directed_graph: bool = _RUNTIME_DEFAULTS["is_directed_graph"]
    linking_top_k: int = _RUNTIME_DEFAULTS["linking_top_k"]
    retrieval_top_k: int = _RUNTIME_DEFAULTS["retrieval_top_k"]
    damping: float = _RUNTIME_DEFAULTS["damping"]
    max_qa_steps: int = _RUNTIME_DEFAULTS["max_qa_steps"]
    qa_top_k: int = _RUNTIME_DEFAULTS["qa_top_k"]
    save_dir: Optional[str] = _RUNTIME_DEFAULTS["save_dir"]
    dataset: Optional[Literal['hotpotqa', 'hotpotqa_train', 'musique', '2wikimultihopqa', 'hover', 'nq', 'popqa']] = _RUNTIME_DEFAULTS["dataset"]
    corpus_len: Optional[int] = _RUNTIME_DEFAULTS["corpus_len"]
    llm_prefetch_workers: int = _RUNTIME_DEFAULTS["llm_prefetch_workers"]

    def __post_init__(self):
        from .validation import validate_runtime
        validate_runtime(self)
        if self.mechanism_ablation not in {
            "full", "wo_source_bound_eligibility", "wo_dependency_closure", "wo_joint_selection"
        }:
            raise ValueError("Unknown mechanism_ablation")
        if not isinstance(self.openie_strict, bool):
            raise ValueError('openie_strict must be a bool')
        if self.openie_prompt_version not in ('origin', 'optimized'):
            raise ValueError('openie_prompt_version must be origin or optimized')
        if self.openie_validation_mode not in ('structural', 'source_verified'):
            raise ValueError('openie_validation_mode must be structural or source_verified')
        if self.openie_max_workers is not None and (isinstance(self.openie_max_workers, bool) or not isinstance(self.openie_max_workers, int) or (not 1 <= self.openie_max_workers <= 8)):
            raise ValueError('openie_max_workers must be an integer between 1 and 8, or None.')
        if self.save_dir is None:
            if self.dataset is None:
                self.save_dir = 'outputs'
            else:
                self.save_dir = os.path.join('outputs', self.dataset)
        logger.debug(f'Initializing the highest level of save_dir to be {self.save_dir}')

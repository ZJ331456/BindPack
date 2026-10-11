"""Observe real pre-selection inputs without modifying production decisions."""
from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from functools import wraps
import gzip
import hashlib
import json
from pathlib import Path
import threading

import numpy as np

from ..evidence.evidence_retrieval import EvidenceRetrieval


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, set):
        return sorted(json_value(v) for v in value)
    return value


def merged_scores(runtime, ids, scores, state):
    """Actual score before greedy-prefix rank encoding, matching production."""
    finite = np.where(np.isfinite(scores), scores, 0.0)
    span = float(finite.max() - finite.min()) if len(finite) else 0.0
    norm = (finite - finite.min()) / span if span > 0 else np.ones(len(finite))
    base = dict(zip(map(int, ids), map(float, norm)))
    weight = float(runtime.cfg.evidence_base_weight)
    merged = {doc_id: weight * value for doc_id, value in base.items()}
    bindings = state.get('_evidence_winning_bindings', {})
    for doc_id, candidate in state.get('evidence_candidates', {}).items():
        goals = runtime._goal_scores(candidate, bindings)
        merged[int(doc_id)] = (weight * base.get(int(doc_id), candidate['base_score'])
                               + (1.0 - weight) * max(goals.values(), default=0.0))
    # The no-candidate fallback has no coverage selector or rank encoding.
    return merged if state.get('evidence_candidates') else dict(zip(map(int, ids), map(float, scores)))


class CaptureSession:
    """Hooks are installed only inside an isolated experimental worker."""

    def __init__(self, rag, directory, ledger, selected_indices=None):
        self.rag, self.ledger = rag, ledger
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.query_indices = {}
        self.selected_indices = selected_indices
        self.verification_parents = defaultdict(deque)
        self.saved = []
        self.retrieval_counts = defaultdict(list)
        self.count_lock = threading.Lock()
        self.original_finalize = EvidenceRetrieval.finalize

    def register_queries(self, queries):
        if len(set(queries)) != len(queries):
            raise ValueError('Capture requires unique questions for exact request attribution')
        self.query_indices = {q: i for i, q in enumerate(queries)}

    def resolve_messages(self, messages):
        text = '\n'.join(str(m.get('content', '')) for m in messages)
        matches = [q for q in self.query_indices if q in text]
        if not matches:
            raise ValueError('LLM request has no attributable original query')
        return max(matches, key=len), 'final_review'

    def _scope_method(self, owner, name, stage):
        original = getattr(owner, name)

        @wraps(original)
        def observed(*args, **kwargs):
            query = kwargs.get('query') or args[0]
            with self.ledger.scope(query, stage):
                return original(*args, **kwargs)
        setattr(owner, name, observed)

    def install(self):
        self.ledger.resolver = self.resolve_messages
        original_infer = self.rag.llm_model.infer
        self.rag.llm_model.infer = self.ledger.wrap_infer(original_infer)
        # DSPyFilter stores a bound infer callable during construction.
        self.rag.rerank_filter.llm_infer_fn = self.rag.llm_model.infer
        for name, stage in (('rerank_facts', 'shared_base'),
                            ('_decompose_query', 'static_qd'),
                            ('_decompose_query_with_hints', 'pcqd'),
                            ('_path_graph_search', 'shared_base_retrieval'),
                            ('_build_qd_track_ranking', 'qd_retrieval'),
                            ('_query_decomposition_retrieval', 'qd_retrieval')):
            self._scope_method(self.rag, name, stage)
        original_dense, original_ppr = self.rag.dense_passage_retrieval, self.rag.run_ppr

        def count(kind, query=None):
            owner = self.ledger.current_owner()
            if owner is None and query in self.query_indices:
                owner = (query, 'shared_base_retrieval')
            if owner is None:
                raise ValueError('Retrieval request has no attributable original query')
            with self.count_lock:
                self.retrieval_counts[owner[0]].append({'kind': kind, 'stage': owner[1]})

        def dense(query):
            count('dense', query)
            return original_dense(query)

        def ppr(*args, **kwargs):
            count('ppr')
            return original_ppr(*args, **kwargs)
        self.rag.dense_passage_retrieval, self.rag.run_ppr = dense, ppr
        self._scope_method(self.rag.evidence_runtime, '_plan', 'dependency_plan')
        runtime = self.rag.evidence_runtime
        original_route = runtime._add_route

        @wraps(original_route)
        def route(state, source, goal, query, requirements=None):
            with self.ledger.scope(state['query'], 'dependency_search'):
                result = original_route(state, source, goal, query, requirements)
            if source.startswith('dag:') and result:
                self.verification_parents[(query, tuple(result[:3]))].append(state['query'])
            return result
        runtime._add_route = route
        original_jobs = runtime._collect_jobs

        def jobs(executor, tasks):
            observed = []
            for func, args in tasks:
                if getattr(func, '__name__', '') == '_verify':
                    key = (args[0], tuple(args[2]))
                    if not self.verification_parents[key]:
                        raise ValueError('Verification lost its parent query')
                    query = self.verification_parents[key].popleft()

                    def execute(*values, _func=func, _query=query):
                        with self.ledger.scope(_query, 'dependency_verification'):
                            return _func(*values)
                    observed.append((execute, args))
                else:
                    observed.append((func, args))
            return original_jobs(executor, observed)
        runtime._collect_jobs = jobs
        session = self

        def finalize(instance, query, ids, scores, ctx, state):
            result = session.original_finalize(instance, query, ids, scores, ctx, state)
            session.save(instance, query, ids, scores, state, result[0])
            return result
        EvidenceRetrieval.finalize = finalize

    def save(self, runtime, query, ids, scores, state, parent_ids):
        local_index = self.query_indices[query]
        index = (self.selected_indices[local_index] if self.selected_indices is not None
                 else local_index)
        order = [int(d) for d in parent_ids[:200]]
        source = merged_scores(runtime, np.asarray(ids), np.asarray(scores, dtype=float), state)
        kept = {key: deepcopy(state[key]) for key in (
            '_evidence_plan', '_evidence_winning_bindings', '_evidence_beams',
            'evidence_candidates', 'evidence_trace') if key in state}
        base = state.get('base', (ids, scores, {}))
        snapshot = json_value({
            'schema_version': 1, 'query': query, 'query_index': index,
            'candidate_ids': order, 'candidate_scores': [source[d] for d in order],
            'score_source': 'pre_selection_merged',
            'ranking_scope': 'real_greedy_parent_before_dag_package',
            'state': kept, 'documents': {d: runtime._document(d) for d in order},
            'common_base_ids': list(base[0][:200]),
            'common_base_scores': list(base[1][:200]),
            'hop': int(base[2].get('hops', self.rag._query_hop_overrides[local_index])),
            'retrieval_calls': self.retrieval_counts[query],
        })
        blob = json.dumps(snapshot, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
        target = self.directory / f'{index:05d}.json.gz'
        with gzip.open(target, 'wb') as stream:
            stream.write(blob)
        self.saved.append({'query_index': index, 'file': target.name,
                           'input_sha256': hashlib.sha256(blob).hexdigest()})

    def close(self):
        EvidenceRetrieval.finalize = self.original_finalize
        for entry in self.saved:
            with gzip.open(self.directory / entry['file'], 'rb') as stream:
                entry['input_sha256'] = hashlib.sha256(stream.read()).hexdigest()
        (self.directory / 'manifest.json').write_text(json.dumps(
            {'schema_version': 1, 'snapshots': sorted(self.saved, key=lambda x: x['query_index'])},
            ensure_ascii=False, indent=2))

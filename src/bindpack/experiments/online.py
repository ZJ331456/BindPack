"""Isolated adapters for capture and equal-budget retrieval controls."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import gzip
import json
from pathlib import Path
import threading
import time

from tqdm import tqdm

from ..retrieval.engine import BindPack
from ..utils.misc_utils import QuerySolution
from ..evaluation.retrieval_eval import RetrievalRecall
from .capture import CaptureSession
from .budget_retrieval import BudgetRetrievalControl
from .request_ledger import LLMCostLedger


def read_snapshot(directory, index):
    with gzip.open(Path(directory) / f'{index:05d}.json.gz', 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def worker_class(options):
    """Build an evaluator-compatible class; production imports stay untouched."""
    class ExperimentalBindPack(BindPack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._experiment_original_infer = self.llm_model.infer
            self.experiment_ledger = LLMCostLedger()
            self.experiment_capture = None
            if options.mode in ('capture', 'no_plan_prune', 'binding_string', 'binding_relation'):
                if options.mode.startswith('binding_'):
                    from .binding_execution import install_binding_control
                    install_binding_control(self.evidence_runtime, options.mode.removeprefix('binding_'))
                self.experiment_capture = CaptureSession(
                    self, options.snapshot_dir, self.experiment_ledger, options.selected_indices)
                self.experiment_capture.install()

        def retrieve(self, queries, num_to_retrieve=None, gold_docs=None):
            if self.experiment_capture is not None:
                batches = _unique_question_batches(queries)
                if len(batches) > 1:
                    return self._capture_duplicate_batches(queries, num_to_retrieve, gold_docs, batches)
                self.experiment_capture.register_queries(queries)
                try:
                    result = super().retrieve(queries, num_to_retrieve, gold_docs)
                    solutions = result[0] if isinstance(result, tuple) else result
                    _complete_snapshots(options.snapshot_dir, options.selected_indices, solutions,
                                        self.experiment_ledger)
                    return result
                finally:
                    self.experiment_capture.close()
                    self.experiment_ledger.write_json(options.ledger_path)

            return self._budget_retrieve(queries, num_to_retrieve, gold_docs)

        def _capture_duplicate_batches(self, queries, num_to_retrieve, gold_docs, batches):
            """Keep duplicate samples while giving each batch its own cost owner."""
            selected = list(options.selected_indices)
            original_hops = list(self._query_hop_overrides or [])
            source = self._query_hop_override_source
            if len(selected) != len(queries) or len(original_hops) != len(queries):
                raise ValueError('Selected indices and hop overrides must align with every capture query')
            if gold_docs is not None and len(gold_docs) != len(queries):
                raise ValueError('Gold passages must align with every capture query')
            capture = self.experiment_capture
            all_solutions, all_records, diagnostics = [], [], []
            batch_receipts = []
            try:
                for batch_index, positions in enumerate(batches):
                    batch_queries = [queries[p] for p in positions]
                    batch_indices = [selected[p] for p in positions]
                    if batch_index:
                        previous = self.experiment_ledger
                        ledger = LLMCostLedger(tokenizer=getattr(previous, '_tokenizer', None),
                                              tokenizer_name=previous.tokenizer_name)
                        ledger.resolver = capture.resolve_messages
                        self.experiment_ledger = capture.ledger = ledger
                        # Rewrap the real cached client, never a previous ledger.
                        self.llm_model.infer = ledger.wrap_infer(self._experiment_original_infer)
                        self.rerank_filter.llm_infer_fn = self.llm_model.infer
                    ledger = self.experiment_ledger
                    capture.selected_indices = batch_indices
                    # Retrieval counters use question text inside one batch;
                    # clear them so repeated text cannot inherit old routes.
                    capture.retrieval_counts.clear()
                    capture.verification_parents.clear()
                    capture.register_queries(batch_queries)
                    self.set_query_hop_overrides([original_hops[p] for p in positions], source)
                    batch_receipts.append({'batch_index': batch_index, 'query_indices': batch_indices,
                                           'n_samples': len(positions)})
                    try:
                        # Final metrics are computed once on all original samples.
                        result = super().retrieve(batch_queries, num_to_retrieve, None)
                        solutions = result[0] if isinstance(result, tuple) else result
                        if len(solutions) != len(positions):
                            raise ValueError('Capture batch dropped a query')
                        _complete_snapshots(options.snapshot_dir, batch_indices, solutions, ledger,
                                            batch_index=batch_index)
                        for index, solution in zip(batch_indices, solutions):
                            solution.retrieval_trace = dict(solution.retrieval_trace or {},
                                experiment_capture={'split_duplicate_question_batches': True,
                                                    'query_index': index, 'batch_index': batch_index})
                        all_solutions.extend(solutions)
                        diagnostic = deepcopy(self.retrieval_diagnostics)
                        for name in ('per_query', 'no_facts_details'):
                            for row in diagnostic.get(name, []):
                                if 'query_idx' in row:
                                    row['query_idx'] = positions[row['query_idx']]
                        diagnostics.append(diagnostic)
                    finally:
                        owners = dict(zip(batch_queries, batch_indices))
                        for record in ledger.get_request_records():
                            all_records.append(dict(record, request_id=len(all_records),
                                query_index=owners[record['query']], batch_index=batch_index))
                if len(all_solutions) != len(queries):
                    raise ValueError('Capture failed to retain every original sample')
                self.retrieval_diagnostics = _merge_capture_diagnostics(diagnostics)
                self.retrieval_diagnostics['split_duplicate_question_batches'] = batch_receipts
                if gold_docs is None:
                    return all_solutions
                metrics, _ = RetrievalRecall(global_config=self.global_config).calculate_metric_scores(
                    gold_docs=gold_docs, retrieved_docs=[s.docs for s in all_solutions],
                    k_list=[1, 2, 5, 10, 20, 30, 50, 100, 150, 200])
                return all_solutions, metrics
            finally:
                self.set_query_hop_overrides(original_hops, source)
                capture.selected_indices = selected
                try:
                    capture.close()
                finally:
                    _write_query_index_ledger(options.ledger_path, all_records,
                                              self.experiment_ledger.tokenizer_name, batch_receipts)

        def _budget_retrieve(self, queries, num_to_retrieve, gold_docs):
            started = time.monotonic()
            original_infer = self.llm_model.infer
            self.llm_model.infer = self.experiment_ledger.wrap_infer(original_infer)
            original_dense = self.dense_passage_retrieval
            encoder_lock = threading.Lock()

            def dense(query):
                # Only LLM requests overlap. NV encoder calls stay serialized.
                with encoder_lock:
                    return original_dense(query)
            self.dense_passage_retrieval = dense

            def one(item):
                local_index, query = item
                index = options.selected_indices[local_index]
                snapshot = read_snapshot(options.snapshot_dir, index)
                if snapshot['query'] != query:
                    raise ValueError('Budget control query differs from frozen source')
                source_cost = snapshot['request_cost']
                base_cost = snapshot['common_base_cost']
                total_budget = max(base_cost['total_tokens'], int(source_cost['total_tokens'] * options.budget_fraction))
                token_budget = max(0, total_budget - base_cost['total_tokens'])
                request_budget = max(0, source_cost['logical_requests'] - base_cost['logical_requests'])
                shared_searches = sum(r['stage'] == 'shared_base_retrieval' for r in snapshot['retrieval_calls'])
                capture_searches = len(snapshot['retrieval_calls'])
                search_budget = max(1, capture_searches - shared_searches + 1)
                control = BudgetRetrievalControl(
                    self, options.mode, token_budget=token_budget,
                    request_budget=request_budget, search_budget=search_budget,
                    tokenizer=self.experiment_ledger.tokenizer)
                owner = f'{index}::{query}'
                with self.experiment_ledger.scope(owner, options.mode):
                    ids, scores, diagnostic = control.retrieve_one(
                        query, snapshot['common_base_ids'], snapshot['common_base_scores'],
                        snapshot['hop'])
                actual = self.experiment_ledger.get_summary(owner)
                diagnostic.update(
                    cost_owner=owner, query_index=index, cost_owner_kind='query_index_and_original_question',
                    shared_base_cost=base_cost,
                    budget_policy='per_query_cap_matched_to_observed_capture_cost',
                    budget_fraction=options.budget_fraction,
                    mandatory_common_base_budget_floor=True,
                    capture_total_token_cap=total_budget,
                    end_to_end_logical_tokens=base_cost['total_tokens'] + actual['total_tokens'],
                    common_base_retrieval_calls=shared_searches,
                    capture_total_retrieval_cap=capture_searches,
                    end_to_end_retrieval_calls=shared_searches + diagnostic['search_count'] - 1,
                    shared_base_replayed=True,
                    search_budget_scope='one_common_initial_ranking_plus_dense_control_routes',
                    baseline_kind='internal_plain_control_not_external_paper_reproduction')
                if diagnostic['end_to_end_logical_tokens'] > max(total_budget, base_cost['total_tokens']):
                    raise ValueError('Budget control exceeded its token envelope')
                if diagnostic['end_to_end_retrieval_calls'] > capture_searches:
                    raise ValueError('Budget control exceeded captured retrieval-call envelope')
                n = int(num_to_retrieve or 200)
                return QuerySolution(
                    question=query,
                    docs=[self.chunk_embedding_store.get_row(self.passage_node_keys[int(d)])['content']
                          for d in ids[:n]], doc_scores=scores[:n],
                    retrieval_trace={'budget_control': diagnostic,
                                     'assigned_hops': snapshot['hop']})

            try:
                with ThreadPoolExecutor(max_workers=8, thread_name_prefix='budget-control') as executor:
                    solutions = list(tqdm(executor.map(one, enumerate(queries)), total=len(queries),
                                          desc=f'{options.mode} budget retrieval'))
            finally:
                owners = {f'{index}::{query}': (index, query)
                          for index, query in zip(options.selected_indices, queries)}
                _write_budget_ledger(self.experiment_ledger, options.ledger_path, owners)
            self.all_retrieval_time = time.monotonic() - started
            if gold_docs is None:
                return solutions
            metrics, _ = RetrievalRecall(global_config=self.global_config).calculate_metric_scores(
                gold_docs=gold_docs, retrieved_docs=[s.docs for s in solutions],
                k_list=[1, 2, 5, 10, 20, 30, 50, 100, 150, 200])
            return solutions, metrics
    return ExperimentalBindPack


def _stage_cost(records, stage):
    rows = [r for r in records if r['stage'] == stage]
    return {'logical_requests': len(rows),
            'prompt_tokens': sum(r['prompt_tokens'] for r in rows),
            'completion_tokens': sum(r['completion_tokens'] for r in rows),
            'total_tokens': sum(r['prompt_tokens'] + r['completion_tokens'] for r in rows)}


def _unique_question_batches(queries):
    """Contiguous batches preserve input order and never discard repeated text."""
    batches, positions, seen = [], [], set()
    for position, query in enumerate(queries):
        if query in seen:
            batches.append(positions)
            positions, seen = [], set()
        positions.append(position)
        seen.add(query)
    if positions:
        batches.append(positions)
    return batches


def _complete_snapshots(directory, indices, solutions, ledger, batch_index=None):
    if len(indices) != len(solutions):
        raise ValueError('Capture snapshot indices and solutions differ')
    for index, solution in zip(indices, solutions):
        path = Path(directory) / f'{index:05d}.json.gz'
        snapshot = read_snapshot(directory, index)
        if snapshot['query'] != solution.question:
            raise ValueError('Capture snapshot question differs from its solution')
        trace = (solution.retrieval_trace or {}).get('evidence', {})
        review = trace.get('improvement_support_semantic_veto', {})
        if review.get('original_parent_doc_ids'):
            snapshot['expected_dag_ids'] = review['original_parent_doc_ids'][:200]
        snapshot['request_cost'] = ledger.get_summary(solution.question)
        snapshot['request_cost']['query_index'] = index
        snapshot['request_records'] = [dict(record, query_index=index)
            for record in ledger.get_request_records(solution.question)]
        snapshot['common_base_cost'] = _stage_cost(snapshot['request_records'], 'shared_base')
        if batch_index is not None:
            snapshot['capture_protocol'] = {'split_duplicate_question_batches': True,
                                           'batch_index': batch_index, 'query_index': index}
        with gzip.open(path, 'wt', encoding='utf-8') as stream:
            json.dump(snapshot, stream, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def _merge_capture_diagnostics(batches):
    """Merge counts, weighted means and raw diagnostic accumulators."""
    total = sum(batch.get('total_queries', 0) for batch in batches)
    merged = {}
    for batch in batches:
        weight = batch.get('total_queries', 0)
        for key, value in batch.items():
            if key == 'module_usage':
                if key in merged and merged[key] != value:
                    raise ValueError('Capture batches used different module flags')
                merged[key] = deepcopy(value)
            elif key.startswith('avg_') or key.endswith('_rate'):
                merged[key] = merged.get(key, 0.) + value * weight / max(1, total)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                merged[key] = merged.get(key, 0) + value
            elif isinstance(value, dict) and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                                for v in value.values()):
                target = merged.setdefault(key, {})
                for name, count in value.items():
                    target[name] = target.get(name, 0) + count
            elif isinstance(value, list):
                merged.setdefault(key, []).extend(deepcopy(value))
            else:
                merged[key] = deepcopy(value)
    if '_pcqd_entity_replacement_den' in merged:
        merged['pcqd_entity_replacement_rate'] = merged.get('_pcqd_entity_replacement_num', 0.) / max(
            1e-9, merged['_pcqd_entity_replacement_den'])
    for rate, numerator in (('pcqd_fallback_rate', '_pcqd_fallback_count'),
                             ('pcqd_conflict_rate', '_pcqd_conflict_sum')):
        if numerator in merged:
            merged[rate] = merged[numerator] / max(1, total)
    return merged


def _write_query_index_ledger(path, records, tokenizer_name, batches):
    """Keep duplicate questions distinct in both per-query costs and records."""
    def summary(rows):
        value = LLMCostLedger._aggregate(rows)
        value['by_stage'] = {stage: LLMCostLedger._aggregate([r for r in rows if r['stage'] == stage])
                             for stage in sorted({r['stage'] for r in rows})}
        value['cost_semantics'] = 'provider logical tokens including cached responses; not physical GPU/HTTP cost'
        return value
    indices = [i for batch in batches for i in batch['query_indices']]
    owners = {record['query_index']: record['query'] for record in records}
    queries = {}
    for index in indices:
        rows = [r for r in records if r['query_index'] == index]
        queries[str(index)] = dict(summary(rows), query_index=index, query=owners.get(index))
    payload = {'schema_version': 1, 'tokenizer': tokenizer_name, 'summary': summary(records),
               'queries': queries, 'requests': records, 'query_owner_kind': 'query_index',
               'split_duplicate_question_batches': batches}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)
    return payload


def _write_budget_ledger(ledger, path, owners):
    payload = ledger.write_json(path)
    payload['query_owner_kind'] = 'query_index_and_original_question'
    for record in payload['requests']:
        index, question = owners[record['query']]
        record.update(query_index=index, original_question=question)
    for owner, summary in payload['queries'].items():
        index, question = owners[owner]
        summary.update(query_index=index, original_question=question)
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)

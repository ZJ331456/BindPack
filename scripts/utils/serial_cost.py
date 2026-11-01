"""Sampling and audited, pooled accounting for the serial 4 x 400 benchmark."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

DATASETS = ('hotpotqa', '2wikimultihopqa', 'musique', 'hover')
SAMPLE_SIZE = 400
SEED = 'BindPack-serial-cost-400-per-dataset-20261029-v1'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(4194304), b''):
            h.update(part)
    return h.hexdigest()


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def stratified_sample(dataset, records, n=SAMPLE_SIZE):
    """Proportional hop/label allocation; SHA ordering never reads outcomes."""
    if not 0 < n <= len(records) or len({r['sample_id'] for r in records}) != len(records):
        raise ValueError('Invalid sample size or duplicate sample IDs')
    groups = defaultdict(list)
    for i, row in enumerate(records):
        groups[(int(row['hop']), str(row.get('label', '')))].append(i)
    keys = sorted(groups)
    quotas = {key: len(groups[key]) * n // len(records) for key in keys}
    remaining = n - sum(quotas.values())
    for key in sorted(keys, key=lambda key: (-(len(groups[key]) * n % len(records)), key))[:remaining]:
        quotas[key] += 1
    def score(i, purpose):
        return hashlib.sha256(f'{SEED}|{dataset}|{purpose}|{records[i]["sample_id"]}'.encode()).digest()
    selected = [i for key in keys for i in sorted(groups[key], key=lambda i: score(i, 'choose'))[:quotas[key]]]
    selected.sort(key=lambda i: score(i, 'order'))
    assert len(selected) == len(set(selected)) == n
    return selected, [dict(hop=key[0], label=key[1], population=len(groups[key]), selected=quotas[key])
                      for key in keys]


def jsonl(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def distribution(values):
    values = sorted(values)
    if not values:
        raise ValueError('No measured durations')
    def percentile(q):
        x = (len(values) - 1) * q; lo = int(x); hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (x - lo)
    return dict(n=len(values), total=sum(values), mean=sum(values) / len(values),
                p50=percentile(.5), p95=percentile(.95), min=values[0], max=values[-1])


def aggregate(root):
    root = Path(root); rows = []; intervals = []; all_latency = []
    totals = dict(queries=0, logical_calls=0, http_attempts=0, input_tokens=0, output_tokens=0,
                  unknown_usage=0, failed_attempts=0)
    for dataset in DATASETS:
        case = root / 'runs' / dataset / 'cases' / dataset / 'full'
        report = read(case / 'report.json'); done = read(case / 'completed.ok')
        assert done['query_count'] == report['coverage']['sample_count'] == SAMPLE_SIZE
        for key, filename in [('report', 'report.json'), ('result', 'result.json'), ('llm_calls', 'llm_calls.jsonl')]:
            assert done[key + '_sha256'] == sha(case / filename), 'Changed completed artifact: ' + filename
        assert read(case / 'audit.json')['status'] == 'passed'
        protocol = read(case / 'protocol.json'); cfg = protocol['configuration']
        assert protocol['query_concurrency'] == cfg['llm_prefetch_workers'] == 1
        assert cfg['stage_ablation'] == cfg['mechanism_ablation'] == 'full'
        assert cfg['source_bound_eligibility'] is False
        assert cfg['pack']['selector_mode'] == 'combined' and cfg['pack']['review_mode'] == 'off'
        assert cfg['pack']['relation_weight'] == .6 and cfg['qa_top_k'] == 5
        assert protocol['qa_executed'] and report['coverage']['qa_n'] == SAMPLE_SIZE
        assert report['cost']['offline_index']['http_attempts'] == 0
        assert report['efficiency']['query_concurrency'] == 1
        assert report['efficiency']['native_llm_concurrency_limit'] == 1
        selected = read(root / 'samples' / (dataset + '.json'))
        assert protocol['dataset']['selected_indices'] == selected['indices']
        latency = []; retrieval = []; reader = []; ids = []
        for event in jsonl(case / 'progress.jsonl'):
            row = event['row']; timing = row['timing']
            assert timing['execution_mode'] == 'serial'
            assert abs(timing['end_to_end_seconds'] - timing['retrieval_seconds'] - timing['reader_seconds']) < 1e-7
            latency.append(timing['end_to_end_seconds']); retrieval.append(timing['retrieval_seconds'])
            reader.append(timing['reader_seconds']); ids.append(row['sample_id'])
        assert len(ids) == SAMPLE_SIZE and ids == selected['sample_ids']
        logical = [e for e in jsonl(case / 'llm_logical_calls.jsonl') if e['phase'].startswith('online_')]
        calls = [e for e in jsonl(case / 'llm_calls.jsonl') if e['phase'].startswith('online_')]
        cost = report['cost']
        assert len(logical) == cost['llm_logical_calls'] and len(calls) == cost['llm_http_attempts']
        assert sum(e.get('prompt_tokens') or 0 for e in calls) == cost['input_tokens_known_total']
        assert sum(e.get('completion_tokens') or 0 for e in calls) == cost['output_tokens_known_total']
        for event in calls:
            start = event['started_unix']; end = start + event['seconds']
            intervals.extend([(start, 1), (max(start, end - .001), -1)])
        totals['queries'] += SAMPLE_SIZE; totals['logical_calls'] += len(logical); totals['http_attempts'] += len(calls)
        totals['input_tokens'] += cost['input_tokens_known_total']; totals['output_tokens'] += cost['output_tokens_known_total']
        totals['unknown_usage'] += cost['attempts_with_unknown_token_usage']; totals['failed_attempts'] += cost['llm_failed_attempts']
        all_latency.extend(latency)
        rows.append(dict(dataset=dataset, n=SAMPLE_SIZE, report=str(case / 'report.json'),
            report_sha256=sha(case / 'report.json'), protocol_sha256=sha(case / 'protocol.json'),
            sample_indices_sha256=sha(root / 'indices' / (dataset + '.json')),
            sample_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            logical_calls=len(logical), http_attempts=len(calls), input_tokens=cost['input_tokens_known_total'],
            output_tokens=cost['output_tokens_known_total'], unknown_usage=cost['attempts_with_unknown_token_usage'],
            latency=distribution(latency), retrieval=distribution(retrieval), reader=distribution(reader),
            efficiency=report['efficiency'], resources=report['resources'], evidence_budget=report['evidence_budget'],
            subset_quality_metrics=report['metrics']))
    active = maximum = 0
    for _, delta in sorted(intervals):
        active += delta; maximum = max(maximum, active)
    assert maximum == 1, f'Physical HTTP requests overlapped: {maximum}'
    assert totals['queries'] == 1600
    result = dict(schema_version=1, status='complete', protocol='current Full; 4 datasets x 400; serial retrieval plus Reader@5',
        query_concurrency=1, client_llm_workers=1, qa_executed=True, n=1600, queries_per_dataset=400,
        totals=totals, per_dataset=rows, latency=distribution(all_latency),
        means=dict(logical_calls_per_query=totals['logical_calls']/1600,
                   http_attempts_per_query=totals['http_attempts']/1600,
                   input_tokens_per_query=totals['input_tokens']/1600,
                   output_tokens_per_query=totals['output_tokens']/1600,
                   wall_seconds_per_query=sum(all_latency)/1600),
        accounting=dict(llm_calls='logical calls, including native cache hits; physical HTTP attempts separately retained',
            wall_time='continuous retrieval + context preparation + Reader for completed queries; excludes model/index loading and checkpoint writes',
            denominator='all 1600 fixed sampled questions, including incorrect/empty predictions',
            tokens='actual online HTTP usage, including failed attempts where usage is known',
            shared_server_cache='vLLM resident and prefix cache not reset; independent empty application caches per dataset',
            historical_index_hours=None, historical_index_reason='Three source indexes have no measured historical construction ledger',
            max_physical_http_overlap=maximum, overlap_tolerance_seconds=.001))
    return result

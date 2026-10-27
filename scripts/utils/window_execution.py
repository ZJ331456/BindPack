"""Native batched retrieval with durable pending solutions and explicit timing scopes."""
from collections import Counter
import json
from pathlib import Path
import time
import uuid


def freeze_protocol(case, protocol, freeze):
    """Retain a serial base protocol; record a separately validated execution revision."""
    from bindpack.evaluation.measurement import digest, write_json
    path=case/'protocol.json'
    if path.exists():
        base=json.loads(path.read_text())
        allowed={'source','query_concurrency'}
        if {k:v for k,v in base.items() if k not in allowed}!={k:v for k,v in protocol.items() if k not in allowed}:
            raise ValueError('Window execution changed method, inputs, models or reader protocol')
    else:
        freeze(path,protocol);base=protocol
    active=case/'execution_protocol_window8.json'
    value=dict(protocol,base_protocol_sha256=digest(base),
        execution_policy='native retrieve(window queries), serial reader/checkpoint; datasets serial',
        timing_policy='window retrieval wall plus individual reader; amortization is throughput, not isolated latency')
    if active.exists() and json.loads(active.read_text())!=value:
        raise ValueError('Frozen window execution protocol changed')
    write_json(active,value)
    return base,value


def iter_windows(rag, case, start, questions, hops, sample_ids, recorder, requested_stop, size=8):
    """Yield ordered solutions; do not re-retrieve a saved unfinished window."""
    from bindpack.evaluation.measurement import jsonl, write_json
    from bindpack.utils.misc_utils import QuerySolution
    import numpy as np
    case=Path(case);pending_path=case/'pending_retrieval_window.json'
    known={w['window_id'] for w in jsonl(case/'retrieval_windows.jsonl')}
    while start<len(questions):
        if requested_stop.is_set():
            raise KeyboardInterrupt('Stopped at a durable query boundary; native window retained')
        if pending_path.exists():
            pending=json.loads(pending_path.read_text())
            lo,hi=pending['start'],pending['end']
            if not lo<=start<=hi or pending['sample_ids']!=sample_ids[lo:hi] or pending['questions']!=questions[lo:hi]:
                raise ValueError('Pending retrieval window differs from frozen inputs/checkpoint')
            if start==hi:
                pending_path.unlink();continue
        else:
            hi=min(start+size,len(questions));lo=start
            window_id='window:'+uuid.uuid4().hex
            recorder.append('retrieval_window_attempts.jsonl',dict(window_id=window_id,start=lo,end=hi,
                sample_ids=sample_ids[lo:hi],started_unix=time.time()))
            with recorder.section('online_retrieval',window_id):
                began=time.perf_counter()
                rag.set_query_hop_overrides(hops[lo:hi],source='benchmark')
                solutions=rag.retrieve(questions[lo:hi],num_to_retrieve=200)
                seconds=time.perf_counter()-began
            if len(solutions)!=hi-lo or [s.question for s in solutions]!=questions[lo:hi]:
                raise ValueError('Native window lost or reordered queries')
            pending=dict(window_id=window_id,start=lo,end=hi,sample_ids=sample_ids[lo:hi],
                questions=questions[lo:hi],retrieval_seconds=seconds,
                solutions=[s.to_dict(top_k=len(s.docs)) for s in solutions])
            write_json(pending_path,pending)
        if pending['window_id'] not in known:
            recorder.append('retrieval_windows.jsonl',{k:v for k,v in pending.items() if k!='solutions'})
            known.add(pending['window_id'])
        for position in range(start,pending['end']):
            if requested_stop.is_set():
                raise KeyboardInterrupt('Stopped at a durable query boundary; native window retained')
            raw=pending['solutions'][position-pending['start']]
            solution=QuerySolution(**raw)
            if solution.doc_scores is not None:solution.doc_scores=np.asarray(solution.doc_scores,dtype=float)
            yield position,solution,dict(execution_mode='native_window8',window_id=pending['window_id'],
                window_size=pending['end']-pending['start'],
                retrieval_window_seconds=pending['retrieval_seconds'],
                retrieval_amortized_seconds=pending['retrieval_seconds']/(pending['end']-pending['start']))
        start=pending['end'];pending_path.unlink()


def augment_accounting(profile,case,rows):
    """Separate the previous serial latency from window throughput on a resumed run."""
    from bindpack.evaluation.measurement import distribution,jsonl
    windows=jsonl(Path(case)/'retrieval_windows.jsonl')
    serial=[r for r in rows if r['timing'].get('execution_mode','serial')=='serial']
    batched=[r for r in rows if r['timing'].get('execution_mode')=='native_window8']
    eff=profile['efficiency']
    eff['serial_query_pipeline_seconds']=eff['query_pipeline_seconds']
    eff['serial_retrieval_seconds']=eff['retrieval_seconds']
    if batched:
        # An unmeasured isolated latency must not inherit an amortized window time.
        eff['query_pipeline_seconds']=distribution([])
        eff['retrieval_seconds']=distribution([])
    eff['native_window_retrieval_seconds']=distribution(w['retrieval_seconds'] for w in windows)
    retrieval=sum(r['timing']['retrieval_seconds'] for r in serial)+sum(w['retrieval_seconds'] for w in windows)
    reader=sum(r['timing']['reader_seconds'] for r in rows)
    eff.update(execution_mode='mixed_serial_and_native_window8' if serial and batched else 'native_window8',
        query_concurrency=8,query_window_size=8,
        execution_segments=dict(serial_queries=len(serial),native_window_queries=len(batched),retrieval_windows=len(windows)),
        completed_retrieval_wall_seconds=retrieval,completed_reader_wall_seconds=reader,
        completed_query_processing_wall_seconds=retrieval+reader,
        processing_seconds_per_query=(retrieval+reader)/len(rows),
        latency_scope='Isolated query latency N/A after native window switch. serial_* describes the preserved serial prefix only; window totals are wall time and processing_seconds_per_query is throughput amortization.')
    profile['measurement_protocol'].update(execution_mode=eff['execution_mode'],query_window_size=8,
        retrieval_cost_attribution='serial calls retain sample IDs; native window retrieval calls use window IDs mapped in retrieval_windows.jsonl; reader calls retain sample IDs')
    return profile

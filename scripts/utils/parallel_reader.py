"""Concurrent native Reader calls, with ordered, durable per-query responses."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time


def iter_readers(stream, rag, case, sample_ids, recorder, requested_stop, workers, prepare_solution):
    from bindpack.evaluation.measurement import jsonl, write_json
    case=Path(case);pending_path=case/'pending_reader_window.json'
    known={row['window_id'] for row in jsonl(case/'reader_windows.jsonl')}
    source=iter(stream)
    while True:
        first=next(source,None)
        if first is None:return
        start,_,timing=first
        retrieval=json.loads((case/'pending_retrieval_window.json').read_text())
        end=retrieval['end'];window_id=timing['window_id']
        if retrieval['window_id']!=window_id or not retrieval['start']<=start<end:
            raise ValueError('Reader and durable retrieval window differ')
        # Do not read ahead into the next retrieval window before checkpointing.
        batch=[first]+[next(source) for _ in range(end-start-1)]
        if [item[0] for item in batch]!=list(range(start,end)):
            raise ValueError('Reader window lost or reordered queries')
        if pending_path.exists():
            pending=json.loads(pending_path.read_text())
            if (pending['window_id']!=window_id or pending['sample_ids']!=sample_ids[retrieval['start']:end]):
                raise ValueError('Saved Reader responses differ from frozen queries')
        else:
            pending=dict(window_id=window_id,start=retrieval['start'],end=end,
                sample_ids=sample_ids[retrieval['start']:end],responses={},seconds=0.0)
            write_json(pending_path,pending)
        prepared={}
        began=time.perf_counter();prior=pending['seconds']
        for position,solution,_ in batch:
            tick=time.perf_counter()
            with recorder.section('online_reader',sample_ids[position]):
                reader_solution,budget=prepare_solution(solution)
            prepared[position]=(reader_solution,budget,time.perf_counter()-tick)
            old=pending['responses'].get(str(position))
            if old is not None and old['budget']!=budget:
                raise ValueError('Reader context changed before checkpoint resume')
        def answer(position):
            reader_solution,budget,preparation_seconds=prepared[position]
            with recorder.thread_section('online_reader',sample_ids[position]):
                tick=time.perf_counter()
                answered,raw,metadata=rag.qa([reader_solution])
                seconds=preparation_seconds+time.perf_counter()-tick
            return dict(answer=answered[0].answer,raw_response=raw[0],metadata=metadata[0],
                        budget=budget,seconds=seconds)
        missing=[p for p,_,_ in batch if str(p) not in pending['responses']]
        if missing:
            with recorder.section('online_reader',window_id):
                with ThreadPoolExecutor(max_workers=workers,thread_name_prefix='bindpack-reader') as pool:
                    futures={pool.submit(answer,p):p for p in missing}
                    for future in as_completed(futures):
                        position=futures[future]
                        pending['responses'][str(position)]=future.result()
                        pending['seconds']=prior+time.perf_counter()-began
                        write_json(pending_path,pending)
            pending['seconds']=prior+time.perf_counter()-began
            write_json(pending_path,pending)
        if window_id not in known:
            recorder.append('reader_windows.jsonl',dict(window_id=window_id,start=pending['start'],
                end=end,sample_ids=pending['sample_ids'],seconds=pending['seconds'],workers=workers))
            known.add(window_id)
        for position,solution,retrieval_timing in batch:
            if requested_stop.is_set():
                raise KeyboardInterrupt('Reader responses and retrieval window retained for resume')
            yield position,solution,dict(retrieval_timing,reader_window_id=window_id,
                reader_window_seconds=pending['seconds'],
                reader_amortized_seconds=pending['seconds']/(end-pending['start']),
                _prepared_reader=pending['responses'][str(position)])
        pending_path.unlink()


def augment_reader_accounting(profile,case,rows,workers):
    from bindpack.evaluation.measurement import jsonl, distribution
    windows=jsonl(Path(case)/'reader_windows.jsonl')
    ids={row['timing']['reader_window_id'] for row in rows}
    assert ids=={window['window_id'] for window in windows}
    reader_wall=sum(window['seconds'] for window in windows)
    efficiency=profile['efficiency']
    efficiency.update(reader_concurrency=workers,reader_window_seconds=distribution(w['seconds'] for w in windows),
        reader_request_seconds=efficiency['reader_seconds'],completed_reader_wall_seconds=reader_wall,
        completed_query_processing_wall_seconds=efficiency['completed_retrieval_wall_seconds']+reader_wall,
        latency_scope='Native retrieval and Reader windows; per-query processing is throughput amortization, not isolated latency')
    efficiency['processing_seconds_per_query']=efficiency['completed_query_processing_wall_seconds']/len(rows)
    profile['measurement_protocol'].update(reader_concurrency=workers,
        reader_cost_attribution='Physical and logical Reader requests retain individual sample IDs; window wall time counted once')

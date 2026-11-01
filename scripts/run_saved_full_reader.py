#!/usr/bin/env python3
"""Complete native Reader@5 on immutable saved Qwen Full rankings."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import copy
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
import urllib.request

PROJECT=Path('/root/BindPack')
SOURCE=PROJECT/'outputs/4_datasets_result/final_full_no_source_20261028/cases/musique/full'
SERIAL=PROJECT/'outputs/serial_cost_1600_20261029/runs/musique/cases/musique/full'
DEFAULT=PROJECT/'outputs/qwen3_qwenemb8b_musique_full_reader_20261031'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()] if Path(p).exists() else []
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temporary=p.with_name(p.name+'.tmp')
    temporary.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temporary.replace(p)

def main(args):
    dataset=args.dataset
    source=PROJECT/f'outputs/4_datasets_result/final_full_no_source_20261028/cases/{dataset}/full'
    serial_source=PROJECT/f'outputs/serial_cost_1600_20261029/runs/{dataset}/cases/{dataset}/full'
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    if not args.frozen_worker:
        snapshot=root/'source_snapshot'
        if not snapshot.exists():
            shutil.copytree(PROJECT/'src',snapshot/'src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            (snapshot/'scripts').mkdir(parents=True)
            shutil.copy2(Path(__file__),snapshot/'scripts'/Path(__file__).name)
        os.execv(sys.executable,[sys.executable,'-B','-u',str(snapshot/'scripts'/Path(__file__).name),
            '--output',str(root),'--dataset',dataset,'--frozen-worker'])
    os.environ['BINDPACK_LLM_MAX_IN_FLIGHT']='32';os.environ['OPENAI_API_KEY']='sk-local-dummy'
    os.environ['TOKENIZERS_PARALLELISM']='false'
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    from bindpack.core.base import BaseRAG
    from bindpack.configuration.runtime import RuntimeConfig
    from bindpack.llm.openai_gpt import CacheOpenAI
    from bindpack.prompts import PromptTemplateManager
    from bindpack.utils.misc_utils import QuerySolution
    from bindpack.evaluation.measured_reader import prepare
    from bindpack.evaluation.measurement import Recorder,digest,usage,ResourceMonitor
    from bindpack.evaluation.metric_report import _qa
    from dataclasses import fields
    if (root/'completed.ok').exists():
        marker=read(root/'completed.ok');assert sha(root/'report.json')==marker['report_sha256']
        print('Already complete; no inference',flush=True);return
    source_report=read(source/'report.json');source_done=read(source/'completed.ok');source_protocol=read(source/'protocol.json')
    assert source_done['query_count']==1000 and sha(source/'report.json')==source_done['report_sha256']
    assert source_report['coverage']['qa_n']==0 and read(source/'audit.json')['status']=='passed'
    assert sha(PROJECT/'datasets'/(dataset+'.json'))==source_protocol['dataset']['data_sha256']
    serial_protocol=read(serial_source/'protocol.json');serial_done=read(serial_source/'completed.ok')
    assert sha(serial_source/'report.json')==serial_done['report_sha256']
    assert sha(serial_source/'result.json')==serial_done['result_sha256']
    cfg=source_protocol['configuration'];scfg=serial_protocol['configuration']
    for key in ('llm_name','embedding_model_name','seed','temperature','max_new_tokens','qa_top_k'):
        assert cfg[key]==scfg[key]
    assert cfg['llm_name']=='qwen3-8b' and cfg['embedding_model_name']=='/root/models/Qwen3-Embedding-8B'
    original=[entry['row'] for entry in lines(source/'progress.jsonl')]
    assert len(original)==1000 and len({r['sample_id'] for r in original})==1000
    serial={entry['row']['sample_id']:entry['row'] for entry in lines(serial_source/'progress.jsonl')}
    protocol=dict(dataset=dataset,questions=1000,mode='native_reader_on_saved_full',model='qwen3-8b',
        embedding='/root/models/Qwen3-Embedding-8B',source_full=str(source),
        source_report_sha256=sha(source/'report.json'),source_progress_sha256=sha(source/'progress.jsonl'),
        source_protocol_sha256=sha(source/'protocol.json'),source_configuration=cfg,
        serial_reuse_source=str(serial_source),serial_report_sha256=sha(serial_source/'report.json'),
        qa_top_k=5,qa_max_tokens=2048,thinking_enabled=False,reader_workers=32,
        reuse_rule='same sample, exact Top5, exact native prepared prompt hash, identical model/seed/temperature/output cap; no outcome selection',
        no_index_build=True,no_embedding_inference=True,no_retrieval_or_plan_rerun=True,
        timing_scope='Reader supplement only; original upstream and reused answers are not new inference cost',
        native_reader_sha256=sha(Path(__file__).resolve().parents[1]/'src/bindpack/core/base.py'))
    if (root/'protocol.json').exists():assert read(root/'protocol.json')==protocol
    else:write(root/'protocol.json',protocol)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('/root/models/Qwen3-8B',local_files_only=True)
    runtime=RuntimeConfig(**{key:value for key,value in cfg.items() if key in {field.name for field in fields(RuntimeConfig)}})
    runtime.llm_base_url='http://127.0.0.1:8035/v1'
    recorder=Recorder(root);llm=CacheOpenAI(cache_dir=str(root/'llm_cache'),global_config=runtime)
    recorder.install_llm(llm,tokenizer)
    rag=SimpleNamespace(global_config=runtime,llm_model=llm,prompt_template_manager=PromptTemplateManager())
    prepared={};eligible={}
    for row in original:
        solution,budget=prepare(rag,QuerySolution(question=row['question'],docs=row['docs']),tokenizer,dataset)
        assert budget['prompt_sha256']==row['reader']['budget']['prompt_sha256'],row['sample_id']
        prepared[row['sample_id']]=(solution,budget)
        old=serial.get(row['sample_id'])
        if (old and old['question']==row['question'] and old['docs'][:5]==solution.docs and
                old['reader']['budget']['prompt_sha256']==budget['prompt_sha256']):
            eligible[row['sample_id']]=old
    print(f'FIXED FULL: 1000 exact native prompts verified; reuse={len(eligible)}; new QA={1000-len(eligible)}',flush=True)
    completed={}
    for entry in lines(root/'progress.jsonl'):
        assert digest(entry['row'])==entry['sha256'];completed[entry['row']['sample_id']]=entry['row']
    def save(row):
        completed[row['sample_id']]=row;recorder.append('progress.jsonl',dict(row=row,sha256=digest(row)))
    for row in original:
        sid=row['sample_id']
        if sid in completed or sid not in eligible:continue
        old=eligible[sid]
        save(dict(sample_id=sid,query_index=row['query_index'],question=row['question'],answer=old['answer'],
            gold_answers=row['gold_answers'],docs=prepared[sid][0].docs,reader=old['reader'],
            provenance=dict(mode='reused_exact_native_reader',source=str(serial_source),
                source_row_sha256=digest(old),prompt_sha256=prepared[sid][1]['prompt_sha256'])))
    pending=[row for row in original if row['sample_id'] not in completed]
    began=time.perf_counter();deadline=time.monotonic()+600
    while pending:
        try:
            with urllib.request.urlopen(runtime.llm_base_url+'/models',timeout=3) as stream:models=json.load(stream)
            assert any(m['id']=='qwen3-8b' and m.get('root')=='/root/models/Qwen3-8B' for m in models['data'])
            write(root/'service_identity.json',models);break
        except Exception:
            if time.monotonic()>deadline:raise
            print('Waiting for local Qwen3-8B service',flush=True);time.sleep(5)
    def answer(row):
        sid=row['sample_id'];solution,budget=prepared[sid]
        with recorder.thread_section('online_reader',sid):
            start=time.perf_counter();solutions,raw,metadata=BaseRAG.qa(rag,[solution]);seconds=time.perf_counter()-start
        return dict(sample_id=sid,query_index=row['query_index'],question=row['question'],
            answer=solutions[0].answer,gold_answers=row['gold_answers'],docs=solution.docs,
            reader=dict(raw_response=raw[0],metadata=metadata[0],budget=budget),reader_seconds=seconds,
            provenance=dict(mode='fresh_native_reader',prompt_sha256=budget['prompt_sha256']))
    monitor=ResourceMonitor(recorder)
    with monitor,recorder.section('online_reader'):
        with ThreadPoolExecutor(max_workers=32) as pool:
            futures={pool.submit(answer,row):row['sample_id'] for row in pending}
            for future in as_completed(futures):
                save(future.result())
                if len(completed)%20==0 or len(completed)==1000:
                    print(f'PROGRESS saved-Full Reader {len(completed)}/1000; new_requests={len(lines(root/"llm_calls.jsonl"))}; elapsed={time.perf_counter()-began:.1f}s',flush=True)
    rows=[completed[row['sample_id']] for row in original]
    data=read(PROJECT/'datasets'/(dataset+'.json'));samples=[data[row['query_index']] for row in rows]
    qa,coverage,details=_qa(rows,samples,dict(dataset=dataset,eval_mode='rag_qa'))
    assert coverage['qa_n']==1000
    assert sha(source/'report.json')==protocol['source_report_sha256']
    assert sha(source/'progress.jsonl')==protocol['source_progress_sha256']
    result=dict(dataset=dataset,method='bindpack',eval_mode='rag_qa',sample_size_effective=1000,
        results=rows,qa_metrics=qa,qa_details=details,source_protocol=protocol)
    write(root/'result.json',result)
    metrics=copy.deepcopy(source_report['metrics']);metrics.update(qa)
    merged=copy.deepcopy(source_report['coverage']);merged.update(coverage)
    calls=lines(root/'llm_calls.jsonl');new_usage=usage(calls)
    report=dict(dataset=dataset,method='bindpack',metrics=metrics,coverage=merged,
        measurement_protocol=protocol,source_retrieval_report=str(source/'report.json'),
        qa_reused_exact_context=len(eligible),qa_new_inference=1000-len(eligible),
        cost=dict(scope='new Reader inference only; reused answers and original upstream costs excluded',**new_usage),
        efficiency=dict(scope='Reader supplement only',wall_seconds=time.perf_counter()-began,reader_workers=32),
        resources=monitor.summary())
    write(root/'report.json',report)
    write(root/'audit.json',dict(status='passed',questions=1000,qa_n=1000,prompts_all_match_saved_full=True,
        exact_answer_reuse=len(eligible),fresh_reader_answers=1000-len(eligible),
        original_full_unchanged=True,new_index_calls=0,new_embedding_calls=0,new_retrieval_calls=0,
        actual_fresh_requests=len(calls),unknown_usage=new_usage['attempts_with_unknown_usage']))
    write(root/'completed.ok',dict(query_count=1000,report_sha256=sha(root/'report.json'),
        result_sha256=sha(root/'result.json'),llm_calls_sha256=sha(root/'llm_calls.jsonl')))
    (root/'RESULTS.md').write_text(f'# {dataset} / Qwen3-8B + Qwen3-Emb-8B：原 Full 的 Reader@5 补充\n\n'+
        '| N | Recall@5 | FCR@5 | DCC@5 | EM | QA F1 |\n|---:|---:|---:|---:|---:|---:|\n| 1000 | '+
        ' | '.join('N/A' if metrics.get(key) is None else f'{metrics[key]*100:.2f}'
                   for key in ('Recall@5','FCR@5','DCC@5','ExactMatch','ClassificationMacroF1' if dataset=='hover' else 'F1'))+' |\n\n'+
        f'固定原始 Full 的证据及顺序；{len(eligible)} 条复用相同原生 prompt 和相同模型生成配置的历史答案，{1000-len(eligible)} 条调用本地原生 Reader 补齐。不重建索引、不重新执行 Route/Plan/Pack；未改动原始检索报告。\n')
    print('PASS SAVED-FULL QA: '+json.dumps(metrics),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--dataset',choices=['hotpotqa','2wikimultihopqa','musique','hover'],default='musique')
    parser.add_argument('--frozen-worker',action='store_true')
    logging.basicConfig(level=logging.WARNING)
    main(parser.parse_args())

#!/usr/bin/env python3
"""Publish a verified four-dataset summary after the detached queue exits."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import select
import time


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def write(p,v):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)


def process_state(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rpartition(') ')[2].split()
        return fields[0],fields[19]
    except FileNotFoundError:return None


def wait_for_process(pid):
    if hasattr(os,'pidfd_open'):
        try:fd=os.pidfd_open(pid)
        except ProcessLookupError:return
        try:select.select([fd],[],[])
        finally:os.close(fd)
        return
    original=process_state(pid)
    while original is not None:
        current=process_state(pid)
        if current is None or current[0]=='Z' or current[1]!=original[1]:return
        time.sleep(30)


def finalize(root):
    protocol=read(root/'queue_protocol.json');state=read(root/'queue_status.json')
    complete=(root/'completed.ok').exists() and state['status']=='complete'
    results={};rows=[]
    for ds in protocol['datasets']:
        for stage in protocol['cases']:
            case=root/'cases'/ds/stage
            if not (case/'completed.ok').exists():continue
            done=read(case/'completed.ok');report=read(case/'report.json')
            assert done['query_count']==1000 and report['coverage']['sample_count']==1000
            for name in ('report','result','llm_calls'):
                suffix='.jsonl' if name=='llm_calls' else '.json'
                assert sha(case/(name+suffix))==done[name+'_sha256']
            assert read(case/'audit.json')['status']=='passed'
            assert read(case/'index_reuse_verified.ok')['source_and_private_unchanged']
            assert read(case/'private_assets_released.ok')['source_unchanged']
            results[ds+'/'+stage]=report
            correct,f1=('ClassificationAccuracy','ClassificationMacroF1') if ds=='hover' else ('ExactMatch','F1')
            metrics=report['metrics']
            rows.append(dict(dataset=ds,case=stage,queries=1000,Recall_at5=metrics['Recall@5'],FCR_at5=metrics['FCR@5'],
                correctness=metrics[correct],answer_F1=metrics[f1],DCC_at5=metrics.get('DCC@5'),
                NodeCov_at5=metrics.get('NodeCov@5'),ClosureGap_at5=metrics.get('ClosureGap@5'),
                latency_mean_seconds=report['efficiency']['query_pipeline_seconds']['mean'],
                processing_seconds_per_query=report['efficiency'].get('processing_seconds_per_query'),
                execution_mode=report['efficiency'].get('execution_mode','serial'),
                llm_calls=report['cost']['llm_logical_calls'],input_tokens=report['cost']['input_tokens_known_total'],
                output_tokens=report['cost']['output_tokens_known_total']))
    if complete:
        assert len(rows)==4*len(protocol['cases']) and sum(r['queries'] for r in rows)==protocol['expected_queries']
        for rel,value in protocol['source_sha256'].items():assert sha(root/'source_snapshot'/rel)==value
        for value in protocol['frozen_inputs'].values():assert sha(value['path'])==value['sha256']
        assert sha(Path(protocol['index_source']).parent/'manifest.json')==protocol['archive_manifest_sha256']
        overlay=root/'window8_transition/overlay_receipt.json'
        if overlay.exists():
            value=read(overlay);snapshot=Path(value['snapshot'])
            for rel,expected in value['source_sha256'].items():assert sha(snapshot/rel)==expected
            for rel,expected in protocol['source_sha256'].items():
                if rel.startswith(('src/','configs/')):assert sha(snapshot/rel)==expected
    status='complete' if complete else state['status']
    write(root/'summary.json',dict(status=status,expected_queries=protocol['expected_queries'],
        completed_queries=1000*len(rows),results=results))
    if rows:
        with (root/'results_at5.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# 最新方案四数据集Full结果','',f'状态：{status}；已完成{len(rows)}组。权重0.60，当前原生hop、相同Qwen3模型与Reader@5，复用历史图/向量索引。','',
        '| 数据集 | Case | Recall@5 | FCR@5 | EM / Accuracy | F1 / Macro-F1 | DCC@5 | NodeCov@5 | ClosureGap@5 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        values=['N/A' if row[k] is None else f'{row[k]:.6f}' for k in ('Recall_at5','FCR_at5','correctness','answer_F1','DCC_at5','NodeCov_at5','ClosureGap_at5')]
        lines.append(f'| {row["dataset"]} | {row["case"]} | '+' | '.join(values)+' |')
    lines+=['','各case的report.json还包含完整延迟、LLM请求/tokens、embedding、索引复用准备、证据预算和资源计量。无可信gold DAG的数据集不填造DCC/NodeCov/ClosureGap。',
        '','原索引建库费用与本次在线费用分开；未保存原始完整成本的索引保留N/A。整张共享GPU采样不等于方法独占GPU成本。']
    (root/'results_at5.md').write_text('\n'.join(lines)+'\n')
    write(root/'finalization.json',dict(status=status,verified_cases=len(rows),verified_queries=len(rows)*1000,
        source_and_input_snapshots_verified=complete,new_model_requests=0))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--wait-pid',type=int);a=p.parse_args()
    if a.wait_pid:
        wait_for_process(a.wait_pid)
    finalize(a.output.resolve())


if __name__=='__main__':main()

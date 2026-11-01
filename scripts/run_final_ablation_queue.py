#!/usr/bin/env python3
"""Fresh retrieval-only mechanism and stage ablations in the authorized order."""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
from zoneinfo import ZoneInfo

PROJECT = Path('/root/BindPack')
ARCHIVE = PROJECT / 'outputs/4_datasets_result'
FULL = ARCHIVE / 'final_full_no_source_20261028'
DEFAULT = PROJECT / 'outputs/ablations_full_20261029'
DATASETS = ('hotpotqa', '2wikimultihopqa', 'musique', 'hover')


def read(path): return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4194304),b''): h.update(block)
    return h.hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def stamp(): return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def ordered_cases():
    cases=[]
    for ds in DATASETS:
        for label,arm in (('wo_dependency','wo_dependency_closure'),('wo_joint_selection','wo_joint_selection')):
            cases.append(dict(group='mechanisms',dataset=ds,label=label,stage='full',mechanism=arm))
    for ds in DATASETS:
        for stage in ('wo_route','wo_plan','wo_pack'):
            cases.append(dict(group='stages',dataset=ds,label=stage,stage=stage,mechanism='full'))
    return cases


def source_hashes(snapshot):
    return {str(p.relative_to(snapshot)):sha(p) for p in snapshot.rglob('*')
            if p.is_file() and p.suffix in ('.py','.json')}


def prepare(root):
    protocol=root/'protocol.json'
    if protocol.exists():
        info=read(protocol)
        assert info['cases']==ordered_cases()
        assert source_hashes(root/'source_snapshot')==info['source_sha256']
        for item in info['datasets'].values():
            for suffix in ('','_corpus'):
                assert sha(root/'frozen_datasets'/(item['dataset']+suffix+'.json'))==item['hashes'][suffix]
        return info
    validation=read(root/'validation/tests.json');assert validation['status']=='passed'
    snapshot=root/'source_snapshot'
    for name in ('src','scripts','configs'):
        shutil.copytree(PROJECT/name,snapshot/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    contract=read(FULL/'manifest.json');datasets={};baselines={}
    for ds in DATASETS:
        case=FULL/'cases'/ds/'full';done=read(case/'completed.ok');assert done['query_count']==1000
        for name in ('report','result','llm_calls'):
            path=case/(name+('.jsonl' if name=='llm_calls' else '.json'))
            assert sha(path)==done[name+'_sha256']
        datasets[ds]=dict(dataset=ds,hashes={})
        for suffix in ('','_corpus'):
            source=PROJECT/'datasets'/(ds+suffix+'.json')
            assert sha(source)==sha(FULL/'input_provenance'/ds/source.name),'Dataset differs from the current Full: '+source.name
            target=root/'frozen_datasets'/source.name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target);datasets[ds]['hashes'][suffix]=sha(target)
        baselines[ds]=dict(case=str(case),metrics=read(case/'report.json')['metrics'],
                           report_sha256=sha(case/'report.json'),query_count=1000)
    # All algorithm groups stay identical to the frozen current Full.
    sys.path.insert(0,str(snapshot/'src'))
    from bindpack import BindPackConfig
    from dataclasses import asdict
    cfg=asdict(BindPackConfig())
    for group in ('route','plan','pack','candidates'):
        assert cfg[group]==contract['full_default_configuration'][group],group+' changed from Full'
    assert cfg['source_bound_eligibility'] is False and cfg['pack']['review_mode']=='off'
    base=read(PROJECT/'configs/ablations_qwen3.json')
    base.update(method_config=str(snapshot/'configs/full_no_source_eligibility.json'),
                datasets=list(DATASETS),datasets_dir=str(root/'frozen_datasets'))
    base['evaluation'].update(eval_mode='retrieve',qa_top_k=5,sample_size=0)
    write(root/'base_config.json',base)
    write(root/'baseline_full.json',baselines)
    info=dict(schema_version=1,created_at=stamp(),cases=ordered_cases(),expected_cases=20,
        queries_per_case=1000,total_queries=20000,datasets=datasets,source_sha256=source_hashes(snapshot),
        source_bound_eligibility=False,support_semantic_veto=False,relation_weight=.6,
        soft_coverage_ablation_excluded_by_user=True,eval_mode='retrieve',qa_executed=False,
        execution='cases serial in exact user order; calibrated native query-window/client LLM parallelism inside each case',
        model_policy='Qwen3-8B and Qwen3-Embedding-8B unchanged; embedding batch4; current vLLM service retained',
        index_policy='same verified shared indexes, private query-only copies; original index build is forbidden',
        upstream_policy='fresh native Route and Plan for all official runs; no saved-response replay or cross-case query caches',
        checkpoint_policy='resume exact ordered completed prefix and durable unfinished window; preserve all original attempts and cost',
        performance_policy='8/16/32 client bounds calibrated on identical 32 original HotpotQA queries; single-pass observations, shared server cache not reset; no isolated-latency claim',
        metric_policy='Recall/FCR and available structural metrics at K=1,3,5,8,10,20,200; QA N/A; throughput amortization and all calls/tokens/resources retained',
        vllm_service_pid=4267,vllm_gpu_memory_utilization=.55,vllm_max_num_seqs=8,
        baseline_contract_sha256=sha(FULL/'manifest.json'))
    write(protocol,info)
    (root/'README.md').write_text(
        '# BindPack final ablations (20 fresh full-data cases)\n\n'
        'First, for HotpotQA, 2WikiMultiHopQA, MuSiQue and HoVer in that order: '
        'w/o Dependency, w/o Joint Selection. Then the same dataset order: w/o Route, w/o Plan, w/o Pack. '
        'Soft Coverage ablation is excluded. Every case uses all 1000 original queries and the same shared index. '
        'Source eligibility and support/semantic veto remain off; relevance weight remains 0.60. '
        'Retrieval only; QA is N/A. Native upstream calls are fresh, not response replay.\n\n'
        'Calibration and preflight are separate from the 20 official cases and are separately accounted. '
        'Within-case parallelism is selected using measured throughput, while cases stay serial. '
        'Completed-window time is throughput amortization, not isolated per-query latency. '
        'The shared vLLM service is not restarted.\n\n'
        'Progress: status.json and run.log. Results: RESULTS.md and summary.json. '
        'Per-case reports, checkpoint/cost ledgers and resource samples are under mechanisms/ and stages/.\n\n'
        'Resume: bash /root/BindPack/scripts/run_final_ablation_queue.sh\n')
    return info


def complete(case,n):
    if not (case/'completed.ok').exists():return False
    done=read(case/'completed.ok');assert done['query_count']==n
    for name in ('report','result','llm_calls'):
        path=case/(name+('.jsonl' if name=='llm_calls' else '.json'))
        assert sha(path)==done[name+'_sha256']
    report=read(case/'report.json');assert report['coverage']['qa_n']==0
    assert read(case/'audit.json')['status']=='passed'
    return True


def summarize(root,status):
    rows=[];baselines=read(root/'baseline_full.json')
    lines=['# 当前Full与20组原生全量消融','',
           '来源资格和末端核验关闭，相关性权重0.60；每组1000题，QA=N/A。仅列完成并通过核验的结果。','',
           '|组别|数据集|方案|N|Recall@5|FCR@5|NodeCov@5|DCC@5|ClosureGap@5|处理秒/题（吞吐摊销）|LLM请求|输入tokens|输出tokens|',
           '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for spec in ordered_cases():
        output=root/spec['group']/spec['dataset']/spec['label'];case=output/'cases'/spec['dataset']/spec['stage']
        if not complete(case,1000):continue
        report=read(case/'report.json');metric=report['metrics'];cost=report['cost'];eff=report['efficiency']
        row=dict(spec,report=str(case/'report.json'),metrics=metric,efficiency=eff,cost=cost,
                 resources=report['resources'],evidence_budget=report['evidence_budget'])
        rows.append(row)
        vals=['N/A' if metric.get(k) is None else f'{metric[k]:.6f}' for k in ('Recall@5','FCR@5','NodeCov@5','DCC@5','ClosureGap@5')]
        seconds=eff.get('processing_seconds_per_query')
        lines.append('|'+ '|'.join([spec['group'],spec['dataset'],spec['label'],'1000',*vals,
            'N/A' if seconds is None else f'{seconds:.4f}',str(cost['llm_http_attempts']),str(cost['input_tokens_known_total']),str(cost['output_tokens_known_total'])])+'|')
    write(root/'summary.json',dict(status=status,completed_cases=len(rows),expected_cases=20,full_references=baselines,qa_executed=False,results=rows))
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--frozen-queue',action='store_true')
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    prepare(root)
    if not args.frozen_queue:
        os.execv(sys.executable,[sys.executable,'-B','-u',str(root/'source_snapshot/scripts/run_final_ablation_queue.py'),
            '--output',str(root),'--frozen-queue',*(['--prepare-only'] if args.prepare_only else [])])
    if args.prepare_only:return
    lock=(PROJECT/'outputs/.bindpack_ablation_queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (root/'run.pid').write_text(str(os.getpid())+'\n')
    state=dict(status='running',phase='calibration',pid=os.getpid(),completed_cases=0,expected_cases=20)
    stop=threading.Event();end=threading.Event();current=[None]
    def stop_requested(*unused):
        stop.set()
        if current[0] is not None and current[0].poll() is None:current[0].send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop_requested);signal.signal(signal.SIGINT,stop_requested)
    def heartbeat():
        while not end.is_set():
            value=dict(state,updated_at=stamp())
            if state.get('worker_output'):
                p=Path(state['worker_output'])/'status.json'
                if p.exists():value['active']=read(p)
            write(root/'status.json',value)
            active=value.get('active',{})
            print(f"{stamp()} PROGRESS {value['completed_cases']}/20 | {value.get('phase')} | {value.get('dataset','')} / {value.get('label','')} | {active.get('completed_queries',0)}/{active.get('expected_queries','?')} | {active.get('phase','preparing')}",flush=True)
            end.wait(30)
    watcher=threading.Thread(target=heartbeat,daemon=True);watcher.start()
    def execute(spec,output,window,workers,n=1000):
        if stop.is_set():raise KeyboardInterrupt('Requested stop')
        case=output/'cases'/spec['dataset']/spec['stage']
        state.update(dataset=spec['dataset'],label=spec['label'],worker_output=str(output))
        settings=root/'settings'/f'workers{workers}.json'
        cfg=read(root/'base_config.json');cfg['runtime']['llm_prefetch_workers']=workers
        if settings.exists():assert read(settings)==cfg
        else:write(settings,cfg)
        if not complete(case,n):
            command=[sys.executable,'-B','-u',str(root/'source_snapshot/scripts/run_bindpack_measured.py'),
                '--frozen-worker','--dataset',spec['dataset'],'--output',str(output),'--config',str(settings),
                '--datasets-dir',str(root/'frozen_datasets'),'--reuse-archive-index',str(ARCHIVE/'shared_indexes'/spec['dataset']),
                '--query-window',str(window),'--case',spec['stage'],'--mechanism-ablation',spec['mechanism'],
                '--source-bound-eligibility','off','--eval-mode','retrieve','--candidate-output-top-k','10']
            if n!=1000:command+=['--query-limit',str(n)]
            if (output/'experiment_config.json').exists():command+=['--resume']
            output.mkdir(parents=True,exist_ok=True);case.mkdir(parents=True,exist_ok=True)
            write(output/'launch.json',dict(spec=spec,command=command,query_window=window,client_llm_workers=workers))
            print(stamp(),'START',state['phase'],spec['dataset'],spec['label'],f'window={window}, client_workers={workers}',flush=True)
            process=subprocess.Popen(command,cwd=PROJECT,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            current[0]=process
            with (case/'run.log').open('a') as log:
                for line in process.stdout:log.write(line);log.flush();print(line,end='',flush=True)
            code=process.wait();current[0]=None
            if stop.is_set():raise KeyboardInterrupt('Stopped at a durable checkpoint; resume uses the same frozen run')
            if code or not complete(case,n):raise RuntimeError('Case failed; queue stopped: '+str(case/'run.log'))
        if not (case/'private_assets_released.ok').exists():
            sys.path.insert(0,str(root/'source_snapshot/scripts'))
            from utils.archived_index import verify_private
            verify_private(case,release=True)
        report=read(case/'report.json')
        print(stamp(),'PASS',spec['dataset'],spec['label'],n,'queries',flush=True)
        return report
    try:
        calibration_path=root/'calibration/selection.json'
        if calibration_path.exists():selection=read(calibration_path)
        else:
            observations=[]
            for workers in (16,8,32):
                spec=dict(group='calibration',dataset='hotpotqa',label=f'window{workers}',stage='full',mechanism='full')
                report=execute(spec,root/'calibration'/spec['label'],workers,workers,32)
                observations.append(dict(query_window=workers,client_llm_workers=workers,
                    processing_seconds_per_query=report['efficiency']['processing_seconds_per_query'],
                    window_retrieval=report['efficiency']['native_window_retrieval_seconds'],
                    http_requests=report['cost']['llm_http_attempts'],resources=report['resources']))
            best=min(observations,key=lambda x:x['processing_seconds_per_query'])
            baseline=next(x for x in observations if x['query_window']==8)
            selection=dict(best,baseline_seconds_per_query=baseline['processing_seconds_per_query'],
                speedup_vs_window8=baseline['processing_seconds_per_query']/best['processing_seconds_per_query'],
                observations=observations,scope='single-pass 32-query HotpotQA throughput calibration; shared vLLM cache not reset; no universal maximum guarantee')
            write(calibration_path,selection)
        window=selection['query_window'];workers=selection['client_llm_workers']
        state.update(phase='preflight',query_window=window,client_llm_workers=workers)
        # Exercise every distinct ablation on the complex HoVer input adapter.
        checks=[dict(group='preflight',dataset='hover',label=x,stage='full',mechanism=a)
                for x,a in (('wo_dependency','wo_dependency_closure'),('wo_joint_selection','wo_joint_selection'))]
        checks += [dict(group='preflight',dataset='hover',label=s,stage=s,mechanism='full') for s in ('wo_route','wo_plan','wo_pack')]
        for spec in checks:execute(spec,root/'preflight'/spec['label'],window,workers,2)
        write(root/'preflight/passed.json',dict(status='passed',cases=5,queries_per_case=2,qa_executed=False))
        state['phase']='official'
        for spec in ordered_cases():
            execute(spec,root/spec['group']/spec['dataset']/spec['label'],window,workers)
            state['completed_cases']+=1;summarize(root,'running')
        state.update(status='complete',phase='complete')
        summarize(root,'complete');write(root/'completed.ok',dict(cases=20,queries=20000,qa_executed=False,finished_at=stamp()))
    except BaseException as exc:
        state.update(status='stopped' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc))
        summarize(root,state['status'])
        raise
    finally:
        end.set();watcher.join(timeout=3);write(root/'status.json',dict(state,updated_at=stamp()))


if __name__=='__main__':main()

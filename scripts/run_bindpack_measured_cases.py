#!/usr/bin/env python3
"""Serial Full checkpoint recovery, then three measured Route/Plan/Pack ablations."""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

from utils.measured_cases import CASES, index_assets, install_ablation_snapshot, read, sha, verify_prefixes, write


def worker_command(root, dataset, stage, smoke=False):
    root = Path(root)
    snapshot = root / ('source_snapshot' if stage == 'full' else 'ablation_source_snapshot')
    protocol = read(root / 'cases' / dataset / 'full/protocol.json')
    source_data = Path(protocol['dataset']['source_data_path']).parent
    command = ['/root/anaconda3/envs/rag/bin/python','-B','-u',str(snapshot/'scripts/run_bindpack_measured.py'),
        '--dataset',dataset,'--output',str(root),'--datasets-dir',str(source_data),
        '--config',str(root/'experiment_config.json'),'--frozen-worker','--resume']
    if stage != 'full':
        command += ['--case',stage,'--reuse-index',str(root/'cases'/dataset/'full/index')]
    if smoke:
        command.append('--smoke')
    return command


def completed(case, expected):
    marker = case / 'completed.ok'
    if not marker.exists():
        return False
    receipt = read(marker)
    if receipt['query_count'] != expected:
        raise ValueError('Completed query count differs: '+str(case))
    for name in ('report','result','llm_calls'):
        suffix = '.jsonl' if name == 'llm_calls' else '.json'
        if receipt[name+'_sha256'] != sha(case/(name+suffix)):
            raise ValueError('Completed case asset changed: '+str(case/name))
    return True


def summarize(root,dataset,expected,status):
    results=[]
    for stage in CASES:
        case=root/'cases'/dataset/stage
        if completed(case,expected):
            report=read(case/'report.json')
            results.append(dict(stage_ablation=stage,queries=expected,report=str(case/'report.json'),
                metrics=report['metrics'],efficiency=report['efficiency'],cost=report['cost'],
                evidence_budget=report['evidence_budget'],resources=report['resources']))
    value=dict(status=status,dataset=dataset,case_order=list(CASES),
        completed_cases=len(results),expected_cases=4,queries_per_case=expected,
        completed_queries=len(results)*expected,expected_queries=4*expected,results=results,
        shared_index_cost_policy='one original Full build; each ablation records independent copy/load cost; historical build usage not added four times')
    write(root/'summary.json',value)
    write(root/'comparison.json',value)
    columns=['Recall@5','FCR@5','ExactMatch','F1','ClassificationAccuracy','ClassificationMacroF1','DCC@5','NodeCov@5','ClosureGap@5']
    lines=['# Measured Full and stage ablations','', '| Case | '+' | '.join(columns)+' |',
           '|---|'+'---:|'*len(columns)]
    for row in results:
        values=['N/A' if row['metrics'].get(k) is None else f"{row['metrics'][k]:.6f}" for k in columns]
        lines.append('| '+row['stage_ablation']+' | '+' | '.join(values)+' |')
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    return value


def freeze_queue(args):
    root=Path(args.output).resolve()
    runtime=root/'serial_extension/queue_runtime/scripts'
    if not runtime.exists():
        runtime.mkdir(parents=True)
        scripts=Path(__file__).resolve().parent
        for name in ('run_bindpack_measured_cases.py','run_bindpack_measured.py','utils/measured_cases.py','utils/__init__.py'):
            target=runtime/name;target.parent.mkdir(parents=True,exist_ok=True)
            if (scripts/name).exists():shutil.copy2(scripts/name,target)
            else:target.write_text('')
    command=[sys.executable,'-B','-u',str(runtime/'run_bindpack_measured_cases.py'),
        '--output',str(root),'--dataset',args.dataset,'--frozen-queue']
    if args.smoke:command.append('--smoke')
    os.execv(sys.executable,command)


def main(args):
    root=Path(args.output).resolve();dataset=args.dataset
    lock=(Path('/root/BindPack/outputs')/'.bindpack_measured_cases.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    base=read(root/'cases'/dataset/'full/protocol.json')
    expected=len(base['dataset']['selected_indices'])
    if bool(base['dataset']['development_fixture']) != args.smoke:
        raise ValueError('Smoke/full flag differs from the frozen Full batch')
    scripts=Path(__file__).resolve().parent
    install_ablation_snapshot(root,scripts)
    preserved=verify_prefixes(root)
    state={'status':'running','case_order':list(CASES),'active_case':None,'worker_pid':None,
           'completed_cases':0,'expected_cases':4,'queries_per_case':expected,
           'preserved_full_queries':preserved,'pid':os.getpid(),'executions':[]}
    stopped=threading.Event();current=[None]
    def stop_requested(signum,frame):
        stopped.set()
        if current[0] is not None and current[0].poll() is None:
            current[0].send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop_requested);signal.signal(signal.SIGINT,stop_requested)
    heartbeat_stop=threading.Event()
    def heartbeat():
        while not heartbeat_stop.is_set():
            try:
                child=read(root/'status.json')
                stage=state['active_case']
                done=child.get('completed_queries',0) if child.get('stage_ablation')==stage else 0
                phase=('starting' if child.get('stage_ablation')!=stage else
                       'finalizing_reports' if done==expected else child.get('phase','starting'))
                value=dict(state,active_completed_queries=done,total_completed_queries=state['completed_cases']*expected+done,
                    expected_queries=4*expected,active_phase=phase,updated_unix_seconds=time.time())
                write(root/'queue_status.json',value)
                print(f"{datetime.now():%Y-%m-%d %H:%M:%S} QUEUE {state['completed_cases']}/4 cases | {stage}: {done}/{expected} | {phase} | total={value['total_completed_queries']}/{4*expected}",flush=True)
            except FileNotFoundError:
                pass
            heartbeat_stop.wait(30)
    watcher=threading.Thread(target=heartbeat,daemon=True);watcher.start()
    marker_archive=root/'serial_extension/single_case_markers';marker_archive.mkdir(parents=True,exist_ok=True)
    try:
        # Original Full workers write a one-case root marker. Preserve it, then
        # publish the root completed.ok only when the four-case queue is done.
        if (root/'completed.ok').exists() and read(root/'completed.ok').get('cases')==1:
            shutil.copy2(root/'completed.ok',marker_archive/'full_completed.ok')
            (root/'completed.ok').unlink()
        for stage in CASES:
            if stopped.is_set():break
            case=root/'cases'/dataset/stage;case.mkdir(parents=True,exist_ok=True)
            state['active_case']=stage
            if completed(case,expected):
                print(f'RETAIN {dataset}/{stage}: {expected} validated queries; no model call',flush=True)
                state['completed_cases']+=1
                continue
            verify_prefixes(root)
            error=case/'error.json'
            if error.exists() and read(error).get('type')=='KeyboardInterrupt':
                shutil.copy2(error,root/'serial_extension'/f'{stage}_planned_stop_error.json')
                error.unlink()
            command=worker_command(root,dataset,stage,args.smoke)
            print(f'{datetime.now():%Y-%m-%d %H:%M:%S} START SERIAL {dataset}/{stage}',flush=True)
            tick=time.perf_counter()
            env=dict(os.environ,PYTHONPATH='/root/.cache/pathcondrag/runtime_deps_qwen3_tf4513',
                     TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
            process=subprocess.Popen(command,cwd=case,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,text=True,bufsize=1,env=env)
            current[0]=process;state['worker_pid']=process.pid
            (case/'run.pid').write_text(str(process.pid)+'\n')
            with (case/'run.log').open('a') as handle:
                for line in process.stdout:
                    handle.write(line);handle.flush();print(line,end='',flush=True)
            code=process.wait();current[0]=None
            state['executions'].append(dict(stage_ablation=stage,returncode=code,
                current_queue_wall_seconds=time.perf_counter()-tick,command=command))
            if stopped.is_set():break
            if code or not completed(case,expected):
                raise RuntimeError('Serial case failed; queue stopped: '+str(case/'run.log'))
            verify_prefixes(root)
            if stage!='full':
                reused=read(case/'index_reuse.json')
                if index_assets(case/'index') != reused['assets']:
                    raise ValueError('Ablation altered private graph/vector inputs: '+stage)
                write(case/'index_reuse_verified.ok',dict(source_and_private_assets_unchanged=True))
            state['completed_cases']+=1
            if stage=='full' and (root/'completed.ok').exists():
                shutil.copy2(root/'completed.ok',marker_archive/'full_completed.ok')
                (root/'completed.ok').unlink()
                if (root/'summary.json').exists():shutil.copy2(root/'summary.json',marker_archive/'full_summary.json')
            summarize(root,dataset,expected,'running')
        state['status']='paused_by_user' if stopped.is_set() else 'complete'
        summary=summarize(root,dataset,expected,state['status'])
        if not stopped.is_set():
            assert summary['completed_cases']==4
            write(root/'completed.ok',dict(cases=4,queries=4*expected,summary_sha256=sha(root/'summary.json')))
    except BaseException as exc:
        state.update(status='failed',error_type=type(exc).__name__,error=str(exc))
        summarize(root,dataset,expected,'failed')
        raise
    finally:
        heartbeat_stop.set();watcher.join(timeout=3)
        if (root/'completed.ok').exists() and read(root/'completed.ok').get('cases')==1:
            shutil.copy2(root/'completed.ok',marker_archive/'full_completed.ok')
            (root/'completed.ok').unlink()
        state['worker_pid']=None;state['updated_unix_seconds']=time.time()
        write(root/'queue_status.json',state)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--dataset',default='hover')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--frozen-queue',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if not args.frozen_queue:freeze_queue(args)
    main(args)

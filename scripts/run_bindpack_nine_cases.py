#!/usr/bin/env python3
"""Resume the old HoVer case, then four joint Full and four w/o Pack cases."""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading

PROJECT=Path('/root/BindPack')
ARCHIVE=PROJECT/'outputs/4_datasets_result'
PREVIOUS=ARCHIVE/'full_no_review_20261026'
DEFAULT=ARCHIVE/'pack_nine_cases_20261027'
DATASETS=('hotpotqa','2wikimultihopqa','musique','hover')


def read(path):return json.loads(Path(path).read_text())
def sha(path,limit=None):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        remaining=limit
        while remaining is None or remaining>0:
            block=stream.read(min(4*1024**2,remaining) if remaining is not None else 4*1024**2)
            if not block:break
            value.update(block)
            if remaining is not None:remaining-=len(block)
    if remaining not in (None,0):raise ValueError('Preserved prefix was truncated: '+str(path))
    return value.hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');temporary.replace(path)


def jobs(previous,new):
    return [dict(category='previous_hover',dataset='hover',case='previous_full',path=str(previous/'cases/hover/full'))]+[
        dict(category='joint_full' if stage=='full' else 'wo_pack',dataset=ds,case=stage,
             path=str(new/'cases'/ds/stage)) for stage in ('full','wo_pack') for ds in DATASETS]


def verify_preserved(protocol):
    for item in protocol['preserved_prefixes']:
        assert sha(item['path'],item['bytes'])==item['sha256'],'Preserved checkpoint/cost prefix changed: '+item['path']
    for item in protocol['immutable_previous_files']:
        assert sha(item['path'])==item['sha256'],'Original HoVer protocol changed: '+item['path']


def previous_snapshot(previous):
    base=read(previous/'queue_protocol.json')
    for rel,value in base['source_sha256'].items():assert sha(previous/'source_snapshot'/rel)==value
    overlay=read(previous/'window8_transition/overlay_receipt.json');snapshot=Path(overlay['snapshot'])
    for rel,value in overlay['source_sha256'].items():assert sha(snapshot/rel)==value
    for rel,value in base['source_sha256'].items():
        if rel.startswith(('src/','configs/')):assert sha(snapshot/rel)==value
    assert (previous/'preflight_passed.ok').exists()
    return snapshot


def prepare(root):
    protocol_path=root/'queue_protocol.json'
    if protocol_path.exists():
        protocol=read(protocol_path);verify_preserved(protocol)
        assert sha(root/'queue_runtime/run_bindpack_nine_cases.py')==protocol['queue_source_sha256']
        return protocol
    if root.exists() and any(root.iterdir()):raise ValueError('Select an empty nine-case output directory')
    root.mkdir(parents=True,exist_ok=True)
    old_snapshot=previous_snapshot(PREVIOUS)
    case=PREVIOUS/'cases/hover/full';count=0
    with (case/'progress.jsonl').open() as stream:
        for line in stream:
            item=json.loads(line);raw=json.dumps(item['row'],ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
            assert hashlib.sha256(raw).hexdigest()==item['sha256'];count+=1
    assert count==520,'Expected the preserved 520-query HoVer checkpoint'
    prefixes=[]
    for name in ('progress.jsonl','llm_calls.jsonl','llm_logical_calls.jsonl','embedding_calls.jsonl',
                 'resource_samples.jsonl','stage_timings.jsonl','execution_sessions.jsonl','retrieval_windows.jsonl'):
        path=case/name
        if path.exists():prefixes.append(dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path)))
    preserved=root/'preservation';preserved.mkdir()
    pending=case/'pending_retrieval_window.json'
    if pending.exists():shutil.copy2(pending,preserved/pending.name)
    immutable=[dict(path=str(case/name),sha256=sha(case/name)) for name in ('protocol.json','index_build.json')]
    method=root/'pack_joint_free.json';shutil.copy2(PROJECT/'configs/pack_joint_free.json',method)
    new=root/'new_scheme'
    spec=importlib.util.spec_from_file_location('native_nine_queue',PROJECT/'scripts/run_bindpack_latest_full.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    native=module.prepare(new,['full','wo_pack'],method,True)
    assert native['effective_pack']['selector_mode']=='combined'
    assert native['effective_pack']['joint_fixed_prefix']==0
    assert native['effective_pack']['relation_weight']==.6 and native['effective_pack']['review_mode']=='off'
    runtime=root/'queue_runtime';runtime.mkdir()
    shutil.copy2(Path(__file__),runtime/'run_bindpack_nine_cases.py')
    plan=jobs(PREVIOUS,new)
    protocol=dict(created_at=datetime.now().astimezone().isoformat(),previous_root=str(PREVIOUS),new_root=str(new),
        previous_snapshot=str(old_snapshot),new_snapshot=str(new/'source_snapshot'),jobs=plan,
        previous_queries_preserved=520,expected_cases=9,expected_total_queries=9000,remaining_queries_at_creation=8480,
        relation_weight=.6,pack_review_mode='off',new_pack_mode='joint_free',
        diagnostic_limit='Best macro-average diagnostic candidate; 2Wiki regressed by one FCR question in validation. Full run is an independent larger experiment, not a claim of prior global qualification.',
        preserved_prefixes=prefixes,immutable_previous_files=immutable,queue_source_sha256=sha(runtime/'run_bindpack_nine_cases.py'),
        index_policy='Reuse the same four verified archived indexes; release only verified private copies after each case',
        execution_policy='previous HoVer first; new Full on all four datasets; then w/o Pack on all four; native window8 inside each case',
        new_preflight='After old HoVer completes, run three real queries per dataset/case (24 total) before the eight full cases; stop on failure',
        metrics='Recall/FCR primary; full QA, available dependency metrics, costs, evidence budgets and resources also saved',
        cache_policy='Old HoVer keeps all caches and costs; each new case starts with its own empty application cache; preflight caches are separate')
    write(protocol_path,protocol)
    for job in plan:
        link=root/'cases'/job['dataset']/job['case'];link.parent.mkdir(parents=True,exist_ok=True)
        link.symlink_to(Path(job['path']),target_is_directory=True)
    (root/'RUN.md').write_text(f'''# 九组串行实验

1. 旧HoVer Full从520/1000接续，沿用其冻结源码、pending窗口、缓存和全部成本。
2. 新联合方案Full：HotpotQA → 2WikiMultiHopQA → MuSiQue → HoVer，各1000题。
3. w/o Pack：相同四数据集顺序，各1000题。不会夹在Full之间执行。

新方案为joint_free，权重0.60、Support / semantic veto关闭；Top2可替换，状态上限4096、beam12、应急搜索时限5秒。它在诊断集有收益但2Wiki曾回退，本轮独立全量验证不预先声称全面提升。

复用四份原索引，原生8题窗口、LLM预取8、embedding batch4；模型与vLLM保持现有Qwen3配置。正式九组共9000题，其中旧520题保留，尚需8480题。新八组之前自动做24题真实预检，失败停止，不跳过错误继续发布结果。各组分别计时/计费，索引历史建库成本不可得时N/A。

结果入口：cases/<dataset>/full、wo_pack；旧HoVer为cases/hover/previous_full。均为原结果位置的链接，不复制索引或结果。新结果实际保存在new_scheme，旧HoVer仍在原目录。

日志：`tail -f {root}/run.log`

中断后续跑：`bash /root/BindPack/scripts/run_bindpack_nine_cases.sh --output {root}`

控制源码及新方案原生源码/配置已冻结，工作区后续变化不会改变队列。失败可修复后沿原断点恢复，不覆盖已完成结果。旧HoVer的520条及账本前缀始终作哈希核验。
''')
    return protocol


def prepare_workers(root,protocol):
    new=Path(protocol['new_root']);snapshot=Path(protocol['new_snapshot'])
    command=[sys.executable,'-B','-u',str(snapshot/'scripts/run_bindpack_latest_full.py'),
        '--output',str(new),'--cases','full','wo_pack','--case-first','--prepare-only','--frozen-queue']
    subprocess.run(command,check=True)
    sys.path[:0]=[str(snapshot/'scripts'),str(snapshot/'src')]
    from utils.archived_index import validate_archive
    verified={}
    for ds in DATASETS:
        cfg=read(new/'runs'/ds/'cases'/ds/'full/protocol.json')['configuration']
        corpus=read(new/'runs'/ds/'metadata'/ds/'corpus.json')
        verified[ds]=validate_archive(ARCHIVE/'shared_indexes'/ds,ds,corpus,cfg['embedding_model_name'],cfg['llm_name'],
            dict(normalized=cfg['embedding_return_as_normalized'],max_sequence_length=cfg['embedding_max_seq_len'],dtype=cfg['embedding_model_dtype']))
        for stage in ('full','wo_pack'):
            p=read(new/'runs'/ds/'cases'/ds/stage/'protocol.json')
            assert p['stage_ablation']==stage and p['query_concurrency']==8
            assert p['configuration']['pack']['joint_fixed_prefix']==0 and p['configuration']['pack']['review_mode']=='off'
    write(root/'prepared.ok',dict(native_case_protocols=8,indexes_verified=4,model_inference_calls=0,index_build_calls=0,
                                  archived_indexes={ds:{k:v[k] for k in ('source','corpus_sha256','models')} for ds,v in verified.items()}))


def passed(job,deep=False):
    case=Path(job['path']);marker=case/'completed.ok'
    if not marker.exists():return False
    receipt=read(marker);assert receipt['query_count']==1000
    if not (case/'private_assets_released.ok').exists():return False
    if deep:
        for name in ('report','result','llm_calls'):
            suffix='.jsonl' if name=='llm_calls' else '.json'
            assert sha(case/(name+suffix))==receipt[name+'_sha256']
        assert read(case/'audit.json')['status']=='passed'
        assert read(case/'index_reuse_verified.ok')['source_and_private_unchanged']
    return True


def publish(root,protocol,status):
    values=[];lines=['# 九组实验结果','',f'状态：{status}。先旧HoVer、四个新Full、四个w/o Pack。','',
        '|顺序|数据集|Case|Recall@5|FCR@5|EM/Accuracy|F1/Macro-F1|DCC@5|NodeCov@5|ClosureGap@5|',
        '|---:|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for number,job in enumerate(protocol['jobs'],1):
        if not passed(job):continue
        report=read(Path(job['path'])/'report.json');m=report['metrics']
        qa=('ClassificationAccuracy','ClassificationMacroF1') if job['dataset']=='hover' else ('ExactMatch','F1')
        keys=['Recall@5','FCR@5',*qa,'DCC@5','NodeCov@5','ClosureGap@5']
        vals=['N/A' if m.get(k) is None else f'{m[k]:.6f}' for k in keys]
        lines.append('|'+ '|'.join([str(number),job['dataset'],job['case'],*vals])+'|')
        values.append(dict(order=number,**job,report=report))
    write(root/'summary.json',dict(status=status,completed_cases=len(values),expected_cases=9,
        completed_queries=1000*len(values),expected_queries=9000,results=values))
    (root/'results_at5.md').write_text('\n'.join(lines)+'\n')
    return len(values)


def main(root,protocol):
    if not (root/'prepared.ok').exists():raise ValueError('Run --prepare-only validation before launch')
    lock=(root/'.queue.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (root/'run.pid').write_text(str(os.getpid())+'\n')
    verify_preserved(protocol);stop=threading.Event();watch_stop=threading.Event();current=[None]
    state=dict(status='running',phase='starting',pid=os.getpid(),expected_cases=9,completed_cases=0)
    def request_stop(number,frame):
        stop.set()
        if current[0] is not None and current[0].poll() is None:current[0].send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    def heartbeat():
        while not watch_stop.is_set():
            try:
                done=sum(passed(job) for job in protocol['jobs']);active={}
                child=Path(protocol['previous_root'] if state['phase']=='resume_previous' else protocol['new_root'])
                if (child/'queue_status.json').exists():active=read(child/'queue_status.json')
                progress=active.get('active',{})
                value=dict(state,completed_cases=done,active=active,updated_at=datetime.now().astimezone().isoformat())
                write(root/'status.json',value)
                print(f"{datetime.now():%Y-%m-%d %H:%M:%S} NINE [{done}/9] | {state['phase']} | {active.get('phase','starting')} {active.get('dataset')}/{active.get('case')} {progress.get('completed_queries','?')}/{progress.get('expected_queries','?')}",flush=True)
            except (FileNotFoundError,json.JSONDecodeError):pass
            watch_stop.wait(30)
    watcher=threading.Thread(target=heartbeat,daemon=True);watcher.start()
    def execute(snapshot,output,extra):
        command=[sys.executable,'-B','-u',str(snapshot/'scripts/run_bindpack_latest_full.py'),
            '--output',str(output),*extra,'--frozen-queue']
        print('START',state['phase'],' '.join(command),flush=True)
        process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            text=True,bufsize=1,start_new_session=True)
        current[0]=process
        with (output/'run.log').open('a') as log:
            for line in process.stdout:log.write(line);log.flush();print(line,end='',flush=True)
        code=process.wait();current[0]=None
        if stop.is_set():raise KeyboardInterrupt('Stopped at durable checkpoint; resume the same nine-case queue')
        if code:raise RuntimeError('Child queue failed; stopped without starting later groups: '+str(output/'run.log'))
        subprocess.run([sys.executable,'-B',str(snapshot/'scripts/finalize_bindpack_latest_full.py'),'--output',str(output)],check=True)
    try:
        old=protocol['jobs'][0]
        if not passed(old,deep=True):
            state['phase']='resume_previous'
            execute(Path(protocol['previous_snapshot']),Path(protocol['previous_root']),['--cases','full','--skip-preflight'])
        assert passed(old,deep=True);verify_preserved(protocol);publish(root,protocol,'running')
        if stop.is_set():raise KeyboardInterrupt('Requested stop')
        state['phase']='new_full_then_wo_pack'
        if not all(passed(job,deep=True) for job in protocol['jobs'][1:]):
            execute(Path(protocol['new_snapshot']),Path(protocol['new_root']),['--cases','full','wo_pack','--case-first'])
        assert all(passed(job,deep=True) for job in protocol['jobs']);verify_preserved(protocol)
        assert publish(root,protocol,'complete')==9
        write(root/'completed.ok',dict(cases=9,queries=9000,old_prefix_queries_preserved=520))
        state['status']='complete'
    except BaseException as exc:
        state.update(status='stopped' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc))
        publish(root,protocol,state['status']);raise
    finally:
        watch_stop.set();watcher.join(timeout=3);state['completed_cases']=sum(passed(job) for job in protocol['jobs'])
        write(root/'status.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--frozen-queue',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args();root=args.output.resolve();protocol=prepare(root)
    if not args.frozen_queue:
        os.execv(sys.executable,[sys.executable,'-B','-u',str(root/'queue_runtime/run_bindpack_nine_cases.py'),*sys.argv[1:],'--frozen-queue'])
    if args.prepare_only:prepare_workers(root,protocol);print('Nine-case queue prepared; no inference',flush=True)
    else:main(root,protocol)

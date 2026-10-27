#!/usr/bin/env python3
"""Serial measured full-dataset runs using immutable archived indexes."""
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

PROJECT=Path('/root/BindPack')
ARCHIVE=PROJECT/'outputs/4_datasets_result'
DEFAULT=ARCHIVE/'full_no_review_20261026'
DATASETS=('hotpotqa','2wikimultihopqa','musique','hover')
CASES=('full','wo_route','wo_plan','wo_pack')


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)


def execution_snapshot(root):
    receipt=root/'window8_transition/overlay_receipt.json'
    if not receipt.exists():return root/'source_snapshot'
    value=read(receipt);snapshot=Path(value['snapshot'])
    base=read(root/'queue_protocol.json')['source_sha256']
    for rel,expected in base.items():
        if rel.startswith(('src/','configs/')):assert sha(snapshot/rel)==expected, 'Algorithm/config changed: '+rel
    for rel,expected in value['source_sha256'].items():assert sha(snapshot/rel)==expected
    return snapshot


def case_order(datasets,cases,case_first=False):
    return [(ds,stage) for stage in cases for ds in datasets] if case_first else [(ds,stage) for ds in datasets for stage in cases]


def prepare(root,cases,method_config=None,case_first=None):
    if (root/'queue_protocol.json').exists():
        protocol=read(root/'queue_protocol.json');assert protocol['cases']==list(cases)
        if case_first is not None:assert protocol.get('case_first',False)==case_first, 'Frozen case order changed'
        if method_config is not None:
            assert protocol.get('method_configuration',{}).get('sha256')==sha(method_config), 'Frozen method configuration changed'
        assert sha(ARCHIVE/'manifest.json')==protocol['archive_manifest_sha256']
        for rel,value in protocol['source_sha256'].items():assert sha(root/'source_snapshot'/rel)==value
        for value in protocol['frozen_inputs'].values():assert sha(value['path'])==value['sha256']
        return protocol
    if any((root/name).exists() for name in ('source_snapshot','runs','cases')):
        raise ValueError('Untracked output tree; select a fresh run directory')
    root.mkdir(parents=True,exist_ok=True)
    snapshot=root/'source_snapshot'
    for name in ('src','scripts','configs'):
        shutil.copytree(PROJECT/name,snapshot/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    config=read(PROJECT/'configs/ablations_qwen3.json')
    method_receipt=None
    if method_config is not None:
        method_config=Path(method_config).resolve()
        target=root/'method_overrides.json';shutil.copy2(method_config,target)
        method_receipt=dict(original_path=str(method_config),path=str(target),sha256=sha(target))
        config['method_config']=str(target)
    frozen_data=root/'frozen_datasets';frozen_data.mkdir(exist_ok=True)
    inputs={}
    if method_receipt:inputs['method_overrides']=dict(path=method_receipt['path'],sha256=method_receipt['sha256'])
    for ds in DATASETS:
        for suffix in ('','_corpus'):
            source=PROJECT/'datasets'/(ds+suffix+'.json');target=frozen_data/source.name
            shutil.copy2(source,target);inputs[source.name]=dict(path=str(target),sha256=sha(target))
        if len(read(frozen_data/(ds+'.json')))!=1000:raise ValueError('Expected all 1000 local queries: '+ds)
    config.update(datasets=list(DATASETS),cases=list(cases),datasets_dir=str(frozen_data))
    config['evaluation'].update(eval_mode='rag_qa',sample_size=0,qa_top_k=5)
    write(root/'run_config.json',config)
    method=read(snapshot/'src/bindpack/configuration/defaults.json')
    if method_receipt:
        override=read(method_receipt['path'])
        if set(override)!={'pack'}:raise ValueError('This queue accepts Pack-only method overrides')
        method['pack'].update(override['pack'])
    assert method['pack']['relation_weight']==.6
    hashes={str(p.relative_to(snapshot)):sha(p) for p in snapshot.rglob('*') if p.is_file() and p.suffix in ('.py','.json')}
    protocol=dict(created_at=datetime.now().astimezone().isoformat(),datasets=list(DATASETS),cases=list(cases),
        queries_per_dataset=1000,expected_queries=4000*len(cases),relation_weight=.6,
        adopted_pack_changes=['goal_redundancy','short_circuit'],source_sha256=hashes,
        method_configuration=method_receipt,effective_pack=method['pack'],case_first=bool(case_first),
        execution_order=[dict(dataset=ds,case=stage) for ds,stage in case_order(DATASETS,cases,bool(case_first))],
        pack_review_mode=method['pack']['review_mode'],
        disabled_components=['support_semantic_veto'] if method['pack']['review_mode']=='off' else [],
        index_source=str(ARCHIVE/'shared_indexes'),dataset_source=str(PROJECT/'datasets'),
        archive_manifest_sha256=sha(ARCHIVE/'manifest.json'),
        frozen_inputs=inputs,
        algorithm='native BindPack with selector_mode='+method['pack'].get('selector_mode','current')+'; no frozen-query replay',query_concurrency=8,llm_prefetch_workers=8,embedding_batch_size=4,
        hop_policy='Current corrected annotation-aware reader, including 2Wiki 2/4 buckets; oracle-assisted budget disclosed',
        cache_policy='fresh independent application cache per case; no historical/preflight cache; shared vLLM not reset',
        accounting='native measured end-to-end retrieval/Route/Plan/Pack+reader; original indexing cost N/A if not measured')
    write(root/'queue_protocol.json',protocol)
    for phase in ('preflight','runs'):
        for ds in DATASETS:
            directory=root/phase/ds;directory.mkdir(parents=True,exist_ok=True)
    (root/'cases').mkdir(exist_ok=True)
    for ds in DATASETS:
        target=root/'cases'/ds
        if not target.exists():target.symlink_to(Path('../runs')/ds/'cases'/ds,target_is_directory=True)
    (root/'RUN.md').write_text(f'''# 最新方案四数据集重跑

Full全量使用每个本地数据集1000题，权重0.60；采用目标感知去重和无效请求短路。配置/源码冻结，后台后续修改工作区不会改变本轮。

末端核验模式：`{method['pack']['review_mode']}`。`off` 只关闭 Support / semantic veto，保留 Pack 贪心覆盖/去重及 DAG 选择；本轮仍标记为 Full，不等同于 wo_pack。旧核验开启的实验断点不混入本轮。

数据：/root/BindPack/datasets；索引：{ARCHIVE}/shared_indexes。当前修复后的native hop规则生效，2Wiki为2/4分桶；不是之前冻结historical prior=2的诊断协议。

运行先在四份完整索引上分别测试3题，检查指标/计量/源码和索引完整性。随后按数据集、case串行全量运行；同一数据集原生8题窗口预取，QA与答案断点逐题保存。模型失败或未通过校验则停止队列。

结果入口：cases/<dataset>/<case>/report.json、metrics_details.json、result.json，以及逐题进度/LLM/embedding/资源账本。实际目录为runs/<dataset>/cases/<dataset>/<case>。

只复制当前正在运行的索引，验证源/副本一致后释放生成的副本，保留所有结果、断点和本case缓存。不会改写旧归档，也不会重建图/向量；历史原始建库成本不可得时保留N/A。

日志：`tail -f {root}/run.log`

中断后继续：`bash /root/BindPack/scripts/run_bindpack_latest_full.sh --output {root} --skip-preflight`

选定case：{', '.join(cases)}；全部规模：{4000*len(cases)}题。
''')
    return protocol


def command(root,ds,stage,preflight=False,prepare_only=False):
    output=root/('preflight' if preflight else 'runs')/ds
    args=[sys.executable,'-B','-u',str(execution_snapshot(root)/'scripts/run_bindpack_measured.py'),
        '--frozen-worker','--dataset',ds,'--case',stage,'--output',str(output),
        '--config',str(root/'run_config.json'),'--datasets-dir',str(root/'frozen_datasets'),
        '--reuse-archive-index',str(ARCHIVE/'shared_indexes'/ds),'--query-window','8']
    if (output/'experiment_config.json').exists():args.append('--resume')
    if preflight:args+=['--query-limit','3']
    if prepare_only:args.append('--prepare-only')
    return args


def completed(case,expected):
    marker=case/'completed.ok'
    if not marker.exists():return False
    value=read(marker);assert value['query_count']==expected
    for name in ('result','report','llm_calls'):
        suffix='.jsonl' if name=='llm_calls' else '.json'
        assert sha(case/(name+suffix))==value[name+'_sha256']
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--cases',nargs='+',choices=CASES,default=['full']);p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--method-config',type=Path,help='Freeze a Pack-only configuration override for this independent run')
    p.add_argument('--case-first',action='store_true',default=None,help='Finish all four Full datasets before starting the next case')
    p.add_argument('--preflight-only',action='store_true');p.add_argument('--skip-preflight',action='store_true')
    p.add_argument('--frozen-queue',action='store_true',help=argparse.SUPPRESS);a=p.parse_args()
    root=a.output.resolve();protocol=prepare(root,a.cases,a.method_config,a.case_first)
    if not a.frozen_queue:
        os.execv(sys.executable,[sys.executable,'-B','-u',str(execution_snapshot(root)/'scripts/run_bindpack_latest_full.py'),*sys.argv[1:],'--frozen-queue'])
    if a.prepare_only:
        for ds,stage in case_order(DATASETS,a.cases,bool(protocol.get('case_first'))):
            subprocess.run(command(root,ds,stage,prepare_only=True),check=True)
        print('Selected full-dataset protocols prepared; no model requests.',flush=True);return
    with (root/'.queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(root/'run.pid').write_text(str(os.getpid())+'\n')
        current=[None];stop=threading.Event();watch_stop=threading.Event()
        state=dict(status='running',pid=os.getpid(),phase='starting',dataset=None,case=None,completed_cases=0,expected_cases=4*len(a.cases),completed_queries=0,expected_queries=protocol['expected_queries'],query_window_size=8)
        previous_summary=read(root/'summary.json') if (root/'summary.json').exists() else {}
        write(root/'summary.json',dict(previous_summary,status='running',execution_mode='native_window8'))
        def request_stop(signum,frame):
            stop.set()
            if current[0] is not None and current[0].poll() is None:current[0].send_signal(signal.SIGTERM)
        signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
        def heartbeat():
            while not watch_stop.is_set():
                try:
                    ds=state['dataset'];phase=state['phase'];progress={}
                    if ds:
                        path=root/('preflight' if phase=='preflight' else 'runs')/ds/'status.json'
                        if path.exists():progress=read(path)
                    write(root/'queue_status.json',dict(state,active=progress,updated_at=datetime.now().astimezone().isoformat()))
                    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} QUEUE {state['completed_cases']}/{state['expected_cases']} cases | {phase} {ds}/{state['case']} {progress.get('completed_queries',0)}/{progress.get('expected_queries','?')} | {progress.get('phase','starting')}",flush=True)
                except (FileNotFoundError,json.JSONDecodeError):pass
                watch_stop.wait(30)
        watcher=threading.Thread(target=heartbeat,daemon=True);watcher.start()
        def execute(ds,stage,preflight):
            assert sha(ARCHIVE/'manifest.json')==protocol['archive_manifest_sha256'], 'Archived index inventory changed'
            output=root/('preflight' if preflight else 'runs')/ds;case=output/'cases'/ds/stage
            state.update(dataset=ds,case=stage,phase='preflight' if preflight else 'full')
            if not completed(case,3 if preflight else 1000):
                case.mkdir(parents=True,exist_ok=True)
                argv=command(root,ds,stage,preflight)
                print('START',state['phase'],ds,stage,flush=True)
                process=subprocess.Popen(argv,cwd=case,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                current[0]=process;(case/'run.pid').write_text(str(process.pid)+'\n')
                with (case/'run.log').open('a') as log:
                    for line in process.stdout:log.write(line);log.flush();print(line,end='',flush=True)
                code=process.wait();current[0]=None
                if stop.is_set():raise KeyboardInterrupt('Requested stop; query checkpoints and costs retained')
                if code or not completed(case,3 if preflight else 1000):raise RuntimeError('Case failed; queue stopped: '+str(case/'run.log'))
            if not (case/'private_assets_released.ok').exists():
                sys.path.insert(0,str(root/'source_snapshot/scripts'))
                from utils.archived_index import verify_private
                verify_private(case,release=True)
            print('PASS',state['phase'],ds,stage,flush=True)
        try:
            if not a.skip_preflight:
                for ds,stage in case_order(DATASETS,a.cases,bool(protocol.get('case_first'))):execute(ds,stage,True)
                write(root/'preflight_passed.ok',dict(datasets=4,queries=12*len(a.cases),cases=list(a.cases),full_corpus_indexes=True,all_cases_passed=True))
            elif not (root/'preflight_passed.ok').exists():raise ValueError('Cannot skip an unvalidated preflight')
            if not a.preflight_only:
                order=protocol.get('execution_order',[dict(dataset=ds,case=stage) for ds,stage in case_order(DATASETS,a.cases)])
                for item in order:
                    ds,stage=item['dataset'],item['case']
                    execute(ds,stage,False);state['completed_cases']+=1;state['completed_queries']+=1000
                    reports={}
                    for d in DATASETS:
                        for c in a.cases:
                            path=root/'cases'/d/c/'report.json'
                            if path.exists():reports[d+'/'+c]=read(path)
                    write(root/'summary.json',dict(status='running',results=reports))
                write(root/'completed.ok',dict(status='complete',cases=state['expected_cases'],queries=protocol['expected_queries']))
            state['status']='preflight_complete' if a.preflight_only else 'complete'
        except BaseException as exc:
            state.update(status='stopped' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc));raise
        finally:
            watch_stop.set();watcher.join(timeout=3);write(root/'queue_status.json',state)


if __name__=='__main__':main()

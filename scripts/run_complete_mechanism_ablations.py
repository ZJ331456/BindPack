#!/usr/bin/env python3
"""Twelve independent, upstream-reexecuted native mechanism experiments."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import argparse,fcntl,hashlib,json,os,shutil,signal,subprocess,sys,threading,time

PROJECT=Path('/root/BindPack')
ARCHIVE=PROJECT/'outputs/4_datasets_result'
REFERENCE=ARCHIVE/'pack_nine_cases_20261027/new_scheme'
DEFAULT=ARCHIVE/'mechanism_complete_20261027'
DATASETS=('hotpotqa','2wikimultihopqa','musique','hover')
ARMS=('wo_dependency_closure','wo_source_bound_eligibility','wo_joint_selection')

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def hashes(snapshot):return {str(p.relative_to(snapshot)):sha(p) for p in snapshot.rglob('*') if p.is_file() and p.suffix in ('.py','.json')}

def prepare(root):
    protocol=root/'protocol.json'
    if protocol.exists():
        p=read(protocol)
        assert hashes(root/'source_snapshot')==p['source_sha256'],'Frozen code changed'
        for info in p['inputs'].values():assert sha(info['path'])==info['sha256']
        for ds,info in p['reference_full'].items():assert sha(info['path'])==info['sha256']
        return p
    snapshot=root/'source_snapshot'
    assert not snapshot.exists(),'Untracked source snapshot'
    for folder in ('src','scripts','configs'):
        shutil.copytree(PROJECT/folder,snapshot/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    inputs={}
    for ds in DATASETS:
        for suffix in ('','_corpus'):
            source=PROJECT/'datasets'/(ds+suffix+'.json')
            reference=REFERENCE/'runs'/ds/source.name
            assert read(source)==read(reference),'Dataset differs from latest Full: '+str(source)
            target=root/'frozen_datasets'/source.name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target);inputs[source.name]={'path':str(target),'sha256':sha(target)}
        assert len(read(root/'frozen_datasets'/(ds+'.json')))==1000
    config=read(PROJECT/'configs/ablations_qwen3.json')
    config.update(datasets=list(DATASETS),datasets_dir=str(root/'frozen_datasets'))
    config['evaluation'].update(eval_mode='retrieve',qa_top_k=5)
    write(root/'run_config.json',config)
    inputs['run_config']={'path':str(root/'run_config.json'),'sha256':sha(root/'run_config.json')}
    refs={ds:{'path':str(REFERENCE/'cases'/ds/'full/report.json'),
              'sha256':sha(REFERENCE/'cases'/ds/'full/report.json')} for ds in DATASETS}
    p={'schema_version':2,'created_at':datetime.now().astimezone().isoformat(),
       'scope':'End-to-end native mechanism interventions in Plan and Pack; Route is freshly rerun',
       'datasets':list(DATASETS),'arms':list(ARMS),'queries_per_case':1000,'ablation_cases':12,
       'upstream_reexecuted':True,'historical_responses_replayed':False,'qa_executed':False,
       'source_sha256':hashes(snapshot),'inputs':inputs,'reference_full':refs,
       'index_source':str(ARCHIVE/'shared_indexes'),'index_rebuilt':False,
       'model_policy':'existing Qwen3-8B vLLM, Qwen3-Embedding-8B, embedding batch4, native window8/prefetch8; vLLM unchanged',
       'review_mode':'off','relation_weight':.6,'budget':5,'state_cap':4096,'beam_width':12,
       'cache_policy':'fresh independent application cache per case; preflight caches never enter full cases; shared vLLM cache not reset',
       'hop_protocol':'same oracle native benchmark hops as latest Full',
       'interventions':{
           'wo_source_bound_eligibility':{
               'disabled':['Plan exact-source verification protocol and quote/answer gates',
                           'Pack/pool branch-compatibility scoring','old/new DAG source qualification',
                           'joint source admission and source retention qualification','source-visible quote requirement'],
               'replacement':'one-pass ordinary intermediate-answer extraction; schema and in-range document IDs retained; unverified records explicitly labeled',
               'retained':['dependency planning/execution','closure selection using unverified claimed sources','joint set selection']},
           'wo_dependency_closure':{
               'disabled':['dependency planner','topological execution','answer propagation','ancestor requirements',
                           'old DAGPackageSelector','joint closure reward','all closure-package search actions'],
               'replacement':'standalone independent questions, parallel independent verification; no flat-node replacement closure reward',
               'soft_weight':'existing missing-proof-channel fallback rho=1; no tuned coefficients',
               'retained':['source verification','pointwise goal scores','soft coverage','support retention','joint search over singleton actions']},
           'wo_joint_selection':{
               'disabled':['entire BudgetedSelector, its joint objective and singleton/package beam search'],
               'replacement':'sequential CoverageSelector then DAGPackageSelector',
               'retained':['source verification and eligibility','dependency execution','old sequential closure selection']}},
       'metrics':'Recall/FCR@3/5/8/10/20/200 plus available gold structural metrics; QA N/A',
       'storage':'top10 passage texts, complete ranking content hashes, full gold ranks and complete native traces; original corpus/index retained',
       'comparison_limit':'Fresh independent upstream runs; no claim that every cross-run difference is caused solely by final Pack ranking'}
    write(protocol,p)
    for phase in ('preflight','runs'):
        for arm in ARMS:
            for ds in DATASETS:
                case=root/phase/arm/ds/'cases'/ds/'full';case.mkdir(parents=True,exist_ok=True)
                if phase=='runs':
                    link=root/'cases'/ds/arm;link.parent.mkdir(parents=True,exist_ok=True)
                    link.symlink_to(case,target_is_directory=True)
    return p

def complete(case,n):
    marker=case/'completed.ok'
    if not marker.exists():return False
    data=read(marker);assert data['query_count']==n
    for name in ('result','report','llm_calls'):
        path=case/(name+('.jsonl' if name=='llm_calls' else '.json'))
        assert sha(path)==data[name+'_sha256']
    assert read(case/'audit.json')['mechanism_audit']['status']=='passed'
    assert read(case/'report.json')['coverage']['qa_n']==0
    return True

def summarize(root,state):
    protocol=read(root/'protocol.json');rows=[]
    lines=['# 完整机制消融（重新执行上游）','',
           '每组1000题；新消融真实执行Route、Plan、Pack，不生成最终QA；复用同一索引。',
           'Full引用最新已完成全量，并另做4000题默认Pack兼容性核验。各关闭项见协议和RUN.md。','',
           '|数据集|方案|Recall@5|FCR@5|ΔRecall|ΔFCR|','|---|---|---:|---:|---:|---:|']
    for ds in DATASETS:
        base=read(protocol['reference_full'][ds]['path'])['metrics']
        lines.append(f'|{ds}|Full（既有全量）|{base["Recall@5"]:.6f}|{base["FCR@5"]:.6f}|—|—|')
        for arm in ARMS:
            case=root/'cases'/ds/arm
            if not (case/'completed.ok').exists():continue
            m=read(case/'report.json')['metrics']
            rows.append({'dataset':ds,'arm':arm,'Recall@5':m['Recall@5'],'FCR@5':m['FCR@5']})
            lines.append(f'|{ds}|{arm}|{m["Recall@5"]:.6f}|{m["FCR@5"]:.6f}|{m["Recall@5"]-base["Recall@5"]:+.6f}|{m["FCR@5"]-base["FCR@5"]:+.6f}|')
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    write(root/'summary.json',{'status':state,'completed_ablation_cases':len(rows),'expected_ablation_cases':12,'results':rows,'qa_executed':False,'upstream_reexecuted':True})

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--frozen-queue',action='store_true');args=p.parse_args()
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True);protocol=prepare(root)
    if not args.frozen_queue:
        os.execv(sys.executable,[sys.executable,'-B','-u',str(root/'source_snapshot/scripts/run_complete_mechanism_ablations.py'),*sys.argv[1:],'--frozen-queue'])
    if args.prepare_only:return
    lock=(root/'.queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (root/'run.pid').write_text(str(os.getpid())+'\n')
    state={'status':'running','pid':os.getpid(),'completed_cases':0,'expected_cases':12,
           'preflight_completed':0,'expected_preflight_cases':12,'phase':'starting'}
    stopping=threading.Event();done=threading.Event();current=[None]
    def stop(*unused):
        stopping.set()
        if current[0] is not None and current[0].poll() is None:current[0].send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def heartbeat():
        while not done.is_set():
            data=dict(state);path=Path(state['worker_output'])/'status.json' if state.get('worker_output') else None
            if path and path.exists():data['active']=read(path)
            data['updated_at']=datetime.now().astimezone().isoformat();write(root/'status.json',data)
            a=data.get('active',{})
            print(f"PROGRESS preflight {data['preflight_completed']}/12 | full {data['completed_cases']}/12 | {data.get('phase')} {data.get('dataset')}/{data.get('arm')} | {a.get('completed_queries',0)}/{a.get('expected_queries','?')} | {a.get('phase','starting')}",flush=True)
            done.wait(30)
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    def execute(ds,arm,preflight):
        out=root/('preflight' if preflight else 'runs')/arm/ds;case=out/'cases'/ds/'full';n=3 if preflight else 1000
        state.update(phase='preflight' if preflight else 'full',dataset=ds,arm=arm,worker_output=str(out))
        if not complete(case,n):
            command=[sys.executable,'-B','-u',str(root/'source_snapshot/scripts/run_bindpack_measured.py'),
                '--frozen-worker','--dataset',ds,'--output',str(out),'--config',str(root/'run_config.json'),
                '--datasets-dir',str(root/'frozen_datasets'),'--reuse-archive-index',str(ARCHIVE/'shared_indexes'/ds),
                '--query-window','8','--eval-mode','retrieve','--candidate-output-top-k','10','--mechanism-ablation',arm]
            if preflight:command+=['--query-limit','3']
            if (out/'experiment_config.json').exists():command+=['--resume']
            print('START',state['phase'],ds,arm,flush=True)
            process=subprocess.Popen(command,cwd=case,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            current[0]=process
            with (case/'run.log').open('a') as log:
                for line in process.stdout:log.write(line);log.flush();print(line,end='',flush=True)
            code=process.wait();current[0]=None
            if stopping.is_set():raise KeyboardInterrupt('Stopped at durable native checkpoint')
            if code or not complete(case,n):raise RuntimeError('Native mechanism case failed: '+str(case/'run.log'))
        if not (case/'private_assets_released.ok').exists():
            sys.path.insert(0,str(root/'source_snapshot/scripts'))
            from utils.archived_index import verify_private
            verify_private(case,release=True)
        print('PASS',state['phase'],ds,arm,flush=True)
    try:
        for ds in DATASETS:
            for arm in ARMS:
                if stopping.is_set():raise KeyboardInterrupt('Requested stop')
                execute(ds,arm,True);state['preflight_completed']+=1
        write(root/'preflight_passed.ok',{'cases':12,'queries':36,'native_mechanism_audits':True,'qa_calls':0})
        assert read(root/'validation/full_compatibility.json')['status']=='passed','Full compatibility must pass before full experiments'
        if not args.preflight_only:
            for arm in ARMS:
                for ds in DATASETS:
                    if stopping.is_set():raise KeyboardInterrupt('Requested stop')
                    execute(ds,arm,False);state['completed_cases']+=1;summarize(root,'running')
            write(root/'completed.ok',{'cases':12,'queries':12000,'qa_executed':False})
        state['status']='preflight_complete' if args.preflight_only else 'complete'
    except BaseException as error:
        state.update(status='stopped' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error));raise
    finally:
        done.set();thread.join(timeout=3);write(root/'status.json',state);summarize(root,state['status'])

if __name__=='__main__':main()

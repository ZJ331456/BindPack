#!/usr/bin/env python3
"""Frozen-upstream Pack controls, queued after the current measured experiment."""
from __future__ import annotations
import argparse,copy,fcntl,gzip,hashlib,json,os,signal,subprocess,sys,threading,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

PROJECT=Path('/root/BindPack')
CODE=Path(__file__).resolve().parents[1]
ARCHIVE=PROJECT/'outputs/4_datasets_result'
SOURCE=ARCHIVE/'mechanism_complete_20261027'
PRIOR=ARCHIVE/'new_full_pilot_20261028'
DATASETS=('hotpotqa','2wikimultihopqa','musique','hover')
ARMS=('full','wo_joint_selection','flat_dependency_support','wo_support_retention','wo_residual_coverage','singleton_search')
ENV={}

def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4194304),b''):h.update(block)
    return h.hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def token(sid):return hashlib.sha256(sid.encode()).hexdigest()+'.json.gz'
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)
def code_hashes(path):return {str(p.relative_to(path)):sha(p) for p in path.rglob('*') if p.is_file() and p.suffix in ('.py','.json')}

def prepare(root):
    if (root/'protocol.json').exists():
        p=read(root/'protocol.json');assert code_hashes(root/'source_snapshot')==p['source_sha256']
        for ds,entry in p['references'].items():
            assert sha(Path(entry['case'])/'progress.jsonl')==entry['progress_sha256']
        assert digest(read(root/'selection_locked.json'))==p['selection_sha256']
        return
    assert read(root/'validation/tests.json')['status']=='passed'
    import shutil
    snapshot=root/'source_snapshot'
    for folder in ('src','scripts','configs'):
        shutil.copytree(PROJECT/folder,snapshot/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    selections={};references={}
    for ds in DATASETS:
        case=SOURCE/'cases'/ds/'wo_source_bound_eligibility'
        done=read(case/'completed.ok');assert done['query_count']==1000
        for name in ('report','result','llm_calls'):
            path=case/(name+('.jsonl' if name=='llm_calls' else '.json'));assert sha(path)==done[name+'_sha256']
        refs={}
        for line in (case/'progress.jsonl').open():
            event=json.loads(line);row=event['row'];assert digest(row)==event['sha256']
            trace=row['retrieval_trace']['evidence'];joint=trace['budgeted_selection']
            ids=trace['improvement_support_semantic_veto']['final_doc_ids']
            refs[row['sample_id']]=dict(query_index=row['query_index'],full_ids=ids,
                s0=joint['reference_doc_ids'],domain=joint['candidate_domain'],objective=joint['objective'],
                rho=joint['soft_weight_effective'],gold_hashes=[hashlib.sha256(d.encode()).hexdigest() for d in row['gold_docs']],
                content_by_id=dict(zip(ids,row['complete_ranking_content_sha256'])))
        assert len(refs)==1000
        ordered=sorted(refs,key=lambda sid:hashlib.sha256(('BindPack/new_full_pilot/20261028|'+ds+'|'+sid).encode()).hexdigest())
        selections[ds]={'pilot':ordered[:200],'validation':ordered[200:400]}
        if ds in ('hotpotqa','2wikimultihopqa'):
            original=read(PRIOR/'selections.json')[ds]
            assert original['pilot']['sample_ids']==selections[ds]['pilot']
            assert original['validation']['sample_ids']==selections[ds]['validation']
        write(root/'metadata'/ds/'references.json',refs)
        references[ds]=dict(case=str(case.resolve()),progress_sha256=sha(case/'progress.jsonl'),report_sha256=sha(case/'report.json'))
    selected={'datasets':selections,'rule':'same fixed SHA256 cohorts as the prior pilot, 200+200 per dataset; disjoint, outcome-blind; all controls run both cohorts regardless of sign'}
    write(root/'selection_locked.json',selected)
    write(root/'protocol.json',dict(schema_version=1,source_sha256=code_hashes(snapshot),references=references,
        datasets=list(DATASETS),arms=list(ARMS),selection_sha256=digest(selected),
        source_bound_eligibility=False,review_mode='off',relation_weight=.6,qa_executed=False,
        upstream_semantics='exact saved Full fact filtering, QD, PCQD, Plan, bindings, source records; no fresh LLM responses',
        numeric_capture='only Route/dense/embedding numbers recomputed using original frozen semantics and full immutable index; capture cost separately recorded',
        controls='fixed Full rho, features, candidate domain, relevance, and capped package bank; only the explicit scoring term or package-action switch changes',
        full_parity='required exact Top200, incumbent, domain; objective tolerance1e-12 before accepting any variant',
        prior_queue=str(PRIOR),new_llm_calls=0,index_rebuilt=False,scope='component diagnostic, not new independent end-to-end main experiments'))
    write(root/'status.json',dict(status='prepared',phase='waiting_for_prior_queue'))

def init_cpu(root,ds):
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['TOKENIZERS_PARALLELISM']='false'
    sys.path[:0]=[str(CODE/'src'),str(CODE/'scripts')]
    import torch
    torch.set_num_threads(1)
    from run_pack_weight_sweep import ReadOnlyPassages
    from transformers import AutoTokenizer
    from bindpack.prompts.prompt_template_manager import PromptTemplateManager
    from types import SimpleNamespace
    root=Path(root)
    ENV.update(root=root,ds=ds,store=ReadOnlyPassages(next((ARCHIVE/'shared_indexes'/ds).rglob('vdb_chunk.parquet'))),
        tokenizer=AutoTokenizer.from_pretrained('/root/models/Qwen3-8B',local_files_only=True),
        reader=SimpleNamespace(prompt_template_manager=PromptTemplateManager(role_mapping={'system':'system','user':'user','assistant':'assistant'})),
        config=read(root/'capture'/ds/'configuration.json'),refs=read(root/'metadata'/ds/'references.json'),
        source_sha=read(root/'protocol.json')['references'][ds]['progress_sha256'])

def cpu_query(job):
    sid,phase=job;root=ENV['root'];ds=ENV['ds'];reference=ENV['refs'][sid]
    from bindpack import BindPackConfig
    from bindpack.stages.pack.stage import PackStage
    from bindpack.evaluation.measured_reader import prepare as reader_prepare
    from bindpack.utils.misc_utils import QuerySolution
    from run_pack_weight_diagnostic import restore_pack_input
    from types import SimpleNamespace
    import numpy as np
    with gzip.open(root/'inputs'/ds/token(sid),'rt') as stream:event=json.load(stream)
    raw=event['input'];assert digest(raw)==event['input_sha256']
    assert event['sample_id']==sid and event['query_index']==reference['query_index']
    assert event['source_progress_sha256']==ENV['source_sha'] and event['recovery_version']==4
    visible_cache={};bank=None;rho=None;records=[]
    arms=('full',) if phase=='preflight' else ARMS
    for arm in arms:
        values=copy.deepcopy(ENV['config']);values.update(source_bound_eligibility=False,
            mechanism_ablation='wo_joint_selection' if arm=='wo_joint_selection' else 'full',
            pack_mechanism_ablation=arm if arm not in ('full','wo_joint_selection') else 'none')
        cfg=BindPackConfig.from_file(**values)
        def visible(question,ordered):
            key=(question,tuple(ordered))
            if key not in visible_cache:
                sol=QuerySolution(question=question,docs=[ENV['store'].texts[int(d)] for d in ordered],doc_scores=np.ones(len(ordered)))
                reading,budget=reader_prepare(ENV['reader'],sol,ENV['tokenizer'],ds,k=5,context_window=8192,output_tokens=2048)
                visible_cache[key]=(dict(zip(ordered,reading.docs)),budget)
            return visible_cache[key]
        runtime=SimpleNamespace(cfg=cfg,_document=lambda d:ENV['store'].texts[int(d)],_trace=lambda:{},_joint_visible=visible)
        begin=time.perf_counter()
        ids,scores,trace=PackStage(runtime).finalize(raw['query'],np.asarray(raw['ids'],int),np.asarray(raw['scores'],float),{},restore_pack_input(raw))
        seconds=time.perf_counter()-begin;ids=ids[:200].tolist();assert set(ids)==set(reference['full_ids'])
        diag=trace.get('budgeted_selection',{})
        if arm=='full':
            assert ids==reference['full_ids'],f'Full ranking parity failed: {ds}/{sid}'
            assert diag['reference_doc_ids']==reference['s0'] and diag['candidate_domain']==reference['domain']
            assert abs(diag['objective']-reference['objective'])<1e-12 and not diag['search_deadline_reached']
            bank=diag['full_package_bank'];rho=diag['soft_weight_effective']
        elif arm=='wo_joint_selection':assert ids[:5]==reference['s0']
        else:
            assert diag['full_package_bank']==bank and diag['soft_weight_effective']==rho
            assert diag['candidate_domain']==reference['domain'] and diag['reference_doc_ids']==reference['s0']
            assert diag['pack_mechanism_ablation']==arm
            if arm=='wo_support_retention':assert diag['reward_loss_weight']==0.
            if arm=='wo_residual_coverage':assert diag['residual_reward_enabled'] is False
            if arm=='singleton_search':assert diag['package_action_count']==0
            if arm=='flat_dependency_support':assert diag['closure_reward_mode']=='flat'
        gold=set(reference['gold_hashes']);selected={reference['content_by_id'][str(d)] for d in ids[:5]}
        recall=len(gold&selected)/len(gold)
        records.append(dict(sample_id=sid,query_index=reference['query_index'],dataset=ds,phase=phase,arm=arm,
            **{'Recall@5':recall,'FCR@5':int(recall==1)},selected_doc_ids=ids[:5],cpu_wall_seconds=seconds,
            objective=diag.get('objective'),package_bank_sha256=digest(diag.get('full_package_bank')),
            soft_weight_effective=diag.get('soft_weight_effective'),predicted_closure=diag.get('predicted_closure'),
            soft_coverage=diag.get('soft_coverage'),lost_support=diag.get('lost_support'),
            qualified_node_count=len(diag.get('eligible_proofs',{})),
            source_witness_count=sum(len(v) for v in diag.get('eligible_proofs',{}).values()),
            retention_node_count=len(diag.get('retention_sources',{})),
            package_action_count=diag.get('package_action_count'),expanded_states=diag.get('expanded_states'),
            state_cap_reached=diag.get('state_cap_reached'),
            source_bound_eligibility=False,qa_executed=False,new_llm_calls=0,new_embedding_calls=0,
            full_parity_passed=True,search_deadline_reached=diag.get('search_deadline_reached',False)))
    return records

def summarize(root):
    result=[];lines=['# 联合选择内部机制消融','',
        '固定同一次上游、来源资格关闭、末端核验关闭。所有对照均跑两批不重叠200题，不按效果剔除负向结果。',
        'Full必须先通过与保存排名、S0、候选域、目标值的精确核对；所有变体固定Full有效rho和capped package bank。QA=N/A。','',
        '|数据集|阶段|方案|题数|Recall@5|FCR@5|','|---|---|---|---:|---:|---:|']
    for ds in DATASETS:
        for phase in ('pilot','validation'):
            path=root/'cpu'/ds/(phase+'.jsonl')
            if not path.exists():continue
            by_arm={a:[] for a in ARMS}
            for line in path.open():
                event=json.loads(line);assert digest(event['records'])==event['sha256']
                for r in event['records']:by_arm[r['arm']].append(r)
            for arm,rows in by_arm.items():
                if not rows:continue
                assert len(rows)==len({r['sample_id'] for r in rows})
                m={k:sum(r[k] for r in rows)/len(rows) for k in ('Recall@5','FCR@5')}
                result.append(dict(dataset=ds,phase=phase,arm=arm,n=len(rows),**m))
                lines.append(f'|{ds}|{phase}|{arm}|{len(rows)}|{m["Recall@5"]:.6f}|{m["FCR@5"]:.6f}|')
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n');write(root/'summary.json',dict(results=result,qa_executed=False))

def cpu_phase(root,ds,phase):
    if phase=='preflight':
        references=read(root/'metadata'/ds/'references.json')
        selected=sorted(references,key=lambda sid:references[sid]['query_index'])[:8]
    else:selected=read(root/'selection_locked.json')['datasets'][ds][phase]
    file=root/'cpu'/ds/(phase+'.jsonl');file.parent.mkdir(parents=True,exist_ok=True)
    existing=set()
    if file.exists():
        for line in file.open():
            event=json.loads(line);assert digest(event['records'])==event['sha256']
            sid=event['records'][0]['sample_id'];assert sid not in existing
            existing.add(sid)
    jobs=[(sid,phase) for sid in selected if sid not in existing]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=init_cpu,initargs=(str(root),ds)) as pool:
        with file.open('a') as stream:
            for records in pool.map(cpu_query,jobs,chunksize=1):
                stream.write(json.dumps(dict(sha256=digest(records),records=records),ensure_ascii=False)+'\n');stream.flush();os.fsync(stream.fileno())
                existing.add(records[0]['sample_id'])
                if len(existing)%10==0:print('CPU',ds,phase,len(existing),'/',len(selected),flush=True)
    assert existing==set(selected)
    if phase=='preflight':write(root/'preflight'/ds/'passed.json',dict(status='passed',queries=8,exact_full_parity=True,new_llm_calls=0))
    else:summarize(root)

def ENV_IDS(root,ds):return list(read(root/'metadata'/ds/'references.json'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ARCHIVE/'joint_mechanisms_20261028')
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--frozen-queue',action='store_true')
    p.add_argument('--cpu-phase',choices=['preflight','pilot','validation']);p.add_argument('--dataset',choices=DATASETS)
    args=p.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    if args.cpu_phase:
        assert args.dataset;cpu_phase(root,args.dataset,args.cpu_phase);return
    prepare(root)
    if not args.frozen_queue:
        os.execv(sys.executable,[sys.executable,'-B','-u',str(root/'source_snapshot/scripts/run_joint_component_study.py'),*sys.argv[1:],'--frozen-queue'])
    if args.prepare_only:return
    lock=(root/'.queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (root/'run.pid').write_text(str(os.getpid())+'\n');state={'status':'waiting','pid':os.getpid(),'phase':'waiting_for_prior_queue'}
    current=[None];stopping=threading.Event()
    def stop(*unused):
        stopping.set()
        if current[0] is not None and current[0].poll() is None:os.killpg(current[0].pid,signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def check_stop():
        if stopping.is_set():raise KeyboardInterrupt('Stopped; all checkpoints retained')
    def execute(command,cpu=False):
        env=dict(os.environ)
        if cpu:env['CUDA_VISIBLE_DEVICES']=''
        process=subprocess.Popen(command,cwd=PROJECT,env=env,stdin=subprocess.DEVNULL,start_new_session=True);current[0]=process
        while process.poll() is None:check_stop();stopping.wait(5)
        current[0]=None;assert process.returncode==0,'Study worker failed; queue stopped'
    try:
        while True:
            check_stop();prior=read(PRIOR/'status.json');pid=prior['pid'];cmd=Path('/proc')/str(pid)/'cmdline'
            try:live=b'run_new_full_pilot.py' in cmd.read_bytes()
            except FileNotFoundError:live=False
            if prior['status']=='complete' and not live:break
            if prior['status'] in ('failed','stopped') and not live:raise RuntimeError('Prior queue did not complete; leave GPU untouched')
            state['prior_phase']=prior.get('phase');state['prior_dataset']=prior.get('dataset');write(root/'status.json',state)
            print('WAIT prior measured queue:',prior.get('phase'),prior.get('dataset'),flush=True);stopping.wait(30)
        for ds in DATASETS:
            check_stop();state.update(status='running',phase='numeric_capture',dataset=ds);write(root/'status.json',state)
            if not (root/'capture'/ds/'completed.ok').exists():
                command=[sys.executable,'-B','-u',str(CODE/'scripts/recover_adopted_pack_inputs.py'),'--output',str(root),'--dataset',ds,'--wait-parity']
                process=subprocess.Popen(command,cwd=PROJECT,stdin=subprocess.DEVNULL,start_new_session=True);current[0]=process
                references=read(root/'metadata'/ds/'references.json')
                first=sorted(references,key=lambda sid:references[sid]['query_index'])[:8]
                while process.poll() is None:
                    check_stop()
                    if not (root/'preflight'/ds/'passed.json').exists() and all((root/'inputs'/ds/token(sid)).exists() for sid in first):
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
                        subprocess.run([sys.executable,'-B','-u',str(CODE/'scripts/run_joint_component_study.py'),'--output',str(root),'--dataset',ds,'--cpu-phase','preflight'],cwd=PROJECT,env=env,check=True)
                    stopping.wait(5)
                current[0]=None;assert process.returncode==0,'Numeric recovery failed; no controls accepted'
            for phase in ('pilot','validation'):
                state.update(phase='cpu_'+phase,dataset=ds);write(root/'status.json',state)
                execute([sys.executable,'-B','-u',str(CODE/'scripts/run_joint_component_study.py'),'--output',str(root),'--dataset',ds,'--cpu-phase',phase],cpu=True)
            write(root/'completed'/ds,dict(status='complete',queries=400,arms=list(ARMS)))
        state.update(status='complete',phase='complete');write(root/'completed.ok',dict(datasets=4,distinct_queries=1600,new_llm_calls=0,qa_executed=False))
    except BaseException as error:
        state.update(status='stopped' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error))
        if current[0] is not None and current[0].poll() is None:
            os.killpg(current[0].pid,signal.SIGTERM);current[0].wait()
        raise
    finally:write(root/'status.json',state);summarize(root)

if __name__=='__main__':main()

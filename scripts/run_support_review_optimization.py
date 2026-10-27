#!/usr/bin/env python3
"""Checkpointed fixed-parent review experiment; never runs Route/Plan/embedding."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from copy import copy,deepcopy
from datetime import datetime
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import threading
import time
from types import SimpleNamespace

PROJECT=Path('/root/BindPack')
BASE=PROJECT/'outputs/4_datasets_result/latest_r060_20261026'
DEFAULT=PROJECT/'outputs/support_review_optimization_20261026'
ARMS=('off','current','compact_two','compact_single','tight_two','prefix_two','adaptive_two','suffix_only')


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def digest(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def checkpoint(case,qid):return case/'checkpoints'/(hashlib.sha256(str(qid).encode()).hexdigest()+'.json')


def prepare(root):
    if (root/'experiment.json').exists():return
    root.mkdir(parents=True,exist_ok=True)
    source=BASE/'source_snapshot'
    shutil.copytree(source/'src',root/'source_snapshot/src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    target=root/'source_snapshot/scripts';target.mkdir(parents=True,exist_ok=True)
    for name in ('compact_support_review.py','run_support_review_optimization.py'):
        shutil.copy2(PROJECT/'scripts'/name,target/name)
    raw=read(BASE/'cases/hotpotqa/full/result.json')
    ids=[r['sample_id'] for r in raw['results']]
    shuffled=sorted(ids,key=lambda s:hashlib.sha256(('support-review-pilot-v1:'+s).encode()).hexdigest())
    write(root/'selection.json',dict(pilot=shuffled[:100],validation=shuffled[100:],
        rule='100 hash-selected pilot queries; remaining 900 are one-time confirmation; no label-based selection'))
    configuration=raw['runtime_config'];method={k:configuration[k] for k in ('candidates','route','plan','pack')}
    method['runtime']={k:v for k,v in configuration.items() if k not in (*method,'stage_ablation')}
    write(root/'method.json',method)
    write(root/'experiment.json',dict(base=str(BASE),source_result_sha256=sha(BASE/'cases/hotpotqa/full/result.json'),
        original_hotpotqa_queries=1000,models=dict(llm='/root/models/Qwen3-8B',embedding='/root/models/Qwen3-Embedding-8B'),
        relation_weight=.6,reader_top_k=5,arms=ARMS,index_rebuild=False,embedding_calls=0,route_plan_rerun=False,
        fixed_parent='exact original_parent_doc_ids from completed measured Full',
        diagnostic_only=True,production_promoted=False,
        selection='pilot100 for candidate selection; remaining900 for confirmation; dataset already inspected historically',
        sources={str(p.relative_to(root/'source_snapshot')):sha(p) for p in (root/'source_snapshot').rglob('*.py')},
        time_scope='Review-stage wall intervals measured. QA reused only for exact identical ordered reader contexts. Incremental wall excludes frozen upstream and is not native end-to-end latency.'))
    write(root/'stopped_previous_queue.json',dict(status=read(BASE/'queue_status.json'),
        preserved_hotpotqa=1000,preserved_2wiki=324,vllm_preserved=True,
        note='Original sessions record KeyboardInterrupt as failed; coordinator records requested stopped. Original ledgers are not rewritten.'))


def source(root,dataset):
    overrides=root/'source_overrides.json'
    if overrides.exists() and dataset in read(overrides):
        entry=read(overrides)[dataset];path=Path(entry['path'])
        assert sha(path)==entry['sha256'],'Frozen extracted source changed'
        return path
    if dataset=='hotpotqa':return BASE/'cases/hotpotqa/full/result.json'
    return PROJECT/f'outputs/pack_150_before_after_20261026/cases/{dataset}/after/result.json'


def run(root,dataset,arm,phase,workers,limit):
    os.environ.setdefault('OPENAI_API_KEY','EMPTY')
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    sys.path.insert(0,str(root/'source_snapshot/src'))
    sys.path.insert(0,str(root/'source_snapshot/scripts'))
    from transformers import AutoTokenizer
    from bindpack import BindPackConfig
    from bindpack.core.base import BaseRAG
    from bindpack.prompts.prompt_template_manager import PromptTemplateManager
    from bindpack.llm.openai_gpt import CacheOpenAI
    from bindpack.evaluation.measurement import Recorder,ResourceMonitor,jsonl,distribution
    from bindpack.evaluation.measured_reader import prepare as reader_prepare
    from bindpack.evaluation.qa_eval import QAExactMatch,QAF1Score
    from bindpack.utils.misc_utils import QuerySolution
    from bindpack.stages.pack.verification.support_semantic_veto import postprocess_support_semantic_veto
    import compact_support_review as compact
    import numpy as np

    class ThreadRecorder(Recorder):
        def tags(self):
            return getattr(self.local,'context',super().tags())
        @contextmanager
        def tagged(self,phase,qid):
            previous=getattr(self.local,'context',None)
            self.local.context=dict(phase=phase,query_id=qid)
            try:yield
            finally:
                if previous is None:del self.local.context
                else:self.local.context=previous

    class Proxy:
        def __init__(self,real,record,qid):self.real,self.record,self.qid,self.reader=real,record,qid,False
        def infer(self,messages,**kw):
            phase='online_reader' if self.reader else 'online_proposer'
            if not self.reader:
                try:
                    if 'PROPOSAL' in json.loads(messages[-1]['content']):phase='online_reviewer'
                except (ValueError,TypeError):pass
            with self.record.tagged(phase,self.qid):return self.real.infer(messages,**kw)

    class Reader:
        qa=BaseRAG.qa
        _extract_llm_text=staticmethod(lambda x:x if isinstance(x,str) else str(x))

    raw=read(source(root,dataset));old_rows=raw['results']
    if dataset=='hotpotqa':
        selection=read(root/'selection.json')
        wanted=selection[phase] if phase in ('pilot','validation') else selection['pilot']+selection['validation']
    else:
        wanted=sorted([r['sample_id'] for r in old_rows],key=lambda s:hashlib.sha256(('review-generalization-v1:'+s).encode()).hexdigest())[:100]
    by_id={r['sample_id']:r for r in old_rows};wanted=wanted[:limit] if limit else wanted
    case=root/'cases'/dataset/(phase+'_'+arm)
    case.mkdir(parents=True,exist_ok=True)
    with (case/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config=BindPackConfig.from_file(root/'method.json',dataset=dataset,save_dir=str(case/'cache'),stage_ablation='full')
        tokenizer=AutoTokenizer.from_pretrained('/root/models/Qwen3-8B',local_files_only=True)
        real=CacheOpenAI.from_experiment_config(config)
        recorder=ThreadRecorder(case/'accounting');recorder.install_llm(real,tokenizer)
        recorder.phase='online_review_experiment'
        prompt_manager=PromptTemplateManager(role_mapping={'system':'system','user':'user','assistant':'assistant'})
        baseline_report=read(source(root,dataset).parent/'metrics_details.json')
        baseline_qa={r['sample_id']:r['qa'] for r in baseline_report['per_query']}
        done=[]

        def process(qid):
            saved=checkpoint(case,qid)
            if saved.exists():return read(saved)
            old=by_id[qid];sem=old['retrieval_trace']['evidence']['improvement_support_semantic_veto']
            parent=list(sem['original_parent_doc_ids']);docs=dict(zip(sem['final_doc_ids'],old['candidate_docs']))
            assert set(parent)==set(docs) and len(parent)==200
            trace=deepcopy(old['retrieval_trace']['evidence']);trace.pop('improvement_support_semantic_veto',None)
            trace['selected_prefix']=[dict(doc_id=d,selection_source='frozen_pre_review') for d in parent[:5]]
            solution=QuerySolution(question=old['question'],docs=[docs[d] for d in parent],
                doc_scores=np.linspace(1.,0.,len(parent)),retrieval_trace={'stage_ablation':'full','evidence':trace})
            rag=Reader();rag.config=rag.global_config=config;rag.prompt_template_manager=prompt_manager
            proxy=Proxy(real,recorder,qid);rag.llm_model=proxy
            runtime=SimpleNamespace(cfg=config,rag=rag,_document=docs.__getitem__,_support_semantic_tokenizer=tokenizer)
            began=time.perf_counter()
            if arm=='current':postprocess_support_semantic_veto(runtime,[(solution,parent)])
            elif arm=='tight_two':
                import tight_support_review
                tight_support_review.run(runtime,solution,parent)
            elif arm=='prefix_two':
                import prefix_support_review
                prefix_support_review.run(runtime,solution,parent)
            elif arm=='adaptive_two':
                import adaptive_support_review
                adaptive_support_review.run(runtime,solution,parent)
            elif arm=='suffix_only':
                import prefix_support_review
                prefix_support_review.run(runtime,solution,parent,mask=False)
            elif arm!='off':compact.run(runtime,solution,parent,reviewer=arm=='compact_two')
            review_seconds=time.perf_counter()-began
            same=solution.docs[:5]==old['candidate_docs'][:5]
            before_reader=time.perf_counter();proxy.reader=True
            reading,budget=reader_prepare(rag,solution,tokenizer,dataset)
            same=same and budget['prompt_sha256']==old['reader']['budget']['prompt_sha256']
            if same:
                solution.answer=old['answer'];reader=deepcopy(old['reader']);reader['reused_identical_ordered_context']=True
            else:
                answered,response,meta=rag.qa([reading]);solution.answer=answered[0].answer
                reader=dict(raw_response=response[0],metadata=meta[0],budget=budget,reused_identical_ordered_context=False)
            reader_seconds=time.perf_counter()-before_reader
            gold=set(old['gold_docs']);final=solution.docs
            qa={**QAExactMatch(config).calculate_metric_scores([old['gold_answers']],[solution.answer])[1][0],
                **QAF1Score(config).calculate_metric_scores([old['gold_answers']],[solution.answer])[1][0]}
            if dataset=='hover':
                from bindpack.evaluation.metric_report import normalize_hover_prediction
                qa['ClassificationAccuracy']=float(normalize_hover_prediction(solution.answer)==normalize_hover_prediction(old['gold_answers'][0]))
            final_ids=[next(d for d in parent if docs[d]==text) for text in final]
            metrics={f'{name}@{k}':v for k in (2,3,5,8,10,20,200)
                for name,v in [('Recall',len(gold&set(final[:k]))/len(gold)),('FCR',float(gold<=set(final[:k]))) ]}
            base={f'{name}@{k}':v for k in (2,3,5,8,10,20,200)
                for name,v in [('Recall',len(gold&set(old['candidate_docs'][:k]))/len(gold)),('FCR',float(gold<=set(old['candidate_docs'][:k]))) ]}
            value=dict(sample_id=qid,query_index=old['query_index'],question=old['question'],answer=solution.answer,
                gold_answers=old['gold_answers'],gold_docs=old['gold_docs'],final_doc_ids=final_ids,
                metrics=metrics,qa=qa,baseline_metrics=base,baseline_qa=baseline_qa[qid],
                reader=reader,review_trace=solution.retrieval_trace['evidence'].get('improvement_support_semantic_veto'),
                timing=dict(review_seconds=review_seconds,reader_seconds=reader_seconds,
                    reader_reused=same,incremental_seconds=time.perf_counter()-began))
            write(saved,value)
            return value

        started=time.perf_counter()
        with ResourceMonitor(recorder) as monitor,ThreadPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(process,qid):qid for qid in wanted}
            for future in as_completed(futures):
                row=future.result();done.append(row)
                status=dict(dataset=dataset,arm=arm,phase=phase,completed=len(done),expected=len(wanted),workers=workers,
                    elapsed_seconds=time.perf_counter()-started,status='running')
                write(root/'status.json',status)
                print(f"{datetime.now():%Y-%m-%d %H:%M:%S} {dataset}/{phase}/{arm} {len(done)}/{len(wanted)}; review={row['timing']['review_seconds']:.2f}s; FCR5={row['metrics']['FCR@5']}; EM={row['qa']['ExactMatch']}",flush=True)
            resources=monitor.summary()
        physical=jsonl(recorder.directory/'llm_calls.jsonl');logical=jsonl(recorder.directory/'llm_logical_calls.jsonl')
        measurements={k:sum(r['metrics'][k] for r in done)/len(done) for k in done[0]['metrics']}
        qa={k:sum(r['qa'][k] for r in done)/len(done) for k in done[0]['qa']}
        base={k:sum(r['baseline_metrics'][k] for r in done)/len(done) for k in done[0]['baseline_metrics']}
        bqa={k:sum(r['baseline_qa'][k] for r in done)/len(done) for k in ('ExactMatch','F1')}
        calls={}
        for phase_name in ('online_proposer','online_reviewer','online_reader'):
            events=[e for e in physical if e['phase']==phase_name]
            calls[phase_name]=dict(http_attempts=len(events),input_tokens=sum(e['prompt_tokens'] or 0 for e in events),
                output_tokens=sum(e['completion_tokens'] or 0 for e in events),seconds=sum(e['seconds'] for e in events),
                failed_attempts=sum(e['status']!='success' for e in events),finish_length=sum(e.get('finish_reason')=='length' for e in events))
        report=dict(dataset=dataset,arm=arm,phase=phase,samples=len(done),metrics={**measurements,**qa},baseline_metrics={**base,**bqa},
            deltas={**{k:measurements[k]-base[k] for k in base},**{k:qa[k]-bqa[k] for k in bqa}},
            review_latency=distribution(r['timing']['review_seconds'] for r in done),
            case_wall_seconds=time.perf_counter()-started,query_concurrency=workers,
            qa_reused_identical_ordered_context=sum(r['timing']['reader_reused'] for r in done),
            cost=calls,logical_calls=len(logical),cache_hits=sum(bool(e['cache_hit']) for e in logical),resources=resources,
            index_rebuilt=False,embedding_calls=0,route_plan_rerun=False,reader_top_k=5,relation_weight=.6,
            evidence_budget=dict(passages=5,actual_prompt_tokens=distribution(r['reader']['budget']['prompt_tokens_local'] for r in done)),
            timing_scope='Review-stage latency and incremental work only. Concurrent throughput is separate from single-query latency; reused reader time not claimed as optimization.',
            input_sha256=sha(source(root,dataset)))
        report['review_outcomes']=dict(Counter((r['review_trace']['base_proposal'].get('abstain_reason')
            or ('accepted' if r['review_trace']['accepted'] else 'local_or_semantic_veto'))
            if r['review_trace'] else 'review_disabled' for r in done))
        report['source_pack_parameters']=raw['runtime_config'].get('pack')
        for k in ('NodeCov@5','DCC@5','ClosureGap@5'):report['metrics'][k]=None
        if dataset=='musique':
            from bindpack.evaluation.structural_metrics import evaluate_structural_metrics
            from bindpack.evaluation.dataset_adapter import get_sample_id
            data=read(PROJECT/'datasets/musique.json');sample_map={get_sample_id(s,dataset):s for s in data};samples=[];struct_rows=[]
            for row in done:
                original=by_id[row['sample_id']];sem=original['retrieval_trace']['evidence']['improvement_support_semantic_veto']
                mapping=dict(zip(sem['final_doc_ids'],original['candidate_docs']))
                ranked=[mapping[d] for d in row['final_doc_ids']]
                item=dict(original,candidate_docs=ranked)
                item.pop('gold_document_ranks',None)
                struct_rows.append(item)
                samples.append(sample_map[row['sample_id']])
            report['structural_metrics']=evaluate_structural_metrics(struct_rows,samples,dataset,ks=(5,))['metrics']
            report['metrics'].update({k:report['structural_metrics'][k] for k in ('NodeCov@5','DCC@5','ClosureGap@5')})
            base_structure=evaluate_structural_metrics([by_id[r['sample_id']] for r in done],samples,dataset,ks=(5,))['metrics']
            for k in ('NodeCov@5','DCC@5','ClosureGap@5'):
                report['baseline_metrics'][k]=base_structure[k]
                report['deltas'][k]=report['metrics'][k]-base_structure[k]
        if dataset=='hover':
            from bindpack.evaluation.metric_report import score_hover_classification
            report['metrics'].update(score_hover_classification([r['answer'] for r in done],
                [by_id[r['sample_id']]['gold_answers'][0] for r in done])[0])
            base_classification=score_hover_classification([by_id[r['sample_id']]['answer'] for r in done],
                [by_id[r['sample_id']]['gold_answers'][0] for r in done])[0]
            report['baseline_metrics'].update(base_classification)
            report['deltas'].update({k:report['metrics'][k]-v for k,v in base_classification.items()})
        write(case/'report.json',report);write(case/'completed.ok',dict(samples=len(done),report_sha256=sha(case/'report.json')))
        write(root/'status.json',dict(status='case_complete',dataset=dataset,arm=arm,phase=phase,completed=len(done)))
        print('REPORT',json.dumps(report,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--dataset',choices=['hotpotqa','2wikimultihopqa','musique','hover'],default='hotpotqa')
    p.add_argument('--arm',choices=ARMS,default='compact_two');p.add_argument('--phase',choices=['pilot','validation','full','generalization'],default='pilot')
    p.add_argument('--workers',type=int,default=1);p.add_argument('--limit',type=int);p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args();prepare(a.output.resolve())
    if a.prepare_only:return
    assert 1<=a.workers<=8
    run(a.output.resolve(),a.dataset,a.arm,a.phase,a.workers,a.limit)


if __name__=='__main__':
    logging.basicConfig(level=logging.WARNING)
    main()

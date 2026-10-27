#!/usr/bin/env python3
"""Serial, checkpointed Pack candidates on the same locked old50 per dataset."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PROJECT = Path('/root/BindPack')
PREVIOUS = PROJECT / 'outputs/component_diagnostic_r060_20261026'
WEIGHTS = PROJECT / 'outputs/pack_weight_diagnostic_20261025'
DEFAULT = PROJECT / 'outputs/pack_candidates_r060_20261026'
DATASETS = ('hotpotqa', '2wikimultihopqa', 'musique', 'hover')
SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parent))
from build_pack_candidates import FEATURES, sources


def read(p): return json.loads(Path(p).read_text())
def digest(v): return hashlib.sha256(json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, v):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + '.tmp'); t.write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False) + '\n'); t.replace(p)


def build(root, arm, features):
    target = root / 'variants' / arm
    changed = sources(root / 'source_snapshot', features)
    if (target / 'source_hashes.json').exists():
        assert read(target / 'features.json') == sorted(features), 'Frozen variant feature set changed'
        expected = read(target / 'source_hashes.json')
        assert all(sha(target / rel) == value for rel, value in expected.items())
        return
    shutil.copytree(root / 'source_snapshot/src', target / 'src', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for rel, text in changed.items(): (target / rel).write_text(text)
    write(target / 'source_hashes.json', {str(p.relative_to(target)): sha(p) for p in (target/'src').rglob('*.py')})
    write(target / 'features.json', sorted(features))


def prepare(root):
    lock = read(PREVIOUS/'selection_locked.ok'); selected = read(PREVIOUS/'selection_manifest.json')
    assert digest(selected) == lock['sha256'] == '719599596273100e9830e7482ef00be1afd75bbb862dc1b266cd1e04b7ea038e'
    if (root/'prepared.ok').exists():
        assert read(root/'selection_locked.ok') == lock
        return
    root.mkdir(parents=True, exist_ok=True)
    write(root/'selection_manifest.json', selected); write(root/'selection_locked.ok', lock)
    shutil.copytree(PROJECT/'src', root/'source_snapshot/src', dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    write(root/'production_base_hashes.json', {str(p.relative_to(PROJECT)): sha(p) for p in (PROJECT/'src').rglob('*') if p.is_file() and p.suffix in {'.py','.json'}})
    for name in (SCRIPT.name, 'build_pack_candidates.py', 'run_pack_weight_sweep.py', 'eval_utils.py'):
        (root/'source_snapshot/scripts').mkdir(exist_ok=True)
        shutil.copy2(PROJECT/'scripts'/name, root/'source_snapshot/scripts'/name)
    write(root/'experiment.json', dict(relation_weight=.6, diagnostic_only=True, questions_per_dataset=50,
        unique_questions=200, selection_sha256=lock['sha256'], arms=FEATURES,
        frozen_inputs=str(PREVIOUS/'bundles'), baseline='Current Full R=.60 including the already adopted HoVer reader.',
        index_reused=True, index_rebuild=False, embedding_inference=False, route_plan_execution=False, serial=True,
        reader_top_k=5, models=dict(llm='/root/models/Qwen3-8B',embedding='/root/models/Qwen3-Embedding-8B'),
        selection_rule='Retain only if every dataset has no Recall@5/FCR@5/QA-primary/correctness/DCC@5 regression and at least one quality metric improves, OR all quality metrics are unchanged and committed logical LLM calls decrease. Neutral engineering changes remain experimental.',
        combination_rule='Test retained features jointly; promote the best non-regressing tested combination or single, prioritizing mean QA, FCR, Recall, correctness, DCC, then fewer logical calls.',
        timing_scope='Warm cache Pack+reader incremental execution only. Not cold end-to-end or offline indexing cost.',
        hop_policy='Same historical old50 protocol. HotpotQA/2Wiki prior=2; existing MuSiQue/HoVer oracle hints unchanged.',
        input_result_hashes={ds:sha(PREVIOUS/'cases'/ds/'combined/result.json') for ds in DATASETS}))
    for ds in DATASETS:
        meta=root/'metadata'/ds;meta.mkdir(parents=True,exist_ok=True)
        for name in ('benchmark.json','corpus.json'):
            if not (meta/name).exists(): (meta/name).symlink_to(PREVIOUS/'metadata'/ds/name)
        manifest=read(PREVIOUS/'metadata'/ds/'sample_manifest.json');manifest['data_path']=str(meta/'benchmark.json')
        write(meta/'sample_manifest.json',manifest)
        method=read(PREVIOUS/'metadata'/ds/'method.json');method['runtime']['save_dir']=str(root/'runtime'/ds/'index')
        assert method['pack']['relation_weight']==.6
        write(meta/'method.json',method)
        cache=root/'runtime'/ds/'index/llm_cache';cache.mkdir(parents=True,exist_ok=True)
        for p in (PREVIOUS/'runtime'/ds/'index/llm_cache').glob('*.sqlite'): shutil.copy2(p,cache/p.name)
    for arm in ('baseline',)+tuple(FEATURES): build(root,arm,[] if arm=='baseline' else [arm])
    write(root/'prepared.ok',dict(selection_sha256=lock['sha256'],model_requests=0))


def token(qid): return hashlib.sha256(str(qid).encode()).hexdigest()+'.json'
def rows(root,ds,arm):
    out=[]
    for q in read(root/'selection_manifest.json')['datasets'][ds]['rows']:
        p=root/'cases'/ds/arm/'checkpoints'/token(q['sample_id'])
        if not p.exists(): break
        saved=read(p);assert digest(saved['row'])==saved['sha256'];out.append(saved['row'])
    return out


def load_sweep(root):
    p=root/'source_snapshot/scripts/run_pack_weight_sweep.py'
    s=importlib.util.spec_from_file_location('candidate_sweep_helpers',p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def finish(root,ds,arm):
    from bindpack.evaluation.metric_report import write_metric_report
    values=rows(root,ds,arm);assert len(values)==50
    meta=root/'metadata'/ds;method=read(meta/'method.json');identity=read(meta/'sample_manifest.json')
    case=root/'cases'/ds/arm
    protocol=dict(variant=arm,features=read(root/'variants'/arm/'features.json'),relation_weight=.6,
        reader_top_k=5,index_rebuilt=False,route_plan_rerun=False,diagnostic_only=True,
        selection_sha256=read(root/'selection_locked.ok')['sha256'])
    write(case/'protocol.json',protocol)
    write(case/'result.json',dict(dataset=ds,method='bindpack',stage_ablation='full',eval_mode='rag_qa',
        data_path=str(meta/'benchmark.json'),corpus_path=str(meta/'corpus.json'),selected_indices=identity['selected_indices'],
        sample_size_effective=50,pack_relation_weight=.6,experimental_variant=arm,diagnostic_selection=True,
        runtime_config=dict(method['runtime'],**{k:method[k] for k in ('candidates','route','plan','pack')}),
        hop_provenance=identity.get('hop_provenance'),results=values))
    write(case/'validated.ok',dict(result_sha256=sha(case/'result.json'),protocol_sha256=digest(protocol),sample_count=50))
    report=write_metric_report(case/'result.json',data_path=meta/'benchmark.json',manifest_path=meta/'sample_manifest.json',ks=(3,5,8,10,20,200))
    c=Counter()
    for row in values:c.update(row['diagnostic_cost'])
    write(case/'cost.json',dict(committed_rows=dict(c),scope='Warm-cache Pack+reader. Logical usage includes real cache metadata; no offline or end-to-end cost claim.'))
    print('REPORT',ds,arm,json.dumps({k:report['metrics'].get(k) for k in ('Recall@5','FCR@5','ExactMatch','F1','ClassificationMacroF1','DCC@5')},ensure_ascii=False),flush=True)


def worker(root,ds,arm,production=False):
    sys.path.insert(0,str(PROJECT/'src' if production else root/'variants'/arm/'src'))
    sys.path.insert(0,str(root/'source_snapshot/scripts'))
    os.environ.setdefault('OPENAI_API_KEY','EMPTY')
    from transformers import AutoTokenizer
    from bindpack import BindPackConfig
    from bindpack.evaluation.measurement import Recorder,ResourceMonitor,jsonl
    from bindpack.evaluation.result_export import detailed_result
    from eval_utils import get_gold_answers,get_gold_docs
    sweep=load_sweep(root);helper=sweep.helpers(WEIGHTS)
    if not production and len(rows(root,ds,arm))==50:finish(root,ds,arm);return
    cfg=BindPackConfig.from_file(root/'metadata'/ds/'method.json',stage_ablation='full')
    assert cfg.pack.relation_weight==.6 and cfg.qa_top_k==5
    tokenizer=AutoTokenizer.from_pretrained('/root/models/Qwen3-8B',local_files_only=True)
    recorder=Recorder(root/('production_parity' if production else 'accounting')/ds/arm)
    store=sweep.ReadOnlyPassages(next((PROJECT/'outputs/4_datasets_result/shared_indexes'/ds).rglob('vdb_chunk.parquet')))
    rag=sweep.make_runtime(cfg,store,recorder,tokenizer)
    reference=read(PREVIOUS/'cases'/ds/'combined/result.json')['results']
    expected={r['sample_id']:r for r in (rows(root,ds,arm) if production else reference)}
    transport=rag.llm_model.openai_client.chat.completions;real=transport.create
    if arm=='baseline' or production:
        def deny(*args,**kwargs):raise AssertionError('Current baseline/production parity generated a cache miss')
        transport.create=deny
    selected=read(root/'selection_manifest.json')['datasets'][ds]['rows'];benchmark=read(root/'metadata'/ds/'benchmark.json')
    try:
        with ResourceMonitor(recorder) as monitor:
            for n,chosen in enumerate(selected,1):
                qid=chosen['sample_id'];checkpoint=root/'cases'/ds/arm/'checkpoints'/token(qid)
                if checkpoint.exists() and not production:continue
                saved=read(PREVIOUS/'bundles'/ds/token(qid));bundle=saved['payload'];assert digest(bundle)==saved['sha256']
                payload=bundle['input'];assert payload['query']==chosen['question']
                before_l=len(jsonl(recorder.directory/'llm_logical_calls.jsonl'));before_h=len(jsonl(recorder.directory/'llm_calls.jsonl'))
                start=time.perf_counter()
                solution,reader,timing=sweep.run_arm(rag,tokenizer,recorder,payload,.6,qid,helper)
                if arm=='baseline' or production:
                    old=expected[qid]
                    assert solution.docs==old['candidate_docs'],(ds,qid,'ranking parity')
                    assert solution.answer==old['answer'],(ds,qid,'answer parity')
                    assert reader['budget']['prompt_sha256']==old['reader']['budget']['prompt_sha256'],(ds,qid,'reader prompt parity')
                if production:continue
                sample=benchmark[chosen['query_index']]
                solution.gold_answers=get_gold_answers([sample],ds)[0];solution.gold_docs=sorted(set(get_gold_docs([sample],ds)[0]))
                row=helper.jsonable(detailed_result(solution,sample,chosen['query_index'],solution.gold_docs,chosen['benchmark_hops'],10,200))
                logical=jsonl(recorder.directory/'llm_logical_calls.jsonl')[before_l:];http=jsonl(recorder.directory/'llm_calls.jsonl')[before_h:]
                cost=dict(logical_calls=len(logical),cache_hits=sum(bool(c.get('cache_hit')) for c in logical),http_attempts=len(http),
                    logical_input_tokens=sum(c.get('metadata',{}).get('prompt_tokens') or 0 for c in logical),
                    logical_output_tokens=sum(c.get('metadata',{}).get('completion_tokens') or 0 for c in logical),
                    physical_input_tokens=sum(c.get('prompt_tokens') or 0 for c in http),physical_output_tokens=sum(c.get('completion_tokens') or 0 for c in http))
                row.update(ranking_complete=True,actual_ranking_length=len(solution.docs),pack_relation_weight=.6,experimental_variant=arm,
                    frozen_pack_input_sha256=digest(payload),reader=reader,timing=dict(timing,incremental_seconds=time.perf_counter()-start),diagnostic_cost=cost)
                write(checkpoint,dict(row=row,sha256=digest(row)))
                write(root/'status.json',dict(dataset=ds,arm=arm,completed=n,total=50,phase='individual' if arm!='combined' else 'combined'))
                print(f'QUERY {ds}/{arm} {n}/50; {time.perf_counter()-start:.2f}s; calls={len(logical)}; HTTP={len(http)}; answer={solution.answer!r}',flush=True)
            write(recorder.directory/'resource_summary.json',monitor.summary())
    finally:
        transport.create=real
        for obj,name,original in reversed(recorder.restorations):setattr(obj,name,original)
    if production:write(root/'production_parity'/ds/'passed.ok',dict(checked=50,new_http_requests=0));return
    finish(root,ds,arm)


def metrics(root,arm):
    values={}
    for ds in DATASETS:
        m=read(root/'cases'/ds/arm/'report.json')['metrics']
        values[ds]={k:m[n] for k,n in dict(recall='Recall@5',fcr='FCR@5',qa='ClassificationMacroF1' if ds=='hover' else 'F1',
            correctness='ClassificationAccuracy' if ds=='hover' else 'ExactMatch',dcc='DCC@5').items()}
    return values


def assess(root,arm):
    base=metrics(root,'baseline');m=metrics(root,arm)
    delta={ds:{k:None if base[ds][k] is None else m[ds][k]-base[ds][k] for k in base[ds]} for ds in DATASETS}
    safe=all(v is None or v>=-1e-10 for d in delta.values() for v in d.values())
    mean={k:sum(d[k] for d in delta.values() if d[k] is not None)/sum(d[k] is not None for d in delta.values()) for k in delta['musique']}
    logical_delta=sum(read(root/'cases'/ds/arm/'cost.json')['committed_rows']['logical_calls']-read(root/'cases'/ds/'baseline/cost.json')['committed_rows']['logical_calls'] for ds in DATASETS)
    gain=any(v>1e-10 for v in mean.values());kept=safe and (gain or logical_delta<0)
    return dict(retain=kept,per_dataset=delta,mean_deltas=mean,logical_call_delta=logical_delta,
        status='retain_quality' if kept and gain else 'retain_efficiency' if kept else 'tradeoff' if gain else 'no_gain_or_regression')


def summarize(root):
    lines=['# Pack 八项独立诊断 · 固定 0.60 / Reader@5','',
        '同一锁定 200 题，每数据集 50 题；复用原索引和相同 Route+Plan 输入，保留已采用的 HoVer reader。此历史选择诊断集不是正式主实验。','',
        '| 数据集 | 版本 | Recall@5 | FCR@5 | EM / Accuracy | F1 / Macro-F1 | DCC@5 | 逻辑调用 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for ds in DATASETS:
        for arm in ('baseline',)+tuple(FEATURES)+('combined',):
            p=root/'cases'/ds/arm/'report.json'
            if not p.exists():continue
            m=read(p)['metrics'];names=('Recall@5','FCR@5','ClassificationAccuracy' if ds=='hover' else 'ExactMatch','ClassificationMacroF1' if ds=='hover' else 'F1','DCC@5')
            calls=read(p.parent/'cost.json')['committed_rows']['logical_calls']
            lines.append(f'| {ds} | {arm} | '+' | '.join('N/A' if m[k] is None else f'{m[k]:.6f}' for k in names)+f' | {calls} |')
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')


def launch(root,arm,production=False):
    for ds in DATASETS:
        print('START',ds,arm,'production-parity' if production else 'diagnostic',flush=True)
        command=[sys.executable,'-B','-u',str(root/'source_snapshot/scripts'/SCRIPT.name),'--output',str(root),'--worker',ds,'--arm',arm]
        if production:command.append('--production')
        subprocess.run(command,check=True)
        if not production:summarize(root)


def promote(root,arm,features):
    receipt=root/'production_promotion.json'
    if receipt.exists():
        previous=read(receipt)
        if previous.get('status')=='applied_and_verified' and previous.get('arm')==arm and previous.get('features')==features:
            assert all(sha(PROJECT/rel)==value for rel,value in previous['changes'].items()), 'Previously promoted source was edited'
            return previous
    changes=sources(root/'source_snapshot',features);hashes=read(root/'production_base_hashes.json');backup={}
    for rel,text in changes.items():
        p=PROJECT/rel
        assert sha(p)==hashes[rel],('Concurrent source change',rel)
        backup[rel]=p.read_text()
    write(root/'production_backup.json',backup)
    for rel,text in changes.items():(PROJECT/rel).write_text(text)
    try:launch(root,arm,production=True)
    except BaseException:
        for rel,text in backup.items():
            assert (PROJECT/rel).read_text()==changes[rel],('Concurrent edit during rollback',rel)
            (PROJECT/rel).write_text(text)
        write(root/'production_promotion.json',dict(status='rolled_back_parity_failure'));raise
    result=dict(status='applied_and_verified' if features else 'baseline_retained',arm=arm,features=features,
        changes={rel:sha(PROJECT/rel) for rel in changes},native_parity_questions=200,new_http_requests=0,relation_weight=.6)
    write(root/'production_promotion.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--worker',choices=DATASETS)
    p.add_argument('--arm',default='baseline');p.add_argument('--production',action='store_true')
    p.add_argument('--no-promote',action='store_true');args=p.parse_args();root=args.output.resolve()
    if args.worker:worker(root,args.worker,args.arm,args.production);return
    prepare(root)
    if args.prepare_only:return
    import fcntl
    with (root/'.run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(root/'run.pid').write_text(str(os.getpid())+'\n')
        launch(root,'baseline')
        decisions={}
        for arm in FEATURES:
            launch(root,arm);decisions[arm]=assess(root,arm)
            write(root/'selection_decision.json',dict(decisions=decisions,retained=[a for a,d in decisions.items() if d['retain']]))
            print('DECISION',arm,json.dumps(decisions[arm],ensure_ascii=False),flush=True)
        retained=[a for a,d in decisions.items() if d['retain']]
        build(root,'combined',retained);launch(root,'combined');decisions['combined']=assess(root,'combined')
        valid=['baseline']+[a for a,d in decisions.items() if d['retain']]
        def key(a):
            d=assess(root,a);return tuple(d['mean_deltas'][k] for k in ('qa','fcr','recall','correctness','dcc'))+(-d['logical_call_delta'],-len(retained if a=='combined' else [] if a=='baseline' else [a]))
        best=max(valid,key=key);features=retained if best=='combined' else [] if best=='baseline' else [best]
        write(root/'final_decision.json',dict(arm=best,features=features,decisions=decisions))
        promotion=promote(root,best,features) if not args.no_promote else dict(status='not_requested')
        summarize(root)
        write(root/'completed.ok',dict(status='complete',questions=200,single_versions=len(FEATURES),final_arm=best,features=features,production=promotion['status']))
        print('COMPLETE',json.dumps(read(root/'completed.ok'),ensure_ascii=False),flush=True)


if __name__=='__main__':main()

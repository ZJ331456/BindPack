#!/usr/bin/env python3
"""Validate paired artifacts and re-time deterministic Pack replay without HTTP."""
from pathlib import Path
from copy import deepcopy
import argparse,hashlib,json,sys,time
import numpy as np
PROJECT=Path('/root/BindPack')
sys.path[:0]=[str(PROJECT/'src'),str(PROJECT/'scripts')]
from capture_joint_inputs import DATASETS,ARCHIVE,read,write,digest,token

def audit(root,retime=False,retime_limit=None,retime_cases=None):
    from bindpack import BindPackConfig
    from bindpack.evaluation.measurement import Recorder,distribution
    from bindpack.evaluation.measured_reader import prepare
    from run_pack_weight_sweep import ReadOnlyPassages,make_runtime
    from run_pack_weight_diagnostic import restore_pack_input
    from bindpack.utils.misc_utils import QuerySolution
    from transformers import AutoTokenizer
    selection=read(root/'selection_locked.json')
    assert digest(selection['datasets'])==selection['sha256']
    tokenizer=AutoTokenizer.from_pretrained('/root/models/Qwen3-8B',local_files_only=True)
    checked=[];fresh_hashes={};parity_errors=[]
    for ds in DATASETS:
        chosen=selection['datasets'][ds]
        assert len({r['sample_id'] for r in chosen})==len({r['question'] for r in chosen})==200
        assert {r['sample_id'] for r in chosen if r['phase']=='pilot'}.isdisjoint(
               r['sample_id'] for r in chosen if r['phase']=='validation')
        receipt_path=root/'capture'/ds/'index_reuse_verified.ok'
        if not receipt_path.exists():
            assert not (root/'completed.ok').exists(), 'Completed experiment has no verified capture'
            continue
        receipt=read(receipt_path)
        assert receipt['source_and_private_unchanged']
        store=ReadOnlyPassages(next((PROJECT/'outputs/4_datasets_result/shared_indexes'/ds).rglob('vdb_chunk.parquet')))
        for output in sorted((root/'cases'/ds).iterdir()):
            if not (output/'completed.ok').exists():continue
            protocol=read(output/'protocol.json');phase=protocol['phase'];arm=protocol['variant']
            should_retime=retime and arm!='review' and (not retime_cases or ds+'/'+output.name in retime_cases)
            cfg=BindPackConfig.from_file(**protocol['configuration'])
            recorder=Recorder(root/'audit_accounting'/ds/output.name)
            rag=make_runtime(cfg,store,recorder,tokenizer);rag.evidence_runtime._joint_tokenizer=tokenizer
            rag.evidence_runtime._joint_features=protocol['features']
            def deny(*a,**kw):raise AssertionError('Audit requested new inference')
            rag.llm_model.openai_client.chat.completions.create=deny
            rag.llm_model.infer=deny
            result=read(output/'result.json');report=read(output/'report.json');marker=read(output/'completed.ok')
            assert hashlib.sha256((output/'result.json').read_bytes()).hexdigest()==marker['result_sha256']
            rows=result['results'];samples=[r for r in chosen if r['phase']==phase]
            assert len(rows)==len(samples)==100
            seconds=[];method_tokens=CounterLike();n_changed=0
            for sample_number,(row,c) in enumerate(zip(rows,samples)):
                assert row['sample_id']==c['sample_id'] and row['query_index']==c['query_index']
                checkpoint=read(output/'checkpoints'/token(c['sample_id']))
                assert digest(checkpoint['row'])==checkpoint['sha256'] and checkpoint['row']==row
                frozen=read(root/'frozen_inputs'/ds/token(c['sample_id']))
                assert digest(frozen['input'])==frozen['input_sha256']==row['frozen_input_sha256']
                s0=frozen['baseline_doc_ids'];final=row['retrieval_trace']['evidence']['improvement_support_semantic_veto']['final_doc_ids']
                fixed=cfg.pack.joint_fixed_prefix if arm not in {'baseline','review'} else 2
                assert final[:fixed]==s0[:fixed] and set(final)==set(s0) and len(final)==len(set(final))==200
                assert row['candidate_docs']==[store.texts[d] for d in final]
                n_changed+=final[:5]!=s0[:5]
                if arm!='review':assert row['pack_model_requests']==0
                if row.get('qa_executed',True):
                    m=row['reader']['metadata']
                    method_tokens.add('input_tokens',m.get('prompt_tokens'))
                    method_tokens.add('output_tokens',m.get('completion_tokens'))
                if should_retime and (retime_limit is None or sample_number<retime_limit):
                    original=frozen['input'];state=restore_pack_input(original)
                    begin=time.perf_counter()
                    ids,scores,trace=rag.evidence_runtime.pack.finalize(original['query'],np.asarray(original['ids'],int),
                        np.asarray(original['scores'],float),{},state)
                    seconds.append(time.perf_counter()-begin)
                    same_rank=ids[:200].tolist()==final
                    if not same_rank:parity_errors.append({'dataset':ds,'case':output.name,'sample_id':c['sample_id']})
                    sol=QuerySolution(question=c['question'],docs=[store.texts[int(d)] for d in ids[:200]],doc_scores=scores[:200])
                    _,budget=prepare(rag,sol,tokenizer,ds)
                    if same_rank:assert budget['prompt_sha256']==row['reader']['budget']['prompt_sha256']
            report['cost'].pop('usage_for_100_reader_contexts',None)
            report['cost']['usage_for_generated_reader_contexts']={**method_tokens.values,
                'generated_contexts':sum(r.get('qa_executed',True) for r in rows),
                'scope':'Usage on actual saved responses, including exact-prompt reuse. Skipped QA is not unknown usage or a zero quality score.'}
            report['evidence_budget'].update(mean_used_passages=sum(r['reader']['budget']['used_passages'] for r in rows)/100,
                mean_prompt_tokens=sum(r['reader']['budget']['prompt_tokens_local'] for r in rows)/100,
                questions_with_partial_passage=sum(bool(r['reader']['budget']['partial_passage_ranks']) for r in rows),
                context_window=8192,reserved_output_tokens=2048)
            logical=output/'selection_accounting/llm_logical_calls.jsonl'
            if arm!='review' and logical.exists():assert not logical.read_text().strip(),'New selector made a logical LLM call'
            if should_retime:
                report['efficiency']['isolated_pack_replay_seconds']=distribution(seconds)
                report['efficiency']['isolated_pack_replay_sample_count']=len(seconds)
                report['efficiency']['isolated_pack_scope']='CPU selection including actual reader-budget visibility checks; no generation, Route/Plan, indexing or answer reuse time.'
            write(output/'report.json',report)
            checked.append({'dataset':ds,'case':output.name,'queries':100,'changed_top5':n_changed,
                'ranking_parity_checked':bool(should_retime),'ranking_parity_sample_count':len(seconds),
                'pack_calls':sum(r['pack_model_requests'] for r in rows)})
    for p in (PROJECT/'src').rglob('*.py'):fresh_hashes[str(p.relative_to(PROJECT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    value={'status':'passed' if not parity_errors else 'failed','checked':checked,'parity_errors':parity_errors,
        'selection_sha256':selection['sha256'],'current_source_hashes':fresh_hashes,
        'new_model_calls':0,'indexes_rebuilt':0,'retimed':retime,
        'source_snapshot_note':'Compare current_source_hashes with the initial source_fingerprint; ranking parity establishes whether timing-only/determinism fixes changed committed selections.'}
    value['resume_preservation_checks']={}
    for name in ['review_window_transition.json','retrieval_priority_update.json']:
        transition=root/name
        if transition.exists():
            hashes=read(transition)['preserved_query_hashes']
            for rel,expected in hashes.items():
                assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==expected,'Checkpoint changed after planned resume: '+rel
            value['resume_preservation_checks'][name]=len(hashes)
    write(root/'final_audit.json',value)
    assert not parity_errors,parity_errors[:3]
    print('AUDIT',value['status'],'cases',len(checked),'queries',sum(c['queries'] for c in checked),flush=True)

class CounterLike:
    def __init__(self):self.values={'input_tokens':0,'output_tokens':0,'unknown_usage_fields':0}
    def add(self,key,value):
        if value is None:self.values['unknown_usage_fields']+=1
        else:self.values[key]+=value

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--retime',action='store_true')
    p.add_argument('--retime-limit',type=int,help='First N fixed-order samples per case; all checkpoint hashes are still checked.')
    p.add_argument('--retime-case',action='append',help='Dataset/case to replay; artifact checks still cover all completed cases.')
    a=p.parse_args()
    if a.retime_limit is not None and a.retime_limit<1:p.error('--retime-limit must be positive')
    audit(a.output,a.retime,a.retime_limit,a.retime_case)

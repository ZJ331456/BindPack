"""Saved-response compatibility audit of the original-task suffix protocol.

No model service, tokenizer model, graph or embedding instance is constructed.
This verifies parser/ranking compatibility, not a new inference quality result.
"""
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path('/root/BindPack/outputs/support_review_optimization_20261026')
sys.path.insert(0,str(ROOT/'source_snapshot/src'))
sys.path.insert(0,str(ROOT/'source_snapshot/scripts'))


def read(p):return json.loads(p.read_text())
def fingerprint(messages):return hashlib.sha256(json.dumps(messages,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--native',action='store_true')
    parser.add_argument('--mode',choices=['standard','suffix_stop'],default='suffix_stop')
    args=parser.parse_args()
    if args.native:sys.path.insert(0,'/root/BindPack/src')
    from bindpack import BindPackConfig
    from bindpack.utils.misc_utils import QuerySolution
    import prefix_support_review
    if args.native:
        from bindpack.stages.pack.verification.support_semantic_veto import postprocess_support_semantic_veto
    results={}
    for ds in ('hotpotqa','2wikimultihopqa','musique','hover'):
        if ds=='hotpotqa':
            source=Path('/root/BindPack/outputs/4_datasets_result/latest_r060_20261026/cases/hotpotqa/full/result.json')
            originals=read(source)['results'];controls=None
        else:
            source=(ROOT/'generalization_sources/2wikimultihopqa/result.json' if ds=='2wikimultihopqa' else
                Path(f'/root/BindPack/outputs/pack_150_before_after_20261026/cases/{ds}/after/result.json'))
            old={r['sample_id']:r for r in read(source)['results']}
            controls=[read(p) for p in (ROOT/f'cases/{ds}/generalization_current/checkpoints').glob('*.json')]
            originals=[old[c['sample_id']] for c in controls]
        config=BindPackConfig.from_file(ROOT/'method.json',dataset=ds)
        if args.native:config.pack.review_mode=args.mode
        passed=0;calls=0;failed=[]
        for index,old in enumerate(originals):
            control=None if controls is None else controls[index]
            saved=old['retrieval_trace']['evidence']['improvement_support_semantic_veto'] if control is None else control['review_trace']
            own=old['retrieval_trace']['evidence']['improvement_support_semantic_veto']
            docs=dict(zip(own['final_doc_ids'],old['candidate_docs']))
            parent=saved['original_parent_doc_ids'];expected=saved['final_doc_ids']
            trace=deepcopy(old['retrieval_trace']['evidence']);trace.pop('improvement_support_semantic_veto',None)
            trace['selected_prefix']=[dict(doc_id=d,selection_source='frozen_pre_review') for d in parent[:5]]
            def stage_for(messages):
                return 'reviewer' if 'PROPOSAL' in json.loads(messages[-1]['content']) else 'proposer'
            def tokens(messages,**kwargs):
                stage=stage_for(messages);pre=saved['llm_stages'][stage]['preflight']
                assert pre is not None and pre['messages_sha256']==fingerprint(messages),'Prompt identity mismatch'
                return range(pre['prompt_tokens'])
            def infer(messages,stop=None,**kwargs):
                nonlocal calls
                stage=stage_for(messages);output=saved['llm_stages'][stage]['output']
                assert output is not None,'Suffix protocol requested an unrecorded stage'
                raw=output['response'];positions=[raw.find(s) for s in (stop or []) if raw.find(s)>=0]
                if positions:raw=raw[:min(positions)]
                calls+=1
                return raw,dict(output['metadata']),True
            rag=SimpleNamespace(llm_model=SimpleNamespace(infer=infer),_extract_llm_text=str)
            runtime=SimpleNamespace(cfg=config,rag=rag,_document=docs.__getitem__,
                _support_semantic_tokenizer=SimpleNamespace(apply_chat_template=tokens))
            solution=QuerySolution(question=old['question'],docs=[docs[d] for d in parent],
                doc_scores=[1.]*200,retrieval_trace={'evidence':trace})
            try:
                if args.native:
                    postprocess_support_semantic_veto(runtime,[(solution,parent)])
                    diag=solution.retrieval_trace['evidence']['improvement_support_semantic_veto']
                else:diag=prefix_support_review.run(runtime,solution,parent,mask=False)
                if diag['final_doc_ids']!=expected:
                    failed.append(dict(sample_id=old['sample_id'],reason='ranking_changed'))
                else:passed+=1
            except Exception as error:
                failed.append(dict(sample_id=old['sample_id'],reason=str(error)))
        results[ds]=dict(queries=len(originals),ranking_parity=passed,logical_calls_replayed=calls,failures=failed)
    output=dict(scope='Saved raw response replay; same prompt hashes, no physical HTTP or new QA inference',
        new_model_requests=0,datasets=results,all_rankings_identical=all(not d['failures'] for d in results.values()))
    output['native_engine']=args.native;output['native_mode']=args.mode if args.native else None
    filename=('native_'+args.mode+'_compatibility.json') if args.native else 'suffix_compatibility_audit.json'
    (ROOT/filename).write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(output,ensure_ascii=False,indent=2))


if __name__=='__main__':main()

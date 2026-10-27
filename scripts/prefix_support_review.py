"""Experimental native task with server-side stopping of unused JSON suffixes.

The exact raw partial response and provider usage are retained. Only omitted
closing delimiters are reconstructed; missing source/check fields never pass.
"""
from copy import deepcopy
import json

import numpy as np

from bindpack.stages.pack.typed_joint_selector import build_joint_prompt
from bindpack.stages.pack.verification.support_guard import support_guard
from bindpack.stages.pack.verification.support_semantic_veto import _state,_preflight
from bindpack.stages.pack.verification.review_prompt import build_verifier_prompt,_proposal,_IDENTIFIER,_CHECKS
from tight_support_review import validate as validate_source_swap


STOPS={
    'proposer':['"reason":','"candidate_quote": ""','"root_quote": ""'],
    'reviewer':['"victim_reason":','"same_entity_event": false','"old_missing_fact": ""','"candidate_supplied_fact": ""'],
}


def decode_prefix(text,stage):
    try:
        value=json.loads(text)
    except (ValueError,TypeError):
        if not isinstance(text,str):return None
        prefix=text.rstrip()
        if prefix.endswith(','):prefix=prefix[:-1].rstrip()
        try:value=json.loads(prefix+('}}' if stage=='proposer' else '}'))
        except (ValueError,TypeError):return None
    return value if isinstance(value,dict) else None


def request(runtime,messages,stage):
    check=_preflight(runtime,messages)
    if not check['allowed']:return None,check
    raw,meta,hit=runtime.rag.llm_model.infer(messages,response_format={'type':'json_object'},stop=STOPS[stage])
    raw=runtime.rag._extract_llm_text(raw)
    decoded=decode_prefix(raw,stage)
    return dict(raw_response=raw,response=json.dumps(decoded,ensure_ascii=False) if decoded is not None else '{}',
        metadata=meta,cache_hit=hit,stop_sequences=STOPS[stage],
        json_closure_reconstructed=decoded is not None and raw.rstrip()[-1:]!='}'),check


def check_review(order,state,proposal,output):
    diag=dict(approved=False,policy='native_semantic_decision_prefix',entailment_guaranteed=False,
        suffix_explanations_generated=False,reviewer_quotes_used_as_evidence=False,rejection_reason=None,checks={})
    def reject(reason):diag['rejection_reason']=reason;return False,diag
    ids,reason=_proposal(order,state,proposal)
    if reason:return reject(reason)
    if output['metadata'].get('finish_reason')!='stop':return reject('review_response_not_completed')
    value=decode_prefix(output['raw_response'],'reviewer')
    if value is None:return reject('incomplete_early_rejection_prefix')
    for key,doc in ids.items():
        x=value.get(key)
        if not isinstance(x,str) or not _IDENTIFIER.fullmatch(x) or int(x[1:])!=doc:return reject('review_document_id_missing_or_mismatch:'+key)
    diag['checks']={k:value.get(k) for k in _CHECKS}
    if any(type(value.get(k)) is not bool for k in _CHECKS):return reject('incomplete_early_rejection_checks')
    failed=[k for k in _CHECKS if not value[k]]
    if failed:return reject('review_conditions_not_met:'+','.join(failed))
    for key in ('old_missing_fact','candidate_supplied_fact'):
        if not isinstance(value.get(key),str) or len(value[key].strip())<8:return reject('missing_semantic_review_explanation:'+key)
        diag[key]=value[key]
    diag.update(approved=True,proposal=ids);return True,diag


def run(runtime,solution,order,mask=True):
    order=[int(d) for d in order]
    assert len(order)==len(set(order))==len(solution.docs)
    assert all(runtime._document(d)==text for d,text in zip(order,solution.docs))
    trace=deepcopy(solution.retrieval_trace['evidence']);state=_state(trace)
    _,early=support_guard(solution.question,order,state,runtime._document,None)
    messages=build_joint_prompt(solution.question,order,state,runtime._document)
    if set(order[2:5])<=set(early['protected_doc_ids']):messages=None
    if messages is not None and mask:
        payload=json.loads(messages[1]['content'])
        for item in payload['passages']:item['protected']=int(item['id'][1:]) in set(early['protected_doc_ids'])
        payload.update(allowed_promote=[f'D{d}' for d in order[5:10]],
            allowed_replace=[f'D{d}' for d in order[2:5] if d not in set(early['protected_doc_ids'])],
            allowed_root=[f'D{d}' for d in order[:5]])
        messages[1]={'role':'user','content':json.dumps(payload,ensure_ascii=False,sort_keys=True)}
    output=preflight=None
    if messages is not None:output,preflight=request(runtime,messages,'proposer')
    proposed,proposal=validate_source_swap(solution.question,order,state,runtime._document,
        output['response'] if output else '{"swap":null}',output['metadata']['finish_reason'] if output else 'stop')
    if proposal.get('promotions'):
        body=decode_prefix(output['raw_response'],'proposer')['swap']
        condition=body.get('missing_condition')
        if not isinstance(condition,str) or len(condition.strip())<8:
            proposed=order;proposal.update(promotions=[],abstain_reason='missing_generated_condition')
        else:proposal['promotions'][0]['missing_condition']=condition
    if output is None:proposal['abstain_reason']='scope_or_local_protection_skip'
    proposal['reason_suffix_generated']=False
    allowed,guard=support_guard(solution.question,order,state,runtime._document,proposal)
    review=review_check=None;approved=False;semantic={'approved':False,'rejection_reason':'no_locally_valid_proposal'}
    if proposed!=order and allowed:
        audit=build_verifier_prompt(solution.question,order,state,runtime._document,proposal)
        review,review_check=request(runtime,audit,'reviewer')
        if review:approved,semantic=check_review(order,state,proposal,review)
    accepted=proposed!=order and allowed and approved;final=proposed if accepted else order
    invariants=dict(top2_preserved=final[:2]==order[:2],top10_set_preserved=set(final[:10])==set(order[:10]),
        top200_set_preserved=set(final)==set(order),suffix_preserved=final[10:]==order[10:],
        unique_documents=len(final)==len(set(final)),protected_support_preserved=set(guard['protected_doc_ids'])<=set(final[:5]))
    assert all(invariants.values())
    diag=dict(enabled=True,experimental=True,gold_labels_used=False,policy='native_task_prefix_stop_masked' if mask else 'native_task_prefix_stop',
        accepted=bool(accepted),original_parent_doc_ids=order,final_doc_ids=final,
        extra_requests=int(output is not None)+int(review is not None),base_proposal=proposal,support_guard=guard,
        independent_verification=semantic,llm_stages={'proposer':{'preflight':preflight,'output':output},
            'reviewer':{'preflight':review_check,'output':review}},promotions=proposal['promotions'] if accepted else [],**invariants)
    docs=dict(zip(order,solution.docs));solution.docs=[docs[d] for d in final]
    if accepted:solution.doc_scores=np.arange(len(final),0,-1,dtype=float)/len(final)
    trace['improvement_support_semantic_veto']=diag;solution.retrieval_trace={'stage_ablation':'full','evidence':trace}
    return diag

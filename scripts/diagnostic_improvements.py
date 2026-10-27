"""Isolated candidates for the locked R=.60 diagnostic; never read gold labels."""
from __future__ import annotations

from copy import deepcopy
import re
from string import Template
import numpy as np

from bindpack.stages.pack.coverage_selector import CoverageSelector
from bindpack.stages.plan.relation_guard import relation_spec
from bindpack.stages.plan.answer_binder import bind_question
from bindpack.stages.plan.answer_validation import _typed_answer_reason
from bindpack.stages.plan.plan_canonicalizer import canonicalize_plan_dependencies
from bindpack.stages.plan.plan_validator import (prune_derived_comparison_nodes,
    validate_retrieval_plan, relation_faithfulness_error)
from bindpack.stages.plan.validation import validate_plan
from bindpack.stages.pack.proof_scope import _specific, _literal, _title
from bindpack.utils.evidence import normalized_text, branch_matches


HOVER_SYSTEM = (
    'Verify the claim using ONLY the provided Wikipedia passages. Silently split '
    'the claim into its individual factual statements and identify which entity '
    'each name, description and pronoun refers to. Follow explicit relationships '
    'across passages: a statement may be supported jointly by several passages '
    'and need not appear verbatim or in a single passage. Check every stated '
    'relation and qualifier, including time, location, identity and conjunctions. '
    'Do not mistake a passage about a related entity for support, and do not use '
    'outside knowledge to fill a missing bridge. Return SUPPORTED when every '
    'necessary statement is supported by the supplied evidence. Return '
    'NOT_SUPPORTED if any necessary statement is contradicted or lacks sufficient '
    'evidence. Do not assume either label in advance. Return exactly one JSON '
    'object with the key answer and the value SUPPORTED or NOT_SUPPORTED; '
    'do not output your analysis or any other text.'
)


def fact_key(question, requirements):
    """Exact questions or recognized same subject/relation, with branch context."""
    spec = relation_spec(question)
    if spec['relation'] not in {'other', 'comparison', 'award_date'} and spec.get('subject'):
        identity = (spec['relation'], normalized_text(spec['subject']))
    else:
        identity = ('question', normalized_text(question))
    context = tuple(sorted((str(k), normalized_text(v)) for k, v in requirements.items()))
    return repr((identity, context))


def coverage_facts(state, candidate, bindings):
    profile = {}
    for source in candidate['sources']:
        if branch_matches(source['requirements'], bindings):
            key = fact_key(source['question'], source['requirements'])
            profile[key] = max(profile.get(key, 0.0), source['score'])
    nodes = {n['id']: n for n in state.get('_evidence_plan', [])}
    for proof in candidate['verified']:
        goal = proof['goal']
        node = nodes.get(goal[4:]) if goal.startswith('dag:') else None
        question = bind_question(node['question'], bindings) if node else None
        if question:
            requirements = {dep: bindings[dep] for dep in node['depends_on'] if dep in bindings}
            key = fact_key(question, requirements)
            profile[key] = max(profile.get(key, 0.0), proof['quality'])
    return profile


class FactCoverageSelector(CoverageSelector):
    """Change coverage identities only; preserve R, weights and redundancy."""

    def select(self, query, ids, scores, ctx, state):
        if self.runtime.cfg.stage_ablation == 'wo_pack' or not state.get('evidence_candidates'):
            return super().select(query, ids, scores, ctx, state)
        from bindpack.stages.pack.redundancy_controller import max_token_redundancy
        candidates = state['evidence_candidates']
        trace = state.get('evidence_trace', self.runtime._trace())
        ids, scores = np.asarray(ids, dtype=int), np.asarray(scores, dtype=float)
        finite = np.where(np.isfinite(scores), scores, 0.0)
        span = float(finite.max() - finite.min()) if len(finite) else 0.0
        norm = (finite - finite.min()) / span if span > 0 else np.ones(len(finite))
        base = dict(zip(ids.tolist(), norm.tolist()))
        bw = float(self.config.base_weight)
        bindings = state.get('_evidence_winning_bindings', {})
        goals = {d:self.scorer.goal_scores(c, bindings) for d,c in candidates.items()}
        profiles = {d:coverage_facts(state,c,bindings) for d,c in candidates.items()}
        # The relevance component intentionally retains its exact old scores.
        relations = {d:max(v.values(),default=0.0) for d,v in goals.items()}
        merged = {d:bw*v for d,v in base.items()}
        for d in candidates:
            merged[d] = bw*base.get(d,candidates[d]['base_score']) + (1-bw)*relations[d]
        ordered = sorted(merged,key=lambda d:(-merged[d],d))
        total = max(1,len({g for p in profiles.values() for g in p}))
        covered, selected, selected_tokens, details = {}, [], [], []
        remaining = set(candidates)
        document_tokens = {d:set(normalized_text(self.runtime._document(d)).split()) for d in remaining}
        for _ in range(min(self.config.budget,len(remaining))):
            ranked=[]
            for d in remaining:
                gain=sum(max(0,v-covered.get(g,0)) for g,v in profiles[d].items())/total
                red=max_token_redundancy(document_tokens[d],selected_tokens)
                utility=(bw*base.get(d,candidates[d]['base_score'])
                    +self.config.relation_weight*relations[d]+self.config.coverage_weight*gain
                    -self.config.redundancy_weight*red)
                ranked.append((utility,merged[d],-d,gain,red,d))
            utility,_,_,gain,red,d=max(ranked)
            selected.append(d);remaining.remove(d);selected_tokens.append(document_tokens[d])
            for g,v in profiles[d].items():covered[g]=max(covered.get(g,0),v)
            details.append(dict(doc_id=d,utility=float(utility),coverage_gain=float(gain),
                redundancy=float(red),goals=goals[d],coverage_facts=profiles[d],verified=candidates[d]['verified']))
        tail=[d for d in ordered if d not in set(selected)]
        top=max(merged.values(),default=1.0)+1.0
        output_scores=[top+(len(selected)-i)/max(1,len(selected)) for i in range(len(selected))]
        trace.update(selected_prefix=details,covered_goals=covered,
            coverage_identity='subject_relation_and_branch_or_exact_question',coverage_fact_count=total)
        return np.asarray(selected+tail),np.asarray(output_scores+[merged[d] for d in tail]),trace


def recover_plan(query, hops, diagnostics, plan_config):
    """Only rescue an otherwise failed plan using existing literal references."""
    limit = int(plan_config.node_budget) if diagnostics.get('routing',{}).get('expand') else max(1,min(4,int(hops)))
    for ordinal, payload in enumerate(diagnostics.get('outputs', [])):
        canonical, changes, reason = canonicalize_plan_dependencies(payload)
        if reason:continue
        nodes, reason = validate_plan(canonical, limit+1)
        if reason:continue
        nodes, removed = prune_derived_comparison_nodes(nodes)
        canonical = dict(canonical,nodes=nodes)
        nodes, reason = validate_retrieval_plan(canonical,limit,int(plan_config.depth_budget))
        if reason or relation_faithfulness_error(query,nodes):continue
        return nodes,dict(recovered=True,output_index=ordinal,canonicalization_changes=changes,
                          pruned_nodes=removed,extra_planner_requests=0)
    return [],dict(recovered=False,extra_planner_requests=0)


def positive_path_fusion(pre_ids, pre_scores, native_ids, native_scores, corpus_size):
    """Keep the original PathSet boost, but never penalize its selected paths."""
    from bindpack.utils.misc_utils import min_max_normalize
    base=np.zeros(corpus_size,dtype=np.float32)
    base[np.asarray(pre_ids,dtype=int)]=np.asarray(pre_scores)
    base=np.asarray(min_max_normalize(base),dtype=np.float32)
    current=np.zeros(corpus_size,dtype=np.float32)
    current[np.asarray(native_ids,dtype=int)]=np.asarray(native_scores)
    changed=int(np.count_nonzero(current<base))
    current=np.maximum(current,base)
    ids=np.argsort(current)[::-1]
    return ids,current[ids],dict(policy='nonnegative_pathset_score_adjustment',restored_penalized_passages=changed)


def complete_chain_package(runtime, result, state, query, selector_ids, selector_scores):
    """A bounded full winning chain, grounded in the existing verifier outputs.

    This is an explicitly fallible alternative selector, not an entailment proof.
    It preserves existing strict-DAG supports and explicit question anchors.
    """
    ids,scores,trace=result
    original=[int(d) for d in ids]
    diag=dict(enabled=True,applied=False,policy='complete_source_chain_with_relevance_loss_bound',
              extra_requests=0,gold_labels_used=False,entailment_guaranteed=False)
    def unchanged(reason):
        diag['reason']=reason
        return ids,scores,dict(trace,complete_chain_package=diag)
    plan=state.get('_evidence_plan',[]);bindings=state.get('_evidence_winning_bindings',{})
    if len(plan)<2 or not {n['id'] for n in plan}<=set(bindings):return unchanged('incomplete_or_short_plan')
    branches=state.get('_evidence_beams',[])
    winning=next((b for b in branches if set(b.get('bindings',{}))==set(bindings)
                  and branch_matches(b['bindings'],bindings)),None)
    if not winning:return unchanged('no_exact_winning_branch')
    proofs=winning.get('proofs',{});source_by_node={}
    for n in plan:
        proof=proofs.get(n['id'],{})
        try:d=int(proof['doc_id']);confidence=float(proof['confidence'])
        except (KeyError,ValueError,TypeError):return unchanged('missing_source_proof')
        document=runtime._document(d)
        quote=str(proof.get('evidence',''));answer=str(proof.get('answer',''))
        if (d not in original[:200] or d not in state['evidence_candidates']
            or not np.isfinite(confidence) or confidence<.85 or not 8<=len(quote)<=1500
            or quote not in document or normalized_text(answer) not in normalized_text(quote)
            or normalized_text(answer)!=normalized_text(bindings[n['id']])
            or _typed_answer_reason(answer,n['answer_type'])
            or not branch_matches(proof.get('requirements',{}),bindings)):
            return unchanged('source_binding_or_type_not_qualified')
        source_by_node[n['id']]=d
    prefix=set(original[:5]);anchors={d for d in prefix if _specific(_title(runtime._document(d)))
        and _literal(_title(runtime._document(d)),query)}
    strict={p['doc_id'] for rows in trace.get('improvement_dag_package',{}).get('eligible_proofs',{}).values()
            for p in rows if p['doc_id'] in prefix}
    chosen=set(source_by_node.values())|anchors|strict
    if len(chosen)>5:return unchanged('package_exceeds_five_passages')
    old_closed=set()
    for n in plan:
        if source_by_node[n['id']] in prefix and set(n['depends_on'])<=old_closed:old_closed.add(n['id'])
    if len(old_closed)==len(plan):return unchanged('full_chain_already_present')
    for d in original:
        if len(chosen)>=5:break
        chosen.add(d)
    base_scores=np.asarray(selector_scores,dtype=float)
    base_ids=np.asarray(selector_ids,dtype=int)
    finite=np.where(np.isfinite(base_scores),base_scores,0)
    span=float(finite.max()-finite.min())
    norm=(finite-finite.min())/span if span>0 else np.ones(len(finite))
    base=dict(zip(base_ids.tolist(),norm.tolist()))
    scorer=runtime.pack.scorer
    def relevance(d):
        candidate=state['evidence_candidates'].get(d)
        relation=max(scorer.goal_scores(candidate,bindings).values(),default=0) if candidate else 0
        return runtime.cfg.pack.base_weight*base.get(d,0)+runtime.cfg.pack.relation_weight*relation
    loss=sum(relevance(d) for d in prefix)-sum(relevance(d) for d in chosen)
    allowance=runtime.cfg.pack.coverage_weight*(len(plan)-len(old_closed))/len(plan)
    if loss>allowance+1e-12:return unchanged('relevance_loss_exceeds_closure_gain_budget')
    output=[d for d in original if d in chosen]+[d for d in original if d not in chosen]
    assert len(output)==len(original)==len(set(output)) and set(output[:200])==set(original[:200])
    selected={d['doc_id']:d for d in trace.get('selected_prefix',[])}
    diag.update(applied=True,reason=None,closed_before=len(old_closed),closed_after=len(plan),
        package_doc_ids=sorted(set(source_by_node.values())),relevance_loss=float(loss),loss_bound=float(allowance))
    trace=dict(trace,complete_chain_package=diag,selected_prefix=[dict(selected.get(d,{}),doc_id=d,
        selection_source='complete_source_chain') for d in output[:5]])
    return np.asarray(output),np.arange(len(output),0,-1,dtype=float)/len(output),trace


def enable_hover_reader(rag):
    if rag.config.dataset=='hover':
        rag.prompt_template_manager.templates['rag_qa_hover'][0]['content']=Template(HOVER_SYSTEM)

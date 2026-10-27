#!/usr/bin/env python3
"""CPU-only counterexamples and safety checks for each frozen candidate."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT=Path('/root/BindPack')
DEFAULT=ROOT/'outputs/pack_candidates_r060_20261026'


def check(root,arm):
    sys.path.insert(0,str(root/'variants'/arm/'src'));sys.path.insert(0,str(ROOT/'tests'))
    import numpy as np
    from bindpack import BindPackConfig
    from bindpack.stages.pack.coverage_selector import CoverageSelector
    from bindpack.stages.pack.evidence_scorer import EvidenceScorer
    from bindpack.stages.pack.joint_selector import _scope
    from bindpack.stages.pack.proof_scope import _plan_reason
    from test_evidence_support import fixture,QUERY
    from bindpack.stages.pack.verification.support_semantic_veto import postprocess_support_semantic_veto
    from test_evidence_dag_package import Engine,parallel,run
    checks=[]
    if arm=='short_circuit':
        runtime,solution,llm,docs=fixture(protect_victim=True)
        old=deepcopy(solution.docs);postprocess_support_semantic_veto(runtime,[(solution,list(range(20)))])
        assert solution.docs==old and len(llm.calls)==1
        checks.append('Local veto preserves ranking with one proposer and zero reviewer calls')
        runtime,solution,llm,docs=fixture()
        # Physical question-title anchors protect all three replacement slots.
        for d,title in zip((2,3,4),('Alpha Example','Beta Example','Gamma Example')):
            docs[d]=title+'\nThis is a literal source article.';solution.docs[d]=docs[d]
        solution.question=QUERY+' Alpha Example, Beta Example and Gamma Example.'
        old=deepcopy(solution.docs);postprocess_support_semantic_veto(runtime,[(solution,list(range(20)))])
        assert solution.docs==old and len(llm.calls)==0
        checks.append('All victims protected: no proposer/reviewer request')
    if arm=='complete_dag_gate':
        runtime,solution,llm,docs=fixture();state={'evidence_trace':deepcopy(solution.retrieval_trace['evidence'])}
        state['evidence_trace']['improvement_dag_package']={'eligible_package_count':1,'prefix_optimization':[{'top_k':5,'coverage_after':['s1']}]}
        assert _scope(QUERY,list(range(20)),state) is None
        state['evidence_trace']['improvement_dag_package']['prefix_optimization'][0]['coverage_after']=['s1','s2']
        assert _scope(QUERY,list(range(20)),state)=='already_supported_dag_package'
        checks.append('Partial root package remains eligible; complete actual Top5 plan abstains')
    if arm in {'progressive_coverage','goal_redundancy','route_pool200'}:
        cfg=BindPackConfig();runtime=SimpleNamespace(cfg=cfg,_trace=lambda:{},_document=lambda d:'Shared Name\nShared topic and different factual evidence '+str(d))
        candidates={i:dict(base_score=1.,sources=[dict(goal='g'+str(i%2),score=.9,requirements={})],verified=[]) for i in range(6)}
        state=dict(evidence_candidates=candidates,evidence_trace={},_evidence_winning_bindings={})
        out,score,trace=CoverageSelector(runtime,EvidenceScorer()).select('Q',np.arange(12),np.linspace(1,0,12),{},state)
        assert len(out)==len(set(out))==12 and set(out)==set(range(12))
        assert cfg.pack.relation_weight==.6
        if arm=='progressive_coverage':assert trace['selected_prefix'][2]['coverage_gain']>0
        if arm=='goal_redundancy':assert trace['selected_prefix'][1]['redundancy']==0
        if arm=='route_pool200':
            assert set(state['evidence_candidates'])==set(range(12))
            assert all(not state['evidence_candidates'][d]['sources'] and not state['evidence_candidates'][d]['verified'] for d in range(6,12))
        checks.append('Preserve all physical IDs and R=.60; no model or invented proof required')
    if arm=='deep_chain_review':
        plan=[dict(id='a',depends_on=[],question='Who directed X?',answer_type='person'),
              dict(id='b',depends_on=['a'],question='Where was ${a.answer} born?',answer_type='place'),
              dict(id='c',depends_on=['b'],question='Where is ${b.answer}?',answer_type='place')]
        trace={'plan':plan};assert _plan_reason(QUERY,trace,{'_evidence_plan':plan}) is None
        parallel_plan=deepcopy(plan);parallel_plan[1]['depends_on']=[]
        assert _plan_reason(QUERY,{'plan':parallel_plan},{'_evidence_plan':parallel_plan})=='parallel_roots_out_of_scope'
        checks.append('Allow serial depth three while retaining parallel-root exclusion')
    if arm=='dag_score_guard':
        e=Engine();state=parallel(e);trace=state['evidence_trace']
        trace['pointwise_relevance_for_dag']={str(i):1. if i<5 else 0. for i in range(220)}
        ids,_,result=run(e,state)
        assert ids.tolist()==list(range(220)) and result['dag_relevance_guard']['rejected']
        checks.append('High relevance loss vetoes an otherwise eligible strict DAG promotion')
    if arm=='budget_contract':
        from bindpack.stages.pack.dag_package_selector import rerank_dag_packages
        for k in (3,5,8,10):
            cfg=BindPackConfig.from_file(pack={'budget':k},qa_top_k=k)
            e=Engine();state=parallel(e);state.update(_pack_budget=k,_pack_node_budget=6,_pack_depth_budget=4)
            ids,diag=rerank_dag_packages(range(220),state,e._document)
            assert [p['top_k'] for p in diag['prefix_optimization']]==[k,2*k]
            assert len(ids)==len(set(ids))==220 and set(ids[:200])==set(range(200))
            assert cfg.qa_top_k==cfg.pack.budget==k
        try:BindPackConfig.from_file(pack={'budget':3})
        except ValueError:pass
        else:raise AssertionError('Mismatched reader/Pack budget was accepted')
        plan=[dict(id='n'+str(i),depends_on=[],question='Who wrote Work '+str(i)+'?',answer_type='person') for i in range(7)]
        state={'_evidence_plan':plan,'_evidence_winning_bindings':{},'_pack_node_budget':7,'evidence_trace':{}}
        _,diag=rerank_dag_packages(range(220),state,lambda i:'Unrelated article')
        assert diag['plan_error']=='missing_exact_winning_branch'
        checks.append('Budgets 3/5/8/10 align; mismatches reject; configured seven-node plan passes node-budget check')
    assert checks
    p=root/'invariant_checks'/f'{arm}.json';p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(dict(status='passed',checks=checks,model_requests=0),indent=2)+'\n')
    print(arm,'PASS',len(checks),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=DEFAULT);p.add_argument('--arm');a=p.parse_args()
    if a.arm:check(a.output,a.arm);return
    from build_pack_candidates import FEATURES
    for arm in FEATURES:subprocess.run([sys.executable,'-B',__file__,'--output',str(a.output),'--arm',arm],check=True)
    (a.output/'invariant_checks/passed.ok').write_text(json.dumps(dict(status='passed',variants=8,model_requests=0))+'\n')


if __name__=='__main__':main()

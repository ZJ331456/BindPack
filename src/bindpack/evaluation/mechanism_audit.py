"""Fail a mechanism run if a disabled component leaks into a saved trace."""


def audit_mechanism_rows(rows, arm, source_bound_eligibility=True):
    from ..stages.mechanisms import mechanisms
    from types import SimpleNamespace
    from dataclasses import asdict
    expected=asdict(mechanisms(SimpleNamespace(mechanism_ablation=arm,
        source_bound_eligibility=source_bound_eligibility)))
    for row in rows:
        trace=row['retrieval_trace']['evidence']
        if arm=='full' and expected['source_bound_eligibility']: continue
        assert trace.get('mechanism_ablation')==arm
        assert trace.get('mechanisms')==expected
        if not expected['source_bound_eligibility']:
            for output in trace.get('verification_outputs',[]):
                assert output.get('source_bound_eligibility') is False
                assert output.get('mode')=='unverified_answer_extraction'
            for source in trace.get('source_records',[]):
                assert source.get('source_verified') is False
            for name in ('improvement_dag_package','budgeted_selection'):
                for proofs in trace.get(name,{}).get('eligible_proofs',{}).values():
                    allowed = ('unbound_extraction_candidate' if not expected['source_binding'] else
                               'eligibility_disabled_unverified')
                    assert all(p.get('check')==allowed for p in proofs)
        if arm=='wo_source_binding':
            assert trace.get('source_binding') is False
            assert trace.get('source_binding_parent_context_exposed') is False
            assert not trace.get('verification_outputs') and not trace.get('branch_scores')
            for source in trace.get('source_records',[]):
                assert source.get('unbound_extraction_candidate') is True
                assert not source.get('requirements') and not source.get('answer') and not source.get('evidence')
        if arm=='wo_parent_answer_propagation':
            assert trace.get('execution_mode')=='dependency_parent_question_descriptions'
            assert trace.get('parent_answer_propagation') is False
            assert all(not r.get('requirements') for r in trace.get('routes',[]))
        if arm=='wo_dependency_closure':
            assert trace.get('execution_mode')=='independent_no_answer_propagation'
            assert trace.get('dependency_edges_executed')==0
            assert all(not n.get('depends_on') and '${' not in n['question'] for n in trace.get('plan',[]))
            assert all(not r['source'].startswith('dag:') and not r['requirements'] for r in trace.get('routes',[]))
            assert not trace.get('improvement_dag_package')
            selected=trace['budgeted_selection']
            assert selected['package_count']==0 and selected['eligible_proofs']=={}
            assert selected['closed_nodes']==[] and selected['predicted_closure']==0
        if arm=='wo_joint_selection':
            assert not trace.get('budgeted_selection')
    return {'status':'passed','rows':len(rows),'arm':arm,'enabled_mechanisms':expected,
            'scope':'native Route+Plan+Pack; no frozen upstream replay'}

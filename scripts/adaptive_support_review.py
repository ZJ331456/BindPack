"""Diagnostic protocol routing using only native plan binding completeness.

This is a post-hoc HotpotQA development hypothesis, not a held-out result.
There are no dataset names, sample IDs or reference labels in the routing rule.
"""


def choose_protocol(trace):
    plan=trace.get('plan') or []
    bindings=trace.get('bindings') or {}
    complete=bool(plan) and isinstance(bindings,dict) and all(
        isinstance(n,dict) and n.get('id') in bindings and bool(bindings[n['id']]) for n in plan)
    return 'native_prefix' if complete else 'tight_citations'


def run(runtime,solution,order):
    route=choose_protocol(solution.retrieval_trace['evidence'])
    if route=='native_prefix':
        from prefix_support_review import run as execute
        diag=execute(runtime,solution,order,mask=False)
    else:
        from tight_support_review import run as execute
        diag=execute(runtime,solution,order)
    diag['budget_route']=route
    diag['budget_route_signal']='native_nonempty_plan_all_nodes_bound'
    return diag

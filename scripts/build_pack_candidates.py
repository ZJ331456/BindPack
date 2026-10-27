"""Build isolated Pack variants from one frozen source; never inspect gold data."""
from __future__ import annotations

import ast
from pathlib import Path


FEATURES = {
    'short_circuit': 'Skip proposer when every victim is protected; skip reviewer after local veto.',
    'complete_dag_gate': 'Abstain for DAG packages only when the actual prefix closes the entire predicted plan.',
    'progressive_coverage': 'Noisy-OR marginal coverage, retaining original goal IDs, R=.60 and all weights.',
    'goal_redundancy': 'Lexical redundancy weighted by overlap of goal support; exempt newly verified goals.',
    'route_pool200': 'Admit existing final Route Top200 passages with their actual base score and no invented goal/proof.',
    'deep_chain_review': 'Allow simple serial plans deeper than two nodes; retain all other scope/source/typed checks.',
    'dag_score_guard': 'Keep strict DAG eligibility, bounding pointwise relevance loss by normalized closure gain.',
    'budget_contract': 'Align greedy/DAG/reader budget and Plan/DAG limits; mark semantic review unsupported outside verified K=5.',
}


def sources(root, features):
    root = Path(root)
    features = set(features)
    if features - FEATURES.keys():
        raise ValueError(features - FEATURES.keys())
    changes = {}

    def edit(name, old, new):
        path = 'src/bindpack/' + name
        text = changes.get(path, (root / path).read_text())
        if text.count(old) != 1:
            raise ValueError((name, 'expected one source anchor', text.count(old), old[:100]))
        changes[path] = text.replace(old, new, 1)

    if 'short_circuit' in features:
        edit('stages/pack/verification/support_semantic_veto.py',
             '                    messages = build_joint_prompt(solution.question, old, state, runtime._document)',
             '''                    _, early_guard = support_guard(solution.question, old, state, runtime._document, None)
                    no_victim = bool(old[2:5]) and set(old[2:5]) <= set(early_guard["protected_doc_ids"])
                    messages = None if no_victim else build_joint_prompt(solution.question, old, state, runtime._document)''')
        edit('stages/pack/verification/support_semantic_veto.py',
             '                                     "proposer_messages": messages})',
             '                                     "proposer_messages": messages, "no_legal_victim": no_victim})')
        edit('stages/pack/verification/support_semantic_veto.py',
             '                    allowed, guard = support_guard(query, old, state, runtime._document, proposal)',
             '''                    if ctx.get("no_legal_victim"):
                        proposal = dict(proposal, abstain_reason="all_replacement_positions_protected")
                    allowed, guard = support_guard(query, old, state, runtime._document, proposal)''')
        edit('stages/pack/verification/support_semantic_veto.py',
             '''                    # Review every accepted base proposal, including locally
                    # vetoed ones, to match the evaluated two-control protocol.
                    if proposed != old:''',
             '''                    # A local veto cannot be overturned by independent review.
                    if proposed != old and allowed:''')
        edit('stages/pack/verification/support_semantic_veto.py',
             '                    accepted = ctx["proposed"] != old and ctx["guard_allowed"] and approved',
             '''                    if ctx["proposed"] != old and not ctx["guard_allowed"]:
                        semantic = dict(semantic, review_requested=False,
                                        rejection_reason="not_reviewed_local_support_veto")
                    accepted = ctx["proposed"] != old and ctx["guard_allowed"] and approved''')
        edit('stages/pack/verification/support_semantic_veto.py',
             '                    counts["semantic_vetoes"] += ctx["proposed"] != old and not approved',
             '                    counts["semantic_vetoes"] += ctx["proposed"] != old and ctx["guard_allowed"] and not approved')

    if 'complete_dag_gate' in features:
        edit('stages/pack/joint_selector.py',
             '''        if dag["eligible_package_count"] > 0:
            return "already_supported_dag_package"''',
             '''        plan = state.get("_evidence_plan", trace.get("plan", []))
        nodes = {n["id"] for n in plan if isinstance(n, dict) and "id" in n}
        prefix = next((p for p in dag.get("prefix_optimization", []) if p.get("top_k") == PREFIX_K), {})
        closed = set(prefix.get("coverage_after", []))
        if dag["eligible_package_count"] > 0 and nodes and nodes <= closed:
            return "already_supported_dag_package"''')

    if 'progressive_coverage' in features:
        edit('stages/pack/coverage_selector.py',
             'gain = sum(max(0.0, score - covered.get(goal, 0.0)) for goal, score in goals.items()) / goal_total',
             'gain = sum(max(0.0, min(1.0, score)) * (1.0 - covered.get(goal, 0.0)) for goal, score in goals.items()) / goal_total')
        edit('stages/pack/coverage_selector.py',
             'covered[goal] = max(covered.get(goal, 0.0), score)',
             'covered[goal] = 1.0 - (1.0 - covered.get(goal, 0.0)) * (1.0 - max(0.0, min(1.0, score)))')

    if 'goal_redundancy' in features:
        edit('stages/pack/coverage_selector.py',
             '        for position in range(min(self.config.budget, len(remaining))):',
             '''        verified_goals = {d: {p["goal"] for p in candidates[d]["verified"]} for d in remaining}
        selected_verified = set()
        for position in range(min(self.config.budget, len(remaining))):''')
        edit('stages/pack/coverage_selector.py',
             '                redundancy = max_token_redundancy(tokens, selected_tokens)',
             '''                redundancy = 0.0
                if not (verified_goals[doc_id] - selected_verified):
                    for previous, previous_tokens in zip(selected, selected_tokens):
                        previous_goals = goal_by_id[previous]
                        union = set(goals) | set(previous_goals)
                        denominator = sum(max(goals.get(g, 0.0), previous_goals.get(g, 0.0)) for g in union)
                        overlap = (sum(min(goals.get(g, 0.0), previous_goals.get(g, 0.0)) for g in union)
                                   / denominator) if denominator else 1.0
                        redundancy = max(redundancy, overlap * max_token_redundancy(tokens, [previous_tokens]))''')
        edit('stages/pack/coverage_selector.py',
             '            selected_tokens.append(document_tokens[chosen])',
             '            selected_tokens.append(document_tokens[chosen])\n            selected_verified.update(verified_goals[chosen])')

    if 'route_pool200' in features:
        edit('stages/pack/coverage_selector.py',
             '        base_weight = float(self.config.base_weight)',
             '''        # Admit only passages already present in the actual Route ranking.
        # New entries have no fabricated retrieval goal or verified proof.
        candidates = dict(candidates)
        added = []
        for d in ids[:min(200, len(ids))]:
            d = int(d)
            if d not in candidates:
                candidates[d] = {"base_score": base_by_id[d], "sources": [], "verified": []}
                added.append(d)
        state["evidence_candidates"] = candidates
        trace["route_pool_extension"] = {"added_doc_ids": added, "no_extra_searches": True}
        base_weight = float(self.config.base_weight)''')

    if 'deep_chain_review' in features:
        edit('stages/pack/proof_scope.py',
             '''    if any(len(ancestors[nid]) > 1 for nid in order):
        return "deep_plan_out_of_scope"
''', '')

    if 'dag_score_guard' in features:
        edit('stages/pack/coverage_selector.py',
             '        ordered = sorted(merged, key=lambda d: (-merged[d], d))',
             '''        trace["pointwise_relevance_for_dag"] = {str(d): float(v) for d, v in merged.items()}
        ordered = sorted(merged, key=lambda d: (-merged[d], d))''')
        edit('stages/pack/dag_package_selector.py',
             '        trace = dict(original_trace, improvement_dag_package=diagnostic)',
             '''        trace = dict(original_trace, improvement_dag_package=diagnostic)
        relevance = original_trace.get("pointwise_relevance_for_dag", {})
        prefix = next((p for p in diagnostic.get("prefix_optimization", []) if p["top_k"] == 5), {})
        plan = state.get("_evidence_plan", [])
        gain = len(prefix.get("coverage_after", [])) - len(prefix.get("coverage_before", []))
        loss = (sum(relevance.get(str(int(d)), 0.0) for d in original_ids[:5])
                - sum(relevance.get(str(int(d)), 0.0) for d in final[:5]))
        allowance = float(state.get("_pack_coverage_weight", 0.45)) * max(0, gain) / max(1, len(plan))
        rejected = loss > allowance + 1e-12
        trace["dag_relevance_guard"] = {"rejected": rejected, "loss": float(loss), "allowance": allowance}
        if rejected:
            trace["improvement_dag_package"] = dict(diagnostic, score_guard_rejected=True)
            return original_ids, original_scores, trace''')
        edit('stages/pack/stage.py',
             '        result = self.coverage.select(query, ids, scores, context, state)',
             '        state["_pack_coverage_weight"] = self.runtime.cfg.pack.coverage_weight\n        result = self.coverage.select(query, ids, scores, context, state)')

    if 'budget_contract' in features:
        edit('configuration/loading.py',
             '''    if not 1 <= config.pack.budget <= 5:
        raise ValueError("Pack prefix budget must be within 1..5")''',
             '''    if not 1 <= config.pack.budget <= 10:
        raise ValueError("Pack prefix budget must be within 1..10")
    if config.qa_top_k != config.pack.budget:
        raise ValueError("Reader qa_top_k must equal the Pack optimization budget")''')
        edit('stages/pack/stage.py',
             '        result = self.coverage.select(query, ids, scores, context, state)',
             '''        state["_pack_budget"] = self.runtime.cfg.pack.budget
        state["_pack_node_budget"] = self.runtime.cfg.plan.node_budget
        state["_pack_depth_budget"] = self.runtime.cfg.plan.depth_budget
        result = self.coverage.select(query, ids, scores, context, state)''')
        edit('stages/pack/dag_package_selector.py',
             'len(node_order) > 6 or max(depth.values(), default=0) > 4',
             'len(node_order) > int(state.get("_pack_node_budget", 6)) or max(depth.values(), default=0) > int(state.get("_pack_depth_budget", 4))')
        edit('stages/pack/dag_package_selector.py',
             '''    final = _optimize_prefix(original, original, 5, 2, choices, ancestors, node_order, leaves, packages, diagnostic)
    final = _optimize_prefix(final, original, 10, 5, choices, ancestors, node_order, leaves, packages, diagnostic)''',
             '''    budget = int(state.get("_pack_budget", 5))
    fixed = min(2, max(0, budget - 1))
    final = _optimize_prefix(original, original, budget, fixed, choices, ancestors, node_order, leaves, packages, diagnostic)
    final = _optimize_prefix(final, original, 2 * budget, budget, choices, ancestors, node_order, leaves, packages, diagnostic)''')
        edit('stages/pack/dag_package_selector.py',
             '                                    for d in final[:5]]',
             '                                    for d in final[:int(state.get("_pack_budget", 5))]]')
        edit('stages/pack/stage.py',
             '        return postprocess_support_semantic_veto(self.runtime, inputs)',
             '''        if self.runtime.cfg.pack.budget != 5:
            for solution, _ in inputs:
                solution.retrieval_trace["evidence"]["semantic_review_scope"] = {
                    "enabled": False, "reason": "only_K5_has_a_validated_semantic_review_protocol"}
            return {"semantic_review_skipped_non5_budget": len(inputs)}
        return postprocess_support_semantic_veto(self.runtime, inputs)''')

    for value in changes.values():
        ast.parse(value)
    return changes

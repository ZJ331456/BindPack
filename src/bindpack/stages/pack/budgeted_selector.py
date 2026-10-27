"""Experimental bounded set selection; no retrieval or model inference.

Closure is credited only to the existing strict citation checks. Lexical
residual coverage is a separate, fallible signal and never creates a proof.
The current no-review result is the common incumbent and ranking domain.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import combinations
import math
import re
import time
import numpy as np

from ...utils.evidence import normalized_text, branch_matches
from .dependency_closure import _ancestors
from .dag_package_selector import _collect_proofs, _package_sets

_STOP = set("a an the and or of in on at to for from by with as is are was were be been "
            "who what which where when why how do does did have has had it its their his her "
            "he she they this that these those one name named person someone something "
            "can could would will provide provides provided following question answer".split())


def residual_features(question, texts):
    """Question-derived lexical facets, with candidate-local IDF (not entailment)."""
    words = normalized_text(question).split()
    facets = {(w,) for w in words if w not in _STOP and len(w) > 2}
    facets.update((a, b) for a, b in zip(words, words[1:])
                  if a not in _STOP and b not in _STOP and len(a) > 2 and len(b) > 2)
    facets = sorted(facets)
    sentences = [[set(normalized_text(s).split()) for s in re.split(r"[.!?;\n]+", t)] for t in texts]
    matrix = np.asarray([[float(any(set(f) <= s for s in ss)) for f in facets]
                         for ss in sentences], dtype=float).reshape(len(texts), len(facets))
    if facets:
        observed = matrix.any(axis=0)
        facets = [f for f,keep in zip(facets,observed) if keep]
        matrix = matrix[:,observed]
    weights = np.log((len(texts) + 1) / (1 + matrix.sum(axis=0))) + 1 if facets else np.zeros(0)
    return facets, matrix, weights


def closed_nodes(selected, choices, ancestors, order, visible=None):
    covered = set()
    for nid in order:
        if not ancestors[nid] <= covered:
            continue
        if any(p['doc_id'] in selected and (visible is None or
               p['evidence'] in visible.get(p['doc_id'], '')) for p in choices.get(nid, [])):
            covered.add(nid)
    return covered


class BudgetedSelector:
    def __init__(self, runtime, scorer):
        self.runtime, self.scorer, self.cfg = runtime, scorer, runtime.cfg.pack

    def _visible(self, question, ordered):
        callback = getattr(self.runtime, '_joint_visible', None)
        if callback is not None:
            return callback(question, ordered)
        from transformers import AutoTokenizer
        from ...evaluation.measured_reader import prepare
        from ...utils.misc_utils import QuerySolution
        if not hasattr(self.runtime, '_joint_tokenizer'):
            self.runtime._joint_tokenizer = AutoTokenizer.from_pretrained(
                self.cfg.review_tokenizer, local_files_only=True)
        sol = QuerySolution(question=question, docs=[self.runtime._document(d) for d in ordered],
                            doc_scores=np.ones(len(ordered)))
        reading, budget = prepare(self.runtime.rag, sol, self.runtime._joint_tokenizer,
                                  self.runtime.cfg.dataset, k=self.cfg.budget,
                                  context_window=self.cfg.review_context_length,
                                  output_tokens=self.runtime.cfg.max_new_tokens)
        return dict(zip(ordered, reading.docs)), budget

    def select(self, question, ids, scores, baseline, state):
        started = time.perf_counter()
        parent, _, trace = baseline
        parent = [int(x) for x in parent]
        k = min(self.cfg.budget, len(parent))
        if k < 3:
            return baseline
        fixed_count = min(k,self.cfg.joint_fixed_prefix)
        s0, fixed = tuple(parent[:k]), frozenset(parent[:fixed_count])
        candidates = deepcopy(state.get('evidence_candidates', {}))
        domain = sorted((set(candidates) & set(parent[:200])) | set(s0), key=parent.index)
        rank = {d: i + 1 for i, d in enumerate(parent)}
        local = {d: i for i, d in enumerate(domain)}
        bindings = state.get('_evidence_winning_bindings', {})
        ev = state.get('evidence_trace', {})
        features = getattr(self.runtime, '_joint_features', None)
        if features is None:
            features = {'residual': self.cfg.selector_mode in {'residual', 'combined'},
                        'risk': self.cfg.selector_mode in {'risk', 'combined'},
                        'sources': self.cfg.selector_mode in {'sources', 'combined'}}
        if features.get('sources'):
            for proof in ev.get('source_records', []):
                d = proof.get('doc_id')
                goal = str(proof.get('goal', ''))
                nid = goal[4:] if goal.startswith('dag:') else None
                if (d in candidates and nid in bindings and
                        normalized_text(proof.get('answer', '')) == normalized_text(bindings[nid]) and
                        branch_matches(proof.get('requirements', {}), bindings)):
                    candidates[d]['verified'].append(dict(proof))
        plan = state.get('_evidence_plan', [])
        ancestors, order, error = _ancestors(plan)
        if error or len(order) > 6:
            ancestors, order, plan = {}, [], []
        nodes = {n['id']: n for n in plan}
        beams = state.get('_evidence_beams') or []
        winning = next((b for b in beams if b.get('bindings') == bindings), {'proofs': {}})
        diagnostic = {'rejected_proofs': []}
        choices = _collect_proofs(nodes, bindings, winning, candidates, rank,
                                  self.runtime._document, diagnostic, include_citations=True)
        leaves = set(order) - {p for n in plan for p in n.get('depends_on', [])}
        texts = [self.runtime._document(d) for d in domain]
        tokens = [set(normalized_text(t).split()) for t in texts]
        finite = np.nan_to_num(np.asarray(scores, float), nan=0, posinf=0, neginf=0)
        span = float(np.ptp(finite)) if len(finite) else 0
        norm = (finite - finite.min()) / span if span else np.ones(len(finite))
        base = dict(zip(map(int, ids), norm.tolist()))
        goals = {d: self.scorer.goal_scores(candidates.get(d, {'sources': [], 'verified': []}), bindings)
                 for d in domain}
        if features.get('residual'):
            facets, soft, weights = residual_features(question, texts)
            goal_names = sorted({g for values in goals.values() for g in values})
            goal_matrix = np.asarray([[goals[d].get(g,0) for g in goal_names] for d in domain],float).reshape(len(domain),len(goal_names))
        else:
            facets, soft, weights = [], np.zeros((len(domain),0)), np.zeros(0)
            goal_names, goal_matrix = [], np.zeros((len(domain),0))
        rel = np.asarray([self.cfg.base_weight * base.get(d, candidates.get(d, {}).get('base_score', 0))
                          + self.cfg.relation_weight * max(goals[d].values(), default=0) for d in domain])
        redundancy = np.zeros((len(domain), len(domain)))
        for a, b in combinations(range(len(domain)), 2):
            ga, gb = goals[domain[a]], goals[domain[b]]
            union = sorted(ga.keys() | gb.keys())
            denominator = sum(max(ga.get(g, 0), gb.get(g, 0)) for g in union)
            overlap = sum(min(ga.get(g, 0), gb.get(g, 0)) for g in union) / denominator if denominator else 1
            if overlap:
                redundancy[a, b] = redundancy[b, a] = overlap * len(tokens[a] & tokens[b]) / max(1, len(tokens[a] | tokens[b]))
        # Retention protects observed literal support, including relations not
        # certified by the stricter closure grammar. It grants no closure credit.
        retention = {}
        records = [dict(p, goal='dag:'+nid) for nid,p in winning.get('proofs',{}).items()]
        if features.get('sources'):records += ev.get('source_records', [])
        for p in records:
            nid = str(p.get('goal',''))[4:]
            d, quote = p.get('doc_id'), str(p.get('evidence',''))
            if (nid in bindings and d in local and len(quote)>=8 and quote in self.runtime._document(d)
                    and normalized_text(p.get('answer','')) == normalized_text(bindings[nid])
                    and normalized_text(p.get('answer','')) in normalized_text(quote)
                    and branch_matches(p.get('requirements',{}),bindings)):
                retention.setdefault(nid,[]).append({'doc_id':d,'evidence':quote})
        base_visible, base_budget = self._visible(question, s0)
        supported0 = {n for n in retention if any(p['doc_id'] in s0 and
                      p['evidence'] in base_visible.get(p['doc_id'], '') for p in retention[n])}
        rho = self.cfg.joint_soft_weight if features.get('residual') else 0
        # A missing proof channel carries no information. In residual variants
        # allocate its coverage weight to the observable lexical channel rather
        # than treating an unmodeled plan as evidence of completeness.
        if features.get('residual') and not choices:
            rho = 1.0
        loss_weight = self.cfg.joint_loss_weight if features.get('risk') else 0
        def evaluate(selected, visible=None):
            indexes = [local[d] for d in selected]
            closed = closed_nodes(selected, choices, ancestors, order, visible)
            closure = (len(closed) / max(1, len(order)) + len(closed & leaves) / max(1, len(leaves))) / 2
            if visible is None:
                lexical = float(np.max(soft[indexes], axis=0) @ weights / weights.sum()) if len(weights) and indexes else 0
                route_coverage = float(np.max(goal_matrix[indexes],axis=0).mean()) if goal_names and indexes else 0
            else:
                ss = [set(normalized_text(t).split()) for text in visible.values()
                      for t in re.split(r"[.!?;\n]+", text)]
                vals = np.asarray([float(any(set(f) <= s for s in ss)) for f in facets])
                lexical = float(vals @ weights / weights.sum()) if len(weights) else 0
                factors = np.asarray([min(1.,len(visible.get(d,''))/max(1,len(texts[local[d]]))) for d in selected])
                route_coverage = float(np.max(goal_matrix[indexes]*factors[:,None],axis=0).mean()) if goal_names else 0
            channels = ([lexical] if len(weights) else []) + ([route_coverage] if goal_names else [])
            coverage = sum(channels)/len(channels) if channels else 0
            red = sum(redundancy[a, b] for a, b in combinations(indexes, 2)) / max(1, k * (k - 1) / 2)
            lost = {n for n in supported0 if not any(p['doc_id'] in selected and
                    (visible is None or p['evidence'] in visible.get(p['doc_id'], '')) for p in retention[n])}
            loss = len(lost) / max(1, len(supported0))
            value = float(rel[indexes].sum() / k + self.cfg.coverage_weight * ((1-rho)*closure + rho*coverage)
                          - self.cfg.redundancy_weight*red - loss_weight*loss)
            return value, {'closed_nodes': sorted(closed), 'predicted_closure': closure,
                           'soft_coverage': coverage, 'lexical_coverage':lexical,
                           'route_goal_coverage':route_coverage,'redundancy': red, 'lost_support': sorted(lost)}
        packages = _package_sets(choices, ancestors, order)
        packages = [p & set(domain) for p in packages if p <= set(domain) and len(p | fixed) <= k]
        packages.sort(key=lambda p: (-evaluate(fixed | p)[0], len(p), tuple(sorted(rank[d] for d in p))))
        units = [frozenset([d]) for d in domain if d not in fixed] + packages[:self.cfg.joint_package_cap]
        buckets = {n: {} for n in range(fixed_count, k+1)}
        buckets[fixed_count][fixed] = evaluate(fixed)[0]
        buckets[k][frozenset(s0)] = evaluate(s0)[0]
        expanded, timed_out, limit_reason = 0, False, None
        deadline = time.perf_counter() + self.cfg.joint_search_seconds
        for size in range(fixed_count, k):
            frontier = sorted(buckets[size], key=lambda s: (-buckets[size][s], tuple(sorted(rank[d] for d in s))))[:self.cfg.joint_beam_width]
            for selected in frontier:
                for unit in units:
                    if expanded >= self.cfg.joint_state_cap or (expanded % 32 == 0 and time.perf_counter() >= deadline):
                        timed_out = True
                        limit_reason = 'state_cap' if expanded >= self.cfg.joint_state_cap else 'deadline'
                        break
                    new = selected | unit
                    if len(new) > k or len(new) <= size or new in buckets[len(new)]:
                        continue
                    buckets[len(new)][new] = evaluate(new)[0]
                    expanded += 1
                if timed_out: break
            if timed_out: break
        finalists = sorted(buckets[k], key=lambda s: (-buckets[k][s], len(s-set(s0)), tuple(sorted(rank[d] for d in s))))[:3]
        if frozenset(s0) not in finalists: finalists.append(frozenset(s0))
        measured = []
        for selected in finalists:
            ordered = tuple(d for d in parent if d in selected)
            visible, budget = (base_visible, base_budget) if ordered == s0 else self._visible(question, ordered)
            value, details = evaluate(selected, visible)
            measured.append((value, -len(selected-set(s0)), tuple(-rank[d] for d in ordered), ordered, details, budget))
        chosen = max(measured, key=lambda x: x[:3])
        base_checked = next(x for x in measured if x[3] == s0)
        if chosen[0] <= base_checked[0] + 1e-10: chosen = base_checked
        final = list(chosen[3]) + [d for d in parent if d not in chosen[3]]
        assert final[:fixed_count] == parent[:fixed_count] and set(final[:200]) == set(parent[:200])
        assert set(final) == set(parent) and len(final) == len(set(final))
        out = dict(trace)
        previous = {p['doc_id']: p for p in trace.get('selected_prefix', [])}
        out['selected_prefix'] = [dict(previous.get(d, {'doc_id': d}),
                                      selection_source='budgeted_selector') for d in chosen[3]]
        out['budgeted_selection'] = dict(enabled=True, mode=self.cfg.selector_mode, features=features,
            gold_labels_used=False, model_requests=0, reference_doc_ids=list(s0), selected_doc_ids=list(chosen[3]),
            candidate_domain=domain, eligible_proofs=choices, retention_sources=retention,
            source_records_considered=len(ev.get('source_records', [])),
            source_alternatives_enabled=bool(features.get('sources')), rejected_proofs=diagnostic['rejected_proofs'],
            objective=chosen[0], reference_objective=base_checked[0], **chosen[4],
            soft_weight_effective=rho,
            soft_coverage_scope='reachable question lexical facets and cached subgoal ranks; not semantic completeness',
            expanded_states=expanded, package_count=len(packages), search_limit_reached=timed_out,
            search_deadline_reached=limit_reason=='deadline',state_cap_reached=limit_reason=='state_cap',
            seconds=time.perf_counter()-started, reader_budget=chosen[5],
            fixed_prefix=fixed_count,top2_preserved=final[:2]==parent[:2], top200_set_preserved=True, approximate_search=True,
            entailment_guaranteed=False)
        return np.asarray(final, int), np.linspace(1, 0, len(final), endpoint=False), out

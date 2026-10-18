"""Text-routed dependency planning using canonical references and pruning.

The planner reads only the question and authorized hop hint. Its prompts and
repair policy are preserved from the evaluated full configuration.
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import json
from .validation import plan_layers, validate_plan
from .question_structure import route_question_structure
from .plan_canonicalizer import canonicalize_plan_dependencies
from .plan_validator import (relation_faithfulness_error, validate_retrieval_plan,
                             prune_derived_comparison_nodes)


class DependencyPlanner:
    """Plan atomic evidence questions under structure-specific capacity policies."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.config = runtime.cfg.plan

    def plan(self, query: str, hops: int):
        routing = route_question_structure(query)
        # Simple dependency chains keep their evaluated prompt and hop capacity;
        # parallel branches use independent node/depth budgets. Both are formal
        # question-structure policies of this planner, with no component flags.
        if routing is not None and not routing["expand"]:
            nodes, reason = self._simple_chain(query, hops)
            self.runtime._plan_diagnostics.setdefault(query, {})["routing"] = routing
            return nodes, reason
        node_budget = max(1, int(self.config.node_budget))
        depth_budget = max(1, int(self.config.depth_budget))
        depth_hint = max(1, int(hops))
        messages = [
            {"role": "system", "content": (
                "Plan evidence retrieval using ONLY the input question. Return ONLY JSON "
                "{nodes:[{id,question,depends_on,answer_type}]}. Each node asks for one "
                "atomic relation or attribute that a single passage can establish. "
                "Keep every unknown intermediate entity as a literal ${ID.answer} "
                "placeholder; list exactly those IDs in depends_on. Never guess an answer. "
                "The depth hint is the expected length of one dependency chain, NOT "
                "the total number of nodes. Independent comparison branches may each "
                "have their own complete chain. Retrieve their attributes separately. "
                "Do not add a final comparison, arithmetic, counting, or aggregation "
                "node: these operations combine retrieved facts, not passages. "
                "For 'which director of Film A and Film B was born earlier?', use four "
                "nodes: s1='Who directed Film A?', s2='When was ${s1.answer} born?', "
                "s3='Who directed Film B?', s4='When was ${s3.answer} born?'. "
                "Do not compress either branch into 'When was the director of Film A born?'. "
                "For 'Where was the writer of Work X born?', retrieve the writer first "
                "and then retrieve that person's birthplace. Preserve all nested "
                "location, time, and relation qualifiers. Use no hashes and no cycles. "
                "Use the fewest atomic nodes that preserve the question's evidence needs."
            )},
            {"role": "user", "content": (
                f"Question: {query}\nExpected dependency depth hint: {depth_hint}\n"
                f"Maximum total retrieval nodes: {node_budget}\n"
                f"Maximum dependency depth: {depth_budget}\n"
                'Syntax: {"nodes":[{"id":"s1","question":"Who wrote Work X?",'
                '"depends_on":[],"answer_type":"person"},{"id":"s2",'
                '"question":"Where was ${s1.answer} born?","depends_on":["s1"],'
                '"answer_type":"place"}]}'
            )},
        ]
        diagnostics = {"attempts": 0, "outputs": [], "validation_errors": [],
                       "depth_hint": depth_hint, "node_budget": node_budget,
                       "depth_budget": depth_budget, "capacity_policy": "atomic_parallel"}
        if routing is not None:
            diagnostics["routing"] = routing
        messages[0]["content"] += (
            " Relation faithfulness: preserve the exact requested relation and attribute. "
            "A death date/place is not a birth date/place; a tenure ending is not a "
            "birth event. Examples illustrate structure, never override the question. "
            "For 'which director of Film A and Film B died earlier?', use "
            "s1='Who directed Film A?', s2='When did ${s1.answer} die?', "
            "s3='Who directed Film B?', s4='When did ${s3.answer} die?'. "
            "Use birth questions only when the input asks about birth. Every unknown "
            "intermediate entity must appear as an existing node's literal "
            "${ID.answer}; do not replace it with an unbound definite description."
        )
        diagnostics.update(validation_mode="canonical_refs", canonicalization_changes=[],
                           relation_faithfulness="explicit_birth_death_end_swap_check")
        self.runtime._plan_diagnostics[query] = diagnostics
        diagnostics["comparison_pruning"] = []
        reason = None
        for attempt in range(2):
            payload, reason = self.runtime._infer_object(messages)
            diagnostics["attempts"] += 1
            diagnostics["outputs"].append(payload)
            if not reason:
                validated_payload = payload
                validated_payload, changes, reason = canonicalize_plan_dependencies(payload)
                diagnostics["canonicalization_changes"].append(
                    {"attempt": attempt + 1, "changes": changes, "error": reason})
                if not reason:
                    # At most one superfluous computation may exceed the
                    # retrieval-node capacity; real retrieval nodes still must
                    # pass the original capacity/depth check after pruning.
                    structural, reason = validate_plan(validated_payload, node_budget + 1)
                    if not reason:
                        structural, removed = prune_derived_comparison_nodes(structural)
                        diagnostics["comparison_pruning"].append(
                            {"attempt": attempt + 1, "removed": removed})
                        validated_payload = dict(validated_payload, nodes=structural)
                if not reason:
                    nodes, reason = validate_retrieval_plan(validated_payload, node_budget, depth_budget)
                if not reason:
                    reason = relation_faithfulness_error(query, nodes)
                if not reason:
                    diagnostics["actual_depth"] = len(plan_layers(nodes))
                    return nodes, None
            diagnostics["validation_errors"].append(reason)
            if attempt == 0:
                messages.extend([
                    {"role": "assistant", "content": json.dumps(payload or {}, ensure_ascii=False)},
                    {"role": "user", "content": (
                        f"Validation failed: {reason}. Correct the JSON with at most "
                        f"{node_budget} atomic retrieval nodes and depth at most {depth_budget}. "
                        "Every dependency must appear as ${ID.answer}. Preserve separate "
                        "parallel branches; omit derived comparison operations. Do not "
                        "guess unknown intermediate entities. Return ONLY corrected JSON."
                    )},
                ])
                messages[-1]["content"] += (
                    " Keep the input question's requested relation and attribute exactly: "
                    "death, birth and tenure end are different facts. If a node depends "
                    "on an earlier answer, write its existing ${ID.answer} explicitly "
                    "in that node's question. Do not remove a needed dependency or "
                    "guess the intermediate entity to bypass validation."
                )
        return [], reason

    def _simple_chain(self, query: str, hops: int) -> Tuple[List[dict], Optional[str]]:
        max_nodes = max(1, min(4, int(hops)))
        messages = [
            {"role": "system", "content": (
                "Plan evidence retrieval for a question. Output ONLY JSON with nodes. "
                "Each node is {id, question, depends_on, answer_type}. "
                "Each question must be answerable from one passage. Unknown answers must "
                "stay as placeholders ${s1.answer}, never guess an intermediate person, "
                "place or answer. List every referenced node in depends_on. Use no other "
                "references. Independent comparison branches should be independent nodes. "
                "Preserve every nested location/relation qualifier in the question. "
                "Resolve the innermost unknown entity before asking the final relationship. "
                "For example, 'When was the person that A was compared with hired by B?' "
                "starts with 'Who was A compared with?', followed by 'When was ${s1.answer} hired by B?'. "
                "Do not carry 'hired by B' into the first subquestion. "
                "For a region north of the region containing X, first locate X, then ask "
                "for the region north of ${s1.answer}; do not replace this with north of X. "
                "Every depends_on ID must appear literally as ${ID.answer} in that node's question. "
                "No cycles. A node's answer_type describes what its passage must establish." )},
            {"role": "user", "content": (
                f"Question: {query}\nMaximum nodes: {max_nodes}\n"
                'Example syntax: {"nodes":[{"id":"s1","question":"Who wrote Work X?",'
                '"depends_on":[],"answer_type":"person"},{"id":"s2",'
                '"question":"Where was ${s1.answer} born?","depends_on":["s1"],'
                '"answer_type":"place"}]}\nUse only the input question to construct the plan.' )},
        ]
        diagnostics = {"attempts": 0, "outputs": [], "validation_errors": []}
        self.runtime._plan_diagnostics[query] = diagnostics
        for attempt in range(2):
            payload, reason = self.runtime._infer_object(messages)
            diagnostics["attempts"] += 1
            diagnostics["outputs"].append(payload)
            nodes, reason = ([], reason) if reason else validate_plan(payload, max_nodes)
            if not reason:
                nodes, removed = prune_derived_comparison_nodes(nodes)
                diagnostics["comparison_pruning"] = [{"attempt": 1, "removed": removed}]
                return nodes, None
            diagnostics["validation_errors"].append(reason)
            if attempt == 0:
                messages = messages + [
                    {"role": "assistant", "content": json.dumps(payload or {}, ensure_ascii=False)},
                    {"role": "user", "content": (
                        f"The plan failed validation: {reason}. Correct the JSON plan, using at most {max_nodes} nodes. "
                        "Every declared dependency must occur as a literal ${ID.answer} placeholder. "
                        "Keep the original question's nested qualifiers, keep unknown entities unbound, "
                        "and return only the corrected JSON. Do not add guessed answers.")},
                ]
        return [], reason

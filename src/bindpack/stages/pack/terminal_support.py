"""Subject, answer type and relation guards shared by DAG proof validation.

Unknown relations need explicit owning subjects and matched literal cues.
No standalone terminal reranker or extra model request is included.
"""

from __future__ import annotations

import re

from ..plan.answer_validation import _attribute_reason, _same_entity, _typed_answer_reason, supported_entity_surface
from ..plan.relation_guard import relation_spec, strict_relation_reason
from ...utils.evidence import normalized_text


def _plain_entity(value):
    return re.sub(r"^(?:the|a|an)\s+", "", str(value).strip(), flags=re.I).rstrip("?. ")


def _literal(value, text):
    value = normalized_text(value)
    return bool(value) and " " + value + " " in " " + normalized_text(text) + " "


def _same_subject(left, right, document):
    return _same_entity(_plain_entity(left), _plain_entity(right), document)


def _subject_surface(entity, text, document):
    return supported_entity_surface(_plain_entity(entity), text, document)


def _question_entities(question):
    """Literal proper-name anchors; category descriptions are not new aliases."""
    names = re.findall(r"\b[A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*)*", question)
    ignored = {"Who", "What", "Which", "Where", "When", "How", "Is", "Was", "The", "A", "At"}
    return [n.rstrip("?.") for n in names if n.rstrip("?.") not in ignored]


def _relation_check(node, question, proof, document, bindings):
    """Return a check name, or a rejection reason with no semantic upgrade."""
    answer, quote = str(proof["answer"]), str(proof["evidence"])
    answer_type = node.get("answer_type", "")
    typed_reason = _typed_answer_reason(answer, answer_type)
    if typed_reason:
        return None, typed_reason
    # A present-but-wrong date is different from an absent date. Keep this
    # bounded type test for the common planner label "time" as well.
    if normalized_text(answer_type) in {"time", "year", "date"} and re.search(
            r"\b(?:born|birth|died|death|year|date)\b", question, re.I) and not re.search(r"\d", answer):
        return None, "answer_type_mismatch"
    checked_question = re.sub(r"\s+on\s+the\s+world\s+map\s*[?.]*$", "?", question, flags=re.I)
    spec = relation_spec(checked_question)
    if spec["relation"] == "comparison":
        return None, "comparison_requires_atomic_evidence"
    deps = node.get("depends_on", [])
    entity = spec.get("subject")
    if entity:
        entity = _plain_entity(entity)
    elif len(deps) == 1 and deps[0] in bindings:
        entity = _plain_entity(bindings[deps[0]])
    else:
        names = _question_entities(question)
        grounded = [n for n in names if _subject_surface(n, quote, document)[0]]
        entity = grounded[0] if len(grounded) == 1 else None
    if not entity:
        return None, "unresolved_quote_subject"
    subject = _subject_surface(entity, quote, document)[0]
    # A document topic licenses pronouns only inside that same document. The
    # existing personal attribute guard performs the actual pronoun check.
    if not subject and spec["relation"] not in {"nationality", "birth_date", "birth_place", "death_date"}:
        return None, "relationship_subject_not_in_quote"
    if spec["relation"] in {"nationality", "birth_date", "birth_place", "death_date"}:
        reason = _attribute_reason(spec["relation"], answer, entity, quote, quote, document)
        return ("local_personal_attribute", None) if reason is None else (None, reason)
    if spec["relation"] != "other":
        reason = strict_relation_reason(spec, question, answer, answer_type, entity, quote,
                                        document, _subject_surface, _same_subject)
        if reason is None:
            return "local_relation_construction", None
        if reason in {"relationship_subject_mismatch", "linked_subject_relationship_not_supported",
                      "answer_type_mismatch", "relationship_direction_conflict"}:
            return None, reason
        # Passive creator leads containing abbreviated titles (e.g. O.K.) can
        # be split by generic sentence heuristics. Match the complete quoted
        # owner before the passive marker, never an unrelated co-occurrence.
        markers = {"director": r"directed\s+by", "author": r"(?:written|authored)\s+by",
                   "composer": r"composed\s+by"}
        if spec["relation"] in markers:
            marker = re.search(markers[spec["relation"]], quote, re.I)
            if marker and _literal(entity, quote[:marker.start()]) and _literal(answer, quote[marker.end():]):
                owner = re.search(re.escape(entity), quote[:marker.start()], re.I)
                prefix = quote[:owner.start()].strip() if owner else ""
                if owner and not prefix and not re.search(r"\b(?:not|never)\b", quote, re.I):
                    return "local_passive_creator", None
        if spec["relation"] == "location" and subject:
            match = re.search(re.escape(subject) + r"\s+(?:is|was|lies)\s+(?:at|in|on)\b", quote, re.I)
            # The answer may include the owning preposition ("at the center"),
            # so retain it in the checked predicate rather than trimming it.
            if match and _literal(answer, quote[match.start() + len(subject):]) and not re.search(r"\b(?:not|never)\b", quote, re.I):
                return "local_location", None
        return None, reason
    # Unknown natural-language relations are not trusted on confidence alone.
    # Admit only explicit matched bounded cue families and one owning clause.
    families = [
        (r"\b(?:live|lives|reside|resides|home)\b", r"\b(?:live[sd]?|resides?|home)\b"),
        (r"\b(?:invented|invent|inventor)\b", r"\b(?:invented|inventor)\b"),
        (r"\b(?:married|spouse|wife|husband)\b", r"\b(?:married|spouse|wife|husband)\b"),
        (r"\b(?:played|portrayed|play|portray)\b", r"\b(?:played|portrayed)\b"),
        (r"\b(?:founded|established|created|built|opened)\b", r"\b(?:founded|established|created|built|opened)\b"),
    ]
    if re.search(r"\b(?:first|earlier|later|older|younger|same|longest|highest|largest)\b", question, re.I):
        return None, "non_atomic_or_comparative_unknown_relation"
    for query_cue, passage_cue in families:
        if not re.search(query_cue, question, re.I):
            continue
        for clause in re.split(r"(?<=[!?;])\s+|\n+|(?<=\.)\s+(?=[A-Z])", quote):
            if (len(clause) <= 450 and _subject_surface(entity, clause, document)[0]
                    and _literal(answer, clause) and re.search(passage_cue, clause, re.I)
                    and not re.search(r"\b(?:not|never|without)\b", clause, re.I)):
                surface = _subject_surface(entity, clause, document)[0]
                mention = re.search(re.escape(surface), clause, re.I)
                marker = re.search(passage_cue, clause, re.I)
                # Obvious relative ownership must not move a spouse's/father's
                # fact to the named entity merely because the names co-occur.
                if mention and (re.match(r"['’]s\b", clause[mention.end():]) or
                                re.search(r"\b(?:father|mother|brother|sister|son|daughter|wife|husband)\s+of\s*$",
                                          clause[:mention.start()], re.I)):
                    continue
                if mention and marker and mention.end() < marker.start() and re.search(
                        r"\b(?:father|mother|brother|sister|son|daughter)\b", clause[mention.end():marker.start()], re.I):
                    continue
                # All declared dependencies must occur in the quoted clause;
                # a multi-parent join cannot inherit an unrelated root's fact.
                if all(dep in bindings and _subject_surface(bindings[dep], clause, document)[0] for dep in deps):
                    return "matched_literal_relation_cue", None
        return None, "relation_cue_not_owned_by_quoted_subject"
    return None, "unknown_relation_not_promoted"

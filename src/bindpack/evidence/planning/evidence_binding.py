"""Source identity and personal attribute checks shared by closed DAG proofs.

These bounded local checks establish quotation and subject support; they do
not certify arbitrary semantic entailment or consult benchmark answers.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from ..evidence_retrieval import normalized_text


def _contains(phrase: str, text: str) -> bool:
    phrase, text = normalized_text(phrase), normalized_text(text)
    return bool(phrase) and f" {phrase} " in f" {text} "


def _name_tokens(name: str) -> List[str]:
    return re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:[-'][A-Za-zÀ-ÖØ-öø-ÿ]+)*", name)


def _name_extension(short: str, full: str) -> bool:
    a, b = _name_tokens(short), _name_tokens(full)
    return (2 <= len(a) <= len(b) <= 6 and a[0].casefold() == b[0].casefold()
            and a[-1].casefold() == b[-1].casefold()
            and all(token[:1].isupper() for token in b)
            and all(x.casefold() in [y.casefold() for y in b] for x in a))


def _identity_names(entity: str, document: str) -> List[str]:
    """Only the title and introductory identity may license shortened names."""
    lines = document.splitlines()
    title = lines[0].strip() if len(lines) > 1 else ""
    body = " ".join(lines[1:]) if title else document
    names: List[str] = []
    if normalized_text(title) == normalized_text(entity):
        names.append(entity)
    # Wikipedia leads commonly start with a full name, possibly a middle name.
    match = re.match(r"\s*([A-ZÀ-ÖØ-Þ][\wÀ-ÖØ-öø-ÿ'-]*(?:\s+[A-ZÀ-ÖØ-Þ][\wÀ-ÖØ-öø-ÿ'-]*){1,5})(?=\s*\(|\s+(?:is|was)\b)", body)
    if match:
        full = match.group(1)
        if normalized_text(full) == normalized_text(entity) or _name_extension(entity, full):
            names.append(full)
    # Explicit alternative spellings/names are evidence, unlike fuzzy matching.
    lead = body[:700]
    if _contains(entity, lead):
        for alias in re.findall(r"(?:also known as|also called|known professionally as|born as)\s+([A-Z][\w'-]*(?:\s+[A-Z][\w'-]*){0,5})", lead):
            names.append(alias)
    return list(dict.fromkeys(names))


def supported_entity_surface(entity: str, quote: str, document: str) -> Tuple[Optional[str], Optional[str]]:
    """Return a literal surface justified by this document's own identity."""
    if _contains(entity, quote):
        return entity, "literal"
    identities = _identity_names(entity, document)
    for name in identities:
        if _contains(name, quote):
            return name, "introductory_full_name"
    if not identities:
        return None, None
    tokens = _name_tokens(entity)
    if len(tokens) < 2:
        return None, None
    surname = tokens[-1]
    for occurrence in re.finditer(r"(?<!\w)" + re.escape(surname) + r"(?!\w)", quote, re.I):
        # A different full name sharing a surname never licenses the binding.
        prefix = quote[:occurrence.start()]
        previous = re.search(r"([A-Z][\w'-]*)\s+$", prefix)
        if previous and previous.group(1) not in {"Mr", "Mrs", "Ms", "Dr", "Sir", "President", "Director"}:
            continue
        return occurrence.group(0), "document_anchored_surname"
    return None, None


def _typed_answer_reason(answer: str, answer_type: str) -> Optional[str]:
    kind = normalized_text(answer_type)
    if kind in {"year", "birth year", "death year"} and not re.search(r"\b\d{1,4}\b", answer):
        return "answer_type_mismatch"
    if kind in {"date", "birth date", "death date"} and not re.search(r"\d|\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b", answer, re.I):
        return "answer_type_mismatch"
    return None


def _same_entity(left: str, right: str, document: str) -> bool:
    if normalized_text(left) == normalized_text(right):
        return True
    # Mere co-occurrence of two names in one identity quote is not equivalence.
    return (any(normalized_text(right) == normalized_text(name) for name in _identity_names(left, document))
            or any(normalized_text(left) == normalized_text(name) for name in _identity_names(right, document)))


def _is_life_date(value: str) -> bool:
    """Recognize dates, not arbitrary parenthesized numeric intervals."""
    tokens = normalized_text(value).split()
    months = {"january", "february", "march", "april", "may", "june", "july", "august",
              "september", "october", "november", "december", "jan", "feb", "mar", "apr",
              "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "bc", "bce", "ad", "ce"}
    if not tokens or any(not token.isdigit() and token not in months for token in tokens):
        return False
    return any(token.isdigit() and len(token) == 4 for token in tokens) or (
        bool({"bc", "bce", "ad", "ce"} & set(tokens)) and any(token.isdigit() for token in tokens))


def _supported_life_range(relation: str, answer: str, entity: str, span: str,
                          document: str) -> bool:
    if relation not in {"birth_date", "death_date"}:
        return False
    for match in re.finditer(r"\(([^()\n]{3,90}?)\s*(?:–|—|\s-\s)\s*([^()\n]{3,90}?)\)", span):
        left, right = match.group(1).strip(), match.group(2).strip()
        if not _is_life_date(left) or not _is_life_date(right):
            continue
        prefix = re.split(r"[.;!?]\s*", span[:match.start()])[-1]
        if not supported_entity_surface(entity, prefix, document)[0]:
            continue
        if re.search(r"\b(?:father|mother|brother|sister|son|daughter|wife|husband)\b", prefix, re.I):
            continue
        value = left if relation == "birth_date" else right
        if _contains(answer, value):
            return True
    return False


def _attribute_reason(relation: str, answer: str, entity: str, span: str,
                      support: str, document: str) -> Optional[str]:
    if relation not in {"nationality", "birth_date", "birth_place", "death_date"}:
        return None
    surface, _ = supported_entity_surface(entity, span, document)
    anchored_support, _ = supported_entity_surface(entity, support, document)
    # Co-reference requires an explicit topic identity in this same document.
    pronoun = bool(_identity_names(entity, document)) and bool(anchored_support) and bool(
        re.search(r"(?:^|[.!?]\s+)(?:He|She|They|His|Her)\b", span))
    if not surface and not pronoun:
        return "relationship_subject_not_supported"
    if relation == "nationality":
        # An Italian film directed by X establishes the film's origin, not X's.
        escaped = re.escape(answer)
        unrelated = re.search(escaped + r"\s+(?:film|movie|novel|book|television\s+series|language|production)\b", span, re.I)
        if unrelated and not re.match(r"\s+(?:director|maker|producer|composer|writer|actor|actress)\b", span[unrelated.end():], re.I):
            return "answer_attribute_belongs_to_other_entity"
        if re.search(r"\b(?:born|birth)\b", span, re.I) and not re.search(
                r"\b(?:nationality|citizen|citizenship|national)\b|" + escaped +
                r"\s+(?:director|writer|author|actor|actress|composer|politician|scientist|singer|musician|filmmaker|screenwriter)\b", span, re.I):
            return "birthplace_does_not_establish_nationality"
        if not re.search(r"\b(?:nationality|citizen|citizenship|national|director|writer|author|actor|actress|composer|politician|person|scientist|singer|musician|filmmaker|screenwriter|is|was)\b", span, re.I):
            return "requested_relationship_not_in_quote"
        own_clause = False
        for clause in re.split(r"[.;!?]\s*", span):
            if _contains(answer, clause) and (supported_entity_surface(entity, clause, document)[0]
                                             or (pronoun and re.match(r"(?:He|She|They|His|Her)\b", clause.strip()))):
                own_clause = True
                break
        if not own_clause:
            return "answer_attribute_belongs_to_other_entity"
    else:
        marker = r"\b(?:born|birth)\b" if relation.startswith("birth") else r"\b(?:died|death)\b"
        matches = list(re.finditer(marker, span, re.I))
        if not matches:
            if _supported_life_range(relation, answer, entity, span, document):
                return None
            return "requested_relationship_not_in_quote"
        for match in matches:
            end = re.search(r"[);.!?]|\b(?:died|death|became|appointed|elected|served)\b", span[match.end():], re.I)
            suffix = span[match.end():match.end() + end.start()] if end else span[match.end():]
            before = span[:match.start()]
            head = re.split(r"[.;!?]\s*", before)[-1]
            if not supported_entity_surface(entity, head, document)[0] and not (
                    pronoun and re.match(r"(?:He|She|They|His|Her)\b", head.strip())):
                continue
            # Possessive relative statements must not move a parent's date to X.
            if re.search(r"\b(?:father|mother|brother|sister|son|daughter|wife|husband)\b[^.;!?]*$", before, re.I):
                continue
            if _contains(answer, suffix):
                return None
        return "answer_not_supported_for_requested_relationship"
    return None

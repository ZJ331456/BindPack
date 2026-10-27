"""Known server-side stops; never synthesize facts, IDs, quotes or decisions."""
import json


STOP_SEQUENCES = {
    "proposer": ['"reason":', '"candidate_quote": ""', '"root_quote": ""'],
    "reviewer": ['"victim_reason":', '"same_entity_event": false',
                 '"old_missing_fact": ""', '"candidate_supplied_fact": ""'],
}


def normalize_prefix(text, stage):
    """Close only a JSON object at a known omitted suffix boundary.

    Incomplete mandatory fields remain absent and are rejected by native
    validators. Unparseable text is returned unchanged for normal abstention.
    """
    try:
        json.loads(text)
        return text, False
    except (ValueError, TypeError):
        if not isinstance(text, str):
            return text, False
    prefix = text.rstrip()
    if prefix.endswith(','):
        prefix = prefix[:-1].rstrip()
    try:
        value = json.loads(prefix + ('}}' if stage == 'proposer' else '}'))
    except (ValueError, TypeError):
        return text, False
    if not isinstance(value, dict):
        return text, False
    return json.dumps(value, ensure_ascii=False), True

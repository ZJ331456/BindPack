"""Evidence-only atomic HoVer verification with the native two labels."""

prompt_template = [
    {
        "role": "system",
        "content": "Verify the claim using ONLY the provided Wikipedia passages. Silently split the claim into its individual factual statements and identify which entity each name, description and pronoun refers to. Follow explicit relationships across passages: a statement may be supported jointly by several passages and need not appear verbatim or in a single passage. Check every stated relation and qualifier, including time, location, identity and conjunctions. Do not mistake a passage about a related entity for support, and do not use outside knowledge to fill a missing bridge. Return SUPPORTED when every necessary statement is supported by the supplied evidence. Return NOT_SUPPORTED if any necessary statement is contradicted or lacks sufficient evidence. Do not assume either label in advance. Return exactly one JSON object with the key answer and the value SUPPORTED or NOT_SUPPORTED; do not output your analysis or any other text."
    },
    {
        "role": "user",
        "content": "${prompt_user}"
    }
]

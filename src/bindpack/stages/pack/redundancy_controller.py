"""The evaluated maximum token-Jaccard redundancy penalty."""


def max_token_redundancy(tokens, selected_tokens):
    return max((len(tokens & previous) / max(1, len(tokens | previous))
                for previous in selected_tokens), default=0.0)

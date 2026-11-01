"""Token-based API estimates; prices are supplied by the experiment owner."""
from decimal import Decimal


DEFAULT_PRICES = dict(input_cny_per_million=.8, output_cny_per_million=2.7,
    cached_input_cny_per_million=.1, cache_creation_cny_per_million=1.25)


def cost_for_usage(input_tokens, output_tokens, cached_tokens=0, created_tokens=0, prices=None):
    prices=prices or DEFAULT_PRICES
    if not all(type(n) is int and n>=0 for n in (input_tokens,output_tokens,cached_tokens,created_tokens)):
        raise ValueError('Token counts must be nonnegative integers')
    if cached_tokens+created_tokens>input_tokens:
        raise ValueError('Cache tokens exceed total input tokens')
    counts={'uncached_input_tokens':input_tokens-cached_tokens-created_tokens,
            'cached_input_tokens':cached_tokens,'cache_creation_input_tokens':created_tokens,
            'output_tokens':output_tokens}
    terms=[(counts['uncached_input_tokens'],'input_cny_per_million'),
           (cached_tokens,'cached_input_cny_per_million'),(created_tokens,'cache_creation_cny_per_million'),
           (output_tokens,'output_cny_per_million')]
    cost=sum(Decimal(n)*Decimal(str(prices[key]))/Decimal(1000000) for n,key in terms)
    standard_cost=(Decimal(input_tokens)*Decimal(str(prices['input_cny_per_million']))+
                   Decimal(output_tokens)*Decimal(str(prices['output_cny_per_million'])))/Decimal(1000000)
    return dict(counts,estimated_cny=float(cost),all_input_at_standard_rate_cny=float(standard_cost))


def summarize_calls(events,prices=None):
    prices=prices or DEFAULT_PRICES
    totals=dict(input_tokens=0,output_tokens=0,cached_tokens=0,created_tokens=0)
    failed=unknown=cache_unknown=0
    for event in events:
        failed+=event.get('status')!='success'
        prompt=event.get('prompt_tokens');completion=event.get('completion_tokens')
        if prompt is None or completion is None:
            unknown+=1;continue
        raw=event.get('usage_details') or {}
        details=raw.get('prompt_tokens_details') or {}
        cache=details.get('cached_tokens',raw.get('prompt_cache_hit_tokens'))
        if cache is None:cache_unknown+=1;cache=0
        created=details.get('cache_creation_input_tokens',0) or 0
        totals['input_tokens']+=prompt;totals['output_tokens']+=completion
        totals['cached_tokens']+=cache;totals['created_tokens']+=created
    costs=cost_for_usage(**totals,prices=prices)
    return dict(physical_requests=len(events),failed_attempts=failed,unknown_usage_attempts=unknown,
        cache_usage_unreported_attempts=cache_unknown,
        **{**totals,**costs},
        cost_scope='actual physical API usage, including known failed-attempt usage; application cache hits incur no new API charge',
        prices_cny_per_million=prices,prices_source='user-supplied prices; provider bill not independently verified')

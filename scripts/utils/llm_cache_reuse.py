"""Copy an explicitly authorized model response cache without changing its keys."""
from pathlib import Path
import sqlite3


def seed_response_cache(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not source.is_file() or source == destination:
        raise ValueError('A separate existing response cache is required')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        with sqlite3.connect(destination) as target:
            count=target.execute('SELECT count(*) FROM cache').fetchone()[0]
        return dict(source=str(source), destination=str(destination), seeded=False, rows=count)
    temporary=destination.with_name(destination.name+'.seed.tmp')
    with sqlite3.connect(f'file:{source}?mode=ro',uri=True) as original, sqlite3.connect(temporary) as target:
        original.backup(target)
        if target.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('Invalid response cache')
        count=target.execute('SELECT count(*) FROM cache').fetchone()[0]
    temporary.replace(destination)
    return dict(source=str(source),destination=str(destination),seeded=True,rows=count,
        eligibility='Existing v2 keys bind exact messages, effective generation parameters and base_url; no relabeling, prompt-only match, or cross-model reuse')

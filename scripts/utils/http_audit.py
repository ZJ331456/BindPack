"""Locate the request log of an existing local vLLM service."""
from pathlib import Path
import stat
from urllib.parse import urlsplit


def locate(base_url):
    url = urlsplit(base_url)
    if url.hostname not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('Provide VLLM_LOG explicitly for a remote endpoint')
    port = str(url.port or (443 if url.scheme == 'https' else 80))
    matches = []
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            args = (process / 'cmdline').read_bytes().decode('utf-8', 'replace').split('\0')
            if 'vllm.entrypoints.openai.api_server' not in args or '--port' not in args:
                continue
            if args[args.index('--port') + 1] != port:
                continue
            output = process / 'fd/1'
            if stat.S_ISREG(output.stat().st_mode):
                # The descriptor remains readable even when its log was deleted.
                matches.append(str(output))
        except (OSError, ValueError, IndexError):
            continue
    if len(matches) != 1:
        raise ValueError(f'Expected one readable local service log, found {len(matches)}; provide VLLM_LOG')
    return matches[0]

#!/usr/bin/env bash
# Start only the configured Qwen3 service; keep an existing service running.
set -euo pipefail
exec /root/anaconda3/bin/python - <<'PY'
import json
from pathlib import Path
import subprocess
import urllib.error
import urllib.request

root = Path('/root/baseline/scripts/outputs/services')
root.mkdir(parents=True, exist_ok=True)
log = root / 'qwen3_reader_055.log'
try:
    with urllib.request.urlopen('http://127.0.0.1:8035/v1/models', timeout=3) as response:
        models = json.load(response)['data']
except urllib.error.URLError:
    models = None
if models is not None:
    if not any(m.get('id') == 'qwen3-8b' and m.get('root') == '/root/models/Qwen3-8B' for m in models):
        raise SystemExit('Port 8035 serves a different model; existing processes were left untouched.')
    print('Qwen3-8B service is already running; kept the current process.')
    print(f'tail -f {log}')
else:
    with log.open('a') as handle:
        proc = subprocess.Popen(['bash', '/root/BindPack/scripts/start_vllm_qwen3.sh'],
            stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True)
    (root / 'qwen3_reader.pid').write_text(str(proc.pid) + '\n')
    print(f'Started Qwen3-8B, PID={proc.pid}; startup status: tail -f {log}')
PY

"""Independent, atomic copies of verified graph/vector index assets."""
import fcntl
import hashlib
import json
from pathlib import Path
import shutil


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def copy_file(source, target, *, reserve_bytes=0):
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name('.' + target.name + '.copy.tmp')
    try:
        try:
            with source.open('rb') as src, tmp.open('wb') as dst:
                fcntl.ioctl(dst.fileno(), 0x40049409, src.fileno())
            shutil.copystat(source, tmp)
        except OSError:
            if shutil.disk_usage(target.parent).free < source.stat().st_size + reserve_bytes:
                raise OSError('Insufficient free space for a physical copy and the result/checkpoint reserve')
            shutil.copy2(source, tmp)
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)

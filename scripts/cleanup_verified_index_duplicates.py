#!/usr/bin/env python3
"""Release only old graph/vector/OpenIE duplicates proven by the canonical archive."""
import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

PROJECT=Path('/root/BindPack')
OUTPUTS=PROJECT/'outputs'
ARCHIVE=OUTPUTS/'4_datasets_result'
OLD_ROOTS=(OUTPUTS/'3multi_hop_datasets_results_10_5',OUTPUTS/'hover_full_measured_20261024')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def allowed(path):
    return path.name=='graph.pickle' or path.suffix=='.parquet' or path.name.startswith('openie_results_ner_') and path.suffix=='.json'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--apply',action='store_true')
    p.add_argument('--receipt',type=Path,default=OUTPUTS/'cleanup_verified_duplicates_20261026.json')
    a=p.parse_args();canonical=defaultdict(list)
    manifest=ARCHIVE/'manifest.json';inventory=json.loads(manifest.read_text());original_manifest=sha(manifest)
    for entry in inventory['files']:
        path=ARCHIVE/entry['path']
        if entry['path'].startswith('shared_indexes/') and allowed(path):
            if not path.is_file() or path.is_symlink() or sha(path)!=entry['sha256']:
                raise ValueError('Invalid canonical index asset: '+str(path))
            canonical[(path.name,path.stat().st_size)].append((path,entry['sha256']))
    active_files=set()
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():continue
        try:
            for fd in (process/'fd').iterdir():
                try:active_files.add(fd.resolve())
                except OSError:pass
        except (FileNotFoundError,PermissionError):pass
    hash_cache={};deletions=[];inodes=defaultdict(list)
    for old in OLD_ROOTS:
        for path in old.rglob('*'):
            if path.is_symlink() or not path.is_file() or not allowed(path):continue
            if 'index' not in path.parts and 'shared_indexes' not in path.parts:continue
            if path.resolve() in active_files:continue
            stat=path.stat();options=canonical.get((path.name,stat.st_size),[])
            if not options:continue
            key=(stat.st_dev,stat.st_ino)
            digest=hash_cache.setdefault(key,None)
            if digest is None:digest=hash_cache[key]=sha(path)
            source=next((p for p,h in options if h==digest),None)
            if source is None:continue
            row=dict(path=str(path),canonical=str(source),bytes=stat.st_size,sha256=digest,
                     device=stat.st_dev,inode=stat.st_ino,links=stat.st_nlink,allocated_bytes=stat.st_blocks*512)
            deletions.append(row);inodes[key].append(row)
    plan=dict(created_at=datetime.now().astimezone().isoformat(),applied=False,
        policy='Only exact SHA256 duplicate graph/vector/OpenIE files in two old completed experiment trees. Preserve canonical four indexes, results, ledgers, checkpoints, source snapshots and all unique assets.',
        canonical_manifest_sha256=original_manifest,files=len(deletions),
        logical_bytes=sum(d['bytes'] for d in deletions),
        estimated_unlinked_inode_bytes=sum(rows[0]['allocated_bytes'] for rows in inodes.values() if len(rows)==rows[0]['links']),
        restoration='Copy each canonical path back to path from this receipt; no index build or LLM needed.',
        before_free_bytes=shutil.disk_usage(OUTPUTS).free,entries=deletions)
    write(a.receipt,plan)
    print(json.dumps({k:v for k,v in plan.items() if k!='entries'},ensure_ascii=False),flush=True)
    if not a.apply:return
    for row in deletions:
        path=Path(row['path']);stat=path.stat()
        if (stat.st_dev,stat.st_ino,stat.st_size)!=(row['device'],row['inode'],row['bytes']):
            raise ValueError('File changed after verification: '+str(path))
        path.unlink()
    if sha(manifest)!=original_manifest:raise ValueError('Canonical archive inventory changed')
    for options in canonical.values():
        for path,digest in options:
            if sha(path)!=digest:raise ValueError('Canonical asset changed')
    plan.update(applied=True,after_free_bytes=shutil.disk_usage(OUTPUTS).free)
    plan['actual_free_increase_bytes']=plan['after_free_bytes']-plan['before_free_bytes']
    write(a.receipt,plan)
    print(json.dumps({k:v for k,v in plan.items() if k!='entries'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()

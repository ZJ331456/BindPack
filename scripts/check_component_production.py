#!/usr/bin/env python3
"""Verify promoted native code against all actual final-candidate predictions."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();root=args.output
    sys.path.insert(0,'/root/BindPack/src')
    path=root/'source_snapshot/scripts/run_pack_weight_sweep.py'
    spec=importlib.util.spec_from_file_location('production_parity_helpers',path)
    sweep=importlib.util.module_from_spec(spec);spec.loader.exec_module(sweep)
    from transformers import AutoTokenizer
    from bindpack import BindPackConfig
    from bindpack.evaluation.measurement import Recorder
    selected=json.loads((root/'final_decision.json').read_text());arm=selected['final_arm'];total=0
    tokenizer=AutoTokenizer.from_pretrained('/root/models/Qwen3-8B',local_files_only=True)
    helper=sweep.helpers(Path('/root/BindPack/outputs/pack_weight_diagnostic_20261025'))
    for ds in sweep.DATASETS:
        cfg=BindPackConfig.from_file(root/'metadata'/ds/'method.json',save_dir=str(root/'runtime'/ds/'index'),stage_ablation='full')
        passage=next((Path('/root/BindPack/outputs/4_datasets_result/shared_indexes')/ds).rglob('vdb_chunk.parquet'))
        store=sweep.ReadOnlyPassages(passage);recorder=Recorder(root/'production_parity'/ds)
        rag=sweep.make_runtime(cfg,store,recorder,tokenizer)
        transport=rag.llm_model.openai_client.chat.completions;real=transport.create
        def deny(*args,**kwargs):raise AssertionError('Promoted code produced a new prompt/cache miss')
        transport.create=deny
        rows=json.loads((root/'cases'/ds/arm/'result.json').read_text())['results']
        try:
            for row in rows:
                key=__import__('hashlib').sha256(str(row['sample_id']).encode()).hexdigest()+'.json'
                payload=json.loads((root/'frozen_inputs'/ds/arm/key).read_text())['input']
                solution,reader,timing=sweep.run_arm(rag,tokenizer,recorder,payload,.6,row['sample_id'],helper)
                assert solution.docs==row['candidate_docs'],(ds,row['sample_id'],'ranking')
                assert solution.answer==row['answer'],(ds,row['sample_id'],'answer')
                assert reader['budget']['prompt_sha256']==row['reader']['budget']['prompt_sha256']
                total+=1
        finally:
            transport.create=real
            for obj,name,original in reversed(recorder.restorations):setattr(obj,name,original)
    assert total==200
    (root/'production_parity.ok').write_text(json.dumps(dict(checked=total,new_requests=0,status='passed'))+'\n')
    print('PRODUCTION PARITY PASS: 200 actual rankings, prompts and answers, zero HTTP requests',flush=True)


if __name__=='__main__':main()

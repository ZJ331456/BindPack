#!/usr/bin/env python3
"""Compatibility only: current native Full must match all 4000 archived inputs."""
from pathlib import Path
import os,sys,json,gzip,hashlib
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
ROOT=Path(__file__).resolve().parents[1]
CACHE=Path('/root/BindPack/outputs/4_datasets_result/shared_pack_inputs/latest_full_20261027')
ENV={}

def init():
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['TOKENIZERS_PARALLELISM']='false'
    sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
    import torch
    torch.set_num_threads(1)

def check(job):
    ds,path=job
    import numpy as np
    from bindpack import BindPackConfig
    from bindpack.stages.pack.stage import PackStage
    from bindpack.evaluation.measured_reader import prepare
    from bindpack.utils.misc_utils import QuerySolution
    from bindpack.prompts.prompt_template_manager import PromptTemplateManager
    from transformers import AutoTokenizer
    from run_pack_weight_diagnostic import restore_pack_input
    from run_pack_weight_sweep import ReadOnlyPassages
    if ds not in ENV:
        cfg=BindPackConfig.from_file(**json.loads((CACHE/'capture'/ds/'configuration.json').read_text()))
        store=ReadOnlyPassages(next((ROOT/'outputs/4_datasets_result/shared_indexes'/ds).rglob('vdb_chunk.parquet')))
        refs=json.loads((CACHE/'metadata'/ds/'references.json').read_text())
        tokenizer=AutoTokenizer.from_pretrained('/root/models/Qwen3-8B',local_files_only=True)
        reader=SimpleNamespace(prompt_template_manager=PromptTemplateManager(role_mapping={'system':'system','user':'user','assistant':'assistant'}))
        ENV[ds]=(cfg,store,refs,tokenizer,reader)
    cfg,store,refs,tokenizer,reader=ENV[ds]
    with gzip.open(path,'rt') as f:event=json.load(f)
    raw=event['input'];assert hashlib.sha256(json.dumps(raw,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()==event['input_sha256']
    cache={}
    def visible(question,ordered):
        key=(question,tuple(ordered))
        if key not in cache:
            sol=QuerySolution(question=question,docs=[store.texts[d] for d in ordered],doc_scores=np.ones(len(ordered)))
            reading,budget=prepare(reader,sol,tokenizer,ds,k=cfg.pack.budget,context_window=cfg.pack.review_context_length,output_tokens=cfg.max_new_tokens)
            cache[key]=(dict(zip(ordered,reading.docs)),budget)
        return cache[key]
    runtime=SimpleNamespace(cfg=cfg,_document=lambda d:store.texts[int(d)],_trace=lambda:{},_joint_visible=visible)
    result=PackStage(runtime).finalize(raw['query'],np.asarray(raw['ids'],int),np.asarray(raw['scores'],float),{},restore_pack_input(raw))
    target=refs[event['sample_id']];diag=result[2]['budgeted_selection']
    assert result[0][:200].tolist()==target['full_top200'],f'Full ranking changed: {ds}/{event["sample_id"]}'
    assert diag['reference_doc_ids']==target['reference_doc_ids'] and diag['candidate_domain']==target['candidate_domain']
    assert abs(diag['objective']-target['objective'])<1e-12 and not diag['search_deadline_reached']
    return ds

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    jobs=[(ds,str(f)) for ds in ('hotpotqa','2wikimultihopqa','musique','hover') for f in sorted((CACHE/'inputs'/ds).glob('*.json.gz'))]
    assert len(jobs)==4000
    from collections import Counter
    counts=Counter()
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=init) as pool:
        for ds in pool.map(check,jobs,chunksize=4):
            counts[ds]+=1
            if sum(counts.values())%100==0:print('FULL PARITY',dict(counts),flush=True)
    assert all(v==1000 for v in counts.values())
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'status':'passed','samples':dict(counts),'full_ranking_and_objective_unchanged':True,'new_llm_calls':0,'new_embedding_calls':0},indent=2)+'\n')

if __name__=='__main__':main()

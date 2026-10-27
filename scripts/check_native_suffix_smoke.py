"""Three real requests paths through the native opt-in suffix engine, no index."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys

PROJECT=Path('/root/BindPack')
ROOT=PROJECT/'outputs/support_review_optimization_20261026'
sys.path.insert(0,str(PROJECT/'src'))


def main():
    os.environ.setdefault('OPENAI_API_KEY','EMPTY')
    from transformers import AutoTokenizer
    from bindpack import BindPackConfig
    from bindpack.llm.openai_gpt import CacheOpenAI
    from bindpack.evaluation.measurement import Recorder,jsonl
    from bindpack.utils.misc_utils import QuerySolution
    from bindpack.stages.pack.verification.support_semantic_veto import postprocess_support_semantic_veto
    from copy import deepcopy
    rows=json.loads((PROJECT/'outputs/4_datasets_result/latest_r060_20261026/cases/hotpotqa/full/result.json').read_text())['results']
    rows=[r for r in rows if r['retrieval_trace']['evidence']['improvement_support_semantic_veto']['llm_stages']['reviewer']['output']][:3]
    cfg=BindPackConfig.from_file(ROOT/'method.json',pack={'review_mode':'suffix_stop'},save_dir=str(ROOT/'native_engine_smoke/cache'))
    tokenizer=AutoTokenizer.from_pretrained('/root/models/Qwen3-8B',local_files_only=True)
    model=CacheOpenAI.from_experiment_config(cfg);recorder=Recorder(ROOT/'native_engine_smoke/accounting')
    recorder.install_llm(model,tokenizer);checked=[]
    for row in rows:
        trace=deepcopy(row['retrieval_trace']['evidence']);old=trace.pop('improvement_support_semantic_veto')
        docs=dict(zip(old['final_doc_ids'],row['candidate_docs']));parent=old['original_parent_doc_ids']
        solution=QuerySolution(question=row['question'],docs=[docs[d] for d in parent],doc_scores=[1.]*200,retrieval_trace={'evidence':trace})
        rag=SimpleNamespace(llm_model=model,_extract_llm_text=str)
        runtime=SimpleNamespace(cfg=cfg,rag=rag,_document=docs.__getitem__,_support_semantic_tokenizer=tokenizer)
        with recorder.section('online_retrieval',row['sample_id']):
            stats=postprocess_support_semantic_veto(runtime,[(solution,parent)])
        new=solution.retrieval_trace['evidence']['improvement_support_semantic_veto']
        assert all(new[k] for k in ('top2_preserved','top10_set_preserved','top200_set_preserved','suffix_preserved','unique_documents','protected_support_preserved'))
        checked.append(dict(sample_id=row['sample_id'],requests=stats['logical_extra_requests'],accepted=new['accepted'],
            ranking_matches_saved=new['final_doc_ids']==old['final_doc_ids'],
            prefix_repairs=sum(bool(v['output'] and v['output'].get('json_closure_reconstructed')) for v in new['llm_stages'].values())))
    calls=jsonl(ROOT/'native_engine_smoke/accounting/llm_calls.jsonl')
    assert calls and all(c['status']=='success' and c['prompt_tokens'] is not None and c['completion_tokens'] is not None for c in calls)
    out=dict(status='passed',native_review_mode=cfg.pack.review_mode,relation_weight=cfg.pack.relation_weight,
        queries=checked,physical_requests=len(calls),new_index=False,new_embedding_calls=0)
    (ROOT/'native_engine_smoke/validation.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(out,ensure_ascii=False))


if __name__=='__main__':main()

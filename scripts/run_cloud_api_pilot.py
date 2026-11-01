#!/usr/bin/env python3
"""Native online-LLM replacement on a verified existing Qwen embedding index."""
from __future__ import annotations
import argparse
from datetime import datetime
from collections import Counter,defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

PROJECT=Path('/root/BindPack')
DEFAULT_OUTPUT=PROJECT/'outputs/api_qwen38_flash_qwenemb8b_pilot4_20261030'
DEFAULT_CREDENTIALS=Path('/root/.config/bindpack/dashscope.json')
ARCHIVE=PROJECT/'outputs/4_datasets_result/shared_indexes'


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(4194304),b''):h.update(part)
    return h.hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)
def stamp():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def select_indices(data,dataset,n):
    sys.path.insert(0,str(PROJECT/'scripts'))
    from eval_utils import get_benchmark_hops,get_sample_id
    hops=get_benchmark_hops(data,dataset);groups=defaultdict(list)
    if n==len(data):
        return list(range(n)),dict(Counter(map(str,hops)))
    for index,hop in enumerate(hops):groups[hop].append(index)
    def key(index):return hashlib.sha256(f'cloud-api-pilot-v1|{dataset}|{get_sample_id(data[index],dataset)}'.encode()).digest()
    for group in groups.values():group.sort(key=key)
    chosen=[]
    while len(chosen)<n:
        for hop in sorted(groups):
            if groups[hop] and len(chosen)<n:chosen.append(groups[hop].pop(0))
    return chosen,dict(Counter(str(hops[index]) for index in chosen))


def write_cost_report(output,dataset):
    sys.path.insert(0,str(PROJECT/'scripts'))
    from utils.api_cost import summarize_calls,cost_for_usage
    case=output/'run/cases'/dataset/'full'
    report=read(case/'report.json');done=read(case/'completed.ok')
    for name,key in [('report.json','report_sha256'),('result.json','result_sha256'),('llm_calls.jsonl','llm_calls_sha256')]:
        assert sha(case/name)==done[key]
    n=done['query_count'];assert report['coverage']['sample_count']==report['coverage']['qa_n']==n
    full=read(output/'protocol.json').get('full_dataset',False)
    assert read(case/'audit.json')['status']=='passed'
    calls=[json.loads(line) for line in (case/'llm_calls.jsonl').read_text().splitlines() if line.strip()]
    online=[call for call in calls if call['phase'].startswith('online_')]
    assert not [call for call in calls if call['phase']=='offline_index']
    costs=summarize_calls(online)
    by_phase={phase:summarize_calls([call for call in online if call['phase']==phase])
              for phase in sorted({call['phase'] for call in online})}
    estimates={str(size):dict(queries=size,estimated_cny=costs['estimated_cny']*size/n,
        all_input_at_standard_rate_cny=costs['all_input_at_standard_rate_cny']*size/n)
        for size in (1000,2000,4000)}
    historical=[]
    # A larger historical token ledger is a separate sensitivity check, not a
    # measurement of the replacement model. It must not overwrite this pilot.
    historical_path=PROJECT/'docs/DASFAA_2027_latex/results/serial_cost.json'
    if historical_path.exists():
        for row in read(historical_path)['per_dataset']:
            point=cost_for_usage(row['input_tokens'],row['output_tokens'])
            historical.append(dict(dataset=row['dataset'],source_queries=row['n'],
                hypothetical_1000_query_cny=point['estimated_cny']*1000/row['n'],
                basis='old local Qwen3-8B tokens at supplied API rates, all input uncached; not qwen3.8-flash measured usage'))
    preflight=read(output/'preflight/api.json') if (output/'preflight/api.json').exists() else None
    setup_cost=None
    if preflight and preflight['status']=='passed':
        usage=preflight['response']['usage'];details=usage.get('prompt_tokens_details') or {}
        setup_cost=cost_for_usage(usage['prompt_tokens'],usage['completion_tokens'],details.get('cached_tokens',0) or 0)
    result=dict(status='complete',dataset=dataset,pilot_queries=None if full else n,queries=n,full_dataset=full,model='qwen3.8-flash',
        embedding='/root/models/Qwen3-Embedding-8B',thinking_enabled=False,index_rebuilt=False,
        extraction_llm='qwen3-8b',replacement_scope='native online Route+Plan+Pack+Reader; historical OpenIE/index retained',
        actual_pilot_api_usage=costs,by_phase=by_phase,estimated_full_online_cost=estimates,
        historical_token_sensitivity=historical,preflight_cost=setup_cost,
        qa_length_limit_count=report['evidence_budget']['reader_finish_length_count'],
        estimate_limits=('This is the completed full dataset; cross-dataset extrapolations remain estimates. ' if full else
            'Hop-diverse questions are a feasibility sample, not a random cost estimator. ')+
            'Cost varies with query mix, retries, generation length and provider cache. Estimates exclude historical indexing, local embedding/GPU costs, and any thinking or batch execution.',
        report=str(case/'report.json'),report_sha256=sha(case/'report.json'),metrics=report['metrics'])
    write(output/'COST_ESTIMATE.json',result)
    metric_keys=('Recall@5','FCR@5','DCC@5','ExactMatch','F1')
    if full:
        write(output/'summary.json',result)
    lines=['# Cloud API '+('full experiment' if full else 'integration pilot'),'',
        f'Model: `qwen3.8-flash` (Alibaba Model Studio), not DeepSeek. Dataset: {dataset}; {n} questions. '
        'Local Qwen3-Embedding-8B and immutable original Qwen3-8B OpenIE index reused. '
        'Native online Route, Plan and Reader calls are new API requests. Pack keeps relevance0.60, joint selection on, source eligibility and final review off. Non-thinking mode; ordinary synchronous API pricing.','',
        '| N | '+' | '.join(metric_keys)+' |','|---:|'+'---:|'*len(metric_keys),
        '| '+str(n)+' | '+' | '.join('N/A' if report['metrics'].get(k) is None else f"{report['metrics'][k]*100:.2f}" for k in metric_keys)+' |','',
        ('All original questions were evaluated, including length-truncated answers.' if full else
         'These scores are an interface diagnostic, not a formal model-factor result.'),'',
        '| Item | Measured |','|---|---:|',
        f"| Physical online API requests | {costs['physical_requests']} |",
        f"| Input tokens | {costs['input_tokens']} |",f"| Cached input tokens (provider-reported) | {costs['cached_tokens']} |",
        f"| Output tokens | {costs['output_tokens']} |",f"| Estimated API charge / pilot | CNY {costs['estimated_cny']:.6f} |",'',
        '| Extrapolated queries | At observed cache rate (CNY) | All input at regular rate (CNY) |',
        '|---:|---:|---:|']
    for size,value in estimates.items():lines.append(f"| {size} | {value['estimated_cny']:.3f} | {value['all_input_at_standard_rate_cny']:.3f} |")
    lines+=['',result['estimate_limits'],'',
        'Prices supplied by user, CNY/million tokens: normal input0.8, output2.7, cache hit0.1, explicit cache creation1.25. '
        'No explicit cache creation or Batch File/Batch Chat service is requested. HTTP parallelism is not Batch File billing. '
        'Usage details and raw provider model identity are retained in the physical request ledger.',
        '', 'Historical local-model token sensitivity (separate from API pilot):','',
        '| Dataset | Old-model tokens at current rates / 1000 queries (CNY) |','|---|---:|']
    for row in historical:lines.append(f"| {row['dataset']} | {row['hypothetical_1000_query_cny']:.3f} |")
    if setup_cost:lines+=['',f"The separate connectivity probe cost is CNY {setup_cost['estimated_cny']:.6f}; not included in the four-query denominator."]
    lines+=['','## Reuse','',
        f'Credentials are read only from `{DEFAULT_CREDENTIALS}` (owner-only permissions), outside source/results snapshots. '
        'No API key is stored in the run config or logs.','',
        '```bash',f'/root/anaconda3/envs/rag/bin/python -B /root/BindPack/scripts/run_cloud_api_pilot.py --dataset {dataset} --sample-size {n} --output {output} --resume','```','',
        'A new output directory creates an independent run/cache. `--resume` keeps completed queries and actual cost history. '
        'Completed reruns validate saved artifacts without calling the API again.']
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    write(output/'completed.ok',dict(queries=n,report_sha256=sha(case/'report.json'),cost_estimate_sha256=sha(output/'COST_ESTIMATE.json')))
    print(json.dumps(dict(queries=n,input_tokens=costs['input_tokens'],cached_input_tokens=costs['cached_tokens'],
        output_tokens=costs['output_tokens'],pilot_estimated_cny=costs['estimated_cny'],full_online_estimates=estimates),indent=2),flush=True)


def main(args):
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    credential_path=args.credentials.resolve()
    if credential_path.stat().st_mode&0o077:raise ValueError('Credentials must be owner-only readable')
    credentials=read(credential_path)
    if credentials['model']!='qwen3.8-flash':raise ValueError('This pilot profile expects qwen3.8-flash')
    snapshot=output/'source_snapshot'
    if not args.resume:
        if (output/'protocol.json').exists():raise ValueError('Existing experiment: use --resume or choose a new output')
        data=read(PROJECT/'datasets'/(args.dataset+'.json'))
        if not 1<=args.sample_size<=len(data):raise ValueError('Invalid sample size')
        indices,hops=select_indices(data,args.dataset,args.sample_size)
        workers=args.workers or 4
        full=len(indices)==len(data)
        for folder in ('src','scripts','configs'):
            shutil.copytree(PROJECT/folder,snapshot/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        write(output/'indices.json',indices)
        config=read(PROJECT/'configs/ablations_qwen3.json')
        config.update(method_config=str(snapshot/'configs/full_no_source_eligibility.json'),datasets=[args.dataset])
        config['runtime'].update(llm_name=credentials['model'],llm_base_url=credentials['base_url'],llm_prefetch_workers=workers,
            embedding_batch_size=args.embedding_batch_size or 4)
        config['evaluation'].update(eval_mode='rag_qa',sample_size=args.sample_size)
        write(output/'config.json',config)
        write(output/'protocol.json',dict(created_at=stamp(),dataset=args.dataset,queries=args.sample_size,indices=indices,
            hop_distribution=hops,full_dataset=full,
            selection='all original indices in dataset order' if full else 'round robin native hop classes, deterministic SHA order; no answers or metrics used',
            llm_model=credentials['model'],base_url=credentials['base_url'],thinking_enabled=False,
            workers=workers,query_window=max(8,workers),reader_workers=max(8,workers),
            qa_max_tokens=args.qa_max_tokens,qa_length_policy='truncate at cap, retain raw response and parse once, no retry',
            gpu_retrieval=args.gpu_retrieval,embedding_batch_size=config['runtime']['embedding_batch_size'],
            credentials_file=str(credential_path),credentials_value_recorded=False,
            embedding_model='/root/models/Qwen3-Embedding-8B',index_extraction_llm='qwen3-8b',
            index_scope='retain original graph/OpenIE and embedding index; replace online model only',
            source_sha256={str(p.relative_to(snapshot)):sha(p) for p in snapshot.rglob('*') if p.is_file()}))
    protocol=read(output/'protocol.json')
    assert protocol['dataset']==args.dataset and protocol['queries']==args.sample_size
    assert credentials['model']==protocol['llm_model'] and credentials['base_url']==protocol['base_url']
    for relative,value in protocol['source_sha256'].items():
        if sha(snapshot/relative)!=value:raise ValueError('Frozen source changed: '+relative)
    workers=protocol.get('workers',4)
    if args.resume:
        if args.workers is not None and args.workers!=workers:raise ValueError('Resume worker limit changed')
        if args.qa_max_tokens is not None and args.qa_max_tokens!=protocol.get('qa_max_tokens'):raise ValueError('Resume QA cap changed')
    use_gpu=protocol.get('gpu_retrieval',False)
    env=dict(os.environ,OPENAI_API_KEY=credentials['api_key'],DASHSCOPE_API_KEY=credentials['api_key'],
        PYTHONPATH='/root/.cache/pathcondrag/runtime_deps_qwen3_tf4513',TOKENIZERS_PARALLELISM='false',
        PYTHONUNBUFFERED='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',
        BINDPACK_RETRIEVAL_DEVICE='cuda' if use_gpu else 'cpu')
    run=output/'run'
    command=[sys.executable,'-B','-u',str(snapshot/'scripts/run_bindpack_measured.py'),'--frozen-worker',
        '--dataset',args.dataset,'--output',str(run),'--config',str(output/'config.json'),
        '--datasets-dir',str(PROJECT/'datasets'),'--reuse-archive-index',str(ARCHIVE/args.dataset),
        '--query-window',str(protocol.get('query_window',8)),'--reader-workers',str(protocol.get('reader_workers',8)),
        '--model-factor-profile','dashscope_qwen3_emb8b','--index-extraction-llm','qwen3-8b',
        '--eval-mode','rag_qa','--source-bound-eligibility','off','--candidate-output-top-k','10']
    if not protocol.get('full_dataset',False):command+=['--query-indices',str(output/'indices.json')]
    if protocol.get('qa_max_tokens') is not None:command+=['--qa-max-tokens',str(protocol['qa_max_tokens'])]
    if (run/'experiment_config.json').exists():command+=['--resume']
    write(output/'launch.json',dict(command=command,credential_environment_names=['OPENAI_API_KEY','DASHSCOPE_API_KEY'],
        env_secret_values_omitted=True,started_at=stamp()))
    (output/'run.pid').write_text(str(os.getpid())+'\n')
    with (output/'run.log').open('a') as log:
        log.write(stamp()+(' START authorized full dataset' if protocol.get('full_dataset') else ' START API integration pilot')+
                  f'; HTTP/Route workers={workers}; GPU retrieval={use_gpu}; thinking=false; QA cap={protocol.get("qa_max_tokens") or 2048}\n');log.flush()
        process=subprocess.Popen(command,cwd=PROJECT,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:
            log.write(line);log.flush();print(line,end='',flush=True)
        code=process.wait()
    if code:raise RuntimeError('API pilot failed; inspect '+str(output/'run.log'))
    sys.path.insert(0,str(snapshot/'scripts'))
    write_cost_report(output,args.dataset)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=['hotpotqa','2wikimultihopqa','musique','hover'],default='musique')
    parser.add_argument('--sample-size',type=int,default=4)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--credentials',type=Path,default=DEFAULT_CREDENTIALS)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--workers',type=int,choices=[4,8,16,32])
    parser.add_argument('--qa-max-tokens',type=int,choices=range(1,2049))
    parser.add_argument('--embedding-batch-size',type=int,choices=[4,8,16,32])
    parser.add_argument('--gpu-retrieval',action='store_true',help='Keep vector matrices and weighted PageRank transition on CUDA')
    main(parser.parse_args())

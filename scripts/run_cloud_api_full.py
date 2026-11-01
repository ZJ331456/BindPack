#!/usr/bin/env python3
"""Fresh API extraction/index and full MuSiQue retrieval/QA with durable accounting."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

PROJECT=Path('/root/BindPack')
DEFAULT=PROJECT/'outputs/api_qwen38_flash_fresh_index_musique_full_20261030'
CREDENTIALS=Path('/root/.config/bindpack/dashscope.json')
PREVIOUS=[PROJECT/'outputs'/name for name in (
    'api_qwen38_flash_fresh_index_pilot_20261030',
    'api_qwen38_flash_qwenemb8b_musique_full_20261030',
    'api_qwen38_flash_qwenemb8b_pilot4_20261030')]

def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(4194304),b''):h.update(part)
    return h.hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)
def stamp():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')

def setup(root):
    sys.path.insert(0,str(PROJECT/'scripts'))
    from eval_utils import build_docs_from_full_corpus,ensure_gold_docs_in_corpus,get_gold_docs,get_benchmark_hops
    data=read(PROJECT/'datasets/musique.json');corpus=read(PROJECT/'datasets/musique_corpus.json')
    docs=ensure_gold_docs_in_corpus(build_docs_from_full_corpus(corpus),
        [sorted(set(group)) for group in get_gold_docs(data,'musique')])
    assert len(data)==1000 and len(docs)==len(set(docs))==11656
    snapshot=root/'source_snapshot'
    for folder in ('src','scripts','configs'):
        shutil.copytree(PROJECT/folder,snapshot/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for stem in ('musique','musique_corpus'):
        destination=root/'frozen_datasets'/(stem+'.json');destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(PROJECT/'datasets'/(stem+'.json'),destination)
    credentials=read(CREDENTIALS)
    assert credentials['model']=='qwen3.8-flash'
    config=read(PROJECT/'configs/ablations_qwen3.json')
    config.update(method_config=str(snapshot/'configs/full_no_source_eligibility.json'),datasets=['musique'])
    config['runtime'].update(llm_name=credentials['model'],llm_base_url=credentials['base_url'],
        llm_prefetch_workers=32,openie_max_workers=8,embedding_batch_size=32)
    config['evaluation'].update(eval_mode='rag_qa',sample_size=1000)
    write(root/'config.json',config)
    from collections import Counter
    protocol=dict(created_at=stamp(),dataset='musique',questions=1000,passages=11656,
        selected_indices=list(range(1000)),hop_distribution=dict(Counter(map(str,get_benchmark_hops(data,'musique')))),
        full_run_authorized=True,index_reused=False,pilot_index_reused=False,
        extraction_model='qwen3.8-flash',online_model='qwen3.8-flash',
        llm_base_url=credentials['base_url'],embedding='/root/models/Qwen3-Embedding-8B',
        thinking_enabled=False,qa_max_tokens=2028,reader_top_k=5,
        native_index_workers=8,query_window=32,reader_workers=32,http_max_in_flight=32,
        gpu_embedding=True,gpu_vector_search=True,gpu_pagerank=True,gpu_synonym_knn=True,
        graph_assembly_device='cpu',pack_relevance_weight=.6,pack_selector='combined',
        source_eligibility=False,support_semantic_veto=False,
        provider_rejection_policy='abstain',
        rejection_details='Only explicit data_inspection_failed; no retry/splitting of rejected content, keep original text/vector, exclude rejected extraction facts; Plan abstains and rejected Reader returns empty scored answer; no samples removed.',
        qa_length_policy='retain truncated native answer; no continuation/retry',
        expected_metrics=['Recall@5','FCR@5','NodeCov@5','DCC@5','ClosureGap@5','ExactMatch','F1'],
        corpus_source_sha256=sha(PROJECT/'datasets/musique_corpus.json'),
        data_source_sha256=sha(PROJECT/'datasets/musique.json'),
        credential_values_recorded=False,credentials_source=str(CREDENTIALS),
        source_sha256={str(path.relative_to(snapshot)):sha(path) for path in sorted(snapshot.rglob('*')) if path.is_file()})
    write(root/'protocol.json',protocol)
    prior=root/'previous_run_records'
    # Keep small, auditable billing/validation records, never old indexes/caches.
    receipt=[]
    for source in PREVIOUS:
        if not source.exists():continue
        chosen=[source/name for name in ('COST_ESTIMATE.json','README.md','API接入与费用说明.md',
            '建库到QA测试与全量费用估算.md','validation.json','STOPPED_BEFORE_INDEX_MODEL_PILOT.json')]
        chosen += [source/'run/cases/musique/full'/name for name in ('llm_calls.jsonl','llm_logical_calls.jsonl','audit.json','index_build.json','error.json')]
        chosen += [source/'validation/provider_schema_probe'/name for name in ('llm_calls.jsonl','checks.json')]
        for path in chosen:
            if not path.is_file():continue
            destination=prior/source.name/path.relative_to(source)
            destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
            assert sha(path)==sha(destination)
            receipt.append(dict(original=str(path),saved=str(destination.relative_to(root)),bytes=path.stat().st_size,sha256=sha(path)))
    write(root/'previous_run_records.json',dict(scope='Historical diagnostic/billing records only; not charged to or used as current Full results/index',files=receipt))
    (root/'RUN.md').write_text(f'''# MuSiQue：API 全流程 Full

启动准备时间：{stamp()}。用户已授权全量。模型为 **qwen3.8-flash**，本地 embedding 为 **Qwen3-Embedding-8B**。

空目录重新执行 11,656 篇原文 NER/OpenIE、全部向量化及建图，然后运行原始顺序的 1,000 题 Route、Plan、Pack、Reader@5。没有拷贝旧模型或小测试的索引/抽取/答案。

保持相关性权重 0.60、联合选择开启；来源资格与末端 Support / semantic veto 关闭。thinking=false；QA 输出上限 2028，截断答案原样计分，不重试续写。建库原生 8 并发；检索窗口、Reader、HTTP 上限均为 32。Embedding、向量计算、加权 PageRank、同义近邻矩阵计算使用 GPU；图组装和 Python 控制仍在 CPU。

最终输出 Recall@5、FCR@5、NodeCov@5、DCC@5、ClosureGap@5、EM、QA F1，以及实际请求、输入/输出/cache tokens、API 费用、阶段耗时和资源采样。

显式 API 内容拒绝记录为失败/弃答，不重试或改写来源规避检查。建库保留原文和 passage 向量，仅不采纳失败关系；Plan 放弃相应模型输出，Reader 被拒绝则空答案计分。最终分母保留全部 1,000 题。服务端不返回 usage 的拒绝会保持未知，金额只汇总已知 token，并显示缺失次数；不伪造 0 token。其他 HTTP 配置/鉴权错误继续报错，传输异常沿用有限重试。

当前实验成本不含 `previous_run_records/` 中的历史测试和已停止任务费用。用户指定的三处旧目录清理记录见 `cleanup.json`。

日志：`run.log`；后台启动输出：`launcher.log`；实时状态：`run/status.json`；建库逐篇断点：`run/cases/musique/full/index/` 的 OpenIE SQLite；逐题断点：`run/cases/musique/full/progress.jsonl`。

完成后原生结果：`run/cases/musique/full/report.json`、`result.json`、`completed.ok`；汇总：`RESULTS.md`、`summary.json`、`API_COST.json`。

```bash
tail -f {root}/run.log
```

续跑使用 `scripts/run_cloud_api_full.sh --output {root} --resume`，保留现有 API 索引、逐题断点、响应缓存和全部原始成本，不重新执行已完成部分。
''')
    (root/'RESULTS.md').write_text('# MuSiQue API Full：运行中\n\n正在从零建库；完整 1,000 题完成后自动写入 Recall@5、FCR@5、DCC@5、EM、QA F1 和费用/时间，当前不填预测成绩。\n')
    return protocol

def cleanup(root):
    receipt=read(root/'previous_run_records.json')
    for row in receipt['files']:assert sha(root/row['saved'])==row['sha256']
    rows=[]
    for path in PREVIOUS:
        assert path.parent.resolve()==(PROJECT/'outputs').resolve() and path not in (root,PROJECT/'outputs')
        if path.is_symlink():raise ValueError('Refusing unexpected symlink: '+str(path))
        if not path.exists():continue
        count=sum(p.stat().st_size for p in path.rglob('*') if p.is_file())
        shutil.rmtree(path)
        rows.append(dict(path=str(path),deleted=True,logical_bytes=count))
    write(root/'cleanup.json',dict(at=stamp(),user_authorized=True,deleted=rows,
        total_logical_bytes=sum(row['logical_bytes'] for row in rows),billing_records_saved=True,
        historical_billing_not_added_to_current_experiment=True))
    print('CLEANUP complete; deleted exactly the three requested previous directories',flush=True)

def summarize(root):
    sys.path.insert(0,str(root/'source_snapshot/scripts'))
    from utils.api_cost import summarize_calls
    case=root/'run/cases/musique/full';report=read(case/'report.json');marker=read(case/'completed.ok')
    assert marker['query_count']==report['coverage']['sample_count']==report['coverage']['qa_n']==1000
    assert all(report['metrics'].get(key) is not None for key in ('Recall@5','FCR@5','DCC@5','ExactMatch','F1'))
    for name,key in [('report.json','report_sha256'),('result.json','result_sha256'),('llm_calls.jsonl','llm_calls_sha256')]:assert sha(case/name)==marker[key]
    calls=[json.loads(line) for line in (case/'llm_calls.jsonl').read_text().splitlines() if line.strip()]
    usage=summarize_calls(calls)
    phases={phase:summarize_calls([event for event in calls if event['phase']==phase]) for phase in sorted({e['phase'] for e in calls})}
    write(root/'API_COST.json',dict(total=usage,by_phase=phases,
        prior_run_costs_excluded=True,includes_fresh_index_and_all_online_requests=True,
        usage_missing_is_unknown_not_zero=usage['unknown_usage_attempts']))
    write(root/'summary.json',dict(status='complete',questions=1000,model='qwen3.8-flash',
        index_model='qwen3.8-flash',embedding='/root/models/Qwen3-Embedding-8B',index_rebuilt=True,
        metrics=report['metrics'],cost=usage,by_phase=phases,efficiency=report['efficiency'],
        resources=report['resources'],report_sha256=sha(case/'report.json')))
    inventory=[dict(path=str(p.relative_to(case/'index')),bytes=p.stat().st_size,sha256=sha(p))
        for p in sorted((case/'index').rglob('*')) if p.is_file() and p.suffix in ('.json','.parquet','.pickle')]
    write(root/'index_identity.json',dict(extraction_model='qwen3.8-flash',embedding='/root/models/Qwen3-Embedding-8B',
        passages=11656,build=read(case/'index_build.json'),inventory=inventory,
        text_normalization='ascii_alnum_lower_v1',fresh_index=True,report_sha256=sha(case/'report.json')))
    keys=['Recall@5','FCR@5','NodeCov@5','DCC@5','ClosureGap@5','ExactMatch','F1']
    table='| 题数 | '+' | '.join(keys)+' |\n|---:|'+'---:|'*len(keys)+'\n| 1000 | '+' | '.join(f'{100*report["metrics"][key]:.2f}' for key in keys)+' |\n'
    (root/'RESULTS.md').write_text('# MuSiQue API Full：完成\n\nqwen3.8-flash 全新建库 + 本地 Qwen3-Embedding-8B + 原生 Full；Reader@5，分母包含全部 1,000 题。指标单位为百分数。\n\n'+table+
        f'\n当前全流程已知 API 费用 ¥{usage["estimated_cny"]:.6f}；实际请求 {usage["physical_requests"]} 次；未知 usage {usage["unknown_usage_attempts"]} 次。费用包含新建库与在线检索/QA，历史任务费用不计入。\n\n'+
        '原始报告：[report.json](run/cases/musique/full/report.json)；逐题结果：[result.json](run/cases/musique/full/result.json)；分阶段费用：[API_COST.json](API_COST.json)。\n')
    write(root/'completed.ok',dict(questions=1000,report_sha256=sha(case/'report.json'),api_cost_sha256=sha(root/'API_COST.json')))
    print('FULL COMPLETE: '+json.dumps(report['metrics'],ensure_ascii=False),flush=True)

def main(args):
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    if CREDENTIALS.stat().st_mode&0o077:raise ValueError('Credentials must be owner-only readable')
    if not (root/'protocol.json').exists():protocol=setup(root)
    else:protocol=read(root/'protocol.json')
    assert protocol['full_run_authorized'] and protocol['questions']==1000 and not protocol['index_reused']
    for relative,expected in protocol['source_sha256'].items():
        if sha(root/'source_snapshot'/relative)!=expected:raise ValueError('Frozen source changed: '+relative)
    if args.prepare_only:
        write(root/'prepared.ok',dict(questions=1000,passages=11656,api_calls=0,source_verified=True))
        print('Prepared fresh Full, no model calls',flush=True);return
    if (root/'completed.ok').exists():
        assert sha(root/'run/cases/musique/full/report.json')==read(root/'completed.ok')['report_sha256']
        print('Full already complete; no new model calls',flush=True);return
    if (root/'run/experiment_config.json').exists() and not args.resume:raise ValueError('Existing full experiment requires --resume')
    credentials=read(CREDENTIALS)
    assert credentials['model']==protocol['extraction_model'] and credentials['base_url']==protocol['llm_base_url']
    env=dict(os.environ,OPENAI_API_KEY=credentials['api_key'],DASHSCOPE_API_KEY=credentials['api_key'],
        PYTHONPATH='/root/.cache/pathcondrag/runtime_deps_qwen3_tf4513',
        BINDPACK_RETRIEVAL_DEVICE='cuda',BINDPACK_KNN_DEVICE='cuda',BINDPACK_PROVIDER_REJECTION_POLICY='abstain',
        PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
    command=[sys.executable,'-B','-u',str(root/'source_snapshot/scripts/run_bindpack_measured.py'),
        '--frozen-worker','--dataset','musique','--output',str(root/'run'),'--config',str(root/'config.json'),
        '--datasets-dir',str(root/'frozen_datasets'),'--query-window','32','--reader-workers','32',
        '--model-factor-profile','dashscope_qwen3_emb8b_fresh','--source-bound-eligibility','off',
        '--eval-mode','rag_qa','--qa-max-tokens','2028','--candidate-output-top-k','10']
    if args.resume:command+=['--resume']
    write(root/'launch.json',dict(command=command,at=stamp(),credentials_values_recorded=False))
    with (root/'run.log').open('a') as log:
        log.write(stamp()+' START AUTHORIZED FULL: fresh API index 11656 passages + 1000 queries; no old index reuse\n');log.flush()
        process=subprocess.Popen(command,cwd=PROJECT,env=env,stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        write(root/'run.pid.json',dict(controller_pid=os.getpid(),worker_pid=process.pid))
        (root/'run.pid').write_text(str(os.getpid())+'\n')
        if args.clean_previous and not (root/'cleanup.json').exists():cleanup(root)
        for line in process.stdout:
            log.write(line);log.flush();print(line,end='',flush=True)
        code=process.wait()
    write(root/'execution_result.json',dict(returncode=code,finished_at=stamp()))
    if code:raise RuntimeError('Full worker stopped; durable checkpoints and actual costs retained. Inspect '+str(root/'run.log'))
    summarize(root)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--clean-previous',action='store_true')
    parser.add_argument('--resume',action='store_true')
    main(parser.parse_args())

#!/usr/bin/env python3
"""Audit the fresh-index pilot and estimate offline/online costs separately."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from utils.api_cost import summarize_calls,DEFAULT_PRICES

def read(path):return json.loads(Path(path).read_text())
def lines(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main(root):
    case=root/'run/cases/musique/full';protocol=read(root/'pilot_protocol.json')
    report=read(case/'report.json');done=read(case/'completed.ok');calls=lines(case/'llm_calls.jsonl')
    for name,key in [('report.json','report_sha256'),('result.json','result_sha256'),('llm_calls.jsonl','llm_calls_sha256')]:
        assert sha(case/name)==done[key]
    for relative,expected in protocol['source_sha256'].items():assert sha(root/'source_snapshot'/relative)==expected
    assert done['query_count']==4 and read(case/'audit.json')['status']=='passed'
    assert read(root/'pilot_execution.json')['returncode']==0
    assert all(event['model']=='qwen3.8-flash' for event in calls)
    native=read(case/'protocol.json')
    assert native['model_factor_profile']=='dashscope_qwen3_emb8b_fresh'
    assert 'index_reuse' not in native
    assert native['configuration']['pack']['relation_weight']==.6
    assert native['configuration']['pack']['review_mode']=='off'
    assert native['configuration']['source_bound_eligibility'] is False
    index=case/'index';openie=read(index/'openie_results_ner_qwen3.8-flash.json')
    publication=openie['publication_report']
    assert publication['document_count']==139 and publication['failed_document_count']==0
    offline=[event for event in calls if event['phase']=='offline_index']
    online=[event for event in calls if event['phase'].startswith('online_')]
    assert len(offline)>0 and all(event['max_tokens']==2028 for event in calls if event['phase']=='online_reader')
    byhash={hashlib.sha256(row['passage'].encode()).hexdigest():row['idx'] for row in openie['docs']}
    uniform_ids=[byhash[value] for value in protocol['uniform_document_sha256']]
    assert len(uniform_ids)==128
    uniform=[event for event in offline if event['query_id'] in set(uniform_ids)]
    offline_cost=summarize_calls(offline);online_cost=summarize_calls(online)
    probe=summarize_calls(lines(root/'validation/provider_schema_probe/llm_calls.jsonl'))
    uniform_cost=summarize_calls(uniform)
    full_n=protocol['full_unique_passages'];scale=full_n/128
    per_doc=[summarize_calls([event for event in uniform if event['query_id']==key])['estimated_cny'] for key in uniform_ids]
    rng=random.Random(20261030)
    boot=sorted(statistics.mean(rng.choices(per_doc,k=128))*full_n for _ in range(4000))
    previous=read(root/'validation/previous_256_online_reference.json')
    previous_n=previous['completed_queries'];previous_usage=previous['usage']
    assert previous_usage['failed_attempts']==previous_usage['unknown_usage_attempts']==0
    stages=report['efficiency']['offline_index']['stages']
    embedding_stages=['chunk_embedding_and_store','entity_embedding_and_store','fact_embedding_and_store']
    embedding_seconds=sum(stages[key]['total'] for key in embedding_stages)
    index_lower_workload=sum(event['seconds'] for event in uniform)/8*scale
    index_batch_linear=read(case/'index_build.json')['seconds']*full_n/139
    index_vector_linear=embedding_seconds*full_n/139
    online_projected=(previous['retrieval_wall_seconds']+previous['qa_wall_seconds'])/previous_n*1000
    values=dict(status='complete',model='qwen3.8-flash',embedding='/root/models/Qwen3-Embedding-8B',
        index_rebuilt=True,old_openie_reused=False,old_graph_reused=False,old_vectors_reused=False,
        pilot_questions=4,pilot_passages=139,uniform_passages_for_cost_estimate=128,
        full_questions=1000,full_passages=full_n,full_run_started=False,
        thinking_enabled=False,qa_max_tokens=2028,prices_cny_per_million=DEFAULT_PRICES,
        measured=dict(index=offline_cost,online=online_cost,schema_probe=probe,
            total_including_probe_cny=offline_cost['estimated_cny']+online_cost['estimated_cny']+probe['estimated_cny'],
            index_build_seconds=read(case/'index_build.json')['seconds'],
            model_load_seconds=report['efficiency']['model_load_seconds'],
            online_retrieval_wall_seconds=report['efficiency']['completed_retrieval_wall_seconds'],
            online_qa_wall_seconds=report['efficiency']['completed_reader_wall_seconds'],
            pilot_launcher_wall_seconds=read(root/'pilot_execution.json')['wall_seconds'],
            finish_length_by_phase=dict(Counter(event['phase'] for event in calls if event.get('finish_reason')=='length')),
            index_extra_requests_beyond_one_ner_one_triple=len(offline)-2*139,
            metrics=report['metrics']),
        estimates=dict(
            index_api_cny=uniform_cost['estimated_cny']*scale,
            index_api_sampling_bootstrap_95pct_cny=[boot[99],boot[3900]],
            bootstrap_scope='Sampling variability only; excludes future provider/retry/systematic changes.',
            online_api_cny_from_fresh_pilot=online_cost['estimated_cny']*250,
            online_api_no_cache_cny_from_fresh_pilot=online_cost['all_input_at_standard_rate_cny']*250,
            online_api_cny_previous_completed_256=previous_usage['estimated_cny']/previous_n*1000,
            online_api_no_cache_cny_previous_completed_256=previous_usage['all_input_at_standard_rate_cny']/previous_n*1000,
            total_point_cny=uniform_cost['estimated_cny']*scale+online_cost['estimated_cny']*250,
            recommended_api_budget_cny=35,api_budget_is_cap=False,
            online_reference_scope='Prior completed 256 API questions, same provider/embedding, historical local-model graph. Forecast only; new API graph can change execution.',
            local_gpu_rental_power_cost_included=False,
            time=dict(index_workers=8,query_window=32,reader_workers=32,
                openie_request_workload_divided_by_8_projected_seconds=index_lower_workload,
                sampled_embedding_linear_projection_seconds=index_vector_linear,
                complete_pilot_index_wall_linear_projection_seconds=index_batch_linear,
                previous_online32_projected_seconds=online_projected,
                retrieval32_projected_seconds=previous['retrieval_wall_seconds']/previous_n*1000,
                qa32_projected_seconds=previous['qa_wall_seconds']/previous_n*1000,
                index_estimated_hours=[2,3],online_estimated_minutes=[35,50],total_estimated_hours=[2.5,4],
                basis='Index request workload /8 plus sampled embedding throughput; raw small-batch wall projection includes long-tail overhead. Graph KNN is not assumed exactly linear. Online throughput uses prior completed 32-query windows, not the underfilled four-query pilot.',
                caveat='Planning ranges, not measured full-run results or a service quota guarantee. Provider inspection rejections need explicit failure handling.')),
        artifact_hashes={name:sha(case/name) for name in ['protocol.json','report.json','result.json','llm_calls.jsonl','index_build.json']})
    write(root/'COST_ESTIMATE.json',values)
    identity=dict(schema='bindpack_fresh_api_index_pilot_v1',llm='qwen3.8-flash',
        embedding=values['embedding'],index_reused=False,passages=139,
        extraction_scope='NER and triples freshly generated by API; structural validation, not semantic entailment proof',
        publication=publication,source_sha256=protocol['source_sha256'],
        index_assets=[dict(path=str(path.relative_to(index)),bytes=path.stat().st_size,sha256=sha(path))
            for path in sorted(index.rglob('*')) if path.is_file() and path.suffix in ('.json','.parquet','.pickle')],
        request_ledger_sha256=sha(case/'llm_calls.jsonl'),full_corpus_index_ready=False)
    write(root/'index_identity.json',identity)
    write(root/'validation/final_audit.json',dict(status='passed',fresh_index=True,
        physical_index_requests=len(offline),physical_online_requests=len(online),
        failed_requests=sum(event['status']!='success' for event in calls),
        missing_usage=sum(event.get('prompt_tokens') is None or event.get('completion_tokens') is None for event in calls),
        snapshot_verified=True,completed_artifacts_verified=True,
        index_publication_failures=publication['failed_document_count'],
        qa_length_responses=sum(event.get('finish_reason')=='length' for event in online if event['phase']=='online_reader'),
        full_run_started=False,api_budget_cny=35,api_budget_is_enforced_cap=False))
    text=f'''# API 换模型：从零建库到检索与 QA 的测试及全量估算

本轮已完成 **qwen3.8-flash 从实体抽取、关系抽取到 Route、Plan、Reader 的真实端到端测试**。Embedding 为本地 `/root/models/Qwen3-Embedding-8B`。旧全量任务已经停止；本轮没有启动新的全量任务。

## 测试范围与结果

- 语料：从 MuSiQue 全部 {full_n:,} 篇原文中固定随机抽 128 篇，另加入 4 条功能测试题所需的 11 篇原文，共 139 篇。费用外推仅用 128 篇随机部分，排除额外加入的原文。
- 问题：沿用最初 4 条的原始索引 882 / 449 / 179 / 295，覆盖 2 / 2 / 3 / 4 跳。
- 在空目录真正执行 API NER/OpenIE，重新生成 passage/entity/fact 向量、关系图和同义边，未拷贝旧索引或旧抽取结果。
- 保持相关性权重 0.60、联合选择开启、来源资格与末端 Support / semantic veto 关闭；thinking=false；Reader@5，QA 上限 2028，达到上限保留截断结果，不续写。
- **139/139 篇抽取完成，4/4 条检索与 QA 完成，HTTP 失败及未知 usage 均为 0。** 有 1 次关系抽取到达 2048；按现有建库有界恢复流程最终完成，全部恢复费用保留。共额外 3 次关系抽取请求；4 次 QA 均正常 stop。

| 小语料功能检查 | Recall@5 | FCR@5 | EM | F1 |
|---|---:|---:|---:|---:|
| 4 题 | 93.75% | 75.00% | 100.00% | 100.00% |

**这是补入测试题原文的小语料接入检查，不是完整语料的正式成绩，不能据此预测全量正确率。**

## 本次实测费用与时间

| 范围 | 实际 API 请求 | 输入 tokens | 输出 tokens | 估算金额 |
|---|---:|---:|---:|---:|
| 139 篇全新建库 | {offline_cost['physical_requests']} | {offline_cost['input_tokens']} | {offline_cost['output_tokens']} | ¥{offline_cost['estimated_cny']:.6f} |
| 4 题检索、计划、QA | {online_cost['physical_requests']} | {online_cost['input_tokens']} | {online_cost['output_tokens']} | ¥{online_cost['estimated_cny']:.6f} |
| 独立接口格式预检 | {probe['physical_requests']} | {probe['input_tokens']} | {probe['output_tokens']} | ¥{probe['estimated_cny']:.6f} |
| 本次总计 | {len(calls)+probe['physical_requests']} | {offline_cost['input_tokens']+online_cost['input_tokens']+probe['input_tokens']} | {offline_cost['output_tokens']+online_cost['output_tokens']+probe['output_tokens']} | **¥{values['measured']['total_including_probe_cny']:.6f}** |

建库 {values['measured']['index_build_seconds']:.2f} 秒；4 题检索/计划窗口 {values['measured']['online_retrieval_wall_seconds']:.2f} 秒，QA 窗口 {values['measured']['online_qa_wall_seconds']:.2f} 秒。启动器总墙钟 {values['measured']['pilot_launcher_wall_seconds']:.2f} 秒（含加载、落盘等）。

GPU 执行 embedding、向量点积、加权 PageRank 与建库同义近邻矩阵计算；图组装、ID 映射和 Python 控制流程仍在 CPU。小测试建库为原生 8 并发；仅 4 题的查询窗口配置 8、HTTP 上限 32，实际不能形成 32 题满载。

## MuSiQue 全量估算：11,656 篇建库 + 1,000 题

| 范围 | API 费用估计 | 耗时估计 |
|---|---:|---:|
| 全新 API 建库，包括抽取、向量与建图 | **约 ¥{values['estimates']['index_api_cny']:.2f}** | **约 2–3 小时** |
| 全量检索、Plan、QA | **约 ¥7–9** | **约 35–50 分钟** |
| 全过程 | **中心估计约 ¥26–28；建议预留 ¥35** | **约 2.5–4 小时** |

金额按用户给出的每百万 tokens 输入 ¥0.8、输出 ¥2.7、真实缓存命中输入 ¥0.1 计算；未使用 Batch 或显式缓存创建。随机 128 篇的实际建库费用为 ¥{uniform_cost['estimated_cny']:.7f}，乘以 {full_n}/128 得到 ¥{values['estimates']['index_api_cny']:.4f}。随机抽样 bootstrap 95% 区间约 ¥{boot[99]:.2f}–{boot[3900]:.2f}，**只反映抽样波动，不涵盖未来限流、重试或负载变化**。本地 GPU 电费/租金未计入 API 费用；¥35 是预算建议，未设置硬性扣费上限。

在线费用交叉参考：新建小语料 4 题外推约 ¥{online_cost['estimated_cny']*250:.2f}；此前同一 API 已完成 256 题的有效窗口（排除未完成窗口及拒绝请求）外推约 ¥{previous_usage['estimated_cny']/previous_n*1000:.2f}，按全部输入无缓存约 ¥{previous_usage['all_input_at_standard_rate_cny']/previous_n*1000:.2f}。**256 题使用旧图，只作在线开销参考，新 API 图会影响检索与计划，不能当作新全量实测。**

耗时口径是**建库 8 并发、全量查询/QA 32 并发**：随机 128 篇的请求服务时间之和 /8 外推约 {index_lower_workload/60:.1f} 分钟，再加向量化、建图及 IO。小测试完整建库墙钟线性外推约 {index_batch_linear/3600:.2f} 小时，包含较明显的少量慢请求尾部开销；因此给出 2–3 小时的建库规划范围，没有将并发度机械地等同于线性提速。此前 256 题满 32 题窗口外推：检索/Plan {values['estimates']['time']['retrieval32_projected_seconds']/60:.1f} 分钟、QA {values['estimates']['time']['qa32_projected_seconds']/60:.1f} 分钟。

## 当前状态和已知问题

旧任务未重启，保留 256 题断点及全部真实费用记录。旧任务出现过服务端 `data_inspection_failed` 内容审查拒绝，不属于解析失败；本轮小测试未出现该错误。全量时间估计以正常服务与明确处理拒绝为前提，不能保证任意语料请求都会被 API 接受，也未改写原文规避服务端检查。

**仅测试已完成，全量等待用户确认。** 本目录的新 API 抽取、原文、向量、图、源码和账本均保留；这是 139 篇小语料索引，不是已经建成的 11,656 篇完整索引。

## 文件与复用

- [COST_ESTIMATE.json](COST_ESTIMATE.json)：实际用量、分阶段费用、全量估算及计算口径。
- [index_identity.json](index_identity.json)：新索引身份、抽取模型、资产哈希与发布检查。
- [pilot_protocol.json](pilot_protocol.json)：随机抽样、原始题目索引、有效配置与源码哈希。
- [report.json](run/cases/musique/full/report.json)：4 条功能测试报告及阶段时间。
- [llm_calls.jsonl](run/cases/musique/full/llm_calls.jsonl)：每次真实 API usage 与逐篇建库归属。
- [final_audit.json](validation/final_audit.json)：完整性检查。

建库协议保留本地结构校验，API 请求改用其支持的 `response_format=json_schema`，不向 API 发送 vLLM 专属 EBNF 参数。参考[百炼结构化输出官方文档](https://help.aliyun.com/zh/model-studio/qwen-structured-output)。结构合法不等于所有关系语义均正确。
'''
    (root/'README.md').write_text(text)
    (root/'建库到QA测试与全量费用估算.md').write_text(text)
    write(root/'completed.ok',dict(queries=4,index_passages=139,full_run_started=False,
        cost_estimate_sha256=sha(root/'COST_ESTIMATE.json'),index_identity_sha256=sha(root/'index_identity.json')))
    print(json.dumps(dict(actual_cny=values['measured']['total_including_probe_cny'],
        estimated_full_index_cny=values['estimates']['index_api_cny'],
        full_api_budget_cny=35,estimated_hours=[2.5,4],full_run_started=False),ensure_ascii=False,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    main(parser.parse_args().output.resolve())

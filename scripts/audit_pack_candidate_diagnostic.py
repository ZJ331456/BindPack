#!/usr/bin/env python3
"""Validate every locked question, paired outcome, native source and cost ledger."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

PROJECT=Path('/root/BindPack')
DEFAULT=PROJECT/'outputs/pack_candidates_r060_20261026'
LABELS={'baseline':'当前 Full 0.60','short_circuit':'无效请求短路','complete_dag_gate':'按实际完整闭合跳过审核',
    'progressive_coverage':'渐进覆盖','goal_redundancy':'目标感知去重','route_pool200':'Route Top200 候选补充',
    'deep_chain_review':'深串行链审核','dag_score_guard':'DAG 相关性损失约束','budget_contract':'预算接口一致性','combined':'保留项组合'}


def read(p):return json.loads(Path(p).read_text())
def digest(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def jsonl(p):return [json.loads(v) for v in Path(p).read_text().splitlines() if v] if Path(p).exists() else []


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=DEFAULT);a=p.parse_args();root=a.output
    protocol=read(root/'experiment.json');done=read(root/'completed.ok');assert done['status']=='complete'
    selected=read(root/'selection_manifest.json');assert digest(selected)==protocol['selection_sha256']==read(root/'selection_locked.ok')['sha256']
    arms=('baseline',)+tuple(protocol['arms'])+('combined',);pairs={};counts=Counter();costs={};quality_reports=0
    for ds,spec in selected['datasets'].items():
        ids=[v['sample_id'] for v in spec['rows']];assert len(ids)==len(set(ids))==50
        baseline=read(root/'cases'/ds/'baseline/result.json')['results']
        detail_base=read(root/'cases'/ds/'baseline/metrics_details.json')['per_query'];pairs[ds]={};costs[ds]={}
        assert sha(Path('/root/BindPack/outputs/component_diagnostic_r060_20261026/cases')/ds/'combined/result.json')==protocol['input_result_hashes'][ds]
        for arm in arms:
            case=root/'cases'/ds/arm;result=read(case/'result.json');report=read(case/'report.json');details=read(case/'metrics_details.json')['per_query']
            rows=result['results'];assert [r['sample_id'] for r in rows]==ids==[r['sample_id'] for r in details]
            assert sha(case/'result.json')==read(case/'validated.ok')['result_sha256']
            assert result['runtime_config']['pack']['relation_weight']==.6
            assert report['coverage']['sample_count']==50
            counter=Counter()
            for row,before,d,bd in zip(rows,baseline,details,detail_base):
                token=hashlib.sha256(str(row['sample_id']).encode()).hexdigest()+'.json';checkpoint=read(case/'checkpoints'/token)
                assert checkpoint['sha256']==digest(row)==digest(checkpoint['row'])
                assert row['frozen_pack_input_sha256']==before['frozen_pack_input_sha256']
                assert len(row['candidate_docs'])==200
                semantic=row['retrieval_trace']['evidence']['improvement_support_semantic_veto']
                assert len(semantic['final_doc_ids'])==len(set(semantic['final_doc_ids']))==200
                assert all(semantic[k] for k in ('top2_preserved','top10_set_preserved','top200_set_preserved','suffix_preserved','unique_documents','protected_support_preserved'))
                counter['changed_top5_order']+=row['candidate_docs'][:5]!=before['candidate_docs'][:5]
                counter['changed_top5_set']+=set(row['candidate_docs'][:5])!=set(before['candidate_docs'][:5])
                for k,name in [('FCR','fcr'),('Recall','recall'),('DCC','dcc')]:
                    if d['at_k']['5'].get(k) is not None:
                        counter[name+'_wins']+=d['at_k']['5'][k]>bd['at_k']['5'][k]
                        counter[name+'_losses']+=d['at_k']['5'][k]<bd['at_k']['5'][k]
                counter['correctness_wins']+=d['qa']['ExactMatch']>bd['qa']['ExactMatch']
                counter['correctness_losses']+=d['qa']['ExactMatch']<bd['qa']['ExactMatch']
                if ds!='hover':
                    counter['token_f1_wins']+=d['qa']['F1']>bd['qa']['F1']
                    counter['token_f1_losses']+=d['qa']['F1']<bd['qa']['F1']
                counter['reader_partial_passages']+=len(row['reader']['budget']['partial_passage_ranks'])
                counter['reader_dropped_passages']+=len(row['reader']['budget']['dropped_passage_ranks'])
            assert counter['reader_partial_passages']==counter['reader_dropped_passages']==0
            pairs[ds][arm]=dict(counter)
            directory=root/'accounting'/ds/arm;logical=jsonl(directory/'llm_logical_calls.jsonl');http=jsonl(directory/'llm_calls.jsonl')
            c=Counter(http_attempts=len(http),logical_calls=len(logical),cache_hits=sum(bool(v.get('cache_hit')) for v in logical),
                failed_http=sum(v['status']!='success' for v in http),unknown_usage=sum(v.get('prompt_tokens') is None or v.get('completion_tokens') is None for v in http),
                physical_input_tokens=sum(v.get('prompt_tokens') or 0 for v in http),physical_output_tokens=sum(v.get('completion_tokens') or 0 for v in http))
            counts.update(c);costs[ds][arm]=dict(c);quality_reports+=1
    for arm in arms:
        hashes=read(root/'variants'/arm/'source_hashes.json')
        assert all(sha(root/'variants'/arm/rel)==value for rel,value in hashes.items())
    promotion_path=root/'production_promotion.json';promotion=read(promotion_path) if promotion_path.exists() else {'status':'not_applied'}
    if promotion['status']=='applied_and_verified':
        assert promotion['native_parity_questions']==200 and promotion['new_http_requests']==0
        assert all(sha(PROJECT/rel)==value for rel,value in promotion['changes'].items())
        base_hashes=read(root/'production_base_hashes.json')
        changed={rel for rel,value in base_hashes.items() if sha(PROJECT/rel)!=value}
        assert changed==set(promotion['changes']), ('Unexpected production source edits',changed)
    assert read(PROJECT/'src/bindpack/configuration/defaults.json')['pack']['relation_weight']==.6
    assert read(root/'invariant_checks/passed.ok')['variants']==8
    assert counts['failed_http']==counts['unknown_usage']==0
    write=lambda path,value:path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    audit=dict(status='passed',unique_questions=200,quality_reports=quality_reports,evaluated_rows=quality_reports*50,
        index_rebuilt=False,embedding_inference=False,route_plan_rerun=False,current_baseline_cache_parity_questions=200,
        invariant_checked_variants=8,production=promotion['status'],incremental_llm_cost=dict(counts),diagnostic_only=True)
    write(root/'audit.json',audit);write(root/'paired_changes.json',pairs);write(root/'costs.json',costs)
    final=read(root/'final_decision.json');decisions=final['decisions']
    lines=['# Pack 逐项诊断结果：固定 0.60','',
        '每个数据集固定最初 50 条，共 200 条；所有版本复用同一 Route+Plan 输入，未建索引、未执行 embedding。基线含已经采用的 HoVer reader。本清单按历史表现选择，反复用于诊断，不能充当独立开发集或正式主实验。','',
        '| 单项 | 处理 | 平均 ΔRecall@5 | 平均 ΔFCR@5 | 平均 Δ回答 F1 / Macro-F1 | 平均 ΔEM / Accuracy | Δ逻辑调用 |',
        '|---|---|---:|---:|---:|---:|---:|']
    names={'retain_quality':'保留：质量收益','retain_efficiency':'保留：效率收益','tradeoff':'未保留：指标有升有降','no_gain_or_regression':'未保留：无收益或退化'}
    for arm in protocol['arms']:
        d=decisions[arm];m=d['mean_deltas'];lines.append(f'| {LABELS[arm]} | {names[d["status"]]} | {m["recall"]:+.6f} | {m["fcr"]:+.6f} | {m["qa"]:+.6f} | {m["correctness"]:+.6f} | {d["logical_call_delta"]:+d} |')
    lines+=['','平均值仅是预先约定的筛选量；普通 QA token F1 与 HoVer Macro-F1 不属于同一种任务指标。合入还要求每个数据集 Recall/FCR/回答主指标/正确率及可用 DCC 均不下降。','',
        '## 最终保留与组合','',f'最终版本：`{final["arm"]}`；采用项：`{final["features"]}`。生产状态：`{promotion["status"]}`。','',
        '| 数据集 | 版本 | Recall@5 | FCR@5 | EM / Accuracy | F1 / Macro-F1 | DCC@5 | 逻辑调用 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for ds in selected['datasets']:
        for arm in dict.fromkeys(('baseline',final['arm'])):
            m=read(root/'cases'/ds/arm/'report.json')['metrics'];ks=('Recall@5','FCR@5','ClassificationAccuracy' if ds=='hover' else 'ExactMatch','ClassificationMacroF1' if ds=='hover' else 'F1','DCC@5')
            calls=read(root/'cases'/ds/arm/'cost.json')['committed_rows']['logical_calls']
            lines.append(f'| {ds} | {LABELS[arm]} | '+' | '.join('N/A' if m[k] is None else f'{m[k]:.6f}' for k in ks)+f' | {calls} |')
    lines+=['','MuSiQue 的独立 gold DAG 指标：','',
        '| 版本 | NodeCov@5 | DCC@5 | ClosureGap@5 |','|---|---:|---:|---:|']
    for arm in dict.fromkeys(('baseline',final['arm'])):
        m=read(root/'cases/musique'/arm/'report.json')['metrics']
        lines.append(f'| {LABELS[arm]} | '+ ' | '.join(f'{m[k]:.6f}' for k in ('NodeCov@5','DCC@5','ClosureGap@5'))+' |')
    lines+=['','## 各方向的具体范围','',
        '- 渐进覆盖采用 noisy-OR 边际累积，保持原 goal 身份及 R=.60；不是上一轮已失败的事实归并实现。',
        '- 目标感知去重按 goal 分布重合度缩放词重合惩罚，并对新 verified goal 减免；这是事实级去重的一个近似，不能视为语义等价判定。',
        '- 候选补充仅加入实际 Route Top200 中已存在的 passage，不伪造额外目标分或 proof；不代表已经测试重新运行 Route/Plan 后的更大检索池。',
        '- 深链审核只放开串行深度限制，仍是一次有界交换，保留并行/多父节点/类型/来源保护。不能解决所有多证据缺失。',
        '- DAG 损失约束保留原严格 proof 准入，用预测闭合的归一化增量限制相关性损失；不代表 gold 闭合。',
        '- 预算接口候选在 K=5 做实际 200 题一致性测试；K=3/8/10 和七节点配置只做 CPU 边界检查，没有声称得到这些 K 的新 QA 曲线。非 K=5 显式停用尚未验证的语义审核。默认无质量/调用收益时不合入该工程候选。','',
        '## 成本与核验','',f'本轮额外实际 HTTP 请求 {counts["http_attempts"]} 次，输入 {counts["physical_input_tokens"]}、输出 {counts["physical_output_tokens"]} tokens；失败 {counts["failed_http"]} 次，usage 缺失 {counts["unknown_usage"]} 次。',
        '','各 case 的 cost.json 统计已提交 50 题的逻辑调用和真实缓存元数据；accounting 保留全部实际尝试。不同候选先后执行，共享私有缓存，因此实际新增 HTTP 数和墙钟时间不能直接当作公平的冷缓存效率排名。调用短路的收益按同输入、相同质量下的逻辑调用减少判断，未将缓存节省当成模型推理加速。',
        '','主代码另修正了审核短路的统计语义：本地保护已否决、未请求 reviewer 的提案不再计为实际语义审核否决。它只改变诊断字段；原生源码的 200 条排名、reader 提示和答案均已零新增 HTTP 复现。说明见 [production_diagnostic_fix.json](production_diagnostic_fix.json)。',
        '','各组同时保存 report.json 和 metrics_details.json，含 Recall/FCR/EM/F1，以及标注允许时的 NodeCov/DCC/ClosureGap；无可用 gold DAG 的数据集相应指标为 N/A。',
        '','完整对照：[comparison.md](comparison.md)；逐题胜负：[paired_changes.json](paired_changes.json)；成本：[costs.json](costs.json)；独立审计：[audit.json](audit.json)；代码采用：[production_promotion.json](production_promotion.json)。']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    write(root/'status.json',dict(phase='complete',arm=final['arm'],features=final['features'],production=promotion['status']))
    print(json.dumps(audit,ensure_ascii=False))


if __name__=='__main__':main()

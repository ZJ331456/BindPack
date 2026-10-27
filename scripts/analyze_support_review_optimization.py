"""Summarize actual review trials without mixing concurrency or replay scopes."""
import argparse
import json
from pathlib import Path


def read(path):return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('/root/BindPack/outputs/support_review_optimization_20261026'))
    a=p.parse_args();root=a.output.resolve();reports={}
    for path in sorted((root/'cases').glob('*/*/report.json')):
        d=read(path);reports[d['dataset']+'/'+path.parent.name]=d
    combined={}
    for arm in ('off','current','compact_single','compact_two','tight_two','prefix_two','adaptive_two'):
        phases=['pilot','validation']
        paths=[root/f'cases/hotpotqa/{phase}_{arm}' for phase in phases]
        if not all((path/'completed.ok').exists() for path in paths):continue
        rows=[read(p) for path in paths for p in (path/'checkpoints').glob('*.json')]
        assert len(rows)==len({r['sample_id'] for r in rows})==1000
        new={k:sum(r['metrics'][k] for r in rows)/1000 for k in ('Recall@5','FCR@5')}
        new.update({k:sum(r['qa'][k] for r in rows)/1000 for k in ('ExactMatch','F1')})
        old={k:sum(r['baseline_metrics'][k] for r in rows)/1000 for k in ('Recall@5','FCR@5')}
        old.update({k:sum(r['baseline_qa'][k] for r in rows)/1000 for k in ('ExactMatch','F1')})
        gains={k:new[k]-old[k] for k in new}
        base=reports.get('hotpotqa/pilot_current')
        pilot=reports.get('hotpotqa/pilot_'+arm)
        reduction=None if base is None or pilot is None else 1-pilot['review_latency']['mean']/base['review_latency']['mean']
        quality=all(v>=-1e-10 for v in gains.values()) and any(v>1e-10 for v in gains.values())
        combined[arm]=dict(samples=1000,metrics=new,baseline_metrics=old,deltas=gains,
            fcr_wins=sum(r['metrics']['FCR@5']>r['baseline_metrics']['FCR@5'] for r in rows),
            fcr_losses=sum(r['metrics']['FCR@5']<r['baseline_metrics']['FCR@5'] for r in rows),
            em_wins=sum(r['qa']['ExactMatch']>r['baseline_qa']['ExactMatch'] for r in rows),
            em_losses=sum(r['qa']['ExactMatch']<r['baseline_qa']['ExactMatch'] for r in rows),
            serial_pilot_review_time_reduction=reduction,
            quality_target_met=quality,time_target_met=reduction is not None and reduction>=.3,
            generalization_gate_passed=quality and reduction is not None and reduction>=.3)
    data=dict(reports=reports,combined_hotpotqa=combined,production_promoted=False,
        production_default_review_mode='standard',
        optional_native_review_modes=['suffix_stop'],
        queue_resumed=False,scope='Fixed-parent diagnostic; reuse exact context answers; no new native end-to-end main run.')
    (root/'summary.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    lines=['# Support / semantic review 优化实验','',
        '权重 0.60，固定同次运行审核前排名；模型、Reader@5、来源校验不变。未重跑索引、Route、Plan、embedding。100 题试验后用其余 900 题确认。所有替代方案均为诊断结果，不覆盖旧 Full。','',
        '| 数据集 / 阶段 / 方案 | 题数 | Recall@5 | FCR@5 | EM / Accuracy | F1 / Macro-F1 | 审核平均秒 | 并发题数 | 审核 HTTP 请求 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for key,d in reports.items():
        m=d['metrics'];cost=d['cost'];n=sum(cost[s]['http_attempts'] for s in ('online_proposer','online_reviewer'))
        correct=m.get('ClassificationAccuracy',m['ExactMatch']);f1=m.get('ClassificationMacroF1',m['F1'])
        lines.append(f"| {key} | {d['samples']} | {m['Recall@5']:.6f} | {m['FCR@5']:.6f} | {correct:.6f} | {f1:.6f} | {d['review_latency']['mean']:.3f} | {d['query_concurrency']} | {n} |")
    for arm,d in combined.items():
        lines+=['',f'## HotpotQA 全 1000 题：{arm}','',
            '| 指标 | 原 Full | 新方案 | 差值 |','|---|---:|---:|---:|']
        for k,v in d['metrics'].items():lines.append(f"| {k} | {d['baseline_metrics'][k]:.6f} | {v:.6f} | {d['deltas'][k]:+.6f} |")
        lines += ['',f"配对 FCR 改善 / 下降：{d['fcr_wins']} / {d['fcr_losses']}；EM：{d['em_wins']} / {d['em_losses']}。",
            f"相同串行 100 题上，审核平均耗时减少 {d['serial_pilot_review_time_reduction']*100:.2f}%；质量目标通过：{d['quality_target_met']}；耗时目标通过：{d['time_target_met']}。"]
    lines+=['','HoVer 使用 Accuracy / Macro-F1，其余三个数据集使用答案 EM / token F1。','',
        '计时只描述审核组件；批处理并发下的平均响应时间不能与串行值直接比较。已保存答案仅在 Reader 输入文档及顺序完全一致时复用，不将省掉的 reader 请求算作方法提速。各 case 的 report.json 另含实际请求、tokens、证据预算与资源计量。','',
        '原全量队列已停止：HotpotQA 完成 1000，2Wiki 断点 324。没有自动恢复旧队列或修改生产默认算法。']
    if any('adaptive_two' in key for key in reports):
        lines+=['','自适应分流是在看到 tight_two 的 1000 题诊断后提出的开发假设；其 HotpotQA 结果属于开发集回测，不能称为新的独立验证或论文主实验。分流只使用 Plan 绑定完成度，不读取 gold / 参考正确性 / 样本 ID / 数据集名称。']
    lines+=['','原生代码提供可选 `pack.review_mode=suffix_stop`，默认仍为 `standard`。该选项只有 HotpotQA 100 题真模型质量/耗时对照；四数据集 1300 题为保存响应的兼容性回放，不是重新推理的泛化成绩。激进压缩和自适应分流均未采纳为生产策略。']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'completed_cases':list(reports),'combined_hotpotqa':combined},ensure_ascii=False))


if __name__=='__main__':main()

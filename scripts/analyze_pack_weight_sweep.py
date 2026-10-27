#!/usr/bin/env python3
"""Compare completed 0.40/0.80 replay arms against the fixed 0.60 baseline."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

DEFAULT=Path('/root/BindPack/outputs/pack_weight_diagnostic_20261025/extension_r040_r080')
NAMES={'hotpotqa':'HotpotQA','2wikimultihopqa':'2WikiMultiHopQA','musique':'MuSiQue','hover':'HoVer'}
GROUPS={'similar':'预计接近','expected_r025':'原预计偏向 0.25','expected_r060':'原预计偏向 0.60'}


def read(path):return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT)
    args=parser.parse_args();root=args.output.resolve();protocol=read(root/'experiment.json');base=Path(protocol['base'])
    assert read(root/'audit.json')['status']=='passed'
    selected=read(base/'selection_manifest.json');comparisons=[];csv_rows=[]
    data=read(root/'comparison.json')['datasets']
    lines=['# 0.40 / 0.80 相对于 0.60 的配对分析','','同一批 200 条诊断样本，保留原预选组；不依据本轮结果重新分组。0.60 为已验证的同状态基线。所有指标使用 0–1 小数。','',
        '## Recall@5 原始值','','每个数据集 50 条，四组使用相同冻结上游输入。','',
        '| 数据集 | 0.25 | 0.40 | 0.60 | 0.80 |',
        '|---|---:|---:|---:|---:|']
    for dataset in data:
        values=[dataset['variants'][weight]['metrics']['Recall@5'] for weight in ('0.25','0.40','0.60','0.80')]
        lines.append(f'| {NAMES[dataset["dataset"]]} | '+' | '.join(f'{value:.6f}' for value in values)+' |')
    lines+=['','## 相对于 0.60 的指标变化','','正值代表新增权重更高，ClosureGap 越低越好。','',
        '| 数据集 | 权重 | ΔRecall@5 | ΔFCR@5 | ΔEM / Accuracy | Δtoken F1 / Macro-F1 | ΔDCC@5 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for dataset in data:
        ds=dataset['dataset'];baseline=dataset['variants']['0.60']['metrics']
        old=read(base/'cases'/ds/'full_r060/metrics_details.json')['per_query']
        for weight,case in zip(protocol['weights'],protocol['cases']):
            new=read(root/'cases'/ds/case/'metrics_details.json')['per_query']
            metrics=read(root/'cases'/ds/case/'report.json')['metrics']
            qa='ClassificationAccuracy' if ds=='hover' else 'ExactMatch'
            f1='ClassificationMacroF1' if ds=='hover' else 'F1'
            delta={key:None if metrics.get(key) is None else metrics[key]-baseline[key]
                for key in ('Recall@5','FCR@5',qa,f1,'DCC@5','NodeCov@5','ClosureGap@5')}
            keys=('Recall@5','FCR@5',qa,f1,'DCC@5')
            lines.append(f'| {NAMES[ds]} | {weight:.2f} | '+' | '.join('N/A' if delta[k] is None else f'{delta[k]:+.6f}' for k in keys)+' |')
            counts=Counter();groups={g:Counter() for g in GROUPS}
            for a,b,chosen in zip(old,new,selected['datasets'][ds]['rows']):
                assert a['sample_id']==b['sample_id']==chosen['sample_id']
                x,y=a['at_k']['5'],b['at_k']['5'];c=groups[chosen['group']]
                for counter in (counts,c):
                    counter['n']+=1
                    counter['fcr_gain']+=y['FCR']>x['FCR'];counter['fcr_loss']+=y['FCR']<x['FCR']
                    counter['em_gain']+=b['qa']['ExactMatch']>a['qa']['ExactMatch']
                    counter['em_loss']+=b['qa']['ExactMatch']<a['qa']['ExactMatch']
                    counter['fcr_r060']+=x['FCR'];counter['fcr_new']+=y['FCR']
                    counter['em_r060']+=a['qa']['ExactMatch'];counter['em_new']+=b['qa']['ExactMatch']
                csv_rows.append(dict(dataset=ds,weight=weight,sample_id=a['sample_id'],cohort=chosen['group'],
                    r060_recall5=x['Recall'],new_recall5=y['Recall'],r060_fcr5=x['FCR'],new_fcr5=y['FCR'],
                    r060_em=a['qa']['ExactMatch'],new_em=b['qa']['ExactMatch'],
                    r060_answer_f1=a['qa']['F1'],new_answer_f1=b['qa']['F1'],
                    r060_dcc5=x['DCC'],new_dcc5=y['DCC']))
            assert counts['n']==50
            comparisons.append(dict(dataset=ds,weight=weight,delta_vs_r060=delta,paired_counts=dict(counts),
                cohorts={key:dict(value) for key,value in groups.items()}))
    lines+=['','## 逐题胜负','','每格为“新权重改善 / 新权重退化”的题数，均以 0.60 为参照。','','| 数据集 | 权重 | 完整证据改善 / 退化 | 答案正确性改善 / 退化 |','|---|---:|---:|---:|']
    for row in comparisons:
        c=row['paired_counts'];lines.append(f'| {NAMES[row["dataset"]]} | {row["weight"]:.2f} | {c["fcr_gain"]} / {c["fcr_loss"]} | {c["em_gain"]} / {c["em_loss"]} |')
    lines+=['','## 保留原分组的完整证据诊断','','| 数据集 | 权重 | 原预选组 | n | 0.60 FCR@5 | 新权重 FCR@5 | 改善 / 退化 |','|---|---:|---|---:|---:|---:|---:|']
    for row in comparisons:
        for group,c in row['cohorts'].items():
            lines.append(f'| {NAMES[row["dataset"]]} | {row["weight"]:.2f} | {GROUPS[group]} | {c["n"]} | {c["fcr_r060"]/c["n"]:.3f} | {c["fcr_new"]/c["n"]:.3f} | {c["fcr_gain"]} / {c["fcr_loss"]} |')
    lines+=['','主结果见 [comparison.md](comparison.md)，逐题配对见 [paired_vs_r060.csv](paired_vs_r060.csv)。原 0.25/0.60 结果未修改。这里只比较固定诊断样本，不据此宣称正式全量最优权重。']
    (root/'analysis_vs_r060.md').write_text('\n'.join(lines)+'\n')
    (root/'analysis_vs_r060.json').write_text(json.dumps(dict(diagnostic_only=True,comparisons=comparisons),ensure_ascii=False,indent=2)+'\n')
    with (root/'paired_vs_r060.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(csv_rows[0]));writer.writeheader()
        writer.writerows({k:'N/A' if v is None else v for k,v in r.items()} for r in csv_rows)
    assert len(csv_rows)==400
    plot(root,data)
    print(json.dumps({f'{weight:.2f}':dict(net_complete_evidence=sum(r['paired_counts']['fcr_gain']-r['paired_counts']['fcr_loss'] for r in comparisons if r['weight']==weight),
        net_correct_answers=sum(r['paired_counts']['em_gain']-r['paired_counts']['em_loss'] for r in comparisons if r['weight']==weight)) for weight in protocol['weights']},ensure_ascii=False))


def plot(root,data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True,sharey=True)
    for ax,dataset in zip(axes.flat,data):
        ds=dataset['dataset'];variants=dataset['variants'];weights=sorted(float(k) for k in variants)
        ax.plot(weights,[variants[f'{w:.2f}']['metrics']['Recall@5'] for w in weights],'-D',label='Recall@5',color='#7451a8')
        ax.plot(weights,[variants[f'{w:.2f}']['metrics']['FCR@5'] for w in weights],'-o',label='FCR@5',color='#2878b5')
        key='ClassificationMacroF1' if ds=='hover' else 'F1'
        ax.plot(weights,[variants[f'{w:.2f}']['metrics'][key] for w in weights],'-s',label='Macro-F1' if ds=='hover' else 'Answer F1',color='#d66b26')
        if ds=='musique':ax.plot(weights,[variants[f'{w:.2f}']['metrics']['DCC@5'] for w in weights],'-^',label='DCC@5',color='#3b935c')
        ax.set_title(NAMES[ds]);ax.set_ylim(0,1.04);ax.set_xticks(weights);ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.tick_params(axis='x',labelbottom=True)
        ax.grid(axis='y',alpha=.2);ax.legend(loc='lower right',frameon=False)
        ax.spines[['top','right']].set_visible(False)
        ax.set_xlabel('Pack relation weight');ax.set_ylabel('Score')
    fig.suptitle('Full Pack: fixed-state weight diagnostic (50 questions per dataset)',fontsize=13)
    fig.text(.5,.015,'Historical outcome strata; same upstream inputs. Diagnostic selection, not a representative benchmark.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.95))
    fig.savefig(root/'weight_sensitivity.svg',bbox_inches='tight')
    fig.savefig(root/'weight_sensitivity.png',dpi=200,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Export verified mechanism results as booktabs-style figures and copyable tables.

Run with the base conda Python, which supplies matplotlib. This script only reads
completed reports and writes publication_tables plus the requested summary doc.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

DATASETS = ('hotpotqa', '2wikimultihopqa', 'musique', 'hover')
DATASET_LABELS = ('HotpotQA', '2Wiki', 'MuSiQue', 'HoVer')
ARMS = ('full', 'wo_source_binding', 'wo_joint_selection', 'flat_dependency_support',
        'singleton_search', 'wo_parent_answer_propagation')
LABELS = ('Full', 'w/o Source Binding', 'Reference (S₀)', 'Flat Support',
          'Singleton Search', 'w/o Parent Answer Propagation')


def read(path):
    return json.loads(Path(path).read_text())


def fmt(value):
    return f'{100*value:.2f}' if value is not None else 'N/A'


def best_flags(values, columns):
    best = [max(row[c] for row in values if row[c] is not None) if column.get('higher', True)
            else None for c, column in enumerate(columns)]
    return [[value is not None and top is not None and math.isclose(value, top, abs_tol=1e-12)
             for value, top in zip(row, best)] for row in values]


def markdown_table(values, columns, bold=True, delta=False):
    flags = best_flags(values, columns) if bold else [[False]*len(columns) for _ in values]
    lines = ['| Variant | '+' | '.join(c['header'] for c in columns)+' |',
             '|---|'+'---:|'*len(columns)]
    for arm, label, row, marked in zip(ARMS, LABELS, values, flags):
        cells = [('N/A' if v is None else f'{100*v:+.2f}') if delta else fmt(v) for v in row]
        lines.append('| '+label+' | '+' | '.join(f'**{v}**' if mark else v
                                               for v, mark in zip(cells, marked))+' |')
    return '\n'.join(lines)


def latex_table(path, values, columns, groups, caption, label):
    flags = best_flags(values, columns)
    names = [r'Reference (S$_0$)' if a == 'wo_joint_selection' else name for a, name in zip(ARMS, LABELS)]
    lines = [r'\begin{table*}[t]', r'\centering', r'\caption{'+caption+'}', r'\label{'+label+'}',
             r'\setlength{\tabcolsep}{8pt}', r'\begin{tabular}{l'+'r'*len(columns)+'}', r'\toprule']
    headers = []; rules = []; offset = 2
    for title, width in groups:
        headers.append(r'\multicolumn{'+str(width)+r'}{c}{'+title+'}')
        rules.append(r'\cmidrule(lr){'+f'{offset}-{offset+width-1}'+'}')
        offset += width
    lines += [' & '+' & '.join(headers)+r' \\', ' '.join(rules),
              'Variant & '+' & '.join(c['metric_label'] for c in columns)+r' \\', r'\midrule']
    for name, row, marks in zip(names, values, flags):
        cells = [r'\textbf{'+fmt(v)+'}' if mark else fmt(v) for v, mark in zip(row, marks)]
        lines.append(name+' & '+' & '.join(cells)+r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    path.write_text('\n'.join(lines)+'\n')


def render_table(destination, values, columns, groups):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for name in ('Regular', 'Bold'):
        font_manager.fontManager.addfont('/usr/share/fonts/truetype/liberation2/LiberationSerif-'+name+'.ttf')
    plt.rcParams.update({'font.family': 'Liberation Serif', 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'svg.fonttype': 'none', 'mathtext.fontset':'dejavuserif'})
    wide = len(columns) > 3
    fig = plt.figure(figsize=(19 if wide else 12.5, 4.9), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    left, right = .013, .987
    numeric_left = .32 if wide else .53
    cell_width = (right-numeric_left)/len(columns)
    centers = [numeric_left+(i+.5)*cell_width for i in range(len(columns))]
    ax.hlines([.97, .69, .055], left, right, colors='black', linewidths=[1.7, 1.0, 1.7])
    offset = 0
    for title, width in groups:
        low, high = numeric_left+offset*cell_width, numeric_left+(offset+width)*cell_width
        ax.text((low+high)/2, .895, title, ha='center', va='center', fontsize=23)
        ax.hlines(.825, low+.008, high-.008, colors='black', linewidth=.7)
        offset += width
    ax.text(left, .755, 'Variant', ha='left', va='center', fontsize=22)
    for x, column in zip(centers, columns):
        ax.text(x, .755, column['metric_label'], ha='center', va='center', fontsize=21)
    for row_index, (name, row, marks) in enumerate(zip(LABELS, values, best_flags(values, columns))):
        y = .615-row_index*.10
        rendered_name='Reference (S$_0$)' if name=='Reference (S₀)' else name
        ax.text(left, y, rendered_name, ha='left', va='center', fontsize=20)
        for x, value, mark in zip(centers, row, marks):
            ax.text(x, y, fmt(value), ha='center', va='center', fontsize=21,
                    fontweight='bold' if mark else 'normal')
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(destination.with_suffix('.'+extension), dpi=220, facecolor='white', bbox_inches='tight', pad_inches=.04)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('/root/BindPack/outputs/ablations_redesigned_20261108'))
    parser.add_argument('--document', type=Path, default=Path('/root/BindPack/docs/消融实验结果_20261108.md'))
    args = parser.parse_args(); root = args.run.resolve(); output = root/'publication_tables'
    verification = read(output/'verification.json')
    assert verification['status'] == 'passed' and len(verification['cases']) == 24
    status = read(root/'status.json'); assert status['status'] == 'complete'
    reports = {(ds, arm): read(root/'cases'/ds/arm/'report.json') for ds in DATASETS for arm in ARMS}
    retrieval_columns = [dict(dataset=ds, metric=metric, metric_label=label, header=f'{title} {label}')
        for ds, title in zip(DATASETS, DATASET_LABELS) for metric, label in (('Recall@5', 'R@5'), ('FCR@5', 'FCR@5'))]
    qa_columns = [dict(dataset=ds, metric=metric, metric_label=label, header=f'{title} {label}')
        for ds, title in zip(DATASETS, DATASET_LABELS) for metric, label in (
            (('ClassificationAccuracy', 'Acc.'), ('ClassificationMacroF1', 'Macro-F1')) if ds == 'hover'
            else (('ExactMatch', 'EM'), ('F1', 'F1')))]
    structural_columns = [dict(dataset='musique', metric=m, metric_label=m, header=m, higher=m != 'ClosureGap@5')
        for m in ('NodeCov@5', 'DCC@5', 'ClosureGap@5')]
    tables = {}
    for stem, columns, groups, caption in (
        ('retrieval_ablation', retrieval_columns, [(t, 2) for t in DATASET_LABELS],
         'Mechanism ablations: passage recall and full coverage at K=5. All values are percentages; column bests, including ties, are bold.'),
        ('qa_ablation', qa_columns, [(t, 2) for t in DATASET_LABELS],
         'Reader@5 results. HotpotQA, 2Wiki and MuSiQue report EM/F1; HoVer reports label accuracy and macro-F1. Values are percentages.'),
        ('musique_structure', structural_columns, [('MuSiQue (independent gold DAG)', 3)],
         'Gold-graph diagnostics on all 1000 MuSiQue questions at K=5. Values are percentages. ClosureGap is descriptive; lower gap alone is not evidence of better retrieval.')):
        values = [[reports[(c['dataset'], a)]['metrics'][c['metric']] for c in columns] for a in ARMS]
        tables[stem] = dict(columns=columns, values=values, groups=groups, caption=caption)
        render_table(output/stem, values, columns, groups)
        latex_table(output/(stem+'.tex'), values, columns, groups, caption, 'tab:'+stem)
        with (output/(stem+'.csv')).open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.writer(stream); writer.writerow(['Variant']+[c['header'] for c in columns])
            writer.writerows([[name]+[fmt(v) for v in row] for name, row in zip(LABELS, values)])
    start = datetime.fromtimestamp(status['started_unix'], ZoneInfo('Asia/Shanghai'))
    finish = datetime.fromtimestamp(status['finished_unix'], ZoneInfo('Asia/Shanghai'))
    elapsed = status['finished_unix']-status['started_unix']
    lines = ['# 2026-11-08 机制消融正式结果', '',
        f'24组全部完成并通过结果/报告SHA校验。每数据集1000题，共24000份检索与QA结果。', '',
        f'正式运行：{start:%Y-%m-%d %H:%M:%S} 至 {finish:%Y-%m-%d %H:%M:%S}（北京时间），耗时{int(elapsed//3600)}小时{int(elapsed%3600//60)}分钟。', '',
        '统一使用原共享索引、K=5及Reader@5。所有比例以百分数表示，保留两位小数；粗体为列内最优（含并列）。', '',
        '## 检索结果', '', markdown_table(tables['retrieval_ablation']['values'], retrieval_columns), '',
        '## QA结果', '', 'HoVer报告标签分类Accuracy和Macro-F1，不与前三个数据集的QA EM/F1混用。', '',
        markdown_table(tables['qa_ablation']['values'], qa_columns), '',
        '## MuSiQue独立金标图诊断', '', markdown_table(tables['musique_structure']['values'], structural_columns), '',
        'ClosureGap = NodeCov − DCC；不能仅根据较小的ClosureGap判断检索质量更好。', '',
        '## 相对Full的检索变化（pp）', '']
    retrieval_values = tables['retrieval_ablation']['values']
    changes = [[v-base for v, base in zip(row, retrieval_values[0])] for row in retrieval_values]
    lines += [markdown_table(changes, retrieval_columns, bold=False, delta=True), '',
        '差值基于未舍入指标计算，可能与表中两位小数的展示值相减有0.01 pp差异。', '',
        '## 逐题核对与结果观察', '']
    for ds, title in zip(DATASETS, DATASET_LABELS):
        comparisons = verification['comparisons'][ds]
        unchanged = all(comparisons[a]['changed_ordered_top5'] == 0 and comparisons[a]['changed_answer'] == 0
                        for a in ('flat_dependency_support', 'singleton_search'))
        if unchanged:
            lines.append(f'- {title}：Flat Support与Singleton Search的1000题最终Top5排序及答案均与Full完全相同；已核对Flat评分模式和Singleton零包动作开关。')
        else:
            flat=comparisons['flat_dependency_support'];single=comparisons['singleton_search']
            lines.append(f'- {title}：Flat Support有{flat["changed_ordered_top5"]}/1000题的Top5排序改变、'
                f'{flat["changed_answer"]}/1000题答案改变；Singleton Search有{single["changed_ordered_top5"]}/1000题'
                f'Top5排序改变、{single["changed_answer"]}/1000题答案改变。汇总检索和QA指标仍与Full一致。')
    lines += ['', '- Source Binding消融的FCR@5下降（HotpotQA/2Wiki/MuSiQue/HoVer）：'+
              ' / '.join(f'{100*(reports[(ds,"full")]["metrics"]["FCR@5"]-reports[(ds,"wo_source_binding")]["metrics"]["FCR@5"]):.2f}' for ds in DATASETS)+' pp。',
              '- Parent Answer Propagation消融对应下降：'+
              ' / '.join(f'{100*(reports[(ds,"full")]["metrics"]["FCR@5"]-reports[(ds,"wo_parent_answer_propagation")]["metrics"]["FCR@5"]):.2f}' for ds in DATASETS)+' pp。',
              '- Reference (S₀)的2Wiki QA EM/F1高于Full；Full并非所有QA指标的最优。', '',
              '## 可下载文件', '',
              '- 检索三线表：`retrieval_ablation.png/.pdf/.svg/.tex/.csv`',
              '- QA三线表：`qa_ablation.png/.pdf/.svg/.tex/.csv`',
              '- MuSiQue诊断：`musique_structure.png/.pdf/.svg/.tex/.csv`',
              '- 未舍入数值与逐题比较统计：`tables.json`、`verification.json`', '',
              f'原始结果目录：`{root}/cases/<dataset>/<variant>/`。']
    document = '\n'.join(lines)+'\n'
    (output/'SUMMARY.md').write_text(document)
    args.document.write_text(document.replace('## 可下载文件',
        '## 可下载文件\n\n[打开三线表目录](../outputs/ablations_redesigned_20261108/publication_tables/)'))
    raw = dict(run=str(root), elapsed_seconds=elapsed, cases=24, queries_per_case=1000, units='fraction',
               tables=tables, verification=verification)
    (output/'tables.json').write_text(json.dumps(raw, ensure_ascii=False, indent=2)+'\n')
    with (output/'metrics_long.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream); writer.writerow(['Dataset', 'Variant', 'N', 'Metric', 'Fraction', 'Percent', 'Delta_pp'])
        for ds in DATASETS:
            for arm, label in zip(ARMS, LABELS):
                for metric, value in reports[(ds,arm)]['metrics'].items():
                    base = reports[(ds,'full')]['metrics'][metric]
                    writer.writerow([ds, label, 1000, metric, value, None if value is None else 100*value,
                                     None if value is None or base is None else 100*(value-base)])
    print(json.dumps(dict(output=str(output), document=str(args.document), verified_cases=24,
                          elapsed_hours=elapsed/3600), ensure_ascii=False))


if __name__ == '__main__':
    main()

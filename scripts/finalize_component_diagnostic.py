#!/usr/bin/env python3
"""Select a non-regressing candidate, promote exact tested helpers, verify cache parity."""
from __future__ import annotations
import argparse
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT=Path('/root/BindPack')
DATASETS=('hotpotqa','2wikimultihopqa','musique','hover')
ARMS=('baseline','coverage_facts','plan_repair','chain_package','route_fusion','hover_reader','combined')
LABELS={'baseline':'原算法0.60','coverage_facts':'覆盖事实归并','plan_repair':'Plan失败格式恢复',
    'chain_package':'完整来源链选择','route_fusion':'Route只增不减融合','hover_reader':'HoVer原子核验reader','combined':'保留项组合'}


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');t.replace(p)


def metrics(root,arm):
    out={}
    for ds in DATASETS:
        m=read(root/'cases'/ds/arm/'report.json')['metrics']
        out[ds]={k:m[name] for k,name in dict(qa='ClassificationMacroF1' if ds=='hover' else 'F1',
            correctness='ClassificationAccuracy' if ds=='hover' else 'ExactMatch',fcr='FCR@5',recall='Recall@5').items()}
    return out


def decision(root):
    baseline=metrics(root,'baseline');candidates={}
    kept=read(root/'selection_decision.json')['retained']
    for arm in ARMS:
        m=metrics(root,arm)
        features=kept if arm=='combined' else ([] if arm=='baseline' else [arm])
        delta={ds:{k:m[ds][k]-baseline[ds][k] for k in baseline[ds]} for ds in DATASETS}
        mean={k:sum(delta[ds][k] for ds in DATASETS)/4 for k in ('qa','correctness','fcr','recall')}
        safe=all(delta[ds][k]>=-1e-10 for ds in DATASETS for k in ('qa','correctness','fcr'))
        improved=any(mean[k]>1e-10 for k in ('qa','correctness','fcr'))
        candidates[arm]=dict(features=features,per_dataset=delta,mean_delta=mean,
            no_dataset_qa_correctness_fcr_regression=safe,improved=improved)
    eligible=[a for a in ARMS if candidates[a]['no_dataset_qa_correctness_fcr_regression'] and
        (a=='baseline' or candidates[a]['improved'])]
    best=max(eligible,key=lambda a:(candidates[a]['mean_delta']['qa'],candidates[a]['mean_delta']['fcr'],
        candidates[a]['mean_delta']['correctness'],-len(candidates[a]['features'])))
    result=dict(final_arm=best,features=candidates[best]['features'],candidates=candidates,
        rule='Promotion additionally requires no dataset-level QA primary, EM/Accuracy or FCR@5 regression. Prefer QA gain, then FCR, then correctness, then fewer changes.',
        diagnostic_only=True)
    write(root/'final_decision.json',result);return result


def definitions(source,names):
    tree=ast.parse(source);found={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    return '\n\n'.join(ast.get_source_segment(source,found[name]) for name in names)+'\n'


def promote(root,selected):
    features=set(selected['features'])
    if not features:
        result=dict(status='baseline_retained',features=[],reason='No tested candidate met the per-dataset non-regression rule.')
        write(root/'production_promotion.json',result);return result
    source=(root/'source_snapshot/scripts/diagnostic_improvements.py').read_text()
    changes={};pack_names=[];stage=PROJECT/'src/bindpack/stages/pack/stage.py'
    stage_text=stage.read_text()
    if 'coverage_facts' in features:
        pack_names+=['fact_key','coverage_facts','FactCoverageSelector']
        stage_text=stage_text.replace('from .coverage_selector import CoverageSelector',
            'from .coverage_selector import CoverageSelector\nfrom .selection_improvements import FactCoverageSelector')
        assert 'self.coverage = CoverageSelector(runtime, self.scorer)' in stage_text
        stage_text=stage_text.replace('self.coverage = CoverageSelector(runtime, self.scorer)',
            'self.coverage = FactCoverageSelector(runtime, self.scorer)')
    if 'chain_package' in features:
        pack_names+=['complete_chain_package']
        stage_text=stage_text.replace('from .dag_package_selector import DAGPackageSelector',
            'from .dag_package_selector import DAGPackageSelector\nfrom .selection_improvements import complete_chain_package')
        assert '        return self.dag_selector.select(result, state)' in stage_text
        stage_text=stage_text.replace('        return self.dag_selector.select(result, state)',
            '        result = self.dag_selector.select(result, state)\n        return complete_chain_package(self.runtime, result, state, query, ids, scores)')
    if pack_names:
        header='''"""Selected source-bound selection improvements, validated in R=.60 diagnostics."""
from __future__ import annotations
import numpy as np
from .coverage_selector import CoverageSelector
from ..plan.relation_guard import relation_spec
from ..plan.answer_binder import bind_question
from ..plan.answer_validation import _typed_answer_reason
from .proof_scope import _specific, _literal, _title
from ...utils.evidence import normalized_text, branch_matches

'''
        changes[stage]=stage_text
        changes[PROJECT/'src/bindpack/stages/pack/selection_improvements.py']=header+definitions(source,pack_names)
    if 'plan_repair' in features:
        header='''"""Conservative recovery after the original planner attempts fail."""
from .plan_canonicalizer import canonicalize_plan_dependencies
from .plan_validator import prune_derived_comparison_nodes, validate_retrieval_plan, relation_faithfulness_error
from .validation import validate_plan

'''
        changes[PROJECT/'src/bindpack/stages/plan/plan_recovery.py']=header+definitions(source,['recover_plan'])
        p=PROJECT/'src/bindpack/stages/plan/dependency_planner.py';text=p.read_text()
        text=text.replace('from .validation import plan_layers, validate_plan',
            'from .validation import plan_layers, validate_plan\nfrom .plan_recovery import recover_plan')
        left,sep,right=text.rpartition('        return [], reason')
        assert sep
        text=left+'''        nodes, recovery = recover_plan(query, hops, diagnostics, self.config)
        diagnostics["deterministic_recovery"] = recovery
        if nodes:
            return nodes, None
        return [], reason'''+right
        changes[p]=text
    if 'route_fusion' in features:
        p=PROJECT/'src/bindpack/stages/route/path_set_consolidation.py';text=p.read_text()
        old='full_ppr_scores[pidx] = 0.7 * full_ppr_scores[pidx] + 0.3 * float(pscore)'
        assert old in text
        changes[p]=text.replace(old,'full_ppr_scores[pidx] = max(full_ppr_scores[pidx], 0.7 * full_ppr_scores[pidx] + 0.3 * float(pscore))')
    if 'hover_reader' in features:
        tree=ast.parse(source)
        value=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='HOVER_SYSTEM' for t in n.targets))
        template=[dict(role='system',content=value),dict(role='user',content='${prompt_user}')]
        changes[PROJECT/'src/bindpack/prompts/templates/rag_qa_hover.py']='"""Evidence-only atomic HoVer verification with the native two labels."""\n\nprompt_template = '+json.dumps(template,ensure_ascii=False,indent=4)+'\n'
    hashes=read(root/'production_base_hashes.json');backup={}
    for p,text in changes.items():
        rel=str(p.relative_to(PROJECT));ast.parse(text)
        if p.exists():
            if sha(p)!=hashes.get(rel):
                result=dict(status='not_applied_concurrent_source_changes',path=rel,features=sorted(features))
                write(root/'production_promotion.json',result);return result
            backup[rel]=p.read_text()
        else:backup[rel]=None
    assert read(PROJECT/'src/bindpack/configuration/defaults.json')['pack']['relation_weight']==.6
    write(root/'production_backup.json',backup)
    for p,text in changes.items():
        p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.py.tmp');tmp.write_text(text);tmp.replace(p)
    applied={str(p.relative_to(PROJECT)):sha(p) for p in changes}
    try:
        command=[sys.executable,'-B',str(root/'source_snapshot/scripts/check_component_production.py'),'--output',str(root)]
        subprocess.run(command,check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
    except Exception:
        for rel,original in backup.items():
            p=PROJECT/rel
            if sha(p)!=applied[rel]:continue
            if original is None:p.unlink()
            else:p.write_text(original)
        result=dict(status='rolled_back_parity_failure',features=sorted(features),attempted_changes=applied)
        write(root/'production_promotion.json',result);return result
    result=dict(status='applied_and_verified',features=sorted(features),changes=applied,
        cached_production_parity_questions=200,relation_weight=.6)
    write(root/'production_promotion.json',result);return result


def report(root,selected,promotion):
    screening=read(root/'selection_decision.json')
    lines=['# 固定0.60：单项筛选与组合结果','',
        '原始每个数据集50条，共200条，全部使用同一锁定清单。筛选与组合规则在新结果生成前记录于experiment.json。该样本按历史表现预选，仅作诊断，不是正式主实验或独立开发集。','',
        '| 单项 | 筛选结果 | 平均Δ回答F1 / Macro-F1 | 平均ΔEM / Accuracy | 平均ΔFCR@5 |',
        '|---|---|---:|---:|---:|']
    names=dict(retain='暂保留并进入组合',tradeoff='有升有降，未合入组合',discard_or_no_gain='剔除或无收益')
    for arm,d in screening['decisions'].items():
        m=d['mean_deltas'];lines.append(f'| {LABELS[arm]} | {names[d["status"]]} | {m["qa"]:+.6f} | {m["correctness"]:+.6f} | {m["fcr"]:+.6f} |')
    lines+=['','这些跨数据集平均值仅用于本诊断的预定义筛选，不作为跨任务通用F1主实验指标。各数据集的原始结果见comparison.md。','',
        '## 组合与最终候选','',f'进入组合：{screening["retained"]}。',
        f'最终选择：{selected["final_arm"]}；对应改动：{selected["features"]}。',
        f'源码处理状态：{promotion["status"]}。','',
        '| 数据集 | 版本 | Recall@5 | FCR@5 | EM / Accuracy | F1 / Macro-F1 |',
        '|---|---|---:|---:|---:|---:|']
    for ds in DATASETS:
        for arm in dict.fromkeys(('baseline','combined',selected['final_arm'])):
            m=read(root/'cases'/ds/arm/'report.json')['metrics']
            qa='ClassificationAccuracy' if ds=='hover' else 'ExactMatch';f1='ClassificationMacroF1' if ds=='hover' else 'F1'
            lines.append(f'| {ds} | {LABELS[arm]} | '+' | '.join(f'{m[k]:.6f}' for k in ('Recall@5','FCR@5',qa,f1))+' |')
    lines+=['','采用主源码还要求每个数据集的回答主指标、EM/Accuracy和FCR@5均不退化，至少一项提升；否则保留原0.60方法。若仅一个单项被保留，其组合与该单项完全相同，复用对应实际预测并记录来源；多个单项的组合重新运行。',
        '','基线首先缓存复现原0.60排名/提示/答案，再产生本轮共同Route、Plan输入。Pack与reader比较共用相同输入，Plan修复从同一pre-Plan状态执行，Route融合从同一pre-PathSet分数重建。权重、模型、并行和hop协议未改。HoVer reader收益是reader改动，不能算作检索贡献。',
        '','成本是私有历史缓存复用条件下的新增请求。每题、各阶段和资源账本保存在runtime/<dataset>/accounting；不作冷缓存端到端速度结论。','',
        '完整质量表：[comparison.md](comparison.md)；筛选：[selection_decision.json](selection_decision.json)；最终选择：[final_decision.json](final_decision.json)；源码状态：[production_promotion.json](production_promotion.json)。']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();root=args.output
    selected=decision(root);promotion=promote(root,selected);report(root,selected,promotion)
    write(root/'completed.ok',dict(status='complete',questions=200,single_versions=5,
        final_arm=selected['final_arm'],features=selected['features'],production=promotion['status'],relation_weight=.6))
    print(json.dumps(dict(final=selected['final_arm'],features=selected['features'],production=promotion['status']),ensure_ascii=False))


if __name__=='__main__':main()

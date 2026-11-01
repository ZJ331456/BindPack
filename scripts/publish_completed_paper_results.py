#!/usr/bin/env python3
"""Publish audited completed Reader/model cases and compile both paper entries."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from zoneinfo import ZoneInfo

PAPER=Path('/root/BindPack/docs/DASFAA_2027_latex')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def main(args):
    batch=args.batch_root.resolve();audit=batch/'paper_updates'/args.tag
    if audit.exists() and (audit/'completed.ok').exists():
        print('Paper publication already complete:',args.tag);return
    audit.mkdir(parents=True,exist_ok=True)
    shutil.copytree(PAPER,audit/'original',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    figures={str(p.relative_to(PAPER)):sha(p) for p in (PAPER/'figures').rglob('*') if p.is_file()}
    artifacts=[]
    for ds in ('hotpotqa','2wikimultihopqa','hover'):
        case=batch/'full_reader'/ds
        if not (case/'completed.ok').exists():continue
        report=read(case/'report.json');marker=read(case/'completed.ok')
        assert marker['query_count']==report['coverage']['sample_count']==report['coverage']['qa_n']==1000
        assert sha(case/'report.json')==marker['report_sha256']
        assert read(case/'audit.json')['status']=='passed'
        filename=f'full_{ds}_qwen3_qwenemb8b_reader.json'
        write(PAPER/'results'/filename,dict(dataset=ds,queries=1000,report=str(case/'report.json'),
            report_sha256=marker['report_sha256'],metrics=report['metrics'],
            qa_reused_exact_context=report['qa_reused_exact_context'],qa_new_inference=report['qa_new_inference']))
        artifacts.append(filename)
    configs=[('dense_qwen','model_factor_qwen38_flash_qwenemb8b_dense.json'),
             ('dense_bge','model_factor_qwen38_flash_bgem3_dense.json'),
             ('bindpack_bge','model_factor_qwen38_flash_bgem3.json')]
    for job,filename in configs:
        case=batch/'model_cases'/job
        if job=='bindpack_bge':case=case/'run/cases/musique/full'
        if not (case/'completed.ok').exists():continue
        report=read(case/'report.json');marker=read(case/'completed.ok');protocol=read(case/'protocol.json')
        assert marker['query_count']==report['coverage']['sample_count']==report['coverage']['qa_n']==1000
        assert sha(case/'report.json')==marker['report_sha256']
        assert read(case/'audit.json')['status']=='passed'
        write(PAPER/'results'/filename,dict(dataset='musique',queries=1000,report=str(case/'report.json'),
            report_sha256=marker['report_sha256'],metrics=report['metrics'],configuration=protocol['configuration']))
        artifacts.append(filename)
    if all((batch/'full_reader'/ds/'completed.ok').exists() for ds in ('hotpotqa','2wikimultihopqa','hover')):
        path=PAPER/'main.tex';text=path.read_text()
        text=text.replace('Full-data BindPack QA is evaluated on the saved MuSiQue rankings with the same Reader@5.',
                          'Full-data BindPack QA and HoVer verification use the saved Full rankings with Reader@5.')
        text=text.replace('Serial cost and QA evaluation uses', 'The separate serial cost/quality benchmark uses')
        path.write_text(text)
        path=PAPER/'RESULTS_UPDATE.md';text=path.read_text()
        text='\n'.join('- 四数据集Full均已固定原始前五篇证据补齐全部1,000题Reader评测。主结果表使用QA的EM/F1及HoVer的Accuracy/Macro-F1；消融组未测QA仍保留N/A。'
            if line.startswith('- 原Full和这些消融最初只运行检索。') else line for line in text.split('\n'))
        path.write_text(text)
    source=audit/'source';shutil.copytree(PAPER,source,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    env=dict(os.environ);env.pop('PYTHONPATH',None)
    with (audit/'sync.log').open('w') as log:
        subprocess.run(['/root/anaconda3/bin/python','-B',str(source/'sync_results.py')],cwd=source,
            env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    old_plot=read(PAPER/'figures/plot_data.json');new_plot=read(source/'figures/plot_data.json')
    for key in ('coverage','closure','cost'):assert old_plot[key]==new_plot[key]
    shutil.copytree(PAPER/'figures',source/'figures',dirs_exist_ok=True)
    build={};env['XDG_CACHE_HOME']='/root/.cache/bindpack_english_tex_cache'
    for kind,name,target in [('main','main',PAPER/'main.pdf'),('local','main-local',PAPER/'local_only/main-local.pdf')]:
        out=audit/kind;out.mkdir(exist_ok=True)
        cmd=['/root/.cache/tectonic-bindpack/tectonic','-c','minimal','-C','-b',
            'http://127.0.0.1:18876/tlextras-2022.0r0.tar','--keep-logs','--outdir',str(out),name+'.tex']
        with (out/'compile.log').open('w') as log:
            subprocess.run(cmd,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        text=(out/(name+'.log')).read_text(errors='replace')
        assert not re.search(r'Overfull \\[hv]box|undefined references|Undefined control sequence',text,re.I)
        info=subprocess.check_output(['pdfinfo',str(out/(name+'.pdf'))],text=True)
        pages=int(re.search(r'Pages:\s+(\d+)',info).group(1))
        assert pages==({'main':16,'local':19}[kind]),(kind,pages)
        build[kind]=dict(pages=pages,pdf_sha256=sha(out/(name+'.pdf')))
    for path in (source/'results').rglob('*'):
        if path.is_file() and path.suffix in ('.tex','.json'):
            relative=path.relative_to(source);target=PAPER/relative
            if not target.exists() or path.read_bytes()!=target.read_bytes():shutil.copy2(path,target)
    for kind,name,target in [('main','main',PAPER/'main.pdf'),('local','main-local',PAPER/'local_only/main-local.pdf')]:
        temporary=target.with_name(target.name+'.tmp');shutil.copy2(audit/kind/(name+'.pdf'),temporary);temporary.replace(target)
    shutil.copy2(audit/'main/main.log',PAPER/'main.log')
    assert all(sha(PAPER/relative)==value for relative,value in figures.items())
    value=dict(status='complete',published_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        artifacts=artifacts,figures_unchanged=True,builds=build)
    validation_path=PAPER/'results/update_validation.json'
    validation=read(validation_path) if validation_path.exists() else {}
    validation.update(latest_automatic_publication=value,submission_pages=build['main']['pages'],
        local_pages=build['local']['pages'],framework_unchanged=True,all_figures_unchanged=True,
        full_qa_queries_per_dataset=read(PAPER/'results/source_manifest.json')['full_qa_queries_per_dataset'])
    write(validation_path,validation)
    write(audit/'completed.ok',value)
    with (PAPER/'RESULTS_UPDATE.md').open('a') as stream:
        stream.write('\n\n'+value['published_at']+'：自动核验完成报告并同步 '+', '.join(artifacts)+
            '。原始检索和模型报告保持不变，完整题数均为1000；Reader复用仅允许相同提示/证据/模型配置。'+
            '主结果表HoVer采用Accuracy与ClassificationMacroF1；未将标签Accuracy代替Macro-F1。'+
            '提交版16页、本地版19页、无溢出，所有图件保持原样。审计：'+str(audit/'completed.ok')+'。\n')
    print('PAPER UPDATED:',json.dumps(value,ensure_ascii=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--batch-root',type=Path,required=True)
    parser.add_argument('--tag',required=True);main(parser.parse_args())

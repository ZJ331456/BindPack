"""Exercise the CPU study's six-arm parity/audit path on a synthetic fixture."""
from dataclasses import asdict
import gzip,hashlib,json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
import numpy as np
from transformers import AutoTokenizer
from bindpack import BindPackConfig
from bindpack.prompts.prompt_template_manager import PromptTemplateManager
from bindpack.stages.pack.stage import PackStage
from run_pack_weight_diagnostic import freeze_pack_input
from test_evidence_dag_package import Engine,chain
import run_joint_component_study as study


class StudyRunnerTests(TestCase):
    def test_all_controls_pass_the_same_input_and_bank_checks_without_models(self):
        engine=Engine();state=chain(engine)
        for c in state['evidence_candidates'].values():
            c.update(sources=[],base_score=0.)
            for p in c['verified']:p['quality']=.95
        cfg=BindPackConfig()
        runtime=SimpleNamespace(cfg=cfg,_document=engine._document,_trace=lambda:{},
            _joint_visible=lambda q,ids:({d:engine._document(d) for d in ids},{}))
        ids=np.arange(220);scores=np.linspace(1,0,220);query='Sample Region'
        result=PackStage(runtime).finalize(query,ids,scores,{},state)
        full=result[0][:200].tolist();diag=result[2]['budgeted_selection']
        ref=dict(query_index=0,full_ids=full,s0=diag['reference_doc_ids'],domain=diag['candidate_domain'],
            objective=diag['objective'],rho=diag['soft_weight_effective'],
            gold_hashes=[hashlib.sha256(engine.documents[15].encode()).hexdigest()],
            content_by_id={str(d):hashlib.sha256(engine.documents[d].encode()).hexdigest() for d in full})
        with TemporaryDirectory() as folder:
            root=Path(folder);path=root/'inputs/hotpotqa'/study.token('fixture');path.parent.mkdir(parents=True)
            raw=freeze_pack_input(query,ids,scores,state)
            event=dict(input=raw,input_sha256=study.digest(raw),sample_id='fixture',query_index=0,
                source_progress_sha256='fixture-only',recovery_version=4)
            with gzip.open(path,'wt') as stream:json.dump(event,stream)
            study.ENV.clear();study.ENV.update(root=root,ds='hotpotqa',source_sha='fixture-only',
                store=SimpleNamespace(texts=engine.documents),refs={'fixture':ref},config=asdict(cfg),
                tokenizer=AutoTokenizer.from_pretrained('/root/models/Qwen3-8B',local_files_only=True),
                reader=SimpleNamespace(prompt_template_manager=PromptTemplateManager(role_mapping={'system':'system','user':'user','assistant':'assistant'})))
            try:
                rows=study.cpu_query(('fixture','pilot'))
                self.assertEqual([r['arm'] for r in rows],list(study.ARMS))
                self.assertTrue(all(r['full_parity_passed'] and not r['new_llm_calls'] and not r['new_embedding_calls'] for r in rows))
                self.assertEqual(rows[0]['selected_doc_ids'],full[:5])
                self.assertEqual(rows[1]['selected_doc_ids'],ref['s0'])
            finally:study.ENV.clear()

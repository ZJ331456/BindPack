"""CPU checks for final commands, frozen indexes and benchmark hops."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))

import eval_dataset
from utils import final_workflow as final
from utils import multi_dataset_retrieval as runner
from utils import multi_dataset_retrieval_nv2 as nv2
from utils import multi_dataset_retrieval_support_semantic_veto as semantic
from utils import workflow_artifacts as artifacts


class MultiDatasetRunnerTests(unittest.TestCase):
    def test_only_final_method_commands_preserve_budgets(self):
        args = SimpleNamespace(python='/python', hippo_root='/hippo', smoke=False,
                               openie_strict=False, openie_prompt_version='optimized',
                               llm_base_url='http://local/v1')
        for name in runner.DATASETS:
            metadata = Path('/output/metadata') / name
            index = Path('/output/shared_indexes') / name
            builder, cases = runner.commands(args, metadata, index, Path('/datasets'), name)
            self.assertEqual(set(cases), {semantic.CASE_NAME})
            for command in (builder, *cases.values()):
                for key, value in (('dataset', name), ('sample_size', '0'),
                                   ('embedding_batch_size', '4'), ('openie_max_workers', '8'),
                                   ('llm_prefetch_workers', '8')):
                    self.assertEqual(command[command.index('--' + key) + 1], value)
            self.assertEqual(builder[builder.index('--save_dir') + 1], str(index))
            self.assertEqual(builder[builder.index('--eval_mode') + 1], 'index_only')
            command = cases[semantic.CASE_NAME]
            self.assertIn('--reuse_index', command)
            parsed = eval_dataset.build_parser().parse_args(command[4:])
            self.assertEqual(parsed.hop_source, 'benchmark')
            self.assertEqual(parsed.improvement_stage, 4)
            self.assertEqual(parsed.max_new_tokens, 2048)
            self.assertEqual(set(parsed.evidence_improvements.split(',')), set(final.FLAGS))
            self.assertEqual(parsed.result_top_k, 10)
            self.assertEqual(parsed.candidate_output_top_k, 200)

    def test_formal_profile_command_contains_no_deleted_settings(self):
        args = SimpleNamespace(python='/python', llm_base_url='http://local/v1')
        context = {'dataset': 'hotpotqa', 'manifest': {
            'data_path': '/data/hotpotqa.json', 'corpus_path': '/data/hotpotqa_corpus.json',
            'runtime': {'embedding_batch_size': 4, 'embedding_model_name': '/root/models/NV-Embed-v2'},
        }}
        command = final.profile_command(args, context, Path('/tmp/case'), [0, 1], semantic.VARIANT)
        parsed = eval_dataset.build_parser().parse_args(command[4:])
        self.assertEqual(parsed.embedding_batch_size, 4)
        self.assertEqual(parsed.llm_prefetch_workers, 8)
        self.assertEqual(parsed.evidence_plan_validation, 'canonical_refs')
        self.assertEqual(parsed.evidence_plan_routing, 'question_structure')
        self.assertFalse(any('adaptive' in item or 'mpce' in item or 'binding_validation' in item
                             or 'selection_top_k' in item or 'hop_force_max' in item for item in command))

    def test_invalid_removed_profiles_are_rejected(self):
        with self.assertRaises(ValueError):
            final.Variant('unsupported', ('adaptive',))
        parser = eval_dataset.build_parser()
        with contextlib.redirect_stderr(io.StringIO()):
            for options in (['--hop_source', 'estimated'], ['--improvement_stage', '0'],
                            ['--use_mpce'], ['--evidence_adaptive_mode', 'both']):
                with self.assertRaises(SystemExit):
                    parser.parse_args(['--dataset', 'hotpotqa', *options])

    def test_private_index_omits_build_journal_and_preserves_initial_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            source = out / 'shared_index'
            model = artifacts.MODEL_DIR
            for name in artifacts.ASSETS:
                path = source / model / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name)
            (source / 'llm_cache').mkdir()
            (source / 'llm_cache/cache.json').write_text('source cache')
            (source / model / 'openie_progress.sqlite').write_text('large journal')
            cache = out / 'initial_cache'
            cache.mkdir()
            (cache / 'cache.json').write_text('frozen cache')
            context = {'clone_source': source, 'cache': cache,
                       'cache_hashes': artifacts.cache_hashes(cache), 'manifest': {
                           'model_dir': model, 'source_asset_sha256': artifacts.asset_hashes(source),
                       }}
            case = out / 'case'
            final.initialize_case(context, case)
            self.assertFalse((case / 'index' / model / 'openie_progress.sqlite').exists())
            for name in artifacts.ASSETS:
                self.assertEqual((case / 'index' / model / name).stat().st_ino,
                                 (source / model / name).stat().st_ino)
            private = case / 'index/llm_cache/cache.json'
            self.assertEqual(private.read_text(), 'frozen cache')
            private.write_text('private request')
            self.assertEqual((cache / 'cache.json').read_text(), 'frozen cache')
            self.assertEqual((source / 'llm_cache/cache.json').read_text(), 'source cache')

    def test_frozen_context_does_not_require_prior_retrieval_case(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            source = out / 'shared_indexes/hotpotqa'
            source.mkdir(parents=True)
            for name in artifacts.ASSETS:
                path = source / artifacts.MODEL_DIR / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name)
            data = out / 'hotpotqa.json'
            corpus = out / 'hotpotqa_corpus.json'
            data.write_text('[]')
            corpus.write_text('[]')
            metadata = out / 'metadata/hotpotqa'
            metadata.mkdir(parents=True)
            cache = metadata / 'initial_llm_cache'
            cache.mkdir()
            (cache / 'cache.json').write_text('frozen cache')
            manifest = {'source_index': str(source), 'model_dir': artifacts.MODEL_DIR,
                        'data_path': str(data), 'corpus_path': str(corpus),
                        'data_sha256': artifacts.sha256(data), 'corpus_sha256': artifacts.sha256(corpus),
                        'source_asset_sha256': artifacts.asset_hashes(source),
                        'initial_cache_sha256': artifacts.cache_hashes(cache)}
            (metadata / 'manifest.json').write_text(json.dumps(manifest))
            with patch.object(artifacts, 'validated_dataset', return_value=([{'question': 'q'}], {'d'}, [2])):
                context = final.context_for(out, 'hotpotqa', out / 'support')
            self.assertEqual(context['clone_source'], source)
            self.assertEqual(artifacts.cache_hashes(context['cache']), manifest['initial_cache_sha256'])
            self.assertFalse((out / 'cases').exists())

    def test_smoke_keeps_original_gold_and_question_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source, output = base / 'source', base / 'run_smoke'
            source.mkdir()
            runner.layout(output, 'hotpotqa')
            rows = [{'title': f'Doc{i}', 'text': 'A source passage with intact words. ' * 4}
                    for i in range(205)]
            samples = [{'_id': f'q{i}', 'question': f'Question {i}?', 'type': 'bridge',
                        'supporting_facts': [[f'Doc{i}', 0]],
                        'context': [[f'Doc{i}', [rows[i]['text']]]]} for i in range(2)]
            (source / 'hotpotqa.json').write_text(json.dumps(samples))
            (source / 'hotpotqa_corpus.json').write_text(json.dumps(rows))
            directory = runner.smoke_dataset(output, source, 'hotpotqa')
            self.assertEqual(json.loads((directory / 'hotpotqa.json').read_text()), samples)
            _, docs, hops = artifacts.validated_dataset(directory / 'hotpotqa.json',
                                                         directory / 'hotpotqa_corpus.json')
            self.assertEqual(len(docs), 20)
            self.assertEqual(hops, [2, 2])

    def test_missing_benchmark_hops_fails_before_model_construction(self):
        args = eval_dataset.build_parser().parse_args(['--dataset', 'musique'])
        with patch.object(eval_dataset, 'load_json', side_effect=[[{'question': 'q'}], []]), \
                patch.object(eval_dataset, 'get_benchmark_hops', return_value=None), \
                patch.object(eval_dataset, 'BindPack') as rag:
            with self.assertRaisesRegex(ValueError, 'benchmark hop labels/priors are missing'):
                eval_dataset.run_eval('musique', args)
        rag.assert_not_called()

    def test_single_hop_priors_are_preserved(self):
        from eval_utils import get_benchmark_hops
        self.assertEqual(get_benchmark_hops([{'question': 'q'}], 'nq'), [1])
        self.assertEqual(get_benchmark_hops([{'question': 'q'}], 'popqa'), [1])

    def test_eval_constructs_retained_config_and_exports_with_mock_runtime(self):
        from bindpack.utils.misc_utils import QuerySolution
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            corpus = [{'title': f'Doc{i}', 'text': f'Intact source passage {i}.'} for i in range(20)]
            sample = {'_id': 'q0', 'question': 'Which sources provide the answer?',
                      'type': 'bridge', 'answer': 'A', 'supporting_facts': [['Doc0', 0]],
                      'context': [[row['title'], [row['text']]] for row in corpus]}
            data_path, corpus_path = out / 'hotpotqa.json', out / 'hotpotqa_corpus.json'
            data_path.write_text(json.dumps([sample]))
            corpus_path.write_text(json.dumps(corpus))
            args = eval_dataset.build_parser().parse_args([
                '--dataset', 'hotpotqa', '--data_path', str(data_path), '--corpus_path', str(corpus_path),
                '--save_dir', str(out / 'index'), '--output', str(out / 'result.json'),
                '--openie_max_workers', '8',
            ])
            documents = [artifacts.passage_text(row) for row in corpus]
            solution = QuerySolution(sample['question'], documents, [1.0 / (i + 1) for i in range(20)])
            fake = Mock()
            fake.retrieve.return_value = ([solution], {'Recall@5': 1.0})
            fake.get_retrieval_diagnostics.return_value = {}
            fake.all_retrieval_time = 0.0
            fake.llm_model.get_request_stats.return_value = {}
            with patch.object(eval_dataset, 'BindPack', return_value=fake) as constructor:
                eval_dataset.run_eval('hotpotqa', args)
            config = constructor.call_args.kwargs['global_config']
            self.assertEqual(config.improvement_stage, 4)
            self.assertEqual(set(config.evidence_improvements.split(',')), set(final.FLAGS))
            fake.set_query_hop_overrides.assert_called_once_with(hops=[2], source='benchmark')
            result = json.loads((out / 'result.json').read_text())
            self.assertEqual(result['hop_distribution'], {'2': 1})
            self.assertEqual(len(result['results'][0]['docs']), 10)
            self.assertEqual(len(result['results'][0]['candidate_docs']), 20)

    def test_nv2_workflow_runs_only_three_final_cases(self):
        self.assertEqual(nv2.METHODS, (semantic.CASE_NAME,))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            out = root / 'outputs/final_nv2'
            out.mkdir(parents=True)
            vlog = out / 'vllm.log'
            vlog.write_text('')
            args = SimpleNamespace(out_root=str(out), smoke=False, embedding_provider='nvembed',
                embedding_model='/root/models/NV-Embed-v2', embedding_batch_size=4,
                vllm_log=str(vlog), llm_base_url='http://local/v1', openie_strict=False,
                openie_prompt_version='optimized', openie_validation_mode='structural',
                reuse_indexes=False)
            with patch.object(runner, 'ROOT', root), \
                    patch.object(runner, 'run', return_value=0) as build, \
                    patch.object(semantic, 'run', return_value=0) as retrieve, \
                    patch.object(runner, 'frozen_hashes', return_value={'stable': 'hash'}), \
                    patch.object(nv2, 'retain_request_logs'), \
                    patch.object(nv2, 'publish_summary', return_value=3):
                self.assertEqual(nv2.run(args), 0)
            build.assert_called_once()
            self.assertTrue(build.call_args.args[0].indexes_only)
            retrieve.assert_called_once()
            self.assertEqual(json.loads((out / 'completed.ok').read_text())['validated_case_count'], 3)

    def test_nv2_reuse_ours_never_calls_index_builder_or_overwrites_old_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            out = root / 'outputs/final_nv2'
            for dataset in runner.DATASETS:
                (out / 'shared_indexes' / dataset).mkdir(parents=True)
                metadata = out / 'metadata' / dataset
                metadata.mkdir(parents=True)
                (metadata / 'manifest.json').write_text(json.dumps({
                    'index_build_status': 'ready', 'runtime': {
                        'embedding_model_name': '/root/models/NV-Embed-v2',
                        'embedding_provider': 'nvembed', 'embedding_batch_size': 4,
                    }}))
            vlog = out / 'vllm.log'
            vlog.write_text('')
            (out / 'completed.ok').write_text('old marker')
            (out / 'comparison.json').write_text('old summary')
            args = SimpleNamespace(out_root=str(out), smoke=False, embedding_provider='nvembed',
                embedding_model='/root/models/NV-Embed-v2', embedding_batch_size=4,
                vllm_log=str(vlog), llm_base_url='http://local/v1', openie_strict=False,
                openie_prompt_version='optimized', openie_validation_mode='structural',
                reuse_indexes=True, case_name='ours')
            with patch.object(runner, 'ROOT', root), \
                    patch.object(runner, 'run') as build, \
                    patch.object(semantic, 'run', return_value=0) as retrieve, \
                    patch.object(runner, 'frozen_hashes', return_value={'stable': 'hash'}), \
                    patch.object(nv2, 'retain_request_logs'), \
                    patch.object(nv2, 'publish_summary', return_value=3):
                self.assertEqual(nv2.run(args), 0)
            build.assert_not_called()
            self.assertEqual(retrieve.call_args.args[0].case_name, 'ours')
            self.assertEqual((out / 'completed.ok').read_text(), 'old marker')
            self.assertEqual((out / 'comparison.json').read_text(), 'old summary')
            self.assertTrue((out / 'ours_completed.ok').is_file())


if __name__ == '__main__':
    unittest.main()

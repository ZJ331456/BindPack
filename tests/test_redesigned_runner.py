"""Protect causal isolation, diverse preflight inputs and durable run reporting."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from bindpack import BindPackConfig
from bindpack.evaluation.dataset_adapter import get_benchmark_hops
import run_redesigned_ablations as runner


class RedesignedRunnerTests(TestCase):
    def test_serial_timing_keeps_shared_upstream_once_and_qa_separate(self):
        windows=[dict(full_seconds=12.,parent_seconds=0.,pack_seconds={},qa_seconds={'full':2.}),
            dict(full_seconds=0.,parent_seconds=0.,pack_seconds={'wo_source_binding':3.},qa_seconds={'wo_source_binding':4.}),
            dict(full_seconds=0.,parent_seconds=0.,pack_seconds={'wo_joint_selection':.1},qa_seconds={'wo_joint_selection':5.}),
            dict(full_seconds=0.,parent_seconds=6.,pack_seconds={},qa_seconds={'wo_parent_answer_propagation':7.})]
        self.assertEqual(runner.case_timing_totals(windows,'wo_source_binding'),dict(
            full_upstream_seconds=12.,pack_replay_seconds=3.,parent_reexecution_seconds=0.,reader_window_seconds=4.))
        self.assertEqual(runner.case_timing_totals(windows,'wo_parent_answer_propagation'),dict(
            full_upstream_seconds=12.,pack_replay_seconds=0.,parent_reexecution_seconds=6.,reader_window_seconds=7.))
        self.assertEqual(runner.case_timing_totals(windows,'full')['reader_window_seconds'],2.)
        self.assertTrue(all(v==0 for v in runner.case_timing_totals([], 'full').values()))

    def test_case_parallelism_is_frozen_per_case_and_bounded(self):
        with TemporaryDirectory() as folder:
            root=Path(folder)
            defaults=runner.case_parallelism(root,32,16)
            self.assertTrue(all(v==dict(llm_workers=32,reader_workers=16) for v in defaults.values()))
            runner.write(root/'concurrency_profile.json',dict(cases={
                'full':dict(llm_workers=32,reader_workers=16),
                'wo_joint_selection':dict(llm_workers=16,reader_workers=32)}))
            profile=runner.case_parallelism(root,32,16)
            self.assertEqual(profile['wo_joint_selection'],dict(llm_workers=16,reader_workers=32))
            self.assertEqual(profile['full'],defaults['full'])
            runner.write(root/'concurrency_profile.json',dict(cases={'full':dict(llm_workers=64,reader_workers=32)}))
            with self.assertRaisesRegex(ValueError,'Invalid concurrency profile'):runner.case_parallelism(root,32,16)

    def test_complete_cases_are_strictly_serial_and_require_validation_before_next(self):
        completed = set()
        queue = runner.serial_case_order(runner.ARMS, completed.__contains__)
        events = []
        for expected in runner.ARMS:
            arm = next(queue)
            self.assertEqual(arm, expected)
            events.append(('start',arm))
            # Model all windows and QA finishing before the completion marker.
            completed.add(arm)
            events.append(('complete',arm))
        with self.assertRaises(StopIteration): next(queue)
        self.assertEqual(events, [(event,arm) for arm in runner.ARMS for event in ('start','complete')])
        blocked = runner.serial_case_order(runner.ARMS, lambda arm: False)
        self.assertEqual(next(blocked),'full')
        with self.assertRaisesRegex(RuntimeError,'has not been validated'): next(blocked)

    def test_each_arm_changes_only_its_mechanism_and_preserves_baseline_parameters(self):
        base = BindPackConfig.from_file(source_bound_eligibility=True,
            pack={"relation_weight": .37}, llm_prefetch_workers=16)
        before = asdict(base)
        switches = {
            "full": ("full", "none"),
            "wo_source_binding": ("wo_source_binding", "none"),
            "wo_joint_selection": ("wo_joint_selection", "none"),
            "flat_dependency_support": ("full", "flat_dependency_support"),
            "singleton_search": ("full", "singleton_search"),
            "wo_parent_answer_propagation": ("wo_parent_answer_propagation", "none"),
        }
        self.assertEqual(set(runner.ARMS), set(switches))
        for arm in runner.ARMS:
            with self.subTest(arm=arm):
                config = runner.arm_config(base, arm)
                self.assertEqual((config.mechanism_ablation,
                    config.pack_mechanism_ablation), switches[arm])
                self.assertLessEqual(int(config.mechanism_ablation != "full")
                    + int(config.pack_mechanism_ablation != "none"), 1)
                actual = asdict(config)
                expected = deepcopy(before)
                for key in ("mechanism_ablation", "pack_mechanism_ablation"):
                    actual.pop(key); expected.pop(key)
                self.assertEqual(actual, expected)
                config.pack.relation_weight = .91
                self.assertEqual(asdict(base), before)

    def test_preflight_prioritizes_native_hop_and_label_combinations(self):
        def hover(hops, label):
            return dict(num_hops=hops, label=label,
                supporting_facts=[[f"Document {i}", 0] for i in range(hops)])

        data = [hover(3, "SUPPORTED"), hover(3, "SUPPORTED"),
            hover(4, "SUPPORTED"), hover(3, "NOT_SUPPORTED"),
            hover(4, "NOT_SUPPORTED"), hover(4, "NOT_SUPPORTED")]
        before = deepcopy(data)
        selected = runner.selected_indices(data, "hover", 4)
        hops = get_benchmark_hops(data, "hover")
        self.assertEqual({(hops[i], data[i]["label"]) for i in selected},
            {(3, "SUPPORTED"), (4, "SUPPORTED"),
             (3, "NOT_SUPPORTED"), (4, "NOT_SUPPORTED")})
        self.assertEqual(len(selected), len(set(selected)))
        self.assertEqual(runner.selected_indices(data, "hover", 0), list(range(6)))
        self.assertEqual(runner.selected_indices(data, "hover", 100),
            runner.selected_indices(data, "hover", 6))
        self.assertEqual(data, before)

    def test_preflight_covers_musique_depths_and_two_wiki_four_support_bucket(self):
        fixtures = {
            "musique": [dict(question_decomposition=[{"question": "Which source?"}]
                * hops) for hops in (2, 2, 3, 2, 4)],
            "2wikimultihopqa": [dict(supporting_facts=[[f"Document {i}", 0]
                for i in range(count)]) for count in (2, 2, 5, 4)],
        }
        for dataset, data in fixtures.items():
            with self.subTest(dataset=dataset):
                hops = get_benchmark_hops(data, dataset)
                selected = runner.selected_indices(data, dataset, len(set(hops)))
                self.assertEqual({hops[i] for i in selected}, set(hops))

    def test_checkpoint_roundtrip_then_corruption_is_rejected_before_loading(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "upstream/0000_0002.pkl.gz"
            payload = dict(indices=[0, 3], questions=["First?", "Second?"],
                full_seconds=1.25, state={"bindings": {"v1": "An answer"}})
            runner.save_window(path, payload)
            self.assertEqual(runner.load_window(path), payload)
            # A damaged gzip must produce the explicit integrity failure,
            # rather than reaching decompression or pickle execution.
            path.write_bytes(path.read_bytes()[:12])
            with self.assertRaisesRegex(ValueError, "Corrupt upstream checkpoint"):
                runner.load_window(path)

    def test_summary_reports_signed_percentage_point_changes_and_preserves_missing_metrics(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            def report(dataset, arm, metrics):
                runner.write(root / "cases" / dataset / arm / "report.json",
                    dict(metrics=metrics, sample_alignment={"sample_count": 17}))
            report("musique", "full", {"Recall@5": .8, "FCR@5": .6,
                "ExactMatch": .5, "F1": .7, "DCC@5": None, "ZeroMetric": 0.})
            report("musique", "wo_source_binding", {"Recall@5": .75,
                "FCR@5": .625, "ExactMatch": .5, "F1": None,
                "DCC@5": .2, "ZeroMetric": .1})
            report("hover", "full", {"Recall@5": .7, "FCR@5": .4,
                "ClassificationAccuracy": .8, "ClassificationMacroF1": .75,
                "DCC@5": None})
            report("hover", "singleton_search", {"Recall@5": .65, "FCR@5": .35,
                "ClassificationAccuracy": .75, "ClassificationMacroF1": .725,
                "DCC@5": None})
            # A partially completed dataset cannot borrow another dataset's
            # Full baseline when computing the differences.
            report("hotpotqa", "wo_joint_selection", {"Recall@5": .9})
            results = runner.summarize(root)
            rows = {(row["dataset"], row["arm"]): row for row in results}
            delta = rows[("musique", "wo_source_binding")]["delta_from_full_pp"]
            for key, expected in {"Recall@5": -5., "FCR@5": 2.5,
                    "ExactMatch": 0., "ZeroMetric": 10.}.items():
                self.assertAlmostEqual(delta[key], expected)
            self.assertNotIn("DCC@5", delta)
            self.assertNotIn("F1", delta)
            self.assertEqual(rows[("hotpotqa", "wo_joint_selection")]["delta_from_full_pp"], {})
            self.assertAlmostEqual(rows[("hover", "singleton_search")]
                ["delta_from_full_pp"]["ClassificationMacroF1"], -2.5)
            self.assertEqual(rows[("musique", "wo_source_binding")]["n"], 17)
            self.assertEqual(len(runner.read(root / "summary.json")["results"]), len(results))
            text = (root / "RESULTS.md").read_text()
            self.assertIn("|hover|Singleton Search|17|65.00|35.00|75.00|72.50|N/A|", text)

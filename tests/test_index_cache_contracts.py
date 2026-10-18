"""Current request cache identity and source-paired OpenIE repair contracts."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bindpack import BindPackConfig
from bindpack.index.openie.openie_quality import normalize_repair_payload, extract_triple_list
from bindpack.index.shared_index_builder import quality_profile
from bindpack.llm.openai_gpt import CacheOpenAI, cache_response


class CurrentContractTests(unittest.TestCase):
    def test_real_llm_initialization_serializes_nested_config_without_mutating_it(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"OPENAI_API_KEY": "EMPTY"}):
            config = BindPackConfig(save_dir=temporary)
            before = asdict(config)
            model = CacheOpenAI.from_experiment_config(config)
            try:
                serialized = json.loads(str(model.llm_config))
                self.assertEqual(serialized["pack"], before["pack"])
                self.assertEqual(serialized["route"], before["route"])
                self.assertEqual(asdict(config), before)
                self.assertFalse(hasattr(config, "generate_params"))
                self.assertFalse(hasattr(config, "max_retries"))
                model.llm_config.pack["coverage_weight"] = 99.
                self.assertEqual(config.pack.coverage_weight, .45)
            finally:
                model.openai_client.close()

    def test_current_cache_preserves_repeated_requests_and_separates_generation_budgets(self):
        calls = []

        @cache_response
        def infer(instance, messages, **kwargs):
            calls.append((messages, kwargs))
            return 'fresh response', {'finish_reason': 'stop'}

        with tempfile.TemporaryDirectory() as temporary:
            model = SimpleNamespace(cache_file_name=str(Path(temporary) / 'cache.sqlite'),
                                    llm_name='qwen3-8b', llm_base_url='http://example/v1',
                                    llm_config=SimpleNamespace(generate_params={
                                        'model': 'qwen3-8b', 'temperature': 0., 'max_tokens': 100}),
                                    _llm_stats_lock=threading.Lock())
            prompt = [{'role': 'user', 'content': 'A question'}]
            self.assertFalse(infer(model, prompt)[2])
            self.assertTrue(infer(model, prompt)[2])
            self.assertFalse(infer(model, prompt, max_tokens=200)[2])
            self.assertEqual(len(calls), 2)
            self.assertEqual(model.llm_cache_hit_count, 1)
            self.assertEqual(model.llm_cache_miss_count, 2)

    def test_unversioned_non_qwen_cache_is_not_silently_reused(self):
        calls = []

        @cache_response
        def infer(instance, messages, **kwargs):
            calls.append(messages)
            return 'current response', {'finish_reason': 'stop'}

        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / 'cache.sqlite'
            prompt = [{'role': 'user', 'content': 'A question'}]
            params = {'model': 'example-model', 'seed': 42, 'temperature': 0., 'max_tokens': 100}
            old_key = hashlib.sha256(json.dumps({'messages': prompt, **params},
                                                sort_keys=True, default=str).encode()).hexdigest()
            with sqlite3.connect(cache) as connection:
                connection.execute('CREATE TABLE cache (key TEXT PRIMARY KEY, message TEXT, metadata TEXT)')
                connection.execute('INSERT INTO cache VALUES (?, ?, ?)',
                                   (old_key, 'stale response', json.dumps({'finish_reason': 'stop'})))
            model = SimpleNamespace(cache_file_name=str(cache), llm_name='example-model',
                                    llm_base_url='http://example/v1',
                                    llm_config=SimpleNamespace(generate_params=params),
                                    _llm_stats_lock=threading.Lock())
            response = infer(model, prompt)
            self.assertEqual(response[0], 'current response')
            self.assertFalse(response[2])
            self.assertEqual(calls, [prompt])

    def test_repair_quotes_are_structurally_paired_with_their_triples(self):
        triple = ['Alpha', 'wrote', 'Work X']
        quote = 'Alpha wrote Work X.'
        self.assertEqual(normalize_repair_payload({'triples': [
            {'triple': triple, 'support_quote': quote}], 'status': 'complete'}),
                         ([triple], [quote], 'complete'))
        self.assertEqual(normalize_repair_payload({'triples': [], 'status': 'empty'}), ([], [], 'empty'))
        with self.assertRaises(ValueError):
            normalize_repair_payload({'triples': [triple], 'support_quotes': [quote]})

    def test_basic_triple_parsing_and_current_quality_profile_remain_supported(self):
        triple = [['Alpha', 'wrote', 'Work X']]
        self.assertEqual(extract_triple_list(json.dumps(triple)), triple)
        self.assertEqual(extract_triple_list(json.dumps({'triples': triple})), triple)
        self.assertEqual(quality_profile(validation_mode='structural')['schema_version'], 1)
        self.assertEqual(quality_profile(validation_mode='source_verified')['schema_version'], 3)
        with self.assertRaises(TypeError):
            quality_profile(legacy=True)


if __name__ == '__main__':
    unittest.main()

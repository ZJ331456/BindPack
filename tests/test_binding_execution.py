"""CPU checks for online binding counterfactuals and literal equivalence."""

import copy
from concurrent.futures import ThreadPoolExecutor
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ModuleType("binding_execution_test_package")
PACKAGE.__path__ = [str(ROOT / "src/bindpack")]
sys.modules[PACKAGE.__name__] = PACKAGE
SPEC = importlib.util.spec_from_file_location(
    PACKAGE.__name__ + ".experiments.binding_execution", ROOT / "src/bindpack/experiments/binding_execution.py")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
BASE = sys.modules[PACKAGE.__name__ + ".evidence.evidence_retrieval"].EvidenceRetrieval


class Runtime(BASE):
    def __init__(self, responses):
        self.responses = iter(responses)
        self.messages = []
        self._verification_diagnostics = {}

    def _infer_object(self, messages):
        self.messages.append(copy.deepcopy(messages))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response if isinstance(response, tuple) else (response, None)


def payload(answer="Rome", evidence="Alpha was born in Rome.", doc_id="D0", confidence=.9):
    return {"hypotheses": [{"answer": answer, "evidence": evidence,
                            "doc_id": doc_id, "confidence": confidence}]}


DOCUMENTS = {0: "Alpha\nAlpha was born in Rome."}
QUESTION = "Where was Alpha born?"


class BindingExecutionTests(unittest.TestCase):
    def test_literal_keeps_original_method_and_exact_outputs(self):
        original = Runtime([payload()])
        expected = original._verify(QUESTION, "place", DOCUMENTS)
        literal = Runtime([payload()])
        before = literal._verify
        self.assertIs(MODULE.install_binding_control(literal, "literal"), literal)
        self.assertEqual(literal._verify, before)
        self.assertEqual(literal._verify(QUESTION, "place", DOCUMENTS), expected)
        self.assertEqual(literal.messages, original.messages)
        self.assertEqual(literal._verification_diagnostics, original._verification_diagnostics)

    def test_all_conditions_keep_identical_first_prompt(self):
        prompts = []
        for mode in ("string", "literal", "relation"):
            runtime = Runtime([payload()])
            MODULE.install_binding_control(runtime, mode)
            result = runtime._verify(QUESTION, "place", DOCUMENTS)
            self.assertEqual(result[0][0]["answer"], "Rome")
            prompts.append(runtime.messages[0])
        self.assertEqual(prompts[0], prompts[1])
        self.assertEqual(prompts[1], prompts[2])

    def test_string_admits_source_mention_without_a_literal_quote(self):
        fabricated_quote = payload(evidence="Rome is Alpha's birthplace.")
        runtime = Runtime([fabricated_quote])
        MODULE.install_binding_control(runtime, "string")
        accepted, rejected, reason = runtime._verify(QUESTION, "place", DOCUMENTS)
        self.assertEqual(accepted[0]["answer"], "Rome")
        self.assertEqual(rejected, [])
        self.assertIsNone(reason)
        self.assertEqual(len(runtime.messages), 1)
        diag = next(iter(runtime._verification_diagnostics.values()))
        self.assertEqual(diag["attempts"], 1)
        self.assertEqual(diag["admission_mode"], "string")
        literal = Runtime([fabricated_quote, {"hypotheses": []}])
        MODULE.install_binding_control(literal, "literal")
        accepted, rejected, _ = literal._verify(QUESTION, "place", DOCUMENTS)
        self.assertEqual(accepted, [])
        self.assertEqual(rejected[0]["reason"], "evidence_not_exact_substring")
        self.assertEqual(len(literal.messages), 2)

    def test_string_answer_can_be_elsewhere_in_source_not_quote(self):
        docs = {0: "Alpha\nAlpha was born in Rome. Alpha was an author."}
        runtime = Runtime([payload(evidence="Alpha was an author.")])
        MODULE.install_binding_control(runtime, "string")
        self.assertEqual(runtime._verify(QUESTION, "place", docs)[0][0]["answer"], "Rome")

    def test_string_still_rejects_missing_answer_invalid_doc_and_low_confidence(self):
        for response, reason in ((payload(answer="Paris"), "answer_not_in_provided_source"),
                                 (payload(doc_id="D2"), "unknown_document_reference"),
                                 (payload(confidence=.2), "low_claimed_confidence"),
                                 (payload(evidence="short"), "invalid_evidence_length"),
                                 (payload(answer="${s1.answer}"), "invalid_answer_or_hash")):
            runtime = Runtime([response, {"hypotheses": []}])
            MODULE.install_binding_control(runtime, "string")
            accepted, rejected, _ = runtime._verify(QUESTION, "place", DOCUMENTS)
            self.assertEqual(accepted, [])
            self.assertEqual(rejected[0]["reason"], reason)

    def test_relation_does_not_admit_wrong_subject_cooccurrence(self):
        docs = {0: "Alpha\nBeta was born in Rome. Alpha later visited Rome."}
        response = payload(evidence="Beta was born in Rome.")
        literal = Runtime([response])
        MODULE.install_binding_control(literal, "literal")
        self.assertEqual(literal._verify(QUESTION, "place", docs)[0][0]["answer"], "Rome")
        relation = Runtime([response, {"hypotheses": []}])
        MODULE.install_binding_control(relation, "relation")
        accepted, rejected, _ = relation._verify(QUESTION, "place", docs)
        self.assertEqual(accepted, [])
        self.assertTrue(rejected)
        diag = next(iter(relation._verification_diagnostics.values()))
        self.assertEqual(diag["relation_scope"], {"relation": "birth_place", "subject": "Alpha"})

    def test_relation_unknown_is_recorded_and_abstains(self):
        docs = {0: "Alpha\nAlpha studied at Example School."}
        response = payload(answer="Example School", evidence="Alpha studied at Example School.")
        runtime = Runtime([response, {"hypotheses": []}])
        MODULE.install_binding_control(runtime, "relation")
        accepted, rejected, _ = runtime._verify("Which school did Alpha study at?", "school", docs)
        self.assertEqual(accepted, [])
        self.assertEqual(rejected[0]["check_outcome"], "unknown")
        self.assertEqual(rejected[0]["relation"], "other")

    def test_relation_literal_failures_are_not_upgraded(self):
        runtime = Runtime([payload(evidence="Rome is Alpha's birthplace."), {"hypotheses": []}])
        MODULE.install_binding_control(runtime, "relation")
        accepted, rejected, _ = runtime._verify(QUESTION, "place", DOCUMENTS)
        self.assertEqual(accepted, [])
        self.assertEqual(rejected[0]["reason"], "evidence_not_exact_substring")

    def test_online_admission_changes_actual_downstream_bound_query(self):
        docs = {0: "Work X\nAlpha wrote Work X.", 1: "Alpha\nAlpha was born in Rome."}
        first = payload(answer="Alpha", evidence="Alpha is the writer of Work X.")
        plan = [{"id": "s1", "question": "Who wrote Work X?", "depends_on": [], "answer_type": "person"},
                {"id": "s2", "question": "Where was ${s1.answer} born?", "depends_on": ["s1"], "answer_type": "place"}]
        observed = {}
        for mode, responses in (("string", [first, payload(doc_id="D1")]),
                                ("literal", [first, {"hypotheses": []}])):
            runtime = Runtime(responses)
            runtime.beam_width, runtime.rank_constant = 1, 10.0
            searches = []

            def add_route(state, source, goal, query, requirements):
                searches.append(query)
                return [1 if "born" in query else 0]

            runtime._add_route = add_route
            runtime._verification_documents = lambda ids: {doc: docs[doc] for doc in ids}
            MODULE.install_binding_control(runtime, mode)
            state = {"_evidence_plan": copy.deepcopy(plan), "evidence_trace": runtime._trace(4),
                     "evidence_candidates": {}}
            runtime._dependency_search([state], None)
            observed[mode] = searches
        self.assertEqual(observed["string"], ["Who wrote Work X?", "Where was Alpha born?"])
        self.assertEqual(observed["literal"], ["Who wrote Work X?"])

    def test_switching_back_to_literal_restores_original_without_stacking(self):
        runtime = Runtime([payload()])
        original = runtime._verify
        MODULE.install_binding_control(runtime, "string")
        MODULE.install_binding_control(runtime, "relation")
        MODULE.install_binding_control(runtime, "literal")
        self.assertEqual(runtime._verify, original)
        self.assertEqual(runtime._verify(QUESTION, "place", DOCUMENTS)[0][0]["answer"], "Rome")

    def test_control_name_preserves_verifier_identity_for_capture_scope(self):
        runtime = Runtime([])
        original = runtime._verify
        for mode in ("string", "relation"):
            MODULE.install_binding_control(runtime, mode)
            self.assertEqual(runtime._verify.__name__, original.__name__)
        MODULE.install_binding_control(runtime, "literal")
        self.assertEqual(runtime._verify, original)

    def test_real_capture_jobs_attribute_threaded_control_requests_to_parent(self):
        capture_module = importlib.import_module(PACKAGE.__name__ + ".experiments.capture")
        ledger_module = importlib.import_module(PACKAGE.__name__ + ".experiments.request_ledger")
        parent_question = "Where was the writer of Work X born?"
        for mode in ("string", "relation"):
            runtime = Runtime([])
            runtime._add_route = lambda state, source, goal, query, requirements=None: [0]
            model = SimpleNamespace(infer=lambda messages: (
                json.dumps(payload()), {"prompt_tokens": 12, "completion_tokens": 20}, False))
            rag = SimpleNamespace(evidence_runtime=runtime, llm_model=model,
                                  rerank_filter=SimpleNamespace(llm_infer_fn=model.infer),
                                  dense_passage_retrieval=lambda query: ([], []), run_ppr=lambda *a, **kw: None)
            for name in ("rerank_facts", "_decompose_query", "_decompose_query_with_hints",
                         "_path_graph_search", "_build_qd_track_ranking", "_query_decomposition_retrieval"):
                setattr(rag, name, lambda *a, **kw: None)
            runtime.rag = rag

            def infer_object(messages):
                response = rag.llm_model.infer(messages)
                return json.loads(response[0]), None

            runtime._infer_object = infer_object
            MODULE.install_binding_control(runtime, mode)
            tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **kw: [1] * 12)
            ledger = ledger_module.LLMCostLedger(tokenizer=tokenizer)
            with tempfile.TemporaryDirectory() as temporary:
                session = capture_module.CaptureSession(rag, temporary, ledger)
                session.register_queries([parent_question])
                session.install()
                try:
                    runtime._add_route({"query": parent_question}, "dag:s1", "dag:s1", QUESTION)
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        results = runtime._collect_jobs(executor, [(runtime._verify, (QUESTION, "place", DOCUMENTS))])
                    self.assertEqual(results[0][0][0]["answer"], "Rome")
                    summary = ledger.get_summary(parent_question)
                    self.assertEqual(summary["logical_requests"], 1)
                    self.assertEqual(summary["by_stage"]["dependency_verification"]["total_tokens"], 32)
                    self.assertEqual(ledger.get_summary(QUESTION)["logical_requests"], 0)
                finally:
                    session.close()

    def test_empty_and_bad_schema_match_bounded_retry_behavior(self):
        for mode in ("string", "relation"):
            runtime = Runtime([{"hypotheses": []}])
            MODULE.install_binding_control(runtime, mode)
            self.assertEqual(runtime._verify(QUESTION, "place", DOCUMENTS), ([], [], None))
            self.assertEqual(len(runtime.messages), 1)
            runtime = Runtime([{"hypotheses": "bad"}, {"hypotheses": "still bad"}])
            MODULE.install_binding_control(runtime, mode)
            accepted, rejected, _ = runtime._verify(QUESTION, "place", DOCUMENTS)
            self.assertEqual(accepted, [])
            self.assertEqual(len(rejected), 2)
            self.assertEqual(len(runtime.messages), 2)

    def test_json_failure_and_transport_exception_propagation(self):
        runtime = Runtime([(None, "invalid_json_object"), {"hypotheses": []}])
        MODULE.install_binding_control(runtime, "string")
        self.assertEqual(runtime._verify(QUESTION, "place", DOCUMENTS)[1][0]["reason"], "invalid_json_object")
        runtime = Runtime([RuntimeError("transport")])
        MODULE.install_binding_control(runtime, "relation")
        with self.assertRaisesRegex(RuntimeError, "transport"):
            runtime._verify(QUESTION, "place", DOCUMENTS)

    def test_invalid_mode_cannot_install_a_gate(self):
        runtime = Runtime([])
        with self.assertRaises(ValueError):
            MODULE.install_binding_control(runtime, "invalid")


if __name__ == "__main__":
    unittest.main()

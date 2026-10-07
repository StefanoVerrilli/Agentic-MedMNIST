from __future__ import annotations

import json
import threading
import time
import unittest
import io
import urllib.error
from unittest.mock import patch

from contracts import (AmbiguityDecision, AutonomousSearchDecision, AutonomousTrainingOptions,
                       ExperimentDecision, LiteratureDecision)
from llm import OllamaDecisionError, OllamaReasoner, _post_json

SAFE_CONFIG = {
    "lr": 0.001,
    "epochs": 4,
    "hidden": 16,
    "batch_size": 64,
    "weight_decay": 0.0001,
    "class_weighting": False,
    "rationale": "bounded fallback",
}


def autonomous_decision(epsilon=1e-7):
    training = dict(lr=.002, weight_decay=.0001, class_weighting=False,
            optimizer="adamw", scheduler="none", label_smoothing=0.,
            early_stopping_patience=0, early_stopping_monitor="val_accuracy",
            early_stopping_min_delta=0., gradient_clip_val=1.,
            adam_beta1=.85, adam_beta2=.995, optimizer_eps=epsilon)
    return dict(action="new_trial", rationale="Test a compact model",
            experiment=dict(bundle_id="compact_experiment", hypothesis="Test a compact model",
                files=[dict(path="experiment.py", code="def build_model(context):\n    pass\n")],
                parameters={}, epochs=2, batch_size=64, training=training))


class OllamaReasonerTests(unittest.TestCase):
    def test_action_repair_names_conflicting_fields_and_preserves_latest_context(self):
        invalid = {**autonomous_decision(), "candidate_id": "autonomous_t0001", "additional_epochs": 2}
        calls, events = [], []

        def transport(url, payload, timeout):
            calls.append(len(payload["messages"]))
            branches = {item["properties"]["action"]["const"]: item for item in payload["format"]["anyOf"]}
            self.assertEqual(branches["new_trial"]["properties"]["candidate_id"]["type"], "null")
            if len(calls) > 1:
                feedback = payload["messages"][-1]["content"]
                self.assertIn("candidate_id must be null or omitted", feedback)
                self.assertIn("additional_epochs must be null or omitted", feedback)
                self.assertIn("$: ", feedback)
            return {"message": {"content": json.dumps(invalid if len(calls) < 3 else autonomous_decision())}}

        result = OllamaReasoner("http://localhost:11434", retries=2, required=True,
            transport=transport).decide(stage="test", system="test", user="test",
            response_model=AutonomousSearchDecision,
            fallback=dict(action="finish_search", rationale="No fallback permitted"),
            audit=lambda event, **kw: events.append(kw))
        self.assertEqual(result.value.action, "new_trial")
        self.assertIsNone(result.value.candidate_id)
        self.assertEqual(calls, [2, 4, 4])
        self.assertEqual(len(events[-1]["rejected_responses"]), 2)

    def test_epsilon_repair_can_reveal_and_then_repair_action_error(self):
        invalid = {**autonomous_decision(.1), "candidate_id": "unassigned"}
        calls = []

        def transport(url, payload, timeout):
            calls.append(payload["format"]["title"])
            if len(calls) == 1:
                content = invalid
            elif len(calls) == 2:
                content = {"optimizer_eps": "2e-7"}
            else:
                previous = json.loads(payload["messages"][-2]["content"])
                self.assertEqual(previous["experiment"]["training"]["optimizer_eps"], 2e-7)
                self.assertEqual(previous["candidate_id"], "unassigned")
                self.assertIn("candidate_id must be null", payload["messages"][-1]["content"])
                content = autonomous_decision(2e-7)
            return {"message": {"content": json.dumps(content)}}

        result = OllamaReasoner("http://localhost:11434", retries=2, required=True,
            transport=transport).decide(stage="test", system="test", user="test",
            response_model=AutonomousSearchDecision,
            fallback=dict(action="finish_search", rationale="No fallback permitted"))
        self.assertEqual(calls, ["AutonomousSearchDecision", "_OptimizerEpsilonRepair", "AutonomousSearchDecision"])
        self.assertEqual(result.value.experiment.training.optimizer_eps, 2e-7)
        self.assertIsNone(result.value.candidate_id)

    def test_length_stop_rejects_even_syntactically_valid_json(self):
        reasoner = OllamaReasoner("http://localhost:11434", retries=0, required=True,
            transport=lambda *args: {"message": {"content": json.dumps(SAFE_CONFIG)}, "done_reason": "length"})
        with self.assertRaisesRegex(OllamaDecisionError, "partial decision.*llm-num-predict"):
            reasoner.decide(stage="test", system="test", user="test",
                response_model=ExperimentDecision, fallback=SAFE_CONFIG)

    def test_timeout_does_not_consume_autonomous_validation_repair(self):
        valid = autonomous_decision()
        invalid = json.loads(json.dumps(valid))
        invalid["experiment"]["training"]["optimizer_eps"] = .1
        calls, events = [], []

        def transport(url, payload, timeout):
            calls.append(len(payload["messages"]))
            if len(calls) == 1:
                raise TimeoutError("timed out")
            if len(calls) == 2:
                return {"message": {"content": json.dumps(invalid)}}
            self.assertIn("experiment.training.optimizer_eps", payload["messages"][-1]["content"])
            self.assertEqual(payload["format"]["properties"]["optimizer_eps"]["type"], "string")
            return {"message": {"content": '{"optimizer_eps": "1e-7"}'}}

        reasoner = OllamaReasoner("http://localhost:11434", transport=transport,
                                 retries=1, required=True)
        result = reasoner.decide(stage="autonomous_search.action_1", system="test", user="test",
            response_model=AutonomousSearchDecision,
            fallback=dict(action="finish_search", rationale="No fallback permitted"),
            audit=lambda event, **kw: events.append(kw))
        self.assertFalse(result.used_fallback)
        self.assertEqual(result.attempts, 3)
        self.assertEqual(result.value.experiment.training.optimizer_eps, 1e-7)
        self.assertEqual(result.value.experiment.training.adam_beta1, .85)
        self.assertEqual(calls, [2, 2, 2])
        self.assertEqual(events[-1]["attempts"], 3)
        expected = AutonomousSearchDecision.model_validate(valid)
        self.assertEqual(result.value, expected)
        self.assertEqual(events[-1]["field_repairs"][0]["rejected_value"], .1)

    def test_focused_epsilon_repair_enforces_bounds_and_preserves_experiment(self):
        for replacement in ("1e-12", "9.9e-12", "2.5e-7", "9.99e-3", "1e-2"):
            for original in (.1, 1e-13):
                with self.subTest(replacement=replacement, original=original):
                    calls = []

                    def transport(url, payload, timeout):
                        calls.append(payload["format"])
                        content = autonomous_decision(original) if len(calls) == 1 else {"optimizer_eps": replacement}
                        return {"message": {"content": json.dumps(content)}}

                    result = OllamaReasoner("http://localhost:11434", transport=transport,
                        retries=1, required=True).decide(stage="test", system="test", user="test",
                        response_model=AutonomousSearchDecision,
                        fallback=dict(action="finish_search", rationale="No fallback permitted"))
                    self.assertEqual(result.value, AutonomousSearchDecision.model_validate(
                        autonomous_decision(float(replacement))))
                    self.assertFalse(result.used_fallback)

    def test_focused_epsilon_repair_rejects_invalid_values_and_extra_changes(self):
        for correction in ({"optimizer_eps": "1e-1"}, {"optimizer_eps": "1e-13"},
                           {"optimizer_eps": "NaN"}, {"optimizer_eps": .1},
                           {"optimizer_eps": "1e-7", "lr": .1}):
            with self.subTest(correction=correction):
                responses = iter((autonomous_decision(.1), correction, correction))
                events = []
                reasoner = OllamaReasoner("http://localhost:11434", retries=2, required=True,
                    transport=lambda *args: {"message": {"content": json.dumps(next(responses))}})
                with self.assertRaisesRegex(OllamaDecisionError, "after 3 attempts"):
                    reasoner.decide(stage="test", system="test", user="test",
                        response_model=AutonomousSearchDecision,
                        fallback=dict(action="finish_search", rationale="No fallback permitted"),
                        audit=lambda event, **kw: events.append(kw))
                self.assertIsNone(events[-1]["decision"])
                self.assertEqual(events[-1]["field_repairs"][0]["rejected_value"], .1)

    def test_multiple_errors_require_full_decision_repair(self):
        invalid = autonomous_decision(.1)
        invalid["experiment"]["training"]["lr"] = -1
        calls = []

        def transport(url, payload, timeout):
            calls.append(payload["format"]["title"])
            return {"message": {"content": json.dumps(invalid if len(calls) == 1 else autonomous_decision())}}

        result = OllamaReasoner("http://localhost:11434", retries=1, required=True,
            transport=transport).decide(stage="test", system="test", user="test",
            response_model=AutonomousSearchDecision,
            fallback=dict(action="finish_search", rationale="No fallback permitted"))
        self.assertEqual(calls, ["AutonomousSearchDecision"] * 2)
        self.assertFalse(result.used_fallback)

    def test_mixed_failures_remain_bounded_and_report_actual_attempts(self):
        for required in (True, False):
            for failures in (("timeout", "invalid", "timeout"),
                             ("invalid", "timeout", "invalid"),
                             ("timeout", "timeout")):
                with self.subTest(required=required, failures=failures):
                    calls, events = [], []

                    def transport(*args):
                        failure = failures[len(calls)]
                        calls.append(failure)
                        if failure == "timeout":
                            raise TimeoutError("timed out")
                        return {"message": {"content": json.dumps({**SAFE_CONFIG, "lr": -1})}}

                    reasoner = OllamaReasoner("http://localhost:11434", transport=transport,
                                             retries=1, required=required)
                    kwargs = dict(stage="test", system="test", user="test",
                        response_model=ExperimentDecision, fallback=SAFE_CONFIG,
                        audit=lambda event, **kw: events.append(kw))
                    if required:
                        with self.assertRaisesRegex(OllamaDecisionError, f"after {len(failures)} attempts"):
                            reasoner.decide(**kwargs)
                        self.assertIsNone(events[-1]["decision"])
                    else:
                        result = reasoner.decide(**kwargs)
                        self.assertTrue(result.used_fallback)
                        self.assertEqual(result.attempts, len(failures))
                    self.assertEqual(calls, list(failures))
                    self.assertEqual(events[-1]["attempts"], len(failures))

    def test_schema_http_400_retries_json_mode_with_full_literature_validation(self):
        valid = {"ideas": [{"idea_id": "compact_model", "target": "architecture",
                 "hypothesis": "Compare compact architectures under equal budgets.",
                 "source_id": "arxiv_2104_05704", "evidence_quote": "convolutional tokenization"}],
                 "proposals": []}
        calls, events = [], []
        def transport(url, payload, timeout):
            calls.append(payload["format"])
            if len(calls) == 1:
                raise urllib.error.HTTPError(url, 400, "Bad Request", {},
                    io.BytesIO(b'{"error":"failed to compile JSON schema grammar"}'))
            self.assertEqual(payload["format"], "json")
            self.assertIn("LiteratureDecision", payload["messages"][0]["content"])
            return {"message": {"content": json.dumps(valid)}}
        reasoner = OllamaReasoner("http://localhost:11434", transport=transport, retries=0, required=True)
        result = reasoner.decide(stage="prior_art.evidence", system="test", user="test",
            response_model=LiteratureDecision, fallback=valid, audit=lambda event, **kw: events.append(kw))
        self.assertFalse(result.used_fallback)
        self.assertEqual(len(calls), 2)
        self.assertEqual(events[-1]["output_format"], "json")
        self.assertIn("compile", events[-1]["format_negotiation"][0])

    def test_json_compatibility_cannot_bypass_required_schema_validation(self):
        calls = 0
        def transport(url, payload, timeout):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise urllib.error.HTTPError(url, 400, "Bad Request", {}, io.BytesIO(b'{"error":"invalid schema"}'))
            return {"message": {"content": json.dumps({**SAFE_CONFIG, "lr": -1})}}
        reasoner = OllamaReasoner("http://localhost:11434", transport=transport, retries=0, required=True)
        with self.assertRaises(OllamaDecisionError):
            reasoner.decide(stage="test", system="test", user="test",
                response_model=ExperimentDecision, fallback=SAFE_CONFIG)

    def test_http_error_body_is_reported_when_json_mode_is_also_rejected(self):
        def transport(url, payload, timeout):
            raise urllib.error.HTTPError(url, 400, "Bad Request", {},
                io.BytesIO(b'{"error":"model backend rejected the request"}'))
        reasoner = OllamaReasoner("http://localhost:11434", transport=transport, retries=0, required=True)
        with self.assertRaisesRegex(OllamaDecisionError, "model backend rejected"):
            reasoner.decide(stage="test", system="test", user="test",
                response_model=ExperimentDecision, fallback=SAFE_CONFIG)

    def test_transport_preserves_server_error_body(self):
        error = urllib.error.HTTPError("http://localhost:11434/api/chat", 400, "Bad Request", {},
                                     io.BytesIO(b'{"error":"unsupported format"}'))
        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaisesRegex(OSError, "unsupported format"):
                _post_json("http://localhost:11434/api/chat", {}, 1)

    def test_schema_valid_response_is_used(self) -> None:
        def transport(url, payload, timeout):
            self.assertTrue(url.endswith("/api/chat"))
            self.assertFalse(payload["stream"])
            self.assertEqual(payload["options"]["temperature"], 0)
            self.assertEqual(payload["format"]["title"], "ExperimentDecision")
            return {"message": {"content": json.dumps(SAFE_CONFIG)}}

        reasoner = OllamaReasoner(
            "http://localhost:11434/v1", transport=transport, retries=0
        )
        result = reasoner.decide(
            stage="test",
            system="test",
            user="test",
            response_model=ExperimentDecision,
            fallback=SAFE_CONFIG,
        )
        self.assertFalse(result.used_fallback)
        self.assertEqual(result.source, "qwen2.5:7b")
        self.assertEqual(result.value.epochs, 4)

    def test_invalid_response_recovers_with_validated_fallback(self) -> None:
        unsafe = {**SAFE_CONFIG, "lr": -1, "epochs": 1_000_000}

        def transport(url, payload, timeout):
            return {"message": {"content": json.dumps(unsafe)}}

        reasoner = OllamaReasoner(
            "http://localhost:11434", transport=transport, retries=1
        )
        result = reasoner.decide(
            stage="test",
            system="test",
            user="test",
            response_model=ExperimentDecision,
            fallback=SAFE_CONFIG,
        )
        self.assertTrue(result.used_fallback)
        self.assertEqual(result.value.lr, 0.001)
        self.assertEqual(result.attempts, 2)

    def test_retry_receives_validation_feedback(self) -> None:
        calls = 0

        def transport(url, payload, timeout):
            nonlocal calls
            calls += 1
            if calls == 1:
                return {"message": {"content": json.dumps({**SAFE_CONFIG, "hidden": 18})}}
            self.assertIn("validation errors", payload["messages"][-1]["content"])
            self.assertEqual(payload["messages"][-2]["role"], "assistant")
            self.assertEqual(json.loads(payload["messages"][-2]["content"])["hidden"], 18)
            return {"message": {"content": json.dumps(SAFE_CONFIG)}}

        reasoner = OllamaReasoner(
            "http://localhost:11434", transport=transport, retries=1, required=True
        )
        result = reasoner.decide(
            stage="repair", system="test", user="test",
            response_model=ExperimentDecision, fallback=SAFE_CONFIG,
        )
        self.assertEqual(result.attempts, 2)
        self.assertFalse(result.used_fallback)

    def test_truncated_response_has_stage_and_completion_diagnostics(self):
        events = []
        def transport(url, payload, timeout):
            return {"message": {"content": '{"lr":', "thinking": "reasoning"},
                    "done_reason": "length", "eval_count": 512}
        reasoner = OllamaReasoner("http://localhost:11434", transport=transport,
                                 retries=0, required=True, num_predict=512)
        with self.assertRaisesRegex(OllamaDecisionError, "autonomous_search.action_1.*llm-num-predict"):
            reasoner.decide(stage="autonomous_search.action_1", system="test", user="test",
                response_model=ExperimentDecision, fallback=SAFE_CONFIG,
                audit=lambda event, **kw: events.append(kw))
        self.assertEqual(events[-1]["status"], "failed")
        self.assertIsNone(events[-1]["decision"])
        rejected = events[-1]["rejected_responses"][0]
        self.assertEqual(rejected["done_reason"], "length")
        self.assertEqual(rejected["completion_tokens"], 512)
        self.assertTrue(rejected["has_thinking"])

    def test_autonomous_training_repair_preserves_agent_selected_active_options(self):
        valid = dict(lr=.002, weight_decay=.0001, class_weighting=False, optimizer="adamw",
            scheduler="cosine", label_smoothing=0., early_stopping_patience=0,
            early_stopping_monitor="val_accuracy", early_stopping_min_delta=0., gradient_clip_val=1.,
            adam_beta1=.85, adam_beta2=.995, optimizer_eps=1e-7, cosine_eta_min=1e-6)
        incomplete = {key: value for key, value in valid.items()
                      if key not in {"adam_beta1", "adam_beta2", "optimizer_eps", "cosine_eta_min"}}
        calls = 0
        def transport(url, payload, timeout):
            nonlocal calls
            calls += 1
            if calls == 1:
                return {"message": {"content": json.dumps(incomplete)}}
            self.assertEqual(json.loads(payload["messages"][-2]["content"]), incomplete)
            self.assertIn("cosine_eta_min", payload["messages"][-1]["content"])
            self.assertIn("adam_beta1", payload["messages"][-1]["content"])
            return {"message": {"content": json.dumps(valid)}}
        reasoner = OllamaReasoner("http://localhost:11434", retries=1, required=True, transport=transport)
        result = reasoner.decide(stage="autonomous_search.action_1", system="test", user="test",
            response_model=AutonomousTrainingOptions, fallback=valid)
        self.assertFalse(result.used_fallback)
        self.assertEqual(result.value.adam_beta1, .85)
        self.assertEqual(result.value.cosine_eta_min, 1e-6)

    def test_empty_thinking_response_cannot_be_used_as_a_decision(self):
        reasoner = OllamaReasoner("http://localhost:11434", retries=0, required=True,
            transport=lambda *args: {"message": {"content": "", "thinking": "unfinished"}})
        with self.assertRaisesRegex(OllamaDecisionError, "no JSON message content"):
            reasoner.decide(stage="test", system="test", user="test",
                response_model=ExperimentDecision, fallback=SAFE_CONFIG)

    def test_timeout_diagnostic_explains_configured_request_limit(self):
        def transport(*args):
            raise TimeoutError("timed out")
        reasoner = OllamaReasoner("http://localhost:11434", retries=0, required=True,
                                 timeout=90, transport=transport)
        with self.assertRaisesRegex(OllamaDecisionError, "timeout=90s.*llm-timeout"):
            reasoner.decide(stage="test", system="test", user="test",
                response_model=ExperimentDecision, fallback=SAFE_CONFIG)

    def test_required_mode_raises_after_invalid_response(self) -> None:
        def transport(url, payload, timeout):
            return {"message": {"content": "not-json"}}

        reasoner = OllamaReasoner(
            "http://localhost:11434",
            transport=transport,
            retries=0,
            required=True,
        )
        with self.assertRaises(OllamaDecisionError):
            reasoner.decide(
                stage="test",
                system="test",
                user="test",
                response_model=ExperimentDecision,
                fallback=SAFE_CONFIG,
            )

    def test_global_lock_allows_only_one_active_ollama_request(self) -> None:
        state = {"active": 0, "maximum": 0}
        state_lock = threading.Lock()

        def transport(url, payload, timeout):
            with state_lock:
                state["active"] += 1
                state["maximum"] = max(state["maximum"], state["active"])
            time.sleep(0.03)
            with state_lock:
                state["active"] -= 1
            content = {"ambiguity_note": "bounded note", "risks": []}
            return {"message": {"content": json.dumps(content)}}

        reasoner = OllamaReasoner(
            "http://localhost:11434", transport=transport, retries=0
        )

        def call() -> None:
            reasoner.decide(
                stage="concurrency",
                system="test",
                user="test",
                response_model=AmbiguityDecision,
                fallback={"ambiguity_note": "fallback note", "risks": []},
            )

        threads = [threading.Thread(target=call) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(state["maximum"], 1)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import threading
import time
import unittest
import io
import urllib.error
from unittest.mock import patch

from contracts import AmbiguityDecision, ExperimentDecision, LiteratureDecision
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


class OllamaReasonerTests(unittest.TestCase):
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

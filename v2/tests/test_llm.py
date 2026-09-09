from __future__ import annotations

import json
import threading
import time
import unittest

from contracts import AmbiguityDecision, ExperimentDecision
from llm import OllamaDecisionError, OllamaReasoner

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

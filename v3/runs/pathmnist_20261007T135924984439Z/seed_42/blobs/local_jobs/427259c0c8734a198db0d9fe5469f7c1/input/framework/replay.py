"""Checksummed decision transcripts with fail-closed offline replay."""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any

from llm import ReasonedDecision, Reasoner
from research import canonical_hash
from failures import is_timeout


def stable_prompt(value: str, root: Path) -> str:
    """Ignore declared execution metadata, never seed, metrics or decision prose.

    File checksums are independently revalidated by the reviewer. They can
    differ between equivalent retrainings because checkpoints embed metadata.
    Exact original requests are retained alongside this replay fingerprint.
    """
    for prefix in (str(root), root.as_posix(), json.dumps(str(root))[1:-1]):
        value = value.replace(prefix, "<RUN>")
    value = re.sub(r"\d{4}-\d\d-\d\d[T ]\d\d:\d\d:\d\d(?:\.\d+)?(?:\+00:00|Z)?", "<TIME>", value)
    value = re.sub(r"\b[a-f0-9]{64}\b", "<CHECKSUM>", value)
    value = re.sub(r'("(?:duration_seconds|elapsed_seconds|total_duration_ns)"\s*:\s*)[0-9.eE+\-]+',
                   r'\g<1>0', value)
    return value


class CachedReasoner:
    def __init__(self, backend: Reasoner, root: Path, *, replay_root: Path | None = None):
        self.backend, self.root = backend, Path(root).resolve()
        self.replay_root = Path(replay_root).resolve() if replay_root else None
        self.model, self.seed = backend.model, backend.seed
        self.directory = self.root / "blobs" / "reasoning"
        self.directory.mkdir(parents=True, exist_ok=True)
        self._sequence = 0
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.backend.enabled

    def decide(self, *, stage: str, system: str, user: str, response_model: Any,
               fallback: Any, audit: Any = None) -> Any:
        started = time.monotonic()
        if audit:
            audit("llm_decision_started", stage=stage, replay=bool(self.replay_root))
        try:
            result = self._decide(stage=stage, system=system, user=user, response_model=response_model,
                                  fallback=fallback, audit=audit)
        except Exception as exc:
            if audit:
                audit("llm_decision_failed", stage=stage, elapsed_seconds=time.monotonic() - started,
                      error_type=type(exc).__name__, error=str(exc)[:8000],
                      failure_kind="llm_timeout" if is_timeout(exc) else "llm_error")
            raise
        if audit:
            audit("llm_decision_finished", stage=stage, elapsed_seconds=time.monotonic() - started,
                  used_fallback=result.used_fallback)
        return result

    def _decide(self, *, stage, system, user, response_model, fallback, audit):
        with self._lock:
            self._sequence += 1
            filename = f"{self._sequence:04d}.json"
            request = {"stage": stage, "system": system, "user": user,
                       "schema": response_model.model_json_schema(), "model": self.model,
                       "seed": self.seed, "fallback": response_model.model_validate(fallback).model_dump(mode="json")}
            stable = {**request, "user": stable_prompt(user, self.root)}
            fingerprint = canonical_hash(stable)
            if self.replay_root:
                source = self.replay_root / "blobs" / "reasoning" / filename
                if not source.exists():
                    raise ValueError(f"missing replay decision {filename} ({stage})")
                envelope = json.loads(source.read_text(encoding="utf-8"))
                payload = envelope["payload"]
                if canonical_hash(payload) != envelope["sha256"]:
                    raise ValueError(f"replay transcript checksum mismatch: {filename}")
                original = payload["request"]
                expected = {**original, "user": stable_prompt(original["user"], self.replay_root)}
                if canonical_hash(expected) != fingerprint:
                    (self.root / "replay_mismatch.json").write_text(json.dumps(
                        {"stage": stage, "original": expected, "replay": stable}, indent=2) + "\n", encoding="utf-8")
                    raise ValueError(f"replay context differs at {filename} ({stage}); refusing a new decision")
                value = response_model.model_validate(payload["decision"])
                result = ReasonedDecision(value, payload["source"], payload["used_fallback"], 0, None)
                if audit:
                    audit("llm_decision", stage=stage, schema=response_model.__name__,
                          status="replayed", source=result.source, attempts=0,
                          decision=value.model_dump(mode="json"))
            else:
                result = self.backend.decide(stage=stage, system=system, user=user,
                                             response_model=response_model, fallback=fallback, audit=audit)
            payload = {"request": request, "request_sha256": fingerprint,
                       "decision": result.value.model_dump(mode="json"), "source": result.source,
                       "used_fallback": result.used_fallback}
            target = self.directory / filename
            if target.exists():
                raise FileExistsError(f"refusing to overwrite reasoning transcript: {target}")
            target.write_text(json.dumps({"payload": payload, "sha256": canonical_hash(payload)},
                                         indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return result

    def assert_replay_complete(self) -> None:
        if self.replay_root:
            expected = len(list((self.replay_root / "blobs" / "reasoning").glob("*.json")))
            if self._sequence != expected:
                raise ValueError(f"replay consumed {self._sequence} of {expected} decisions")

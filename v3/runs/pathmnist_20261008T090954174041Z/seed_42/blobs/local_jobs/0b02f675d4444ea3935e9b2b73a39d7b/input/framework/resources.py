"""Run-scoped Ollama/CUDA handoff, without stopping the Ollama service."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import json
import time
import urllib.request

from failures import WorkerOperationError
from llm import _GLOBAL_OLLAMA_LOCK, _normalise_base_url, _post_json


_CURRENT = ContextVar("run_resources", default=None)


def current_resources():
    return _CURRENT.get()


def _get_json(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


class RunResources:
    def __init__(self, root, *, base_url=None, model=None, audit=None, replay=False,
                 timeout=30.0, post=None, get=None, monotonic=None, sleep=None):
        self.root = root.resolve()
        self.base_url = _normalise_base_url(base_url) if base_url else None
        self.model, self.audit, self.replay, self.timeout = model, audit, replay, timeout
        self.post, self.get = post or _post_json, get or _get_json
        self.clock, self.sleep = monotonic or time.monotonic, sleep or time.sleep

    @contextmanager
    def activate(self):
        token = _CURRENT.set(self)
        try:
            yield self
        finally:
            _CURRENT.reset(token)

    def record(self, event, **details):
        if self.audit:
            self.audit(event, **details)

    def release_for_worker(self, device, **details):
        if device != "cuda" or self.replay or not self.base_url:
            return
        started = self.clock()
        deadline = started + self.timeout
        self.record("ollama_unload_started", model=self.model, **details)
        def remaining():
            value = deadline - self.clock()
            if value <= 0:
                raise TimeoutError("Ollama model release exceeded its deadline")
            return value
        acquired = False
        try:
            acquired = _GLOBAL_OLLAMA_LOCK.acquire(timeout=remaining())
            if not acquired:
                raise TimeoutError("Ollama handoff lock exceeded its deadline")
            response = self.post(f"{self.base_url}/api/generate",
                                 {"model": self.model, "prompt": "", "stream": False, "keep_alive": 0},
                                 remaining())
            if not isinstance(response, dict) or response.get("error"):
                raise ValueError(f"Ollama unload failed: {response}")
            expected = self.model if ":" in self.model.rsplit("/", 1)[-1] else self.model + ":latest"
            while True:
                snapshot = self.get(f"{self.base_url}/api/ps", remaining())
                models = snapshot.get("models")
                if not isinstance(models, list) or any(not isinstance(item, dict) for item in models):
                    raise ValueError("Invalid Ollama running-model response")
                if not any(expected in {item.get("name"), item.get("model")} or
                           self.model in {item.get("name"), item.get("model")} for item in models):
                    remaining()
                    break
                self.sleep(min(1.0, remaining()))
            self.record("ollama_unload_finished", model=self.model,
                        elapsed_seconds=self.clock() - started, **details)
        except Exception as exc:
            self.record("ollama_unload_failed", model=self.model, error=str(exc),
                        elapsed_seconds=self.clock() - started, failure_kind="resource_handoff", **details)
            raise WorkerOperationError(str(exc), operation="resource_handoff", kind="resource_handoff") from exc
        finally:
            if acquired:
                _GLOBAL_OLLAMA_LOCK.release()

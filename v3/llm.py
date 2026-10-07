"""Strict, sequential Ollama reasoning seam.

Only the declared judgement points use this module.  Calls target Ollama's
native ``/api/chat`` endpoint with a JSON Schema, temperature zero and an
in-process global lock: even if callers are threaded, only one model request is
active at a time.  Invalid or unavailable responses use a validated heuristic
fallback unless ``required=True``.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
import urllib.error
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

DecisionT = TypeVar("DecisionT", bound=BaseModel)
AuditSink = Callable[..., Any]
Transport = Callable[[str, dict[str, Any], float], dict[str, Any]]
_GLOBAL_OLLAMA_LOCK = threading.Lock()


class OllamaDecisionError(RuntimeError):
    """Raised when Ollama is required and no validated response is obtained."""


class OllamaHTTPError(OSError):
    """Retain the bounded server error body, without logging the request."""

    def __init__(self, code: int, detail: str):
        self.code = code
        super().__init__(f"HTTP {code}: {detail}")


@dataclass(frozen=True)
class ReasonedDecision(Generic[DecisionT]):
    value: DecisionT
    source: str
    used_fallback: bool
    attempts: int
    request_id: str | None


class Reasoner(Protocol):
    """Provider-neutral interface consumed by hybrid agents."""

    model: str
    seed: int

    @property
    def enabled(self) -> bool: ...

    def decide(
        self,
        *,
        stage: str,
        system: str,
        user: str,
        response_model: type[DecisionT],
        fallback: DecisionT | dict[str, Any],
        audit: AuditSink | None = None,
    ) -> ReasonedDecision[DecisionT]: ...


class OllamaReasoner:
    """Small provider seam for schema-constrained local Ollama decisions."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str = "qwen2.5:7b",
        *,
        timeout: float = 90.0,
        retries: int = 1,
        required: bool = False,
        keep_alive: str = "5m",
        seed: int = 42,
        num_predict: int = 16384,
        transport: Transport | None = None,
    ):
        self.base_url = _normalise_base_url(base_url) if base_url else None
        self.model = model
        self.timeout = timeout
        self.retries = max(0, retries)
        self.required = required
        self.keep_alive = keep_alive
        self.seed = seed
        self.num_predict = max(512, int(num_predict))
        self._transport = transport or _post_json

    @classmethod
    def from_env(cls, *, required: bool = False, seed: int = 42) -> OllamaReasoner:
        return cls(
            base_url=os.environ.get("AGENTIC_LLM_BASE"),
            model=os.environ.get("AGENTIC_LLM_MODEL", "qwen2.5:7b"),
            timeout=float(os.environ.get("AGENTIC_LLM_TIMEOUT", "90")),
            retries=int(os.environ.get("AGENTIC_LLM_RETRIES", "1")),
            required=required,
            keep_alive=os.environ.get("AGENTIC_LLM_KEEP_ALIVE", "5m"),
            seed=seed,
            num_predict=int(os.environ.get("AGENTIC_LLM_NUM_PREDICT", "16384")),
        )

    @property
    def enabled(self) -> bool:
        return self.base_url is not None

    def decide(
        self,
        *,
        stage: str,
        system: str,
        user: str,
        response_model: type[DecisionT],
        fallback: DecisionT | dict[str, Any],
        audit: AuditSink | None = None,
    ) -> ReasonedDecision[DecisionT]:
        validated_fallback = response_model.model_validate(fallback)
        schema = response_model.model_json_schema()
        if not self.enabled:
            _audit(
                audit,
                "llm_decision",
                stage=stage,
                schema=response_model.__name__,
                status="heuristic",
                source="heuristic",
                attempts=0,
                decision=validated_fallback.model_dump(mode="json"),
            )
            return ReasonedDecision(validated_fallback, "heuristic", True, 0, None)

        request_id = uuid.uuid4().hex
        schema_text = json.dumps(schema, ensure_ascii=False, sort_keys=True)
        messages = [
            {
                "role": "system",
                "content": (
                    system
                    + " Return only JSON conforming exactly to this schema: "
                    + schema_text
                ),
            },
            {"role": "user", "content": user},
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": schema,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": 0,
                "seed": self.seed,
                "num_predict": self.num_predict,
            },
        }

        errors: list[str] = []
        error_details: list[str] = []
        rejected_responses: list[dict[str, Any]] = []
        format_negotiation: list[str] = []
        # A transport failure produces no decision to repair. Keep its retry
        # budget separate so a timeout cannot consume JSON correction attempts.
        transport_failures = validation_failures = 0
        attempt = 0
        while True:
            attempt += 1
            started = time.monotonic()
            response = None
            content = None
            try:
                with _GLOBAL_OLLAMA_LOCK:
                    try:
                        response = self._transport(
                            f"{self.base_url}/api/chat", payload, self.timeout
                        )
                    except OSError as exc:
                        # Some Ollama backends cannot compile a nested schema.
                        # JSON mode remains model reasoning: Pydantic still
                        # enforces the complete original contract below.
                        if getattr(exc, "code", None) != 400 or payload["format"] == "json":
                            raise
                        format_negotiation.append(_error_detail(exc))
                        payload["format"] = "json"
                        response = self._transport(
                            f"{self.base_url}/api/chat", payload, self.timeout
                        )
                content = response["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("Ollama returned no JSON message content")
                parsed = json.loads(content)
                decision = response_model.model_validate(parsed)
                _audit(
                    audit,
                    "llm_decision",
                    stage=stage,
                    schema=response_model.__name__,
                    status="validated",
                    source=self.model,
                    attempts=attempt,
                    request_id=request_id,
                    elapsed_seconds=round(time.monotonic() - started, 4),
                    prompt_tokens=response.get("prompt_eval_count"),
                    completion_tokens=response.get("eval_count"),
                    total_duration_ns=response.get("total_duration"),
                    output_format="json" if payload["format"] == "json" else "json_schema",
                    format_negotiation=format_negotiation,
                    decision=decision.model_dump(mode="json"),
                )
                return ReasonedDecision(
                    decision, self.model, False, attempt, request_id
                )
            except (KeyError, TypeError, ValueError, ValidationError, OSError) as exc:
                errors.append(type(exc).__name__)
                if isinstance(exc, ValidationError):
                    detail = "; ".join(
                        f"{'.'.join(map(str, row['loc']))}: {row['msg']}"
                        for row in exc.errors(include_url=False)[:8]
                    )
                else:
                    detail = _error_detail(exc)
                if isinstance(exc, TimeoutError):
                    detail += f" (request timeout={self.timeout}s; increase --llm-timeout for slow generation)"
                if isinstance(response, dict):
                    rejected_responses.append({
                        "attempt": attempt,
                        "done_reason": response.get("done_reason"),
                        "completion_tokens": response.get("eval_count"),
                        "content_preview": content[:4000] if isinstance(content, str) else None,
                        "content_length": len(content) if isinstance(content, str) else None,
                        "has_thinking": bool(response.get("message", {}).get("thinking"))
                            if isinstance(response.get("message"), dict) else False,
                    })
                    if response.get("done_reason") == "length":
                        detail += (f" (generation reached num_predict={self.num_predict}; "
                                   "the response may be truncated; increase --llm-num-predict)")
                    if response.get("error"):
                        detail += f" (server error: {str(response['error'])[:4000]})"
                error_details.append(detail)
                if isinstance(exc, OSError):
                    transport_failures += 1
                    retry = transport_failures <= self.retries
                else:
                    validation_failures += 1
                    retry = validation_failures <= self.retries
                if retry and not isinstance(exc, OSError):
                    if isinstance(content, str) and content.strip():
                        messages.append({"role": "assistant", "content": content})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Your previous JSON was rejected. Correct these "
                                "validation errors and return a complete replacement: "
                                + detail
                            ),
                        }
                    )
                if not retry:
                    break

        error_summary = ",".join(errors)
        _audit(
            audit,
            "llm_decision",
            stage=stage,
            schema=response_model.__name__,
            status="fallback" if not self.required else "failed",
            source="heuristic:fallback",
            attempts=attempt,
            request_id=request_id,
            error_types=error_summary,
            error_details=error_details,
            rejected_responses=rejected_responses,
            timeout_seconds=self.timeout,
            num_predict=self.num_predict,
            output_format="json" if payload["format"] == "json" else "json_schema",
            format_negotiation=format_negotiation,
            decision=(
                None if self.required else validated_fallback.model_dump(mode="json")
            ),
        )
        if self.required:
            raise OllamaDecisionError(
                f"Ollama stage {stage} returned no valid {response_model.__name__} after "
                f"{attempt} attempts ({error_summary}): "
                f"{error_details[-1] if error_details else 'unknown validation error'}. "
                f"See llm_decision in decision_log.jsonl (request_id={request_id})."
            )
        return ReasonedDecision(
            validated_fallback,
            f"heuristic:fallback:{errors[-1] if errors else 'unknown'}",
            True,
            attempt,
            request_id,
        )


def _normalise_base_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    for suffix in ("/v1", "/api"):
        base = base.removesuffix(suffix)
    if not base.startswith(("http://", "https://")):
        raise ValueError("AGENTIC_LLM_BASE must start with http:// or https://")
    return base


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise OllamaHTTPError(exc.code, _error_detail(exc)) from exc


def _error_detail(exc: BaseException) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        try:
            body = exc.read(8192).decode("utf-8", errors="replace")
            if body:
                try:
                    document = json.loads(body)
                    body = str(document.get("error", body)) if isinstance(document, dict) else body
                except ValueError:
                    pass
                return f"HTTP {exc.code}: {body}"[:4000]
        except OSError:
            return str(exc)[:4000]
        finally:
            exc.close()
    return str(exc)[:4000]


def ollama_model_digest(
    base_url: str | None, model: str, *, timeout: float = 5.0
) -> str | None:
    """Resolve an Ollama tag to its immutable digest when the server exposes it."""
    if not base_url:
        return None
    try:
        request = urllib.request.Request(
            f"{_normalise_base_url(base_url)}/api/tags", method="GET"
        )
        with (
            _GLOBAL_OLLAMA_LOCK,
            urllib.request.urlopen(request, timeout=timeout) as response,
        ):
            payload = json.load(response)
        for item in payload.get("models", []):
            names = {str(item.get("name", "")), str(item.get("model", ""))}
            if model in names:
                digest = item.get("digest")
                return str(digest) if digest else None
    except (OSError, ValueError, TypeError, KeyError):
        return None
    return None


def _audit(sink: AuditSink | None, event: str, **details: Any) -> None:
    if sink is not None:
        sink(event, **details)

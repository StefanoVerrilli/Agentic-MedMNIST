"""The 'hybrid' seam: LLM reasoning where judgement is needed, deterministic
fallback everywhere else.

Provider-agnostic on purpose (satisfies the "avoid a model owned by a company"
principle): it talks to any OpenAI-compatible endpoint, so you point it at a
local open model (Ollama / vLLM) via env vars. With no endpoint configured it
returns the heuristic fallback, so the whole pipeline still runs offline and
reproducibly.

    export AGENTIC_LLM_BASE=http://localhost:11434/v1   # e.g. Ollama
    export AGENTIC_LLM_MODEL=llama3.1
"""
from __future__ import annotations

import json
import os
import urllib.request


def _extract_json(text: str) -> str:
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end != -1 else "{}"


def reason_json(system: str, user: str, fallback: dict) -> dict:
    """Return a dict merged over `fallback`. Uses the LLM when configured,
    otherwise (or on any error) returns the deterministic fallback tagged as
    such, so every decision records its provenance."""
    base = os.environ.get("AGENTIC_LLM_BASE")
    if not base:
        return {**fallback, "source": "heuristic"}
    model = os.environ.get("AGENTIC_LLM_MODEL", "llama3.1")
    key = os.environ.get("AGENTIC_LLM_KEY", "not-needed")
    try:
        payload = {
            "model": model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        req = urllib.request.Request(
            base.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {key}"},
        )
        resp = json.load(urllib.request.urlopen(req, timeout=60))
        data = json.loads(_extract_json(resp["choices"][0]["message"]["content"]))
        return {**fallback, **data, "source": model}
    except Exception as exc:  # any failure -> stay reproducible
        return {**fallback, "source": f"heuristic(fallback:{type(exc).__name__})"}

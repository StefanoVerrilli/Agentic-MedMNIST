"""Stable failure categories shared by the controller and isolated workers."""
from __future__ import annotations

import errno
import re
import subprocess


TERMINAL_FAILURES = {"storage_error", "controller_invariant", "resource_handoff"}


def is_timeout(exc):
    return (isinstance(exc, (TimeoutError, subprocess.TimeoutExpired)) or
            isinstance(getattr(exc, "reason", None), TimeoutError) or
            "timeout" in str(exc).lower() or "timed out" in str(exc).lower())


def failure_kind(exc, operation):
    if getattr(exc, "failure_kind", None):
        return exc.failure_kind
    message = str(exc).lower()
    if isinstance(exc, OSError) and exc.errno in {errno.ENOSPC, errno.EIO, errno.EROFS, errno.EDQUOT}:
        return "storage_error"
    if any(text in message for text in ("no space left on device", "input/output error", "disk quota exceeded")):
        return "storage_error"
    if is_timeout(exc):
        return "timeout"
    if isinstance(exc, MemoryError) or "out of memory" in message:
        return "oom"
    if operation == "verify":
        return "code_validation"
    if operation in {"train", "infer"}:
        return "training_error"
    return "controller_invariant"


def failure_signature(error, kind):
    text = str(error)
    frames = re.findall(r'File "([^"]+)", line \d+, in ([^\n]+)', text)
    generated = [(path, fn.strip()) for path, fn in frames if "/bundle/" in path.replace("\\", "/")]
    frame = (generated or frames)[-1] if (generated or frames) else ("", "")
    filename = frame[0].replace("\\", "/").rsplit("/", 1)[-1]
    last = text.strip().splitlines()[-1] if text.strip() else kind
    exception = re.search(r'(?:[\w.]+(?:Error|Exception)):', last)
    if exception:
        last = last[exception.start():]
    last = re.sub(r'\b[a-f0-9]{32,64}\b', "<id>", last)
    last = re.sub(r'(?<![a-zA-Z_])-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', "<n>", last)
    last = re.sub(r'[A-Za-z]:[\\/][^\s]+|/(?:[^\s/]+/)+[^\s]+', "<path>", last)
    return f"{kind}|{filename}:{frame[1]}|{last}"


class WorkerOperationError(RuntimeError):
    def __init__(self, message, *, operation, kind, signature=None):
        super().__init__(message)
        self.operation = operation
        self.failure_kind = kind
        self.failure_signature = signature or failure_signature(message, kind)


class SearchCircuitBreaker(RuntimeError):
    """A terminal operational failure, rather than a scientific search budget."""

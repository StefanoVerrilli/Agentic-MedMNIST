"""Run-scoped code storage and local subprocess execution."""
from __future__ import annotations

import ast
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import os
import sys
import shutil
import subprocess
import time
import uuid

from contracts import (GeneratedBundle, GeneratedBundleReference, GeneratedCodeDecision,
                       TrainConfig, TrainResult, WorkerConfiguration, sha256_file)
from research import canonical_hash, contained_path
from failures import WorkerOperationError, failure_kind
from resources import current_resources


def archive_bundle(bb, decision: GeneratedCodeDecision, source: str, *, parent=None):
    existing = list((bb.root / "generated").glob("*/v*/manifest.json"))
    settings = bb.get_optional("worker_evidence")
    limit = settings.configuration.max_bundles if settings else 32
    autonomous = getattr(bb.get_optional("search_plan"), "policy", "legacy") == "agent_autonomous"
    if not autonomous and len(existing) >= limit:
        raise ValueError("run bundle budget exhausted")
    directory = bb.root / "generated" / decision.bundle_id
    version = 1 + len(list(directory.glob("v*/manifest.json")))
    destination = directory / f"v{version:03d}"
    # Parse syntax only: generated Python is never imported by the controller.
    for item in decision.files:
        path = contained_path(destination, item.path)
        if path.suffix != ".py" or ".." in PurePosixPath(item.path).parts:
            raise ValueError("invalid generated source path")
        ast.parse(item.code, filename=item.path)
    destination.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for item in decision.files:
        path = contained_path(destination, item.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(item.code, encoding="utf-8", newline="\n")
        hashes[item.path] = sha256_file(path)
    document = {"bundle_id": decision.bundle_id, "version": version,
                "files": hashes, "entrypoint": "experiment.py"}
    manifest = destination / "manifest.json"
    manifest.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")
    reference = GeneratedBundleReference(path=manifest.relative_to(bb.root).as_posix(),
        sha256=sha256_file(manifest), bundle_id=decision.bundle_id, version=version,
        parameters=decision.parameters)
    artifact = GeneratedBundle(reference=reference, hypothesis=decision.hypothesis,
        author="experimental_development", source=source, files=hashes,
        parent_sha256=parent.sha256 if parent else None)
    bb.put(f"generated_{decision.bundle_id}", artifact, producer="experimental_development")
    return reference


def validate_bundle(root: Path, reference: GeneratedBundleReference) -> Path:
    root = root.resolve()
    def reject_symlinks(raw: Path):
        if raw.is_symlink():
            raise ValueError("generated bundle symlinks are forbidden")
        for parent in raw.parents:
            if parent == root:
                break
            if parent.is_symlink():
                raise ValueError("generated bundle symlinks are forbidden")
    reject_symlinks(root / reference.path)
    manifest = contained_path(root, reference.path)
    if manifest.is_symlink() or sha256_file(manifest) != reference.sha256:
        raise ValueError("generated manifest checksum mismatch")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(data.get("files"), dict) or not 1 <= len(data["files"]) <= 16:
        raise ValueError("invalid generated file manifest")
    if data["bundle_id"] != reference.bundle_id or data["version"] != reference.version:
        raise ValueError("bundle identity mismatch")
    if data.get("entrypoint") != "experiment.py" or "experiment.py" not in data["files"]:
        raise ValueError("missing generated entrypoint")
    actual = {path.relative_to(manifest.parent).as_posix() for path in manifest.parent.rglob("*")
              if path.is_file() and path != manifest}
    if actual != set(data["files"]):
        raise ValueError("generated bundle has unrecorded or missing files")
    for name, digest in data["files"].items():
        if not name.endswith(".py"):
            raise ValueError("generated bundles contain Python sources only")
        path = contained_path(manifest.parent, name)
        reject_symlinks(manifest.parent / name)
        if sha256_file(path) != digest:
            raise ValueError(f"generated source checksum mismatch: {name}")
    return manifest.parent


def copy_bundle(origin: Path, destination: Path, reference: GeneratedBundleReference):
    source = validate_bundle(origin, reference)
    target = contained_path(destination, reference.path).parent
    if target.exists():
        validate_bundle(destination, reference)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)
    validate_bundle(destination, reference)


class LocalWorker:
    """Run generated code in a child of the current Python environment."""

    def __init__(self, configuration: WorkerConfiguration | None = None, *, coordinator=None):
        self.configuration = configuration or WorkerConfiguration()
        self.coordinator = coordinator or current_resources()

    def preflight(self, device: str):
        import torch
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in the current Python environment; use --device cpu")
        return {"backend": "local.subprocess", "python_executable": sys.executable,
                "isolation": {"enforced": False, "execution": "local_subprocess"}}

    def execute(self, root: Path, config: TrainConfig, operation: str, arrays: dict,
                *, checkpoint: Path | None = None, epochs: int | None = None, evidence=None,
                resume: Path | None = None, start_epoch: int = 0, candidate_id=None,
                representation=None):
        root = root.resolve()
        job = root / "blobs" / "local_jobs" / uuid.uuid4().hex
        started = time.monotonic()
        coordinator = self.coordinator
        if coordinator and coordinator.root != root:
            raise ValueError("worker resource coordinator belongs to another run")
        def event(name, **details):
            if coordinator:
                coordinator.record(name, job_id=job.name, operation=operation, candidate_id=candidate_id,
                                   elapsed_seconds=time.monotonic() - started, **details)
        event("local_job_created", array_shapes={key: list(value.shape) for key, value in arrays.items()},
              array_bytes=sum(value.nbytes for value in arrays.values()))
        try:
            output = self._execute(root, config, operation, arrays, job=job, event=event,
                                   checkpoint=checkpoint, epochs=epochs, evidence=evidence,
                                   resume=resume, start_epoch=start_epoch, representation=representation)
            event("local_job_completed")
            return output
        except Exception as exc:
            event("local_job_failed", error=str(exc)[:8000], failure_kind=failure_kind(exc, operation))
            raise

    def _execute(self, root, config, operation, arrays, *, job, event,
                 checkpoint=None, epochs=None, evidence=None, resume=None, start_epoch=0,
                 representation=None):
        import numpy as np
        if operation not in {"verify", "train", "infer", "strategy"}:
            raise ValueError("unsupported local operation")
        root = root.resolve()
        inputs, output = job / "input", job / "output"
        framework = inputs / "framework"
        framework.mkdir(parents=True, exist_ok=False)
        output.mkdir()
        work = job / "work"
        work.mkdir()
        for path in Path(__file__).parent.glob("*.py"):
            shutil.copyfile(path, framework / path.name)
        if config.generated_bundle:
            shutil.copytree(validate_bundle(root, config.generated_bundle), inputs / "bundle")
        if checkpoint:
            shutil.copyfile(checkpoint, inputs / "model.ckpt")
        if resume:
            shutil.copyfile(resume, inputs / "resume.pt")
        event("local_job_arrays_saving")
        np.savez(inputs / "data.npz", **arrays)
        event("local_job_arrays_saved", data_npz_bytes=(inputs / "data.npz").stat().st_size)
        request = {"operation": operation, "config": config.model_dump(mode="json"),
                   "epochs": epochs if epochs is not None else config.epochs,
                   "evidence": evidence or [], "start_epoch": start_epoch, "resume": resume is not None,
                   "representation": representation or getattr(self.coordinator, "representation", None)}
        (inputs / "request.json").write_text(json.dumps(request) + "\n", encoding="utf-8")
        command = [sys.executable, str(framework / "worker_runtime.py"), "--job", str(job)]
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        # CREATE_SUSPENDED prevents descendants escaping before Job Object assignment.
        options = ({"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP | 0x4}
                   if os.name == "nt" else {"start_new_session": True})
        log = job / "process.log"
        if self.coordinator:
            self.coordinator.release_for_worker(config.device, job_id=job.name, operation=operation)
        event("local_job_process_starting")
        with log.open("wb") as handle:
            tree = None
            if os.name == "nt":
                from process_control import WindowsProcessTree
                try:
                    tree = WindowsProcessTree()
                except Exception as exc:
                    raise WorkerOperationError(f"Cannot create worker process tree: {exc}",
                                               operation=operation, kind="controller_invariant") from exc
            try:
                process = subprocess.Popen(command, cwd=work, env=environment,
                                           stdout=handle, stderr=subprocess.STDOUT, **options)
                deadline = time.monotonic() + self.configuration.timeout_seconds
                try:
                    if tree:
                        try:
                            tree.assign(process)
                        except Exception as exc:
                            raise WorkerOperationError(f"Cannot attach worker process tree: {exc}",
                                                       operation=operation, kind="controller_invariant") from exc
                    event("local_job_process_started", child_pid=process.pid)
                    while True:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise subprocess.TimeoutExpired(command, self.configuration.timeout_seconds)
                        try:
                            code = process.wait(timeout=min(15.0, remaining))
                            break
                        except subprocess.TimeoutExpired:
                            if time.monotonic() >= deadline:
                                raise
                            event("local_job_process_heartbeat", child_pid=process.pid)
                except BaseException as exc:
                    try:
                        if tree:
                            tree.terminate()
                        else:
                            import signal
                            try:
                                os.killpg(process.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                    finally:
                        if process.poll() is None:
                            process.kill()
                        process.wait(timeout=30)
                    event("local_job_process_exited", child_pid=process.pid, return_code=process.returncode,
                          timed_out=isinstance(exc, subprocess.TimeoutExpired))
                    if isinstance(exc, subprocess.TimeoutExpired):
                        raise TimeoutError("local operation exceeded its timeout") from exc
                    raise
            finally:
                if tree:
                    try:
                        tree.terminate()
                    finally:
                        tree.close()
        event("local_job_process_exited", child_pid=process.pid, return_code=code)
        if code:
            with log.open("rb") as handle:
                handle.seek(max(0, log.stat().st_size - 8000))
                detail = handle.read().decode("utf-8", "replace")
            report = output / "failure.json"
            failure = json.loads(report.read_text(encoding="utf-8")) if report.exists() else {}
            raise WorkerOperationError(f"local process exit {code}: {detail}",
                operation=failure.get("operation", operation),
                kind=failure.get("failure_kind") or failure_kind(RuntimeError(detail), operation),
                signature=failure.get("failure_signature"))
        if config.generated_bundle:
            validate_bundle(root, config.generated_bundle)
        return output

def prepared_arrays(prepared, splits, batch_size=128, *, seed=42, corruption_sigma=None):
    import numpy as np
    from ml import make_loader
    arrays = {}
    coordinator = current_resources()
    started = time.monotonic()
    if coordinator:
        coordinator.record("local_job_arrays_preparing", splits=splits)
    for split in splits:
        loader = make_loader(prepared, split=split, batch_size=batch_size, shuffle=False,
                             seed=seed, corruption_sigma=corruption_sigma)
        images, targets = [], []
        for image, target in loader:
            images.append(image.numpy())
            targets.append(target.numpy())
        arrays[f"{split}_images"] = np.concatenate(images)
        arrays[f"{split}_targets"] = np.concatenate(targets)
    if coordinator:
        coordinator.representation = {"normalization": prepared.normalization,
            "augmentations": list(prepared.augmentations), "mean": list(prepared.mean), "std": list(prepared.std),
            "layout": "NCHW", "already_prepared": True}
        coordinator.record("local_job_arrays_prepared", elapsed_seconds=time.monotonic() - started,
            array_shapes={key: list(value.shape) for key, value in arrays.items()},
            array_bytes=sum(value.nbytes for value in arrays.values()))
    return arrays


@dataclass
class RemoteModel:
    root: Path
    checkpoint: Path
    checkpoint_sha256: str
    config: TrainConfig
    coordinator: object = None

    def __post_init__(self):
        if self.coordinator is None:
            self.coordinator = current_resources()

    def predict(self, prepared, split, batch_size, *, seed=42, corruption_sigma=None):
        import numpy as np
        if sha256_file(self.checkpoint) != self.checkpoint_sha256:
            raise ValueError("run checkpoint checksum mismatch")
        data = prepared_arrays(prepared, [split], batch_size, seed=seed,
                               corruption_sigma=corruption_sigma)
        targets = data.pop(f"{split}_targets")
        options = {"coordinator": self.coordinator} if self.coordinator else {}
        output = LocalWorker(self.config.worker, **options).execute(self.root, self.config, "infer",
            {"images": data[f"{split}_images"]}, checkpoint=self.checkpoint)
        logits = np.load(output / "logits.npy", allow_pickle=False)
        if logits.shape != (len(targets), 9) or not np.issubdtype(logits.dtype, np.floating) or not np.isfinite(logits).all():
            raise ValueError("worker predictions must be finite N x 9 logits in original order")
        values = logits.astype("float64") - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(values)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities, logits, targets


def train_remote(prepared, config: TrainConfig, checkpoint: Path):
    from ml import TrainingOutput
    root = checkpoint.resolve().parent
    while root.name != "blobs" and root != root.parent:
        root = root.parent
    if root.name != "blobs":
        raise ValueError("run checkpoints must reside in run/blobs")
    root = root.parent
    arrays = prepared_arrays(prepared, ["train", "val"], config.batch_size, seed=config.seed)
    worker = LocalWorker(config.worker)
    resume, previous, start_epoch, previous_history = None, None, 0, []
    targets = config.segment_targets if config.execution_mode == "agent_autonomous" and config.segment_targets else [config.epochs]
    for target in targets:
        segment_config = config.model_copy(update={"epochs": target})
        if config.execution_mode == "agent_autonomous":
            output = worker.execute(root, segment_config, "train", arrays,
                checkpoint=previous, resume=resume, start_epoch=start_epoch)
        else:
            output = worker.execute(root, segment_config, "train", arrays)
        segment_data = json.loads((output / "result.json").read_text(encoding="utf-8"))
        if config.execution_mode == "agent_autonomous":
            TrainResult.model_validate({**segment_data, "checkpoint_path": "model.ckpt", "checkpoint_sha256": "0" * 64,
                "seed": config.seed, "device": config.device})
            if segment_data["history"][:start_epoch] != previous_history:
                raise ValueError("continuation rewrote recorded history")
            if not start_epoch < segment_data["epochs_completed"] <= target:
                raise ValueError("worker returned invalid continuation epoch count")
            if segment_data.get("stop_reason") not in {"segment_complete", "early_stopping"}:
                raise ValueError("autonomous worker must report stopping reason")
            if segment_data["stop_reason"] == "segment_complete" and segment_data["epochs_completed"] != target:
                raise ValueError("autonomous worker stopped before its requested target")
            resume, previous = output / "resume.pt", output / "model.ckpt"
            if not resume.is_file():
                raise ValueError("missing continuation state")
            start_epoch = segment_data["epochs_completed"]
            previous_history = segment_data["history"]
            if segment_data["stop_reason"] == "early_stopping":
                break
    result_data = json.loads((output / "result.json").read_text(encoding="utf-8"))
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(output / "model.ckpt", checkpoint)
    log = checkpoint.with_suffix(".training.jsonl")
    shutil.copyfile(output / "training.jsonl", log)
    result_data.update(checkpoint_path=str(checkpoint.resolve()),
        checkpoint_sha256=sha256_file(checkpoint), training_log_path=str(log.resolve()),
        seed=config.seed, device=config.device, framework="local.python")
    if config.execution_mode == "agent_autonomous":
        resume_target = checkpoint.with_suffix(".resume.pt")
        shutil.copyfile(resume, resume_target)
        result_data.update(resume_path=str(resume_target.resolve()), resume_sha256=sha256_file(resume_target))
    result = TrainResult.model_validate(result_data)
    if not 1 <= result.epochs_completed <= config.epochs:
        raise ValueError("worker exceeded the epoch budget")
    return TrainingOutput(RemoteModel(root, checkpoint, result.checkpoint_sha256, config), result)

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
import uuid

from contracts import (GeneratedBundle, GeneratedBundleReference, GeneratedCodeDecision,
                       TrainConfig, TrainResult, WorkerConfiguration, sha256_file)
from research import canonical_hash, contained_path


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

    def __init__(self, configuration: WorkerConfiguration | None = None):
        self.configuration = configuration or WorkerConfiguration()

    def preflight(self, device: str):
        import torch
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in the current Python environment; use --device cpu")
        return {"backend": "local.subprocess", "python_executable": sys.executable,
                "isolation": {"enforced": False, "execution": "local_subprocess"}}

    def execute(self, root: Path, config: TrainConfig, operation: str, arrays: dict,
                *, checkpoint: Path | None = None, epochs: int | None = None, evidence=None,
                resume: Path | None = None, start_epoch: int = 0):
        import numpy as np
        if operation not in {"verify", "train", "infer", "strategy"}:
            raise ValueError("unsupported local operation")
        root = root.resolve()
        job = root / "blobs" / "local_jobs" / uuid.uuid4().hex
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
        np.savez(inputs / "data.npz", **arrays)
        request = {"operation": operation, "config": config.model_dump(mode="json"),
                   "epochs": epochs if epochs is not None else config.epochs,
                   "evidence": evidence or [], "start_epoch": start_epoch, "resume": resume is not None}
        (inputs / "request.json").write_text(json.dumps(request) + "\n", encoding="utf-8")
        command = [sys.executable, str(framework / "worker_runtime.py"), "--job", str(job)]
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        options = ({"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
                   if os.name == "nt" else {"start_new_session": True})
        log = job / "process.log"
        with log.open("wb") as handle:
            process = subprocess.Popen(command, cwd=work, env=environment,
                                       stdout=handle, stderr=subprocess.STDOUT, **options)
            try:
                code = process.wait(timeout=self.configuration.timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   capture_output=True, check=False, timeout=30)
                else:
                    import signal
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=30)
                raise TimeoutError("local operation exceeded its timeout") from exc
        if code:
            with log.open("rb") as handle:
                handle.seek(max(0, log.stat().st_size - 8000))
                detail = handle.read().decode("utf-8", "replace")
            raise RuntimeError(f"local process exit {code}: {detail}")
        if config.generated_bundle:
            validate_bundle(root, config.generated_bundle)
        return output

def prepared_arrays(prepared, splits, batch_size=128, *, seed=42, corruption_sigma=None):
    import numpy as np
    from ml import make_loader
    arrays = {}
    for split in splits:
        loader = make_loader(prepared, split=split, batch_size=batch_size, shuffle=False,
                             seed=seed, corruption_sigma=corruption_sigma)
        images, targets = [], []
        for image, target in loader:
            images.append(image.numpy())
            targets.append(target.numpy())
        arrays[f"{split}_images"] = np.concatenate(images)
        arrays[f"{split}_targets"] = np.concatenate(targets)
    return arrays


@dataclass
class RemoteModel:
    root: Path
    checkpoint: Path
    checkpoint_sha256: str
    config: TrainConfig

    def predict(self, prepared, split, batch_size, *, seed=42, corruption_sigma=None):
        import numpy as np
        if sha256_file(self.checkpoint) != self.checkpoint_sha256:
            raise ValueError("run checkpoint checksum mismatch")
        data = prepared_arrays(prepared, [split], batch_size, seed=seed,
                               corruption_sigma=corruption_sigma)
        targets = data.pop(f"{split}_targets")
        output = LocalWorker(self.config.worker).execute(self.root, self.config, "infer",
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

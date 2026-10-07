"""Run-scoped code storage and fail-closed SSH/container transport."""
from __future__ import annotations

import ast
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import time
import uuid
import zipfile

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


class SSHWorker:
    def __init__(self, configuration: WorkerConfiguration):
        self.configuration = configuration

    def _ssh(self, command: str, *, stdin=None, timeout=None):
        result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            "-o", "StrictHostKeyChecking=yes", self.configuration.host, command],
            input=stdin, capture_output=True, text=True,
            timeout=timeout or self.configuration.timeout_seconds + 60, check=False)
        if result.returncode:
            raise RuntimeError("worker SSH failure: " + result.stderr[-8000:])
        return result.stdout

    def _service(self, operation: str, device: str, *extra, timeout_seconds=None):
        seconds = timeout_seconds or self.configuration.timeout_seconds
        argv = [operation, "--image", self.configuration.image, "--device", device,
                "--memory", str(self.configuration.memory_gib),
                "--timeout", str(seconds), *extra]
        source = Path(__file__).with_name("worker_service.py").read_text(encoding="utf-8")
        command = shlex.join([self.configuration.python, "-", *argv])
        return json.loads(self._ssh(command, stdin=source,
                         timeout=150 if operation == "preflight" else seconds + 120))

    def preflight(self, device: str):
        result = self._service("preflight", device)
        self.configuration = self.configuration.model_copy(update={"image": result["image_digest"]})
        return result

    def execute(self, root: Path, config: TrainConfig, operation: str, arrays: dict,
                *, checkpoint: Path | None = None, epochs: int | None = None, evidence=None,
                resume: Path | None = None, start_epoch: int = 0):
        import numpy as np
        deadline = time.monotonic() + self.configuration.timeout_seconds
        def remaining():
            value = int(deadline - time.monotonic())
            if value < 1:
                raise TimeoutError("worker operation exhausted its wall-clock budget")
            return value
        identifier = uuid.uuid4().hex
        job = root / "blobs" / "worker_jobs" / identifier
        inputs = job / "input"
        framework = inputs / "framework"
        framework.mkdir(parents=True, exist_ok=False)
        for path in Path(__file__).parent.glob("*.py"):
            shutil.copyfile(path, framework / path.name)
        if config.generated_bundle:
            source = validate_bundle(root, config.generated_bundle)
            shutil.copytree(source, inputs / "bundle")
        if checkpoint:
            shutil.copyfile(checkpoint, inputs / "model.ckpt")
        if resume:
            shutil.copyfile(resume, inputs / "resume.pt")
        np.savez(inputs / "data.npz", **arrays)
        request = {"operation": operation, "config": config.model_dump(mode="json"),
                   "epochs": epochs or config.epochs, "evidence": evidence or [],
                   "start_epoch": start_epoch, "resume": resume is not None}
        (inputs / "request.json").write_text(json.dumps(request) + "\n", encoding="utf-8")
        remote_job = self.configuration.root.rstrip("/") + "/" + identifier
        self._ssh(shlex.join(["mkdir", "-p", "--", remote_job]), timeout=min(30, remaining()))
        options = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=10"]
        upload = subprocess.run(["scp", *options, "-r", str(inputs),
                                 f"{self.configuration.host}:{remote_job}/"],
                                capture_output=True, text=True, timeout=remaining(), check=False)
        if upload.returncode:
            raise RuntimeError("worker input upload failed: " + upload.stderr[-2000:])
        result = self._service("execute", config.device, "--job", remote_job, timeout_seconds=remaining())
        if result["image_digest"] != self.configuration.image:
            raise ValueError("worker image changed during execution")
        archive = job / "response.zip"
        download = subprocess.run(["scp", *options,
            f"{self.configuration.host}:{remote_job}/response.zip", str(archive)],
            capture_output=True, text=True, timeout=remaining(), check=False)
        if download.returncode:
            raise RuntimeError("worker output download failed: " + download.stderr[-2000:])
        output = job / "output"
        output.mkdir()
        limits = {"result.json": 2_000_000, "logits.npy": 100_000_000,
                  "resume.pt": 1_073_741_824,
                  "model.ckpt": 1_073_741_824, "training.jsonl": 2_000_000,
                  "strategy.json": 200_000, "container.log": 8_000_000}
        with zipfile.ZipFile(archive) as stored:
            names = stored.namelist()
            if len(names) != len(set(names)):
                raise ValueError("duplicate worker outputs")
            for item in stored.infolist():
                if item.filename not in limits or item.file_size > limits[item.filename]:
                    raise ValueError("unexpected or oversized worker output")
                with stored.open(item) as src, (output / item.filename).open("xb") as dst:
                    shutil.copyfileobj(src, dst)
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
            raise ValueError("remote checkpoint checksum mismatch")
        data = prepared_arrays(prepared, [split], batch_size, seed=seed,
                               corruption_sigma=corruption_sigma)
        targets = data.pop(f"{split}_targets")
        output = SSHWorker(self.config.worker).execute(self.root, self.config, "infer",
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
        raise ValueError("remote checkpoints must reside in run/blobs")
    root = root.parent
    arrays = prepared_arrays(prepared, ["train", "val"], config.batch_size, seed=config.seed)
    worker = SSHWorker(config.worker)
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
        seed=config.seed, device=config.device, framework="isolated.python")
    if config.execution_mode == "agent_autonomous":
        resume_target = checkpoint.with_suffix(".resume.pt")
        shutil.copyfile(resume, resume_target)
        result_data.update(resume_path=str(resume_target.resolve()), resume_sha256=sha256_file(resume_target))
    result = TrainResult.model_validate(result_data)
    if not 1 <= result.epochs_completed <= config.epochs:
        raise ValueError("worker exceeded the epoch budget")
    return TrainingOutput(RemoteModel(root, checkpoint, result.checkpoint_sha256, config), result)

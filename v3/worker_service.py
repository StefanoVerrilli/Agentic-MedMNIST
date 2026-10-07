"""Trusted Linux service, sent over SSH stdin; never imports generated sources."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import uuid
import zipfile


def docker_flags(image: str, memory: int, device: str) -> list[str]:
    flags = ["docker", "run", "--rm", "--network=none", "--read-only",
             "--user=65534:65534", "--cap-drop=ALL",
             "--security-opt=no-new-privileges", "--pids-limit=256",
             f"--memory={memory}g", f"--memory-swap={memory}g", "--cpus=4",
             "--tmpfs=/tmp:rw,nosuid,nodev,size=512m,mode=1777",
             "--env=PYTHONDONTWRITEBYTECODE=1", "--env=HOME=/tmp",
             "--env=CUBLAS_WORKSPACE_CONFIG=:4096:8"]
    if device == "cuda":
        flags += ["--gpus=all"]
    return flags


def image_id(image: str) -> str:
    info = json.loads(subprocess.check_output(["docker", "info", "--format", "{{json .}}"], timeout=30))
    if info.get("OSType") != "linux":
        raise RuntimeError("isolated experiments require Linux containers")
    if not any("seccomp" in value for value in info.get("SecurityOptions", [])):
        raise RuntimeError("Docker seccomp protection is required")
    item = json.loads(subprocess.check_output(["docker", "image", "inspect", image], timeout=30))[0]
    if item.get("Os") != "linux":
        raise RuntimeError("worker image must be Linux")
    return item["Id"]


def preflight(image: str, device: str, memory: int) -> dict:
    digest = image_id(image)
    probe = '''import json, pathlib, socket, sys, torch, numpy, lightning, pydantic
denied = False
try:
    pathlib.Path('/original_probe').write_text('forbidden')
except OSError:
    denied = True
assert denied
assert not pathlib.Path('/var/run/docker.sock').exists()
assert not pathlib.Path('/input').exists()
assert not pathlib.Path('/root/.ssh').exists()
if DEVICE == 'cuda':
    assert torch.cuda.is_available(), 'CUDA unavailable in worker image'
print(json.dumps({'read_only_root': denied, 'network': 'none', 'user': 65534,
                  'capabilities': 'none', 'docker_socket': False, 'cuda': torch.cuda.is_available(),
                  'python_version': sys.version, 'packages': {'torch': torch.__version__,
                  'numpy': numpy.__version__, 'lightning': lightning.__version__, 'pydantic': pydantic.__version__}}))
'''.replace("DEVICE", repr(device))
    result = subprocess.run([*docker_flags(digest, memory, device), "--entrypoint=python", digest,
                             "-c", probe], check=True, capture_output=True, text=True, timeout=90)
    return {"image_digest": digest, "isolation": json.loads(result.stdout)}


def execute(job: Path, image: str, memory: int, device: str, timeout: int) -> dict:
    job = job.resolve(strict=True)
    inputs = job / "input"
    if not inputs.is_dir() or inputs.is_symlink():
        raise ValueError("missing regular input directory")
    output = job / "output"
    output.mkdir(exist_ok=False)
    output.chmod(0o777)
    # SCP preserves private controller permissions; make only job inputs readable.
    for path in [inputs, *inputs.rglob("*")]:
        if path.is_symlink():
            raise ValueError("symlinks are forbidden in worker input")
        path.chmod(0o755 if path.is_dir() else 0o644)
    digest = image_id(image)
    name = "agentic-" + uuid.uuid4().hex
    command = [*docker_flags(digest, memory, device), "--name", name,
               "--mount", f"type=bind,source={inputs},target=/input,readonly",
               "--mount", f"type=bind,source={output},target=/output",
               "--workdir=/tmp", "--entrypoint=python", digest,
               "/input/framework/worker_runtime.py"]
    log = job / "container.log"
    try:
        with log.open("wb") as handle:
            result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                                    timeout=timeout, check=False)
        if result.returncode:
            raise RuntimeError(f"container exit {result.returncode}: " + log.read_bytes()[-8000:].decode("utf-8", "replace"))
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30, check=False)
    # Only fixed, regular outputs are exported. No generated archive is trusted.
    limits = {"result.json": 2_000_000, "logits.npy": 100_000_000,
              "resume.pt": 1_073_741_824,
              "model.ckpt": 1_073_741_824, "training.jsonl": 2_000_000,
              "strategy.json": 200_000}
    archive = job / "response.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as target:
        for filename, limit in limits.items():
            path = output / filename
            if not path.exists() and not path.is_symlink():
                continue
            if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
                raise ValueError(f"invalid worker output: {filename}")
            target.write(path, filename)
        if log.stat().st_size > 8_000_000:
            log.write_bytes(log.read_bytes()[-8_000_000:])
        target.write(log, "container.log")
    return {"image_digest": digest, "archive": str(archive)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["preflight", "execute"])
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--memory", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--job")
    args = parser.parse_args()
    if args.operation == "preflight":
        result = preflight(args.image, args.device, args.memory)
    else:
        result = execute(Path(args.job), args.image, args.memory, args.device, args.timeout)
    print(json.dumps(result))


if __name__ == "__main__":
    main()

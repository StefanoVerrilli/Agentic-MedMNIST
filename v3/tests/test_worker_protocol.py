from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from contracts import TrainConfig, WorkerConfiguration
from remote import SSHWorker
from worker_service import docker_flags, execute


class WorkerProtocolTests(unittest.TestCase):
    def test_sandbox_arguments_and_gpu_are_explicit(self):
        cpu = docker_flags("image", 16, "cpu")
        for required in ("--network=none", "--read-only", "--user=65534:65534", "--cap-drop=ALL",
                         "--security-opt=no-new-privileges", "--pids-limit=256", "--memory=16g",
                         "--memory-swap=16g"):
            self.assertIn(required, cpu)
        self.assertNotIn("--gpus=all", cpu)
        self.assertIn("--gpus=all", docker_flags("image", 16, "cuda"))
        self.assertNotIn("--privileged", cpu)

    def test_timeout_always_removes_the_remote_container(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory)
            (job / "input").mkdir()
            with patch("worker_service.image_id", return_value="sha256:" + "a" * 64), \
                 patch("worker_service.subprocess.run", side_effect=[subprocess.TimeoutExpired("docker", 1),
                     SimpleNamespace(returncode=0)]) as called, self.assertRaises(subprocess.TimeoutExpired):
                execute(job, "image", 1, "cpu", 1)
            self.assertEqual(called.call_args.args[0][:3], ["docker", "rm", "-f"])

    def test_container_failure_is_reported_and_not_exported(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory)
            (job / "input").mkdir()
            with patch("worker_service.image_id", return_value="sha256:" + "a" * 64), \
                 patch("worker_service.subprocess.run", side_effect=[SimpleNamespace(returncode=137),
                     SimpleNamespace(returncode=0)]), self.assertRaisesRegex(RuntimeError, "137"):
                execute(job, "image", 1, "cpu", 1)
            self.assertFalse((job / "response.zip").exists())

    def test_controller_rejects_traversal_archive_and_image_changes(self):
        configuration = WorkerConfiguration(host="worker", root="/srv/jobs", image="sha256:" + "a" * 64)
        config = TrainConfig(lr=.001, epochs=1, hidden=16, batch_size=8, weight_decay=0,
                            class_weighting=False, seed=42, device="cpu", rationale="test", source="test", worker=configuration)
        with tempfile.TemporaryDirectory() as directory:
            worker = SSHWorker(configuration)
            def transfer(command, **kwargs):
                if "-r" not in command:
                    with zipfile.ZipFile(command[-1], "w") as stored:
                        stored.writestr("../escape.py", "pass")
                return SimpleNamespace(returncode=0, stderr="")
            with patch.object(worker, "_ssh", return_value=""), \
                 patch.object(worker, "_service", return_value={"image_digest": configuration.image}), \
                 patch("remote.subprocess.run", side_effect=transfer), self.assertRaisesRegex(ValueError, "unexpected"):
                worker.execute(Path(directory), config, "infer", {"images": np.zeros((1, 3, 28, 28))})
            self.assertFalse((Path(directory) / "escape.py").exists())
            with patch.object(worker, "_ssh", return_value=""), \
                 patch.object(worker, "_service", return_value={"image_digest": "sha256:" + "b" * 64}), \
                 patch("remote.subprocess.run", return_value=SimpleNamespace(returncode=0, stderr="")), \
                 self.assertRaisesRegex(ValueError, "image changed"):
                worker.execute(Path(directory), config, "infer", {"images": np.zeros((1, 3, 28, 28))})

    def test_ssh_transport_requires_known_host_and_key_authentication(self):
        worker = SSHWorker(WorkerConfiguration(host="worker", root="/srv/jobs", image="test/image:v3"))
        with patch("remote.subprocess.run", return_value=SimpleNamespace(returncode=1,
                    stdout="", stderr="host unavailable")) as called, self.assertRaises(RuntimeError):
            worker.preflight("cpu")
        command = called.call_args.args[0]
        self.assertIn("BatchMode=yes", command)
        self.assertIn("StrictHostKeyChecking=yes", command)

"""Small synthetic subprocess checks; never launch a dataset pipeline."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from autonomous import experiment_config
from contracts import AutonomousExperimentDecision, Blackboard, GeneratedSource, TrainConfig, WorkerConfiguration
from generated_agents import fallback_decision
from ml import prepare_data, train_model
from remote import LocalWorker, archive_bundle
from contracts import RepresentationDecision
from tests.helpers import make_bundle


class LocalExecutionTests(unittest.TestCase):
    def test_timeout_terminates_spawned_descendant_and_records_process_exit(self):
        from generated_agents import generated_config
        from resources import RunResources
        source = ("import subprocess, sys, time\n"
                  "flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0\n"
                  "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], creationflags=flags)\n"
                  "print('DESCENDANT_PID=' + str(child.pid), flush=True)\n"
                  "time.sleep(60)\n")
        with tempfile.TemporaryDirectory() as directory:
            bb = Blackboard(Path(directory) / "run")
            decision = fallback_decision().model_copy(update={"files": [GeneratedSource(path="experiment.py", code=source)]})
            reference = archive_bundle(bb, decision, "test")
            configuration = WorkerConfiguration(timeout_seconds=15)
            config = generated_config(decision, reference, configuration, 42, "cpu", 1, "test")
            resources = RunResources(bb.root, audit=bb.record_event)
            with resources.activate(), self.assertRaises(TimeoutError):
                LocalWorker(configuration).execute(bb.root, config, "verify", {})
            log = next((bb.blob_dir / "local_jobs").glob("*/process.log")).read_text(encoding="utf-8")
            pid_line = next(row for row in log.splitlines() if row.startswith("DESCENDANT_PID="))
            descendant_pid = int(pid_line.split("=")[1])
            if os.name == "nt":
                import ctypes
                from ctypes import wintypes
                api = ctypes.WinDLL("kernel32", use_last_error=True)
                api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
                api.OpenProcess.restype = wintypes.HANDLE
                api.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
                api.WaitForSingleObject.restype = wintypes.DWORD
                api.CloseHandle.argtypes = [wintypes.HANDLE]
                handle = api.OpenProcess(0x1000 | 0x100000, False, descendant_pid)
                if handle:
                    try:
                        self.assertEqual(api.WaitForSingleObject(handle, 5000), 0)
                    finally:
                        api.CloseHandle(handle)
                else:
                    self.assertEqual(ctypes.get_last_error(), 87)  # PID no longer exists
            else:
                status = Path(f"/proc/{descendant_pid}/stat")
                if status.exists():
                    self.assertEqual(status.read_text().split(")", 1)[1].split()[0], "Z")
            events = [json.loads(row) for row in (bb.root / "decision_log.jsonl").read_text().splitlines()]
            self.assertTrue(any(row["event"] == "local_job_process_exited" and row["timed_out"] for row in events))

    def test_no_remote_arguments_or_worker_are_required(self):
        from run import build_parser, validate_args, worker_configuration
        args = build_parser().parse_args(["--execution-mode", "agent_autonomous", "--device", "cpu",
            "--ollama-base", "http://localhost:11434"])
        validate_args(args)
        self.assertEqual(worker_configuration(args), WorkerConfiguration())
        worker = LocalWorker()
        evidence = worker.preflight("cpu")
        self.assertEqual(evidence["python_executable"], sys.executable)
        self.assertFalse(evidence["isolation"]["enforced"])
        options = {option for action in build_parser()._actions for option in action.option_strings}
        self.assertNotIn("--worker-host", options)
        self.assertNotIn("--worker-image", options)

    def test_same_interpreter_and_run_scoped_working_directory(self):
        config = TrainConfig(lr=.001, epochs=1, hidden=16, batch_size=8, weight_decay=0,
            class_weighting=False, seed=42, device="cpu", rationale="test", source="test")
        with tempfile.TemporaryDirectory() as directory, patch("remote.subprocess.Popen") as launch, \
                patch("process_control.WindowsProcessTree"):
            process = MagicMock()
            process.wait.return_value = 0
            launch.return_value = process
            output = LocalWorker().execute(Path(directory), config, "strategy", {})
            command = launch.call_args.args[0]
            self.assertEqual(command[0], sys.executable)
            self.assertEqual(command[-2], "--job")
            self.assertTrue(launch.call_args.kwargs["cwd"].is_relative_to(Path(directory)))
            self.assertTrue(output.is_relative_to(Path(directory)))
            self.assertNotIn("shell", launch.call_args.kwargs)

    def test_actual_child_process_imports_bundle_only_in_job_and_reports_failure(self):
        from generated_agents import generated_config
        with tempfile.TemporaryDirectory() as directory:
            bb = Blackboard(Path(directory) / "run")
            decision = fallback_decision().model_copy(update={"files": [GeneratedSource(path="experiment.py",
                code="import os\nraise RuntimeError('local child failure pid=' + str(os.getpid()))\n")]})
            reference = archive_bundle(bb, decision, "test")
            config = generated_config(decision, reference, WorkerConfiguration(), 42, "cpu", 1, "test")
            with self.assertRaisesRegex(RuntimeError, "local child failure pid="):
                LocalWorker().execute(bb.root, config, "verify", {})
            failure = json.loads(next((bb.blob_dir / "local_jobs").glob("*/output/failure.json")).read_text())
            self.assertEqual(failure["failure_kind"], "code_validation")
            self.assertEqual(failure["operation"], "verify")
            self.assertEqual((bb.root / "generated" / decision.bundle_id / "v001" / "experiment.py").read_text(), decision.files[0].code)

    def test_timeout_stops_child_and_preserves_job_log(self):
        from generated_agents import generated_config
        with tempfile.TemporaryDirectory() as directory:
            bb = Blackboard(Path(directory) / "run")
            decision = fallback_decision().model_copy(update={"files": [GeneratedSource(path="experiment.py",
                code="import time\ntime.sleep(60)\n")]})
            reference = archive_bundle(bb, decision, "test")
            worker = WorkerConfiguration(timeout_seconds=1)
            config = generated_config(decision, reference, worker, 42, "cpu", 1, "test")
            with self.assertRaisesRegex(TimeoutError, "local operation"):
                LocalWorker(worker).execute(bb.root, config, "verify", {})
            self.assertEqual(len(list((bb.blob_dir / "local_jobs").glob("*/process.log"))), 1)

    def test_actual_local_adaptive_training_and_checkpoint_continuation(self):
        source = "import torch\ntorch.set_num_threads(1)\n" + Path(__file__).resolve().parents[1].joinpath("default_experiment.py").read_text(encoding="utf-8")
        decision = AutonomousExperimentDecision(bundle_id="local_probe", hypothesis="Verify local run-scoped continuation",
            files=[GeneratedSource(path="experiment.py", code=source)], epochs=2, batch_size=9,
            parameters={"hidden": 16, "depth": 1, "scales": [4, 7], "dropout": .1},
            training=dict(lr=.001, weight_decay=.0001, class_weighting=False, optimizer="adamw", scheduler="none",
                label_smoothing=0., early_stopping_patience=0, early_stopping_monitor="val_accuracy",
                early_stopping_min_delta=0., gradient_clip_val=1., adam_beta1=.9, adam_beta2=.999, optimizer_eps=1e-8))
        with tempfile.TemporaryDirectory() as directory:
            bb = Blackboard(Path(directory) / "run")
            reference = archive_bundle(bb, decision, "test")
            config = experiment_config(decision, reference, WorkerConfiguration(), 42, "cpu", "test")
            verified = LocalWorker().execute(bb.root, config, "verify", {}, epochs=1)
            self.assertEqual(json.loads((verified / "result.json").read_text(encoding="utf-8")), {"verified": True})
            prepared = prepare_data(make_bundle(), RepresentationDecision(normalization="standardize",
                augmentations=[], rationale="Synthetic local test data"))
            continuous = train_model(prepared, config, bb.blob_dir / "continuous.ckpt")
            segmented = train_model(prepared, config.model_copy(update={"segment_targets": [1, 2]}), bb.blob_dir / "segmented.ckpt")
            self.assertEqual(continuous.result.history, segmented.result.history)
            self.assertTrue(Path(segmented.result.resume_path).is_file())
            np.testing.assert_array_equal(continuous.model.predict(prepared, "val", 9)[1],
                                          segmented.model.predict(prepared, "val", 9)[1])
            self.assertEqual(segmented.result.framework, "local.python")

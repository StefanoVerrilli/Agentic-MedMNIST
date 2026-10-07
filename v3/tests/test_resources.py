"""GPU handoff uses a simulated Ollama server; no real models are unloaded."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from contracts import TrainConfig, WorkerConfiguration
from failures import WorkerOperationError
from remote import LocalWorker
from resources import RunResources, current_resources
from llm import OllamaReasoner
from replay import CachedReasoner


class ResourceTests(unittest.TestCase):
    def test_next_decision_reloads_after_handoff_and_reports_loading_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls, events = [], []
            loaded = [False]
            def chat(url, payload, timeout):
                calls.append("chat")
                self.assertEqual(payload["keep_alive"], "5m")
                loaded[0] = True
                return {"message": {"content": json.dumps(self.config().model_dump(mode="json"))}, "load_duration": 123}
            def unload(*args):
                calls.append("unload")
                loaded[0] = False
                return {"done": True}
            resources = RunResources(root, base_url="http://localhost:11434", model="test",
                post=unload, get=lambda *args: {"models": [{"name": "test:latest"}] if loaded[0] else []},
                audit=lambda name, **kw: events.append((name, kw)))
            reasoner = CachedReasoner(OllamaReasoner("http://localhost:11434", model="test", transport=chat), root)
            with resources.activate():
                for stage in ("first", "second"):
                    reasoner.decide(stage=stage, system="test", user="test", response_model=TrainConfig,
                                    fallback=self.config(), audit=resources.record)
                    if stage == "first":
                        resources.release_for_worker("cuda")
            self.assertEqual(calls, ["chat", "unload", "chat"])
            self.assertEqual([kw["load_duration_ns"] for name, kw in events if name == "llm_request_finished"], [123, 123])
            self.assertEqual(sum(name == "llm_decision_finished" for name, _ in events), 2)

    def config(self):
        return TrainConfig(lr=.001, epochs=1, hidden=16, batch_size=8, weight_decay=0,
            class_weighting=False, seed=42, device="cuda", rationale="Test handoff", source="test")

    def test_unload_verified_before_launch_and_scope_is_restored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls, events = [], []
            snapshots = iter([{"models": [{"name": "test:latest", "size_vram": 100}]},
                              {"models": [{"name": "another:latest"}]}])
            resources = RunResources(root, base_url="http://localhost:11434/api", model="test",
                post=lambda url, payload, timeout: calls.append(("unload", url, payload)) or {"done": True},
                get=lambda url, timeout: calls.append(("ps", url)) or next(snapshots), sleep=lambda _: None,
                audit=lambda event, **details: events.append((event, details)))
            process = MagicMock()
            process.wait.return_value = 0
            def launch(*args, **kwargs):
                calls.append(("launch",))
                return process
            with resources.activate(), patch("remote.subprocess.Popen", side_effect=launch), \
                    patch("process_control.WindowsProcessTree"):
                self.assertIs(current_resources(), resources)
                output = LocalWorker().execute(root, self.config(), "strategy", {})
            self.assertIsNone(current_resources())
            self.assertEqual([row[0] for row in calls], ["unload", "ps", "ps", "launch"])
            self.assertEqual(calls[0][2], {"model": "test", "prompt": "", "stream": False, "keep_alive": 0})
            self.assertTrue(output.is_relative_to(root))
            names = [event for event, _ in events]
            for name in ("local_job_arrays_saved", "ollama_unload_finished", "local_job_process_started",
                         "local_job_process_exited", "local_job_completed"):
                self.assertIn(name, names)

    def test_deadline_or_invalid_response_prevents_worker_launch(self):
        for malformed in (False, True):
            with self.subTest(malformed=malformed), tempfile.TemporaryDirectory() as directory:
                now = [0.0]
                resources = RunResources(Path(directory), base_url="http://localhost:11434", model="test",
                    timeout=2., monotonic=lambda: now[0], sleep=lambda seconds: now.__setitem__(0, now[0]+seconds),
                    post=lambda *args: {"done": True},
                    get=lambda *args: {} if malformed else {"models": [{"model": "test:latest"}]})
                with resources.activate(), patch("remote.subprocess.Popen") as launch:
                    with self.assertRaises(WorkerOperationError) as raised:
                        LocalWorker().execute(Path(directory), self.config(), "verify", {})
                    self.assertEqual(raised.exception.failure_kind, "resource_handoff")
                    launch.assert_not_called()

    def test_cpu_offline_and_replay_do_not_contact_ollama(self):
        with tempfile.TemporaryDirectory() as directory:
            for base_url, replay, device in ((None, False, "cuda"),
                    ("http://localhost:11434", True, "cuda"), ("http://localhost:11434", False, "cpu")):
                post, get = MagicMock(), MagicMock()
                resources = RunResources(Path(directory), base_url=base_url, replay=replay, model="test", post=post, get=get)
                resources.release_for_worker(device)
                post.assert_not_called()
                get.assert_not_called()

    def test_failed_unload_is_terminal_and_scope_is_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            resources = RunResources(Path(directory), base_url="http://localhost:11434", model="test",
                                     post=MagicMock(side_effect=OSError("server unavailable")))
            with self.assertRaises(WorkerOperationError), resources.activate():
                resources.release_for_worker("cuda")
            self.assertIsNone(current_resources())

    def test_heartbeat_does_not_reset_process_deadline(self):
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            now = [0.0]
            events = []
            resources = RunResources(Path(directory), audit=lambda name, **kw: events.append(name))
            process = MagicMock()
            def wait(timeout):
                now[0] += timeout
                if now[0] < 20:
                    raise subprocess.TimeoutExpired("worker", timeout)
                return 0
            process.wait.side_effect = wait
            with resources.activate(), patch("remote.subprocess.Popen", return_value=process), \
                    patch("process_control.WindowsProcessTree"), \
                    patch("remote.time.monotonic", side_effect=lambda: now[0]):
                LocalWorker(WorkerConfiguration(timeout_seconds=20)).execute(
                    Path(directory), self.config().model_copy(update={"device": "cpu"}), "strategy", {})
            self.assertIn("local_job_process_heartbeat", events)
            self.assertEqual([call.kwargs["timeout"] for call in process.wait.call_args_list], [15., 5.])

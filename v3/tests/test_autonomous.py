"""Unit tests with synthetic data and a transport double; no real pipeline jobs."""
import contextlib
import io
import json
from pathlib import Path
import random
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from adaptive_training import fit_adaptive, validate_resume
from autonomous import AutonomousSearchAgent, continuation_identity, experiment_config
from contracts import (AutonomousExperimentDecision, AutonomousSearchDecision, Blackboard,
                       SearchPlan, TrainConfig, sha256_file)
from generated_agents import fallback_decision
from llm import ReasonedDecision
from replay import CachedReasoner
from tests import test_generated as generated_test_support
from tests.test_generated import SimulatedWorker, WORKER


def experiment(epochs=2, **training_updates):
    training = dict(lr=.002, weight_decay=.0001, class_weighting=False, optimizer="adamw",
        scheduler="none", label_smoothing=.0, early_stopping_patience=0,
        early_stopping_monitor="val_accuracy", early_stopping_min_delta=0., gradient_clip_val=1.,
        adam_beta1=.9, adam_beta2=.999, optimizer_eps=1e-8)
    training.update(training_updates)
    # Untrusted experiment is archived, never imported or run in these tests.
    payload = fallback_decision().model_dump()
    payload.update(epochs=epochs, batch_size=8, training=training,
        files=[{"path": "experiment.py", "code": "raise RuntimeError('never execute on controller')\n"}])
    return AutonomousExperimentDecision.model_validate(payload)


class ScriptedReasoner:
    model, seed, enabled = "scripted-agent", 42, True

    def __init__(self, actions, *, fallback=False):
        self.actions, self.calls, self.fallback = list(actions), [], fallback

    def decide(self, **kwargs):
        self.calls.append(kwargs)
        value = kwargs["response_model"].model_validate(self.actions.pop(0))
        if kwargs.get("audit"):
            kwargs["audit"]("llm_decision", stage=kwargs["stage"], source=self.model, decision=value.model_dump(mode="json"))
        return ReasonedDecision(value, self.model, self.fallback, 1, None)


class AdaptiveWorker(SimulatedWorker):
    def execute(self, root, config, operation, arrays, **kwargs):
        output = super().execute(root, config, operation, arrays, **kwargs)
        if operation == "train":
            path = output / "result.json"
            result = json.loads(path.read_text(encoding="utf-8"))
            result["stop_reason"] = "segment_complete"
            path.write_text(json.dumps(result), encoding="utf-8")
            (output / "resume.pt").write_bytes(b"opaque latest-state checkpoint")
        return output


class AutonomousTests(unittest.TestCase):
    def setUp(self):
        SimulatedWorker.calls = []

    def populate(self, directory):
        return generated_test_support.GeneratedTests().populate(Path(directory) / "run")[0]

    def actions(self):
        return [dict(action="new_trial", rationale="Test initial training", experiment=experiment().model_dump()),
                dict(action="continue_trial", rationale="Validation suggests additional training", candidate_id="autonomous_t0001", additional_epochs=3),
                dict(action="finish_search", rationale="Sufficient validation evidence")]

    def test_cli_rejects_external_budgets_and_requires_live_decisions(self):
        from run import build_parser, validate_args
        base = ["--execution-mode", "agent_autonomous",
                "--ollama-base", "http://localhost:11434"]
        args = build_parser().parse_args(base)
        validate_args(args)
        self.assertTrue(args.allow_generated_code and args.skip_baseline and args.require_llm)
        self.assertIsNone(args.max_epochs)
        for flag in ("--max-epochs", "--search-epochs", "--search-rounds", "--search-trials", "--max-generated-bundles"):
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, "external search budgets"):
                validate_args(build_parser().parse_args([*base, flag, "1"]))
        for flag in ("--offline", "--no-search"):
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                validate_args(build_parser().parse_args([*base, flag]))

    def test_llm_output_budget_is_configurable_independently_of_epochs(self):
        from run import build_parser, validate_args
        base = ["--execution-mode", "agent_autonomous", "--ollama-base", "http://localhost:11434"]
        with patch.dict("os.environ", {"AGENTIC_LLM_NUM_PREDICT": "32768"}):
            args = build_parser().parse_args(base)
        validate_args(args)
        self.assertEqual(args.llm_num_predict, 32768)
        self.assertIsNone(args.max_epochs)
        with self.assertRaisesRegex(ValueError, "llm-num-predict"):
            validate_args(build_parser().parse_args([*base, "--llm-num-predict", "0"]))

    def test_code_only_pipeline_double_skips_baseline_freezes_and_replays(self):
        from agents import IngestionAgent
        from governance import read_artefact
        from run import build_parser, resolve_limits, run_once, validate_args
        from tests.helpers import make_bundle
        actions = self.actions()
        class PipelineReasoner(ScriptedReasoner):
            def decide(self, **kwargs):
                if kwargs["response_model"] is AutonomousSearchDecision:
                    return super().decide(**kwargs)
                value = kwargs["response_model"].model_validate(kwargs["fallback"])
                if kwargs.get("audit"):
                    kwargs["audit"]("llm_decision", stage=kwargs["stage"], source=self.model, decision=value.model_dump(mode="json"))
                return ReasonedDecision(value, self.model, False, 1, None)
        def ingestion(**kwargs):
            loader = kwargs.pop("loader", lambda **kw: make_bundle())
            return IngestionAgent(loader=loader, **kwargs)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), \
                patch("run.OllamaReasoner", side_effect=lambda **kw: PipelineReasoner(actions)), \
                patch("run.ollama_model_digest", return_value="a" * 64), \
                patch("run.IngestionAgent", side_effect=ingestion), \
                patch("run.run_baseline", side_effect=AssertionError("baseline must not execute")), \
                patch("autonomous.LocalWorker", AdaptiveWorker), patch("remote.LocalWorker", AdaptiveWorker):
            args = build_parser().parse_args(["--execution-mode", "agent_autonomous", "--device", "cpu",
                "--ollama-base", "http://simulated-llm"])
            validate_args(args)
            root = Path(directory)
            original = root / "original" / "seed_42"
            summary = run_once(args, 42, original, resolve_limits(args))
            self.assertTrue(summary["status"].startswith("completed"), summary)
            self.assertIsNone(summary["baseline_accuracy"])
            self.assertEqual(summary["baseline_status"], "not_executed")
            self.assertIsNone(summary["accuracy_delta"])
            acceptance = read_artefact(original, "acceptance_report")
            criterion = next(item for item in acceptance["criteria"] if item["requirement"] == "WP4_BASELINE_COMPARISON")
            self.assertEqual(criterion["status"], "failed")
            best = summary["_best_configuration"]
            frozen = run_once(args, 43, root / "original" / "seed_43", resolve_limits(args), frozen_best=best)
            self.assertTrue(frozen["status"].startswith("completed"), frozen)
            args.replay_run = str(original)
            repeated = run_once(args, 42, root / "replay" / "seed_42", resolve_limits(args))
            self.assertTrue(repeated["status"].startswith("completed"), repeated)
            report = json.loads((root / "replay" / "seed_42" / "replay_comparison.json").read_text(encoding="utf-8"))
            self.assertTrue(report["payload"]["matched"], report)

    def test_unbounded_agent_epochs_and_explicit_active_options(self):
        value = experiment(5001, early_stopping_patience=2000)
        payload = value.model_dump()
        del payload["epochs"]
        with self.assertRaises(ValueError):
            AutonomousExperimentDecision.model_validate(payload)
        payload = value.model_dump()
        del payload["training"]["adam_beta1"]
        with self.assertRaises(ValueError):
            AutonomousExperimentDecision.model_validate(payload)
        SearchPlan(policy="agent_autonomous", progressive_budget=False)
        with self.assertRaises(ValueError):
            SearchPlan(policy="agent_autonomous", progressive_budget=False, max_trials=4)
        with tempfile.TemporaryDirectory() as directory:
            from remote import archive_bundle
            bb = self.populate(directory)
            reference = archive_bundle(bb, value, "test")
            config = experiment_config(value, reference, WORKER, 42, "cpu", "agent")
            self.assertEqual((config.epochs, config.early_stopping_patience), (5001, 2000))
            with self.assertRaises(ValueError):
                TrainConfig.model_validate({**config.model_dump(), "execution_mode": "legacy"})

    def test_agent_search_continues_exact_requested_target_and_freezes(self):
        with tempfile.TemporaryDirectory() as directory, patch("autonomous.LocalWorker", AdaptiveWorker), patch("remote.LocalWorker", AdaptiveWorker):
            bb = self.populate(directory)
            reasoner = ScriptedReasoner(self.actions())
            AutonomousSearchAgent(reasoner, worker=WORKER, seed=42, device="cpu").run(bb)
            best = bb.get("best_configuration")
            self.assertEqual(best.train_config.epochs, 5)
            self.assertEqual(best.train_config.segment_targets, [2, 5])
            self.assertEqual(bb.get("selected_training_result").epochs_completed, 5)
            self.assertEqual(bb.get("search_plan").policy, "agent_autonomous")
            training = [call for call in SimulatedWorker.calls if call[0] == "train"]
            self.assertEqual([call[3]["start_epoch"] for call in training], [0, 2])
            self.assertIsNotNone(training[1][3]["resume"])
            self.assertTrue(all("test_targets" not in keys for _, keys, _, _ in SimulatedWorker.calls))
            self.assertEqual(Blackboard.open(bb.root).get("autonomous_action_0003").decision.action, "finish_search")
            from agents import TrainingAgent
            with patch("agents.train_model", side_effect=AssertionError("must use frozen checkpoint")):
                TrainingAgent().run(bb)
            self.assertEqual(bb.get("train_result").checkpoint_sha256, bb.get("selected_training_result").checkpoint_sha256)

    def test_repeated_seed_uses_recorded_segments(self):
        from ml import train_model
        with tempfile.TemporaryDirectory() as directory, patch("autonomous.LocalWorker", AdaptiveWorker), patch("remote.LocalWorker", AdaptiveWorker):
            bb = self.populate(directory)
            AutonomousSearchAgent(ScriptedReasoner(self.actions()), worker=WORKER, seed=42, device="cpu").run(bb)
            config = bb.get("best_configuration").train_config.model_copy(update={"seed": 43})
            before = len(SimulatedWorker.calls)
            output = train_model(bb.get_blob("prepared_data"), config, bb.blob_dir / "repeated.ckpt")
            training = [call for call in SimulatedWorker.calls[before:] if call[0] == "train"]
            self.assertEqual([call[3]["start_epoch"] for call in training], [0, 2])
            self.assertEqual(output.result.epochs_completed, 5)

    def test_decision_transcript_replays_without_backend_calls(self):
        with tempfile.TemporaryDirectory() as directory, patch("autonomous.LocalWorker", AdaptiveWorker), patch("remote.LocalWorker", AdaptiveWorker):
            first = self.populate(Path(directory) / "first")
            backend = ScriptedReasoner(self.actions())
            recorded = CachedReasoner(backend, first.root)
            AutonomousSearchAgent(recorded, worker=WORKER, seed=42, device="cpu").run(first)
            second = self.populate(Path(directory) / "second")
            unused = ScriptedReasoner([])
            replay = CachedReasoner(unused, second.root, replay_root=first.root)
            AutonomousSearchAgent(replay, worker=WORKER, seed=42, device="cpu").run(second)
            replay.assert_replay_complete()
            self.assertEqual(unused.calls, [])
            self.assertEqual(first.get("best_configuration").selected_config_hash, second.get("best_configuration").selected_config_hash)

    def test_fallback_and_unknown_continuation_fail_without_training(self):
        for actions, fallback in (([dict(action="finish_search", rationale="fallback")], True),
                                  ([dict(action="continue_trial", rationale="invalid parent", candidate_id="missing", additional_epochs=1)], False)):
            with tempfile.TemporaryDirectory() as directory, patch("autonomous.LocalWorker", side_effect=AssertionError("no job should start")):
                bb = self.populate(directory)
                with self.assertRaises(ValueError):
                    AutonomousSearchAgent(ScriptedReasoner(actions, fallback=fallback), worker=WORKER, seed=42, device="cpu").run(bb)

    def test_failed_segment_is_not_ranked_as_completed(self):
        class TimeoutWorker(AdaptiveWorker):
            def execute(self, root, config, operation, arrays, **kwargs):
                if operation == "train":
                    raise TimeoutError("worker time exhausted")
                return super().execute(root, config, operation, arrays, **kwargs)
        actions = [self.actions()[0], self.actions()[-1]]
        with tempfile.TemporaryDirectory() as directory, patch("autonomous.LocalWorker", TimeoutWorker):
            bb = self.populate(directory)
            with self.assertRaisesRegex(ValueError, "without a successful trial"):
                AutonomousSearchAgent(ScriptedReasoner(actions), worker=WORKER, seed=42, device="cpu").run(bb)
            self.assertEqual(bb.get("trial_0001").status, "failed")
            self.assertIsNone(bb.get_optional("best_configuration"))

    def test_memory_interruptions_are_explicit_and_resume_checksums_are_audited(self):
        class MemoryWorker(AdaptiveWorker):
            def execute(self, root, config, operation, arrays, **kwargs):
                if operation == "train":
                    raise RuntimeError("local process exit 137: out of memory")
                return super().execute(root, config, operation, arrays, **kwargs)
        with tempfile.TemporaryDirectory() as directory, patch("autonomous.LocalWorker", MemoryWorker):
            bb = self.populate(directory)
            with self.assertRaises(ValueError):
                AutonomousSearchAgent(ScriptedReasoner([self.actions()[0], self.actions()[-1]]),
                    worker=WORKER, seed=42, device="cpu").run(bb)
            self.assertEqual(bb.get("trial_0001").stop_reason, "memory")
        with tempfile.TemporaryDirectory() as directory, patch("autonomous.LocalWorker", AdaptiveWorker), patch("remote.LocalWorker", AdaptiveWorker):
            from governance import check_integrity
            bb = self.populate(directory)
            AutonomousSearchAgent(ScriptedReasoner(self.actions()), worker=WORKER, seed=42, device="cpu").run(bb)
            state = bb.root / bb.get("selected_training_result").resume_path
            state.write_bytes(b"tampered")
            self.assertTrue(any("resume_path" in issue for issue in check_integrity(bb.root)))

    def test_ranking_accepts_different_durations_only_in_autonomous_policy(self):
        from search import rank_trials
        actions = [self.actions()[0], dict(action="new_trial", rationale="Alternative architecture",
                    experiment=experiment(3).model_copy(update={"parameters": {"hidden": 32}}).model_dump()), self.actions()[-1]]
        with tempfile.TemporaryDirectory() as directory, patch("autonomous.LocalWorker", AdaptiveWorker), patch("remote.LocalWorker", AdaptiveWorker):
            bb = self.populate(directory)
            AutonomousSearchAgent(ScriptedReasoner(actions), worker=WORKER, seed=42, device="cpu").run(bb)
            trials = [bb.get("trial_0001"), bb.get("trial_0002")]
            with self.assertRaises(ValueError):
                rank_trials(trials, accuracy_tolerance=.005)
            rank_trials(trials, accuracy_tolerance=.005, equal_epoch_budgets=False)


class AdaptiveTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def context(self, directory, epochs):
        rng = np.random.default_rng(77)
        config = dict(execution_mode="agent_autonomous", **experiment().training.model_dump(),
            epochs=epochs, batch_size=8, seed=42, device="cpu", generated_bundle=None)
        return dict(config=config, output=str(directory), epochs=epochs, start_epoch=0, resume=None,
            checkpoint=str(Path(directory) / "model.ckpt"), data={
                "train_images": rng.normal(size=(18, 3, 28, 28)).astype("float32"), "train_targets": np.tile(np.arange(9), 2),
                "val_images": rng.normal(size=(9, 3, 28, 28)).astype("float32"), "val_targets": np.arange(9)})

    def model(self):
        return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 28 * 28, 16),
            torch.nn.ReLU(), torch.nn.Dropout(.2), torch.nn.Linear(16, 9))

    def seed(self):
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

    def test_continuous_and_resumed_training_match_exactly(self):
        for scheduler in ("none", "reduce_on_plateau"):
            with self.subTest(scheduler=scheduler), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for name in ("full", "first", "last"):
                    (root / name).mkdir()
                full = self.context(root / "full", 4)
                full["config"].update(scheduler=scheduler)
                self.seed()
                full_result = fit_adaptive(full, self.model())
                first = self.context(root / "first", 2)
                first["config"].update(scheduler=scheduler)
                self.seed()
                fit_adaptive(first, self.model())
                last = self.context(root / "last", 4)
                last["config"].update(scheduler=scheduler)
                last.update(start_epoch=2, resume=str(root / "first" / "resume.pt"), checkpoint=str(root / "first" / "model.ckpt"))
                resumed_result = fit_adaptive(last, self.model())
                self.assertEqual(full_result, resumed_result)
                a = torch.load(root / "full" / "resume.pt", weights_only=False)
                b = torch.load(root / "last" / "resume.pt", weights_only=False)
                for key in a["model"]:
                    torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
                torch.testing.assert_close(a["loader_rng"], b["loader_rng"], rtol=0, atol=0)
                validate_resume(last, resumed_result)

    def test_resume_rejects_changed_optimizer_data_or_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "first").mkdir()
            first = self.context(root / "first", 1)
            self.seed()
            fit_adaptive(first, self.model())
            for variant in ("optimizer", "data", "epoch"):
                last = self.context(root, 2)
                last.update(start_epoch=1, resume=str(root / "first" / "resume.pt"), checkpoint=str(root / "first" / "model.ckpt"))
                if variant == "optimizer": last["config"]["lr"] = .01
                elif variant == "data": last["data"]["train_images"][0, 0, 0, 0] += 1
                else: last["start_epoch"] = 0
                with self.subTest(variant=variant), self.assertRaisesRegex(ValueError, "incompatible"):
                    fit_adaptive(last, self.model())

    def test_best_and_latest_states_are_separate_and_early_stop_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            context = self.context(directory, 6)
            context["config"].update(early_stopping_patience=1, early_stopping_min_delta=1.)
            self.seed()
            result = fit_adaptive(context, self.model())
            self.assertEqual((result["stop_reason"], result["epochs_completed"], result["best_epoch"]), ("early_stopping", 2, 1))
            state = torch.load(Path(directory) / "resume.pt", weights_only=False)
            best = torch.load(Path(directory) / "model.ckpt", weights_only=True)
            self.assertTrue(any(not torch.equal(best[key], state["model"][key]) for key in best))
            validate_resume(context, result)
            context.update(start_epoch=2, resume=str(Path(directory) / "resume.pt"))
            with self.assertRaises(ValueError):
                fit_adaptive(context, self.model())

    def test_scheduler_state_is_preserved_and_exhausted_one_cycle_is_rejected(self):
        for scheduler in ("cosine", "one_cycle"):
            with self.subTest(scheduler=scheduler), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "first").mkdir()
                (root / "last").mkdir()
                first = self.context(root / "first", 2)
                first["config"].update(scheduler=scheduler, one_cycle_pct_start=.3)
                self.seed()
                fit_adaptive(first, self.model())
                last = self.context(root / "last", 3)
                last["config"].update(scheduler=scheduler, one_cycle_pct_start=.3)
                last.update(start_epoch=2, resume=str(root / "first" / "resume.pt"), checkpoint=str(root / "first" / "model.ckpt"))
                if scheduler == "one_cycle":
                    with self.assertRaisesRegex(ValueError, "OneCycle schedule exhausted"):
                        fit_adaptive(last, self.model())
                else:
                    fit_adaptive(last, self.model())
                    state = torch.load(root / "last" / "resume.pt", weights_only=False)
                    self.assertEqual(state["scheduler"]["T_max"], 2)
                    self.assertEqual(state["scheduler"]["last_epoch"], 3)

    def test_invalid_resume_payload_is_not_exported_as_success(self):
        with tempfile.TemporaryDirectory() as directory:
            context = self.context(directory, 1)
            self.seed()
            result = fit_adaptive(context, self.model())
            state = torch.load(Path(directory) / "resume.pt", weights_only=False)
            del state["optimizer"]
            torch.save(state, Path(directory) / "resume.pt")
            with self.assertRaisesRegex(ValueError, "incomplete continuation state"):
                validate_resume(context, result)

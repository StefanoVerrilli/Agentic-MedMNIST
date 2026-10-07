from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from agents import (ArchitectureResearchAgent, DataAuditAgent, FrozenConfigurationAgent,
                    IngestionAgent, PreprocessingAgent, ProfilingAmbiguityAgent, ReviewerConsistencyAgent)
from contracts import (Blackboard, GeneratedCodeDecision, GeneratedSource, TrainConfig, WorkerConfiguration,
                       WorkerEvidence, sha256_file)
from generated_agents import ExperimentalDevelopmentAgent, GeneratedSearchAgent, fallback_decision
from governance import check_integrity, read_artefact
from llm import OllamaReasoner
from remote import RemoteModel, archive_bundle, copy_bundle, validate_bundle
from tests.helpers import make_bundle

WORKER = WorkerConfiguration()


class SimulatedWorker:
    """Transport double: never imports or executes bundle sources."""
    calls = []

    def __init__(self, configuration):
        self.configuration = configuration

    def preflight(self, device):
        return {"python_executable": "simulated-python", "isolation": {"enforced": False}}

    def execute(self, root, config, operation, arrays, **kwargs):
        self.calls.append((operation, tuple(arrays), config.generated_bundle, kwargs))
        output = root / "blobs" / "simulated" / str(len(self.calls))
        output.mkdir(parents=True)
        if operation == "verify":
            (output / "result.json").write_text('{"verified": true}', encoding="utf-8")
        elif operation == "strategy":
            (output / "strategy.json").write_text(json.dumps([
                {"parameters": {"hidden": 32, "dropout": 0.2}, "epochs": config.epochs}]), encoding="utf-8")
        elif operation == "train":
            assert set(arrays) == {"train_images", "train_targets", "val_images", "val_targets"}
            rows = [{"epoch": epoch, "train_loss": 1.0, "val_accuracy": 0.1}
                    for epoch in range(1, config.epochs + 1)]
            (output / "result.json").write_text(json.dumps({"final_train_loss": 1.0,
                "final_val_accuracy": 0.1, "best_val_accuracy": 0.1, "best_epoch": 1,
                "epochs_completed": len(rows), "history": rows}), encoding="utf-8")
            (output / "training.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            (output / "model.ckpt").write_bytes(b"opaque checkpoint: MUST NOT be deserialized on controller")
        elif operation == "infer":
            assert set(arrays) == {"images"}
            logits = np.zeros((len(arrays["images"]), 9), dtype="float32")
            logits[:, 0] = 1
            np.save(output / "logits.npy", logits)
        return output


class GeneratedTests(unittest.TestCase):
    def setUp(self):
        SimulatedWorker.calls = []

    def populate(self, path):
        bb = Blackboard(path)
        reasoner = OllamaReasoner(base_url=None)
        bb.put("worker_evidence", WorkerEvidence(configuration=WORKER,
            isolation={"enforced": False}), producer="test")
        IngestionAgent(loader=lambda **kw: make_bundle()).run(bb)
        ProfilingAmbiguityAgent(reasoner).run(bb)
        PreprocessingAgent(reasoner).run(bb)
        DataAuditAgent().run(bb)
        ArchitectureResearchAgent(reasoner).run(bb)
        return bb, reasoner

    def test_archiving_does_not_execute_code_and_versions_are_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            bb = Blackboard(Path(directory) / "run")
            decision = GeneratedCodeDecision(bundle_id="custom_model", hypothesis="New architecture hypothesis",
                files=[GeneratedSource(path="experiment.py", code="raise RuntimeError('must never execute locally')\n")])
            first = archive_bundle(bb, decision, "test")
            original = validate_bundle(bb.root, first).joinpath("experiment.py").read_bytes()
            second = archive_bundle(bb, decision, "test", parent=first)
            self.assertEqual((first.version, second.version), (1, 2))
            self.assertEqual(validate_bundle(bb.root, first).joinpath("experiment.py").read_bytes(), original)
            self.assertEqual(Blackboard.open(bb.root).get("generated_custom_model").parent_sha256, first.sha256)

    def test_tampered_and_unrecorded_sources_fail_integrity_and_reviewer(self):
        with tempfile.TemporaryDirectory() as directory:
            bb, reasoner = self.populate(Path(directory) / "run")
            ref = archive_bundle(bb, fallback_decision(), "test")
            folder = validate_bundle(bb.root, ref)
            (folder / "experiment.py").write_text("# changed", encoding="utf-8")
            self.assertTrue(check_integrity(bb.root))
            report = ReviewerConsistencyAgent(reasoner).review(bb, "experimental_development")
            self.assertEqual(report.action, "stop")
            self.assertTrue(any("generated code integrity" in issue for issue in report.deterministic_issues))

    def test_unrecorded_source_and_unsafe_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            bb = Blackboard(Path(directory) / "run")
            ref = archive_bundle(bb, fallback_decision(), "test")
            (validate_bundle(bb.root, ref) / "hidden.py").write_text("pass", encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_bundle(bb.root, ref)
        for path in ("../outside.py", "/absolute.py", "x/../../escape.py", "x\\escape.py"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                GeneratedSource(path=path, code="pass")

    def test_bundle_ceiling_and_run_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            bb = Blackboard(Path(directory) / "one")
            worker = WORKER.model_copy(update={"max_bundles": 1})
            bb.put("worker_evidence", WorkerEvidence(configuration=worker,
                isolation={}), producer="test")
            ref = archive_bundle(bb, fallback_decision(), "test")
            with self.assertRaises(ValueError):
                archive_bundle(bb, fallback_decision(), "test")
            target = Path(directory) / "two"
            copy_bundle(bb.root, target, ref)
            self.assertEqual(sha256_file(validate_bundle(target, ref) / "experiment.py"),
                             sha256_file(validate_bundle(bb.root, ref) / "experiment.py"))

    def test_generated_config_requires_worker_and_bundle(self):
        with self.assertRaises(ValueError):
            TrainConfig(model_family="run_generated", lr=0.001, epochs=1, hidden=16,
                batch_size=8, weight_decay=0.0, class_weighting=False,
                seed=42, device="cpu", rationale="test", source="test")
    def test_search_freezes_actual_bundle_and_never_transmits_test_targets(self):
        with tempfile.TemporaryDirectory() as directory, patch("remote.LocalWorker", SimulatedWorker), \
                patch("generated_agents.LocalWorker", SimulatedWorker):
            bb, reasoner = self.populate(Path(directory) / "run")
            ExperimentalDevelopmentAgent(reasoner).run(bb)
            GeneratedSearchAgent(reasoner, worker=WORKER, seed=42, device="cpu", max_trials=4,
                rounds=2, search_epochs=1, final_epochs=2, accuracy_tolerance=0.005).run(bb)
            best = bb.get("best_configuration")
            self.assertEqual(best.train_config.model_family, "run_generated")
            self.assertLessEqual(best.train_config.epochs, 2)
            selected = next(bb.get(name) for name in bb.get("search_report").trial_artefacts
                            if bb.get(name).candidate_id == best.selected_candidate_id)
            self.assertEqual(selected.epoch_budget, 2)
            validate_bundle(bb.root, best.train_config.generated_bundle)
            self.assertFalse(bb.get("search_report").test_metrics_used)
            self.assertEqual(bb.get("search_plan").framework, "local.python")
            self.assertTrue(all("test_images" not in keys and "test_targets" not in keys
                                for _, keys, _, _ in SimulatedWorker.calls))
            for operation, keys, _, _ in SimulatedWorker.calls:
                if operation == "infer":
                    self.assertEqual(keys, ("images",))
            self.assertEqual(check_integrity(bb.root), [])
            self.assertEqual(ReviewerConsistencyAgent(reasoner).review(bb, "model_search").severity, "ok")

    def test_final_budget_failure_cannot_select_earlier_success(self):
        with tempfile.TemporaryDirectory() as directory, patch("remote.LocalWorker", SimulatedWorker), \
                patch("generated_agents.LocalWorker", SimulatedWorker):
            bb, reasoner = self.populate(Path(directory) / "run")
            ExperimentalDevelopmentAgent(reasoner).run(bb)
            execute = SimulatedWorker.execute
            def fail_final(self, root, config, operation, arrays, **kwargs):
                if operation == "train" and config.epochs == 2:
                    raise RuntimeError("final-budget OOM")
                return execute(self, root, config, operation, arrays, **kwargs)
            with patch.object(SimulatedWorker, "execute", fail_final), self.assertRaises(RuntimeError):
                GeneratedSearchAgent(reasoner, worker=WORKER, seed=42, device="cpu", max_trials=2,
                    rounds=2, search_epochs=1, final_epochs=2, accuracy_tolerance=0.005).run(bb)
            self.assertIsNone(bb.get_optional("best_configuration"))

    def test_remote_model_refuses_changed_checkpoint_and_invalid_logits(self):
        with tempfile.TemporaryDirectory() as directory, patch("remote.LocalWorker", SimulatedWorker), \
                patch("generated_agents.LocalWorker", SimulatedWorker):
            bb, reasoner = self.populate(Path(directory) / "run")
            ExperimentalDevelopmentAgent(reasoner).run(bb)
            GeneratedSearchAgent(reasoner, worker=WORKER, seed=42, device="cpu", max_trials=1,
                rounds=1, search_epochs=1, final_epochs=1, accuracy_tolerance=0.005).run(bb)
            trial = bb.get("trial_001")
            checkpoint = bb.root / trial.checkpoint_path
            proxy = RemoteModel(bb.root, checkpoint, trial.checkpoint_sha256, trial.config)
            checkpoint.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                proxy.predict(bb.get_blob("prepared_data"), "test", 8)

    def test_nonfinite_and_wrong_shape_predictions_are_rejected(self):
        from generated_agents import generated_config
        with tempfile.TemporaryDirectory() as directory, patch("remote.LocalWorker", SimulatedWorker):
            bb, _ = self.populate(Path(directory) / "run")
            decision = fallback_decision()
            reference = archive_bundle(bb, decision, "test")
            config = generated_config(decision, reference, WORKER, 42, "cpu", 1, "test")
            checkpoint = bb.blob_dir / "opaque.ckpt"
            checkpoint.write_bytes(b"opaque")
            proxy = RemoteModel(bb.root, checkpoint, sha256_file(checkpoint), config)
            execute = SimulatedWorker.execute
            for invalid in (np.full((18, 9), np.nan, dtype="float32"), np.zeros((18, 8), dtype="float32"),
                            np.zeros((18, 9), dtype="complex64")):
                def invalid_predict(self, root, config, operation, arrays, **kwargs):
                    result = execute(self, root, config, operation, arrays, **kwargs)
                    np.save(result / "logits.npy", invalid)
                    return result
                with patch.object(SimulatedWorker, "execute", invalid_predict), self.assertRaises(ValueError):
                    proxy.predict(bb.get_blob("prepared_data"), "test", 8)

    def test_repairs_bind_selection_to_the_corrected_source_version(self):
        from llm import ReasonedDecision
        class RepairReasoner:
            enabled = True
            def decide(self, **kwargs):
                decision = kwargs["fallback"]
                if "correct_" in kwargs["stage"]:
                    decision = decision.model_copy(update={"files": [GeneratedSource(path=item.path,
                        code=item.code + "\n# corrected after worker feedback\n") for item in decision.files]})
                return ReasonedDecision(decision, "repair_test", False, 1, None)
        execute = SimulatedWorker.execute
        def needs_repair(self, root, config, operation, arrays, **kwargs):
            if operation == "verify" and config.generated_bundle.version == 1:
                raise RuntimeError("broken first version")
            return execute(self, root, config, operation, arrays, **kwargs)
        with tempfile.TemporaryDirectory() as directory, patch("remote.LocalWorker", SimulatedWorker), \
                patch("generated_agents.LocalWorker", SimulatedWorker), patch.object(SimulatedWorker, "execute", needs_repair):
            bb, _ = self.populate(Path(directory) / "run")
            reasoner = RepairReasoner()
            ExperimentalDevelopmentAgent(reasoner).run(bb)
            GeneratedSearchAgent(reasoner, worker=WORKER, seed=42, device="cpu", max_trials=1,
                rounds=1, search_epochs=1, final_epochs=1, accuracy_tolerance=0.005).run(bb)
            reference = bb.get("best_configuration").train_config.generated_bundle
            self.assertEqual(reference.version, 2)
            self.assertIn("corrected after worker feedback", (validate_bundle(bb.root, reference) / "experiment.py").read_text())
            self.assertEqual(bb.get("generated_multiscale_experiment").reference, reference)

    def test_code_repair_is_limited_to_two_new_versions(self):
        from llm import ReasonedDecision
        class AlwaysRepair:
            enabled = True
            def decide(self, **kwargs):
                return ReasonedDecision(kwargs["fallback"], "repair_test", False, 1, None)
        execute = SimulatedWorker.execute
        def always_broken(self, root, config, operation, arrays, **kwargs):
            if operation == "verify":
                raise RuntimeError("unrepairable module")
            return execute(self, root, config, operation, arrays, **kwargs)
        with tempfile.TemporaryDirectory() as directory, patch("remote.LocalWorker", SimulatedWorker), \
                patch("generated_agents.LocalWorker", SimulatedWorker), patch.object(SimulatedWorker, "execute", always_broken):
            bb, _ = self.populate(Path(directory) / "run")
            reasoner = AlwaysRepair()
            ExperimentalDevelopmentAgent(reasoner).run(bb)
            with self.assertRaises(RuntimeError):
                GeneratedSearchAgent(reasoner, worker=WORKER, seed=42, device="cpu", max_trials=1,
                    rounds=1, search_epochs=1, final_epochs=1, accuracy_tolerance=0.005).run(bb)
            self.assertEqual(len(list((bb.root / "generated").glob("*/v*/manifest.json"))), 3)
            self.assertEqual(bb.get("trial_001").status, "failed")

    def test_full_autonomous_pipeline_frozen_seed_and_replay(self):
        from run import build_parser, run_once, resolve_limits
        with tempfile.TemporaryDirectory() as directory, patch("remote.LocalWorker", SimulatedWorker), \
                patch("generated_agents.LocalWorker", SimulatedWorker), contextlib.redirect_stdout(io.StringIO()):
            args = build_parser().parse_args(["--offline", "--device", "cpu", "--allow-generated-code",
                "--search-trials", "1", "--search-rounds", "1", "--search-epochs", "1", "--max-epochs", "1"])
            root = Path(directory)
            def ingestion(**kwargs):
                loader = kwargs.pop("loader", lambda **kw: make_bundle())
                return IngestionAgent(loader=loader, **kwargs)
            with patch("run.IngestionAgent", side_effect=ingestion):
                original = root / "original" / "seed_42"
                result = run_once(args, 42, original, resolve_limits(args))
                self.assertTrue(result["status"].startswith("completed"), result)
                self.assertFalse(read_artefact(original, "run_manifest")["lightning_cli"])
                best = result["_best_configuration"]
                with patch("generated_agents.propose_code", side_effect=AssertionError("frozen seed generated code")):
                    frozen = run_once(args, 47, root / "original" / "seed_47", resolve_limits(args), frozen_best=best)
                self.assertTrue(frozen["status"].startswith("completed"), frozen)
                args.replay_run = str(original)
                replay = root / "replay" / "seed_42"
                repeated = run_once(args, 42, replay, resolve_limits(args))
                self.assertTrue(repeated["status"].startswith("completed"), repeated)
                comparison = json.loads((replay / "replay_comparison.json").read_text())
                self.assertTrue(comparison["payload"]["matched"], comparison)
                ref = best.train_config.generated_bundle
                validate_bundle(root / "original" / "seed_47", ref)
                self.assertEqual(check_integrity(replay), [])

    def test_failed_local_preflight_does_not_create_experiment(self):
        from run import main
        with tempfile.TemporaryDirectory() as directory, patch("remote.LocalWorker.preflight",
                side_effect=RuntimeError("Local interpreter unavailable")):
            with self.assertRaises(RuntimeError):
                main(["--allow-generated-code",
                      "--offline", "--device", "cpu", "--output-root", directory])
            self.assertEqual(list(Path(directory).iterdir()), [])

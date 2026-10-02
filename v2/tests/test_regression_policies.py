"""Regression coverage for search, abstention, portability and evidence scope."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from agents import (AbstentionOODAgent, DataAuditAgent, IngestionAgent, ModelSearchAgent,
                    PreprocessingAgent, ProfilingAmbiguityAgent)
from contracts import Blackboard, CandidateProposal, EpochMetrics, TrainResult, _sha256_json, sha256_file
from governance import check_integrity, evidence_references
from llm import OllamaReasoner, ReasonedDecision
from ml import prepare_data
from run import build_parser, code_tree_sha256, validate_args
from search import (candidate_hash, default_candidates, lightning_config_payload,
                    proposal_to_config, proposal_to_representation, rank_trials)
from tests.helpers import make_bundle


class PolicyRegressions(unittest.TestCase):
    def test_wide_limits_and_agent_epoch_stopping_choices_survive_export(self):
        data = default_candidates(1)[0].model_dump()
        data.update(model_family="vision_transformer", hidden=1024, depth=48,
                    epochs=800, early_stopping_patience=200, early_stopping_monitor="val_loss",
                    early_stopping_min_delta=.015, batch_size=4096, num_heads=32,
                    mlp_ratio=16, patch_size=14, lr=.1, weight_decay=.5, dropout=.8,
                    label_smoothing=.4, gradient_clip_val=25, optimizer="adamw",
                    adam_beta1=.8, adam_beta2=.98, optimizer_eps=1e-6, scheduler="cosine", cosine_eta_min=1e-5)
        candidate = CandidateProposal.model_validate(data)
        config = proposal_to_config(candidate, seed=42, device="cpu", epochs=600, source="test")
        self.assertEqual((config.epochs, config.early_stopping_patience), (600, 200))
        payload = lightning_config_payload(config, proposal_to_representation(candidate), data_root=None,
            train_limit=None, val_limit=None, test_limit=None, download=False)
        callback = payload["trainer"]["callbacks"][1]["init_args"]
        self.assertEqual((callback["monitor"], callback["mode"], callback["patience"], callback["min_delta"]),
                         ("val_loss", "min", 200, .015))
        self.assertEqual(payload["model"]["adam_beta2"], .98)
        self.assertEqual(payload["model"]["cosine_eta_min"], 1e-5)
        self.assertNotEqual(candidate_hash(candidate), candidate_hash(candidate.model_copy(update={"epochs": 799})))
        args = build_parser().parse_args(["--max-epochs", "1000", "--search-epochs", "200",
                                         "--search-trials", "256", "--search-rounds", "16"])
        validate_args(args)
        self.assertEqual(build_parser().parse_args([]).max_epochs, 100)

    def test_cli_instantiates_exported_callback_and_optimizer_options(self):
        from lightning.pytorch.cli import LightningCLI
        from lightning.pytorch.callbacks import EarlyStopping
        from lightning_components import PathMNISTLitModule, PathMNISTDataModule
        candidate = CandidateProposal.model_validate({**default_candidates(1)[0].model_dump(),
            "model_family": "residual_cnn", "epochs": 7, "early_stopping_patience": 40, "early_stopping_monitor": "val_macro_f1",
            "early_stopping_min_delta": .02, "optimizer": "sgd", "momentum": .7,
            "nesterov": False, "scheduler": "none", "channel_cap": 512})
        config = proposal_to_config(candidate, seed=42, device="cpu", epochs=100, source="test")
        payload = lightning_config_payload(config, proposal_to_representation(candidate), data_root=None,
            train_limit=27, val_limit=18, test_limit=18, download=False)
        with tempfile.TemporaryDirectory() as directory:
            payload["trainer"].update(default_root_dir=directory, enable_progress_bar=False)
            cli = LightningCLI(PathMNISTLitModule, PathMNISTDataModule, args=payload,
                               run=False, save_config_callback=None)
            self.assertEqual(cli.trainer.max_epochs, 7)
            stopping = next(c for c in cli.trainer.callbacks if isinstance(c, EarlyStopping))
            self.assertEqual((stopping.patience, stopping.monitor), (40, "val_macro_f1"))
            self.assertEqual(cli.trainer.checkpoint_callback.monitor, "val_accuracy")
            optimizer = cli.model.configure_optimizers()
            self.assertEqual(optimizer.param_groups[0]["momentum"], .7)
            self.assertFalse(optimizer.param_groups[0]["nesterov"])
            self.assertEqual(cli.model.hparams.channel_cap, 512)

    def test_expanded_architecture_shapes_execute_and_invalid_combinations_fail(self):
        import torch
        from lightning_components import PathMNISTLitModule
        from pydantic import ValidationError
        for options in ({"hidden": 256, "depth": 1, "num_heads": 16, "patch_size": 14, "mlp_ratio": 8},
                        {"hidden": 8, "depth": 12, "num_heads": 1, "patch_size": 28}):
            model = PathMNISTLitModule(model_family="vision_transformer", **options)
            self.assertEqual(tuple(model(torch.randn(2, 3, 28, 28)).shape), (2, 9))
        data = default_candidates(1)[0].model_dump()
        with self.assertRaises(ValidationError):
            CandidateProposal.model_validate({**data, "optimizer": "sgd", "momentum": 0, "nesterov": True})
        with self.assertRaises(ValidationError):
            CandidateProposal.model_validate({**data, "scheduler": "cosine", "cosine_eta_min": 1, "lr": .01})

    def test_real_trainer_honors_agent_loss_stopping_and_optimizer(self):
        import torch
        from ml import train_model
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            data = default_candidates(1)[0].model_dump()
            data.update(model_family="tiny_cnn", hidden=8, depth=1, dropout=0,
                        epochs=20, lr=1e-7, optimizer="sgd", momentum=.5, nesterov=False,
                        scheduler="none", early_stopping_monitor="val_loss",
                        early_stopping_patience=2, early_stopping_min_delta=1.0)
            candidate = CandidateProposal.model_validate(data)
            config = proposal_to_config(candidate, seed=42, device="cpu", epochs=100, source="test")
            prepared = prepare_data(make_bundle(), proposal_to_representation(candidate))
            with tempfile.TemporaryDirectory() as directory:
                output = train_model(prepared, config, Path(directory) / "selected.ckpt",
                                     enable_progress_bar=False, echo_epoch_log=False)
                self.assertEqual(output.result.epochs_completed, 3)
                self.assertEqual(output.model.hparams.momentum, .5)
                self.assertFalse(output.model.hparams.nesterov)
        finally:
            torch.set_num_threads(previous_threads)

    def test_source_identity_normalizes_paths_and_line_endings_but_detects_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tests").mkdir()
            source = root / "tests" / "sample.py"
            source.write_bytes(b"a = 1\nb = 2\n")
            original = code_tree_sha256(root)
            source.write_bytes(b"a = 1\r\nb = 2\r\n")
            self.assertEqual(original, code_tree_sha256(root))
            self.assertNotEqual(original, code_tree_sha256(root, algorithm="legacy-native-v1"))
            source.write_bytes(b"a = 3\nb = 2\n")
            self.assertNotEqual(original, code_tree_sha256(root))
            with self.assertRaises(ValueError):
                code_tree_sha256(root, algorithm="unknown")

    def test_nested_evidence_and_historical_frozen_origin_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "seed_47"
            origin = root.parent / "seed_42"
            (root / "artefacts").mkdir(parents=True)
            origin.mkdir()
            source = origin / "best.yaml"
            source.write_bytes(b"original configuration")
            checkpoint = root / "ablation.ckpt"
            checkpoint.write_bytes(b"checkpoint")
            payload = {"frozen_best": {"train_config": {"seed": 42},
                "lightning_config_path": "best.yaml", "lightning_config_sha256": sha256_file(source)},
                "scenarios": [{"checkpoint_path": checkpoint.name, "checkpoint_sha256": sha256_file(checkpoint)}]}
            (root / "artefacts" / "one.json").write_text(json.dumps(
                {"payload": payload, "payload_sha256": _sha256_json(payload)}), encoding="utf-8")
            self.assertEqual(check_integrity(root), [])
            checkpoint.write_bytes(b"corrupt")
            self.assertIn("scenarios[0]", " ".join(check_integrity(root)))
            source.write_bytes(b"corrupt source")
            self.assertIn("frozen_best", " ".join(check_integrity(root)))
            with self.assertRaises(ValueError):
                list(evidence_references(root, {"path": "../../escape", "sha256": "a" * 64}))
            # A local snapshot must never silently fall back to a sibling.
            payload["frozen_reference_scope"] = "local_snapshot"
            refs = list(evidence_references(root, payload))
            self.assertEqual(refs[0][1], root / "best.yaml")

    def test_export_instantiates_the_same_checkpoint_and_stopping_policy(self):
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
        candidate = default_candidates(1)[0]
        config = proposal_to_config(candidate, seed=42, device="cpu", epochs=8, source="test")
        for patience in (0, 3):
            current = config.model_copy(update={"early_stopping_patience": patience})
            payload = lightning_config_payload(current, proposal_to_representation(candidate),
                data_root=None, train_limit=None, val_limit=None, test_limit=None, download=False)
            specs = payload["trainer"]["callbacks"]
            checkpoint = ModelCheckpoint(**specs[0]["init_args"])
            self.assertEqual((checkpoint.monitor, checkpoint.mode, checkpoint.save_top_k), ("val_accuracy", "max", 1))
            self.assertEqual(len(specs), 2 if patience else 1)
            if patience:
                stop = EarlyStopping(**specs[1]["init_args"])
                self.assertEqual((stop.monitor, stop.mode, stop.patience), ("val_accuracy", "max", patience))

    def test_ood_failure_disables_acceptance_and_test_cannot_select_method(self):
        bundle = make_bundle()
        candidate = default_candidates(1)[0]
        config = proposal_to_config(candidate, seed=47, device="cpu", epochs=2, source="test")
        prepared = prepare_data(bundle, proposal_to_representation(candidate))
        clean = np.full((18, 9), .01)
        clean[np.arange(18), bundle.targets["val"]] = .92
        clean /= clean.sum(axis=1, keepdims=True)
        uncertain = np.full_like(clean, 1 / 9)
        with tempfile.TemporaryDirectory() as directory:
            for val_good, test_good in ((False, True), (True, False), (True, True)):
                with self.subTest(validation=val_good, test=test_good):
                    bb = Blackboard(Path(directory) / f"{val_good}_{test_good}")
                    bb.put("train_config", config, producer="test")
                    blobs = {"probabilities": {"val": clean, "test": clean,
                             "val_targets": bundle.targets["val"], "test_targets": bundle.targets["test"]},
                             "prepared_data": prepared, "model": object()}
                    seeds = []
                    def outputs(model, data, split, *args, **kwargs):
                        seeds.append(kwargs["seed"])
                        good = val_good if split == "val" else test_good
                        return uncertain if good else clean, None, None
                    with patch.object(bb, "get_blob", side_effect=blobs.__getitem__), patch("agents.predict_outputs", side_effect=outputs):
                        AbstentionOODAgent().run(bb)
                    report = bb.get("abstention_report")
                    self.assertEqual(report.detector_status, "eligible" if val_good else "no_eligible_detector")
                    self.assertEqual(report.ood_pass, val_good and test_good)
                    self.assertEqual(report.test_coverage, 1 if val_good else 0)
                    self.assertEqual(report.human_review_count, 0 if val_good else 18)
                    self.assertEqual(set(seeds), {1729})
                    self.assertEqual(report.ood_scenarios[0].test_false_accept_rate, 0 if test_good else 1)

    def test_adaptive_slots_and_finalists_use_comparable_budgets(self):
        bundle = make_bundle()
        reasoner = OllamaReasoner(base_url=None)
        calls, configs = [], []
        class SearchReasoner:
            def decide(self, **kwargs):
                calls.append(kwargs)
                candidates = [default_candidates(1)[0].model_copy(update={
                    "candidate_id": f"adaptive_{len(calls)}_{i}", "lr": .00123 * len(calls) + i * .0001}) for i in range(2)]
                return ReasonedDecision(value=kwargs["fallback"].model_copy(update={"candidates": candidates, "stop": True}),
                                        source="test:adaptive", used_fallback=False, attempts=1, request_id=None)
        def trainer(prepared, config, checkpoint):
            configs.append(config)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"test checkpoint")
            result = TrainResult(final_train_loss=.5, final_val_accuracy=.7, best_val_accuracy=.7,
                best_epoch=1, epochs_completed=1, history=[EpochMetrics(epoch=1, train_loss=.5, val_accuracy=.7)],
                checkpoint_path=str(checkpoint), checkpoint_sha256=sha256_file(checkpoint), seed=config.seed, device="cpu")
            return SimpleNamespace(model=object(), result=result)
        def predictor(model, prepared, split, *args, **kwargs):
            self.assertEqual(split, "val")
            targets = prepared.bundle.targets[split]
            probabilities = np.full((len(targets), 9), .01)
            probabilities[np.arange(len(targets)), targets] = .92
            return probabilities, targets
        with tempfile.TemporaryDirectory() as directory:
            bb = Blackboard(Path(directory) / "run")
            IngestionAgent(loader=lambda **kwargs: bundle).run(bb)
            ProfilingAmbiguityAgent(reasoner).run(bb)
            PreprocessingAgent(reasoner).run(bb)
            DataAuditAgent().run(bb)
            ModelSearchAgent(SearchReasoner(), max_trials=8, rounds=3, search_epochs=1, final_epochs=4,
                             trainer=trainer, predictor=predictor).run(bb)
            report = bb.get("search_report")
            trials = [bb.get(name) for name in report.trial_artefacts]
            self.assertEqual(len(trials), 8)
            self.assertEqual(len(calls), 2)  # final comparison has no LLM capacity
            self.assertEqual(sum(t.decision_source == "test:adaptive" for t in trials), 4)
            self.assertTrue(all(t.decision_source == "promotion:finalist" for t in trials[-2:]))
            self.assertEqual([t.config.epochs for t in trials], [1, 1, 1, 2, 2, 2, 4, 4])
            previous = {t.config_hash for t in trials if t.config.epochs == 2}
            self.assertTrue({t.config_hash for t in trials[-2:]} <= previous)
            self.assertIn("epoch_budget", calls[1]["user"])
            with self.assertRaises(ValueError):
                rank_trials([trials[0], trials[-1]], accuracy_tolerance=.005)
            finalists = trials[-2:]
            expected = rank_trials(finalists, accuracy_tolerance=.005).config_hash
            reversed_times = [t.model_copy(update={"duration_seconds": 1000 if i else 0}) for i, t in enumerate(finalists)]
            self.assertEqual(rank_trials(reversed_times, accuracy_tolerance=.005).config_hash, expected)
            self.assertIn("learning_curve_tail", calls[1]["user"])
            class ShortHorizon:
                def decide(self, **kwargs):
                    candidate = default_candidates(1)[0].model_copy(update={"epochs": 1,
                        "early_stopping_patience": 35, "early_stopping_monitor": "val_loss",
                        "early_stopping_min_delta": .01})
                    return ReasonedDecision(value=kwargs["fallback"].model_copy(update={"candidates": [candidate]}),
                        source="test:short", used_fallback=False, attempts=1, request_id=None)
            short = Blackboard(Path(directory) / "short")
            IngestionAgent(loader=lambda **kwargs: bundle).run(short)
            ProfilingAmbiguityAgent(reasoner).run(short)
            PreprocessingAgent(reasoner).run(short)
            DataAuditAgent().run(short)
            ModelSearchAgent(ShortHorizon(), max_trials=1, rounds=1, search_epochs=1, final_epochs=4,
                             trainer=trainer, predictor=predictor).run(short)
            selected = short.get("train_config")
            self.assertEqual((selected.epochs, selected.early_stopping_patience), (1, 35))
            self.assertEqual(selected.early_stopping_monitor, "val_loss")
            trial = short.get(short.get("search_report").trial_artefacts[0])
            self.assertEqual((trial.epoch_budget, trial.requested_epochs, trial.config.epochs), (4, 1, 1))


if __name__ == "__main__":
    unittest.main()

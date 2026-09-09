from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from agents import ModelSearchAgent, _ood_auc
from contracts import (
    BestConfiguration,
    Blackboard,
    EpochMetrics,
    SearchReport,
    TrainResult,
    TrialResult,
    sha256_file,
)
from llm import OllamaReasoner
from search import (
    candidate_hash,
    default_candidates,
    proposal_to_config,
    proposal_to_representation,
    rank_trials,
)
from tests.helpers import make_bundle


class SearchTests(unittest.TestCase):
    def test_four_round_fallback_portfolio_is_executable_and_unique(self) -> None:
        candidates = [
            candidate
            for round_index in range(1, 5)
            for candidate in default_candidates(round_index)
        ]
        hashes = [candidate_hash(candidate) for candidate in candidates]
        self.assertEqual(len(candidates), 16)
        self.assertEqual(len(set(hashes)), len(hashes))

        resnet = next(item for item in candidates if item.model_family == "resnet18")
        other_depth = resnet.model_copy(update={"depth": 4 if resnet.depth != 4 else 2})
        self.assertEqual(candidate_hash(resnet), candidate_hash(other_depth))

    def test_rank_uses_macro_f1_only_inside_accuracy_tolerance(self) -> None:
        candidate = default_candidates(1)[0]
        config = proposal_to_config(
            candidate, seed=42, device="cpu", epochs=2, source="test"
        )

        def trial(name: str, accuracy: float, macro_f1: float) -> TrialResult:
            return TrialResult(
                candidate_id=name,
                config_hash=(name[0] * 64),
                round_index=1,
                status="completed",
                config=config,
                representation=proposal_to_representation(candidate),
                validation_accuracy=accuracy,
                validation_macro_f1=macro_f1,
                validation_balanced_accuracy=macro_f1,
                best_epoch=1,
                checkpoint_path="x.ckpt",
                checkpoint_sha256="a" * 64,
                duration_seconds=1,
                decision_source="test",
            )

        trials = [trial("alpha", 0.80, 0.70), trial("bravo", 0.796, 0.76)]
        selected = rank_trials(trials, accuracy_tolerance=0.005)
        self.assertEqual(selected.candidate_id, "bravo")
        selected = rank_trials(trials, accuracy_tolerance=0.001)
        self.assertEqual(selected.candidate_id, "alpha")

    def test_model_search_never_requests_test_predictions(self) -> None:
        bundle = make_bundle()
        calls: list[str] = []
        epoch_budgets: list[int] = []

        def trainer(prepared, config, checkpoint):
            epoch_budgets.append(config.epochs)
            path = Path(checkpoint)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(config.model_family.encode("utf-8"))
            score = 2 if config.model_family == "resnet18" else 1
            result = TrainResult(
                final_train_loss=0.5,
                final_val_accuracy=0.7,
                best_val_accuracy=0.7,
                best_epoch=1,
                epochs_completed=1,
                history=[EpochMetrics(epoch=1, train_loss=0.5, val_accuracy=0.7)],
                checkpoint_path=str(path),
                checkpoint_sha256=sha256_file(path),
                seed=config.seed,
                device=config.device,
            )
            return SimpleNamespace(model=SimpleNamespace(score=score), result=result)

        def predictor(model, prepared, split, batch_size, **kwargs):
            calls.append(split)
            targets = np.asarray(prepared.bundle.targets[split])
            probabilities = np.full((len(targets), 9), 0.01, dtype="float64")
            predictions = targets.copy()
            if model.score == 1:
                predictions[::3] = (predictions[::3] + 1) % 9
            probabilities[np.arange(len(targets)), predictions] = 0.92
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            return probabilities, targets

        with tempfile.TemporaryDirectory() as directory:
            from agents import IngestionAgent, ProfilingAmbiguityAgent

            # Use real lightweight stages to populate validated manifests/profile.
            bb = Blackboard(Path(directory) / "run")
            IngestionAgent(loader=lambda **kwargs: bundle).run(bb)
            reasoner = OllamaReasoner(base_url=None)
            ProfilingAmbiguityAgent(reasoner).run(bb)
            ModelSearchAgent(
                reasoner,
                max_trials=2,
                rounds=2,
                search_epochs=1,
                final_epochs=2,
                trainer=trainer,
                predictor=predictor,
            ).run(bb)
            report = bb.get("search_report")
            best = bb.get("best_configuration")
            self.assertIsInstance(report, SearchReport)
            self.assertIsInstance(best, BestConfiguration)
            self.assertEqual(calls, ["val", "val"])
            self.assertEqual(epoch_budgets, [1, 1])
            self.assertFalse(report.test_metrics_used)
            self.assertEqual(
                sha256_file(Path(directory) / "run" / "best_config.yaml"),
                best.lightning_config_sha256,
            )

    def test_ood_auc_rewards_uncertain_corrupted_samples(self) -> None:
        clean = np.full((9, 9), 0.01)
        clean[np.arange(9), np.arange(9)] = 0.92
        clean /= clean.sum(axis=1, keepdims=True)
        corrupted = np.full((9, 9), 1.0 / 9.0)
        self.assertEqual(_ood_auc(clean, corrupted, "max_softmax"), 1.0)
        self.assertEqual(_ood_auc(clean, corrupted, "predictive_entropy"), 1.0)


if __name__ == "__main__":
    unittest.main()

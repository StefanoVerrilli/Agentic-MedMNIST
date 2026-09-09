from __future__ import annotations

import json
import tempfile
import unittest
from types import SimpleNamespace

from agents import ReviewerConsistencyAgent
from contracts import Blackboard
from llm import OllamaReasoner


class FaultInjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reviewer = ReviewerConsistencyAgent(OllamaReasoner(base_url=None))

    def test_at_least_ninety_percent_of_modelled_faults_are_detected(self) -> None:
        builders = [
            self._wrong_dataset,
            self._wrong_channels,
            self._empty_split,
            self._bad_profile_counts,
            self._non_official_split,
            self._unsafe_augmentation,
            self._leaky_statistics,
            self._seed_mismatch,
            self._bad_confusion_matrix,
            self._bad_abstention_sum,
        ]
        detected = 0
        for builder in builders:
            with (
                self.subTest(fault=builder.__name__),
                tempfile.TemporaryDirectory() as path,
            ):
                bb = Blackboard(path)
                stage = builder(bb)
                findings = self.reviewer._deterministic_findings(bb, stage)
                detected += bool(findings)
        self.assertGreaterEqual(detected / len(builders), 0.90)

    def test_llm_cannot_downgrade_a_deterministic_critical_finding(self) -> None:
        def transport(url, payload, timeout):
            response = {
                "severity": "ok",
                "action": "continue",
                "issues": [],
                "comment": "looks fine",
            }
            return {"message": {"content": json.dumps(response)}}

        reviewer = ReviewerConsistencyAgent(
            OllamaReasoner("http://localhost:11434", transport=transport, retries=0)
        )
        with tempfile.TemporaryDirectory() as path:
            bb = Blackboard(path)
            self._wrong_dataset(bb)
            report = reviewer.review(bb, "ingestion")
            self.assertEqual(report.severity, "critical")
            self.assertEqual(report.action, "stop")

    @staticmethod
    def _base_manifest(**changes):
        data = {
            "dataset": "pathmnist",
            "n_channels": 3,
            "n_classes": 9,
            "loaded_split_sizes": {"train": 18, "val": 9, "test": 9},
            "subsampled": False,
        }
        data.update(changes)
        return SimpleNamespace(**data)

    def _wrong_dataset(self, bb):
        bb.artefacts["dataset_manifest"] = self._base_manifest(dataset="pneumoniamnist")
        return "ingestion"

    def _wrong_channels(self, bb):
        bb.artefacts["dataset_manifest"] = self._base_manifest(n_channels=1)
        return "ingestion"

    def _empty_split(self, bb):
        bb.artefacts["dataset_manifest"] = self._base_manifest(
            loaded_split_sizes={"train": 18, "val": 0, "test": 9}
        )
        return "ingestion"

    def _bad_profile_counts(self, bb):
        bb.artefacts["dataset_manifest"] = self._base_manifest()
        bb.artefacts["data_profile"] = SimpleNamespace(
            class_counts={"train": [1] * 9, "val": [1] * 9, "test": [1] * 9},
            profiled_fraction=1.0,
            train_channel_std_unit=[0.1, 0.1, 0.1],
        )
        return "profiling"

    def _base_preprocessing(self, bb):
        bb.artefacts["dataset_manifest"] = self._base_manifest()
        bb.artefacts["split_manifest"] = SimpleNamespace(
            strategy="official",
            train_size=18,
            val_size=9,
            test_size=9,
            train_stats_only=True,
        )
        bb.artefacts["representation_plan"] = SimpleNamespace(
            augmentations=[], augmentation_factor=1
        )

    def _non_official_split(self, bb):
        self._base_preprocessing(bb)
        bb.artefacts["split_manifest"].strategy = "random"
        return "preprocessing"

    def _unsafe_augmentation(self, bb):
        self._base_preprocessing(bb)
        bb.artefacts["representation_plan"] = SimpleNamespace(
            augmentations=["color_shift"], augmentation_factor=2
        )
        return "preprocessing"

    def _leaky_statistics(self, bb):
        self._base_preprocessing(bb)
        bb.artefacts["split_manifest"].train_stats_only = False
        return "preprocessing"

    def _seed_mismatch(self, bb):
        bb.artefacts["train_config"] = SimpleNamespace(seed=42, epochs=2)
        bb.artefacts["train_result"] = SimpleNamespace(
            seed=43,
            epochs_completed=2,
            checkpoint_path="missing.pt",
            checkpoint_sha256="0" * 64,
        )
        return "training"

    def _bad_confusion_matrix(self, bb):
        bb.artefacts["dataset_manifest"] = self._base_manifest()
        bb.artefacts["evaluation_report"] = SimpleNamespace(
            n_samples=9,
            confusion_matrix=[[0] * 9 for _ in range(9)],
            accuracy=0.5,
        )
        return "evaluation"

    def _bad_abstention_sum(self, bb):
        bb.artefacts["abstention_report"] = SimpleNamespace(
            test_coverage=0.7,
            abstain_rate=0.4,
            accuracy_on_covered=0.8,
            base_test_accuracy=0.7,
            ood_scope="controlled_corruption_proxy",
        )
        return "abstention"


if __name__ == "__main__":
    unittest.main()

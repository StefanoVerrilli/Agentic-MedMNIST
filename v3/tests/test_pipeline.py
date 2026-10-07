from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from agents import (
    ArchitectureResearchAgent,
    IngestionAgent,
    PreprocessingAgent,
    ProfilingAmbiguityAgent,
    ReportingAgent,
    ReviewerConsistencyAgent,
)
from contracts import AnomalyReport, ArchitectureResearch, Blackboard, RunManifest
from llm import OllamaReasoner
from ml import evaluate_probabilities, stratified_indices
from orchestrator import Orchestrator
from run import select_task_configuration
from tests.helpers import make_bundle


class PipelineTests(unittest.TestCase):
    def test_cross_cutting_research_is_auditable_and_test_blind(self) -> None:
        bundle = make_bundle()
        with tempfile.TemporaryDirectory() as directory:
            bb = Blackboard(directory)
            reasoner = OllamaReasoner(base_url=None)
            IngestionAgent(loader=lambda **kwargs: bundle).run(bb)
            ProfilingAmbiguityAgent(reasoner).run(bb)
            profile = bb.get("data_profile")
            self.assertEqual(set(profile.class_counts), {"train", "val"})
            self.assertEqual(profile.test_to_train_prevalence_ratio, [])
            self.assertEqual(profile.samples_profiled, 45)
            ArchitectureResearchAgent(reasoner).run(bb)
            research = bb.get("architecture_research")
            self.assertIsInstance(research, ArchitectureResearch)
            self.assertFalse(research.test_metrics_used)
            self.assertIn("compact_transformer", research.architecture_priorities)
            self.assertGreaterEqual(len(research.evidence_sources), 3)
            self.assertEqual(set(research.evidence_provenance.values()), {"unverified_external_reference"})

    def test_research_revision_receives_previous_brief_and_review(self):
        from llm import ReasonedDecision
        calls = []

        class Capture:
            enabled = True

            def decide(self, **kwargs):
                calls.append(kwargs)
                return ReasonedDecision(kwargs["fallback"], "test", False, 1, None)

        class Reviewer:
            def review(self, bb, stage, *, attempt=1):
                report = AnomalyReport(stage=stage, attempt=attempt,
                    severity="warning" if attempt == 1 else "ok",
                    action="revise" if attempt == 1 else "continue",
                    comment="Clarify contingent proposals" if attempt == 1 else "Complete", source="test")
                bb.put("anomaly_architecture_research", report, producer="reviewer")
                return report

        with tempfile.TemporaryDirectory() as directory:
            bb = Blackboard(directory)
            IngestionAgent(loader=lambda **kwargs: make_bundle()).run(bb)
            ProfilingAmbiguityAgent(OllamaReasoner(base_url=None)).run(bb)
            Orchestrator([ArchitectureResearchAgent(Capture())], Reviewer(), max_stage_retries=1).run(bb)
            self.assertEqual(len(calls), 2)
            self.assertIn("Clarify contingent proposals", calls[1]["user"])
            self.assertIn("previous_research", calls[1]["user"])
            self.assertIn("immediately available", calls[1]["system"])
            self.assertEqual(bb.get("execution_status").status, "completed")

    def test_stateful_stage_is_not_restarted_after_exception(self):
        from autonomous import AutonomousSearchAgent
        calls = []

        class Stateful:
            name = "model_search"
            retry_on_exception = AutonomousSearchAgent.retry_on_exception

            def run(self, bb):
                calls.append("run")
                raise RuntimeError("exhausted decision retries after a completed trial")

        with tempfile.TemporaryDirectory() as directory:
            bb = Blackboard(directory)
            with self.assertRaisesRegex(RuntimeError, "exhausted decision"):
                Orchestrator([Stateful()], None, max_stage_retries=3).run(bb)
            self.assertEqual(calls, ["run"])
            self.assertEqual(bb.get("execution_status").status, "failed:model_search")
            self.assertFalse(any(row["event"] == "stage_retry" for row in bb.log))

    def test_official_validation_split_is_preserved(self) -> None:
        bundle = make_bundle()

        def loader(**kwargs):
            return bundle

        with tempfile.TemporaryDirectory() as directory:
            bb = Blackboard(directory)
            reasoner = OllamaReasoner(base_url=None)
            IngestionAgent(loader=loader).run(bb)
            ProfilingAmbiguityAgent(reasoner).run(bb)
            PreprocessingAgent(reasoner).run(bb)
            split = bb.get("split_manifest")
            prepared = bb.get_blob("prepared_data")
            self.assertEqual(split.strategy, "official")
            self.assertEqual(split.train_size, 27)
            self.assertEqual(split.val_size, 18)
            self.assertEqual(split.test_size, 18)
            np.testing.assert_array_equal(prepared.bundle.images["val"], bundle.images["val"])
            findings = ReviewerConsistencyAgent(reasoner)._deterministic_findings(
                bb, "preprocessing"
            )
            self.assertEqual(findings, [])

    def test_stratified_subsample_is_reproducible_and_covers_all_classes(self) -> None:
        labels = np.repeat(np.arange(9), np.arange(1, 10) + 4)
        first = stratified_indices(labels, 27, 123)
        second = stratified_indices(labels, 27, 123)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(set(labels[first].tolist()), set(range(9)))

    def test_multiclass_evaluation_has_all_required_metrics(self) -> None:
        targets = np.repeat(np.arange(9), 2)
        probabilities = np.eye(9)[targets]
        report = evaluate_probabilities(probabilities, targets, make_bundle().labels)
        self.assertEqual(report.accuracy, 1.0)
        self.assertEqual(report.macro_f1, 1.0)
        self.assertEqual(report.roc_auc_ovr_macro, 1.0)
        self.assertIsNotNone(report.accuracy_ci95_low)
        self.assertLessEqual(report.accuracy_ci95_low, report.accuracy)
        self.assertGreaterEqual(report.accuracy_ci95_high, report.accuracy)
        self.assertTrue(all(row.recall_ci95_low is not None for row in report.per_class))
        self.assertEqual(sum(sum(row) for row in report.confusion_matrix), 18)

    def test_task_configuration_selection_is_independent_of_test_status(self) -> None:
        summaries = [
            {
                "seed": 1,
                "status": "completed",
                "best_validation_accuracy": 0.75,
                "best_validation_macro_f1": 0.72,
                "best_config_path": "seed_1/best_config.yaml",
            },
            {
                "seed": 2,
                "status": "vetoed:comparison",
                "best_validation_accuracy": 0.80,
                "best_validation_macro_f1": 0.77,
                "best_config_path": "seed_2/best_config.yaml",
            },
        ]
        selected = select_task_configuration(summaries, accuracy_tolerance=0.005)
        self.assertIsNotNone(selected)
        self.assertEqual(selected["seed"], 2)

    def test_final_dossier_contains_reporting_review(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bb = Blackboard(directory)
            bb.put(
                "run_manifest",
                RunManifest(
                    run_id="test",
                    dataset="pathmnist",
                    seed=42,
                    created_at="2026-01-01T00:00:00+00:00",
                    python_version="3.12",
                    platform="test",
                    package_versions={},
                    llm_mode="heuristic",
                    llm_model="none",
                ),
                producer="test",
            )
            reasoner = OllamaReasoner(base_url=None)
            orchestrator = Orchestrator(
                [ReportingAgent()],
                ReviewerConsistencyAgent(reasoner),
                max_stage_retries=0,
            )
            orchestrator.run(bb)
            dossier = json.loads(
                (Path(directory) / "dossier.json").read_text(encoding="utf-8")
            )
            self.assertIn("anomaly_reporting", dossier["latest_artefacts"])
            self.assertEqual(dossier["status"], "completed")

    def test_orchestrator_retries_then_promotes(self) -> None:
        calls: list[str] = []

        class Dummy:
            name = "dummy"

            def run(self, bb):
                calls.append("run")

            def revise(self, bb, report):
                calls.append("remediate")
                return True

        class Reviewer:
            count = 0

            def review(self, bb, stage, *, attempt=1):
                self.count += 1
                warning = self.count == 1
                return AnomalyReport(
                    stage=stage,
                    attempt=attempt,
                    severity="warning" if warning else "ok",
                    action="revise" if warning else "continue",
                    comment="retry" if warning else "pass",
                    source="test",
                )

        with tempfile.TemporaryDirectory() as directory:
            orchestrator = Orchestrator([Dummy()], Reviewer(), max_stage_retries=1)
            bb = Blackboard(directory)
            orchestrator.run(bb)
            self.assertEqual(calls, ["run", "remediate", "run"])
            self.assertEqual(bb.get("execution_status").status, "completed")

    def test_orchestrator_does_not_blindly_retry_without_remediation(self) -> None:
        calls: list[str] = []

        class Dummy:
            name = "dummy"

            def run(self, bb):
                calls.append("run")

        class Reviewer:
            def review(self, bb, stage, *, attempt=1):
                return AnomalyReport(
                    stage=stage,
                    attempt=attempt,
                    severity="warning",
                    action="revise",
                    comment="unresolved",
                    source="test",
                )

        with tempfile.TemporaryDirectory() as directory:
            bb = Blackboard(directory)
            orchestrator = Orchestrator([Dummy()], Reviewer(), max_stage_retries=2)
            orchestrator.run(bb)
            self.assertEqual(calls, ["run"])
            self.assertTrue(any(row["event"] == "review_unresolved" for row in bb.log))


if __name__ == "__main__":
    unittest.main()

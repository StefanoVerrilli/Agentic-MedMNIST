"""Reviewer-driven corrections preserve experiments and frozen evaluation evidence."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agents import ReportingAgent, ReviewerConsistencyAgent
from contracts import (AnomalyReport, Blackboard, ExecutionState, ReportSummary, ReviewRequest,
                       RunConfiguration, sha256_file)
from orchestrator import Orchestrator
from tests.test_autonomous import AdaptiveWorker, ScriptedReasoner, experiment
from tests import test_autonomous as support
from tests.test_generated import WORKER
from autonomous import AutonomousSearchAgent
from governance import check_integrity
from replay import CachedReasoner


def request():
    return ReviewRequest(request_id="comparison", problem="Stopping claim lacks comparison",
                         correction="Run another candidate", required_evidence="Validation checkpoint metrics")


class SequenceReviewer:
    def __init__(self, revisions, stage="model_search", stop=False):
        self.revisions, self.stage, self.stop = revisions, stage, stop
        self.calls = []

    def review(self, bb, stage, attempt=1):
        self.calls.append(attempt)
        action = "stop" if self.stop else "revise" if len(self.calls) <= self.revisions else "continue"
        report = AnomalyReport(stage=stage, attempt=attempt, severity="critical" if self.stop else
            "warning" if action == "revise" else "ok", action=action, source="test",
            comment="Need new comparison" if action == "revise" else "Evidence accepted",
            requests=[request()] if action == "revise" else [])
        bb.put("anomaly_" + stage, report, producer="reviewer")
        return report


class RevisionTests(unittest.TestCase):
    def test_interrupted_search_restores_feedback_and_identifiers(self):
        class InterruptingReasoner(ScriptedReasoner):
            def decide(self, **kwargs):
                if not self.actions:
                    raise KeyboardInterrupt()
                return super().decide(**kwargs)

        with tempfile.TemporaryDirectory() as directory, patch("autonomous.LocalWorker", AdaptiveWorker), \
                patch("remote.LocalWorker", AdaptiveWorker):
            bb = support.AutonomousTests().populate(directory)
            first = InterruptingReasoner([
                dict(action="new_trial", experiment=experiment().model_dump(), rationale="First candidate"),
                dict(action="finish_search", rationale="Proposed stop")])
            with self.assertRaises(KeyboardInterrupt):
                Orchestrator([AutonomousSearchAgent(first, worker=WORKER, seed=42, device="cpu")],
                    SequenceReviewer(1), max_stage_retries=0,
                    review_revision_stages=("model_search",)).run(bb)
            self.assertEqual(bb.get("execution_status").status, "paused:model_search")
            reopened = Blackboard.open(bb.root)
            candidate = experiment(lr=.003).model_dump()
            candidate["bundle_id"] = "new_candidate"
            second = ScriptedReasoner([
                dict(action="new_trial", experiment=candidate, rationale="Answer reviewer"),
                dict(action="finish_search", rationale="Comparison now available")])
            Orchestrator([AutonomousSearchAgent(second, worker=WORKER, seed=42, device="cpu")],
                SequenceReviewer(0), max_stage_retries=0,
                review_revision_stages=("model_search",)).run(reopened)
            self.assertEqual(reopened.get("anomaly_model_search").attempt, 2)
            self.assertEqual(reopened.get("search_report").completed_trials, 2)
            self.assertEqual(json.loads(second.calls[0]["user"])["review_feedback"]["requests"][0]["request_id"], "comparison")
            self.assertEqual(check_integrity(reopened.root), [])

    def test_report_feedback_replays_without_backend_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            original, replayed = Blackboard(Path(directory) / "original"), Blackboard(Path(directory) / "replayed")
            for bb in (original, replayed):
                bb.put("run_configuration", RunConfiguration(parameters={"execution_mode": "agent_autonomous",
                    "reviewer_revisions": True}), producer="test")
            backend = ScriptedReasoner([
                dict(narrative="Initial result", response_to_review="Initial report"),
                dict(severity="warning", action="revise", comment="Explain coverage",
                     requests=[request().model_dump()]),
                dict(narrative="Corrected result", response_to_review="Coverage explained"),
                dict(severity="ok", action="continue", comment="Explanation accepted")])
            recorded = CachedReasoner(backend, original.root)
            unused = ScriptedReasoner([])
            replay = CachedReasoner(unused, replayed.root, replay_root=original.root)
            for bb, reasoner in ((original, recorded), (replayed, replay)):
                reviewer = ReviewerConsistencyAgent(reasoner)
                with patch.object(reviewer, "_deterministic_findings", return_value=[]):
                    Orchestrator([ReportingAgent(reasoner)], reviewer, max_stage_retries=0,
                                 review_revision_stages=("reporting",)).run(bb)
            replay.assert_replay_complete()
            self.assertEqual(unused.calls, [])
            self.assertEqual(original.get("report_summary"), replayed.get("report_summary"))

    def test_search_survives_three_rejections_and_preserves_trials(self):
        with tempfile.TemporaryDirectory() as directory, patch("autonomous.LocalWorker", AdaptiveWorker), \
                patch("remote.LocalWorker", AdaptiveWorker):
            bb = support.AutonomousTests().populate(directory)
            actions = []
            for index in range(4):
                candidate = experiment(lr=.002 + index * .001).model_dump()
                candidate["bundle_id"] = f"candidate_{index}"
                actions.extend([dict(action="new_trial", experiment=candidate, rationale="New validation evidence"),
                                dict(action="finish_search", rationale="Checkpoint comparison supports selection")])
            backend = ScriptedReasoner(actions)
            agent = AutonomousSearchAgent(backend, worker=WORKER, seed=42, device="cpu")
            reviewer = SequenceReviewer(3)
            Orchestrator([agent], reviewer, max_stage_retries=0,
                         review_revision_stages=("model_search",)).run(bb)
            self.assertEqual(reviewer.calls, [1, 2, 3, 4])
            self.assertEqual(bb.get("search_report").completed_trials, 4)
            self.assertEqual(bb.get("search_report").families_evaluated, ["run_generated"])
            self.assertEqual(len({bb.get(n).candidate_id for n in bb.get("search_report").trial_artefacts}), 4)
            self.assertEqual(bb._versions["best_configuration"], 4)
            self.assertEqual(check_integrity(bb.root), [])
            for call in backend.calls:
                payload = json.loads(call["user"])
                self.assertEqual(payload["test_split"], "locked")
                self.assertNotIn("evaluation_report", payload)
            self.assertEqual(json.loads(backend.calls[2]["user"])["review_feedback"]["requests"][0]["request_id"], "comparison")
            self.assertEqual(bb.get("review_response_model_search").review_attempt, 3)
            self.assertEqual(len([e for e in bb.log if e["event"] == "search_configuration_frozen"]), 1)
            self.assertEqual(Blackboard.open(bb.root).get("search_report").completed_trials, 4)

    def test_search_cannot_reopen_after_test(self):
        with tempfile.TemporaryDirectory() as directory:
            bb = Blackboard(directory)
            bb.put("evaluation_report", ExecutionState(status="evaluated"), producer="test")
            with self.assertRaisesRegex(ValueError, "after test"):
                AutonomousSearchAgent(ScriptedReasoner([]), worker=WORKER, seed=42, device="cpu").run(bb)

    def test_report_revisions_do_not_change_prediction_files(self):
        with tempfile.TemporaryDirectory() as directory:
            bb = Blackboard(directory)
            checkpoint, predictions = bb.root / "model.ckpt", bb.root / "predictions.npz"
            checkpoint.write_bytes(b"frozen model")
            predictions.write_bytes(b"frozen predictions")
            before = [sha256_file(checkpoint), sha256_file(predictions)]
            backend = ScriptedReasoner([dict(narrative=f"Evidence based report {i}",
                response_to_review="Corrected the requested explanation") for i in range(4)])
            Orchestrator([ReportingAgent(backend)], SequenceReviewer(3, "reporting"),
                max_stage_retries=0, review_revision_stages=("reporting",)).run(bb)
            self.assertEqual(bb._versions["report_summary"], 4)
            self.assertEqual(before, [sha256_file(checkpoint), sha256_file(predictions)])
            self.assertEqual(bb.get("review_response_reporting").review_attempt, 3)
            self.assertEqual(check_integrity(bb.root), [])

    def test_advisory_warning_is_not_a_correction_but_requests_are(self):
        with tempfile.TemporaryDirectory() as directory:
            bb = Blackboard(directory)
            bb.put("run_configuration", RunConfiguration(parameters={"execution_mode": "agent_autonomous",
                "reviewer_revisions": True}), producer="test")
            backend = ScriptedReasoner([
                dict(severity="warning", action="continue", issues=[], comment="Document limited OOD scope",
                     observations=["Controlled corruption only"]),
                dict(severity="ok", action="continue", issues=[], comment="Missing comparison",
                     requests=[request().model_dump()])])
            reviewer = ReviewerConsistencyAgent(backend)
            with patch.object(reviewer, "_deterministic_findings", return_value=[]):
                first = reviewer.review(bb, "model_search")
                second = reviewer.review(bb, "model_search", attempt=2)
            self.assertEqual((first.action, first.severity), ("continue", "warning"))
            self.assertEqual((second.action, second.severity), ("revise", "warning"))

    def test_integrity_veto_never_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            bb = Blackboard(directory)
            backend = ScriptedReasoner([dict(narrative="Report", response_to_review="Initial report")])
            reviewer = SequenceReviewer(0, "reporting", stop=True)
            Orchestrator([ReportingAgent(backend)], reviewer, max_stage_retries=0,
                         review_revision_stages=("reporting",)).run(bb)
            self.assertEqual(bb.get("execution_status").status, "vetoed:reporting")
            self.assertEqual(len(backend.calls), 1)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from contracts import Blackboard, ExperimentDecision, RunManifest


def manifest(run_id: str = "test") -> RunManifest:
    return RunManifest(
        run_id=run_id,
        dataset="pathmnist",
        seed=42,
        created_at="2026-01-01T00:00:00+00:00",
        python_version="3.12",
        platform="test",
        package_versions={},
        llm_mode="heuristic",
        llm_model="none",
    )


class ContractTests(unittest.TestCase):
    def test_invalid_llm_training_values_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ExperimentDecision(
                lr=-1,
                epochs=1_000_000,
                hidden=64,
                batch_size=32,
                weight_decay=0,
                class_weighting=False,
                rationale="unsafe values",
            )

    def test_blackboard_versions_and_checksums_every_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bb = Blackboard(directory, run_id="version-test")
            bb.put("run_manifest", manifest("version-test"), producer="test")
            bb.put("run_manifest", manifest("version-test"), producer="test")
            files = sorted((Path(directory) / "artefacts").glob("*.json"))
            self.assertEqual(len(files), 2)
            self.assertIn("v001", files[0].name)
            self.assertIn("v002", files[1].name)
            first = json.loads(files[0].read_text(encoding="utf-8"))
            self.assertEqual(len(first["payload_sha256"]), 64)
            self.assertEqual(first["version"], 1)
            lines = (
                (Path(directory) / "decision_log.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            )
            self.assertEqual(len(lines), 2)

    def test_dossier_contains_latest_post_review_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bb = Blackboard(directory)
            bb.put("run_manifest", manifest(), producer="test")
            bb.write_dossier(status="completed")
            dossier = json.loads(
                (Path(directory) / "dossier.json").read_text(encoding="utf-8")
            )
            self.assertIn("run_manifest", dossier["latest_artefacts"])
            self.assertEqual(dossier["decision_events"], len(bb.log))


if __name__ == "__main__":
    unittest.main()

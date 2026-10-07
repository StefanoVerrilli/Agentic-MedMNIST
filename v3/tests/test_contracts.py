from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from contracts import ArchitectureResearch, Blackboard, ExperimentDecision, RunManifest


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
    def test_research_review_gets_full_prose_and_other_previews_are_labelled(self):
        research = ArchitectureResearch(
            analysis="Complete analysis. " * 400,
            transformer_guidance="Complete transformer guidance. " * 150,
            representation_priorities=["unit scaling"], parametrization_priorities=["control capacity"],
            architecture_priorities=["resnet18", "compact_transformer"],
            evidence_sources=["https://arxiv.org/abs/2104.05704"], source="test")
        with tempfile.TemporaryDirectory() as directory:
            bb = Blackboard(directory)
            bb.put("architecture_research", research, producer="test")
            context = json.loads(bb.review_context("architecture_research", max_chars=1000))
            self.assertEqual(context["artefacts"]["architecture_research"]["analysis"], research.analysis)
            self.assertEqual(context["artefacts"]["architecture_research"]["transformer_guidance"], research.transformer_guidance)
            self.assertEqual(context["compacted_fields"], [])
            preview = json.loads(bb.review_context("general"))
            self.assertTrue(preview["artefacts"]["architecture_research"]["analysis"].endswith("..."))
            self.assertIn("architecture_research.analysis", [row["path"] for row in preview["compacted_fields"]])
            self.assertEqual(bb.get("architecture_research").analysis, research.analysis)

    def test_conditional_architecture_options_are_enforced(self) -> None:
        from contracts import CandidateProposal

        base = {
            "candidate_id": "bad_heads", "model_family": "vision_transformer",
            "hidden": 40, "depth": 2, "dropout": 0.1,
            "normalization": "standardize", "augmentations": ["hflip"],
            "optimizer": "adamw", "scheduler": "one_cycle", "lr": 3e-4,
            "weight_decay": 1e-4, "class_weighting": False,
            "label_smoothing": 0.05, "batch_size": 64, "num_heads": 3,
            "rationale": "invalid divisibility must be rejected",
        }
        with self.assertRaises(ValidationError):
            CandidateProposal.model_validate(base)

        cnn = CandidateProposal.model_validate(
            {**base, "candidate_id": "canonical_cnn", "model_family": "tiny_cnn",
             "hidden": 32, "num_heads": 3, "patch_size": 2}
        )
        self.assertEqual(cnn.patch_size, 4)
        self.assertEqual(cnn.num_heads, 4)

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

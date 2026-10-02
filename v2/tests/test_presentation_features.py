from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from agents import (DataAuditAgent, IngestionAgent, PreprocessingAgent, PriorArtScoutAgent,
                    ProfilingAmbiguityAgent, ReviewerConsistencyAgent)
from contracts import (Blackboard, ExtensionProposal, HumanReviewQueue, LiteratureDecision,
                       PriorArtBrief, RepresentationDecision, sha256_file)
from extensions import clear_registry, gate_extensions, recipe
from governance import resolve_human_review
from llm import OllamaReasoner
from replay import CachedReasoner
from research import LiteratureStore, canonical_hash, citation_issues
from tests.helpers import make_bundle


class PresentationFeaturesTests(unittest.TestCase):
    def tearDown(self):
        clear_registry()

    def populate(self, root, bundle=None):
        bb = Blackboard(root)
        reasoner = OllamaReasoner(base_url=None)
        IngestionAgent(loader=lambda **kw: bundle or make_bundle()).run(bb)
        ProfilingAmbiguityAgent(reasoner).run(bb)
        return bb, reasoner

    def test_scout_sources_are_snapshotted_cited_and_consumed_before_preprocessing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bb, reasoner = self.populate(root / "run")
            PriorArtScoutAgent(reasoner, LiteratureStore(root / "cache")).run(bb)
            brief = bb.get("prior_art_brief")
            self.assertEqual(citation_issues(brief, bb.root), [])
            self.assertEqual(brief.retrieval_mode, "bundled")
            calls = []
            class Capture:
                seed = 42
                def decide(self, **kw):
                    calls.append(kw["user"])
                    return reasoner.decide(**kw)
            PreprocessingAgent(Capture()).run(bb)
            self.assertIn(brief.ideas[0].source_id, calls[0])
            self.assertIn("hypothesis_to_validate", calls[0])

    def test_hallucinated_quote_and_tampered_snapshot_trigger_veto(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bb, reasoner = self.populate(root / "run")
            PriorArtScoutAgent(reasoner, LiteratureStore(root / "cache")).run(bb)
            brief = bb.get("prior_art_brief")
            invalid = brief.model_copy(update={"ideas": [brief.ideas[0].model_copy(
                update={"evidence_quote": "This is an invented observation without supporting evidence."})]})
            bb.put("prior_art_brief", invalid, producer="fault")
            self.assertEqual(ReviewerConsistencyAgent(reasoner).review(bb, "prior_art").action, "stop")
            (bb.root / brief.sources[0].snapshot_path).write_text("tampered", encoding="utf-8")
            self.assertTrue(citation_issues(brief, bb.root))

    def test_arxiv_search_and_cache_replay_never_refetch(self):
        xml = '''<feed xmlns="http://www.w3.org/2005/Atom"><entry>
        <id>http://arxiv.org/abs/2104.05704v4</id><title>Compact Transformers</title>
        <summary>Convolutional tokenization is studied for small-scale learning.</summary></entry></feed>'''
        calls = []
        def transport(url):
            calls.append(url)
            return xml.encode()
        with tempfile.TemporaryDirectory() as directory, patch("research.time.sleep"):
            root = Path(directory)
            store = LiteratureStore(root / "cache", online=True, transport=transport)
            first, mode, errors = store.retrieve(root / "one")
            self.assertEqual(mode, "online")
            self.assertEqual(len(calls), 2)
            self.assertTrue(any("search_query" in url for url in calls))
            second, mode, _ = store.retrieve(root / "two")
            self.assertEqual(mode, "cache")
            self.assertEqual(len(calls), 2)
            self.assertEqual([s.snapshot_sha256 for s in first], [s.snapshot_sha256 for s in second])
            (root / "cache" / Path(first[0].snapshot_path).name).write_text("bad", encoding="utf-8")
            with self.assertRaises(ValueError):
                store.retrieve(root / "three")

    def test_live_requirement_cannot_be_satisfied_by_curated_excerpts(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                LiteratureStore(Path(directory) / "cache", require_live=True).retrieve(Path(directory) / "run")

    def test_blob_reopen_is_typed_and_original_object_mutation_is_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = make_bundle()
            bb, _ = self.populate(Path(directory) / "run", bundle)
            original = bundle.images["train"].copy()
            bundle.images["train"][:] = 0
            restored = Blackboard.open(bb.root)
            np.testing.assert_array_equal(restored.get_blob("raw_dataset").images["train"], original)
            descriptor = restored.get("blob_raw_dataset")
            (restored.root / descriptor.path).write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                restored.get_blob("raw_dataset")

    def test_actual_train_validation_overlap_blocks_pretraining_gate(self):
        bundle = make_bundle()
        bundle.images["val"][0] = bundle.images["train"][0]
        with tempfile.TemporaryDirectory() as directory:
            bb, reasoner = self.populate(Path(directory) / "run", bundle)
            PreprocessingAgent(reasoner).run(bb)
            DataAuditAgent().run(bb)
            self.assertFalse(bb.get("data_audit_report").passed)
            self.assertEqual(bb.get("data_audit_report").train_validation_overlap, 1)
            self.assertEqual(ReviewerConsistencyAgent(reasoner).review(bb, "data_audit").action, "stop")

    def test_extension_requires_hash_bound_review_and_never_runs_architecture_spec(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bb, reasoner = self.populate(root / "run")
            PriorArtScoutAgent(reasoner, LiteratureStore(root / "cache")).run(bb)
            brief = bb.get("prior_art_brief")
            proposal = ExtensionProposal(proposal_id="rotate_three", kind="augmentation",
                description="Compose rotations into another exact orientation variant.",
                source_ids=[brief.sources[0].source_id], operations=["rotate90", "rotate180"],
                label_preservation_rationale="Reviewer confirms tissue type is independent of orientation.")
            architecture = ExtensionProposal(proposal_id="new_family", kind="architecture",
                description="A new architecture requiring a developer-owned registry implementation.",
                source_ids=[brief.sources[1].source_id], implementation_requirements=["Implement a tested registry entry"])
            brief = brief.model_copy(update={"proposals": [proposal, architecture]})
            pending = gate_extensions(brief)
            self.assertTrue(all(entry.status == "pending_review" for entry in pending.entries))
            with self.assertRaises(ValueError):
                recipe("approved_rotate_three")
            approvals = root / "approvals.json"
            approvals.write_text(json.dumps({"approvals": {p.proposal_id: {
                "spec_sha256": canonical_hash(p.model_dump(mode="json")), "decision": "approve",
                "reviewer": "Independent reviewer", "rationale": "Review of the cited, bounded hypothesis"}
                for p in (proposal, architecture)}}), encoding="utf-8")
            approved = gate_extensions(brief, approvals)
            self.assertEqual([entry.status for entry in approved.entries], ["approved", "reviewed_spec"])
            from ml import apply_augmentation
            image = make_bundle().images["train"][0]
            np.testing.assert_array_equal(apply_augmentation(image, "approved_rotate_three"), np.rot90(image, 3))
            with self.assertRaises(ValueError):
                recipe("approved_new_family")

    def test_cache_replay_uses_no_backend_and_rejects_changed_context(self):
        from contracts import AmbiguityDecision
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = OllamaReasoner(base_url=None)
            kwargs = dict(stage="ambiguity", system="bounded", user="train count=27",
                          response_model=AmbiguityDecision,
                          fallback={"ambiguity_note": "Ambiguity risk", "risks": []})
            first = CachedReasoner(backend, root / "one").decide(**kwargs)
            replay = CachedReasoner(backend, root / "two", replay_root=root / "one")
            with patch.object(backend, "decide", side_effect=AssertionError("network/provider called")):
                second = replay.decide(**kwargs)
            self.assertEqual(first.value, second.value)
            replay.assert_replay_complete()
            changed = CachedReasoner(backend, root / "three", replay_root=root / "one")
            with self.assertRaises(ValueError):
                changed.decide(**{**kwargs, "user": "train count=28"})

    def test_human_responses_are_versioned_and_bound_to_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bb, _ = self.populate(root / "run")
            queue_path = bb.blob_dir / "queue.json"
            queue_path.write_text(json.dumps({"cases": [{"case_id": "test_1"}, {"case_id": "test_2"}]}), encoding="utf-8")
            bb.put("human_review_queue", HumanReviewQueue(path="blobs/queue.json",
                sha256=sha256_file(queue_path), count=2), producer="test")
            bb.write_dossier(status="completed")
            response = root / "responses.json"
            response.write_text(json.dumps({"decisions": [{"case_id": "test_1", "reviewer": "Reviewer",
                "decision": "correct", "label": 2, "comment": "Reviewed the benchmark image"}]}), encoding="utf-8")
            result = resolve_human_review(bb.root, response)
            self.assertEqual(result.pending_count, 1)
            self.assertEqual(result.queue_sha256, sha256_file(queue_path))
            response.write_text(json.dumps({"decisions": [{"case_id": "outside", "reviewer": "Reviewer",
                "decision": "defer", "comment": "No evidence available"}]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                resolve_human_review(bb.root, response)

    def test_real_lightning_end_to_end_and_offline_replay_match(self):
        import torch
        from run import build_parser, resolve_limits, run_once
        from governance import assess

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args = build_parser().parse_args(["--offline", "--device", "cpu", "--max-epochs", "1",
                    "--search-trials", "1", "--search-rounds", "1", "--search-epochs", "1"])
                original = root / "original" / "seed_42"
                replay = root / "replay" / "seed_42"
                def ingestion(**kwargs):
                    loader = kwargs.pop("loader", lambda **kw: make_bundle())
                    return IngestionAgent(loader=loader, **kwargs)
                with patch("run.IngestionAgent", side_effect=ingestion), contextlib.redirect_stdout(io.StringIO()):
                    result = run_once(args, 42, original, resolve_limits(args))
                    self.assertTrue(result["status"].startswith("completed"))
                    args.replay_run = str(original)
                    repeated = run_once(args, 42, replay, resolve_limits(args))
                comparison = json.loads((replay / "replay_comparison.json").read_text())
                self.assertTrue(comparison["payload"]["matched"], comparison)
                report = assess(replay)
                self.assertTrue(report.technical_complete, report.model_dump())
                self.assertFalse(report.acceptance_ready)
                self.assertFalse(report.trl7_evidence_complete)
                self.assertEqual(result["agentic_accuracy"], repeated["agentic_accuracy"])
                # A frozen seed and its replay must carry their own source YAML,
                # without requiring the first seed in the replay destination.
                from contracts import BestConfiguration
                from governance import check_integrity, read_artefact
                selected = BestConfiguration.model_validate(read_artefact(original, "best_configuration"))
                frozen_root = original.parent / "seed_47"
                frozen_replay = root / "frozen_replay" / "seed_47"
                args.replay_run = None
                with patch("run.IngestionAgent", side_effect=ingestion), contextlib.redirect_stdout(io.StringIO()):
                    run_once(args, 47, frozen_root, resolve_limits(args), frozen_best=selected)
                    args.replay_run = str(frozen_root)
                    embedded = BestConfiguration.model_validate(read_artefact(frozen_root, "run_configuration")["frozen_best"])
                    run_once(args, 47, frozen_replay, resolve_limits(args), frozen_best=embedded)
                self.assertEqual(check_integrity(frozen_replay), [])
                frozen_comparison = json.loads((frozen_replay / "replay_comparison.json").read_text())
                self.assertTrue(frozen_comparison["payload"]["matched"], frozen_comparison)
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()

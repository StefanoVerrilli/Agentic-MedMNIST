"""Fail-closed operational continuation into an independent child run."""
from __future__ import annotations

import shutil
from pathlib import Path

from contracts import Blackboard, ResumeManifest, TrialResult, utc_now
from governance import check_integrity


def verify_store(bb):
    issues = check_integrity(bb.root)
    if issues:
        raise ValueError("resume integrity failure: " + "; ".join(issues))
    from remote import validate_bundle
    from generated_agents import configuration_hash
    from research import contained_path
    from contracts import sha256_file
    for name, item in bb.artefacts.items():
        if isinstance(item, TrialResult):
            if configuration_hash(item.config) != item.config_hash:
                raise ValueError(f"trial configuration checksum mismatch: {name}")
            if item.config.generated_bundle:
                validate_bundle(bb.root, item.config.generated_bundle)
            if item.status == "completed":
                references = [(item.checkpoint_path, item.checkpoint_sha256)]
                if item.config.execution_mode == "agent_autonomous" or item.resume_path or item.resume_sha256:
                    references.append((item.resume_path, item.resume_sha256))
                for path, checksum in references:
                    if not path or not checksum or sha256_file(contained_path(bb.root, path)) != checksum:
                        raise ValueError(f"completed trial checkpoint/resume checksum mismatch: {name}")
                if item.training_result:
                    result = item.training_result
                    if (result.history != item.learning_curve or result.epochs_completed != item.epochs_completed
                            or result.checkpoint_sha256 != item.checkpoint_sha256 or result.resume_sha256 != item.resume_sha256
                            or result.best_epoch != item.best_epoch or result.stop_reason != item.stop_reason
                            or result.checkpoint_path != item.checkpoint_path or result.resume_path != item.resume_path
                            or result.seed != item.config.seed or result.device != item.config.device):
                        raise ValueError(f"trial training result inconsistent: {name}")


def inspect_parent(root):
    """Read and verify a parent without writing even a log entry into it."""
    root = Path(root).resolve()
    if any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError("resume parent cannot contain symlinks")
    bb = Blackboard.open(root)
    verify_store(bb)
    status = bb.get("execution_status")
    stage = status.stage
    if not stage or not status.status.startswith(("paused:", "failed:")):
        raise ValueError("resume requires a recorded paused/failed stage")
    if status.status.startswith("failed:"):
        # Historical LLM failures used failed rather than paused. Explicit migration.
        exceptions = [e for e in bb.log if e["event"] in {"stage_exception", "review_exception"}
                      and e.get("stage") == stage]
        if not exceptions or exceptions[-1].get("error_type") != "OllamaDecisionError":
            raise ValueError("terminal failure is not safely resumable")
    if status.status not in {f"paused:{stage}", f"failed:{stage}", f"paused:review:{stage}", f"failed:review:{stage}"}:
        raise ValueError("execution status/stage mismatch")
    config, manifest = bb.get("run_configuration"), bb.get("run_manifest")
    if not manifest.code_sha256:
        raise ValueError("resume requires the parent's recorded code hash")
    from run import parse_seeds
    if manifest.seed not in parse_seeds(config.parameters["seeds"]):
        raise ValueError("run configuration seed mismatch")
    promoted = list(dict.fromkeys(e["stage"] for e in bb.log if e["event"] == "stage_promoted"))
    if stage == "reporting" and stage in promoted:
        # A final report may be refreshed after baseline/ablation added evidence.
        last_start = max((e["sequence"] for e in bb.log if e["event"] == "stage_started"
                          and e.get("stage") == stage), default=0)
        last_promotion = max(e["sequence"] for e in bb.log if e["event"] == "stage_promoted"
                             and e.get("stage") == stage)
        if last_start > last_promotion:
            promoted.remove(stage)
    if stage in promoted:
        raise ValueError("failed stage was already promoted; inconsistent state")
    if (stage == "model_search" and not status.status.startswith(("paused:review:", "failed:review:"))
            and config.parameters.get("execution_mode") != "agent_autonomous"):
        raise ValueError("partial model_search resume currently requires agent_autonomous")
    if bb.get_optional("dataset_manifest"):
        from ml import index_fingerprint
        bundle = bb.get_blob("raw_dataset")
        dataset = bb.get("dataset_manifest")
        if bundle.selected_index_sha256 != dataset.selected_index_sha256:
            raise ValueError("dataset provenance mismatch")
        for split in ("train", "val"):
            if index_fingerprint(split, bundle.selected_indices[split], bundle.targets[split]) != dataset.selected_index_sha256[split]:
                raise ValueError("split provenance mismatch")
        if bb.get_optional("split_manifest"):
            split = bb.get("split_manifest")
            if split.seed != manifest.seed or split.selected_index_sha256 != dataset.selected_index_sha256:
                raise ValueError("preprocessing split provenance mismatch")
            prepared = bb.get_blob("prepared_data")
            if prepared.bundle.selected_index_sha256 != bundle.selected_index_sha256:
                raise ValueError("prepared dataset provenance mismatch")
    for item in bb.artefacts.values():
        if isinstance(item, TrialResult) and item.config.seed != manifest.seed:
            raise ValueError("trial seed mismatch")
    if stage == "model_search" and config.parameters.get("execution_mode") == "agent_autonomous":
        from autonomous import AutonomousSearchState
        AutonomousSearchState.restore(bb)
    return bb, promoted


def create_child(parent_root, child_root, *, code_sha256):
    parent, promoted = inspect_parent(parent_root)
    child_root = Path(child_root).resolve()
    if child_root == parent.root or parent.root in child_root.parents:
        raise ValueError("resume child must be outside parent run")
    if child_root.exists():
        raise FileExistsError("resume child directory already exists")
    # Physical copies, never hard links: future writes cannot affect the parent.
    shutil.copytree(parent.root, child_root)
    child = Blackboard.open(child_root)
    child.run_id = child_root.parent.name + "/" + child_root.name
    stage = parent.get("execution_status").stage
    child.record_event("run_resume_started", parent_run=str(parent.root), stage=stage)
    try:
        if child.log[:-1] != parent.log or child.registry != parent.registry:
            raise ValueError("parent changed while resume snapshot was copied")
        verify_store(child)
        child.record_event("run_resume_integrity_checked", parent_run=str(parent.root), stage=stage)
        state = None
        if stage == "model_search" and not parent.get("execution_status").status.startswith(("paused:review:", "failed:review:")):
            from autonomous import AutonomousSearchState
            state = AutonomousSearchState.restore(child)
        transcripts = sorted(p.name for p in (child.blob_dir / "reasoning").glob("*.json"))
        resume = ResumeManifest(parent_run_path=str(parent.root), parent_run_id=parent.run_id,
            parent_execution_status=parent.get("execution_status").status, parent_stage=stage,
            parent_code_sha256=parent.get("run_manifest").code_sha256,
            resume_code_sha256=code_sha256, resumed_at=utc_now(),
            last_imported_artefact=parent.registry[-1]["path"],
            last_imported_trial=state.trial_names[-1] if state and state.trial_names else None,
            first_new_decision=f"autonomous_search.action_{state.sequence + 1}" if state else None,
            inherited_transcripts=transcripts, promoted_stages=promoted,
            review_only=parent.get("execution_status").status.startswith(("paused:review:", "failed:review:")))
        child.put("resume_manifest", resume, producer="resume")
        child.record_event("run_resume_state_restored", parent_run=str(parent.root), stage=stage,
            search_sequence=state.sequence if state else None, candidate_sequence=state.candidate_sequence if state else None,
            completed_trials=sum(t.status == "completed" for t in state.trials) if state else 0,
            failed_trials=sum(t.status == "failed" for t in state.trials) if state else 0,
            first_new_decision=resume.first_new_decision)
        return child
    except Exception as exc:
        child.record_event("run_resume_failed", parent_run=str(parent.root), stage=stage, error=str(exc))
        raise

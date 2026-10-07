"""Agent-directed search: no external trial, round or epoch horizon."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import time

from contracts import (AutonomousSearchDecision, AutonomousSearchEvent, BestConfiguration,
                       SearchPlan, SearchReport, TrainConfig, TrainResult, TrialResult, sha256_file)
from generated_agents import CODE_INTERFACE, _representation, configuration_hash
from ml import evaluate_probabilities, predict_probabilities
from remote import RemoteModel, SSHWorker, archive_bundle, prepared_arrays
from search import rank_trials


AUTONOMOUS_INTERFACE = CODE_INTERFACE.split("Optional propose(context)")[0].replace(
    "epochs (hard allocated epoch ceiling)", "epochs (agent-selected cumulative target)") + '''
This run is agent_autonomous. YOU choose every active experimental parameter and
all search actions. No allocated epoch ceiling, trial count or round count exists.
Choose new_trial (complete experiment, explicit training options and initial epochs),
continue_trial (existing candidate_id and positive additional_epochs), or finish_search.
Use validation only. Justify every choice and stop when further research is not useful.
Experiments must support exact continuation. train(context) saves BOTH model.ckpt
(best checkpoint) and resume.pt (latest state). context['epochs'] is the cumulative
agent-selected target; segment_epochs is the newly requested duration, start_epoch is
the previously completed epoch, resume is a read-only input state or None.
History is cumulative. Return stop_reason='segment_complete' or 'early_stopping'.
The resume.pt dictionary must contain format='agent-resume-v1', identity, data_identity,
epochs_completed, model, optimizer, scheduler, rng, loader_rng, history, best,
best_epoch, stale, stopped. Obtain identity using autonomous.continuation_identity(config)
and data_identity using adaptive_training.data_identity(data). Preserve preprocessing and other custom state too.
worker_runtime.fit_model supports this protocol and selects the explicitly supplied
optimizer/scheduler/loss controls. A custom loop must preserve equivalent state.
Continuation must not change code, parameters, optimizer or preprocessing. Create a
new trial for changes. A scheduler with an exhausted fixed horizon cannot be extended.
Operational limits are one hour per operation and the configured worker memory;
resource interruption is a failed segment, never a completed trial. Decisions must
account for observed runtime and memory. Epochs have no fixed upper bound.
Use only dependencies already installed in the worker image. Preserve learned
preprocessing parameters in the checkpoint. All source code belongs to this run.
'''


def continuation_identity(config):
    """Stable executable identity, independent of the next training horizon."""
    from research import canonical_hash
    payload = config.model_dump(mode="json") if hasattr(config, "model_dump") else dict(config)
    for name in ("epochs", "worker", "source", "rationale", "schema_version", "segment_targets"):
        payload.pop(name, None)
    if payload.get("generated_bundle"):
        payload["generated_bundle"] = dict(payload["generated_bundle"])
        payload["generated_bundle"].pop("path", None)
    return canonical_hash(payload)


def experiment_config(experiment, reference, worker, seed, device, source):
    return TrainConfig(**experiment.training.model_dump(), execution_mode="agent_autonomous",
        model_family="run_generated", generated_bundle=reference, worker=worker,
        epochs=experiment.epochs, batch_size=experiment.batch_size, hidden=64, depth=2,
        seed=seed, device=device, rationale=experiment.hypothesis, source=source)


class AutonomousSearchAgent:
    name = "model_search"

    def __init__(self, reasoner, *, worker, seed, device, accuracy_tolerance=0.005):
        self.reasoner, self.worker, self.seed, self.device = reasoner, worker, seed, device
        self.accuracy_tolerance = accuracy_tolerance

    def run(self, bb):
        if not bb.get("data_audit_report").passed:
            raise ValueError("autonomous search requires a passed data audit")
        if not self.reasoner.enabled and not getattr(self.reasoner, "replay_root", None):
            raise ValueError("autonomous search requires live decisions or recorded replay")
        bb.put("search_plan", SearchPlan(policy="agent_autonomous", progressive_budget=False,
            framework="isolated.python", accuracy_tolerance=self.accuracy_tolerance), producer=self.name)
        prepared = bb.get_blob("prepared_data")
        latest, results, trials, names, identities = {}, {}, [], [], set()
        sequence, candidate_sequence = 0, 0
        while True:
            sequence += 1
            evidence = [{"candidate_id": t.candidate_id, "status": t.status,
                "validation_accuracy": t.validation_accuracy, "validation_macro_f1": t.validation_macro_f1,
                "epochs_completed": t.epochs_completed, "requested_target": t.config.epochs,
                "duration_seconds": t.duration_seconds, "stop_reason": t.stop_reason, "error": t.error,
                "configuration": t.config.model_dump(mode="json", exclude={"worker"}),
                "learning_curve": [row.model_dump(mode="json") for row in t.learning_curve]} for t in trials]
            decision = self.reasoner.decide(stage=f"autonomous_search.action_{sequence}",
                system=AUTONOMOUS_INTERFACE,
                user=json.dumps({"profile": bb.get("data_profile").model_dump(mode="json"),
                    "research": bb.get("architecture_research").model_dump(mode="json"),
                    "validation_evidence": evidence, "test_split": "locked",
                    "resources": {"operation_timeout_seconds": self.worker.timeout_seconds,
                                  "memory_gib": self.worker.memory_gib}}, ensure_ascii=False, sort_keys=True),
                response_model=AutonomousSearchDecision,
                fallback=AutonomousSearchDecision(action="finish_search", rationale="No fallback is permitted."),
                audit=bb.record_event)
            if decision.used_fallback:
                raise ValueError("autonomous decisions cannot use heuristic fallback")
            action = decision.value
            bb.put(f"autonomous_action_{sequence:04d}", AutonomousSearchEvent(decision=action,
                source=decision.source), producer=self.name)
            if action.action == "finish_search":
                if not latest:
                    raise ValueError("cannot finish search without a successful trial")
                break
            parent = None
            if action.action == "new_trial":
                reference = archive_bundle(bb, action.experiment, decision.source)
                config = experiment_config(action.experiment, reference, self.worker, self.seed, self.device, decision.source)
                identity = continuation_identity(config)
                if identity in identities:
                    raise ValueError("duplicate autonomous configuration; continue the existing trial instead")
                identities.add(identity)
                candidate_sequence += 1
                candidate_id = f"autonomous_t{candidate_sequence:04d}"
            else:
                parent = latest.get(action.candidate_id)
                if parent is None or not parent.resume_path or parent.stop_reason == "early_stopping":
                    raise ValueError("continuation requires a successful resumable trial without early stopping")
                config = parent.config.model_copy(update={"epochs": parent.epochs_completed + action.additional_epochs})
                candidate_id = parent.candidate_id
            trial, result = self._segment(bb, prepared, config, candidate_id, sequence, parent)
            trials.append(trial)
            name = f"trial_{sequence:04d}"
            names.append(name)
            bb.put(name, trial, producer=self.name)
            if result is not None:
                latest[candidate_id], results[candidate_id] = trial, result
        selected = rank_trials(latest.values(), accuracy_tolerance=self.accuracy_tolerance, equal_epoch_budgets=False)
        self._freeze(bb, selected, results[selected.candidate_id], trials, names)

    def _segment(self, bb, prepared, config, candidate_id, sequence, parent):
        started = time.monotonic()
        checkpoint = bb.blob_dir / "search" / f"{candidate_id}_s{sequence:04d}.ckpt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        result, validation, error, failure_kind = None, None, None, None
        try:
            worker = SSHWorker(config.worker)
            if parent is None:
                verified = worker.execute(bb.root, config, "verify", {}, epochs=1)
                if json.loads((verified / "result.json").read_text(encoding="utf-8")) != {"verified": True}:
                    raise ValueError("worker verification failed")
            resume = bb.root / parent.resume_path if parent else None
            previous = bb.root / parent.checkpoint_path if parent else None
            if parent and (sha256_file(resume) != parent.resume_sha256 or sha256_file(previous) != parent.checkpoint_sha256):
                raise ValueError("continuation input checksum mismatch")
            start_epoch = parent.epochs_completed if parent else 0
            output = worker.execute(bb.root, config, "train",
                prepared_arrays(prepared, ["train", "val"], config.batch_size, seed=config.seed),
                checkpoint=previous, resume=resume, start_epoch=start_epoch)
            data = json.loads((output / "result.json").read_text(encoding="utf-8"))
            if data.get("stop_reason") not in {"segment_complete", "early_stopping"}:
                raise ValueError("autonomous training must report its stop reason")
            state = output / "resume.pt"
            if not state.is_file():
                raise ValueError("autonomous training must save latest continuation state")
            shutil.copyfile(output / "model.ckpt", checkpoint)
            resume_target = checkpoint.with_suffix(".resume.pt")
            shutil.copyfile(state, resume_target)
            log = checkpoint.with_suffix(".training.jsonl")
            shutil.copyfile(output / "training.jsonl", log)
            data.update(checkpoint_path=checkpoint.relative_to(bb.root).as_posix(),
                checkpoint_sha256=sha256_file(checkpoint), resume_path=resume_target.relative_to(bb.root).as_posix(),
                resume_sha256=sha256_file(resume_target), training_log_path=log.relative_to(bb.root).as_posix(),
                seed=config.seed, device=config.device, framework="isolated.python")
            result = TrainResult.model_validate(data)
            if not start_epoch < result.epochs_completed <= config.epochs:
                raise ValueError("autonomous segment has invalid completed epoch count")
            if result.stop_reason == "segment_complete" and result.epochs_completed != config.epochs:
                raise ValueError("segment stopped before its agent-selected target")
            if parent and result.history[:start_epoch] != parent.learning_curve:
                raise ValueError("continuation rewrote prior history")
            model = RemoteModel(bb.root, checkpoint, result.checkpoint_sha256, config)
            probabilities, targets = predict_probabilities(model, prepared, "val", config.batch_size,
                device=config.device, seed=config.seed)
            validation = evaluate_probabilities(probabilities, targets, prepared.bundle.labels, split="val")
        except Exception as exc:
            result = None
            error = str(exc)[:8000]
            failure_kind = ("timeout" if isinstance(exc, (TimeoutError, subprocess.TimeoutExpired)) else
                            "memory" if isinstance(exc, MemoryError) or "container exit 137" in error or "out of memory" in error.lower()
                            else "training_failure")
            bb.record_event("autonomous_segment_failed", candidate_id=candidate_id, error=error,
                interrupted=failure_kind in {"timeout", "memory"}, failure_kind=failure_kind)
        trial = TrialResult(candidate_id=candidate_id, config_hash=configuration_hash(config),
            round_index=sequence, epoch_budget=None, requested_epochs=config.epochs,
            parent_candidate_id=parent.candidate_id if parent else None,
            status="completed" if result else "failed", config=config, representation=_representation(bb),
            validation_accuracy=validation.accuracy if result else None,
            validation_macro_f1=validation.macro_f1 if result else None,
            validation_balanced_accuracy=validation.balanced_accuracy if result else None,
            epochs_completed=result.epochs_completed if result else None,
            learning_curve=result.history if result else [], best_epoch=result.best_epoch if result else None,
            checkpoint_path=result.checkpoint_path if result else None,
            checkpoint_sha256=result.checkpoint_sha256 if result else None,
            resume_path=result.resume_path if result else None, resume_sha256=result.resume_sha256 if result else None,
            stop_reason=result.stop_reason if result else failure_kind,
            training_log_path=result.training_log_path if result else None,
            duration_seconds=time.monotonic() - started, decision_source=config.source, error=error)
        return trial, result

    def _freeze(self, bb, selected, result, trials, names):
        targets = [trial.config.epochs for trial in trials
                   if trial.candidate_id == selected.candidate_id and trial.status == "completed"]
        selected = selected.model_copy(update={"config": selected.config.model_copy(update={"segment_targets": targets})})
        payload = {"format": "isolated-experiment-v1", "train_config": selected.config.model_dump(mode="json"),
                   "representation": selected.representation.model_dump(mode="json"),
                   "search_policy": "agent_autonomous", "checkpoint": selected.checkpoint_path}
        settings = bb.get_optional("run_configuration")
        if settings:
            payload["data"] = {key: settings.parameters.get(key) for key in
                ("data_root", "train_limit", "val_limit", "test_limit", "quick", "no_download")}
        path = bb.root / "best_config_v001.yaml"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (bb.root / "best_config.yaml").write_bytes(path.read_bytes())
        bb.put("best_configuration", BestConfiguration(selected_candidate_id=selected.candidate_id,
            selected_config_hash=selected.config_hash, train_config=selected.config,
            representation=selected.representation, validation_accuracy=selected.validation_accuracy,
            validation_macro_f1=selected.validation_macro_f1,
            selection_rule="agent-selected durations; validation accuracy tolerance then macro-F1; test locked",
            lightning_config_path=path.relative_to(bb.root).as_posix(), lightning_config_sha256=sha256_file(path)), producer=self.name)
        bb.put("train_config", selected.config, producer=self.name)
        bb.put("selected_training_result", result, producer=self.name)
        bb.put("search_report", SearchReport(trial_artefacts=names,
            selection_rule="agent_durations_accuracy_tolerance_then_macro_f1",
            completed_trials=sum(t.status == "completed" for t in trials),
            failed_trials=sum(t.status == "failed" for t in trials), selected_candidate_id=selected.candidate_id,
            selected_config_hash=selected.config_hash, selected_validation_accuracy=selected.validation_accuracy,
            selected_validation_macro_f1=selected.validation_macro_f1), producer=self.name)

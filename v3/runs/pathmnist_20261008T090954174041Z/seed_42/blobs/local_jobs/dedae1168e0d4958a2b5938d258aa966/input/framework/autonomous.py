"""Agent-directed search: no external trial, round or epoch horizon."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
import shutil
import time

from contracts import (AutonomousRecoveryDecision, AutonomousSearchDecision, AutonomousSearchEvent, BestConfiguration,
                       SearchPlan, SearchReport, TrainConfig, TrainResult, TrialResult, ReviewResponse, sha256_file)
from generated_agents import CODE_INTERFACE, _representation, configuration_hash
from ml import evaluate_probabilities, predict_probabilities
from remote import RemoteModel, LocalWorker, archive_bundle, prepared_arrays
from search import rank_trials
from llm import OllamaDecisionError
from failures import (SearchCircuitBreaker, TERMINAL_FAILURES,
                      failure_kind as classify_failure, failure_signature, is_timeout)


AUTONOMOUS_INTERFACE = CODE_INTERFACE.split("Optional propose(context)")[0].replace(
    "epochs (allocated epoch horizon)", "epochs (agent-selected cumulative target)") + '''
This run is agent_autonomous. YOU choose every active experimental parameter and
all search actions. No allocated epoch ceiling, trial count or round count exists.
Choose new_trial (complete experiment, explicit training options and initial epochs),
continue_trial (existing candidate_id and positive additional_epochs), or finish_search.
For new_trial, experiment is required; candidate_id and additional_epochs must be null.
The framework assigns the new candidate_id; do not invent one for new_trial.
For continue_trial, experiment must be null. For finish_search, all three arguments
must be null. Do not include fields belonging to another action.
Choose only from available_actions and resumable_candidate_ids supplied in the request.
In experiment.training, explicitly supply every active optimizer and scheduler option:
sgd requires momentum and nesterov; adam/adamw require adam_beta1, adam_beta2 and
optimizer_eps.
optimizer_eps must be between 1e-12 and 0.01 inclusive (for example 1e-8);
it is the numerical stability epsilon, not the learning rate or an Adam beta.
cosine requires cosine_eta_min; one_cycle requires one_cycle_pct_start;
reduce_on_plateau requires plateau_factor and plateau_patience. scheduler='none'
needs no scheduler-specific options. These choices are required even when you choose
a value equal to a schema default. Select the values yourself; omit inactive options.
Use validation only. Justify every choice and stop when further research is not useful.
Experiments must support exact continuation. train(context) saves BOTH model.ckpt
(best checkpoint) and resume.pt (latest state), both inside context['output']. context['epochs'] is the cumulative
agent-selected target; segment_epochs is the newly requested duration, start_epoch is
the previously completed epoch, resume is a read-only input state or None.
resume is a filesystem path string, NOT a state dictionary. Load it with
torch.load(context['resume'], map_location='cpu', weights_only=False), or use fit_model.
History is cumulative. Return stop_reason='segment_complete' or 'early_stopping'.
Verification trains a synthetic epoch, then resumes into a new output directory for
epoch two. Honour context['start_epoch']; use context.get('scheduler_horizon', context['epochs'])
for a custom fixed-horizon scheduler. Verification disables early stopping for this probe.
The resume.pt dictionary must contain format='agent-resume-v1', identity, data_identity,
epochs_completed, model, optimizer, scheduler, rng, loader_rng, history, best,
best_epoch, stale, stopped. Obtain identity using autonomous.continuation_identity(config)
and data_identity using adaptive_training.data_identity(data). Preserve preprocessing and other custom state too.
worker_runtime.fit_model supports this protocol and selects the explicitly supplied
optimizer/scheduler/loss controls. A custom loop must preserve equivalent state.
Continuation must not change code, parameters, optimizer or preprocessing. Create a
new trial for changes. A scheduler with an exhausted fixed horizon cannot be extended.
The local subprocess timeout is configurable (default one hour per operation);
resource interruption is a failed segment, never a completed trial. Decisions must
account for observed runtime and memory. Epochs have no fixed upper bound.
Use only dependencies already installed in the current Python environment. Preserve learned
preprocessing parameters in the checkpoint. All source code belongs to this run.
''' + "\nHelper-based starting example (adapt build_model to your hypothesis):\n" + \
    Path(__file__).with_name("default_experiment.py").read_text(encoding="utf-8")


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


@dataclass
class AutonomousSearchState:
    sequence: int = 0
    candidate_sequence: int = 0
    trials: list[TrialResult] = field(default_factory=list)
    trial_names: list[str] = field(default_factory=list)
    latest: dict[str, TrialResult] = field(default_factory=dict)
    results: dict[str, TrainResult] = field(default_factory=dict)
    identities: set[str] = field(default_factory=set)
    consecutive_failures: int = 0
    same_failures: int = 0
    last_signature: str | None = None

    @classmethod
    def empty(cls):
        return cls()

    @classmethod
    def restore(cls, bb):
        from resume import verify_store
        verify_store(bb)
        state = cls.empty()
        names = sorted((n for n in bb.artefacts if n.startswith("trial_")),
                       key=lambda n: int(n.removeprefix("trial_")))
        for name in names:
            trial = bb.get(name)
            if trial.config.execution_mode != "agent_autonomous":
                raise ValueError("cannot restore non-autonomous partial search")
            if int(name.removeprefix("trial_")) != trial.round_index:
                raise ValueError("trial sequence mismatch")
            state.sequence = max(state.sequence, trial.round_index)
            state.candidate_sequence = max(state.candidate_sequence, int(trial.candidate_id.removeprefix("autonomous_t")))
            state.trials.append(trial)
            state.trial_names.append(name)
            state.identities.add(continuation_identity(trial.config))
            if trial.status == "completed":
                previous = state.latest.get(trial.candidate_id)
                if previous and (continuation_identity(previous.config) != continuation_identity(trial.config)
                                 or trial.epochs_completed <= previous.epochs_completed
                                 or trial.learning_curve[:previous.epochs_completed] != previous.learning_curve):
                    raise ValueError("persisted continuation identity/history mismatch")
                state.latest[trial.candidate_id] = trial
                state.results[trial.candidate_id] = restore_training_result(trial)
                state.consecutive_failures = state.same_failures = 0
                state.last_signature = None
            else:
                state.consecutive_failures += 1
                state.same_failures = state.same_failures + 1 if trial.failure_signature == state.last_signature else 1
                state.last_signature = trial.failure_signature
        # An accepted action without a persisted trial is an unfinished operation.
        # Reserve its identifiers; never overwrite its generated code or checkpoints.
        for name in bb.artefacts:
            if name.startswith("autonomous_action_"):
                seq = int(name.removeprefix("autonomous_action_"))
                if seq > state.sequence:
                    action = bb.get(name).decision
                    state.sequence = seq
                    if action.action == "new_trial":
                        state.candidate_sequence += 1
        return state


def restore_training_result(trial):
    """Legacy migration derives metrics only from the recorded cumulative curve."""
    if trial.training_result is not None:
        return trial.training_result
    if (not trial.learning_curve or len(trial.learning_curve) != trial.epochs_completed
            or trial.best_epoch is None or not 1 <= trial.best_epoch <= trial.epochs_completed
            or trial.resume_path is None or trial.resume_sha256 is None
            or trial.stop_reason not in {"segment_complete", "early_stopping"}):
        raise ValueError("legacy trial lacks lossless training evidence; cannot resume")
    final = trial.learning_curve[-1]
    best = trial.learning_curve[trial.best_epoch - 1]
    return TrainResult(final_train_loss=final.train_loss, final_val_accuracy=final.val_accuracy,
        best_val_accuracy=best.val_accuracy, best_epoch=trial.best_epoch,
        epochs_completed=trial.epochs_completed, history=trial.learning_curve,
        checkpoint_path=trial.checkpoint_path, checkpoint_sha256=trial.checkpoint_sha256,
        resume_path=trial.resume_path, resume_sha256=trial.resume_sha256, stop_reason=trial.stop_reason,
        training_log_path=trial.training_log_path, lightning_csv_path=trial.lightning_csv_path,
        seed=trial.config.seed, device=trial.config.device, framework="local.python")


class AutonomousSearchAgent:
    name = "model_search"
    # Decisions already have bounded retries. Restarting the entire stage would
    # discard its in-memory history and reuse checkpoint/candidate identifiers.
    retry_on_exception = False

    def __init__(self, reasoner, *, worker, seed, device, accuracy_tolerance=0.005,
                 max_consecutive_failures=3, max_same_signature_failures=2):
        self.reasoner, self.worker, self.seed, self.device = reasoner, worker, seed, device
        self.accuracy_tolerance = accuracy_tolerance
        if min(max_consecutive_failures, max_same_signature_failures) < 1:
            raise ValueError("failure thresholds must be positive")
        self.max_consecutive_failures = max_consecutive_failures
        self.max_same_signature_failures = max_same_signature_failures

    def run(self, bb):
        if bb.get_optional("evaluation_report") is not None:
            raise ValueError("cannot reopen model search after test evaluation")
        if not bb.get("data_audit_report").passed:
            raise ValueError("autonomous search requires a passed data audit")
        if not self.reasoner.enabled and not getattr(self.reasoner, "replay_root", None):
            raise ValueError("autonomous search requires live decisions or recorded replay")
        bb.put("search_plan", SearchPlan(policy="agent_autonomous", progressive_budget=False,
            framework="local.python", accuracy_tolerance=self.accuracy_tolerance), producer=self.name)
        prepared = bb.get_blob("prepared_data")
        state = AutonomousSearchState.restore(bb) if (bb.get_optional("resume_manifest") or
            any(n.startswith("trial_") for n in bb.artefacts)) else AutonomousSearchState.empty()
        latest, results, trials, names, identities = state.latest, state.results, state.trials, state.trial_names, state.identities
        sequence, candidate_sequence = state.sequence, state.candidate_sequence
        consecutive_failures, same_failures = state.consecutive_failures, state.same_failures
        last_signature = state.last_signature
        while True:
            sequence += 1
            resumable = sorted(candidate_id for candidate_id, trial in latest.items()
                               if trial.resume_path and trial.stop_reason != "early_stopping"
                               and trial.config.scheduler != "one_cycle")
            evidence = [{"candidate_id": t.candidate_id, "status": t.status,
                "validation_accuracy": t.validation_accuracy, "validation_macro_f1": t.validation_macro_f1,
                "epochs_completed": t.epochs_completed, "requested_target": t.config.epochs,
                "duration_seconds": t.duration_seconds, "stop_reason": t.stop_reason, "error": t.error,
                "failed_operation": t.failed_operation, "failure_kind": t.failure_kind,
                "failure_signature": t.failure_signature,
                "configuration": t.config.model_dump(mode="json", exclude={"worker"}),
                "learning_curve": [row.model_dump(mode="json") for row in t.learning_curve]} for t in trials]
            repair = None
            if trials and trials[-1].failure_kind == "code_validation":
                from remote import validate_bundle
                reference = trials[-1].config.generated_bundle
                directory = validate_bundle(bb.root, reference)
                repair = {"instruction": "Repair the existing program, do not redesign the experiment.",
                          "previous_sources": {p.relative_to(directory).as_posix(): p.read_text(encoding="utf-8")
                                               for p in sorted(directory.rglob("*.py"))},
                          "traceback": trials[-1].error, "operation": trials[-1].failed_operation,
                          "input_shapes": {"train_images": [9, 3, 28, 28], "val_images": [9, 3, 28, 28]},
                          "contract": AUTONOMOUS_INTERFACE}
            available = (["new_trial"] + (["continue_trial"] if resumable else [])
                         + (["finish_search"] if latest else [])) if not repair else ["new_trial"]
            decision_started = time.monotonic()
            bb.record_event("autonomous_decision_started", decision_sequence=sequence,
                completed_trials=sum(t.status == "completed" for t in trials),
                failed_trials=sum(t.status == "failed" for t in trials),
                consecutive_failures=consecutive_failures, resumable_candidate_ids=resumable,
                available_actions=available)
            try:
                decision = self.reasoner.decide(stage=f"autonomous_search.action_{sequence}",
                    system=AUTONOMOUS_INTERFACE,
                    user=json.dumps({"profile": bb.get("data_profile").model_dump(mode="json"),
                        "research": bb.get("architecture_research").model_dump(mode="json"),
                        "validation_evidence": evidence, "test_split": "locked",
                        "review_feedback": (bb.get("anomaly_model_search").model_dump(mode="json")
                                            if bb.get_optional("anomaly_model_search") else None),
                        "revision_instruction": "Address every corrective request with new evidence or a justified "
                            "contestation. Compare checkpoint metrics, not curve maxima. Use archived program source "
                            "and generated bundle parameters to identify active model settings; generic architecture "
                            "defaults do not establish instantiated model shape. If previous evidence "
                            "was insufficient, change the correction strategy. Only the reviewer can close a request.",
                        "available_actions": available, "repair": repair,
                        "resumable_candidate_ids": resumable,
                        "resources": {"operation_timeout_seconds": self.worker.timeout_seconds,
                                      "backend": "local.subprocess", "memory_limit_enforced": False}}, ensure_ascii=False, sort_keys=True),
                    response_model=AutonomousSearchDecision,
                    fallback=AutonomousSearchDecision(action="finish_search", rationale="No fallback is permitted."),
                    audit=bb.record_event)
            except OllamaDecisionError as exc:
                bb.record_event("autonomous_decision_failed", decision_sequence=sequence,
                    elapsed_seconds=time.monotonic() - decision_started, error=str(exc)[:8000],
                    failure_kind="llm_timeout" if is_timeout(exc) else "llm_error")
                if not latest:
                    raise
                decision = self.reasoner.decide(stage=f"autonomous_search.recovery_{sequence}",
                    system="The proposed trial could not be validated. Valid completed trials exist. "
                        "Choose finish_search or continue_trial of an eligible candidate. No new experiment is allowed.",
                    user=json.dumps({"validation_evidence": evidence, "test_split": "locked",
                                     "resumable_candidate_ids": resumable, "failed_decision": str(exc)}),
                    response_model=AutonomousRecoveryDecision,
                    fallback=AutonomousRecoveryDecision(action="finish_search", rationale="No fallback permitted"),
                    audit=bb.record_event)
                if decision.used_fallback:
                    raise OllamaDecisionError("autonomous recovery requires a live validated decision")
                from llm import ReasonedDecision
                decision = ReasonedDecision(AutonomousSearchDecision.model_validate(decision.value.model_dump()),
                    decision.source, False, decision.attempts, decision.request_id)
                repair = None
            bb.record_event("autonomous_decision_finished", decision_sequence=sequence,
                elapsed_seconds=time.monotonic() - decision_started, selected_action=decision.value.action)
            if decision.used_fallback:
                raise ValueError("autonomous decisions cannot use heuristic fallback")
            action = decision.value
            if repair and action.action != "new_trial":
                raise ValueError("code validation failure requires repair before another search action")
            bb.put(f"autonomous_action_{sequence:04d}", AutonomousSearchEvent(decision=action,
                source=decision.source), producer=self.name)
            if action.action == "finish_search":
                if not latest:
                    raise ValueError("cannot finish search without a successful trial")
                break
            parent = None
            if action.action == "new_trial":
                previous_reference = trials[-1].config.generated_bundle if repair else None
                experiment = action.experiment
                if previous_reference:
                    experiment = experiment.model_copy(update={"bundle_id": previous_reference.bundle_id})
                reference = archive_bundle(bb, experiment, decision.source, parent=previous_reference)
                config = experiment_config(action.experiment, reference, self.worker, self.seed, self.device, decision.source)
                identity = continuation_identity(config)
                if identity in identities:
                    raise ValueError("duplicate autonomous configuration; continue the existing trial instead")
                identities.add(identity)
                candidate_sequence += 1
                candidate_id = f"autonomous_t{candidate_sequence:04d}"
            else:
                parent = latest.get(action.candidate_id)
                if action.candidate_id not in resumable:
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
                consecutive_failures = same_failures = 0
                last_signature = None
            else:
                consecutive_failures += 1
                same_failures = same_failures + 1 if trial.failure_signature == last_signature else 1
                last_signature = trial.failure_signature
                if (trial.failure_kind in TERMINAL_FAILURES or
                        consecutive_failures >= self.max_consecutive_failures or
                        same_failures >= self.max_same_signature_failures):
                    bb.record_event("autonomous_circuit_breaker", candidate_id=candidate_id,
                        failure_kind=trial.failure_kind, failure_signature=last_signature,
                        consecutive_failures=consecutive_failures, same_signature_failures=same_failures)
                    raise SearchCircuitBreaker(f"autonomous search stopped after {consecutive_failures} "
                                               f"consecutive failures ({trial.failure_kind})")
        selected = rank_trials(latest.values(), accuracy_tolerance=self.accuracy_tolerance, equal_epoch_budgets=False)
        self._freeze(bb, selected, results[selected.candidate_id], trials, names, rationale=action.rationale)
        previous = bb.get_optional("anomaly_model_search")
        if previous and previous.action == "revise":
            bb.put("review_response_model_search", ReviewResponse(stage=self.name,
                review_attempt=previous.attempt, response=action.rationale,
                evidence=names + ["search_report", "best_configuration"],
                evidence_versions=[{"artefact": entry["artefact"], "version": entry["version"],
                    "artefact_path": entry["path"], "payload_sha256": entry["sha256"]}
                    for entry in bb.registry if entry["artefact"] in names
                    or (entry["artefact"] in {"search_report", "best_configuration"}
                        and entry["version"] == bb._versions[entry["artefact"]])]), producer=self.name)

    def revise(self, bb, report):
        # run() restores the append-only trials and action sequence on the next iteration.
        bb.record_event("search_revision_requested", attempt=report.attempt,
                        requests=[r.model_dump(mode="json") for r in report.requests],
                        issues=report.llm_issues)
        return True

    def _segment(self, bb, prepared, config, candidate_id, sequence, parent):
        started = time.monotonic()
        checkpoint = bb.blob_dir / "search" / f"{candidate_id}_s{sequence:04d}.ckpt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        result, validation, error, failure_kind = None, None, None, None
        operation, signature = "verify", None
        representation = {"normalization": prepared.normalization, "augmentations": list(prepared.augmentations),
                          "mean": list(prepared.mean), "std": list(prepared.std),
                          "layout": "NCHW", "already_prepared": True}
        try:
            worker = LocalWorker(config.worker)
            if parent is None:
                verified = worker.execute(bb.root, config, "verify", {}, epochs=1,
                                          candidate_id=candidate_id, representation=representation)
                if json.loads((verified / "result.json").read_text(encoding="utf-8")) != {"verified": True}:
                    raise ValueError("worker verification failed")
            operation = "result_validation"
            resume = bb.root / parent.resume_path if parent else None
            previous = bb.root / parent.checkpoint_path if parent else None
            if parent and (sha256_file(resume) != parent.resume_sha256 or sha256_file(previous) != parent.checkpoint_sha256):
                raise ValueError("continuation input checksum mismatch")
            start_epoch = parent.epochs_completed if parent else 0
            operation = "preparation"
            arrays = prepared_arrays(prepared, ["train", "val"], config.batch_size, seed=config.seed)
            operation = "train"
            output = worker.execute(bb.root, config, "train", arrays,
                checkpoint=previous, resume=resume, start_epoch=start_epoch,
                candidate_id=candidate_id, representation=representation)
            operation = "result_validation"
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
                seed=config.seed, device=config.device, framework="local.python")
            result = TrainResult.model_validate(data)
            if not start_epoch < result.epochs_completed <= config.epochs:
                raise ValueError("autonomous segment has invalid completed epoch count")
            if result.stop_reason == "segment_complete" and result.epochs_completed != config.epochs:
                raise ValueError("segment stopped before its agent-selected target")
            if parent and result.history[:start_epoch] != parent.learning_curve:
                raise ValueError("continuation rewrote prior history")
            model = RemoteModel(bb.root, checkpoint, result.checkpoint_sha256, config)
            operation = "infer"
            probabilities, targets = predict_probabilities(model, prepared, "val", config.batch_size,
                device=config.device, seed=config.seed)
            validation = evaluate_probabilities(probabilities, targets, prepared.bundle.labels, split="val")
        except Exception as exc:
            result = None
            error = str(exc)[:8000]
            operation = getattr(exc, "operation", operation)
            failure_kind = classify_failure(exc, operation)
            signature = getattr(exc, "failure_signature", None) or failure_signature(error, failure_kind)
            bb.record_event("autonomous_segment_failed", candidate_id=candidate_id, error=error,
                interrupted=failure_kind in {"timeout", "oom"}, failure_kind=failure_kind,
                operation=operation, failure_signature=signature)
        trial = TrialResult(training_result=result, candidate_id=candidate_id, config_hash=configuration_hash(config),
            round_index=sequence, epoch_budget=None, requested_epochs=config.epochs,
            parent_candidate_id=parent.candidate_id if parent else None,
            status="completed" if result else "failed", config=config, representation=_representation(bb),
            failed_operation=operation if not result else None,
            failure_kind=failure_kind, failure_signature=signature,
            validation_accuracy=validation.accuracy if result else None,
            validation_macro_f1=validation.macro_f1 if result else None,
            validation_balanced_accuracy=validation.balanced_accuracy if result else None,
            epochs_completed=result.epochs_completed if result else None,
            learning_curve=result.history if result else [], best_epoch=result.best_epoch if result else None,
            checkpoint_path=result.checkpoint_path if result else None,
            checkpoint_sha256=result.checkpoint_sha256 if result else None,
            resume_path=result.resume_path if result else None, resume_sha256=result.resume_sha256 if result else None,
            stop_reason=result.stop_reason if result else {"oom": "memory", "training_error": "training_failure"}.get(failure_kind, failure_kind),
            training_log_path=result.training_log_path if result else None,
            duration_seconds=time.monotonic() - started, decision_source=config.source, error=error)
        return trial, result

    def _freeze(self, bb, selected, result, trials, names, *, rationale=""):
        targets = [trial.config.epochs for trial in trials
                   if trial.candidate_id == selected.candidate_id and trial.status == "completed"]
        selected = selected.model_copy(update={"config": selected.config.model_copy(update={"segment_targets": targets})})
        payload = {"format": "run-experiment-v1", "train_config": selected.config.model_dump(mode="json"),
                   "representation": selected.representation.model_dump(mode="json"),
                   "search_policy": "agent_autonomous", "checkpoint": selected.checkpoint_path}
        settings = bb.get_optional("run_configuration")
        if settings:
            payload["data"] = {key: settings.parameters.get(key) for key in
                ("data_root", "train_limit", "val_limit", "test_limit", "quick", "no_download")}
        version = bb._versions.get("best_configuration", 0) + 1
        path = bb.root / f"best_config_v{version:03d}.yaml"
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
        from remote import validate_bundle
        program = (validate_bundle(bb.root, selected.config.generated_bundle) / "experiment.py").read_text(encoding="utf-8")
        bb.put("search_report", SearchReport(trial_artefacts=names,
            stopping_rationale=rationale, selected_program=program,
            families_evaluated=sorted({t.config.model_family for t in trials if t.status == "completed"}),
            evaluated_bundles=sorted({t.config.generated_bundle.bundle_id for t in trials
                                     if t.status == "completed" and t.config.generated_bundle}),
            validation_trials=[{"candidate_id": t.candidate_id, "round_index": t.round_index,
                "checkpoint_accuracy": t.validation_accuracy, "checkpoint_macro_f1": t.validation_macro_f1,
                "best_epoch": t.best_epoch, "curve_max_accuracy": max((r.val_accuracy for r in t.learning_curve), default=None),
                "bundle": t.config.generated_bundle.model_dump(mode="json") if t.config.generated_bundle else None,
                "generic_architecture_fields": "framework defaults; inspect archived build_model for actual parameter use",
                "duration_seconds": t.duration_seconds} for t in trials if t.status == "completed"],
            selection_rule="agent_durations_accuracy_tolerance_then_macro_f1",
            completed_trials=sum(t.status == "completed" for t in trials),
            failed_trials=sum(t.status == "failed" for t in trials), selected_candidate_id=selected.candidate_id,
            selected_config_hash=selected.config_hash, selected_validation_accuracy=selected.validation_accuracy,
            selected_validation_macro_f1=selected.validation_macro_f1), producer=self.name)

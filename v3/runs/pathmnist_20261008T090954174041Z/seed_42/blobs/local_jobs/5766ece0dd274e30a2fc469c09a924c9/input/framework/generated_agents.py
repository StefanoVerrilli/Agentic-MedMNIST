"""Experimental code agents, integrating exclusively through the Blackboard."""
from __future__ import annotations

import json
from pathlib import Path
import time

from contracts import (BestConfiguration, GeneratedCodeDecision, GeneratedDevelopmentPlan,
                       GeneratedSource, GeneratedVerification, SearchPlan, SearchReport,
                       TrainConfig, TrialResult, sha256_file)
from ml import train_model, evaluate_probabilities, predict_probabilities
from remote import archive_bundle, validate_bundle, LocalWorker
from research import canonical_hash
from search import rank_trials


CODE_INTERFACE = '''Write a complete run-scoped Python experiment, never modify framework files.
experiment.py must expose build_model(context), train(context), predict(context).
build_model returns torch.nn.Module producing finite float logits [batch,9] for float NCHW 28x28 RGB.
context is a dict: config (seed/device/batch_size/lr/epochs/early_stopping...);
parameters (your free JSON parameters); data (numpy arrays); output (absolute job output directory);
checkpoint (input checkpoint path); epochs (allocated epoch horizon).
Training data keys: train_images, train_targets, val_images, val_targets.
All image arrays (including inference) are finite float32 NCHW [N,3,28,28],
already normalized and augmented by the framework's representation plan.
Do not transpose NCHW inputs or apply channel normalization again. context['representation']
describes the applied transformation. Keep arrays on CPU and transfer only batches
to context['config']['device']; never silently force CPU or move the full dataset to CUDA.
Use worker_runtime.fit_model and batched_logits by default; custom loops are allowed
only if they preserve the same device, input, checkpoint and continuation contracts.
Inference data keys: images only. Test targets must never enter training or search.
train must save Path(context['output']) / 'model.ckpt' and return final_train_loss, final_val_accuracy,
best_val_accuracy, best_epoch, epochs_completed, history.
history rows: epoch (contiguous starting at 1), train_loss, val_accuracy;
optional val_loss, val_macro_f1, learning_rate. Never exceed context['epochs'].
predict returns numpy logits in original image order, loading context['checkpoint'].
Custom preprocessing, losses, optimizers, training loops and analysis are allowed.
worker_runtime.fit_model(context, model, loss_fn=None, optimizer=None) and
worker_runtime.batched_logits(model, images, config) are optional helpers.
lightning_components.build_network provides the seven built-in model families.
Its signature is build_network(model_family, *, in_channels, n_classes, hidden, depth,
dropout, patch_size=4, num_heads=4, mlp_ratio=4, pooling='cls', ...).
Use n_classes=9 (not num_classes); no image_size argument exists. Families include
feature_pyramid_transformer (hidden=256, depth=1, four ResNet stages, ST dot-product MoS(2), GT negative Euclidean MoS(4), convolutional channel rendering RT), and multi_scale_transformer with scales=(2,4,7), hidden=64, depth=2, num_heads=4, pooling='mean'.
Optional propose(context) implements your own search strategy. It sees context['evidence']
(prior validation results only), no datasets, and returns a list of objects containing
'parameters' (JSON object), optional 'epochs' (1..1000) and 'batch_size' (4..4096).
Code executes in local subprocesses using the current Python environment. Use context
paths, never hardcode /input or /output. Write new code and artifacts only within this run.
Preserve learned preprocessing in checkpoints. Use installed Python dependencies.
'''


def fallback_decision() -> GeneratedCodeDecision:
    return GeneratedCodeDecision(bundle_id="multiscale_experiment",
        hypothesis="Fuse fine and coarse tissue tokens with a compact multi-scale Transformer.",
        files=[GeneratedSource(path="experiment.py", code=Path(__file__).with_name("default_experiment.py").read_text(encoding="utf-8"))],
        parameters={"hidden": 64, "depth": 2, "scales": [2, 4, 7], "dropout": 0.1},
        epochs=100, batch_size=128)


def propose_code(reasoner, bb, stage, fallback, context):
    result = reasoner.decide(stage=stage, system=CODE_INTERFACE,
        user=json.dumps(context, ensure_ascii=False), response_model=GeneratedCodeDecision,
        fallback=fallback, audit=bb.record_event)
    return result.value, result.source


def generated_config(decision, reference, worker, seed, device, ceiling, source):
    return TrainConfig(model_family="run_generated", generated_bundle=reference, worker=worker,
        lr=0.0005, epochs=min(decision.epochs, ceiling), hidden=64, depth=2, dropout=0.1,
        batch_size=decision.batch_size, weight_decay=0.0001, class_weighting=False,
        seed=seed, device=device, rationale=decision.hypothesis, source=source)


def configuration_hash(config):
    payload = config.model_dump(mode="json", exclude={"seed", "device", "worker", "source", "rationale", "segment_targets"})
    payload["generated_bundle"].pop("path", None)
    return canonical_hash(payload)


class ExperimentalDevelopmentAgent:
    name = "experimental_development"

    def __init__(self, reasoner):
        self.reasoner = reasoner

    def run(self, bb):
        if not bb.get("data_audit_report").passed:
            raise ValueError("code development requires the data audit")
        decision, source = propose_code(self.reasoner, bb, self.name, fallback_decision(),
            {"profile": bb.get("data_profile").model_dump(mode="json"),
             "research": bb.get("architecture_research").model_dump(mode="json"),
             "test_split": "locked"})
        for attempt in range(3):
            try:
                reference = archive_bundle(bb, decision, source)
                break
            except (ValueError, SyntaxError) as exc:
                if attempt == 2 or not (self.reasoner.enabled or getattr(self.reasoner, "replay_root", None)):
                    raise
                decision, source = propose_code(self.reasoner, bb,
                    f"experimental_development.syntax_{attempt + 1}", decision,
                    {"previous": decision.model_dump(mode="json"), "syntax_error": str(exc)[:4000]})
        bb.put("generated_development_plan", GeneratedDevelopmentPlan(decision=decision,
            reference=reference), producer=self.name)


class GeneratedSearchAgent:
    name = "model_search"

    def __init__(self, reasoner, *, worker, seed, device, max_trials, rounds,
                 search_epochs, final_epochs, accuracy_tolerance):
        self.reasoner, self.worker, self.seed, self.device = reasoner, worker, seed, device
        self.max_trials, self.rounds = max_trials, rounds
        self.search_epochs, self.final_epochs = search_epochs, final_epochs
        self.accuracy_tolerance = accuracy_tolerance

    def _trial(self, bb, decision, reference, ceiling, index, sequence, source):
        started = time.monotonic()
        deadline = started + self.worker.timeout_seconds
        prepared = bb.get_blob("prepared_data")
        error, output, validation = None, None, None
        config = generated_config(decision, reference, self.worker, self.seed, self.device, ceiling, source)
        checkpoint = bb.blob_dir / "search" / f"generated_trial_{sequence:03d}.ckpt"
        for attempt in range(3):
            bb.record_event("search_trial_started", stage=self.name, candidate_id=reference.bundle_id,
                epoch_budget=ceiling, bundle_sha256=reference.sha256, attempt=attempt + 1)
            try:
                def budgeted_config():
                    remaining = int(deadline - time.monotonic())
                    if remaining < 1:
                        raise TimeoutError("generated trial exhausted its wall-clock budget")
                    return config.model_copy(update={"worker": self.worker.model_copy(
                        update={"timeout_seconds": remaining})})
                validate_bundle(bb.root, reference)
                operation_config = budgeted_config()
                verified = LocalWorker(operation_config.worker).execute(bb.root, operation_config, "verify", {}, epochs=1)
                if json.loads((verified / "result.json").read_text(encoding="utf-8")) != {"verified": True}:
                    raise ValueError("worker did not confirm verification")
                bb.put(f"generated_verification_{sequence:03d}", GeneratedVerification(
                    reference=reference, passed=True), producer=self.name)
                output = train_model(prepared, budgeted_config(), checkpoint)
                output.model.config = budgeted_config()
                probabilities, targets = predict_probabilities(output.model, prepared, "val",
                    config.batch_size, device=config.device, seed=config.seed)
                validation = evaluate_probabilities(probabilities, targets, prepared.bundle.labels, split="val")
                error = None
                break
            except (Exception) as exc:
                error = str(exc)[:8000]
                bb.put(f"generated_verification_{sequence:03d}", GeneratedVerification(
                    reference=reference, passed=False, error=error), producer=self.name)
                bb.record_event("generated_trial_error", trial=sequence, attempt=attempt + 1,
                    bundle_sha256=reference.sha256, error=error)
                if attempt == 2 or not (self.reasoner.enabled or getattr(self.reasoner, "replay_root", None)):
                    break
                if time.monotonic() >= deadline:
                    break
                corrected, correction_source = propose_code(self.reasoner, bb,
                    f"experimental_development.correct_{sequence}_{attempt + 1}", decision,
                    {"previous": decision.model_dump(mode="json"), "error": error,
                     "instruction": "Correct this experiment. Preserve the interface and budget."})
                try:
                    reference = archive_bundle(bb, corrected, correction_source, parent=reference)
                except (ValueError, SyntaxError) as archive_error:
                    error = str(archive_error)
                    break
                decision, source = corrected, correction_source
                config = generated_config(decision, reference, self.worker, self.seed, self.device, ceiling, source)
        success = error is None and validation is not None
        return TrialResult(candidate_id=f"{reference.bundle_id}_t{sequence:03d}",
            config_hash=configuration_hash(config), round_index=index, epoch_budget=ceiling,
            requested_epochs=decision.epochs, status="completed" if success else "failed",
            config=config, representation=_representation(bb),
            validation_accuracy=validation.accuracy if success else None,
            validation_macro_f1=validation.macro_f1 if success else None,
            validation_balanced_accuracy=validation.balanced_accuracy if success else None,
            epochs_completed=output.result.epochs_completed if success else None,
            learning_curve=output.result.history if success else [],
            best_epoch=output.result.best_epoch if success else None,
            checkpoint_path=checkpoint.relative_to(bb.root).as_posix() if success else None,
            checkpoint_sha256=output.result.checkpoint_sha256 if success else None,
            training_log_path=Path(output.result.training_log_path).relative_to(bb.root).as_posix() if success else None,
            duration_seconds=time.monotonic() - started, decision_source=source, error=error)

    def run(self, bb):
        if not bb.get("data_audit_report").passed:
            raise ValueError("generated search requires a passed audit")
        development = bb.get("generated_development_plan")
        decision, reference = development.decision, development.reference
        ceilings = [self.final_epochs if index == self.rounds else min(self.final_epochs, self.search_epochs * index)
                    for index in range(1, self.rounds + 1)]
        bb.put("search_plan", SearchPlan(max_trials=self.max_trials, rounds=self.rounds,
            search_epochs=self.search_epochs, final_epochs=self.final_epochs,
            framework="local.python",
            accuracy_tolerance=self.accuracy_tolerance, round_epoch_budgets=ceilings), producer=self.name)
        trials, names, decisions = [], [], {}
        seen = set()
        for index, ceiling in enumerate(ceilings, 1):
            slots = self.max_trials // self.rounds + (index <= self.max_trials % self.rounds)
            successful = [trial for trial in trials if trial.status == "completed"]
            latest = {}
            for trial in successful:
                latest[trial.config.generated_bundle.sha256 + canonical_hash(trial.config.generated_bundle.parameters)] = trial
            promotions = sorted(latest.values(), key=lambda trial: (
                trial.validation_accuracy, trial.validation_macro_f1), reverse=True)
            final = index == self.rounds and bool(promotions)
            if index > 1 and not final:
                decision, source = propose_code(self.reasoner, bb,
                    f"experimental_development.round_{index}", decision,
                    {"previous": decision.model_dump(mode="json"),
                     "validation_evidence": _evidence(trials), "allocated_epochs": ceiling})
                reference = archive_bundle(bb, decision, source, parent=reference)
            else:
                source = "frozen_promotion" if final else "experimental_development"
            strategies = []
            if not final:
                try:
                    config = generated_config(decision, reference, self.worker, self.seed, self.device, ceiling, source)
                    result = LocalWorker(self.worker).execute(bb.root, config, "strategy", {}, evidence=_evidence(trials))
                    strategies = json.loads((result / "strategy.json").read_text(encoding="utf-8"))
                    if not isinstance(strategies, list) or len(strategies) > 256:
                        raise ValueError("propose must return at most 256 suggestions")
                except Exception as exc:
                    bb.record_event("generated_strategy_failed", error=str(exc)[:2000], round=index)
                    strategies = []
            for slot in range(slots):
                active, ref = decision, reference
                if final:
                    if slot >= len(promotions):
                        break
                    promoted = promotions[slot]
                    ref = promoted.config.generated_bundle
                    active = decisions[promoted.candidate_id]
                elif slot and slot - 1 < len(strategies):
                    suggestion = strategies[slot - 1]
                    if not isinstance(suggestion, dict) or set(suggestion) - {"parameters", "epochs", "batch_size"}:
                        bb.record_event("generated_strategy_rejected", round=index, slot=slot)
                    else:
                        try:
                            payload = decision.model_dump()
                            payload.update(suggestion)
                            active = GeneratedCodeDecision.model_validate(payload)
                            ref = reference.model_copy(update={"parameters": active.parameters})
                        except ValueError as exc:
                            bb.record_event("generated_strategy_rejected", error=str(exc)[:1000])
                proposed_config = generated_config(active, ref, self.worker, self.seed, self.device, ceiling, source)
                identity = (configuration_hash(proposed_config), ceiling)
                if identity in seen:
                    bb.record_event("generated_duplicate_skipped", round=index, config_hash=identity[0])
                    continue
                seen.add(identity)
                trial = self._trial(bb, active, ref, ceiling, index, len(trials) + 1, source)
                trials.append(trial)
                # Corrections can replace source and parameters; persist actual executable identity.
                if trial.config.generated_bundle.sha256 != ref.sha256:
                    bundle_dir = validate_bundle(bb.root, trial.config.generated_bundle)
                    active = active.model_copy(update={"files": [GeneratedSource(path=path.relative_to(bundle_dir).as_posix(),
                        code=path.read_text(encoding="utf-8")) for path in sorted(bundle_dir.rglob("*.py"))],
                        "parameters": trial.config.generated_bundle.parameters,
                        "epochs": trial.requested_epochs, "batch_size": trial.config.batch_size})
                decisions[trial.candidate_id] = active
                name = f"trial_{len(trials):03d}"
                names.append(name)
                bb.put(name, trial, producer=self.name)
        finalist = [trial for trial in trials if trial.round_index == self.rounds and trial.status == "completed"]
        selected = rank_trials(finalist, accuracy_tolerance=self.accuracy_tolerance)
        payload = {"format": "run-experiment-v1", "train_config": selected.config.model_dump(mode="json"),
                   "representation": selected.representation.model_dump(mode="json")}
        settings = bb.get_optional("run_configuration")
        if settings:
            payload["data"] = {key: settings.parameters.get(key) for key in
                ("data_root", "train_limit", "val_limit", "test_limit", "quick", "no_download")}
        path = bb.root / "best_config_v001.yaml"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (bb.root / "best_config.yaml").write_bytes(path.read_bytes())
        best = BestConfiguration(selected_candidate_id=selected.candidate_id,
            selected_config_hash=selected.config_hash, train_config=selected.config,
            representation=selected.representation, validation_accuracy=selected.validation_accuracy,
            validation_macro_f1=selected.validation_macro_f1,
            selection_rule="highest allocated epoch ceiling; validation accuracy tolerance then macro-F1; test locked",
            lightning_config_path=path.relative_to(bb.root).as_posix(), lightning_config_sha256=sha256_file(path))
        bb.put("train_config", selected.config, producer=self.name)
        bb.put("best_configuration", best, producer=self.name)
        bb.put("search_report", SearchReport(trial_artefacts=names,
            completed_trials=len(successful := [trial for trial in trials if trial.status == "completed"]),
            failed_trials=len(trials) - len(successful), selected_candidate_id=selected.candidate_id,
            selected_config_hash=selected.config_hash, selected_validation_accuracy=selected.validation_accuracy,
            selected_validation_macro_f1=selected.validation_macro_f1), producer=self.name)


def _evidence(trials):
    return [{"candidate_id": trial.candidate_id, "status": trial.status, "epoch_budget": trial.epoch_budget,
             "validation_accuracy": trial.validation_accuracy, "validation_macro_f1": trial.validation_macro_f1,
             "error": trial.error, "epochs_completed": trial.epochs_completed,
             "parameters": trial.config.generated_bundle.parameters,
             "learning_curve": [row.model_dump(mode="json") for row in trial.learning_curve]}
            for trial in trials]


def _representation(bb):
    from contracts import RepresentationDecision
    plan = bb.get("representation_plan")
    return RepresentationDecision(normalization=plan.normalization, augmentations=plan.augmentations,
                                  rationale=plan.rationale)

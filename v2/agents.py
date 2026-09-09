"""Cohesive agents for the PathMNIST hybrid training workflow."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, Protocol

from contracts import (
    AbstentionReport,
    AmbiguityDecision,
    AnomalyReport,
    BestConfiguration,
    Blackboard,
    CandidateProposal,
    DataProfile,
    DatasetManifest,
    ExperimentDecision,
    OODScenario,
    PredictionArtifact,
    ReportingStatus,
    RepresentationDecision,
    RepresentationPlan,
    ReviewDecision,
    SearchDecision,
    SearchPlan,
    SearchReport,
    SplitManifest,
    TrainConfig,
    TrialResult,
    sha256_file,
    utc_now,
)
from llm import Reasoner
from ml import (
    DatasetBundle,
    blank_fraction,
    channel_statistics,
    count_duplicate_images,
    evaluate_probabilities,
    inverse_frequency_class_weights,
    load_pathmnist,
    predict_outputs,
    predict_probabilities,
    prepare_data,
    save_prediction_arrays,
    train_model,
)
from search import (
    candidate_hash,
    default_candidates,
    lightning_config_payload,
    proposal_to_config,
    proposal_to_representation,
    rank_trials,
    safe_trial_name,
)


class Agent(Protocol):
    name: str

    def run(self, bb: Blackboard) -> None: ...


class IngestionAgent:
    """Load only PathMNIST and preserve its official train/val/test split."""

    name = "ingestion"

    def __init__(
        self,
        *,
        seed: int = 42,
        train_limit: int | None = None,
        val_limit: int | None = None,
        test_limit: int | None = None,
        data_root: str | Path | None = None,
        download: bool = True,
        loader: Callable[..., DatasetBundle] = load_pathmnist,
    ):
        self.seed = seed
        self.train_limit = train_limit
        self.val_limit = val_limit
        self.test_limit = test_limit
        self.data_root = data_root
        self.download = download
        self.loader = loader

    def run(self, bb: Blackboard) -> None:
        bundle = self.loader(
            seed=self.seed,
            train_limit=self.train_limit,
            val_limit=self.val_limit,
            test_limit=self.test_limit,
            data_root=self.data_root,
            download=self.download,
        )
        loaded_sizes = {
            split: len(bundle.targets[split]) for split in ("train", "val", "test")
        }
        manifest = DatasetManifest(
            dataset="pathmnist",
            task="multi-class",
            n_channels=3,
            n_classes=9,
            labels=list(bundle.labels),
            official_split_sizes=bundle.official_sizes,
            loaded_split_sizes=loaded_sizes,
            selected_index_sha256=bundle.selected_index_sha256,
            data_reference=bundle.data_reference,
            archive_md5=bundle.archive_md5,
            license=bundle.license,
            test_domain_note=bundle.test_domain_note,
            subsampled=any(
                loaded_sizes[key] < bundle.official_sizes[key] for key in loaded_sizes
            ),
        )
        bb.put_blob("raw_dataset", bundle, producer=self.name)
        bb.put("dataset_manifest", manifest, producer=self.name)


class ProfilingAmbiguityAgent:
    """Profile every loaded sample and reason only about ambiguity risks."""

    name = "profiling"

    def __init__(self, reasoner: Reasoner):
        self.reasoner = reasoner

    def run(self, bb: Blackboard) -> None:
        import numpy as np

        bundle = _bundle(bb)
        manifest = _manifest(bb)
        counts = {
            split: np.bincount(bundle.targets[split], minlength=bundle.n_classes)
            .astype(int)
            .tolist()
            for split in ("train", "val", "test")
        }
        positive = [value for value in counts["train"] if value > 0]
        imbalance = max(positive) / min(positive)
        proportions = {
            split: (np.asarray(values, dtype="float64") / sum(values)).tolist()
            for split, values in counts.items()
        }
        prevalence_ratio = (
            np.asarray(proportions["test"])
            / np.maximum(np.asarray(proportions["train"]), 1e-12)
        ).tolist()
        test_positive = [value for value in counts["test"] if value > 0]
        test_imbalance = max(test_positive) / min(test_positive)
        mean, std = channel_statistics(bundle.images["train"])
        total_loaded = sum(manifest.loaded_split_sizes.values())
        total_official = sum(manifest.official_split_sizes.values())
        fallback_risks = [
            "The official test set comes from a different clinical center."
        ]
        if imbalance > 1.5:
            fallback_risks.append("Class imbalance can hide minority-class errors.")
        if manifest.subsampled:
            fallback_risks.append(
                "Quick mode profiles a stratified subset, not the full dataset."
            )
        fallback = AmbiguityDecision(
            ambiguity_note=(
                f"PathMNIST is a nine-class histology task with train imbalance "
                f"ratio {imbalance:.2f}; tissue-boundary and inter-center shift "
                "remain the main ambiguity risks."
            ),
            risks=fallback_risks,
        )
        decision = self.reasoner.decide(
            stage="profiling.ambiguity",
            system=(
                "You are an independent medical-image data auditor. Do not make "
                "clinical claims. Identify concrete ambiguity and dataset-shift risks."
            ),
            user=(
                f"Dataset=PathMNIST; labels={list(bundle.labels)}; class_counts={counts}; "
                f"class_proportions={proportions}; "
                f"test_to_train_prevalence_ratio={prevalence_ratio}; "
                f"train_imbalance={imbalance:.4f}; loaded_sizes="
                f"{manifest.loaded_split_sizes}; official_sizes="
                f"{manifest.official_split_sizes}; test_domain="
                f"{manifest.test_domain_note}"
            ),
            response_model=AmbiguityDecision,
            fallback=fallback,
            audit=bb.record_event,
        )
        shape = bundle.images["train"].shape[1:]
        if len(shape) == 2:
            shape = (*shape, 1)
        profile = DataProfile(
            dataset="pathmnist",
            n_channels=3,
            n_classes=9,
            samples_profiled=total_loaded,
            profiled_fraction=1.0,
            official_data_fraction=round(total_loaded / total_official, 8),
            class_counts=counts,
            class_proportions={
                split: [round(float(value), 8) for value in values]
                for split, values in proportions.items()
            },
            test_to_train_prevalence_ratio=[
                round(float(value), 8) for value in prevalence_ratio
            ],
            imbalance_ratio=round(float(imbalance), 6),
            test_imbalance_ratio=round(float(test_imbalance), 6),
            image_shape_hwc=tuple(int(x) for x in shape),
            train_channel_mean_unit=[round(float(x), 8) for x in mean],
            train_channel_std_unit=[round(float(x), 8) for x in std],
            blank_image_fraction=round(blank_fraction(bundle.images["train"]), 8),
            duplicate_train_samples=count_duplicate_images(bundle.images["train"]),
            ambiguity_note=decision.value.ambiguity_note,
            ambiguity_risks=decision.value.risks,
            decision_source=decision.source,
        )
        bb.put("data_profile", profile, producer=self.name)


class PreprocessingAgent:
    """Choose a safe representation, then apply it lazily and deterministically."""

    name = "preprocessing"

    def __init__(
        self,
        reasoner: Reasoner,
        *,
        forced_decision: RepresentationDecision | None = None,
        forced_source: str = "forced",
    ):
        self.reasoner = reasoner
        self.forced_decision = forced_decision
        self.forced_source = forced_source

    def run(self, bb: Blackboard) -> None:
        bundle = _bundle(bb)
        profile = _profile(bb)
        manifest = _manifest(bb)
        fallback = RepresentationDecision(
            normalization="standardize",
            augmentations=["hflip"],
            rationale=(
                "Use train-only per-channel standardization. Horizontal flips "
                "preserve PathMNIST tissue identity and provide one tracked "
                "enrichment scenario."
            ),
        )
        if self.forced_decision is None:
            decision = self.reasoner.decide(
                stage="preprocessing.representation",
                system=(
                    "You design the input representation for 28x28 H&E tissue "
                    "patch classification. Choose only from the schema guardrails. "
                    "The listed flips and 90-degree rotation are orientation-safe "
                    "for tissue-type labels; application will be train-only."
                ),
                user=(
                    f"classes=9; train_counts={profile.class_counts['train']}; "
                    f"imbalance={profile.imbalance_ratio}; channel_mean="
                    f"{profile.train_channel_mean_unit}; channel_std="
                    f"{profile.train_channel_std_unit}; design_evidence=train_only"
                ),
                response_model=RepresentationDecision,
                fallback=fallback,
                audit=bb.record_event,
            )
            chosen, source = decision.value, decision.source
        else:
            chosen, source = self.forced_decision, self.forced_source
            bb.record_event(
                "forced_decision",
                stage="preprocessing.representation",
                source=source,
                value=chosen.model_dump(mode="json"),
            )

        prepared = prepare_data(bundle, chosen)
        plan = RepresentationPlan(
            normalization=chosen.normalization,
            augmentations=list(prepared.augmentations),
            train_channel_mean_unit=[round(x, 8) for x in prepared.mean],
            train_channel_std_unit=[round(x, 8) for x in prepared.std],
            augmentation_factor=1 + len(prepared.augmentations),
            rationale=chosen.rationale,
            source=source,
        )
        split = SplitManifest(
            dataset="pathmnist",
            seed=self.reasoner.seed,
            train_size=manifest.loaded_split_sizes["train"],
            val_size=manifest.loaded_split_sizes["val"],
            test_size=manifest.loaded_split_sizes["test"],
            augmented_train_size=prepared.augmented_train_size,
            selected_index_sha256=manifest.selected_index_sha256,
        )
        bb.put_blob("prepared_data", prepared, producer=self.name)
        bb.put("representation_plan", plan, producer=self.name)
        bb.put("split_manifest", split, producer=self.name)


class ExperimentDesignAgent:
    """Select a bounded training configuration from profile + representation."""

    name = "experiment_design"

    def __init__(
        self,
        reasoner: Reasoner,
        *,
        seed: int = 42,
        device: str = "cpu",
        max_epochs: int = 8,
    ):
        if max_epochs < 1 or max_epochs > 50:
            raise ValueError("max_epochs must be between 1 and 50")
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        self.reasoner = reasoner
        self.seed = seed
        self.device = device
        self.max_epochs = max_epochs

    def run(self, bb: Blackboard) -> None:
        profile = _profile(bb)
        representation = _representation(bb)
        fallback = ExperimentDecision(
            model_family="tiny_cnn",
            lr=1e-3,
            epochs=min(5 if profile.imbalance_ratio > 1.5 else 4, self.max_epochs),
            hidden=32 if representation.augmentations else 16,
            batch_size=128,
            weight_decay=1e-4,
            class_weighting=profile.imbalance_ratio > 1.5,
            optimizer="adamw",
            scheduler="cosine",
            label_smoothing=0.0,
            rationale=(
                "Bounded tiny-CNN configuration based on class balance and the "
                f"representation factor {representation.augmentation_factor}."
            ),
        )
        decision = self.reasoner.decide(
            stage="experiment_design.configuration",
            system=(
                "Configure a deliberately small CNN. Values outside the JSON "
                "Schema are invalid. Prefer stable, low-cost settings and explain "
                "how the data profile and representation informed the choice."
            ),
            user=(
                f"PathMNIST classes=9; profiled_samples={profile.samples_profiled}; "
                f"imbalance={profile.imbalance_ratio}; normalization="
                f"{representation.normalization}; augmentations="
                f"{representation.augmentations}; maximum_epochs={self.max_epochs}"
            ),
            response_model=ExperimentDecision,
            fallback=fallback,
            audit=bb.record_event,
        )
        selected = decision.value
        epochs = min(selected.epochs, self.max_epochs)
        if epochs != selected.epochs:
            bb.record_event(
                "guardrail_applied",
                stage="experiment_design.configuration",
                field="epochs",
                proposed=selected.epochs,
                accepted=epochs,
            )
        config = TrainConfig(
            model_family=selected.model_family,
            lr=selected.lr,
            epochs=epochs,
            hidden=selected.hidden,
            depth=selected.depth,
            dropout=selected.dropout,
            batch_size=selected.batch_size,
            weight_decay=selected.weight_decay,
            class_weighting=selected.class_weighting,
            optimizer=selected.optimizer,
            scheduler=selected.scheduler,
            label_smoothing=selected.label_smoothing,
            seed=self.seed,
            device=self.device,
            rationale=selected.rationale,
            source=decision.source,
        )
        bb.put("train_config", config, producer=self.name)


class ModelSearchAgent:
    """Iteratively search bounded models using validation evidence only."""

    name = "model_search"

    def __init__(
        self,
        reasoner: Reasoner,
        *,
        seed: int = 42,
        device: str = "cpu",
        max_trials: int = 6,
        rounds: int = 2,
        search_epochs: int = 3,
        final_epochs: int = 15,
        accuracy_tolerance: float = 0.005,
        data_root: str | None = None,
        train_limit: int | None = None,
        val_limit: int | None = None,
        test_limit: int | None = None,
        download: bool = True,
        trainer: Callable[..., Any] = train_model,
        predictor: Callable[..., Any] = predict_probabilities,
    ):
        if not 1 <= max_trials <= 24:
            raise ValueError("max_trials must be between 1 and 24")
        if not 1 <= rounds <= 4:
            raise ValueError("rounds must be between 1 and 4")
        if max_trials < rounds:
            raise ValueError("max_trials must be at least the number of rounds")
        if not 1 <= search_epochs <= 12:
            raise ValueError("search_epochs must be between 1 and 12")
        if not search_epochs <= final_epochs <= 50:
            raise ValueError("final_epochs must be between search_epochs and 50")
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        self.reasoner = reasoner
        self.seed = seed
        self.device = device
        self.max_trials = max_trials
        self.rounds = rounds
        self.search_epochs = search_epochs
        self.final_epochs = final_epochs
        self.accuracy_tolerance = accuracy_tolerance
        self.data_root = data_root
        self.train_limit = train_limit
        self.val_limit = val_limit
        self.test_limit = test_limit
        self.download = download
        self.trainer = trainer
        self.predictor = predictor

    def run(self, bb: Blackboard) -> None:
        bundle = _bundle(bb)
        profile = _profile(bb)
        manifest = _manifest(bb)
        plan = SearchPlan(
            max_trials=self.max_trials,
            rounds=self.rounds,
            search_epochs=self.search_epochs,
            final_epochs=self.final_epochs,
            accuracy_tolerance=self.accuracy_tolerance,
        )
        bb.put("search_plan", plan, producer=self.name)

        trials: list[TrialResult] = []
        proposals_by_hash: dict[str, CandidateProposal] = {}
        seen: set[str] = set()
        trial_names: list[str] = []
        for round_index in range(1, self.rounds + 1):
            remaining = self.max_trials - len(trials)
            if remaining <= 0:
                break
            rounds_left = self.rounds - round_index + 1
            round_cap = min(4, max(1, (remaining + rounds_left - 1) // rounds_left))
            fallback_candidates = default_candidates(round_index)[:round_cap]
            fallback = SearchDecision(
                analysis=(
                    "Deterministic diverse candidate set; selection uses validation "
                    "accuracy and macro-F1, never test metrics."
                ),
                candidates=fallback_candidates,
                stop=False,
            )
            prior = [
                {
                    "candidate_id": trial.candidate_id,
                    "config_hash": trial.config_hash,
                    "configuration": trial.config.model_dump(
                        mode="json",
                        exclude={
                            "schema_version",
                            "epochs",
                            "early_stopping_patience",
                            "gradient_clip_val",
                            "seed",
                            "device",
                            "rationale",
                            "source",
                        },
                    ),
                    "representation": trial.representation.model_dump(
                        mode="json", exclude={"rationale"}
                    ),
                    "validation_accuracy": trial.validation_accuracy,
                    "validation_macro_f1": trial.validation_macro_f1,
                    "validation_balanced_accuracy": trial.validation_balanced_accuracy,
                    "best_epoch": trial.best_epoch,
                    "status": trial.status,
                    "error": trial.error,
                }
                for trial in trials
            ]
            decision = self.reasoner.decide(
                stage=f"model_search.round_{round_index}",
                system=(
                    "You are a bounded ML search agent for PathMNIST. Propose "
                    "diverse configurations only from the JSON schema. Use only "
                    "training metadata and prior validation results. The test set "
                    "is locked and must never influence a proposal. Avoid duplicate "
                    "configurations and excessive weight decay."
                ),
                user=(
                    f"round={round_index}/{self.rounds}; candidate_budget={round_cap}; "
                    f"train_samples={manifest.loaded_split_sizes['train']}; "
                    f"validation_samples={manifest.loaded_split_sizes['val']}; "
                    f"train_counts={profile.class_counts['train']}; "
                    f"train_imbalance={profile.imbalance_ratio}; "
                    f"channel_mean={profile.train_channel_mean_unit}; "
                    f"channel_std={profile.train_channel_std_unit}; "
                    f"prior_validation_results={json.dumps(prior, sort_keys=True)}"
                ),
                response_model=SearchDecision,
                fallback=fallback,
                audit=bb.record_event,
            )
            candidate_entries = [
                (candidate, decision.source) for candidate in decision.value.candidates
            ]
            candidate_entries.extend(
                (candidate, "heuristic:portfolio") for candidate in fallback_candidates
            )
            accepted = 0
            for candidate, candidate_source in candidate_entries:
                if accepted >= round_cap or len(trials) >= self.max_trials:
                    break
                digest = candidate_hash(candidate)
                if digest in seen:
                    bb.record_event(
                        "search_candidate_skipped",
                        round_index=round_index,
                        candidate_id=candidate.candidate_id,
                        reason="duplicate_configuration",
                        config_hash=digest,
                    )
                    continue
                seen.add(digest)
                proposals_by_hash[digest] = candidate
                accepted += 1
                # All candidates receive the same budget, so metrics from later
                # agentic rounds remain directly comparable with earlier trials.
                trial_epochs = self.search_epochs
                config = proposal_to_config(
                    candidate,
                    seed=self.seed,
                    device=self.device,
                    epochs=trial_epochs,
                    source=candidate_source,
                )
                representation = proposal_to_representation(candidate)
                prepared = prepare_data(bundle, representation)
                artefact_name = safe_trial_name(candidate.candidate_id, digest)
                checkpoint = bb.blob_dir / "search" / f"{artefact_name}.ckpt"
                started = time.monotonic()
                bb.record_event(
                    "search_trial_started",
                    round_index=round_index,
                    candidate_id=candidate.candidate_id,
                    config_hash=digest,
                    epochs=trial_epochs,
                )
                try:
                    output = self.trainer(prepared, config, checkpoint)
                    probabilities, targets = self.predictor(
                        output.model,
                        prepared,
                        "val",
                        config.batch_size,
                        device=config.device,
                        seed=config.seed,
                    )
                    evaluation = evaluate_probabilities(
                        probabilities, targets, bundle.labels, split="val"
                    )
                    actual_checkpoint = Path(output.result.checkpoint_path)
                    checkpoint_relative = _relative_to_run(actual_checkpoint, bb.root)
                    trial = TrialResult(
                        candidate_id=candidate.candidate_id,
                        config_hash=digest,
                        round_index=round_index,
                        status="completed",
                        config=config,
                        representation=representation,
                        validation_accuracy=evaluation.accuracy,
                        validation_macro_f1=evaluation.macro_f1,
                        validation_balanced_accuracy=evaluation.balanced_accuracy,
                        best_epoch=output.result.best_epoch,
                        checkpoint_path=checkpoint_relative,
                        checkpoint_sha256=output.result.checkpoint_sha256,
                        training_log_path=(
                            _relative_to_run(
                                Path(output.result.training_log_path), bb.root
                            )
                            if output.result.training_log_path is not None
                            else None
                        ),
                        lightning_csv_path=(
                            _relative_to_run(
                                Path(output.result.lightning_csv_path), bb.root
                            )
                            if output.result.lightning_csv_path is not None
                            else None
                        ),
                        duration_seconds=round(time.monotonic() - started, 4),
                        decision_source=candidate_source,
                    )
                # A failed candidate must be recorded without aborting the
                # remaining independent trials; the final guard rejects an
                # entirely failed search.
                except Exception as exc:  # noqa: BLE001
                    trial = TrialResult(
                        candidate_id=candidate.candidate_id,
                        config_hash=digest,
                        round_index=round_index,
                        status="failed",
                        config=config,
                        representation=representation,
                        duration_seconds=round(time.monotonic() - started, 4),
                        decision_source=candidate_source,
                        error=f"{type(exc).__name__}: {str(exc)[:400]}",
                    )
                bb.put(artefact_name, trial, producer=self.name)
                trial_names.append(artefact_name)
                trials.append(trial)
                bb.record_event(
                    "search_trial_finished",
                    round_index=round_index,
                    candidate_id=candidate.candidate_id,
                    config_hash=digest,
                    status=trial.status,
                    validation_accuracy=trial.validation_accuracy,
                    validation_macro_f1=trial.validation_macro_f1,
                )
            minimum_trials = min(self.max_trials, max(3, self.rounds))
            if (
                decision.value.stop
                and sum(t.status == "completed" for t in trials) >= minimum_trials
                and round_index >= min(2, self.rounds)
            ):
                bb.record_event(
                    "search_stop_requested",
                    round_index=round_index,
                    source=decision.source,
                )
                break
            if decision.value.stop:
                bb.record_event(
                    "search_stop_rejected",
                    round_index=round_index,
                    reason=("adaptive_round_or_minimum_validation_trials_not_reached"),
                    minimum_trials=minimum_trials,
                )

        selected = rank_trials(trials, accuracy_tolerance=self.accuracy_tolerance)
        candidate = proposals_by_hash[selected.config_hash]
        final_config = selected.config.model_copy(
            update={
                "epochs": self.final_epochs,
                "early_stopping_patience": min(5, max(0, self.final_epochs // 4)),
                "source": f"agentic_search:selected:{selected.decision_source}",
                "rationale": (
                    f"Selected from {sum(t.status == 'completed' for t in trials)} "
                    "validation-only trials. " + candidate.rationale
                ),
            }
        )
        representation = proposal_to_representation(candidate)
        prepared = prepare_data(bundle, representation)
        lightning_payload = lightning_config_payload(
            final_config,
            representation,
            data_root=self.data_root,
            train_limit=self.train_limit,
            val_limit=self.val_limit,
            test_limit=self.test_limit,
            download=self.download,
            class_weights=(
                inverse_frequency_class_weights(
                    bundle.targets["train"], bundle.n_classes
                )
                if final_config.class_weighting
                else None
            ),
        )
        config_path = bb.root / "best_config.yaml"
        config_path.write_text(
            json.dumps(lightning_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        config_sha = sha256_file(config_path)
        plan_artifact = RepresentationPlan(
            normalization=representation.normalization,
            augmentations=list(prepared.augmentations),
            train_channel_mean_unit=[round(x, 8) for x in prepared.mean],
            train_channel_std_unit=[round(x, 8) for x in prepared.std],
            augmentation_factor=1 + len(prepared.augmentations),
            rationale=representation.rationale,
            source=final_config.source,
        )
        split = SplitManifest(
            dataset="pathmnist",
            seed=self.seed,
            train_size=manifest.loaded_split_sizes["train"],
            val_size=manifest.loaded_split_sizes["val"],
            test_size=manifest.loaded_split_sizes["test"],
            augmented_train_size=prepared.augmented_train_size,
            selected_index_sha256=manifest.selected_index_sha256,
        )
        report = SearchReport(
            trial_artefacts=trial_names,
            completed_trials=sum(trial.status == "completed" for trial in trials),
            failed_trials=sum(trial.status == "failed" for trial in trials),
            selected_candidate_id=selected.candidate_id,
            selected_config_hash=selected.config_hash,
            selected_validation_accuracy=float(selected.validation_accuracy),
            selected_validation_macro_f1=float(selected.validation_macro_f1),
        )
        best = BestConfiguration(
            selected_candidate_id=selected.candidate_id,
            selected_config_hash=selected.config_hash,
            train_config=final_config,
            representation=representation,
            validation_accuracy=float(selected.validation_accuracy),
            validation_macro_f1=float(selected.validation_macro_f1),
            selection_rule="accuracy within tolerance, then macro-F1; test locked",
            lightning_config_path=str(config_path.relative_to(bb.root)),
            lightning_config_sha256=config_sha,
        )
        bb.put_blob("prepared_data", prepared, producer=self.name)
        bb.put("representation_plan", plan_artifact, producer=self.name)
        bb.put("split_manifest", split, producer=self.name)
        bb.put("train_config", final_config, producer=self.name)
        bb.put("search_report", report, producer=self.name)
        bb.put("best_configuration", best, producer=self.name)


class FrozenConfigurationAgent:
    """Replay the validation-selected task configuration on another seed."""

    name = "configuration_replay"

    def __init__(
        self,
        selected: BestConfiguration,
        *,
        seed: int,
        device: str,
        data_root: str | None = None,
        train_limit: int | None = None,
        val_limit: int | None = None,
        test_limit: int | None = None,
        download: bool = True,
    ):
        self.selected = selected
        self.seed = seed
        self.device = device
        self.data_root = data_root
        self.train_limit = train_limit
        self.val_limit = val_limit
        self.test_limit = test_limit
        self.download = download

    def run(self, bb: Blackboard) -> None:
        bundle = _bundle(bb)
        manifest = _manifest(bb)
        config = self.selected.train_config.model_copy(
            update={
                "seed": self.seed,
                "device": self.device,
                "source": (
                    "configuration_replay:" + self.selected.selected_config_hash[:12]
                ),
            }
        )
        representation = self.selected.representation
        prepared = prepare_data(bundle, representation)
        config_path = bb.root / "best_config.yaml"
        payload = lightning_config_payload(
            config,
            representation,
            data_root=self.data_root,
            train_limit=self.train_limit,
            val_limit=self.val_limit,
            test_limit=self.test_limit,
            download=self.download,
            class_weights=(
                inverse_frequency_class_weights(
                    bundle.targets["train"], bundle.n_classes
                )
                if config.class_weighting
                else None
            ),
        )
        config_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        best = self.selected.model_copy(
            update={
                "train_config": config,
                "lightning_config_path": str(config_path.relative_to(bb.root)),
                "lightning_config_sha256": sha256_file(config_path),
                "selection_rule": (
                    self.selected.selection_rule
                    + "; frozen and replayed without test-dependent adaptation"
                ),
            }
        )
        plan = RepresentationPlan(
            normalization=representation.normalization,
            augmentations=list(prepared.augmentations),
            train_channel_mean_unit=[round(x, 8) for x in prepared.mean],
            train_channel_std_unit=[round(x, 8) for x in prepared.std],
            augmentation_factor=1 + len(prepared.augmentations),
            rationale=representation.rationale,
            source=config.source,
        )
        split = SplitManifest(
            dataset="pathmnist",
            seed=self.seed,
            train_size=manifest.loaded_split_sizes["train"],
            val_size=manifest.loaded_split_sizes["val"],
            test_size=manifest.loaded_split_sizes["test"],
            augmented_train_size=prepared.augmented_train_size,
            selected_index_sha256=manifest.selected_index_sha256,
        )
        bb.record_event(
            "configuration_replayed",
            selected_candidate_id=best.selected_candidate_id,
            selected_config_hash=best.selected_config_hash,
            original_validation_accuracy=best.validation_accuracy,
            test_metrics_used=False,
        )
        bb.put_blob("prepared_data", prepared, producer=self.name)
        bb.put("representation_plan", plan, producer=self.name)
        bb.put("split_manifest", split, producer=self.name)
        bb.put("train_config", config, producer=self.name)
        bb.put("best_configuration", best, producer=self.name)


class TrainingAgent:
    """Retrain the selected model through Lightning and persist its checkpoint."""

    name = "training"

    def run(self, bb: Blackboard) -> None:
        prepared = bb.get_blob("prepared_data")
        config = _train_config(bb)
        version = 1 + sum(item["artefact"] == "train_result" for item in bb.registry)
        checkpoint = bb.blob_dir / f"agentic_model_v{version:03d}.ckpt"
        output = train_model(prepared, config, checkpoint)
        actual_checkpoint = Path(output.result.checkpoint_path)
        relative_result = output.result.model_copy(
            update={
                "checkpoint_path": _relative_to_run(actual_checkpoint, bb.root),
                "training_log_path": (
                    _relative_to_run(Path(output.result.training_log_path), bb.root)
                    if output.result.training_log_path is not None
                    else None
                ),
                "lightning_csv_path": (
                    _relative_to_run(Path(output.result.lightning_csv_path), bb.root)
                    if output.result.lightning_csv_path is not None
                    else None
                ),
                "lightning_config_path": (
                    str(bb.get("best_configuration").lightning_config_path)
                    if isinstance(
                        bb.get_optional("best_configuration"), BestConfiguration
                    )
                    else None
                ),
            }
        )
        bb.put_blob("model", output.model, producer=self.name)
        bb.put("train_result", relative_result, producer=self.name)


class EvaluationAgent:
    """Evaluate once on the official test split with multiclass metrics."""

    name = "evaluation"

    def run(self, bb: Blackboard) -> None:
        prepared = bb.get_blob("prepared_data")
        model = bb.get_blob("model")
        config = _train_config(bb)
        val_probs, val_logits, val_targets = predict_outputs(
            model,
            prepared,
            "val",
            config.batch_size,
            device=config.device,
            seed=config.seed,
        )
        test_probs, test_logits, test_targets = predict_outputs(
            model,
            prepared,
            "test",
            config.batch_size,
            device=config.device,
            seed=config.seed,
        )
        report = evaluate_probabilities(
            test_probs, test_targets, prepared.bundle.labels, split="test"
        )
        version = 1 + sum(
            item["artefact"] == "prediction_artifact" for item in bb.registry
        )
        prediction_path = bb.blob_dir / f"predictions_v{version:03d}.npz"
        prediction_sha = save_prediction_arrays(
            prediction_path,
            validation_probabilities=val_probs,
            validation_logits=val_logits,
            validation_targets=val_targets,
            test_probabilities=test_probs,
            test_logits=test_logits,
            test_targets=test_targets,
            test_predictions=test_probs.argmax(axis=1),
            test_confidence=test_probs.max(axis=1),
        )
        bb.put_blob(
            "probabilities",
            {
                "val": val_probs,
                "val_logits": val_logits,
                "val_targets": val_targets,
                "test": test_probs,
                "test_logits": test_logits,
                "test_targets": test_targets,
            },
            producer=self.name,
        )
        bb.put(
            "prediction_artifact",
            PredictionArtifact(
                path=str(prediction_path.relative_to(bb.root)),
                sha256=prediction_sha,
                validation_samples=len(val_targets),
                test_samples=len(test_targets),
            ),
            producer=self.name,
        )
        bb.put("evaluation_report", report, producer=self.name)


class AbstentionOODAgent:
    """Calibrate selective prediction and validate OOD scores without test tuning."""

    name = "abstention"

    def __init__(
        self,
        *,
        target_validation_coverage: float = 0.80,
        corruption_sigma: float = 0.25,
        corruption_sigmas: tuple[float, ...] | None = None,
    ):
        if not 0.0 < target_validation_coverage <= 1.0:
            raise ValueError("target_validation_coverage must be in (0, 1]")
        if corruption_sigma <= 0:
            raise ValueError("corruption_sigma must be positive")
        self.target_validation_coverage = target_validation_coverage
        sigmas = corruption_sigmas or (0.15, corruption_sigma, 0.40)
        if not sigmas or any(value <= 0 for value in sigmas):
            raise ValueError("all corruption sigmas must be positive")
        self.corruption_sigmas = tuple(dict.fromkeys(float(value) for value in sigmas))

    def run(self, bb: Blackboard) -> None:
        import numpy as np

        probabilities = bb.get_blob("probabilities")
        prepared = bb.get_blob("prepared_data")
        model = bb.get_blob("model")
        config = _train_config(bb)
        scenarios: list[OODScenario] = []
        method_validation: dict[str, list[float]] = {
            "max_softmax": [],
            "predictive_entropy": [],
        }
        method_test: dict[str, list[float]] = {
            "max_softmax": [],
            "predictive_entropy": [],
        }
        ood_arrays: dict[str, Any] = {}
        for sigma in self.corruption_sigmas:
            val_ood, _, _ = predict_outputs(
                model,
                prepared,
                "val",
                config.batch_size,
                device=config.device,
                seed=config.seed,
                corruption_sigma=sigma,
            )
            test_ood, _, _ = predict_outputs(
                model,
                prepared,
                "test",
                config.batch_size,
                device=config.device,
                seed=config.seed,
                corruption_sigma=sigma,
            )
            key = str(sigma).replace(".", "_")
            ood_arrays[f"validation_gaussian_{key}"] = val_ood
            ood_arrays[f"test_gaussian_{key}"] = test_ood
            for method in ("max_softmax", "predictive_entropy"):
                val_auc = _ood_auc(probabilities["val"], val_ood, method)
                test_auc = _ood_auc(probabilities["test"], test_ood, method)
                if val_auc is not None:
                    method_validation[method].append(val_auc)
                if test_auc is not None:
                    method_test[method].append(test_auc)
                scenarios.append(
                    OODScenario(
                        corruption=f"unit_scale_gaussian_noise_sigma_{sigma}",
                        score=method,
                        validation_auroc=round(val_auc, 6)
                        if val_auc is not None
                        else None,
                        test_auroc=round(test_auc, 6) if test_auc is not None else None,
                    )
                )
        selected_method = max(
            method_validation,
            key=lambda name: (
                sum(method_validation[name]) / max(len(method_validation[name]), 1),
                name == "max_softmax",
            ),
        )
        val_confidence = 1.0 - _ood_score(probabilities["val"], selected_method)
        quantile = max(0.0, 1.0 - self.target_validation_coverage)
        threshold = float(np.quantile(val_confidence, quantile, method="lower"))
        validation_covered = val_confidence >= threshold

        test_probs = probabilities["test"]
        test_targets = probabilities["test_targets"]
        test_confidence = 1.0 - _ood_score(test_probs, selected_method)
        prediction = test_probs.argmax(axis=1)
        covered = test_confidence >= threshold
        base_accuracy = float((prediction == test_targets).mean())
        covered_accuracy = (
            float((prediction[covered] == test_targets[covered]).mean())
            if covered.any()
            else 0.0
        )

        validation_values = method_validation[selected_method]
        test_values = method_test[selected_method]
        validation_ood_auc = (
            sum(validation_values) / len(validation_values)
            if validation_values
            else None
        )
        ood_auc = sum(test_values) / len(test_values) if test_values else None
        ood_pass = bool(test_values and min(test_values) >= 0.5)
        ood_path = bb.blob_dir / "ood_probabilities.npz"
        ood_sha = save_prediction_arrays(ood_path, **ood_arrays)

        coverage = float(covered.mean())
        report = AbstentionReport(
            method=selected_method,
            threshold=round(threshold, 8),
            target_validation_coverage=self.target_validation_coverage,
            validation_coverage=round(float(validation_covered.mean()), 6),
            test_coverage=round(coverage, 6),
            abstain_rate=round(1.0 - coverage, 6),
            base_test_accuracy=round(base_accuracy, 6),
            accuracy_on_covered=round(covered_accuracy, 6),
            selective_risk=round(1.0 - covered_accuracy, 6),
            human_review_count=int((~covered).sum()),
            ood_scope="controlled_corruption_proxy",
            ood_corruption="unit_scale_gaussian_noise_multi_severity",
            ood_auroc=round(ood_auc, 6) if ood_auc is not None else None,
            ood_validation_auroc=(
                round(validation_ood_auc, 6) if validation_ood_auc is not None else None
            ),
            ood_scenarios=scenarios,
            ood_pass=ood_pass,
            ood_evidence_path=str(ood_path.relative_to(bb.root)),
            ood_evidence_sha256=ood_sha,
        )
        bb.put("abstention_report", report, producer=self.name)


class ReportingAgent:
    """Assemble a dossier; the orchestrator refreshes it after final review."""

    name = "reporting"

    def run(self, bb: Blackboard) -> None:
        if bb.get_optional("ablation_report") is not None:
            complete_through = "ablation"
        elif bb.get_optional("comparison_report") is not None:
            complete_through = "comparison"
        elif bb.get_optional("abstention_report") is not None:
            complete_through = "abstention"
        else:
            complete_through = "pipeline"
        path = bb.write_dossier(status="pipeline_reporting")
        bb.put(
            "reporting_status",
            ReportingStatus(
                dossier_path=str(path.relative_to(bb.root)),
                artefact_versions=len(bb.registry) + 1,
                decision_events=len(bb.log),
                complete_through_stage=complete_through,
            ),
            producer=self.name,
        )
        bb.write_dossier(status="pipeline_complete")


class ReviewerConsistencyAgent:
    """Independent deterministic audit plus schema-constrained Ollama review."""

    name = "reviewer"
    _RANK: ClassVar[dict[str, int]] = {"ok": 0, "warning": 1, "critical": 2}

    def __init__(self, reasoner: Reasoner):
        self.reasoner = reasoner

    def review(self, bb: Blackboard, stage: str, *, attempt: int = 1) -> AnomalyReport:
        findings = self._deterministic_findings(bb, stage)
        deterministic_severity = max(
            (level for level, _ in findings),
            key=self._RANK.get,
            default="ok",
        )
        deterministic_messages = [f"{level}:{message}" for level, message in findings]
        fallback = ReviewDecision(
            severity="ok",
            action="continue",
            issues=[],
            comment=(
                "Deterministic reviewer invariants passed."
                if not findings
                else "Deterministic findings require policy handling."
            ),
        )
        decision = self.reasoner.decide(
            stage=f"review.{stage}",
            system=(
                "You are an independent software and ML consistency reviewer. "
                "Inspect only the supplied evidence. You may raise severity but "
                "must never dismiss deterministic findings. No clinical claims. "
                "The supplied current UTC timestamp is authoritative."
            ),
            user=(
                f"current_utc={utc_now()}; stage={stage}; "
                f"deterministic_findings={deterministic_messages}; "
                f"stage_focused_evidence={bb.review_context(stage)}"
            ),
            response_model=ReviewDecision,
            fallback=fallback,
            audit=bb.record_event,
        )
        llm_severity = decision.value.severity
        severity = max((deterministic_severity, llm_severity), key=self._RANK.get)
        if decision.value.action == "stop":
            severity = "critical"
        elif decision.value.action == "revise" and severity == "ok":
            severity = "warning"
        action = (
            "stop"
            if severity == "critical"
            else ("revise" if severity == "warning" else "continue")
        )
        report = AnomalyReport(
            stage=stage,
            attempt=attempt,
            severity=severity,
            action=action,
            deterministic_issues=deterministic_messages,
            llm_issues=decision.value.issues,
            comment=decision.value.comment,
            source=decision.source,
        )
        bb.put(f"anomaly_{stage}", report, producer=self.name)
        return report

    def _deterministic_findings(
        self, bb: Blackboard, stage: str
    ) -> list[tuple[str, str]]:
        findings: list[tuple[str, str]] = []

        def require(name: str) -> Any | None:
            value = bb.get_optional(name)
            if value is None:
                findings.append(("critical", f"missing required artefact '{name}'"))
            return value

        if stage == "ingestion":
            manifest = require("dataset_manifest")
            if manifest is not None:
                if getattr(manifest, "dataset", None) != "pathmnist":
                    findings.append(("critical", "dataset is not PathMNIST"))
                if getattr(manifest, "n_channels", None) != 3:
                    findings.append(("critical", "PathMNIST must have three channels"))
                if getattr(manifest, "n_classes", None) != 9:
                    findings.append(("critical", "PathMNIST must have nine classes"))
                if any(v <= 0 for v in manifest.loaded_split_sizes.values()):
                    findings.append(("critical", "an official split is empty"))
        elif stage == "profiling":
            manifest = require("dataset_manifest")
            profile = require("data_profile")
            if manifest is not None and profile is not None:
                for split in ("train", "val", "test"):
                    if (
                        sum(profile.class_counts[split])
                        != manifest.loaded_split_sizes[split]
                    ):
                        findings.append(
                            ("critical", f"{split} class counts do not sum")
                        )
                profiled_samples = getattr(profile, "samples_profiled", None)
                if profiled_samples != sum(manifest.loaded_split_sizes.values()):
                    findings.append(
                        ("critical", "not every loaded sample was profiled")
                    )
                if (
                    getattr(profile, "profiled_fraction", 0.0) < 0.99
                    and not manifest.subsampled
                ):
                    findings.append(("critical", "less than 99% profiled in full mode"))
                if any(value <= 0 for value in profile.train_channel_std_unit):
                    findings.append(("critical", "zero-variance input channel"))
        elif stage == "preprocessing":
            manifest = require("dataset_manifest")
            representation = require("representation_plan")
            split = require("split_manifest")
            if manifest is not None and split is not None:
                if split.strategy != "official":
                    findings.append(("critical", "official splits were not preserved"))
                expected = manifest.loaded_split_sizes
                actual = {
                    "train": split.train_size,
                    "val": split.val_size,
                    "test": split.test_size,
                }
                if actual != expected:
                    findings.append(
                        ("critical", "split sizes changed during preprocessing")
                    )
                if not split.train_stats_only:
                    findings.append(
                        ("critical", "normalization stats are not train-only")
                    )
            if representation is not None:
                allowed = {"hflip", "vflip", "rotate90"}
                bad = set(representation.augmentations) - allowed
                if bad:
                    findings.append(
                        ("critical", f"unsafe augmentations: {sorted(bad)}")
                    )
                if representation.augmentation_factor != 1 + len(
                    representation.augmentations
                ):
                    findings.append(("critical", "augmentation factor is inconsistent"))
        elif stage == "experiment_design":
            config = require("train_config")
            if config is not None and config.source == "":
                findings.append(("critical", "training config has no provenance"))
        elif stage == "model_search":
            plan = require("search_plan")
            report = require("search_report")
            best = require("best_configuration")
            if plan is not None and not plan.test_locked:
                findings.append(("critical", "test split was not locked during search"))
            if report is not None:
                if report.test_metrics_used:
                    findings.append(
                        ("critical", "test metrics influenced model search")
                    )
                if report.completed_trials < 1:
                    findings.append(
                        ("critical", "model search has no successful trial")
                    )
            if report is not None and best is not None:
                if report.selected_config_hash != best.selected_config_hash:
                    findings.append(("critical", "selected search hashes disagree"))
                config_path = bb.root / best.lightning_config_path
                if not config_path.exists():
                    findings.append(("critical", "best Lightning config is missing"))
                elif sha256_file(config_path) != best.lightning_config_sha256:
                    findings.append(("critical", "best Lightning config hash mismatch"))
        elif stage == "configuration_replay":
            best = require("best_configuration")
            config = require("train_config")
            split = require("split_manifest")
            if best is not None:
                if best.test_metrics_used:
                    findings.append(("critical", "replayed config used test metrics"))
                config_path = bb.root / best.lightning_config_path
                if not config_path.exists():
                    findings.append(
                        ("critical", "replayed Lightning config is missing")
                    )
                elif sha256_file(config_path) != best.lightning_config_sha256:
                    findings.append(
                        ("critical", "replayed Lightning config hash mismatch")
                    )
            if config is not None and split is not None and config.seed != split.seed:
                findings.append(
                    ("critical", "replayed config and split seeds disagree")
                )
        elif stage == "training":
            config = require("train_config")
            result = require("train_result")
            if config is not None and result is not None:
                if result.seed != config.seed:
                    findings.append(("critical", "training seed differs from config"))
                if not 1 <= result.epochs_completed <= config.epochs:
                    findings.append(("critical", "training epoch count is invalid"))
                if getattr(result, "best_epoch", 1) > result.epochs_completed:
                    findings.append(("critical", "best epoch exceeds completed epochs"))
                checkpoint = Path(result.checkpoint_path)
                if not checkpoint.is_absolute():
                    checkpoint = bb.root / checkpoint
                if not checkpoint.exists():
                    findings.append(("critical", "checkpoint is missing"))
                elif sha256_file(checkpoint) != result.checkpoint_sha256:
                    findings.append(("critical", "checkpoint checksum mismatch"))
        elif stage == "evaluation":
            manifest = require("dataset_manifest")
            report = require("evaluation_report")
            if manifest is not None and report is not None:
                if report.n_samples != manifest.loaded_split_sizes["test"]:
                    findings.append(("critical", "test sample count mismatch"))
                matrix_total = sum(sum(row) for row in report.confusion_matrix)
                if matrix_total != report.n_samples:
                    findings.append(("critical", "confusion matrix count mismatch"))
                chance = 1.0 / manifest.n_classes
                if report.accuracy + 1e-6 < chance:
                    findings.append(("warning", "accuracy is below random chance"))
                prediction = require("prediction_artifact")
                if prediction is not None:
                    path = bb.root / prediction.path
                    if not path.exists():
                        findings.append(("critical", "prediction evidence is missing"))
                    elif sha256_file(path) != prediction.sha256:
                        findings.append(
                            ("critical", "prediction evidence hash mismatch")
                        )
                    if prediction.test_samples != report.n_samples:
                        findings.append(("critical", "prediction/test counts disagree"))
        elif stage == "abstention":
            report = require("abstention_report")
            if report is not None:
                if abs(report.test_coverage + report.abstain_rate - 1.0) > 1e-5:
                    findings.append(
                        ("critical", "coverage and abstention do not sum to one")
                    )
                if report.accuracy_on_covered < report.base_test_accuracy:
                    findings.append(
                        ("warning", "abstention did not lower covered-set error")
                    )
                if report.ood_scope != "controlled_corruption_proxy":
                    findings.append(
                        ("critical", "OOD scope is not explicitly qualified")
                    )
                ood_auroc = getattr(report, "ood_auroc", None)
                if ood_auroc is None:
                    findings.append(("warning", "OOD AUROC is undefined"))
                elif ood_auroc < 0.5:
                    findings.append(("warning", "OOD AUROC is below random chance"))
                if getattr(report, "ood_evidence_path", None):
                    path = bb.root / report.ood_evidence_path
                    if not path.exists():
                        findings.append(("critical", "OOD evidence is missing"))
                    elif sha256_file(path) != getattr(
                        report, "ood_evidence_sha256", None
                    ):
                        findings.append(("critical", "OOD evidence hash mismatch"))
        elif stage == "reporting":
            status = require("reporting_status")
            if status is not None and not (bb.root / status.dossier_path).exists():
                findings.append(("critical", "dossier file is missing"))
        elif stage == "comparison":
            comparison = require("comparison_report")
            baseline = require("baseline_report")
            if comparison is not None and not comparison.baseline_met_or_exceeded:
                findings.append(("warning", "agentic primary metric is below baseline"))
            if baseline is not None:
                checkpoint = bb.root / baseline.checkpoint_path
                if not baseline.checkpoint_sha256:
                    findings.append(("critical", "baseline checkpoint has no checksum"))
                elif (
                    not checkpoint.exists()
                    or sha256_file(checkpoint) != baseline.checkpoint_sha256
                ):
                    findings.append(
                        ("critical", "baseline checkpoint integrity failed")
                    )
        elif stage == "ablation":
            report = require("ablation_report")
            if report is not None and len(report.scenarios) < 3:
                findings.append(("critical", "fewer than three ablation scenarios"))
            if report is not None:
                for scenario in report.scenarios:
                    if not scenario.checkpoint_path or not scenario.checkpoint_sha256:
                        findings.append(
                            (
                                "critical",
                                f"ablation {scenario.name} lacks checkpoint provenance",
                            )
                        )
                        continue
                    checkpoint = bb.root / scenario.checkpoint_path
                    if (
                        not checkpoint.exists()
                        or sha256_file(checkpoint) != scenario.checkpoint_sha256
                    ):
                        findings.append(
                            (
                                "critical",
                                f"ablation {scenario.name} checkpoint integrity failed",
                            )
                        )
        return findings


def _bundle(bb: Blackboard) -> DatasetBundle:
    value = bb.get_blob("raw_dataset")
    if not isinstance(value, DatasetBundle):
        raise TypeError("raw_dataset blob does not satisfy DatasetBundle")
    return value


def _manifest(bb: Blackboard) -> DatasetManifest:
    value = bb.get("dataset_manifest")
    if not isinstance(value, DatasetManifest):
        raise TypeError("dataset_manifest has the wrong contract")
    return value


def _profile(bb: Blackboard) -> DataProfile:
    value = bb.get("data_profile")
    if not isinstance(value, DataProfile):
        raise TypeError("data_profile has the wrong contract")
    return value


def _representation(bb: Blackboard) -> RepresentationPlan:
    value = bb.get("representation_plan")
    if not isinstance(value, RepresentationPlan):
        raise TypeError("representation_plan has the wrong contract")
    return value


def _train_config(bb: Blackboard) -> TrainConfig:
    value = bb.get("train_config")
    if not isinstance(value, TrainConfig):
        raise TypeError("train_config has the wrong contract")
    return value


def _relative_to_run(path: Path, run_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(run_root.resolve()))
    except ValueError:
        return str(path)


def _ood_score(probabilities: Any, method: str) -> Any:
    import numpy as np

    values = np.asarray(probabilities, dtype="float64")
    if method == "max_softmax":
        return 1.0 - values.max(axis=1)
    if method == "predictive_entropy":
        clipped = np.clip(values, 1e-12, 1.0)
        return -(clipped * np.log(clipped)).sum(axis=1) / math.log(values.shape[1])
    raise ValueError(f"unknown OOD score: {method}")


def _ood_auc(
    clean_probabilities: Any, ood_probabilities: Any, method: str
) -> float | None:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    clean = _ood_score(clean_probabilities, method)
    ood = _ood_score(ood_probabilities, method)
    labels = np.concatenate([np.zeros(len(clean)), np.ones(len(ood))])
    try:
        return float(roc_auc_score(labels, np.concatenate([clean, ood])))
    except ValueError:
        return None

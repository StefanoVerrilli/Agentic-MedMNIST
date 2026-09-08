"""Cohesive agents for the PathMNIST hybrid training workflow."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol

from contracts import (
    AbstentionReport,
    AmbiguityDecision,
    AnomalyReport,
    Blackboard,
    DataProfile,
    DatasetManifest,
    EvalReport,
    ExperimentDecision,
    RepresentationDecision,
    RepresentationPlan,
    ReportingStatus,
    ReviewDecision,
    SplitManifest,
    TrainConfig,
    TrainResult,
    sha256_file,
)
from llm import Reasoner
from ml import (
    DatasetBundle,
    blank_fraction,
    channel_statistics,
    count_duplicate_images,
    load_pathmnist,
    predict_probabilities,
    prepare_data,
    train_tiny_cnn,
    evaluate_probabilities,
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
            split: int(len(bundle.targets[split]))
            for split in ("train", "val", "test")
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
                loaded_sizes[key] < bundle.official_sizes[key]
                for key in loaded_sizes
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
            split: np.bincount(
                bundle.targets[split], minlength=bundle.n_classes
            ).astype(int).tolist()
            for split in ("train", "val", "test")
        }
        positive = [value for value in counts["train"] if value > 0]
        imbalance = max(positive) / min(positive)
        mean, std = channel_statistics(bundle.images["train"])
        total_loaded = sum(manifest.loaded_split_sizes.values())
        total_official = sum(manifest.official_split_sizes.values())
        fallback_risks = [
            "The official test set comes from a different clinical center."
        ]
        if imbalance > 1.5:
            fallback_risks.append("Class imbalance can hide minority-class errors.")
        if manifest.subsampled:
            fallback_risks.append("Quick mode profiles a stratified subset, not the full dataset.")
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
            imbalance_ratio=round(float(imbalance), 6),
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
                    f"{profile.train_channel_std_unit}; ambiguity="
                    f"{profile.ambiguity_note}"
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
        if max_epochs < 1 or max_epochs > 20:
            raise ValueError("max_epochs must be between 1 and 20")
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
            lr=1e-3,
            epochs=min(5 if profile.imbalance_ratio > 1.5 else 4, self.max_epochs),
            hidden=32 if representation.augmentations else 16,
            batch_size=128,
            weight_decay=1e-4,
            class_weighting=profile.imbalance_ratio > 1.5,
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
            lr=selected.lr,
            epochs=epochs,
            hidden=selected.hidden,
            batch_size=selected.batch_size,
            weight_decay=selected.weight_decay,
            class_weighting=selected.class_weighting,
            seed=self.seed,
            device=self.device,
            rationale=selected.rationale,
            source=decision.source,
        )
        bb.put("train_config", config, producer=self.name)


class TrainingAgent:
    """Train the tiny CNN deterministically and persist its best checkpoint."""

    name = "training"

    def run(self, bb: Blackboard) -> None:
        prepared = bb.get_blob("prepared_data")
        config = _train_config(bb)
        version = 1 + sum(
            item["artefact"] == "train_result" for item in bb.registry
        )
        checkpoint = bb.blob_dir / f"agentic_model_v{version:03d}.pt"
        output = train_tiny_cnn(prepared, config, checkpoint)
        relative_result = output.result.model_copy(
            update={"checkpoint_path": str(checkpoint.relative_to(bb.root))}
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
        val_probs, val_targets = predict_probabilities(
            model,
            prepared,
            "val",
            config.batch_size,
            device=config.device,
            seed=config.seed,
        )
        test_probs, test_targets = predict_probabilities(
            model,
            prepared,
            "test",
            config.batch_size,
            device=config.device,
            seed=config.seed,
        )
        report = evaluate_probabilities(test_probs, test_targets, prepared.bundle.labels)
        bb.put_blob(
            "probabilities",
            {
                "val": val_probs,
                "val_targets": val_targets,
                "test": test_probs,
                "test_targets": test_targets,
            },
            producer=self.name,
        )
        bb.put("evaluation_report", report, producer=self.name)


class AbstentionOODAgent:
    """Calibrate abstention on validation and test a controlled OOD proxy."""

    name = "abstention"

    def __init__(
        self,
        *,
        target_validation_coverage: float = 0.80,
        corruption_sigma: float = 0.25,
    ):
        if not 0.0 < target_validation_coverage <= 1.0:
            raise ValueError("target_validation_coverage must be in (0, 1]")
        if corruption_sigma <= 0:
            raise ValueError("corruption_sigma must be positive")
        self.target_validation_coverage = target_validation_coverage
        self.corruption_sigma = corruption_sigma

    def run(self, bb: Blackboard) -> None:
        import numpy as np
        from sklearn.metrics import roc_auc_score

        probabilities = bb.get_blob("probabilities")
        val_confidence = probabilities["val"].max(axis=1)
        quantile = max(0.0, 1.0 - self.target_validation_coverage)
        threshold = float(np.quantile(val_confidence, quantile, method="lower"))
        validation_covered = val_confidence >= threshold

        test_probs = probabilities["test"]
        test_targets = probabilities["test_targets"]
        test_confidence = test_probs.max(axis=1)
        prediction = test_probs.argmax(axis=1)
        covered = test_confidence >= threshold
        base_accuracy = float((prediction == test_targets).mean())
        covered_accuracy = (
            float((prediction[covered] == test_targets[covered]).mean())
            if covered.any()
            else 0.0
        )

        prepared = bb.get_blob("prepared_data")
        model = bb.get_blob("model")
        config = _train_config(bb)
        ood_probs, _ = predict_probabilities(
            model,
            prepared,
            "test",
            config.batch_size,
            device=config.device,
            seed=config.seed,
            corruption_sigma=self.corruption_sigma,
        )
        ood_score = np.concatenate(
            [1.0 - test_confidence, 1.0 - ood_probs.max(axis=1)]
        )
        ood_labels = np.concatenate(
            [np.zeros(len(test_confidence)), np.ones(len(ood_probs))]
        )
        try:
            ood_auc = float(roc_auc_score(ood_labels, ood_score))
        except ValueError:
            ood_auc = None

        coverage = float(covered.mean())
        report = AbstentionReport(
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
            ood_corruption=f"unit_scale_gaussian_noise_sigma_{self.corruption_sigma}",
            ood_auroc=round(ood_auc, 6) if ood_auc is not None else None,
        )
        bb.put("abstention_report", report, producer=self.name)


class ReportingAgent:
    """Assemble a dossier; the orchestrator refreshes it after final review."""

    name = "reporting"

    def run(self, bb: Blackboard) -> None:
        path = bb.write_dossier(status="pipeline_reporting")
        bb.put(
            "reporting_status",
            ReportingStatus(
                dossier_path=str(path.relative_to(bb.root)),
                artefact_versions=len(bb.registry) + 1,
                decision_events=len(bb.log),
                complete_through_stage="abstention",
            ),
            producer=self.name,
        )
        bb.write_dossier(status="pipeline_complete")


class ReviewerConsistencyAgent:
    """Independent deterministic audit plus schema-constrained Ollama review."""

    name = "reviewer"
    _RANK = {"ok": 0, "warning": 1, "critical": 2}

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
                "must never dismiss deterministic findings. No clinical claims."
            ),
            user=(
                f"stage={stage}; deterministic_findings={deterministic_messages}; "
                f"latest_artefacts={bb.artefact_summary()}"
            ),
            response_model=ReviewDecision,
            fallback=fallback,
            audit=bb.record_event,
        )
        llm_severity = decision.value.severity
        severity = max(
            (deterministic_severity, llm_severity), key=self._RANK.get
        )
        if decision.value.action == "stop":
            severity = "critical"
        elif decision.value.action == "revise" and severity == "ok":
            severity = "warning"
        action = "stop" if severity == "critical" else (
            "revise" if severity == "warning" else "continue"
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
                    if sum(profile.class_counts[split]) != manifest.loaded_split_sizes[split]:
                        findings.append(("critical", f"{split} class counts do not sum"))
                profiled_samples = getattr(profile, "samples_profiled", None)
                if profiled_samples != sum(manifest.loaded_split_sizes.values()):
                    findings.append(("critical", "not every loaded sample was profiled"))
                if getattr(profile, "profiled_fraction", 0.0) < 0.99 and not manifest.subsampled:
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
                    findings.append(("critical", "split sizes changed during preprocessing"))
                if not split.train_stats_only:
                    findings.append(("critical", "normalization stats are not train-only"))
            if representation is not None:
                allowed = {"hflip", "vflip", "rotate90"}
                bad = set(representation.augmentations) - allowed
                if bad:
                    findings.append(("critical", f"unsafe augmentations: {sorted(bad)}"))
                if representation.augmentation_factor != 1 + len(
                    representation.augmentations
                ):
                    findings.append(("critical", "augmentation factor is inconsistent"))
        elif stage == "experiment_design":
            config = require("train_config")
            if config is not None and config.source == "":
                findings.append(("critical", "training config has no provenance"))
        elif stage == "training":
            config = require("train_config")
            result = require("train_result")
            if config is not None and result is not None:
                if result.seed != config.seed:
                    findings.append(("critical", "training seed differs from config"))
                if result.epochs_completed != config.epochs:
                    findings.append(("critical", "training history is incomplete"))
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
                if report.accuracy < chance:
                    findings.append(("warning", "accuracy is below random chance"))
        elif stage == "abstention":
            report = require("abstention_report")
            if report is not None:
                if abs(report.test_coverage + report.abstain_rate - 1.0) > 1e-5:
                    findings.append(("critical", "coverage and abstention do not sum to one"))
                if report.accuracy_on_covered < report.base_test_accuracy:
                    findings.append(("warning", "abstention did not lower covered-set error"))
                if report.ood_scope != "controlled_corruption_proxy":
                    findings.append(("critical", "OOD scope is not explicitly qualified"))
        elif stage == "reporting":
            status = require("reporting_status")
            if status is not None and not (bb.root / status.dossier_path).exists():
                findings.append(("critical", "dossier file is missing"))
        elif stage == "comparison":
            comparison = require("comparison_report")
            if comparison is not None and not comparison.baseline_met_or_exceeded:
                findings.append(("warning", "agentic primary metric is below baseline"))
        elif stage == "ablation":
            report = require("ablation_report")
            if report is not None and len(report.scenarios) < 3:
                findings.append(("critical", "fewer than three ablation scenarios"))
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

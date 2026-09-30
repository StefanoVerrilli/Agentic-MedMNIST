"""Typed artefact contracts and the append-only Blackboard.

Agents never call one another. Their only integration surface is a validated
Pydantic artefact written to the Blackboard. Every version is immutable,
checksummed and represented in the persistent decision log.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Severity = Literal["ok", "warning", "critical"]
Action = Literal["continue", "revise", "stop"]
Augmentation = Literal[
    "hflip", "vflip", "rotate90", "rotate180", "brightness", "contrast"
]
ModelFamily = Literal[
    "tiny_cnn",
    "residual_cnn",
    "resnet18",
    "vision_transformer",
    "compact_transformer",
]
OptimizerName = Literal["adam", "adamw", "sgd"]
SchedulerName = Literal["none", "cosine", "one_cycle", "reduce_on_plateau"]
PoolingName = Literal["cls", "mean", "attention"]
PositionalEncodingName = Literal["learned", "sinusoidal"]


def _validate_model_options(model: Any) -> None:
    if model.model_family in {"vision_transformer", "compact_transformer"}:
        if model.hidden % model.num_heads != 0:
            raise ValueError("hidden must be divisible by num_heads")
    if model.model_family == "resnet18" and model.hidden > 64:
        raise ValueError("resnet18 hidden cannot exceed 64")
    if model.model_family == "residual_cnn" and model.hidden > 96:
        raise ValueError("residual_cnn hidden cannot exceed 96")


def _canonicalize_model_options(value: Any) -> Any:
    """Reset architecture-irrelevant knobs instead of rejecting a proposal."""
    if not isinstance(value, dict):
        return value
    data = dict(value)
    family = data.get("model_family", "tiny_cnn")
    defaults = {
        "patch_size": 4,
        "num_heads": 4,
        "mlp_ratio": 4,
        "pooling": "cls",
        "positional_encoding": "learned",
        "tokenizer_layers": 2,
    }
    if family not in {"vision_transformer", "compact_transformer"}:
        data.update(defaults)
    elif family == "compact_transformer":
        data["patch_size"] = 4
    else:
        data["tokenizer_layers"] = 2
    return data


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StrictModel(BaseModel):
    """Base for LLM responses and nested records: reject unknown fields."""

    model_config = ConfigDict(
        extra="forbid", str_strip_whitespace=True, allow_inf_nan=False
    )


class Artefact(StrictModel):
    """Base for persisted agent hand-offs."""

    schema_version: str = "1.0"


# --- Strict schemas for the four hybrid judgement points --------------------


class AmbiguityDecision(StrictModel):
    ambiguity_note: str = Field(min_length=3, max_length=6000)
    risks: list[str] = Field(default_factory=list, max_length=8)


class RepresentationDecision(StrictModel):
    normalization: Literal["unit", "standardize"]
    augmentations: list[Augmentation] = Field(default_factory=list, max_length=3)
    rationale: str = Field(min_length=3, max_length=6000)


class ExperimentDecision(StrictModel):
    model_family: ModelFamily = "tiny_cnn"
    lr: float = Field(ge=1e-5, le=3e-2)
    epochs: int = Field(ge=1, le=50)
    hidden: int = Field(ge=16, le=192, multiple_of=8)
    depth: int = Field(default=2, ge=1, le=8)
    dropout: float = Field(default=0.0, ge=0.0, le=0.6)
    batch_size: Literal[16, 32, 64, 128, 256]
    weight_decay: float = Field(ge=0.0, le=0.1)
    class_weighting: bool
    optimizer: OptimizerName = "adamw"
    scheduler: SchedulerName = "none"
    label_smoothing: float = Field(default=0.0, ge=0.0, le=0.2)
    patch_size: Literal[2, 4, 7] = 4
    num_heads: Literal[2, 3, 4, 6, 8] = 4
    mlp_ratio: Literal[2, 3, 4] = 4
    pooling: PoolingName = "cls"
    positional_encoding: PositionalEncodingName = "learned"
    tokenizer_layers: int = Field(default=2, ge=1, le=3)
    early_stopping_patience: int = Field(default=3, ge=0, le=10)
    gradient_clip_val: float = Field(default=1.0, ge=0.0, le=5.0)
    rationale: str = Field(min_length=3, max_length=6000)

    @model_validator(mode="before")
    @classmethod
    def canonicalize_model_options(cls, value: Any) -> Any:
        return _canonicalize_model_options(value)

    @model_validator(mode="after")
    def valid_model_options(self) -> ExperimentDecision:
        _validate_model_options(self)
        return self


class CandidateProposal(StrictModel):
    """One bounded model/representation hypothesis proposed by the LLM."""

    candidate_id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,31}$")
    model_family: ModelFamily
    hidden: int = Field(ge=16, le=192, multiple_of=8)
    depth: int = Field(ge=1, le=8)
    dropout: float = Field(ge=0.0, le=0.6)
    normalization: Literal["unit", "standardize"]
    augmentations: list[Augmentation] = Field(default_factory=list, max_length=3)
    optimizer: OptimizerName
    scheduler: SchedulerName
    lr: float = Field(ge=1e-5, le=3e-2)
    weight_decay: float = Field(ge=0.0, le=0.1)
    class_weighting: bool
    label_smoothing: float = Field(ge=0.0, le=0.2)
    batch_size: Literal[16, 32, 64, 128, 256]
    patch_size: Literal[2, 4, 7] = 4
    num_heads: Literal[2, 3, 4, 6, 8] = 4
    mlp_ratio: Literal[2, 3, 4] = 4
    pooling: PoolingName = "cls"
    positional_encoding: PositionalEncodingName = "learned"
    tokenizer_layers: int = Field(default=2, ge=1, le=3)
    early_stopping_patience: int = Field(default=2, ge=0, le=10)
    gradient_clip_val: float = Field(default=1.0, ge=0.0, le=5.0)
    rationale: str = Field(min_length=3, max_length=6000)

    @model_validator(mode="before")
    @classmethod
    def canonicalize_model_options(cls, value: Any) -> Any:
        return _canonicalize_model_options(value)

    @model_validator(mode="after")
    def unique_augmentations(self) -> CandidateProposal:
        if len(set(self.augmentations)) != len(self.augmentations):
            raise ValueError("augmentations must be unique")
        _validate_model_options(self)
        return self


class SearchDecision(StrictModel):
    """Schema-constrained proposal for one search round."""

    analysis: str = Field(min_length=3, max_length=12000)
    candidates: list[CandidateProposal] = Field(min_length=1, max_length=4)
    stop: bool = False


class ArchitectureResearchDecision(StrictModel):
    """Cross-cutting guidance spanning representation, tuning and architecture."""

    analysis: str = Field(min_length=3, max_length=12000)
    representation_priorities: list[str] = Field(min_length=1, max_length=8)
    parametrization_priorities: list[str] = Field(min_length=1, max_length=8)
    architecture_priorities: list[ModelFamily] = Field(min_length=1, max_length=5)
    transformer_guidance: str = Field(min_length=3, max_length=6000)
    risks: list[str] = Field(default_factory=list, max_length=8)


class ReviewDecision(StrictModel):
    severity: Severity
    action: Action
    issues: list[str] = Field(default_factory=list, max_length=30)
    comment: str = Field(min_length=2, max_length=12000)


# --- Artefacts exchanged between agents ------------------------------------


class RunManifest(Artefact):
    run_id: str
    dataset: Literal["pathmnist"]
    seed: int = Field(ge=0)
    created_at: str
    python_version: str
    platform: str
    package_versions: dict[str, str]
    llm_mode: Literal["ollama", "heuristic"]
    llm_model: str
    llm_model_digest: str | None = None
    code_sha256: str | None = None
    lightning_cli: bool = True
    deterministic_torch: bool = True


class DatasetManifest(Artefact):
    dataset: Literal["pathmnist"]
    task: Literal["multi-class"]
    n_channels: Literal[3]
    n_classes: Literal[9]
    labels: list[str] = Field(min_length=9, max_length=9)
    official_split_sizes: dict[str, int]
    loaded_split_sizes: dict[str, int]
    selected_index_sha256: dict[str, str]
    data_reference: str
    archive_md5: str = Field(min_length=32, max_length=32)
    license: str
    test_domain_note: str
    subsampled: bool

    @model_validator(mode="after")
    def validate_splits(self) -> DatasetManifest:
        expected = {"train", "val", "test"}
        if set(self.official_split_sizes) != expected:
            raise ValueError("official_split_sizes must contain train, val and test")
        if set(self.loaded_split_sizes) != expected:
            raise ValueError("loaded_split_sizes must contain train, val and test")
        if set(self.selected_index_sha256) != expected:
            raise ValueError("selected_index_sha256 must contain train, val and test")
        for split in expected:
            loaded = self.loaded_split_sizes[split]
            official = self.official_split_sizes[split]
            if loaded <= 0 or loaded > official:
                raise ValueError(f"invalid loaded size for {split}: {loaded}")
        return self


class DataProfile(Artefact):
    dataset: Literal["pathmnist"]
    n_channels: Literal[3]
    n_classes: Literal[9]
    samples_profiled: int = Field(gt=0)
    profiled_fraction: float = Field(gt=0.0, le=1.0)
    official_data_fraction: float = Field(gt=0.0, le=1.0)
    class_counts: dict[str, list[int]]
    class_proportions: dict[str, list[float]] = Field(default_factory=dict)
    test_to_train_prevalence_ratio: list[float] = Field(default_factory=list)
    imbalance_ratio: float = Field(ge=1.0)
    test_imbalance_ratio: float = Field(default=1.0, ge=1.0)
    image_shape_hwc: tuple[int, int, int]
    train_channel_mean_unit: list[float] = Field(min_length=3, max_length=3)
    train_channel_std_unit: list[float] = Field(min_length=3, max_length=3)
    blank_image_fraction: float = Field(ge=0.0, le=1.0)
    duplicate_train_samples: int = Field(ge=0)
    ambiguity_note: str
    ambiguity_risks: list[str]
    decision_source: str

    @model_validator(mode="after")
    def validate_counts(self) -> DataProfile:
        if set(self.class_counts) != {"train", "val", "test"}:
            raise ValueError("class_counts must contain train, val and test")
        if any(len(v) != self.n_classes for v in self.class_counts.values()):
            raise ValueError("every class-count vector must match n_classes")
        if self.class_proportions and (
            set(self.class_proportions) != {"train", "val", "test"}
            or any(len(v) != self.n_classes for v in self.class_proportions.values())
        ):
            raise ValueError("class_proportions must contain three n-class vectors")
        if (
            self.test_to_train_prevalence_ratio
            and len(self.test_to_train_prevalence_ratio) != self.n_classes
        ):
            raise ValueError("prevalence-ratio vector must match n_classes")
        return self


class RepresentationPlan(Artefact):
    normalization: Literal["unit", "standardize"]
    augmentations: list[Augmentation]
    stats_source: Literal["train"] = "train"
    train_channel_mean_unit: list[float] = Field(min_length=3, max_length=3)
    train_channel_std_unit: list[float] = Field(min_length=3, max_length=3)
    augmentation_factor: int = Field(ge=1, le=4)
    rationale: str
    source: str


class SplitManifest(Artefact):
    dataset: Literal["pathmnist"]
    strategy: Literal["official"] = "official"
    seed: int = Field(ge=0)
    train_size: int = Field(gt=0)
    val_size: int = Field(gt=0)
    test_size: int = Field(gt=0)
    augmented_train_size: int = Field(gt=0)
    selected_index_sha256: dict[str, str]
    train_stats_only: bool = True
    official_test_separate_center: bool = True


class TrainConfig(Artefact):
    model_family: ModelFamily = "tiny_cnn"
    lr: float = Field(ge=1e-5, le=3e-2)
    epochs: int = Field(ge=1, le=50)
    hidden: int = Field(ge=16, le=192, multiple_of=8)
    depth: int = Field(default=2, ge=1, le=8)
    dropout: float = Field(default=0.0, ge=0.0, le=0.6)
    batch_size: Literal[16, 32, 64, 128, 256]
    weight_decay: float = Field(ge=0.0, le=0.1)
    class_weighting: bool
    optimizer: OptimizerName = "adamw"
    scheduler: SchedulerName = "none"
    label_smoothing: float = Field(default=0.0, ge=0.0, le=0.2)
    patch_size: Literal[2, 4, 7] = 4
    num_heads: Literal[2, 3, 4, 6, 8] = 4
    mlp_ratio: Literal[2, 3, 4] = 4
    pooling: PoolingName = "cls"
    positional_encoding: PositionalEncodingName = "learned"
    tokenizer_layers: int = Field(default=2, ge=1, le=3)
    early_stopping_patience: int = Field(default=3, ge=0, le=10)
    gradient_clip_val: float = Field(default=1.0, ge=0.0, le=5.0)
    seed: int = Field(ge=0)
    device: Literal["cpu", "cuda"]
    rationale: str
    source: str

    @model_validator(mode="before")
    @classmethod
    def canonicalize_model_options(cls, value: Any) -> Any:
        return _canonicalize_model_options(value)

    @model_validator(mode="after")
    def valid_model_options(self) -> TrainConfig:
        _validate_model_options(self)
        return self


class EpochMetrics(StrictModel):
    epoch: int = Field(ge=1)
    train_loss: float = Field(ge=0.0)
    val_loss: float | None = Field(default=None, ge=0.0)
    val_accuracy: float = Field(ge=0.0, le=1.0)
    val_macro_f1: float | None = Field(default=None, ge=0.0, le=1.0)
    learning_rate: float | None = Field(default=None, ge=0.0)


class TrainResult(Artefact):
    final_train_loss: float = Field(ge=0.0)
    final_val_accuracy: float = Field(ge=0.0, le=1.0)
    best_val_accuracy: float = Field(ge=0.0, le=1.0)
    best_epoch: int = Field(ge=1)
    epochs_completed: int = Field(ge=1)
    history: list[EpochMetrics]
    checkpoint_path: str
    checkpoint_sha256: str = Field(min_length=64, max_length=64)
    training_log_path: str | None = None
    lightning_csv_path: str | None = None
    seed: int = Field(ge=0)
    device: Literal["cpu", "cuda"]
    framework: Literal["lightning.pytorch"] = "lightning.pytorch"
    lightning_config_path: str | None = None

    @model_validator(mode="after")
    def validate_history(self) -> TrainResult:
        if len(self.history) != self.epochs_completed:
            raise ValueError("history length must equal epochs_completed")
        expected_epochs = list(range(1, self.epochs_completed + 1))
        if [row.epoch for row in self.history] != expected_epochs:
            raise ValueError("history epochs must be contiguous and one-indexed")
        if self.best_epoch > self.epochs_completed:
            raise ValueError("best_epoch cannot exceed epochs_completed")
        return self


class SearchPlan(Artefact):
    max_trials: int = Field(ge=1, le=24)
    rounds: int = Field(ge=1, le=4)
    search_epochs: int = Field(ge=1, le=12)
    final_epochs: int = Field(ge=1, le=50)
    accuracy_tolerance: float = Field(default=0.005, ge=0.0, le=0.05)
    objective: Literal["validation_accuracy_then_macro_f1"] = (
        "validation_accuracy_then_macro_f1"
    )
    test_locked: Literal[True] = True
    framework: Literal["lightning.pytorch"] = "lightning.pytorch"

    @model_validator(mode="after")
    def validate_epoch_budgets(self) -> SearchPlan:
        if self.search_epochs > self.final_epochs:
            raise ValueError("search_epochs cannot exceed final_epochs")
        return self


class ArchitectureResearch(Artefact):
    """Auditable research brief consumed by design/search stages."""

    analysis: str
    representation_priorities: list[str] = Field(min_length=1, max_length=8)
    parametrization_priorities: list[str] = Field(min_length=1, max_length=8)
    architecture_priorities: list[ModelFamily] = Field(min_length=1, max_length=5)
    transformer_guidance: str
    risks: list[str] = Field(default_factory=list, max_length=8)
    evidence_sources: list[str] = Field(min_length=1, max_length=8)
    test_metrics_used: Literal[False] = False
    source: str

class TrialResult(Artefact):
    candidate_id: str
    config_hash: str = Field(min_length=64, max_length=64)
    round_index: int = Field(ge=1, le=4)
    status: Literal["completed", "failed"]
    config: TrainConfig
    representation: RepresentationDecision
    validation_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    validation_macro_f1: float | None = Field(default=None, ge=0.0, le=1.0)
    validation_balanced_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    best_epoch: int | None = Field(default=None, ge=1)
    checkpoint_path: str | None = None
    checkpoint_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    training_log_path: str | None = None
    lightning_csv_path: str | None = None
    duration_seconds: float = Field(ge=0.0)
    decision_source: str
    error: str | None = None

    @model_validator(mode="after")
    def completed_trial_has_evidence(self) -> TrialResult:
        if self.status == "completed":
            required = (
                self.validation_accuracy,
                self.validation_macro_f1,
                self.validation_balanced_accuracy,
                self.best_epoch,
                self.checkpoint_path,
                self.checkpoint_sha256,
            )
            if any(value is None for value in required):
                raise ValueError("completed trial is missing validation evidence")
        return self


class SearchReport(Artefact):
    trial_artefacts: list[str] = Field(min_length=1)
    completed_trials: int = Field(ge=1)
    failed_trials: int = Field(ge=0)
    selected_candidate_id: str
    selected_config_hash: str = Field(min_length=64, max_length=64)
    selected_validation_accuracy: float = Field(ge=0.0, le=1.0)
    selected_validation_macro_f1: float = Field(ge=0.0, le=1.0)
    selection_rule: Literal["accuracy_tolerance_then_macro_f1"] = (
        "accuracy_tolerance_then_macro_f1"
    )
    test_metrics_used: Literal[False] = False


class BestConfiguration(Artefact):
    selected_candidate_id: str
    selected_config_hash: str = Field(min_length=64, max_length=64)
    train_config: TrainConfig
    representation: RepresentationDecision
    validation_accuracy: float = Field(ge=0.0, le=1.0)
    validation_macro_f1: float = Field(ge=0.0, le=1.0)
    selection_rule: str
    test_metrics_used: Literal[False] = False
    lightning_config_path: str
    lightning_config_sha256: str = Field(min_length=64, max_length=64)


class ClassMetrics(StrictModel):
    label: int = Field(ge=0, le=8)
    name: str
    support: int = Field(ge=0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)


class EvalReport(Artefact):
    split: Literal["val", "test"] = "test"
    n_samples: int = Field(gt=0)
    accuracy: float = Field(ge=0.0, le=1.0)
    balanced_accuracy: float = Field(ge=0.0, le=1.0)
    macro_precision: float = Field(ge=0.0, le=1.0)
    macro_recall: float = Field(ge=0.0, le=1.0)
    macro_f1: float = Field(ge=0.0, le=1.0)
    roc_auc_ovr_macro: float | None = Field(default=None, ge=0.0, le=1.0)
    per_class: list[ClassMetrics] = Field(min_length=9, max_length=9)
    confusion_matrix: list[list[int]] = Field(min_length=9, max_length=9)
    primary_metric: Literal["accuracy"] = "accuracy"

    @model_validator(mode="after")
    def validate_multiclass_evidence(self) -> EvalReport:
        if any(len(row) != 9 for row in self.confusion_matrix):
            raise ValueError("confusion_matrix must be 9x9")
        if sum(sum(row) for row in self.confusion_matrix) != self.n_samples:
            raise ValueError("confusion_matrix must sum to n_samples")
        if [item.label for item in self.per_class] != list(range(9)):
            raise ValueError("per_class labels must be ordered from 0 through 8")
        if sum(item.support for item in self.per_class) != self.n_samples:
            raise ValueError("per-class support must sum to n_samples")
        return self


class PredictionArtifact(Artefact):
    path: str
    sha256: str = Field(min_length=64, max_length=64)
    validation_samples: int = Field(gt=0)
    test_samples: int = Field(gt=0)
    includes_probabilities: Literal[True] = True
    includes_logits: Literal[True] = True
    includes_targets: Literal[True] = True


class OODScenario(StrictModel):
    corruption: str = Field(min_length=3)
    score: Literal["max_softmax", "predictive_entropy"]
    validation_auroc: float | None = Field(default=None, ge=0.0, le=1.0)
    test_auroc: float | None = Field(default=None, ge=0.0, le=1.0)


class AbstentionReport(Artefact):
    method: Literal["max_softmax", "predictive_entropy"] = "max_softmax"
    threshold: float = Field(ge=0.0, le=1.0)
    calibrated_on: Literal["validation"] = "validation"
    target_validation_coverage: float = Field(gt=0.0, le=1.0)
    validation_coverage: float = Field(ge=0.0, le=1.0)
    test_coverage: float = Field(ge=0.0, le=1.0)
    abstain_rate: float = Field(ge=0.0, le=1.0)
    base_test_accuracy: float = Field(ge=0.0, le=1.0)
    accuracy_on_covered: float = Field(ge=0.0, le=1.0)
    selective_risk: float = Field(ge=0.0, le=1.0)
    human_review_count: int = Field(ge=0)
    ood_scope: Literal["controlled_corruption_proxy"]
    ood_corruption: str
    ood_auroc: float | None = Field(default=None, ge=0.0, le=1.0)
    ood_validation_auroc: float | None = Field(default=None, ge=0.0, le=1.0)
    ood_scenarios: list[OODScenario] = Field(default_factory=list)
    ood_pass: bool = False
    ood_evidence_path: str = Field(min_length=1)
    ood_evidence_sha256: str = Field(min_length=64, max_length=64)


class BaselineReport(Artefact):
    dataset: Literal["pathmnist"]
    split_fingerprint: str
    seed: int = Field(ge=0)
    config: TrainConfig
    accuracy: float = Field(ge=0.0, le=1.0)
    macro_f1: float = Field(ge=0.0, le=1.0)
    checkpoint_path: str
    checkpoint_sha256: str | None = None
    training_log_path: str | None = None
    lightning_csv_path: str | None = None


class ComparisonReport(Artefact):
    primary_metric: Literal["accuracy"] = "accuracy"
    agentic_accuracy: float = Field(ge=0.0, le=1.0)
    baseline_accuracy: float = Field(ge=0.0, le=1.0)
    accuracy_delta: float = Field(ge=-1.0, le=1.0)
    common_seed: int = Field(ge=0)
    common_split_fingerprint: str
    baseline_met_or_exceeded: bool


class AblationScenario(StrictModel):
    name: str
    normalization: Literal["unit", "standardize"]
    augmentations: list[Augmentation]
    seed: int = Field(ge=0)
    accuracy: float = Field(ge=0.0, le=1.0)
    macro_f1: float = Field(ge=0.0, le=1.0)
    checkpoint_path: str | None = None
    checkpoint_sha256: str | None = None
    training_log_path: str | None = None
    lightning_csv_path: str | None = None


class AblationReport(Artefact):
    scenarios: list[AblationScenario] = Field(min_length=3)
    common_split_fingerprint: str


class ReportingStatus(Artefact):
    dossier_path: str
    artefact_versions: int = Field(gt=0)
    decision_events: int = Field(gt=0)
    complete_through_stage: str


class AnomalyReport(Artefact):
    stage: str
    attempt: int = Field(ge=1)
    severity: Severity
    action: Action
    deterministic_issues: list[str] = Field(default_factory=list)
    llm_issues: list[str] = Field(default_factory=list)
    comment: str
    source: str


# --- Blackboard -------------------------------------------------------------


class Blackboard:
    """Append-only artefact store plus in-memory heavy-object exchange."""

    _VALID_NAME = re.compile(r"^[a-z][a-z0-9_]*$")

    def __init__(self, root: str | Path, run_id: str | None = None):
        self.root = Path(root)
        self.run_id = run_id or self.root.name
        self.artefact_dir = self.root / "artefacts"
        self.blob_dir = self.root / "blobs"
        if (self.root / "decision_log.jsonl").exists() or any(
            self.artefact_dir.glob("*.json")
        ):
            raise FileExistsError(
                f"run directory is not empty; refusing to overwrite: {self.root}"
            )
        self.artefact_dir.mkdir(parents=True, exist_ok=True)
        self.blob_dir.mkdir(parents=True, exist_ok=True)
        self.artefacts: dict[str, Artefact] = {}
        self.blobs: dict[str, Any] = {}
        self.registry: list[dict[str, Any]] = []
        self.log: list[dict[str, Any]] = []
        self._versions: dict[str, int] = {}
        self._artefact_sequence = 0
        self._event_sequence = 0
        self.decision_log_path = self.root / "decision_log.jsonl"

    def put(self, name: str, artefact: Artefact, *, producer: str) -> Path:
        self._check_name(name)
        if not isinstance(artefact, Artefact):
            raise TypeError("Blackboard artefacts must inherit from Artefact")

        self._artefact_sequence += 1
        version = self._versions.get(name, 0) + 1
        self._versions[name] = version
        payload = artefact.model_dump(mode="json")
        digest = _sha256_json(payload)
        filename = f"{self._artefact_sequence:04d}_{name}_v{version:03d}.json"
        path = self.artefact_dir / filename
        envelope = {
            "run_id": self.run_id,
            "artefact": name,
            "artefact_type": type(artefact).__name__,
            "version": version,
            "sequence": self._artefact_sequence,
            "created_at": utc_now(),
            "producer": producer,
            "payload_sha256": digest,
            "payload": payload,
        }
        _atomic_write_json(path, envelope)
        self.artefacts[name] = artefact
        entry = {
            "artefact": name,
            "type": type(artefact).__name__,
            "version": version,
            "path": str(path.relative_to(self.root)),
            "sha256": digest,
            "producer": producer,
        }
        self.registry.append(entry)
        self.record_event("artefact_written", **entry)
        return path

    def get(self, name: str) -> Artefact:
        return self.artefacts[name]

    def get_optional(self, name: str) -> Artefact | None:
        return self.artefacts.get(name)

    def put_blob(self, name: str, obj: Any, *, producer: str) -> None:
        self._check_name(name)
        self.blobs[name] = obj
        self.record_event(
            "blob_registered",
            blob=name,
            producer=producer,
            python_type=type(obj).__name__,
        )

    def get_blob(self, name: str) -> Any:
        return self.blobs[name]

    def record_event(self, event: str, **details: Any) -> dict[str, Any]:
        self._event_sequence += 1
        row = {
            "sequence": self._event_sequence,
            "timestamp": utc_now(),
            "run_id": self.run_id,
            "event": event,
            **_json_safe(details),
        }
        self.log.append(row)
        with self.decision_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return row

    def artefact_summary(self, max_chars: int = 12000) -> str:
        """Return valid, compact JSON with the newest artefacts first."""
        return self.review_context("general", max_chars=max_chars)

    def review_context(self, stage: str, *, max_chars: int = 12000) -> str:
        """Build stage-focused evidence without cutting JSON mid-artefact.

        The previous implementation truncated one large JSON string from the
        front. Once a run grew, that routinely hid the current-stage artefact
        from the reviewer. This method prioritises the evidence required for
        the current gate, compacts prose, and only appends complete artefacts.
        """
        focus_by_stage = {
            "ingestion": ["dataset_manifest", "run_manifest"],
            "profiling": ["data_profile", "dataset_manifest"],
            "preprocessing": ["representation_plan", "split_manifest", "data_profile"],
            "architecture_research": ["architecture_research", "data_profile"],
            "experiment_design": [
                "train_config",
                "representation_plan",
                "data_profile",
            ],
            "model_search": [
                "search_report", "best_configuration", "search_plan",
                "architecture_research",
            ],
            "configuration_replay": [
                "best_configuration",
                "train_config",
                "representation_plan",
                "split_manifest",
            ],
            "training": ["train_result", "train_config", "best_configuration"],
            "evaluation": ["evaluation_report", "prediction_artifact", "train_result"],
            "abstention": ["abstention_report", "evaluation_report"],
            "comparison": ["comparison_report", "baseline_report", "evaluation_report"],
            "ablation": ["ablation_report", "best_configuration"],
            "reporting": [
                "reporting_status",
                "comparison_report",
                "ablation_report",
                "abstention_report",
                "search_report",
            ],
        }
        preferred = focus_by_stage.get(stage, [])
        newest = list(reversed(self.artefacts))
        ordered = list(dict.fromkeys([*preferred, *newest]))
        selected: dict[str, Any] = {}
        omitted: list[str] = []
        for name in ordered:
            artefact = self.artefacts.get(name)
            if artefact is None:
                continue
            raw = (
                artefact.model_dump(mode="json")
                if isinstance(artefact, BaseModel)
                else _json_safe(
                    vars(artefact) if hasattr(artefact, "__dict__") else artefact
                )
            )
            selected[name] = _compact_for_review(raw)
            candidate = {
                "stage": stage,
                "artefacts": selected,
                "omitted_artefacts": omitted,
            }
            text = json.dumps(candidate, ensure_ascii=False, sort_keys=True)
            if len(text) > max_chars and len(selected) > 1:
                selected.pop(name)
                omitted.append(name)
        return json.dumps(
            {
                "stage": stage,
                "artefacts": selected,
                "omitted_artefacts": omitted,
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    def write_dossier(self, *, status: str) -> Path:
        dossier = {
            "run_id": self.run_id,
            "status": status,
            "updated_at": utc_now(),
            "latest_artefacts": {
                name: item.model_dump(mode="json")
                for name, item in self.artefacts.items()
            },
            "artefact_registry": self.registry,
            "decision_log": str(self.decision_log_path.name),
            "decision_events": len(self.log),
        }
        path = self.root / "dossier.json"
        _atomic_write_json(path, dossier)
        return path

    def _check_name(self, name: str) -> None:
        if not self._VALID_NAME.fullmatch(name):
            raise ValueError(f"invalid Blackboard name: {name!r}")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(data: Any) -> str:
    canonical = json.dumps(
        data, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _json_safe(data: Any) -> Any:
    return json.loads(json.dumps(data, ensure_ascii=False, default=str))


def _compact_for_review(data: Any) -> Any:
    """Bound verbose prose while preserving all structured numeric evidence."""
    if isinstance(data, str):
        return data if len(data) <= 400 else data[:397] + "..."
    if isinstance(data, list):
        return [_compact_for_review(item) for item in data[:20]]
    if isinstance(data, dict):
        return {str(key): _compact_for_review(value) for key, value in data.items()}
    return data

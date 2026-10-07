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
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Severity = Literal["ok", "warning", "critical"]
Action = Literal["continue", "revise", "stop"]
Augmentation = Annotated[str, Field(pattern=r"^(hflip|vflip|rotate90|rotate180|brightness|contrast|approved_[a-z][a-z0-9_]{2,31})$")]
ModelFamily = Literal[
    "tiny_cnn",
    "residual_cnn",
    "resnet18",
    "vision_transformer",
    "compact_transformer",
    "multi_scale_transformer",
    "feature_pyramid_transformer",
]
OptimizerName = Literal["adam", "adamw", "sgd"]
SchedulerName = Literal["none", "cosine", "one_cycle", "reduce_on_plateau"]
PoolingName = Literal["cls", "mean", "attention"]
PositionalEncodingName = Literal["learned", "sinusoidal"]


def _validate_model_options(model: Any) -> None:
    if model.optimizer == "sgd" and model.nesterov and model.momentum <= 0:
        raise ValueError("Nesterov requires positive momentum")
    if model.scheduler == "cosine" and model.cosine_eta_min > model.lr:
        raise ValueError("cosine_eta_min cannot exceed lr")
    if model.model_family == "tiny_cnn" and model.depth > 4:
        raise ValueError("tiny_cnn depth cannot exceed four pooling blocks for 28x28 input")
    if model.model_family in {"vision_transformer", "compact_transformer", "multi_scale_transformer"}:
        if model.hidden % model.num_heads != 0:
            raise ValueError("hidden must be divisible by num_heads")
    if model.model_family == "multi_scale_transformer":
        validate_scales(model.scales)
    if model.model_family == "resnet18" and model.hidden > 256:
        raise ValueError("resnet18 hidden cannot exceed 256")
    if model.model_family == "residual_cnn" and model.hidden > 512:
        raise ValueError("residual_cnn hidden cannot exceed 512")


def _canonicalize_model_options(value: Any) -> Any:
    """Reset architecture-irrelevant knobs instead of rejecting a proposal."""
    if not isinstance(value, dict):
        return value
    data = dict(value)
    family = data.get("model_family", "tiny_cnn")
    if family == "multi_scale_transformer":
        data.setdefault("pooling", "mean")
    defaults = {
        "patch_size": 4,
        "num_heads": 4,
        "mlp_ratio": 4,
        "pooling": "cls",
        "positional_encoding": "learned",
        "tokenizer_layers": 2,
    }
    if family != "multi_scale_transformer":
        data["scales"] = (2, 4, 7)
    elif "scales" in data:
        if not isinstance(data["scales"], (tuple, list)) or any(type(item) is not int for item in data["scales"]):
            raise ValueError("scales must be a list of integer patch sizes")
        data["scales"] = tuple(sorted(data["scales"]))
    if family not in {"vision_transformer", "compact_transformer", "multi_scale_transformer"}:
        data.update(defaults)
    elif family == "compact_transformer":
        data["patch_size"] = 4
    else:
        data["tokenizer_layers"] = 2
        if family == "multi_scale_transformer":
            data["patch_size"] = 4
    if family not in {"tiny_cnn", "residual_cnn"}:
        data["channel_cap"] = 256
    if family == "feature_pyramid_transformer":
        data.update(depth=1, pooling="mean")
    if data.get("optimizer", "adamw") == "sgd":
        data.update(adam_beta1=0.9, adam_beta2=0.999, optimizer_eps=1e-8)
    else:
        data.update(momentum=0.9, nesterov=True)
    for scheduler, values in {
        "one_cycle": {"one_cycle_pct_start": 0.1},
        "reduce_on_plateau": {"plateau_factor": 0.5, "plateau_patience": 1},
        "cosine": {"cosine_eta_min": 0.0},
    }.items():
        if data.get("scheduler", "none") != scheduler:
            data.update(values)
    if data.get("early_stopping_patience") == 0:
        data.update(early_stopping_monitor="val_accuracy", early_stopping_min_delta=0.0)
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
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True,
                              allow_inf_nan=False, frozen=True)


# --- Strict schemas for the four hybrid judgement points --------------------


class AmbiguityDecision(StrictModel):
    ambiguity_note: str = Field(min_length=3, max_length=6000)
    risks: list[str] = Field(default_factory=list, max_length=8)


class RepresentationDecision(StrictModel):
    normalization: Literal["unit", "standardize"]
    augmentations: list[Augmentation] = Field(default_factory=list, max_length=8)
    rationale: str = Field(min_length=3, max_length=6000)

    @model_validator(mode="after")
    def approved_augmentations_only(self) -> RepresentationDecision:
        from extensions import validate_representation
        validate_representation(self.augmentations)
        return self


def validate_scales(scales: tuple[int, ...]) -> None:
    if not 2 <= len(scales) <= 4 or len(set(scales)) != len(scales):
        raise ValueError("multi-scale models need two to four distinct scales")
    if any(scale not in {2, 4, 7, 14} for scale in scales):
        raise ValueError("scales must be selected from 2, 4, 7, 14")


class ExecutionOptions(StrictModel):
    """Shared tunable network/optimizer controls; defaults preserve old checkpoints."""
    channel_cap: int = Field(default=256, ge=8, le=4096, multiple_of=8)
    scales: tuple[Literal[2, 4, 7, 14], ...] = (2, 4, 7)
    momentum: float = Field(default=0.9, ge=0.0, lt=1.0)
    nesterov: bool = True
    adam_beta1: float = Field(default=0.9, ge=0.0, lt=1.0)
    adam_beta2: float = Field(default=0.999, ge=0.0, lt=1.0)
    optimizer_eps: float = Field(default=1e-8, ge=1e-12, le=1e-2)
    one_cycle_pct_start: float = Field(default=0.1, gt=0.0, lt=1.0)
    plateau_factor: float = Field(default=0.5, gt=0.0, lt=1.0)
    plateau_patience: int = Field(default=1, ge=0, le=500)
    cosine_eta_min: float = Field(default=0.0, ge=0.0, le=1.0)


def execution_options(config: Any) -> dict[str, Any]:
    return {name: getattr(config, name) for name in ExecutionOptions.model_fields}


class ExperimentDecision(ExecutionOptions):
    model_family: ModelFamily = "tiny_cnn"
    lr: float = Field(ge=1e-7, le=1.0)
    epochs: int = Field(ge=1, le=1000)
    hidden: int = Field(ge=8, le=1024, multiple_of=8)
    depth: int = Field(default=2, ge=1, le=48)
    dropout: float = Field(default=0.0, ge=0.0, le=0.95)
    batch_size: int = Field(ge=4, le=4096)
    weight_decay: float = Field(ge=0.0, le=1.0)
    class_weighting: bool
    optimizer: OptimizerName = "adamw"
    scheduler: SchedulerName = "none"
    label_smoothing: float = Field(default=0.0, ge=0.0, le=0.5)
    patch_size: Literal[1, 2, 4, 7, 14, 28] = 4
    num_heads: int = Field(default=4, ge=1, le=32)
    mlp_ratio: int = Field(default=4, ge=1, le=16)
    pooling: PoolingName = "cls"
    positional_encoding: PositionalEncodingName = "learned"
    tokenizer_layers: int = Field(default=2, ge=1, le=5)
    early_stopping_patience: int = Field(default=3, ge=0, le=500)
    early_stopping_monitor: Literal["val_accuracy", "val_macro_f1", "val_loss"] = "val_accuracy"
    early_stopping_min_delta: float = Field(default=0.0, ge=0.0, le=1.0)
    gradient_clip_val: float = Field(default=1.0, ge=0.0, le=100.0)
    rationale: str = Field(min_length=3, max_length=6000)

    @model_validator(mode="before")
    @classmethod
    def canonicalize_model_options(cls, value: Any) -> Any:
        return _canonicalize_model_options(value)

    @model_validator(mode="after")
    def valid_model_options(self) -> ExperimentDecision:
        _validate_model_options(self)
        return self


class CandidateProposal(ExecutionOptions):
    """One bounded model/representation hypothesis proposed by the LLM."""

    candidate_id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,31}$")
    model_family: ModelFamily
    epochs: int = Field(default=100, ge=1, le=1000)
    hidden: int = Field(ge=8, le=1024, multiple_of=8)
    depth: int = Field(ge=1, le=48)
    dropout: float = Field(ge=0.0, le=0.95)
    normalization: Literal["unit", "standardize"]
    augmentations: list[Augmentation] = Field(default_factory=list, max_length=8)
    optimizer: OptimizerName
    scheduler: SchedulerName
    lr: float = Field(ge=1e-7, le=1.0)
    weight_decay: float = Field(ge=0.0, le=1.0)
    class_weighting: bool
    label_smoothing: float = Field(ge=0.0, le=0.5)
    batch_size: int = Field(ge=4, le=4096)
    patch_size: Literal[1, 2, 4, 7, 14, 28] = 4
    num_heads: int = Field(default=4, ge=1, le=32)
    mlp_ratio: int = Field(default=4, ge=1, le=16)
    pooling: PoolingName = "cls"
    positional_encoding: PositionalEncodingName = "learned"
    tokenizer_layers: int = Field(default=2, ge=1, le=5)
    early_stopping_patience: int = Field(default=2, ge=0, le=500)
    early_stopping_monitor: Literal["val_accuracy", "val_macro_f1", "val_loss"] = "val_accuracy"
    early_stopping_min_delta: float = Field(default=0.0, ge=0.0, le=1.0)
    gradient_clip_val: float = Field(default=1.0, ge=0.0, le=100.0)
    rationale: str = Field(min_length=3, max_length=6000)

    @model_validator(mode="before")
    @classmethod
    def canonicalize_model_options(cls, value: Any) -> Any:
        return _canonicalize_model_options(value)

    @model_validator(mode="after")
    def unique_augmentations(self) -> CandidateProposal:
        from extensions import validate_representation
        validate_representation(self.augmentations)
        if len(set(self.augmentations)) != len(self.augmentations):
            raise ValueError("augmentations must be unique")
        _validate_model_options(self)
        return self


class SearchDecision(StrictModel):
    """Schema-constrained proposal for one search round."""

    analysis: str = Field(min_length=3, max_length=12000)
    candidates: list[CandidateProposal] = Field(min_length=1, max_length=16)
    stop: bool = False


class ArchitectureResearchDecision(StrictModel):
    """Cross-cutting guidance spanning representation, tuning and architecture."""

    analysis: str = Field(min_length=3, max_length=12000)
    representation_priorities: list[str] = Field(min_length=1, max_length=8)
    parametrization_priorities: list[str] = Field(min_length=1, max_length=8)
    architecture_priorities: list[ModelFamily] = Field(min_length=1, max_length=7)
    transformer_guidance: str = Field(min_length=3, max_length=6000)
    risks: list[str] = Field(default_factory=list, max_length=8)


class SourceRecord(StrictModel):
    source_id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,63}$")
    title: str = Field(min_length=3, max_length=1000)
    url: str = Field(pattern=r"^https://arxiv\.org/abs/[0-9.]+(v[0-9]+)?$")
    text: str = Field(min_length=10, max_length=20000)
    origin: Literal["arxiv_api", "curated_excerpt"]
    retrieved_at: str
    snapshot_path: str
    snapshot_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class CitedIdea(StrictModel):
    idea_id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,31}$")
    target: Literal["representation", "architecture", "training"]
    hypothesis: str = Field(min_length=3, max_length=2000)
    source_id: str
    evidence_quote: str = Field(min_length=10, max_length=600)
    # A quoted observation is evidence; its transfer to PathMNIST is a hypothesis.
    status: Literal["hypothesis_to_validate"] = "hypothesis_to_validate"


class ExtensionProposal(StrictModel):
    proposal_id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,31}$")
    kind: Literal["augmentation", "architecture"]
    description: str = Field(min_length=10, max_length=3000)
    source_ids: list[str] = Field(min_length=1, max_length=8)
    operations: list[Literal["hflip", "vflip", "rotate90", "rotate180"]] = Field(default_factory=list, max_length=3)
    label_preservation_rationale: str = Field(default="", max_length=2000)
    implementation_requirements: list[str] = Field(default_factory=list, max_length=10)


class LiteratureDecision(StrictModel):
    ideas: list[CitedIdea] = Field(min_length=1, max_length=12)
    proposals: list[ExtensionProposal] = Field(default_factory=list, max_length=4)


class PriorArtBrief(Artefact):
    query: str
    sources: list[SourceRecord] = Field(min_length=1, max_length=8)
    ideas: list[CitedIdea] = Field(min_length=1, max_length=12)
    proposals: list[ExtensionProposal] = Field(default_factory=list, max_length=4)
    source: str
    retrieval_mode: Literal["online", "cache", "bundled"]
    retrieval_errors: list[str] = Field(default_factory=list)
    test_metrics_used: Literal[False] = False


class ExtensionGateEntry(StrictModel):
    proposal_id: str
    spec_sha256: str
    status: Literal["pending_review", "approved", "reviewed_spec", "rejected"]
    reason: str
    registry_name: str | None = None
    reviewer: str | None = None


class ExtensionGateReport(Artefact):
    entries: list[ExtensionGateEntry] = Field(default_factory=list)
    approvals_sha256: str | None = None


class ApprovedExtension(StrictModel):
    proposal: ExtensionProposal
    spec_sha256: str
    reviewer: str = Field(min_length=2)
    rationale: str = Field(min_length=3)


class ExtensionRegistry(Artefact):
    entries: list[ApprovedExtension] = Field(default_factory=list)


class BlobReference(Artefact):
    kind: Literal["dataset", "prepared", "model", "arrays"]
    path: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    metadata: dict[str, Any] = Field(default_factory=dict)


class DataAuditReport(Artefact):
    passed: bool
    checks: dict[str, bool]
    issues: list[str]
    train_validation_overlap: int = Field(ge=0)
    test_inspected: Literal[False] = False


class HumanReviewQueue(Artefact):
    path: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    count: int = Field(ge=0)
    split: Literal["test"] = "test"
    purpose: Literal["benchmark_review"] = "benchmark_review"


class HumanReviewDecision(StrictModel):
    case_id: str
    reviewer: str = Field(min_length=2, max_length=200)
    decision: Literal["confirm", "correct", "defer"]
    label: int | None = Field(default=None, ge=0, le=8)
    comment: str = Field(min_length=3, max_length=2000)

    @model_validator(mode="after")
    def correction_requires_label(self) -> HumanReviewDecision:
        if (self.decision == "correct") != (self.label is not None):
            raise ValueError("only a correction must provide a label")
        return self


class HumanReviewResolution(Artefact):
    queue_sha256: str
    decisions: list[HumanReviewDecision]
    pending_count: int = Field(ge=0)
    created_at: str


class AcceptanceCriterion(StrictModel):
    requirement: str
    status: Literal["passed", "failed", "pending"]
    evidence: list[str]
    detail: str


class AcceptanceReport(Artefact):
    criteria: list[AcceptanceCriterion]
    kpis: dict[str, Any]
    technical_complete: bool
    acceptance_ready: bool
    trl7_evidence_complete: bool = False
    independent_signoff: str | None = None


class ExecutionState(Artefact):
    status: str
    stage: str | None = None


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
    code_hash_algorithm: Literal["legacy-native-v1", "canonical-text-v2"] = "legacy-native-v1"
    lightning_cli: bool = True
    deterministic_torch: bool = True


class RunConfiguration(Artefact):
    parameters: dict[str, Any]
    frozen_best: dict[str, Any] | None = None
    frozen_reference_scope: Literal["origin_seed", "local_snapshot"] = "origin_seed"
    extension_registry: list[ApprovedExtension] = Field(default_factory=list)


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
    verified_archive_md5: str | None = Field(default=None, min_length=32, max_length=32)
    archive_sha256: str | None = Field(default=None, min_length=64, max_length=64)
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
        if set(self.class_counts) != {"train", "val"}:
            raise ValueError("selection profile must contain train and val only")
        if any(len(v) != self.n_classes for v in self.class_counts.values()):
            raise ValueError("every class-count vector must match n_classes")
        if self.class_proportions and (
            set(self.class_proportions) != {"train", "val"}
            or any(len(v) != self.n_classes for v in self.class_proportions.values())
        ):
            raise ValueError("class_proportions must contain train and val vectors")
        if self.test_to_train_prevalence_ratio:
            raise ValueError("selection profile must not contain test prevalence")
        return self


class RepresentationPlan(Artefact):
    normalization: Literal["unit", "standardize"]
    augmentations: list[Augmentation]
    stats_source: Literal["train"] = "train"
    train_channel_mean_unit: list[float] = Field(min_length=3, max_length=3)
    train_channel_std_unit: list[float] = Field(min_length=3, max_length=3)
    augmentation_factor: int = Field(ge=1, le=9)
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


class GeneratedBundleReference(StrictModel):
    path: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    bundle_id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,31}$")
    version: int = Field(ge=1)
    parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def json_parameters(self) -> GeneratedBundleReference:
        try:
            json.dumps(self.parameters, allow_nan=False)
        except (ValueError, TypeError) as exc:
            raise ValueError("bundle parameters must contain finite JSON values") from exc
        return self


class WorkerConfiguration(StrictModel):
    """Local subprocess execution controls, using the running Python environment."""
    timeout_seconds: int = Field(default=3600, ge=1)
    max_bundles: int = Field(default=32, ge=1, le=1024)

    @model_validator(mode="before")
    @classmethod
    def read_previous_configuration(cls, value):
        if isinstance(value, dict):
            return {key: item for key, item in value.items()
                    if key not in {"host", "root", "image", "python", "memory_gib"}}
        return value


class GeneratedSource(StrictModel):
    path: str = Field(pattern=r"^[a-zA-Z_][a-zA-Z0-9_/]*\.py$")
    code: str = Field(min_length=1, max_length=120000)


class GeneratedCodeDecision(StrictModel):
    bundle_id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,31}$")
    hypothesis: str = Field(min_length=3, max_length=6000)
    files: list[GeneratedSource] = Field(min_length=1, max_length=16)
    parameters: dict[str, Any] = Field(default_factory=dict)
    epochs: int = Field(default=100, ge=1, le=1000)
    batch_size: int = Field(default=128, ge=4, le=4096)

    @model_validator(mode="after")
    def entrypoint_present(self) -> GeneratedCodeDecision:
        paths = [item.path for item in self.files]
        if "experiment.py" not in paths or len(set(paths)) != len(paths):
            raise ValueError("unique source paths including experiment.py are required")
        try:
            json.dumps(self.parameters, allow_nan=False)
        except (ValueError, TypeError) as exc:
            raise ValueError("experiment parameters must contain finite JSON values") from exc
        return self


class GeneratedBundle(Artefact):
    reference: GeneratedBundleReference
    hypothesis: str
    author: str
    source: str
    files: dict[str, str]
    parent_sha256: str | None = None


class AutonomousTrainingOptions(ExecutionOptions):
    lr: float = Field(ge=1e-7, le=1.0)
    weight_decay: float = Field(ge=0.0, le=1.0)
    class_weighting: bool
    optimizer: OptimizerName
    scheduler: SchedulerName
    label_smoothing: float = Field(ge=0.0, le=0.5)
    early_stopping_patience: int = Field(ge=0)
    early_stopping_monitor: Literal["val_accuracy", "val_macro_f1", "val_loss"]
    early_stopping_min_delta: float = Field(ge=0.0, le=1.0)
    gradient_clip_val: float = Field(ge=0.0, le=100.0)

    @model_validator(mode="after")
    def explicit_active_options(self):
        required = ({"momentum", "nesterov"} if self.optimizer == "sgd" else
                    {"adam_beta1", "adam_beta2", "optimizer_eps"})
        required |= {"none": set(), "cosine": {"cosine_eta_min"},
                     "one_cycle": {"one_cycle_pct_start"},
                     "reduce_on_plateau": {"plateau_factor", "plateau_patience"}}[self.scheduler]
        missing = required - self.model_fields_set
        if missing:
            raise ValueError(f"agent must explicitly choose active options: {sorted(missing)}")
        if self.optimizer == "sgd" and self.nesterov and self.momentum <= 0:
            raise ValueError("Nesterov requires positive momentum")
        if self.scheduler == "cosine" and self.cosine_eta_min > self.lr:
            raise ValueError("cosine_eta_min cannot exceed lr")
        return self


class AutonomousExperimentDecision(GeneratedCodeDecision):
    epochs: int = Field(ge=1, strict=True)
    batch_size: int = Field(ge=4, le=4096)
    training: AutonomousTrainingOptions


def _autonomous_action_schema(schema: dict[str, Any]) -> None:
    """Expose the same action-specific arguments to JSON generators and Python."""
    properties = schema.pop("properties")
    schema.pop("required", None)
    schema.pop("additionalProperties", None)
    branches = []
    for action, active in {
        "new_trial": {"experiment"},
        "continue_trial": {"candidate_id", "additional_epochs"},
        "finish_search": set(),
    }.items():
        fields = deepcopy(properties)
        fields["action"] = {"type": "string", "const": action}
        for name in ("experiment", "candidate_id", "additional_epochs"):
            fields[name] = (next(item for item in fields[name]["anyOf"] if item.get("type") != "null")
                            if name in active else {"type": "null", "default": None})
        branches.append({"type": "object", "properties": fields,
                         "required": ["action", "rationale", *sorted(active)],
                         "additionalProperties": False})
    schema["anyOf"] = branches


class AutonomousSearchDecision(StrictModel):
    model_config = ConfigDict(json_schema_extra=_autonomous_action_schema)

    action: Literal["new_trial", "continue_trial", "finish_search"]
    rationale: str = Field(min_length=3, max_length=6000)
    experiment: AutonomousExperimentDecision | None = None
    candidate_id: str | None = Field(default=None, min_length=1)
    additional_epochs: int | None = Field(default=None, ge=1, strict=True)

    @model_validator(mode="after")
    def action_arguments(self):
        issues = []
        if self.action == "new_trial":
            if self.experiment is None:
                issues.append("experiment must be a complete experiment object")
            inactive = ("candidate_id", "additional_epochs")
        elif self.action == "continue_trial":
            if not self.candidate_id:
                issues.append("candidate_id must identify an existing resumable trial")
            if self.additional_epochs is None:
                issues.append("additional_epochs must be a positive integer")
            inactive = ("experiment",)
        else:
            inactive = ("experiment", "candidate_id", "additional_epochs")
        issues.extend(f"{name} must be null or omitted" for name in inactive if getattr(self, name) is not None)
        if issues:
            raise ValueError(f"{self.action}: " + "; ".join(issues))
        return self


class AutonomousSearchEvent(Artefact):
    decision: AutonomousSearchDecision
    source: str


class GeneratedDevelopmentPlan(Artefact):
    decision: GeneratedCodeDecision
    reference: GeneratedBundleReference


class GeneratedVerification(Artefact):
    reference: GeneratedBundleReference
    passed: bool
    error: str | None = None


class WorkerEvidence(Artefact):
    configuration: WorkerConfiguration
    image_digest: str | None = None  # Historical metadata only; never used to run jobs.
    isolation: dict[str, Any]
    python_executable: str | None = None


class TrainConfig(Artefact, ExecutionOptions):
    execution_mode: Literal["legacy", "agent_autonomous"] = "legacy"
    segment_targets: list[Annotated[int, Field(ge=1)]] = Field(default_factory=list)
    model_family: ModelFamily | Literal["run_generated"] = "tiny_cnn"
    generated_bundle: GeneratedBundleReference | None = None
    worker: WorkerConfiguration | None = None
    lr: float = Field(ge=1e-7, le=1.0)
    epochs: int = Field(ge=1)
    hidden: int = Field(ge=8, le=1024, multiple_of=8)
    depth: int = Field(default=2, ge=1, le=48)
    dropout: float = Field(default=0.0, ge=0.0, le=0.95)
    batch_size: int = Field(ge=4, le=4096)
    weight_decay: float = Field(ge=0.0, le=1.0)
    class_weighting: bool
    optimizer: OptimizerName = "adamw"
    scheduler: SchedulerName = "none"
    label_smoothing: float = Field(default=0.0, ge=0.0, le=0.5)
    patch_size: Literal[1, 2, 4, 7, 14, 28] = 4
    num_heads: int = Field(default=4, ge=1, le=32)
    mlp_ratio: int = Field(default=4, ge=1, le=16)
    pooling: PoolingName = "cls"
    positional_encoding: PositionalEncodingName = "learned"
    tokenizer_layers: int = Field(default=2, ge=1, le=5)
    early_stopping_patience: int = Field(default=3, ge=0)
    early_stopping_monitor: Literal["val_accuracy", "val_macro_f1", "val_loss"] = "val_accuracy"
    early_stopping_min_delta: float = Field(default=0.0, ge=0.0, le=1.0)
    gradient_clip_val: float = Field(default=1.0, ge=0.0, le=100.0)
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
        if self.segment_targets != sorted(set(self.segment_targets)):
            raise ValueError("recorded segment targets must strictly increase")
        if self.execution_mode == "legacy" and self.epochs > 1000:
            raise ValueError("legacy epochs cannot exceed 1000")
        if self.execution_mode == "legacy" and self.early_stopping_patience > 500:
            raise ValueError("legacy patience cannot exceed 500")
        _validate_model_options(self)
        if self.model_family == "run_generated" and self.generated_bundle is None:
            raise ValueError("run_generated requires a run-scoped bundle")
        if self.model_family == "run_generated" and self.worker is None:
            object.__setattr__(self, "worker", WorkerConfiguration())
        if self.model_family != "run_generated" and self.generated_bundle is not None:
            raise ValueError("generated bundles belong only to run_generated models")
        return self


class EpochMetrics(StrictModel):
    epoch: int = Field(ge=1)
    train_loss: float = Field(ge=0.0)
    val_loss: float | None = Field(default=None, ge=0.0)
    val_accuracy: float = Field(ge=0.0, le=1.0)
    val_macro_f1: float | None = Field(default=None, ge=0.0, le=1.0)
    learning_rate: float | None = Field(default=None, ge=0.0)


class TrainResult(Artefact):
    resume_path: str | None = None
    resume_sha256: str | None = None
    stop_reason: Literal["segment_complete", "early_stopping"] | None = None
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
    framework: Literal["lightning.pytorch", "isolated.python", "local.python"] = "lightning.pytorch"
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
    policy: Literal["legacy", "agent_autonomous"] = "legacy"
    max_trials: int | None = Field(default=None, ge=1, le=256)
    rounds: int | None = Field(default=None, ge=1, le=16)
    search_epochs: int | None = Field(default=None, ge=1, le=1000)
    final_epochs: int | None = Field(default=None, ge=1, le=1000)
    accuracy_tolerance: float = Field(default=0.005, ge=0.0, le=0.05)
    objective: Literal["validation_accuracy_then_macro_f1"] = (
        "validation_accuracy_then_macro_f1"
    )
    test_locked: Literal[True] = True
    framework: Literal["lightning.pytorch", "isolated.python", "local.python"] = "lightning.pytorch"
    progressive_budget: bool = True
    round_epoch_budgets: list[int] = Field(default_factory=list, max_length=16)

    @model_validator(mode="after")
    def validate_epoch_budgets(self) -> SearchPlan:
        if self.policy == "agent_autonomous":
            if any(value is not None for value in (self.max_trials, self.rounds, self.search_epochs, self.final_epochs)) or self.round_epoch_budgets or self.progressive_budget:
                raise ValueError("autonomous search cannot have external search budgets")
            return self
        if any(value is None for value in (self.max_trials, self.rounds, self.search_epochs, self.final_epochs)) or not self.round_epoch_budgets:
            raise ValueError("legacy search requires budgets")
        if self.search_epochs > self.final_epochs:
            raise ValueError("search_epochs cannot exceed final_epochs")
        if len(self.round_epoch_budgets) != self.rounds:
            raise ValueError("round_epoch_budgets must match rounds")
        if self.round_epoch_budgets != sorted(self.round_epoch_budgets):
            raise ValueError("round epoch budgets must be non-decreasing")
        if any(value > self.final_epochs for value in self.round_epoch_budgets):
            raise ValueError("round epoch budget cannot exceed final_epochs")
        return self


class ArchitectureResearch(Artefact):
    """Auditable research brief consumed by design/search stages."""

    analysis: str
    representation_priorities: list[str] = Field(min_length=1, max_length=8)
    parametrization_priorities: list[str] = Field(min_length=1, max_length=8)
    architecture_priorities: list[ModelFamily] = Field(min_length=1, max_length=7)
    transformer_guidance: str
    risks: list[str] = Field(default_factory=list, max_length=8)
    evidence_sources: list[str] = Field(min_length=1, max_length=8)
    evidence_provenance: dict[str, Literal["retrieved_abstract_only", "unverified_curated_excerpt",
                                         "unverified_external_reference"]] = Field(default_factory=dict)
    test_metrics_used: Literal[False] = False
    source: str

class TrialResult(Artefact):
    epoch_budget: int | None = Field(default=None, ge=1)
    requested_epochs: int | None = Field(default=None, ge=1)
    epochs_completed: int | None = Field(default=None, ge=1)
    resume_path: str | None = None
    resume_sha256: str | None = None
    parent_candidate_id: str | None = None
    stop_reason: str | None = None
    learning_curve: list[EpochMetrics] = Field(default_factory=list)
    candidate_id: str
    config_hash: str = Field(min_length=64, max_length=64)
    round_index: int = Field(ge=1)
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
        if self.config.execution_mode == "legacy" and (self.round_index > 16 or len(self.learning_curve) > 1000):
            raise ValueError("legacy trial exceeds round/history limits")
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
    selection_rule: Literal["highest_budget_accuracy_tolerance_then_macro_f1", "agent_durations_accuracy_tolerance_then_macro_f1"] = (
        "highest_budget_accuracy_tolerance_then_macro_f1"
    )
    test_metrics_used: Literal[False] = False
    families_evaluated: list[ModelFamily] = Field(default_factory=list)
    missing_families: list[ModelFamily] = Field(default_factory=list)


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
    recall_ci95_low: float | None = Field(default=None, ge=0.0, le=1.0)
    recall_ci95_high: float | None = Field(default=None, ge=0.0, le=1.0)


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
    accuracy_ci95_low: float | None = Field(default=None, ge=0.0, le=1.0)
    accuracy_ci95_high: float | None = Field(default=None, ge=0.0, le=1.0)

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
    validation_false_accept_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    test_false_accept_rate: float | None = Field(default=None, ge=0.0, le=1.0)


class RiskCoveragePoint(StrictModel):
    target_validation_coverage: float = Field(gt=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    validation_coverage: float = Field(ge=0.0, le=1.0)
    validation_risk: float = Field(ge=0.0, le=1.0)
    test_coverage: float = Field(ge=0.0, le=1.0)
    test_risk: float = Field(ge=0.0, le=1.0)


class AbstentionReport(Artefact):
    detector_status: Literal["legacy_unchecked", "eligible", "no_eligible_detector"] = "legacy_unchecked"
    minimum_scenario_auroc: float = Field(default=0.65, ge=0.5, le=1.0)
    maximum_false_accept_rate: float = Field(default=0.20, ge=0.0, le=1.0)
    corruption_seed: int | None = Field(default=None, ge=0)
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
    risk_coverage_curve: list[RiskCoveragePoint] = Field(default_factory=list)


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
    """Append-only typed store; heavy objects have persisted references too."""

    _VALID_NAME = re.compile(r"^[a-z][a-z0-9_]*$")

    def __init__(self, root: str | Path, run_id: str | None = None):
        self.root = Path(root).resolve()
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
        self.artefacts[name] = artefact.model_copy(deep=True)
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

    @classmethod
    def open(cls, root: str | Path) -> Blackboard:
        """Reopen an existing store without overwriting any artefact versions."""
        bb = cls.__new__(cls)
        bb.root = Path(root).resolve()
        bb.artefact_dir, bb.blob_dir = bb.root / "artefacts", bb.root / "blobs"
        bb.decision_log_path = bb.root / "decision_log.jsonl"
        bb.artefacts, bb.registry, bb._versions = {}, [], {}
        bb._artefact_sequence = 0
        bb.log = [json.loads(line) for line in bb.decision_log_path.read_text(encoding="utf-8").splitlines()]
        if [row["sequence"] for row in bb.log] != list(range(1, len(bb.log) + 1)):
            raise ValueError("decision log sequence is inconsistent")
        bb._event_sequence = len(bb.log)
        types = {item.__name__: item for item in Artefact.__subclasses__()}
        for path in sorted(bb.artefact_dir.glob("*.json")):
            envelope = json.loads(path.read_text(encoding="utf-8"))
            if _sha256_json(envelope["payload"]) != envelope["payload_sha256"]:
                raise ValueError(f"artefact checksum mismatch: {path.name}")
            name = envelope["artefact"]
            if envelope["version"] != bb._versions.get(name, 0) + 1:
                raise ValueError("artefact versions are not contiguous")
            if envelope["sequence"] != bb._artefact_sequence + 1:
                raise ValueError("artefact sequence is not contiguous")
            bb.run_id = envelope["run_id"]
            bb._versions[name] = envelope["version"]
            bb._artefact_sequence = envelope["sequence"]
            bb.artefacts[name] = types[envelope["artefact_type"]].model_validate(envelope["payload"])
            if name in {"approved_extensions", "run_configuration"}:
                from extensions import restore_registry
                data = envelope["payload"]
                restore_registry(data.get("entries", data.get("extension_registry", [])))
            bb.registry.append({"artefact": name, "type": envelope["artefact_type"],
                "version": envelope["version"], "path": path.relative_to(bb.root).as_posix(),
                "sha256": envelope["payload_sha256"], "producer": envelope["producer"]})
        if not bb.registry:
            raise ValueError("cannot reopen an empty Blackboard")
        return bb

    def get(self, name: str) -> Artefact:
        value = self.artefacts[name]
        return value.model_copy(deep=True) if isinstance(value, Artefact) else value

    def get_optional(self, name: str) -> Artefact | None:
        return self.get(name) if name in self.artefacts else None

    def put_blob(self, name: str, obj: Any, *, producer: str) -> None:
        from blobio import save_blob

        self._check_name(name)
        reference_name = f"blob_{name}"
        version = self._versions.get(reference_name, 0) + 1
        reference = save_blob(self, name, obj, version)
        self.put(reference_name, reference, producer=producer)

    def get_blob(self, name: str) -> Any:
        from blobio import load_blob

        reference = self.get(f"blob_{name}")
        if not isinstance(reference, BlobReference):
            raise TypeError("blob hand-off requires a BlobReference")
        return load_blob(self.root, reference)

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
            "prior_art": ["prior_art_brief", "extension_gate_report", "data_profile"],
            "data_audit": ["data_audit_report", "split_manifest", "data_profile"],
            "preprocessing": ["representation_plan", "split_manifest", "data_profile"],
            "architecture_research": ["architecture_research", "extension_gate_report",
                                      "representation_plan", "data_profile"],
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
        # Execution configuration and transport descriptors are audited by
        # deterministic checks, not sent as noisy inputs to domain reasoning.
        newest = [name for name in reversed(self.artefacts)
                  if name not in {"run_configuration", "execution_status", "acceptance_report"}
                  and not name.startswith("blob_")]
        ordered = list(dict.fromkeys([*preferred, *newest]))
        selected: dict[str, Any] = {}
        compacted: list[dict[str, Any]] = []
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
            changes: list[dict[str, Any]] = []
            # Research prose is the subject of this gate, not a preview. The
            # primary artefact may exceed the soft context budget to stay intact.
            selected[name] = (raw if stage == "architecture_research" and name == "architecture_research"
                              else _compact_for_review(raw, path=name, changes=changes))
            candidate = {
                "stage": stage,
                "artefacts": selected,
                "omitted_artefacts": omitted,
                "compacted_fields": [*compacted, *changes],
            }
            text = json.dumps(candidate, ensure_ascii=False, sort_keys=True)
            if len(text) > max_chars and len(selected) > 1:
                selected.pop(name)
                omitted.append(name)
            else:
                compacted.extend(changes)
        return json.dumps(
            {
                "stage": stage,
                "artefacts": selected,
                "omitted_artefacts": omitted,
                "compacted_fields": compacted,
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


def _compact_for_review(data: Any, *, path: str = "", changes: list | None = None) -> Any:
    """Bound verbose prose while preserving all structured numeric evidence."""
    if isinstance(data, str):
        if len(data) > 400:
            if changes is not None:
                changes.append({"path": path, "original_chars": len(data), "shown_chars": 397})
            return data[:397] + "..."
        return data
    if isinstance(data, list):
        if len(data) > 20 and changes is not None:
            changes.append({"path": path, "original_items": len(data), "shown_items": 20})
        return [_compact_for_review(item, path=f"{path}[{index}]", changes=changes)
                for index, item in enumerate(data[:20])]
    if isinstance(data, dict):
        return {str(key): _compact_for_review(value, path=f"{path}.{key}", changes=changes)
                for key, value in data.items()}
    return data

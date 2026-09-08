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
Augmentation = Literal["hflip", "vflip", "rotate90"]


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
    ambiguity_note: str = Field(min_length=3, max_length=600)
    risks: list[str] = Field(default_factory=list, max_length=8)


class RepresentationDecision(StrictModel):
    normalization: Literal["unit", "standardize"]
    augmentations: list[Augmentation] = Field(default_factory=list, max_length=3)
    rationale: str = Field(min_length=3, max_length=800)


class ExperimentDecision(StrictModel):
    lr: float = Field(ge=1e-5, le=1e-2)
    epochs: int = Field(ge=1, le=20)
    hidden: Literal[16, 32, 64]
    batch_size: Literal[32, 64, 128, 256]
    weight_decay: float = Field(ge=0.0, le=0.1)
    class_weighting: bool
    rationale: str = Field(min_length=3, max_length=800)


class ReviewDecision(StrictModel):
    severity: Severity
    action: Action
    issues: list[str] = Field(default_factory=list, max_length=10)
    comment: str = Field(min_length=2, max_length=800)


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
    def validate_splits(self) -> "DatasetManifest":
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
    imbalance_ratio: float = Field(ge=1.0)
    image_shape_hwc: tuple[int, int, int]
    train_channel_mean_unit: list[float] = Field(min_length=3, max_length=3)
    train_channel_std_unit: list[float] = Field(min_length=3, max_length=3)
    blank_image_fraction: float = Field(ge=0.0, le=1.0)
    duplicate_train_samples: int = Field(ge=0)
    ambiguity_note: str
    ambiguity_risks: list[str]
    decision_source: str

    @model_validator(mode="after")
    def validate_counts(self) -> "DataProfile":
        if set(self.class_counts) != {"train", "val", "test"}:
            raise ValueError("class_counts must contain train, val and test")
        if any(len(v) != self.n_classes for v in self.class_counts.values()):
            raise ValueError("every class-count vector must match n_classes")
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
    lr: float = Field(ge=1e-5, le=1e-2)
    epochs: int = Field(ge=1, le=20)
    hidden: Literal[16, 32, 64]
    batch_size: Literal[32, 64, 128, 256]
    weight_decay: float = Field(ge=0.0, le=0.1)
    class_weighting: bool
    seed: int = Field(ge=0)
    device: Literal["cpu", "cuda"]
    rationale: str
    source: str


class EpochMetrics(StrictModel):
    epoch: int = Field(ge=1)
    train_loss: float = Field(ge=0.0)
    val_accuracy: float = Field(ge=0.0, le=1.0)


class TrainResult(Artefact):
    final_train_loss: float = Field(ge=0.0)
    final_val_accuracy: float = Field(ge=0.0, le=1.0)
    best_val_accuracy: float = Field(ge=0.0, le=1.0)
    best_epoch: int = Field(ge=1)
    epochs_completed: int = Field(ge=1)
    history: list[EpochMetrics]
    checkpoint_path: str
    checkpoint_sha256: str = Field(min_length=64, max_length=64)
    seed: int = Field(ge=0)
    device: Literal["cpu", "cuda"]


class ClassMetrics(StrictModel):
    label: int = Field(ge=0, le=8)
    name: str
    support: int = Field(ge=0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)


class EvalReport(Artefact):
    split: Literal["test"] = "test"
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


class AbstentionReport(Artefact):
    method: Literal["max_softmax"] = "max_softmax"
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


class BaselineReport(Artefact):
    dataset: Literal["pathmnist"]
    split_fingerprint: str
    seed: int = Field(ge=0)
    config: TrainConfig
    accuracy: float = Field(ge=0.0, le=1.0)
    macro_f1: float = Field(ge=0.0, le=1.0)
    checkpoint_path: str


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
            "blob_registered", blob=name, producer=producer, python_type=type(obj).__name__
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

    def artefact_summary(self, max_chars: int = 6000) -> str:
        data = {
            name: (
                artefact.model_dump(mode="json")
                if isinstance(artefact, BaseModel)
                else _json_safe(vars(artefact) if hasattr(artefact, "__dict__") else artefact)
            )
            for name, artefact in self.artefacts.items()
        }
        text = json.dumps(data, ensure_ascii=False, sort_keys=True)
        return text if len(text) <= max_chars else text[:max_chars] + "..."

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

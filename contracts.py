"""Artefact contracts + blackboard.

Agents never call each other. They only read/write versioned, typed artefacts
through the Blackboard. These dataclasses ARE the interfaces between agents,
which is what lets any single agent be swapped without touching the others
(maximum cohesion, minimum coupling).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# --- Artefacts exchanged between agents (one schema per hand-off) -------------

@dataclass
class DataProfile:            # produced by ProfilingAmbiguityAgent (WP2)
    dataset: str
    n_channels: int
    n_classes: int
    n_samples: int
    class_counts: list[int]
    imbalance_ratio: float          # max/min class count
    ambiguity_note: str             # hybrid: LLM (or heuristic) commentary


@dataclass
class RepresentationPlan:     # produced by PreprocessingAgent (hybrid): the
    normalization: str          # "architected" best input representation.
    augmentations: list[str]    # normalization: "unit" | "standardize"
    rationale: str              # augmentations: label-preserving transforms only
    source: str


@dataclass
class SplitManifest:          # produced by PreprocessingAgent
    seed: int
    train_size: int
    val_size: int
    test_size: int


@dataclass
class TrainConfig:            # produced by ExperimentDesignAgent (WP3, hybrid)
    lr: float
    epochs: int
    hidden: int
    batch_size: int
    rationale: str
    source: str                     # "heuristic" or the LLM model name


@dataclass
class TrainResult:
    final_train_loss: float
    val_accuracy: float


@dataclass
class EvalReport:
    accuracy: float
    macro_precision: float
    per_class_precision: list[float]


@dataclass
class AbstentionReport:       # produced by AbstentionOODAgent (WP5)
    threshold: float
    coverage: float                 # fraction of cases the system commits to
    abstain_rate: float             # fraction sent to human review ("unclear")
    accuracy_on_covered: float


@dataclass
class AnomalyReport:          # produced by ReviewerConsistencyAgent (WP6)
    stage: str
    severity: str                   # "ok" | "warning" | "critical"
    issues: list[str] = field(default_factory=list)
    comment: str = ""


# --- Blackboard: the shared, versioned artefact store -------------------------

class Blackboard:
    """Two stores: `artefacts` (small, typed, JSON-persisted, audited) and
    `blobs` (in-memory heavy objects like numpy arrays / trained weights).
    Every write is appended to a decision log for traceability (WP1/WP6 KPI)."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        (self.root / "artefacts").mkdir(parents=True, exist_ok=True)
        self.artefacts: dict[str, Any] = {}
        self.blobs: dict[str, Any] = {}
        self.log: list[dict] = []

    def put(self, name: str, artefact: Any) -> None:
        self.artefacts[name] = artefact
        path = self.root / "artefacts" / f"{name}.json"
        path.write_text(json.dumps(asdict(artefact), indent=2, default=str))
        self.log.append({"op": "put", "artefact": name})

    def get(self, name: str) -> Any:
        return self.artefacts[name]

    def put_blob(self, name: str, obj: Any) -> None:
        self.blobs[name] = obj

    def get_blob(self, name: str) -> Any:
        return self.blobs[name]

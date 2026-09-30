"""Pure helpers for bounded, validation-only agentic model search."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from typing import Any

from contracts import (
    CandidateProposal,
    RepresentationDecision,
    TrainConfig,
    TrialResult,
)


def candidate_hash(candidate: CandidateProposal) -> str:
    """Hash only executable choices, excluding names and prose rationale."""
    payload = candidate.model_dump(mode="json", exclude={"candidate_id", "rationale"})
    if candidate.model_family == "resnet18":
        # ResNet-18 has fixed topology; the generic ``depth`` field is ignored
        # by its builder and therefore must not create a false-new trial.
        payload["depth"] = "fixed_resnet18"
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def proposal_to_config(
    candidate: CandidateProposal,
    *,
    seed: int,
    device: str,
    epochs: int,
    source: str,
) -> TrainConfig:
    return TrainConfig(
        model_family=candidate.model_family,
        lr=candidate.lr,
        epochs=epochs,
        hidden=candidate.hidden,
        depth=candidate.depth,
        dropout=candidate.dropout,
        batch_size=candidate.batch_size,
        weight_decay=candidate.weight_decay,
        class_weighting=candidate.class_weighting,
        optimizer=candidate.optimizer,
        scheduler=candidate.scheduler,
        label_smoothing=candidate.label_smoothing,
        patch_size=candidate.patch_size,
        num_heads=candidate.num_heads,
        mlp_ratio=candidate.mlp_ratio,
        pooling=candidate.pooling,
        positional_encoding=candidate.positional_encoding,
        tokenizer_layers=candidate.tokenizer_layers,
        early_stopping_patience=min(candidate.early_stopping_patience, max(0, epochs - 1)),
        gradient_clip_val=candidate.gradient_clip_val,
        seed=seed,
        device=device,
        rationale=candidate.rationale,
        source=source,
    )


def proposal_to_representation(candidate: CandidateProposal) -> RepresentationDecision:
    return RepresentationDecision(
        normalization=candidate.normalization,
        augmentations=candidate.augmentations,
        rationale=candidate.rationale,
    )


def rank_trials(
    trials: Iterable[TrialResult], *, accuracy_tolerance: float
) -> TrialResult:
    """Select by accuracy first, macro-F1 inside a declared tolerance band."""
    completed = [trial for trial in trials if trial.status == "completed"]
    if not completed:
        raise RuntimeError("model search produced no successful trial")
    max_accuracy = max(float(trial.validation_accuracy) for trial in completed)
    eligible = [
        trial
        for trial in completed
        if float(trial.validation_accuracy) >= max_accuracy - accuracy_tolerance
    ]
    return max(
        eligible,
        key=lambda trial: (
            float(trial.validation_macro_f1),
            float(trial.validation_accuracy),
            -trial.duration_seconds,
            trial.candidate_id,
        ),
    )


def default_candidates(round_index: int) -> list[CandidateProposal]:
    """Diverse deterministic fallbacks used offline or after invalid LLM output."""
    pools: dict[int, list[dict[str, Any]]] = {
        1: [
            {
                "candidate_id": "compact_transformer",
                "model_family": "compact_transformer",
                "hidden": 48,
                "depth": 2,
                "dropout": 0.10,
                "normalization": "standardize",
                "augmentations": ["hflip", "contrast"],
                "optimizer": "adamw",
                "scheduler": "cosine",
                "lr": 0.0005,
                "weight_decay": 0.0001,
                "class_weighting": False,
                "label_smoothing": 0.05,
                "batch_size": 128,
                "num_heads": 4,
                "mlp_ratio": 3,
                "pooling": "attention",
                "positional_encoding": "learned",
                "tokenizer_layers": 2,
                "rationale": "Convolutional tokenization adds a useful local-image prior before attention.",
            },
            {
                "candidate_id": "residual_balanced",
                "model_family": "residual_cnn",
                "hidden": 32,
                "depth": 3,
                "dropout": 0.10,
                "normalization": "standardize",
                "augmentations": ["hflip"],
                "optimizer": "adamw",
                "scheduler": "cosine",
                "lr": 0.001,
                "weight_decay": 0.0001,
                "class_weighting": False,
                "label_smoothing": 0.05,
                "batch_size": 128,
                "rationale": "Residual blocks, light regularisation and the strongest prior ablation.",
            },
            {
                "candidate_id": "resnet18_conservative",
                "model_family": "resnet18",
                "hidden": 32,
                "depth": 3,
                "dropout": 0.10,
                "normalization": "standardize",
                "augmentations": ["hflip", "vflip"],
                "optimizer": "adamw",
                "scheduler": "cosine",
                "lr": 0.0005,
                "weight_decay": 0.0001,
                "class_weighting": False,
                "label_smoothing": 0.05,
                "batch_size": 128,
                "rationale": "A CIFAR-style ResNet-18 tests additional capacity without ImageNet assumptions.",
            },
            {
                "candidate_id": "tiny_regularized",
                "model_family": "tiny_cnn",
                "hidden": 32,
                "depth": 2,
                "dropout": 0.15,
                "normalization": "standardize",
                "augmentations": ["hflip"],
                "optimizer": "adamw",
                "scheduler": "cosine",
                "lr": 0.001,
                "weight_decay": 0.0001,
                "class_weighting": False,
                "label_smoothing": 0.0,
                "batch_size": 128,
                "rationale": "Low-cost control close to the best representation ablation.",
            },
            {
                "candidate_id": "residual_weighted",
                "model_family": "residual_cnn",
                "hidden": 24,
                "depth": 2,
                "dropout": 0.20,
                "normalization": "standardize",
                "augmentations": ["hflip", "rotate90"],
                "optimizer": "adamw",
                "scheduler": "reduce_on_plateau",
                "lr": 0.0008,
                "weight_decay": 0.001,
                "class_weighting": True,
                "label_smoothing": 0.05,
                "batch_size": 128,
                "rationale": "Tests a smaller weighted residual model for weak minority classes.",
            },
        ],
        2: [
            {
                "candidate_id": "vit_patch4",
                "model_family": "vision_transformer",
                "hidden": 64,
                "depth": 3,
                "dropout": 0.10,
                "normalization": "standardize",
                "augmentations": ["hflip", "vflip"],
                "optimizer": "adamw",
                "scheduler": "cosine",
                "lr": 0.0003,
                "weight_decay": 0.0001,
                "class_weighting": False,
                "label_smoothing": 0.05,
                "batch_size": 128,
                "patch_size": 2,
                "num_heads": 8,
                "mlp_ratio": 3,
                "pooling": "mean",
                "positional_encoding": "sinusoidal",
                "rationale": "A fine-grained patch-2 ViT tests global interactions over 196 tissue tokens.",
            },
            {
                "candidate_id": "residual_low_lr",
                "model_family": "residual_cnn",
                "hidden": 32,
                "depth": 3,
                "dropout": 0.15,
                "normalization": "standardize",
                "augmentations": ["hflip"],
                "optimizer": "adamw",
                "scheduler": "cosine",
                "lr": 0.0005,
                "weight_decay": 0.0001,
                "class_weighting": False,
                "label_smoothing": 0.05,
                "batch_size": 128,
                "rationale": "Refines the residual candidate with a lower learning rate.",
            },
            {
                "candidate_id": "residual_wide",
                "model_family": "residual_cnn",
                "hidden": 48,
                "depth": 3,
                "dropout": 0.20,
                "normalization": "standardize",
                "augmentations": ["hflip"],
                "optimizer": "adamw",
                "scheduler": "cosine",
                "lr": 0.0008,
                "weight_decay": 0.0005,
                "class_weighting": False,
                "label_smoothing": 0.05,
                "batch_size": 128,
                "rationale": "Tests whether moderate width improves tissue texture modelling.",
            },
            {
                "candidate_id": "resnet18_low_decay",
                "model_family": "resnet18",
                "hidden": 32,
                "depth": 3,
                "dropout": 0.0,
                "normalization": "standardize",
                "augmentations": ["hflip"],
                "optimizer": "adamw",
                "scheduler": "cosine",
                "lr": 0.0008,
                "weight_decay": 0.00001,
                "class_weighting": False,
                "label_smoothing": 0.05,
                "batch_size": 128,
                "rationale": "Refines ResNet-18 while avoiding the prior excessive weight decay.",
            },
            {
                "candidate_id": "residual_sgd",
                "model_family": "residual_cnn",
                "hidden": 32,
                "depth": 3,
                "dropout": 0.10,
                "normalization": "standardize",
                "augmentations": ["hflip"],
                "optimizer": "sgd",
                "scheduler": "cosine",
                "lr": 0.01,
                "weight_decay": 0.0001,
                "class_weighting": False,
                "label_smoothing": 0.0,
                "batch_size": 128,
                "rationale": "Provides an optimizer-diverse candidate with momentum SGD.",
            },
        ],
        3: [
            {
                "candidate_id": "residual_deep",
                "model_family": "residual_cnn",
                "hidden": 48,
                "depth": 4,
                "dropout": 0.20,
                "normalization": "standardize",
                "augmentations": ["hflip", "vflip"],
                "optimizer": "adamw",
                "scheduler": "cosine",
                "lr": 0.0003,
                "weight_decay": 0.0005,
                "class_weighting": False,
                "label_smoothing": 0.05,
                "batch_size": 128,
                "rationale": "Tests additional residual depth with conservative optimisation.",
            },
            {
                "candidate_id": "resnet18_wide",
                "model_family": "resnet18",
                "hidden": 48,
                "depth": 2,
                "dropout": 0.15,
                "normalization": "standardize",
                "augmentations": ["hflip", "vflip", "rotate90"],
                "optimizer": "adamw",
                "scheduler": "cosine",
                "lr": 0.0003,
                "weight_decay": 0.0001,
                "class_weighting": False,
                "label_smoothing": 0.05,
                "batch_size": 64,
                "rationale": "Tests more capacity with all guarded orientation-safe transforms.",
            },
            {
                "candidate_id": "tiny_wide",
                "model_family": "tiny_cnn",
                "hidden": 64,
                "depth": 3,
                "dropout": 0.25,
                "normalization": "standardize",
                "augmentations": ["hflip"],
                "optimizer": "adamw",
                "scheduler": "reduce_on_plateau",
                "lr": 0.0005,
                "weight_decay": 0.001,
                "class_weighting": False,
                "label_smoothing": 0.05,
                "batch_size": 128,
                "rationale": "A wide but inexpensive CNN probes whether residual links are necessary.",
            },
            {
                "candidate_id": "residual_adam",
                "model_family": "residual_cnn",
                "hidden": 32,
                "depth": 3,
                "dropout": 0.10,
                "normalization": "standardize",
                "augmentations": ["hflip", "rotate90"],
                "optimizer": "adam",
                "scheduler": "none",
                "lr": 0.0005,
                "weight_decay": 0.00001,
                "class_weighting": False,
                "label_smoothing": 0.0,
                "batch_size": 128,
                "rationale": "Provides an Adam control without scheduling or strong regularisation.",
            },
        ],
        4: [
            {
                "candidate_id": "residual_unit_scale",
                "model_family": "residual_cnn",
                "hidden": 48,
                "depth": 3,
                "dropout": 0.10,
                "normalization": "unit",
                "augmentations": ["hflip"],
                "optimizer": "adamw",
                "scheduler": "cosine",
                "lr": 0.0005,
                "weight_decay": 0.0001,
                "class_weighting": False,
                "label_smoothing": 0.05,
                "batch_size": 128,
                "rationale": "Checks whether train-statistic standardisation is actually beneficial.",
            },
            {
                "candidate_id": "resnet18_sgd",
                "model_family": "resnet18",
                "hidden": 32,
                "depth": 2,
                "dropout": 0.10,
                "normalization": "standardize",
                "augmentations": ["hflip", "vflip"],
                "optimizer": "sgd",
                "scheduler": "cosine",
                "lr": 0.005,
                "weight_decay": 0.0001,
                "class_weighting": False,
                "label_smoothing": 0.05,
                "batch_size": 128,
                "rationale": "Tests momentum SGD on the fixed-depth residual architecture.",
            },
            {
                "candidate_id": "tiny_weighted",
                "model_family": "tiny_cnn",
                "hidden": 48,
                "depth": 3,
                "dropout": 0.20,
                "normalization": "standardize",
                "augmentations": ["hflip", "vflip"],
                "optimizer": "adamw",
                "scheduler": "cosine",
                "lr": 0.0008,
                "weight_decay": 0.0005,
                "class_weighting": True,
                "label_smoothing": 0.05,
                "batch_size": 128,
                "rationale": "A weighted compact model challenges gains attributed to model capacity.",
            },
            {
                "candidate_id": "residual_small_batch",
                "model_family": "residual_cnn",
                "hidden": 32,
                "depth": 4,
                "dropout": 0.25,
                "normalization": "standardize",
                "augmentations": ["hflip", "vflip", "rotate90"],
                "optimizer": "adamw",
                "scheduler": "reduce_on_plateau",
                "lr": 0.0003,
                "weight_decay": 0.001,
                "class_weighting": True,
                "label_smoothing": 0.10,
                "batch_size": 64,
                "rationale": "Combines strong regularisation and a smaller batch for the final bounded probe.",
            },
        ],
    }
    template = pools.get(round_index, pools[4])
    return [CandidateProposal.model_validate(item) for item in template]


def safe_trial_name(candidate_id: str, config_hash: str) -> str:
    clean = re.sub(r"[^a-z0-9_]", "_", candidate_id.lower()).strip("_")
    return f"trial_{clean}_{config_hash[:8]}"


def lightning_config_payload(
    config: TrainConfig,
    representation: RepresentationDecision,
    *,
    data_root: str | None,
    train_limit: int | None,
    val_limit: int | None,
    test_limit: int | None,
    download: bool,
    class_weights: list[float] | None = None,
) -> dict[str, Any]:
    """Return a JSON-compatible configuration accepted by LightningCLI."""
    return {
        "seed_everything": config.seed,
        "trainer": {
            "max_epochs": config.epochs,
            "accelerator": "gpu" if config.device == "cuda" else "cpu",
            "devices": 1,
            "deterministic": True,
            "gradient_clip_val": config.gradient_clip_val,
            "logger": False,
            "enable_progress_bar": True,
            "num_sanity_val_steps": 0,
        },
        "model": {
            "model_family": config.model_family,
            "hidden": config.hidden,
            "depth": config.depth,
            "dropout": config.dropout,
            "optimizer": config.optimizer,
            "scheduler": config.scheduler,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "label_smoothing": config.label_smoothing,
            "patch_size": config.patch_size,
            "num_heads": config.num_heads,
            "mlp_ratio": config.mlp_ratio,
            "pooling": config.pooling,
            "positional_encoding": config.positional_encoding,
            "tokenizer_layers": config.tokenizer_layers,
            "class_weights": class_weights,
        },
        "data": {
            "data_root": data_root,
            "batch_size": config.batch_size,
            "seed": config.seed,
            "normalization": representation.normalization,
            "augmentations": representation.augmentations,
            "train_limit": train_limit,
            "val_limit": val_limit,
            "test_limit": test_limit,
            "download": download,
        },
    }

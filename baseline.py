"""Conventional baseline and deterministic representation ablations."""
from __future__ import annotations

from pathlib import Path

from contracts import (
    AblationReport,
    AblationScenario,
    BaselineReport,
    RepresentationDecision,
    TrainConfig,
)
from ml import (
    DatasetBundle,
    common_split_fingerprint,
    evaluate_probabilities,
    predict_probabilities,
    prepare_data,
    train_tiny_cnn,
)


def run_baseline(
    bundle: DatasetBundle,
    root: str | Path,
    *,
    seed: int,
    device: str,
    epochs: int,
) -> BaselineReport:
    """Train a fixed non-agentic baseline on the exact same selected splits."""
    root_path = Path(root)
    representation = RepresentationDecision(
        normalization="standardize",
        augmentations=[],
        rationale="Fixed conventional preprocessing: train-only standardization.",
    )
    prepared = prepare_data(bundle, representation)
    config = TrainConfig(
        lr=1e-3,
        epochs=epochs,
        hidden=16,
        batch_size=128,
        weight_decay=1e-4,
        class_weighting=False,
        seed=seed,
        device=device,
        rationale="Fixed conventional control with the agentic run's epoch budget.",
        source="baseline:fixed",
    )
    checkpoint = root_path / "blobs" / "baseline_model.pt"
    output = train_tiny_cnn(prepared, config, checkpoint)
    probabilities, targets = predict_probabilities(
        output.model,
        prepared,
        "test",
        config.batch_size,
        device=device,
        seed=seed,
    )
    evaluation = evaluate_probabilities(probabilities, targets, bundle.labels)
    return BaselineReport(
        dataset="pathmnist",
        split_fingerprint=common_split_fingerprint(bundle),
        seed=seed,
        config=config,
        accuracy=evaluation.accuracy,
        macro_f1=evaluation.macro_f1,
        checkpoint_path=str(checkpoint.relative_to(root_path)),
    )


def run_representation_ablations(
    bundle: DatasetBundle,
    root: str | Path,
    *,
    template_config: TrainConfig,
) -> AblationReport:
    """Run the three proposal-required representation scenarios."""
    root_path = Path(root)
    choices = [
        ("representation_off", "unit", []),
        ("standardize_only", "standardize", []),
        ("standardize_hflip", "standardize", ["hflip"]),
    ]
    results: list[AblationScenario] = []
    for name, normalization, augmentations in choices:
        decision = RepresentationDecision(
            normalization=normalization,
            augmentations=augmentations,
            rationale=f"Deterministic ablation scenario: {name}.",
        )
        prepared = prepare_data(bundle, decision)
        config = template_config.model_copy(
            update={
                "source": f"ablation:{name}",
                "rationale": f"Common config; only representation varies ({name}).",
            }
        )
        checkpoint = root_path / "blobs" / f"ablation_{name}.pt"
        output = train_tiny_cnn(prepared, config, checkpoint)
        probabilities, targets = predict_probabilities(
            output.model,
            prepared,
            "test",
            config.batch_size,
            device=config.device,
            seed=config.seed,
        )
        evaluation = evaluate_probabilities(probabilities, targets, bundle.labels)
        results.append(
            AblationScenario(
                name=name,
                normalization=normalization,
                augmentations=augmentations,
                seed=config.seed,
                accuracy=evaluation.accuracy,
                macro_f1=evaluation.macro_f1,
            )
        )
    return AblationReport(
        scenarios=results,
        common_split_fingerprint=common_split_fingerprint(bundle),
    )

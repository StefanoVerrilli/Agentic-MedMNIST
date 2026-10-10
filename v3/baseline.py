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
    train_model,
)


def _relative_to_root(path: str | Path, root: str | Path) -> str:
    """Return a stable run-relative path even when Lightning returns absolute paths."""
    candidate = Path(path)
    root_path = Path(root)
    try:
        return candidate.resolve().relative_to(root_path.resolve()).as_posix()
    except ValueError:
        return candidate.as_posix()


def run_baseline(
    bundle: DatasetBundle,
    root: str | Path,
    *,
    seed: int,
    device: str,
    epochs: int,
    worker=None,
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
        model_family="tiny_cnn",
        lr=1e-3,
        epochs=epochs,
        hidden=16,
        batch_size=128,
        weight_decay=1e-4,
        class_weighting=False,
        optimizer="adamw",
        scheduler="cosine",
        label_smoothing=0.0,
        seed=seed,
        device=device,
        rationale="Fixed conventional control with the agentic run's epoch budget.",
        source="baseline:fixed",
        worker=worker,
    )
    checkpoint = root_path / "blobs" / "baseline_model.ckpt"
    output = train_model(prepared, config, checkpoint)
    probabilities, targets = predict_probabilities(
        output.model,
        prepared,
        "test",
        config.batch_size,
        device=device,
        seed=seed,
    )
    evaluation = evaluate_probabilities(
        probabilities, targets, bundle.labels, split="test"
    )
    actual_checkpoint = Path(output.result.checkpoint_path)
    return BaselineReport(
        dataset="pathmnist",
        split_fingerprint=common_split_fingerprint(bundle),
        seed=seed,
        config=config,
        accuracy=evaluation.accuracy,
        macro_f1=evaluation.macro_f1,
        checkpoint_path=_relative_to_root(actual_checkpoint, root_path),
        checkpoint_sha256=output.result.checkpoint_sha256,
        training_log_path=(
            _relative_to_root(output.result.training_log_path, root_path)
            if output.result.training_log_path is not None
            else None
        ),
        lightning_csv_path=(
            _relative_to_root(output.result.lightning_csv_path, root_path)
            if output.result.lightning_csv_path is not None
            else None
        ),
    )


def run_representation_ablations(
    bundle: DatasetBundle,
    root: str | Path,
    *,
    template_config: TrainConfig,
    completed_scenarios: list[AblationScenario] | None = None,
    on_scenario=None,
) -> AblationReport:
    """Run the three proposal-required representation scenarios."""
    root_path = Path(root)
    choices = [
        ("representation_off", "unit", []),
        ("standardize_only", "standardize", []),
        ("standardize_hflip", "standardize", ["hflip"]),
    ]
    completed = {item.name: item for item in (completed_scenarios or [])}
    if len(completed) != len(completed_scenarios or []) or set(completed) - {item[0] for item in choices}:
        raise ValueError("invalid completed ablation scenarios")
    results: list[AblationScenario] = []
    for name, normalization, augmentations in choices:
        if name in completed:
            item = completed[name]
            if (item.seed != template_config.seed or item.normalization != normalization
                    or item.augmentations != augmentations):
                raise ValueError("completed ablation protocol mismatch")
            from contracts import sha256_file
            from research import contained_path
            if not item.checkpoint_path or not item.checkpoint_sha256 or sha256_file(
                    contained_path(root_path, item.checkpoint_path)) != item.checkpoint_sha256:
                raise ValueError("completed ablation checkpoint checksum mismatch")
            results.append(item)
            continue
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
        checkpoint = root_path / "blobs" / f"ablation_{name}.ckpt"
        output = train_model(prepared, config, checkpoint)
        probabilities, targets = predict_probabilities(
            output.model,
            prepared,
            "test",
            config.batch_size,
            device=config.device,
            seed=config.seed,
        )
        evaluation = evaluate_probabilities(
            probabilities, targets, bundle.labels, split="test"
        )
        actual_checkpoint = Path(output.result.checkpoint_path)
        results.append(
            AblationScenario(
                name=name,
                normalization=normalization,
                augmentations=augmentations,
                seed=config.seed,
                accuracy=evaluation.accuracy,
                macro_f1=evaluation.macro_f1,
                checkpoint_path=_relative_to_root(actual_checkpoint, root_path),
                checkpoint_sha256=output.result.checkpoint_sha256,
                training_log_path=(
                    _relative_to_root(output.result.training_log_path, root_path)
                    if output.result.training_log_path is not None
                    else None
                ),
                lightning_csv_path=(
                    _relative_to_root(output.result.lightning_csv_path, root_path)
                    if output.result.lightning_csv_path is not None
                    else None
                ),
            )
        )
        if on_scenario is not None:
            on_scenario(results[-1])
    return AblationReport(
        scenarios=results,
        common_split_fingerprint=common_split_fingerprint(bundle),
    )

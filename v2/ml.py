"""Deterministic, shared ML primitives for agentic and baseline experiments."""

from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from contracts import (
    ClassMetrics,
    EvalReport,
    RepresentationDecision,
    TrainConfig,
    TrainResult,
    sha256_file,
)


@dataclass(frozen=True)
class DatasetBundle:
    dataset: str
    n_channels: int
    n_classes: int
    labels: tuple[str, ...]
    task: str
    official_sizes: dict[str, int]
    images: dict[str, Any]
    targets: dict[str, Any]
    selected_indices: dict[str, Any]
    selected_index_sha256: dict[str, str]
    data_reference: str
    archive_md5: str
    license: str
    test_domain_note: str


@dataclass(frozen=True)
class PreparedData:
    bundle: DatasetBundle
    normalization: str
    augmentations: tuple[str, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]

    @property
    def augmented_train_size(self) -> int:
        return len(self.bundle.targets["train"]) * (1 + len(self.augmentations))


@dataclass(frozen=True)
class TrainingOutput:
    model: Any
    result: TrainResult


def load_pathmnist(
    *,
    seed: int,
    train_limit: int | None = None,
    val_limit: int | None = None,
    test_limit: int | None = None,
    data_root: str | Path | None = None,
    download: bool = True,
) -> DatasetBundle:
    """Load PathMNIST's official train/validation/test splits.

    Optional limits are deterministic and stratified. A limit of ``None``
    keeps the complete official split.
    """
    import medmnist
    import numpy as np
    from medmnist import INFO

    flag = "pathmnist"
    info = INFO[flag]
    data_class = getattr(medmnist, info["python_class"])
    kwargs: dict[str, Any] = {"download": download}
    if data_root is not None:
        kwargs["root"] = str(data_root)

    datasets = {
        split: data_class(split=split, **kwargs) for split in ("train", "val", "test")
    }
    limits = {"train": train_limit, "val": val_limit, "test": test_limit}
    images: dict[str, Any] = {}
    targets: dict[str, Any] = {}
    selected: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    for offset, split in enumerate(("train", "val", "test")):
        labels = np.asarray(datasets[split].labels).reshape(-1).astype("int64")
        indices = stratified_indices(labels, limits[split], seed + offset)
        images[split] = np.asarray(datasets[split].imgs)[indices]
        targets[split] = labels[indices]
        selected[split] = indices
        hashes[split] = index_fingerprint(split, indices, targets[split])

    label_names = tuple(
        info["label"][str(index)] for index in range(len(info["label"]))
    )
    return DatasetBundle(
        dataset=flag,
        n_channels=int(info["n_channels"]),
        n_classes=len(label_names),
        labels=label_names,
        task=str(info["task"]),
        official_sizes={k: int(v) for k, v in info["n_samples"].items()},
        images=images,
        targets=targets,
        selected_indices=selected,
        selected_index_sha256=hashes,
        data_reference=str(info["url"]),
        archive_md5=str(info["MD5"]),
        license=str(info["license"]),
        test_domain_note=(
            "CRC-VAL-HE-7K test images originate from a different clinical center."
        ),
    )


def stratified_indices(labels: Any, limit: int | None, seed: int) -> Any:
    """Return deterministic class-proportional indices without sklearn."""
    import numpy as np

    y = np.asarray(labels).reshape(-1)
    n = len(y)
    if limit is None or limit <= 0 or limit >= n:
        return np.arange(n, dtype="int64")
    classes, counts = np.unique(y, return_counts=True)
    if limit < len(classes):
        raise ValueError(
            f"limit {limit} is smaller than the {len(classes)} observed classes"
        )

    exact = counts * (float(limit) / n)
    allocation = np.maximum(1, np.floor(exact).astype(int))
    allocation = np.minimum(allocation, counts)
    while int(allocation.sum()) > limit:
        candidates = np.where(allocation > 1)[0]
        choice = candidates[np.argmax(allocation[candidates] - exact[candidates])]
        allocation[choice] -= 1
    while int(allocation.sum()) < limit:
        candidates = np.where(allocation < counts)[0]
        choice = candidates[np.argmax(exact[candidates] - allocation[candidates])]
        allocation[choice] += 1

    rng = np.random.default_rng(seed)
    chunks = [
        rng.choice(np.flatnonzero(y == cls), size=int(take), replace=False)
        for cls, take in zip(classes, allocation)
    ]
    result = np.concatenate(chunks).astype("int64")
    rng.shuffle(result)
    return result


def index_fingerprint(split: str, indices: Any, labels: Any) -> str:
    import numpy as np

    digest = hashlib.sha256(split.encode("utf-8"))
    digest.update(np.ascontiguousarray(indices, dtype="int64").tobytes())
    digest.update(np.ascontiguousarray(labels, dtype="int64").tobytes())
    return digest.hexdigest()


def common_split_fingerprint(bundle: DatasetBundle) -> str:
    digest = hashlib.sha256()
    for split in ("train", "val", "test"):
        digest.update(bundle.selected_index_sha256[split].encode("ascii"))
    return digest.hexdigest()


def channel_statistics(
    images: Any, batch_size: int = 2048
) -> tuple[list[float], list[float]]:
    """Compute unit-scale per-channel statistics without a full float copy."""
    import numpy as np

    total = np.zeros(3, dtype="float64")
    total_sq = np.zeros(3, dtype="float64")
    pixels = 0
    for start in range(0, len(images), batch_size):
        batch = np.asarray(images[start : start + batch_size], dtype="float64") / 255.0
        if batch.ndim == 3:
            batch = batch[..., None]
        flat = batch.reshape(-1, batch.shape[-1])
        total += flat.sum(axis=0)
        total_sq += np.square(flat).sum(axis=0)
        pixels += len(flat)
    mean = total / pixels
    variance = np.maximum(total_sq / pixels - np.square(mean), 0.0)
    std = np.sqrt(variance)
    return mean.tolist(), std.tolist()


def count_duplicate_images(images: Any) -> int:
    seen: set[bytes] = set()
    duplicates = 0
    for image in images:
        digest = hashlib.sha256(memoryview(image)).digest()
        if digest in seen:
            duplicates += 1
        else:
            seen.add(digest)
    return duplicates


def blank_fraction(images: Any, batch_size: int = 4096) -> float:
    import numpy as np

    blank = 0
    for start in range(0, len(images), batch_size):
        batch = np.asarray(images[start : start + batch_size])
        flattened = batch.reshape(len(batch), -1)
        blank += int((np.ptp(flattened, axis=1) <= 1).sum())
    return float(blank / len(images))


def prepare_data(
    bundle: DatasetBundle,
    decision: RepresentationDecision,
) -> PreparedData:
    mean, std = channel_statistics(bundle.images["train"])
    safe_std = [max(value, 1e-6) for value in std]
    return PreparedData(
        bundle=bundle,
        normalization=decision.normalization,
        augmentations=tuple(dict.fromkeys(decision.augmentations)),
        mean=tuple(float(x) for x in mean),
        std=tuple(float(x) for x in safe_std),
    )


def resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    import torch

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return requested


def train_model(
    prepared: PreparedData,
    config: TrainConfig,
    checkpoint_path: str | Path,
    *,
    enable_progress_bar: bool = True,
    echo_epoch_log: bool = True,
) -> TrainingOutput:
    """Train one bounded candidate through the shared Lightning backend."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger

    from lightning_components import (
        MetricsHistory,
        PathMNISTLitModule,
        PreparedDataModule,
    )

    seed_everything(config.seed)
    device = resolve_device(config.device)
    class_weights = (
        inverse_frequency_class_weights(
            prepared.bundle.targets["train"], prepared.bundle.n_classes
        )
        if config.class_weighting
        else None
    )

    model = PathMNISTLitModule(
        model_family=config.model_family,
        in_channels=prepared.bundle.n_channels,
        n_classes=prepared.bundle.n_classes,
        hidden=config.hidden,
        depth=config.depth,
        dropout=config.dropout,
        patch_size=config.patch_size,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        pooling=config.pooling,
        positional_encoding=config.positional_encoding,
        tokenizer_layers=config.tokenizer_layers,
        optimizer=config.optimizer,
        scheduler=config.scheduler,
        lr=config.lr,
        weight_decay=config.weight_decay,
        label_smoothing=config.label_smoothing,
        class_weights=class_weights,
    )
    data_module = PreparedDataModule(
        prepared, batch_size=config.batch_size, seed=config.seed
    )
    target = Path(checkpoint_path)
    if target.suffix != ".ckpt":
        target = target.with_suffix(".ckpt")
    target.parent.mkdir(parents=True, exist_ok=True)
    epoch_log_path = target.with_suffix(".training.jsonl")
    csv_logger = CSVLogger(
        save_dir=str(target.parent / "lightning_logs"),
        name=target.stem,
        version=0,
    )
    checkpoint = ModelCheckpoint(
        dirpath=target.parent,
        filename=target.stem,
        monitor="val_accuracy",
        mode="max",
        save_top_k=1,
        save_last=False,
        auto_insert_metric_name=False,
        enable_version_counter=False,
    )
    history_callback = MetricsHistory(
        echo=echo_epoch_log,
        label=target.stem,
        jsonl_path=epoch_log_path,
    )
    callbacks: list[Any] = [checkpoint, history_callback]
    if config.early_stopping_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor="val_accuracy",
                mode="max",
                patience=config.early_stopping_patience,
                check_finite=True,
            )
        )
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        accelerator="gpu" if device == "cuda" else "cpu",
        devices=1,
        deterministic=True,
        logger=csv_logger,
        enable_checkpointing=True,
        callbacks=callbacks,
        enable_progress_bar=enable_progress_bar,
        enable_model_summary=False,
        gradient_clip_val=config.gradient_clip_val,
        num_sanity_val_steps=0,
        log_every_n_steps=10,
    )
    trainer.fit(model, datamodule=data_module)
    if not checkpoint.best_model_path:
        raise RuntimeError("Lightning produced no best checkpoint")
    target = Path(checkpoint.best_model_path)
    csv_metrics_path = Path(csv_logger.log_dir) / "metrics.csv"
    best_model = PathMNISTLitModule.load_from_checkpoint(
        str(target), map_location=device
    ).to(device)
    history = history_callback.history
    if not history:
        raise RuntimeError("Lightning produced no epoch history")
    best_row = max(history, key=lambda row: row.val_accuracy)
    result = TrainResult(
        final_train_loss=history[-1].train_loss,
        final_val_accuracy=history[-1].val_accuracy,
        best_val_accuracy=best_row.val_accuracy,
        best_epoch=best_row.epoch,
        epochs_completed=len(history),
        history=history,
        checkpoint_path=str(target),
        checkpoint_sha256=sha256_file(target),
        training_log_path=str(epoch_log_path.resolve()),
        lightning_csv_path=(
            str(csv_metrics_path.resolve()) if csv_metrics_path.exists() else None
        ),
        seed=config.seed,
        device=device,
    )
    return TrainingOutput(model=best_model, result=result)


def train_tiny_cnn(
    prepared: PreparedData,
    config: TrainConfig,
    checkpoint_path: str | Path,
) -> TrainingOutput:
    """Compatibility alias; model family now comes from ``TrainConfig``."""
    return train_model(prepared, config, checkpoint_path)


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def inverse_frequency_class_weights(targets: Any, n_classes: int) -> list[float]:
    import numpy as np

    counts = np.bincount(targets, minlength=n_classes).astype("float64")
    inverse = counts.sum() / np.maximum(counts, 1.0)
    inverse /= inverse.mean()
    return inverse.astype("float32").tolist()


def tiny_cnn(in_channels: int, n_classes: int, hidden: int) -> Any:
    from lightning_components import build_network

    return build_network(
        "tiny_cnn",
        in_channels=in_channels,
        n_classes=n_classes,
        hidden=hidden,
        depth=2,
        dropout=0.0,
    )


def make_loader(
    prepared: PreparedData,
    *,
    split: str,
    batch_size: int,
    shuffle: bool,
    seed: int,
    corruption_sigma: float | None = None,
) -> Any:
    import torch
    from torch.utils.data import DataLoader, Dataset

    if split not in {"train", "val", "test"}:
        raise ValueError(f"unknown split: {split}")
    images = prepared.bundle.images[split]
    targets = prepared.bundle.targets[split]
    augmentations = prepared.augmentations if split == "train" else ()

    class ArrayDataset(Dataset):
        def __len__(self) -> int:
            return len(targets) * (1 + len(augmentations))

        def __getitem__(self, index: int) -> tuple[Any, Any]:
            import numpy as np

            base_index = index % len(targets)
            variant = index // len(targets)
            image = images[base_index]
            if variant:
                image = apply_augmentation(image, augmentations[variant - 1])
            array = np.asarray(image, dtype="float32") / 255.0
            if array.ndim == 2:
                array = array[..., None]
            array = np.ascontiguousarray(array.transpose(2, 0, 1))
            tensor = torch.from_numpy(array)
            if corruption_sigma is not None:
                generator = torch.Generator().manual_seed(seed + base_index)
                noise = torch.randn(tensor.shape, generator=generator)
                tensor = torch.clamp(tensor + noise * corruption_sigma, 0.0, 1.0)
            if prepared.normalization == "standardize":
                mean = torch.tensor(prepared.mean, dtype=tensor.dtype)[:, None, None]
                std = torch.tensor(prepared.std, dtype=tensor.dtype)[:, None, None]
                tensor = (tensor - mean) / std
            return tensor, torch.tensor(int(targets[base_index]), dtype=torch.long)

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        ArrayDataset(),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=31,
        generator=generator,
        drop_last=False,
    )


def apply_augmentation(image: Any, augmentation: str) -> Any:
    import numpy as np

    if augmentation == "hflip":
        return np.flip(image, axis=1)
    if augmentation == "vflip":
        return np.flip(image, axis=0)
    if augmentation == "rotate90":
        return np.rot90(image, k=1, axes=(0, 1))
    if augmentation == "rotate180":
        return np.rot90(image, k=2, axes=(0, 1))
    if augmentation == "brightness":
        return np.clip(np.asarray(image, dtype="float32") * 1.10, 0, 255).astype(
            image.dtype
        )
    if augmentation == "contrast":
        values = np.asarray(image, dtype="float32")
        channel_mean = values.mean(axis=(0, 1), keepdims=True)
        return np.clip((values - channel_mean) * 1.10 + channel_mean, 0, 255).astype(
            image.dtype
        )
    raise ValueError(f"augmentation is outside the guardrail set: {augmentation}")


def predict_probabilities(
    model: Any,
    prepared: PreparedData,
    split: str,
    batch_size: int,
    *,
    device: str,
    seed: int = 42,
    corruption_sigma: float | None = None,
) -> tuple[Any, Any]:
    probabilities, _, targets = predict_outputs(
        model,
        prepared,
        split,
        batch_size,
        device=device,
        seed=seed,
        corruption_sigma=corruption_sigma,
    )
    return probabilities, targets


def predict_outputs(
    model: Any,
    prepared: PreparedData,
    split: str,
    batch_size: int,
    *,
    device: str,
    seed: int = 42,
    corruption_sigma: float | None = None,
) -> tuple[Any, Any, Any]:
    import numpy as np
    import torch

    loader = make_loader(
        prepared,
        split=split,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        corruption_sigma=corruption_sigma,
    )
    probabilities: list[Any] = []
    logits_output: list[Any] = []
    targets: list[Any] = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            logits = model(inputs.to(device))
            logits_output.append(logits.cpu().numpy())
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            targets.append(labels.numpy())
    return (
        np.concatenate(probabilities),
        np.concatenate(logits_output),
        np.concatenate(targets),
    )


def save_prediction_arrays(path: str | Path, **arrays: Any) -> str:
    """Persist auditable per-sample outputs and return their SHA-256."""
    import numpy as np

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, target)
    return sha256_file(target)


def evaluate_probabilities(
    probabilities: Any,
    targets: Any,
    label_names: tuple[str, ...],
    *,
    split: str = "test",
) -> EvalReport:
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    labels = list(range(len(label_names)))
    prediction = np.asarray(probabilities).argmax(axis=1)
    y = np.asarray(targets)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, prediction, labels=labels, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y, prediction, average="macro", zero_division=0
    )
    auc: float | None
    try:
        one_hot = np.eye(len(label_names), dtype="float64")[y]
        auc = float(
            roc_auc_score(one_hot, probabilities, average="macro", multi_class="ovr")
        )
    except ValueError:
        auc = None
    return EvalReport(
        split=split,
        n_samples=len(y),
        accuracy=round(float(accuracy_score(y, prediction)), 6),
        balanced_accuracy=round(float(balanced_accuracy_score(y, prediction)), 6),
        macro_precision=round(float(macro_precision), 6),
        macro_recall=round(float(macro_recall), 6),
        macro_f1=round(float(macro_f1), 6),
        roc_auc_ovr_macro=round(auc, 6) if auc is not None else None,
        per_class=[
            ClassMetrics(
                label=index,
                name=label_names[index],
                support=int(support[index]),
                precision=round(float(precision[index]), 6),
                recall=round(float(recall[index]), 6),
                f1=round(float(f1[index]), 6),
            )
            for index in labels
        ],
        confusion_matrix=confusion_matrix(y, prediction, labels=labels)
        .astype(int)
        .tolist(),
    )

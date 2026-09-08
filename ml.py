"""Deterministic, shared ML primitives for agentic and baseline experiments."""
from __future__ import annotations

import copy
import hashlib
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from contracts import (
    ClassMetrics,
    EpochMetrics,
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
        split: data_class(split=split, **kwargs)
        for split in ("train", "val", "test")
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


def channel_statistics(images: Any, batch_size: int = 2048) -> tuple[list[float], list[float]]:
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


def train_tiny_cnn(
    prepared: PreparedData,
    config: TrainConfig,
    checkpoint_path: str | Path,
) -> TrainingOutput:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    import torch.nn as nn

    seed_everything(config.seed)
    device = resolve_device(config.device)
    model = tiny_cnn(
        prepared.bundle.n_channels, prepared.bundle.n_classes, config.hidden
    ).to(device)
    weights = None
    if config.class_weighting:
        counts = np.bincount(
            prepared.bundle.targets["train"], minlength=prepared.bundle.n_classes
        ).astype("float64")
        inverse = counts.sum() / np.maximum(counts, 1.0)
        inverse /= inverse.mean()
        weights = torch.tensor(inverse, dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    train_loader = make_loader(
        prepared,
        split="train",
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
    )

    history: list[EpochMetrics] = []
    best_accuracy = -1.0
    best_epoch = 1
    best_state: dict[str, Any] | None = None
    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum = 0.0
        observations = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
            batch_n = int(targets.shape[0])
            loss_sum += float(loss.detach().cpu()) * batch_n
            observations += batch_n
        val_probs, val_targets = predict_probabilities(
            model, prepared, "val", config.batch_size, device=device, seed=config.seed
        )
        val_accuracy = float((val_probs.argmax(axis=1) == val_targets).mean())
        epoch_loss = loss_sum / max(observations, 1)
        history.append(
            EpochMetrics(
                epoch=epoch,
                train_loss=round(epoch_loss, 6),
                val_accuracy=round(val_accuracy, 6),
            )
        )
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    target = Path(checkpoint_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "train_config": config.model_dump(mode="json"),
            "normalization": prepared.normalization,
            "augmentations": list(prepared.augmentations),
            "mean": list(prepared.mean),
            "std": list(prepared.std),
            "dataset": prepared.bundle.dataset,
        },
        target,
    )
    result = TrainResult(
        final_train_loss=history[-1].train_loss,
        final_val_accuracy=history[-1].val_accuracy,
        best_val_accuracy=round(best_accuracy, 6),
        best_epoch=best_epoch,
        epochs_completed=len(history),
        history=history,
        checkpoint_path=str(target),
        checkpoint_sha256=sha256_file(target),
        seed=config.seed,
        device=device,
    )
    return TrainingOutput(model=model, result=result)


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


def tiny_cnn(in_channels: int, n_classes: int, hidden: int) -> Any:
    import torch.nn as nn

    return nn.Sequential(
        nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(hidden, hidden * 2, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Linear(hidden * 2 * 4 * 4, n_classes),
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
        num_workers=0,
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
    targets: list[Any] = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            logits = model(inputs.to(device))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            targets.append(labels.numpy())
    return np.concatenate(probabilities), np.concatenate(targets)


def evaluate_probabilities(
    probabilities: Any,
    targets: Any,
    label_names: tuple[str, ...],
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

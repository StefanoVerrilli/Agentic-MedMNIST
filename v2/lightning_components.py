"""Lightning modules shared by the CLI and the agentic training backend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import Callback
from torch import nn

from contracts import (
    Augmentation,
    EpochMetrics,
    ModelFamily,
    OptimizerName,
    RepresentationDecision,
    SchedulerName,
)
from ml import PreparedData, load_pathmnist, make_loader, prepare_data


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.body(inputs) + self.skip(inputs))


class VisionTransformer(nn.Module):
    """Small ViT/CCT for 28x28 inputs, trained without external weights."""

    def __init__(self, *, in_channels: int, n_classes: int, hidden: int,
                 depth: int, dropout: float, convolutional_tokenizer: bool):
        super().__init__()
        if convolutional_tokenizer:
            mid = max(16, hidden // 2)
            self.tokenizer = nn.Sequential(
                nn.Conv2d(in_channels, mid, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(mid), nn.GELU(),
                nn.Conv2d(mid, hidden, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(hidden), nn.GELU(),
            )
        else:
            # 28 / 4 = 7: a compact 7x7 patch grid (49 visual tokens).
            self.tokenizer = nn.Conv2d(in_channels, hidden, 4, stride=4)
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden))
        self.position = nn.Parameter(torch.zeros(1, 50, hidden))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_classes))
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(inputs).flatten(2).transpose(1, 2)
        cls = self.class_token.expand(inputs.shape[0], -1, -1)
        tokens = torch.cat((cls, tokens), dim=1)
        tokens = tokens + self.position[:, : tokens.shape[1]]
        return self.head(self.norm(self.encoder(tokens)[:, 0]))


def build_network(
    model_family: str,
    *,
    in_channels: int,
    n_classes: int,
    hidden: int,
    depth: int,
    dropout: float,
) -> nn.Module:
    if model_family == "tiny_cnn":
        layers: list[nn.Module] = []
        channels = in_channels
        width = hidden
        for block in range(depth):
            layers.extend(
                [
                    nn.Conv2d(channels, width, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(width),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                ]
            )
            channels = width
            width = min(width * 2, 256)
        layers.extend(
            [
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(channels, n_classes),
            ]
        )
        return nn.Sequential(*layers)

    if model_family == "residual_cnn":
        layers = [
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        ]
        channels = hidden
        for block in range(depth):
            out_channels = min(hidden * (2 ** min(block, 2)), 256)
            layers.append(
                ResidualBlock(channels, out_channels, stride=1 if block == 0 else 2)
            )
            channels = out_channels
        layers.extend(
            [
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(channels, n_classes),
            ]
        )
        return nn.Sequential(*layers)

    if model_family == "resnet18":
        return _small_resnet18(
            in_channels=in_channels,
            n_classes=n_classes,
            base_width=hidden,
            dropout=dropout,
        )
    if model_family in {"vision_transformer", "compact_transformer"}:
        return VisionTransformer(
            in_channels=in_channels,
            n_classes=n_classes,
            hidden=hidden,
            depth=depth,
            dropout=dropout,
            convolutional_tokenizer=model_family == "compact_transformer",
        )
    raise ValueError(f"unknown model family: {model_family}")


class SmallResNet18(nn.Module):
    """CIFAR-style ResNet-18 suitable for 28x28 PathMNIST patches."""

    def __init__(
        self, *, in_channels: int, n_classes: int, base_width: int, dropout: float
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )
        widths = [base_width, base_width * 2, base_width * 4, base_width * 8]
        blocks: list[nn.Module] = []
        current = base_width
        for stage, width in enumerate(widths):
            blocks.append(ResidualBlock(current, width, stride=1 if stage == 0 else 2))
            blocks.append(ResidualBlock(width, width))
            current = width
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(current, n_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(inputs)))


def _small_resnet18(
    *, in_channels: int, n_classes: int, base_width: int, dropout: float
) -> SmallResNet18:
    return SmallResNet18(
        in_channels=in_channels,
        n_classes=n_classes,
        base_width=base_width,
        dropout=dropout,
    )


class PathMNISTLitModule(pl.LightningModule):
    def __init__(
        self,
        model_family: ModelFamily = "residual_cnn",
        in_channels: int = 3,
        n_classes: int = 9,
        hidden: int = 32,
        depth: int = 3,
        dropout: float = 0.1,
        optimizer: OptimizerName = "adamw",
        scheduler: SchedulerName = "cosine",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.05,
        class_weights: list[float] | None = None,
    ):
        super().__init__()
        if in_channels < 1 or n_classes < 2:
            raise ValueError("in_channels must be positive and n_classes at least 2")
        if hidden not in {16, 24, 32, 48, 64}:
            raise ValueError("hidden must be one of 16, 24, 32, 48 or 64")
        if depth not in {2, 3, 4}:
            raise ValueError("depth must be one of 2, 3 or 4")
        if not 0.0 <= dropout <= 0.5:
            raise ValueError("dropout must be between 0.0 and 0.5")
        if not 1e-5 <= lr <= 1e-2:
            raise ValueError("lr must be between 1e-5 and 1e-2")
        if not 0.0 <= weight_decay <= 0.1:
            raise ValueError("weight_decay must be between 0.0 and 0.1")
        if not 0.0 <= label_smoothing <= 0.2:
            raise ValueError("label_smoothing must be between 0.0 and 0.2")
        if class_weights is not None and len(class_weights) != n_classes:
            raise ValueError("class_weights must contain one value per class")
        if class_weights is not None and any(value <= 0 for value in class_weights):
            raise ValueError("class_weights must all be positive")
        self.save_hyperparameters()
        self.network = build_network(
            model_family,
            in_channels=in_channels,
            n_classes=n_classes,
            hidden=hidden,
            depth=depth,
            dropout=dropout,
        )
        weight = (
            torch.tensor(class_weights, dtype=torch.float32)
            if class_weights is not None
            else None
        )
        self.loss_fn = nn.CrossEntropyLoss(
            weight=weight, label_smoothing=label_smoothing
        )
        self.register_buffer(
            "_val_confusion",
            torch.zeros(n_classes, n_classes, dtype=torch.long),
            persistent=False,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        inputs, targets = batch
        loss = self.loss_fn(self(inputs), targets)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=targets.shape[0],
        )
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        inputs, targets = batch
        logits = self(inputs)
        loss = self.loss_fn(logits, targets)
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == targets).float().mean()
        n_classes = int(self.hparams.n_classes)
        encoded = targets.to(torch.long) * n_classes + predictions.to(torch.long)
        counts = torch.bincount(encoded, minlength=n_classes * n_classes)
        self._val_confusion += counts.reshape(n_classes, n_classes)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=targets.shape[0],
        )
        self.log(
            "val_accuracy",
            accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=targets.shape[0],
        )

    def on_validation_epoch_start(self) -> None:
        self._val_confusion.zero_()

    def on_validation_epoch_end(self) -> None:
        matrix = self._val_confusion.to(dtype=torch.float32)
        true_positive = torch.diag(matrix)
        predicted = matrix.sum(dim=0)
        support = matrix.sum(dim=1)
        precision = true_positive / predicted.clamp_min(1.0)
        recall = true_positive / support.clamp_min(1.0)
        denominator = precision + recall
        f1 = torch.where(
            denominator > 0,
            2.0 * precision * recall / denominator,
            torch.zeros_like(denominator),
        )
        observed = support > 0
        macro_f1 = f1[observed].mean() if observed.any() else f1.mean()
        self.log(
            "val_macro_f1",
            macro_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        inputs, targets = batch
        logits = self(inputs)
        accuracy = (logits.argmax(dim=1) == targets).float().mean()
        self.log(
            "test_accuracy",
            accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=targets.shape[0],
        )

    def configure_optimizers(self):
        name = self.hparams.optimizer
        if name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        elif name == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        elif name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.lr,
                momentum=0.9,
                nesterov=True,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            raise ValueError(f"unknown optimizer: {name}")

        scheduler_name = self.hparams.scheduler
        if scheduler_name == "none":
            return optimizer
        if scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, int(self.trainer.max_epochs))
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        if scheduler_name == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=1
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val_accuracy"},
            }
        raise ValueError(f"unknown scheduler: {scheduler_name}")


class PreparedDataModule(pl.LightningDataModule):
    """In-memory DataModule used by the agentic search and final training."""

    def __init__(self, prepared: PreparedData, *, batch_size: int, seed: int):
        super().__init__()
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        self.prepared = prepared
        self.batch_size = batch_size
        self.seed = seed

    def train_dataloader(self):
        return make_loader(
            self.prepared,
            split="train",
            batch_size=self.batch_size,
            shuffle=True,
            seed=self.seed,
        )

    def val_dataloader(self):
        return make_loader(
            self.prepared,
            split="val",
            batch_size=self.batch_size,
            shuffle=False,
            seed=self.seed,
        )

    def test_dataloader(self):
        return make_loader(
            self.prepared,
            split="test",
            batch_size=self.batch_size,
            shuffle=False,
            seed=self.seed,
        )


class PathMNISTDataModule(pl.LightningDataModule):
    """Serializable DataModule exposed by ``lightning_cli.py``."""

    def __init__(
        self,
        data_root: str | None = None,
        batch_size: int = 128,
        seed: int = 42,
        normalization: Literal["unit", "standardize"] = "standardize",
        augmentations: list[Augmentation] | None = None,
        train_limit: int | None = None,
        val_limit: int | None = None,
        test_limit: int | None = None,
        download: bool = True,
        num_workers: int = 0,
    ):
        super().__init__()
        if batch_size not in {32, 64, 128, 256}:
            raise ValueError("batch_size must be one of 32, 64, 128 or 256")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        selected_augmentations = augmentations or []
        if len(set(selected_augmentations)) != len(selected_augmentations):
            raise ValueError("augmentations must be unique")
        for name, limit in {
            "train_limit": train_limit,
            "val_limit": val_limit,
            "test_limit": test_limit,
        }.items():
            if limit is not None and limit != 0 and limit < 9:
                raise ValueError(f"{name} must be 0, at least 9, or null")
        if num_workers != 0:
            raise ValueError(
                "num_workers is fixed to 0 for strict deterministic array loading"
            )
        self.save_hyperparameters()
        self.prepared: PreparedData | None = None

    def setup(self, stage: str | None = None) -> None:
        if self.prepared is not None:
            return
        decision = RepresentationDecision(
            normalization=self.hparams.normalization,
            augmentations=list(self.hparams.augmentations or []),
            rationale="Configuration supplied through LightningCLI.",
        )
        bundle = load_pathmnist(
            seed=int(self.hparams.seed),
            train_limit=self.hparams.train_limit,
            val_limit=self.hparams.val_limit,
            test_limit=self.hparams.test_limit,
            data_root=self.hparams.data_root,
            download=bool(self.hparams.download),
        )
        self.prepared = prepare_data(bundle, decision)

    def _loader(self, split: str, *, shuffle: bool):
        if self.prepared is None:
            raise RuntimeError("PathMNISTDataModule.setup() has not run")
        loader = make_loader(
            self.prepared,
            split=split,
            batch_size=int(self.hparams.batch_size),
            shuffle=shuffle,
            seed=int(self.hparams.seed),
        )
        return loader

    def train_dataloader(self):
        return self._loader("train", shuffle=True)

    def val_dataloader(self):
        return self._loader("val", shuffle=False)

    def test_dataloader(self):
        return self._loader("test", shuffle=False)


class MetricsHistory(Callback):
    """Collect the epoch evidence required by the immutable TrainResult."""

    def __init__(
        self,
        *,
        echo: bool = True,
        label: str = "training",
        jsonl_path: str | Path | None = None,
    ):
        super().__init__()
        self.history: list[EpochMetrics] = []
        self.echo = echo
        self.label = label
        self.jsonl_path = Path(jsonl_path) if jsonl_path is not None else None

    def on_fit_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self.jsonl_path is not None and trainer.is_global_zero:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            self.jsonl_path.write_text("", encoding="utf-8")

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        if "train_loss" not in metrics or "val_accuracy" not in metrics:
            return
        train_loss = float(metrics["train_loss"].detach().cpu())
        val_loss_metric = metrics.get("val_loss")
        val_loss = (
            float(val_loss_metric.detach().cpu()) if val_loss_metric is not None else None
        )
        val_accuracy = float(metrics["val_accuracy"].detach().cpu())
        val_f1_metric = metrics.get("val_macro_f1")
        val_macro_f1 = (
            float(val_f1_metric.detach().cpu()) if val_f1_metric is not None else None
        )
        learning_rate = None
        if trainer.optimizers:
            learning_rate = float(trainer.optimizers[0].param_groups[0]["lr"])
        row = EpochMetrics(
            epoch=int(trainer.current_epoch) + 1,
            train_loss=round(train_loss, 6),
            val_loss=round(val_loss, 6) if val_loss is not None else None,
            val_accuracy=round(val_accuracy, 6),
            val_macro_f1=(
                round(val_macro_f1, 6) if val_macro_f1 is not None else None
            ),
            learning_rate=(
                round(learning_rate, 10) if learning_rate is not None else None
            ),
        )
        self.history.append(row)
        if not trainer.is_global_zero:
            return
        payload = row.model_dump(mode="json")
        if self.jsonl_path is not None:
            with self.jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
        if self.echo:
            values = [
                f"epoch={row.epoch:03d}/{int(trainer.max_epochs):03d}",
                f"train_loss={row.train_loss:.6f}",
            ]
            if row.val_loss is not None:
                values.append(f"val_loss={row.val_loss:.6f}")
            values.append(f"val_acc={row.val_accuracy:.6f}")
            if row.val_macro_f1 is not None:
                values.append(f"val_macro_f1={row.val_macro_f1:.6f}")
            if row.learning_rate is not None:
                values.append(f"lr={row.learning_rate:.8g}")
            print(f"[lightning:{self.label}] " + " ".join(values), flush=True)

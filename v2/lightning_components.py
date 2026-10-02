"""Lightning modules shared by the CLI and the agentic training backend."""

from __future__ import annotations

import json
import math
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
    PoolingName,
    PositionalEncodingName,
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
                 depth: int, dropout: float, convolutional_tokenizer: bool,
                 patch_size: int, num_heads: int, mlp_ratio: int,
                 pooling: PoolingName, positional_encoding: PositionalEncodingName,
                 tokenizer_layers: int):
        super().__init__()
        if convolutional_tokenizer:
            layers: list[nn.Module] = []
            channels = in_channels
            for index in range(tokenizer_layers):
                out_channels = hidden if index == tokenizer_layers - 1 else max(16, hidden // 2)
                layers.extend([
                    nn.Conv2d(channels, out_channels, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels), nn.GELU(),
                ])
                channels = out_channels
            self.tokenizer = nn.Sequential(*layers)
            grid = math.ceil(28 / (2 ** tokenizer_layers))
        else:
            self.tokenizer = nn.Conv2d(in_channels, hidden, patch_size, stride=patch_size)
            grid = 28 // patch_size
        self.pooling = pooling
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden))
        token_count = grid * grid + 1
        position = _sinusoidal_position(token_count, hidden)
        if positional_encoding == "learned":
            self.position = nn.Parameter(torch.zeros(1, token_count, hidden))
            nn.init.trunc_normal_(self.position, std=0.02)
        else:
            self.register_buffer("position", position, persistent=True)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=num_heads,
            dim_feedforward=hidden * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(hidden)
        self.attention_pool = nn.Linear(hidden, 1) if pooling == "attention" else None
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_classes))
        nn.init.trunc_normal_(self.class_token, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(inputs).flatten(2).transpose(1, 2)
        cls = self.class_token.expand(inputs.shape[0], -1, -1)
        tokens = torch.cat((cls, tokens), dim=1)
        tokens = tokens + self.position[:, : tokens.shape[1]]
        encoded = self.norm(self.encoder(tokens))
        if self.pooling == "cls":
            pooled = encoded[:, 0]
        elif self.pooling == "mean":
            pooled = encoded[:, 1:].mean(dim=1)
        else:
            weights = torch.softmax(self.attention_pool(encoded[:, 1:]), dim=1)
            pooled = (weights * encoded[:, 1:]).sum(dim=1)
        return self.head(pooled)


def _sinusoidal_position(tokens: int, hidden: int) -> torch.Tensor:
    positions = torch.arange(tokens, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, hidden, 2, dtype=torch.float32)
        * (-math.log(10000.0) / hidden)
    )
    encoding = torch.zeros(1, tokens, hidden)
    encoding[0, :, 0::2] = torch.sin(positions * frequencies)
    encoding[0, :, 1::2] = torch.cos(positions * frequencies)
    return encoding


def build_network(
    model_family: str,
    *,
    in_channels: int,
    n_classes: int,
    hidden: int,
    depth: int,
    dropout: float,
    patch_size: int = 4,
    num_heads: int = 4,
    mlp_ratio: int = 4,
    pooling: PoolingName = "cls",
    positional_encoding: PositionalEncodingName = "learned",
    tokenizer_layers: int = 2,
    channel_cap: int = 256,
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
            width = min(width * 2, channel_cap)
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
            out_channels = min(hidden * (2 ** min(block, 2)), channel_cap)
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
            patch_size=patch_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            pooling=pooling,
            positional_encoding=positional_encoding,
            tokenizer_layers=tokenizer_layers,
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
        patch_size: int = 4,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        pooling: PoolingName = "cls",
        positional_encoding: PositionalEncodingName = "learned",
        tokenizer_layers: int = 2,
        optimizer: OptimizerName = "adamw",
        scheduler: SchedulerName = "cosine",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.05,
        class_weights: list[float] | None = None,
        channel_cap: int = 256,
        momentum: float = 0.9,
        nesterov: bool = True,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        optimizer_eps: float = 1e-8,
        one_cycle_pct_start: float = 0.1,
        plateau_factor: float = 0.5,
        plateau_patience: int = 1,
        cosine_eta_min: float = 0.0,
    ):
        super().__init__()
        if in_channels < 1 or n_classes < 2:
            raise ValueError("in_channels must be positive and n_classes at least 2")
        from contracts import ExecutionOptions
        ExecutionOptions(channel_cap=channel_cap, momentum=momentum, nesterov=nesterov,
                         adam_beta1=adam_beta1, adam_beta2=adam_beta2, optimizer_eps=optimizer_eps,
                         one_cycle_pct_start=one_cycle_pct_start, plateau_factor=plateau_factor,
                         plateau_patience=plateau_patience, cosine_eta_min=cosine_eta_min)
        if optimizer == "sgd" and nesterov and momentum <= 0:
            raise ValueError("Nesterov requires positive momentum")
        if scheduler == "cosine" and cosine_eta_min > lr:
            raise ValueError("cosine_eta_min cannot exceed lr")
        if not 8 <= hidden <= 1024 or hidden % 8:
            raise ValueError("hidden must be a multiple of 8 between 8 and 1024")
        if not 1 <= depth <= 48 or (model_family == "tiny_cnn" and depth > 4):
            raise ValueError("depth must be 1..48 (1..4 for tiny_cnn pooling)")
        if model_family == "resnet18" and hidden > 256:
            raise ValueError("resnet18 base width cannot exceed 256")
        if model_family == "residual_cnn" and hidden > 512:
            raise ValueError("residual_cnn base width cannot exceed 512")
        if not 0.0 <= dropout <= 0.95:
            raise ValueError("dropout must be between 0.0 and 0.95")
        if not 1e-7 <= lr <= 1.0:
            raise ValueError("lr must be between 1e-7 and 1.0")
        if model_family in {"vision_transformer", "compact_transformer"}:
            if not 1 <= num_heads <= 32 or hidden % num_heads:
                raise ValueError("num_heads must be 1..32 and divide hidden")
            if not 1 <= mlp_ratio <= 16 or not 1 <= tokenizer_layers <= 5:
                raise ValueError("invalid MLP ratio or tokenizer depth")
            if patch_size not in {1, 2, 4, 7, 14, 28}:
                raise ValueError("patch size must divide the 28x28 input")
        if not 0.0 <= weight_decay <= 1.0:
            raise ValueError("weight_decay must be between 0.0 and 1.0")
        if not 0.0 <= label_smoothing <= 0.5:
            raise ValueError("label_smoothing must be between 0.0 and 0.5")
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
            patch_size=patch_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            pooling=pooling,
            positional_encoding=positional_encoding,
            tokenizer_layers=tokenizer_layers,
            channel_cap=channel_cap,
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
                betas=(self.hparams.adam_beta1, self.hparams.adam_beta2),
                eps=self.hparams.optimizer_eps,
            )
        elif name == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
                betas=(self.hparams.adam_beta1, self.hparams.adam_beta2),
                eps=self.hparams.optimizer_eps,
            )
        elif name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.lr,
                momentum=self.hparams.momentum,
                nesterov=self.hparams.nesterov,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            raise ValueError(f"unknown optimizer: {name}")

        scheduler_name = self.hparams.scheduler
        if scheduler_name == "none":
            return optimizer
        if scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, int(self.trainer.max_epochs)), eta_min=self.hparams.cosine_eta_min
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        if scheduler_name == "one_cycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=float(self.hparams.lr),
                total_steps=int(self.trainer.estimated_stepping_batches),
                pct_start=self.hparams.one_cycle_pct_start,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }
        if scheduler_name == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=self.hparams.plateau_factor, patience=self.hparams.plateau_patience
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
        if batch_size not in {16, 32, 64, 128, 256}:
            raise ValueError("batch_size must be one of 16, 32, 64, 128 or 256")
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

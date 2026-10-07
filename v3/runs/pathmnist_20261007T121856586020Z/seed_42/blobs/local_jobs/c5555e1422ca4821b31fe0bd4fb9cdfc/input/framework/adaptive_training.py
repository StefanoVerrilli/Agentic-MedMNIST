"""Trusted, resumable training helper used only inside the Linux worker.

State is saved at epoch boundaries. The best inference checkpoint and latest
optimizer/RNG state have separate files and separate purposes.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import random
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from autonomous import continuation_identity


def data_identity(data):
    digest = hashlib.sha256()
    for key in sorted(data):
        value = np.ascontiguousarray(data[key])
        digest.update(f"{key}:{value.shape}:{value.dtype}".encode())
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def make_optimizer(config, model):
    if config["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"],
                               momentum=config["momentum"], nesterov=config["nesterov"])
    factory = torch.optim.AdamW if config["optimizer"] == "adamw" else torch.optim.Adam
    return factory(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"],
                   betas=(config["adam_beta1"], config["adam_beta2"]), eps=config["optimizer_eps"])


def make_scheduler(config, optimizer, steps_per_epoch, horizon):
    if config["scheduler"] == "none":
        return None
    if config["scheduler"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=horizon, eta_min=config["cosine_eta_min"])
    if config["scheduler"] == "one_cycle":
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config["lr"],
            total_steps=horizon * steps_per_epoch, pct_start=config["one_cycle_pct_start"])
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min" if config["early_stopping_monitor"] == "val_loss" else "max",
        factor=config["plateau_factor"], patience=config["plateau_patience"])


def fit_adaptive(context, model, *, loss_fn=None, optimizer=None, scheduler=None):
    from worker_runtime import batched_logits
    from ml import evaluate_probabilities
    config, data = context["config"], context["data"]
    start, target = context.get("start_epoch", 0), context["epochs"]
    if not 0 <= start < target:
        raise ValueError("training target must exceed completed epochs")
    device = config["device"]
    model = model.to(device)
    if loss_fn is None:
        weights = None
        if config["class_weighting"]:
            counts = np.bincount(data["train_targets"], minlength=9)
            if np.any(counts == 0):
                raise ValueError("class weighting requires all nine training classes")
            weights = torch.as_tensor(counts.sum() / (9 * counts), dtype=torch.float32, device=device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=config["label_smoothing"])
    optimizer = optimizer if optimizer is not None else make_optimizer(config, model)
    generator = torch.Generator().manual_seed(config["seed"])
    loader = DataLoader(TensorDataset(torch.from_numpy(data["train_images"]), torch.from_numpy(data["train_targets"])),
        batch_size=config["batch_size"], shuffle=True, generator=generator, num_workers=0)
    identity, fingerprint = continuation_identity(config), data_identity(data)
    state = None
    if context.get("resume"):
        state = torch.load(context["resume"], map_location="cpu", weights_only=False)
        if (state.get("format") != "agent-resume-v1" or state.get("identity") != identity or
                state.get("data_identity") != fingerprint or state.get("epochs_completed") != start or state.get("stopped")):
            raise ValueError("continuation state/configuration/data are incompatible")
    elif start:
        raise ValueError("continuation requires a latest-state checkpoint")
    initial_horizon = state["initial_horizon"] if state else target
    scheduler = scheduler if scheduler is not None else make_scheduler(config, optimizer, len(loader), initial_horizon)
    history, best, best_epoch, stale = [], None, 0, 0
    output = Path(context["output"])
    if state:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if (scheduler is None) != (state["scheduler"] is None):
            raise ValueError("continuation scheduler changed")
        if scheduler is not None:
            scheduler.load_state_dict(state["scheduler"])
        if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR) and target * len(loader) > scheduler.total_steps:
            raise ValueError("OneCycle schedule exhausted: choose a new trial rather than resetting it")
        generator.set_state(state["loader_rng"])
        history, best, best_epoch, stale = state["history"], state["best"], state["best_epoch"], state["stale"]
        shutil.copyfile(context["checkpoint"], output / "model.ckpt")
        restore_rng(state["rng"])
    stopped = False
    for epoch in range(start + 1, target + 1):
        model.train()
        total_loss, samples = 0.0, 0
        for images, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images.to(device)), targets.to(device))
            if not torch.isfinite(loss):
                raise ValueError("training loss is not finite")
            loss.backward()
            if config["gradient_clip_val"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_val"])
            optimizer.step()
            if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()
            total_loss += float(loss.detach()) * len(targets)
            samples += len(targets)
        logits = batched_logits(model, data["val_images"], config)
        probabilities = torch.softmax(torch.from_numpy(logits), 1).numpy()
        report = evaluate_probabilities(probabilities, data["val_targets"], tuple(map(str, range(9))), split="val")
        val_loss = float(torch.nn.functional.cross_entropy(torch.from_numpy(logits), torch.from_numpy(data["val_targets"])))
        row = {"epoch": epoch, "train_loss": total_loss / samples, "val_loss": val_loss,
               "val_accuracy": report.accuracy, "val_macro_f1": report.macro_f1,
               "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(row)
        monitor = config["early_stopping_monitor"]
        score = -row[monitor] if monitor == "val_loss" else row[monitor]
        if best is None or score > best + config["early_stopping_min_delta"]:
            best, best_epoch, stale = score, epoch, 0
            torch.save(model.state_dict(), output / "model.ckpt")
        else:
            stale += 1
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(row[monitor])
        elif scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()
        stopped = bool(config["early_stopping_patience"] and stale >= config["early_stopping_patience"])
        state = {"format": "agent-resume-v1", "identity": identity, "data_identity": fingerprint,
            "initial_horizon": initial_horizon, "epochs_completed": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict() if scheduler else None,
            "rng": capture_rng(), "loader_rng": generator.get_state(), "history": history,
            "best": best, "best_epoch": best_epoch, "stale": stale, "stopped": stopped}
        torch.save(state, output / "resume.pt")
        if stopped:
            break
    return {"final_train_loss": history[-1]["train_loss"], "final_val_accuracy": history[-1]["val_accuracy"],
            "best_val_accuracy": history[best_epoch - 1]["val_accuracy"], "best_epoch": best_epoch,
            "epochs_completed": len(history), "history": history,
            "stop_reason": "early_stopping" if stopped else "segment_complete"}


def validate_resume(context, result):
    """Only the isolated worker deserializes custom training state."""
    from contracts import TrainResult
    state = torch.load(Path(context["output"]) / "resume.pt", map_location="cpu", weights_only=False)
    required = {"format", "identity", "data_identity", "epochs_completed", "model", "optimizer", "scheduler", "rng",
                "loader_rng", "history", "best", "best_epoch", "stale", "stopped"}
    if not isinstance(state, dict) or not required <= state.keys():
        raise ValueError("incomplete continuation state")
    if state["format"] != "agent-resume-v1" or state["identity"] != continuation_identity(context["config"]):
        raise ValueError("continuation state identity mismatch")
    if state["data_identity"] != data_identity(context["data"]):
        raise ValueError("continuation state dataset mismatch")
    if not isinstance(state["optimizer"], dict) or not isinstance(state["model"], dict) or not isinstance(state["loader_rng"], torch.Tensor):
        raise ValueError("invalid model/optimizer/loader continuation state")
    if not isinstance(state["rng"], dict) or not {"python", "numpy", "torch", "cuda"} <= state["rng"].keys():
        raise ValueError("incomplete random generator state")
    # Validate history and metrics before exporting any purportedly successful segment.
    TrainResult.model_validate({**result, "checkpoint_path": "model.ckpt", "checkpoint_sha256": "0" * 64,
        "seed": context["config"]["seed"], "device": context["config"]["device"]})
    if state["epochs_completed"] != result["epochs_completed"] or state["history"] != result["history"]:
        raise ValueError("continuation state and training result disagree")
    if state["best_epoch"] != result["best_epoch"] or bool(state["stopped"]) != (result["stop_reason"] == "early_stopping"):
        raise ValueError("continuation stopping state disagrees with result")

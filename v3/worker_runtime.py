"""Local subprocess entrypoint and optional run-scoped training helpers."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

from contracts import TrainConfig


def fit_model(context: dict, model, *, loss_fn=None, optimizer=None, scheduler=None) -> dict:
    """Optional helper; an experiment may replace this entire training loop."""
    if context["config"].get("execution_mode") == "agent_autonomous":
        from adaptive_training import fit_adaptive
        return fit_adaptive(context, model, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler)
    from torch.utils.data import DataLoader, TensorDataset
    from ml import evaluate_probabilities
    config = context["config"]
    data = context["data"]
    device = config["device"]
    model = model.to(device)
    loss_fn = loss_fn or torch.nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                               weight_decay=config["weight_decay"])
    generator = torch.Generator().manual_seed(config["seed"])
    loader = DataLoader(TensorDataset(torch.from_numpy(data["train_images"]),
                                     torch.from_numpy(data["train_targets"])),
                        batch_size=config["batch_size"], shuffle=True, generator=generator)
    history, best, best_epoch, stale = [], None, 1, 0
    output = Path(context["output"])
    labels = tuple(str(index) for index in range(9))
    for epoch in range(1, context["epochs"] + 1):
        model.train()
        losses = []
        for images, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images.to(device)), targets.to(device))
            if not torch.isfinite(loss):
                raise ValueError("training loss is not finite")
            loss.backward()
            if config["gradient_clip_val"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_val"])
            optimizer.step()
            losses.append((float(loss.detach()), len(targets)))
        logits = batched_logits(model, data["val_images"], config)
        probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        report = evaluate_probabilities(probabilities, data["val_targets"], labels, split="val")
        val_loss = float(torch.nn.functional.cross_entropy(torch.from_numpy(logits),
                                                          torch.from_numpy(data["val_targets"])))
        row = {"epoch": epoch, "train_loss": sum(loss * size for loss, size in losses) / sum(size for _, size in losses),
               "val_loss": val_loss, "val_accuracy": report.accuracy,
               "val_macro_f1": report.macro_f1, "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(row)
        monitor = config["early_stopping_monitor"]
        score = -row[monitor] if monitor == "val_loss" else row[monitor]
        if best is None or score > best + config["early_stopping_min_delta"]:
            best, best_epoch, stale = score, epoch, 0
            torch.save(model.state_dict(), output / "model.ckpt")
        else:
            stale += 1
        print(json.dumps(row), flush=True)
        if config["early_stopping_patience"] and stale >= config["early_stopping_patience"]:
            break
    return {"final_train_loss": history[-1]["train_loss"],
            "final_val_accuracy": history[-1]["val_accuracy"],
            "best_val_accuracy": history[best_epoch - 1]["val_accuracy"],
            "best_epoch": best_epoch, "epochs_completed": len(history), "history": history}


def batched_logits(model, images: np.ndarray, config: dict) -> np.ndarray:
    model = model.to(config["device"]).eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(images), config["batch_size"]):
            batch = torch.from_numpy(images[start:start + config["batch_size"]]).to(config["device"])
            values.append(model(batch).detach().cpu().numpy())
    return np.concatenate(values)


def builtin_model(context):
    from lightning_components import build_network
    config = context["config"]
    return build_network(config["model_family"], in_channels=3, n_classes=9,
        **{key: config[key] for key in ("hidden", "depth", "dropout", "patch_size", "num_heads",
                                      "mlp_ratio", "pooling", "positional_encoding", "tokenizer_layers",
                                      "channel_cap", "scales")})


def builtin_train(context):
    # Native Lightning, with already normalized arrays and no dataset download.
    import lightning.pytorch as pl
    from lightning_components import PathMNISTLitModule, MetricsHistory
    from contracts import execution_options
    from torch.utils.data import DataLoader, TensorDataset
    config = TrainConfig.model_validate(context["config"])
    arguments = {key: getattr(config, key) for key in (
        "model_family", "hidden", "depth", "dropout", "patch_size", "num_heads", "mlp_ratio",
        "pooling", "positional_encoding", "tokenizer_layers", "optimizer", "scheduler", "lr",
        "weight_decay", "label_smoothing")}
    from ml import inverse_frequency_class_weights
    arguments["class_weights"] = (inverse_frequency_class_weights(context["data"]["train_targets"], 9)
                                  if config.class_weighting else None)
    model = PathMNISTLitModule(**execution_options(config), **arguments)
    output = Path(context["output"])
    history = MetricsHistory(jsonl_path=output / "training.jsonl", echo=True)
    callbacks = [history, pl.callbacks.ModelCheckpoint(dirpath=output, filename="model",
                  monitor=config.early_stopping_monitor,
                  mode="min" if config.early_stopping_monitor == "val_loss" else "max",
                  save_top_k=1)]
    if config.early_stopping_patience:
        callbacks.append(pl.callbacks.EarlyStopping(monitor=config.early_stopping_monitor,
            mode="min" if config.early_stopping_monitor == "val_loss" else "max",
            patience=config.early_stopping_patience, min_delta=config.early_stopping_min_delta))
    loaders = []
    for split in ("train", "val"):
        data = context["data"]
        loaders.append(DataLoader(TensorDataset(torch.from_numpy(data[split + "_images"]),
            torch.from_numpy(data[split + "_targets"])), batch_size=config.batch_size,
            shuffle=split == "train", generator=torch.Generator().manual_seed(config.seed)))
    trainer = pl.Trainer(max_epochs=context["epochs"], accelerator="gpu" if config.device == "cuda" else "cpu",
        devices=1, deterministic=True, gradient_clip_val=config.gradient_clip_val,
        logger=False, callbacks=callbacks, enable_progress_bar=False, num_sanity_val_steps=0)
    trainer.fit(model, *loaders)
    rows = [row.model_dump(mode="json") for row in history.history]
    best = max(rows, key=lambda row: row["val_accuracy"])
    return {"final_train_loss": rows[-1]["train_loss"], "final_val_accuracy": rows[-1]["val_accuracy"],
            "best_val_accuracy": best["val_accuracy"], "best_epoch": best["epoch"],
            "epochs_completed": len(rows), "history": rows}


def builtin_predict(context):
    from lightning_components import PathMNISTLitModule
    model = PathMNISTLitModule.load_from_checkpoint(context["checkpoint"], map_location="cpu", weights_only=False)
    return batched_logits(model, context["data"]["images"], context["config"])


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    job = parser.parse_args().job.resolve()
    inputs, output = job / "input", job / "output"
    request = json.loads((inputs / "request.json").read_text(encoding="utf-8"))
    config = TrainConfig.model_validate(request["config"]).model_dump(mode="json")
    from ml import seed_everything
    seed_everything(config["seed"])
    with np.load(inputs / "data.npz", allow_pickle=False) as stored:
        data = {name: stored[name] for name in stored.files}
    context = {"config": config, "parameters": (config.get("generated_bundle") or {}).get("parameters", {}),
               "data": data, "output": str(output), "checkpoint": str(inputs / "model.ckpt"),
               "epochs": min(request["epochs"], config["epochs"])}
    context.update(start_epoch=request.get("start_epoch", 0),
                   resume=str(inputs / "resume.pt") if request.get("resume") else None)
    if config.get("execution_mode") == "agent_autonomous":
        if request["operation"] == "train" and request["epochs"] != config["epochs"]:
            raise ValueError("autonomous request must preserve the agent-selected epoch target")
        context["epochs"] = request["epochs"]
    context["segment_epochs"] = context["epochs"] - context["start_epoch"]
    context["evidence"] = request.get("evidence", [])
    custom = config["model_family"] == "run_generated"
    if custom:
        sys.path.insert(0, str(inputs / "bundle"))
        spec = importlib.util.spec_from_file_location("run_experiment", inputs / "bundle" / "experiment.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        build, train, predict = module.build_model, module.train, module.predict
    else:
        build, train, predict = builtin_model, builtin_train, builtin_predict
    operation = request["operation"]
    if operation == "strategy":
        callback = getattr(module, "propose", None) if custom else None
        result = callback(context) if callback else []
        (output / "strategy.json").write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
    elif operation == "verify":
        inputs = torch.randn(4, 3, 28, 28, device=config["device"])
        model = build(context).to(config["device"])
        logits = model(inputs)
        if logits.shape != (4, 9) or not torch.isfinite(logits).all():
            raise ValueError("build_model must produce finite batch x 9 logits")
        torch.nn.functional.cross_entropy(logits, torch.arange(4, device=config["device"])).backward()
        if not any(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                   for parameter in model.parameters()):
            raise ValueError("model must support finite gradients")
        rng = np.random.default_rng(config["seed"])
        context["data"] = {"train_images": rng.normal(size=(9, 3, 28, 28)).astype("float32"),
                           "val_images": rng.normal(size=(9, 3, 28, 28)).astype("float32"),
                           "train_targets": np.arange(9), "val_targets": np.arange(9)}
        context["epochs"] = 1
        context.update(start_epoch=0, segment_epochs=1, resume=None)
        result = train(context)
        context["checkpoint"] = str(output / "model.ckpt")
        context["data"] = {"images": context["data"]["val_images"]}
        values = np.asarray(predict(context))
        if values.shape != (9, 9) or not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
            raise ValueError("checkpoint inference failed verification")
        (output / "result.json").write_text(json.dumps({"verified": True}, allow_nan=False), encoding="utf-8")
    elif operation == "train":
        if set(data) != {"train_images", "train_targets", "val_images", "val_targets"}:
            raise ValueError("training input must contain train and validation only")
        result = train(context)
        if config.get("execution_mode") == "agent_autonomous":
            from adaptive_training import validate_resume
            validate_resume(context, result)
        (output / "result.json").write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
        (output / "training.jsonl").write_text("".join(json.dumps(row, allow_nan=False) + "\n"
            for row in result["history"]), encoding="utf-8")
        if not (output / "model.ckpt").is_file():
            raise ValueError("train must save model.ckpt inside context['output']")
    elif operation == "infer":
        if set(data) != {"images"}:
            raise ValueError("inference receives images only, never targets")
        values = np.asarray(predict(context))
        if values.shape != (len(data["images"]), 9) or not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
            raise ValueError("predict must return finite N x 9 logits")
        np.save(output / "logits.npy", values, allow_pickle=False)
    else:
        raise ValueError("unsupported worker operation")


if __name__ == "__main__":
    main()

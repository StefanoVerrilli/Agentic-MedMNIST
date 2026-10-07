"""Exercise trusted worker helpers locally, without importing generated Python."""
from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from contracts import TrainConfig
from worker_runtime import builtin_train, builtin_predict, fit_model


class WorkerRuntimeTests(unittest.TestCase):
    def context(self, directory, family="tiny_cnn"):
        config = TrainConfig(model_family=family, lr=.001, epochs=1, hidden=16, depth=1,
            batch_size=9, weight_decay=0, class_weighting=False, seed=42,
            device="cpu", rationale="Trusted helper smoke test", source="test")
        rng = np.random.default_rng(42)
        return {"config": config.model_dump(mode="json"), "parameters": {},
                "data": {"train_images": rng.normal(size=(9, 3, 28, 28)).astype("float32"),
                         "val_images": rng.normal(size=(9, 3, 28, 28)).astype("float32"),
                         "train_targets": np.arange(9), "val_targets": np.arange(9)},
                "output": directory, "checkpoint": str(Path(directory) / "model.ckpt"), "epochs": 1}

    def test_native_lightning_worker_training_and_checkpoint_inference(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as directory:
                context = self.context(directory, "multi_scale_transformer")
                context["config"]["scales"] = [4, 7]
                result = builtin_train(context)
                self.assertEqual(result["epochs_completed"], 1)
                self.assertTrue(Path(context["checkpoint"]).is_file())
                self.assertTrue(Path(directory, "training.jsonl").is_file())
                context["data"] = {"images": context["data"]["val_images"]}
                logits = builtin_predict(context)
                self.assertEqual(logits.shape, (9, 9))
                self.assertTrue(np.isfinite(logits).all())
        finally:
            torch.set_num_threads(previous)

    def test_custom_training_helper_accepts_loss_and_optimizer(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as directory:
                context = self.context(directory)
                model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 28 * 28, 9))
                optimizer = torch.optim.SGD(model.parameters(), lr=.001)
                result = fit_model(context, model, loss_fn=torch.nn.CrossEntropyLoss(), optimizer=optimizer)
                self.assertEqual(result["epochs_completed"], 1)
                self.assertEqual(result["history"][0]["learning_rate"], .001)
                self.assertTrue(Path(directory, "model.ckpt").is_file())
        finally:
            torch.set_num_threads(previous)

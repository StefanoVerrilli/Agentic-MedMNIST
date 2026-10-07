"""Exercise trusted worker helpers locally, without importing generated Python."""
from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from contracts import TrainConfig
from worker_runtime import (builtin_train, builtin_predict, fit_model, validate_image_data,
                            verify_experiment, enforce_model_device)


class WorkerRuntimeTests(unittest.TestCase):
    def test_input_boundary_rejects_wrong_layout_dtype_and_nonfinite_values(self):
        for images in (np.zeros((9, 28, 28, 3), dtype="float32"),
                       np.zeros((9, 3, 28, 28), dtype="uint8"),
                       np.full((9, 3, 28, 28), np.nan, dtype="float32")):
            with self.subTest(shape=images.shape, dtype=images.dtype), self.assertRaises(ValueError):
                validate_image_data({"train_images": images})
        images = np.zeros((9, 3, 28, 28), dtype="float32")
        validate_image_data({"train_images": images, "train_targets": np.arange(9)})
        with self.assertRaises(ValueError):
            validate_image_data({"train_images": images, "train_targets": np.arange(8)})

    def test_device_guard_rejects_silent_cpu_execution_and_removes_hook(self):
        model = torch.nn.Linear(2, 2)
        with enforce_model_device("cuda"), self.assertRaisesRegex(ValueError, "configured device"):
            model(torch.zeros(1, 2))
        self.assertEqual(model(torch.zeros(1, 2)).shape, (1, 2))

    def test_verification_catches_all_three_observed_normalization_bugs(self):
        def build(context):
            return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3*28*28, 9))
        def predict(context):
            raise AssertionError("broken training must fail before inference")
        for transform in (lambda x: x.transpose(0, 3, 1, 2) - np.zeros((1, 3, 1, 1)),
                          lambda x: x - np.zeros((1, 1, 1, 3)), lambda x: x - np.zeros(3)):
            with tempfile.TemporaryDirectory() as directory:
                context = self.context(directory)
                def train(ctx):
                    transform(ctx["data"]["train_images"])
                with self.assertRaisesRegex(ValueError, "broadcast"):
                    verify_experiment(context, build, train, predict)

    def test_verification_catches_wrong_build_network_keywords(self):
        from lightning_components import build_network
        with tempfile.TemporaryDirectory() as directory:
            def build(context):
                return build_network("tiny_cnn", num_classes=9)
            with self.assertRaisesRegex(TypeError, "num_classes"):
                verify_experiment(self.context(directory), build, lambda _: None, lambda _: None)

    def test_verification_probes_resume_path_and_batched_inference(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for broken_resume in (False, True):
                with self.subTest(broken_resume=broken_resume), tempfile.TemporaryDirectory() as directory:
                    context = self.context(directory)
                    context["config"].update(execution_mode="agent_autonomous", early_stopping_patience=0,
                                             scheduler="one_cycle")
                    def build(ctx):
                        return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3*28*28, 9))
                    calls = []
                    def train(ctx):
                        calls.append((ctx["start_epoch"], ctx["resume"]))
                        if broken_resume and ctx["resume"]:
                            ctx["resume"].get("epochs_completed")
                        return fit_model(ctx, build(ctx))
                    def predict(ctx):
                        from worker_runtime import batched_logits
                        model = build(ctx)
                        model.load_state_dict(torch.load(ctx["checkpoint"], weights_only=True))
                        return batched_logits(model, ctx["data"]["images"], ctx["config"])
                    if broken_resume:
                        with self.assertRaisesRegex(AttributeError, "get"):
                            verify_experiment(context, build, train, predict)
                    else:
                        verify_experiment(context, build, train, predict)
                        self.assertTrue(Path(directory, "result.json").is_file())
                        self.assertTrue(Path(directory, "continuation_probe", "resume.pt").is_file())
                    self.assertEqual([call[0] for call in calls], [0, 1])
                    self.assertIsInstance(calls[1][1], str)
        finally:
            torch.set_num_threads(previous)

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

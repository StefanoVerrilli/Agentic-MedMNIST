"""Compact transformer depth-8 experiment for PathMNIST 28x28 histopathology."""
import torch
from lightning_components import build_network
from worker_runtime import fit_model, batched_logits


def build_model(context):
    parameters = context["parameters"]
    return build_network(
        "compact_transformer",
        in_channels=3,
        n_classes=9,
        hidden=parameters.get("hidden", 256),
        depth=parameters.get("depth", 8),
        dropout=parameters.get("dropout", 0.2),
        patch_size=parameters.get("patch_size", 4),
        num_heads=parameters.get("num_heads", 4),
        mlp_ratio=parameters.get("mlp_ratio", 4),
        pooling=parameters.get("pooling", "cls"),
    )


def train(context):
    return fit_model(context, build_model(context))


def predict(context):
    model = build_model(context)
    model.load_state_dict(torch.load(context["checkpoint"], map_location="cpu", weights_only=True))
    return batched_logits(model, context["data"]["images"], context["config"])
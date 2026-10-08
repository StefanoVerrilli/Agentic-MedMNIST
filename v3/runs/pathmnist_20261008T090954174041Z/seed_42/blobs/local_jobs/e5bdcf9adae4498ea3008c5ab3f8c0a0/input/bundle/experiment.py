"""Multi-scale transformer experiment for PathMNIST 28x28 histopathology classification."""
import torch
from lightning_components import build_network
from worker_runtime import fit_model, batched_logits


def build_model(context):
    parameters = context["parameters"]
    return build_network(
        "multi_scale_transformer",
        in_channels=3,
        n_classes=9,
        hidden=parameters.get("hidden", 128),
        depth=parameters.get("depth", 2),
        dropout=parameters.get("dropout", 0.1),
        scales=tuple(parameters.get("scales", [2, 4, 7])),
        num_heads=parameters.get("num_heads", 4),
        mlp_ratio=parameters.get("mlp_ratio", 4),
        pooling=parameters.get("pooling", "mean"),
    )


def train(context):
    return fit_model(context, build_model(context))


def predict(context):
    model = build_model(context)
    model.load_state_dict(torch.load(context["checkpoint"], map_location="cpu", weights_only=True))
    return batched_logits(model, context["data"]["images"], context["config"])
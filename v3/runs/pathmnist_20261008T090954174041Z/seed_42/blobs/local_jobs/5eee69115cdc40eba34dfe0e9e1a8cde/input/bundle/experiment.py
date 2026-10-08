import torch
from lightning_components import build_network
from worker_runtime import fit_model, batched_logits


def build_model(context):
    p = context["parameters"]
    return build_network("resnet18", in_channels=3, n_classes=9,
        hidden=p.get("hidden", 64), depth=p.get("depth", 2),
        dropout=p.get("dropout", 0.25))


def train(context):
    return fit_model(context, build_model(context))


def predict(context):
    model = build_model(context)
    model.load_state_dict(torch.load(context["checkpoint"], map_location="cpu", weights_only=True))
    return batched_logits(model, context["data"]["images"], context["config"])
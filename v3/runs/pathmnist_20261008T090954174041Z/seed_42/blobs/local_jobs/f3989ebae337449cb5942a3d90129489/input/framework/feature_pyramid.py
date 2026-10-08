"""ECCV 2020 FPT operators, with a compact PathMNIST backbone and classifier.

Reference: https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123730324.pdf
The mixture splits queries/keys, but shares full-width values across components.
"""
from __future__ import annotations

import torch
from torch import nn


class MixtureAttention(nn.Module):
    def __init__(self, width: int, components: int, *, euclidean: bool):
        super().__init__()
        if width < 1 or width % components:
            raise ValueError("width must be positive and divisible by mixture components")
        self.components = components
        self.euclidean = euclidean
        self.query = nn.Conv2d(width, width, 1)
        self.key = nn.Conv2d(width, width, 1)
        self.value = nn.Conv2d(width, width, 1)
        self.gate = nn.Linear(width, components, bias=False)

    def attention_weights(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """Accept projected B,C,H,W tensors; return B,query_pixels,key_pixels."""
        q = query.flatten(2).transpose(1, 2)
        k = key.flatten(2).transpose(1, 2)
        gates = self.gate(k.mean(dim=1)).softmax(dim=-1)
        weights = []
        for index, (part_q, part_k) in enumerate(zip(q.chunk(self.components, -1), k.chunk(self.components, -1))):
            similarity = (-torch.cdist(part_q, part_k, p=2, compute_mode="donot_use_mm_for_euclid_dist")
                          if self.euclidean else part_q @ part_k.transpose(-2, -1))
            weights.append(gates[:, index, None, None] * similarity.softmax(dim=-1))
        return torch.stack(weights).sum(dim=0)

    def forward(self, destination: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        weights = self.attention_weights(self.query(destination), self.key(source))
        values = self.value(source).flatten(2).transpose(1, 2)
        return (weights @ values).transpose(1, 2).reshape_as(destination)


class SelfTransformer(MixtureAttention):
    def __init__(self, width: int):
        super().__init__(width, 2, euclidean=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return super().forward(inputs, inputs)


class GroundingTransformer(MixtureAttention):
    def __init__(self, width: int):
        super().__init__(width, 4, euclidean=True)


class RenderingTransformer(nn.Module):
    """Fine-to-coarse channel rendering (Eq. 6), without spatial attention."""
    def __init__(self, width: int, stride: int):
        super().__init__()
        self.query = nn.Conv2d(width, width, 1)
        self.key = nn.Conv2d(width, width, 1)
        self.value = nn.Conv2d(width, width, 1)
        self.query_refine = nn.Conv2d(width, width, 3, padding=1)
        self.downsample = nn.Conv2d(width, width, 3, stride=stride, padding=1)
        self.refine = nn.Conv2d(width, width, 3, padding=1)

    def forward(self, destination: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        channel_weights = self.key(source).mean(dim=(-2, -1), keepdim=True)
        weighted = self.query_refine(self.query(destination) * channel_weights)
        downsampled = self.downsample(self.value(source))
        if weighted.shape != downsampled.shape:
            raise ValueError("rendering source and destination have incompatible pyramid shapes")
        return self.refine(weighted + downsampled)


class FeaturePyramidTransformerBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        if width < 4 or width % 4:
            raise ValueError("FPT width must be positive and divisible by four")
        self.self_transformers = nn.ModuleList(SelfTransformer(width) for _ in range(4))
        self.grounding = nn.ModuleDict({f"{src}_{dst}": GroundingTransformer(width)
                                       for dst in range(4) for src in range(dst + 1, 4)})
        self.rendering = nn.ModuleDict({f"{src}_{dst}": RenderingTransformer(width, 2 ** (dst - src))
                                       for dst in range(4) for src in range(dst)})
        self.fusion = nn.ModuleList(nn.Conv2d(5 * width, width, 3, padding=1) for _ in range(4))

    def forward(self, pyramid: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(pyramid) != 4:
            raise ValueError("FPT requires four pyramid levels")
        outputs = []
        for dst, original in enumerate(pyramid):
            branches = [original, self.self_transformers[dst](original)]
            for src, source in enumerate(pyramid):
                if src > dst:
                    branches.append(self.grounding[f"{src}_{dst}"](original, source))
                elif src < dst:
                    branches.append(self.rendering[f"{src}_{dst}"](original, source))
            outputs.append(self.fusion[dst](torch.cat(branches, dim=1)))
        return outputs


class FeaturePyramidTransformer(nn.Module):
    def __init__(self, *, in_channels: int, n_classes: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        from lightning_components import SmallResNet18
        backbone = SmallResNet18(in_channels=in_channels, n_classes=n_classes, base_width=32, dropout=0)
        self.stem, self.blocks = backbone.stem, backbone.blocks
        self.projections = nn.ModuleList(nn.Conv2d(channels, hidden, 1) for channels in (32, 64, 128, 256))
        self.transformer = FeaturePyramidTransformerBlock(hidden)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(4 * hidden, n_classes))

    def pyramid(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        if inputs.shape[-2:] != (28, 28):
            raise ValueError("FeaturePyramidTransformer expects 28x28 images")
        features = self.stem(inputs)
        pyramid = []
        for stage, projection in enumerate(self.projections):
            features = self.blocks[2 * stage + 1](self.blocks[2 * stage](features))
            pyramid.append(projection(features))
        return pyramid

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        transformed = self.transformer(self.pyramid(inputs))
        return self.head(torch.cat([level.mean(dim=(-2, -1)) for level in transformed], dim=1))

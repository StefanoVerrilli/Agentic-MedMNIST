"""Conventional (non-agentic) baseline: one tightly-coupled script doing the
same job. Kept ONLY so the agentic run has something to be compared against
(the WP4 comparison KPI). Deliberately monolithic to make the contrast visible.
"""
from __future__ import annotations


def run_baseline(flag: str = "pneumoniamnist", n: int = 2000) -> dict:
    import medmnist, numpy as np, torch, torch.nn as nn
    from medmnist import INFO
    torch.manual_seed(42)
    info = INFO[flag]
    DataClass = getattr(medmnist, info["python_class"])
    tr, te = DataClass(split="train", download=True), DataClass(split="test", download=True)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(tr.imgs), size=min(n, len(tr.imgs)), replace=False)

    def prep(x):
        x = x.astype("float32") / 255.0
        return x[:, None] if x.ndim == 3 else x.transpose(0, 3, 1, 2)

    xt = torch.tensor(prep(tr.imgs[idx])); yt = torch.tensor(tr.labels[idx].ravel()).long()
    ch, k = info["n_channels"], len(info["label"])
    model = nn.Sequential(nn.Conv2d(ch, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                          nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                          nn.Flatten(), nn.Linear(32 * 7 * 7, k))
    opt, lossf = torch.optim.Adam(model.parameters(), 1e-3), nn.CrossEntropyLoss()
    for _ in range(3):
        for i in range(0, len(xt), 64):
            opt.zero_grad(); loss = lossf(model(xt[i:i + 64]), yt[i:i + 64])
            loss.backward(); opt.step()
    with torch.no_grad():
        xe = torch.tensor(prep(te.imgs)); pred = model(xe).argmax(1).numpy()
    return {"accuracy": round(float((pred == te.labels.ravel()).mean()), 4)}

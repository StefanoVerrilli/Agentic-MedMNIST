"""The agents. One cohesive responsibility each; they communicate only through
the Blackboard. Heavy libs are imported lazily inside `run` so the orchestration
layer can be read/tested without torch installed.

Hybrid points (LLM reasoning): ProfilingAmbiguityAgent, ExperimentDesignAgent,
ReviewerConsistencyAgent. Everything else is deterministic.
"""
from __future__ import annotations

from typing import Protocol

from contracts import (AbstentionReport, AnomalyReport, Blackboard, DataProfile,
                       EvalReport, RepresentationPlan, SplitManifest, TrainConfig,
                       TrainResult)
from llm import reason_json

SEED = 42


class Agent(Protocol):
    name: str
    def run(self, bb: Blackboard) -> None: ...


class IngestionAgent:
    """WP2 upstream: pull a MedMNIST dataset (subsampled) and stash raw arrays."""
    name = "ingestion"

    def __init__(self, flag: str = "pneumoniamnist", n: int = 2000):
        self.flag, self.n = flag, n

    def run(self, bb: Blackboard) -> None:
        import medmnist, numpy as np
        from medmnist import INFO
        info = INFO[self.flag]
        DataClass = getattr(medmnist, info["python_class"])
        tr = DataClass(split="train", download=True)
        te = DataClass(split="test", download=True)
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(tr.imgs), size=min(self.n, len(tr.imgs)), replace=False)
        bb.put_blob("raw", {
            "flag": self.flag,
            "n_channels": info["n_channels"],
            "n_classes": len(info["label"]),
            "x_train": tr.imgs[idx], "y_train": tr.labels[idx].ravel(),
            "x_test": te.imgs, "y_test": te.labels.ravel(),
        })


class ProfilingAmbiguityAgent:
    """WP2 innovation: profile the data and flag ambiguity (hybrid comment)."""
    name = "profiling"

    def run(self, bb: Blackboard) -> None:
        import numpy as np
        raw = bb.get_blob("raw")
        counts = np.bincount(raw["y_train"], minlength=raw["n_classes"]).tolist()
        imbalance = max(counts) / max(1, min(c for c in counts if c > 0))
        decision = reason_json(
            system="You are a medical-imaging data auditor. Reply with JSON "
                   '{"ambiguity_note": "<one sentence>"}.',
            user=f"Dataset {raw['flag']}, class counts {counts}, "
                 f"imbalance ratio {imbalance:.2f}. Comment on ambiguity risk.",
            fallback={"ambiguity_note":
                      f"Imbalance ratio {imbalance:.2f}; "
                      f"{'watch minority-class ambiguity.' if imbalance > 1.5 else 'classes reasonably balanced.'}"},
        )
        bb.put("data_profile", DataProfile(
            dataset=raw["flag"], n_channels=raw["n_channels"],
            n_classes=raw["n_classes"], n_samples=int(len(raw["y_train"])),
            class_counts=counts, imbalance_ratio=round(float(imbalance), 3),
            ambiguity_note=decision["ambiguity_note"]))


class PreprocessingAgent:
    """Representation architect (hybrid). Not just standard preprocessing: it
    *reasons* about the best input representation for the task (normalization
    scheme + which label-preserving augmentations apply), records that choice as
    a versioned RepresentationPlan, then applies it deterministically. Decision
    is agentic; application is reproducible. A guardrail keeps augmentations
    within a known label-preserving set (the reviewer re-checks this in WP6)."""
    name = "preprocessing"
    ALLOWED_AUG = {"hflip", "rotate90"}

    def run(self, bb: Blackboard) -> None:
        import numpy as np
        raw, prof = bb.get_blob("raw"), bb.get("data_profile")

        # --- hybrid decision: architect the representation -------------------
        d = reason_json(
            system="You are a medical-imaging representation architect. Choose the "
                   "best input representation for a classifier. Reply JSON with keys "
                   'normalization ("unit"|"standardize"), augmentations (a subset of '
                   '["hflip","rotate90"] that PRESERVE the label for THIS task), '
                   "rationale.",
            user=f"Task {prof.dataset}: {prof.n_classes} classes, {prof.n_channels} "
                 f"channels, imbalance {prof.imbalance_ratio}. {prof.ambiguity_note}",
            fallback={"normalization": "standardize", "augmentations": [],
                      "rationale": "Per-channel standardization stabilises training; "
                                   "no augmentation unless known label-preserving."})
        norm = d["normalization"] if d["normalization"] in ("unit", "standardize") else "unit"
        aug = [a for a in d["augmentations"] if a in self.ALLOWED_AUG]  # guardrail
        bb.put("representation", RepresentationPlan(
            normalization=norm, augmentations=aug,
            rationale=str(d["rationale"]), source=str(d["source"])))

        # --- deterministic application of the chosen plan -------------------
        def to_chw(x):
            x = x.astype("float32") / 255.0
            return x[:, None, :, :] if x.ndim == 3 else x.transpose(0, 3, 1, 2)

        x, y = to_chw(raw["x_train"]), raw["y_train"]
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(len(x)); cut = int(0.8 * len(x))
        x_tr, y_tr = x[perm[:cut]], y[perm[:cut]]
        x_va, y_va = x[perm[cut:]], y[perm[cut:]]
        x_te, y_te = to_chw(raw["x_test"]), raw["y_test"]

        if norm == "standardize":                       # stats from TRAIN only
            mu = x_tr.mean((0, 2, 3), keepdims=True)
            sd = x_tr.std((0, 2, 3), keepdims=True) + 1e-6
            x_tr, x_va, x_te = (x_tr - mu) / sd, (x_va - mu) / sd, (x_te - mu) / sd

        for a in aug:                                   # train-only expansion
            flip = x_tr[:, :, :, ::-1] if a == "hflip" else np.rot90(x_tr, 1, (2, 3))
            x_tr = np.concatenate([x_tr, flip]); y_tr = np.concatenate([y_tr, y_tr])

        bb.put_blob("data", {
            "x_train": np.ascontiguousarray(x_tr), "y_train": y_tr,
            "x_val": np.ascontiguousarray(x_va), "y_val": y_va,
            "x_test": np.ascontiguousarray(x_te), "y_test": y_te})
        bb.put("split", SplitManifest(seed=SEED, train_size=int(len(x_tr)),
                                      val_size=int(len(x_va)), test_size=int(len(y_te))))


class ExperimentDesignAgent:
    """WP3 innovation (hybrid): pick a training config FROM the data profile AND
    the representation the preprocessing agent architected. Each augmentation
    roughly doubles the train set, so effective capacity is scaled accordingly."""
    name = "experiment_design"

    def run(self, bb: Blackboard) -> None:
        p = bb.get("data_profile")
        rep = bb.get("representation")
        augmented = bool(rep.augmentations)
        base = {"lr": 1e-3, "epochs": 4 if p.imbalance_ratio > 1.5 else 3,
                # more (augmented) data supports more capacity
                "hidden": 32 if augmented else 16, "batch_size": 64,
                "rationale": f"Heuristic from class balance, channels and "
                             f"representation (norm={rep.normalization}, "
                             f"aug={rep.augmentations or 'none'})."}
        d = reason_json(
            system="You configure a tiny CNN for the representation the "
                   "preprocessing agent chose. Reply JSON with keys "
                   "lr, epochs, hidden, batch_size, rationale.",
            user=f"Profile: {p.n_classes} classes, {p.n_channels} channels, "
                 f"{p.n_samples} base samples, imbalance {p.imbalance_ratio}. "
                 f"Representation: normalization={rep.normalization}, "
                 f"augmentations={rep.augmentations} (each ~doubles the train set). "
                 f"Ambiguity: {p.ambiguity_note}",
            fallback=base)
        bb.put("train_config", TrainConfig(
            lr=float(d["lr"]), epochs=int(d["epochs"]), hidden=int(d["hidden"]),
            batch_size=int(d["batch_size"]), rationale=str(d["rationale"]),
            source=str(d["source"])))


class TrainingAgent:
    """Deterministic: train a deliberately tiny CNN. Not the star of the show."""
    name = "training"

    def run(self, bb: Blackboard) -> None:
        import torch, torch.nn as nn, numpy as np
        torch.manual_seed(SEED)
        d, cfg = bb.get_blob("data"), bb.get("train_config")
        prof = bb.get("data_profile")
        model = _tiny_cnn(prof.n_channels, prof.n_classes, cfg.hidden)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        lossf = nn.CrossEntropyLoss()
        xt = torch.tensor(d["x_train"]); yt = torch.tensor(d["y_train"]).long()
        last = 0.0
        for _ in range(cfg.epochs):
            for i in range(0, len(xt), cfg.batch_size):
                opt.zero_grad()
                out = model(xt[i:i + cfg.batch_size])
                loss = lossf(out, yt[i:i + cfg.batch_size])
                loss.backward(); opt.step(); last = float(loss)
        with torch.no_grad():
            xv = torch.tensor(d["x_val"])
            va = (model(xv).argmax(1).numpy() == d["y_val"]).mean()
        bb.put_blob("model", model)
        bb.put("train_result", TrainResult(final_train_loss=round(last, 4),
                                           val_accuracy=round(float(va), 4)))


class EvaluationAgent:
    """Deterministic test-set metrics + stash probabilities for abstention."""
    name = "evaluation"

    def run(self, bb: Blackboard) -> None:
        import torch, numpy as np
        d = bb.get_blob("data"); model = bb.get_blob("model")
        prof = bb.get("data_profile")
        with torch.no_grad():
            probs = torch.softmax(model(torch.tensor(d["x_test"])), 1).numpy()
        bb.put_blob("test_probs", probs)
        pred, y = probs.argmax(1), d["y_test"]
        precs = []
        for c in range(prof.n_classes):
            tp = int(((pred == c) & (y == c)).sum())
            fp = int(((pred == c) & (y != c)).sum())
            precs.append(round(tp / (tp + fp), 4) if tp + fp else 0.0)
        bb.put("eval", EvalReport(accuracy=round(float((pred == y).mean()), 4),
                                  macro_precision=round(float(np.mean(precs)), 4),
                                  per_class_precision=precs))


class AbstentionOODAgent:
    """WP5 innovation: refuse to commit when unconfident ('no clear finding')."""
    name = "abstention"

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def run(self, bb: Blackboard) -> None:
        import numpy as np
        probs = bb.get_blob("test_probs"); y = bb.get_blob("data")["y_test"]
        conf, pred = probs.max(1), probs.argmax(1)
        covered = conf >= self.threshold
        cov = float(covered.mean())
        acc_cov = float((pred[covered] == y[covered]).mean()) if covered.any() else 0.0
        bb.put("abstention", AbstentionReport(
            threshold=self.threshold, coverage=round(cov, 4),
            abstain_rate=round(1 - cov, 4), accuracy_on_covered=round(acc_cov, 4)))


class ReportingAgent:
    """Assemble the final dossier from whatever artefacts exist."""
    name = "reporting"

    def run(self, bb: Blackboard) -> None:
        import json
        dossier = {k: _as_dict(v) for k, v in bb.artefacts.items()}
        (bb.root / "dossier.json").write_text(json.dumps(dossier, indent=2, default=str))


# --- Cross-cutting reviewer (WP6): runs after every stage --------------------

class ReviewerConsistencyAgent:
    """WP6 innovation: independent, cross-cutting consistency / anomaly check.
    Deterministic invariants + an optional hybrid comment. Can flag 'critical'
    to let the orchestrator veto promotion to the next stage."""
    name = "reviewer"

    def review(self, bb: Blackboard, stage: str) -> AnomalyReport:
        issues: list[str] = []
        a = bb.artefacts
        if stage == "preprocessing":
            s = a["split"]
            if s.val_size == 0 or s.train_size == 0:
                issues.append("empty train or val split")
            rp = a.get("representation")
            if rp:
                if rp.normalization not in ("unit", "standardize"):
                    issues.append(f"unknown normalization '{rp.normalization}'")
                bad = [x for x in rp.augmentations if x not in {"hflip", "rotate90"}]
                if bad:
                    issues.append(f"non-label-preserving augmentations: {bad}")
        if stage == "training":
            r = a["train_result"]
            if not 0.0 <= r.val_accuracy <= 1.0:
                issues.append("val_accuracy out of [0,1]")
        if stage == "evaluation":
            e = a["eval"]
            base = 1.0 / a["data_profile"].n_classes
            if e.accuracy < base:
                issues.append(f"accuracy {e.accuracy} below chance {base:.2f}")
        severity = "critical" if issues else "ok"
        comment = reason_json(
            system='Reply JSON {"comment":"<short consistency note>"}.',
            user=f"Stage {stage} produced artefacts: {list(a)}. Issues: {issues}.",
            fallback={"comment": "invariants passed" if not issues else "; ".join(issues)},
        )["comment"]
        rep = AnomalyReport(stage=stage, severity=severity, issues=issues, comment=comment)
        bb.put(f"anomaly_{stage}", rep)
        return rep


# --- small helpers -----------------------------------------------------------

def _tiny_cnn(in_ch: int, n_classes: int, hidden: int):
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(in_ch, hidden, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(hidden, hidden * 2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(hidden * 2 * 7 * 7, n_classes))


def _as_dict(obj):
    from dataclasses import asdict, is_dataclass
    return asdict(obj) if is_dataclass(obj) else obj

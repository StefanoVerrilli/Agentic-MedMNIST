"""Regenerate the presentation figures from the PathMNIST run artefacts.

Usage: python make_figures.py --npz /path/to/pathmnist.npz
Requires numpy and matplotlib. Writes PDF/PNG files next to this script.
"""
import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
RUN = HERE.parents[1] / "v3/runs/pathmnist_20261008T090954174041Z/seed_42"

GREEN = "#5E7A29"
BLUE = "#2a78d6"
GREY = "#b9b8b1"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

CLASSES = ["Adipose", "Background", "Debris", "Lymphocytes", "Mucus",
           "Smooth muscle", "Normal mucosa", "Cancer stroma", "Adenocarcinoma"]
ABBR = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "legend.frameon": False,
    "savefig.transparent": True,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})


def artefact(name):
    data = json.loads((RUN / "artefacts" / f"{name}.json").read_text())
    return data.get("payload", data)


def save(fig, name):
    fig.savefig(HERE / name, dpi=300)
    plt.close(fig)


def pick(images, labels, cls, n, seed):
    idx = np.flatnonzero(labels == cls)
    return images[np.random.default_rng(seed).choice(idx, n, replace=False)]


def tile_grid(rows, row_labels, name, width, gap=2):
    n_rows, n_cols = len(rows), len(rows[0])
    step = 28 + gap
    mosaic = np.full((n_rows * step - gap, n_cols * step - gap, 3), 255, dtype="uint8")
    for r, imgs in enumerate(rows):
        for c, img in enumerate(imgs):
            mosaic[r * step:r * step + 28, c * step:c * step + 28] = img
    fig, ax = plt.subplots(figsize=(width, width * mosaic.shape[0] / mosaic.shape[1]))
    ax.imshow(mosaic, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([r * step + 14 for r in range(n_rows)], row_labels, fontsize=11, color=INK)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    save(fig, name)


def samples(d):
    rows = [pick(d["train_images"], d["train_labels"].ravel(), c, 8, 7 + c) for c in range(9)]
    labels = [f"{n} ({a})" for n, a in zip(CLASSES, ABBR)]
    tile_grid(rows, labels, "fig_samples.png", width=4.6)


def class_distribution(d):
    tr = np.bincount(d["train_labels"].ravel(), minlength=9) / len(d["train_labels"]) * 100
    te = np.bincount(d["test_labels"].ravel(), minlength=9) / len(d["test_labels"]) * 100
    y = np.arange(9)[::-1]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    h = 0.38
    ax.barh(y + h / 2 + 0.01, tr, h, color=GREEN, label="Train (89,996)")
    ax.barh(y - h / 2 - 0.01, te, h, color=BLUE, label="Test, other centre (7,180)")
    ax.set_yticks(y, CLASSES)
    ax.set_xlabel("Share of split (%)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, fontsize=8)
    save(fig, "fig_class_distribution.pdf")


def augmentations(d):
    img = pick(d["train_images"], d["train_labels"].ravel(), 5, 1, 21)[0]
    f = img.astype("float32")
    mean = f.mean(axis=(0, 1), keepdims=True)
    views = [
        ("Original\n", img),
        ("Horizontal\nflip", np.flip(img, 1)),
        ("Vertical\nflip", np.flip(img, 0)),
        ("Rotate\n90°", np.rot90(img, 1)),
        ("Rotate\n180°", np.rot90(img, 2)),
        ("Brightness\n×1.1", np.clip(f * 1.1, 0, 255).astype("uint8")),
        ("Contrast\n×1.1", np.clip((f - mean) * 1.1 + mean, 0, 255).astype("uint8")),
    ]
    fig, axes = plt.subplots(1, 7, figsize=(7.0, 1.3), gridspec_kw={"wspace": 0.08})
    for ax, (title, im) in zip(axes, views):
        ax.imshow(im, interpolation="nearest")
        ax.set_title(title, fontsize=8, color=INK2 if not title.startswith("Original") else INK)
        ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    save(fig, "fig_augmentations.png")


def ambiguity(d):
    rows = [pick(d["train_images"], d["train_labels"].ravel(), c, 7, 50 + c) for c in (7, 5, 2)]
    tile_grid(rows, ["Cancer stroma", "Smooth muscle", "Debris"], "fig_ambiguity.png", width=6.0)


def ood_noise(d):
    img = pick(d["test_images"], d["test_labels"].ravel(), 6, 1, 11)[0].astype("float32") / 255
    rng = np.random.default_rng(1729)
    fig, axes = plt.subplots(1, 4, figsize=(4.4, 1.35), gridspec_kw={"wspace": 0.08})
    for ax, sigma in zip(axes, [0, 0.15, 0.25, 0.40]):
        im = np.clip(img + rng.normal(0, sigma, img.shape), 0, 1) if sigma else img
        ax.imshow(im, interpolation="nearest")
        ax.set_title("Clean" if sigma == 0 else f"σ = {sigma:.2f}", fontsize=8)
        ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    save(fig, "fig_ood_noise.png")


def stage_timeline():
    starts = {}
    durations = {}
    for line in (RUN / "decision_log.jsonl").read_text().splitlines():
        e = json.loads(line)
        if e.get("event") == "stage_started":
            starts.setdefault(e["stage"], e["timestamp"])
        if e.get("event") == "stage_promoted":
            from datetime import datetime
            t0 = datetime.fromisoformat(starts[e["stage"]])
            t1 = datetime.fromisoformat(e["timestamp"])
            durations[e["stage"]] = (t1 - t0).total_seconds() / 60
    names = {"ingestion": "Ingestion", "profiling": "Profiling", "prior_art": "Prior-art scout",
             "preprocessing": "Representation", "data_audit": "Data audit",
             "architecture_research": "Architecture research", "model_search": "Model search (8 trials)",
             "training": "Training (reuse best)", "evaluation": "Evaluation", "abstention": "Abstention / OOD",
             "reporting": "Reporting"}
    stages = list(durations)
    vals = [durations[s] for s in stages]
    y = np.arange(len(stages))[::-1]
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    colors = [GREEN if s == "model_search" else GREY for s in stages]
    ax.barh(y, vals, 0.62, color=colors)
    ax.set_xscale("log")
    ax.set_xlim(0.2, 2000)
    ax.set_yticks(y, [names[s] for s in stages])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Wall-clock minutes (log scale)")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    for yy, v in zip(y, vals):
        txt = f"{int(v // 60)} h {int(v % 60)} min" if v >= 60 else f"{v:.1f} min"
        ax.text(v * 1.08, yy, txt, va="center", fontsize=8, color=INK2)
    save(fig, "fig_stage_timeline.pdf")


def search_trials():
    trials = artefact("0051_search_report_v001")["validation_trials"]
    acc = [t["checkpoint_accuracy"] * 100 for t in trials]
    resnet = ["resnet18" in t["bundle"]["bundle_id"] for t in trials]
    labels = ["CCT\nd4 w128", "Multi-scale\nd2 w128", "CCT\nd4 w256", "CCT\nd6 w256",
              "CCT\nd8 w256", "ResNet-18\nbase", "ResNet-18\ntuned", "ResNet-18\ndrop .25"]
    x = np.arange(1, len(trials) + 1)
    fig, ax = plt.subplots(figsize=(5.0, 2.1))
    for xi, a, r in zip(x, acc, resnet):
        ax.scatter(xi, a, s=60 if xi == 7 else 34, color=GREEN if r else BLUE, zorder=3,
                   edgecolor="white", linewidth=1.5)
        ax.text(xi, a + 0.06, f"{a:.2f}", ha="center", fontsize=8, color=INK if xi == 7 else INK2,
                fontweight="bold" if xi == 7 else "normal")
    ax.set_xticks(x, [f"T{i}\n{l}" for i, l in zip(x, labels)], fontsize=6.5)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(98.55, 99.75)
    ax.set_ylabel("Validation accuracy (%)")
    ax.grid(axis="x", visible=False)
    ax.scatter([], [], color=BLUE, s=34, label="Transformer family")
    ax.scatter([], [], color=GREEN, s=34, label="ResNet-18 family")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7.5, ncol=2)
    save(fig, "fig_search_trials.pdf")


def training_curve():
    h = artefact("0053_train_result_v001")["history"]
    ep = [r["epoch"] for r in h]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(4.8, 1.95), gridspec_kw={"wspace": 0.35})
    a1.plot(ep, [r["val_accuracy"] * 100 for r in h], color=GREEN, lw=2)
    a1.set_title("Validation accuracy (%)", fontsize=9, loc="left", color=INK)
    a1.set_ylim(96.5, 100)
    a2.plot(ep, [r["train_loss"] for r in h], color=BLUE, lw=2, label="Train")
    a2.plot(ep, [r["val_loss"] for r in h], color=GREEN, lw=2, label="Validation")
    a2.set_title("Loss", fontsize=9, loc="left", color=INK)
    a2.legend(fontsize=7, loc="upper right")
    for ax in (a1, a2):
        ax.axvline(17, color=INK2, lw=1, ls="--")
        ax.set_xlabel("Epoch")
        ax.set_xlim(0.5, 29.5)
    a1.text(17.4, 96.8, "checkpoint\n(epoch 17)", fontsize=7, color=INK2)
    save(fig, "fig_training_curve.pdf")


def confusion():
    cm = np.array(artefact("0058_evaluation_report_v001")["confusion_matrix"], dtype=float)
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    cmap = LinearSegmentedColormap.from_list("g", ["#f7f9f2", "#c9d6a8", "#8aa64f", GREEN, "#2c3a12"])
    fig, ax = plt.subplots(figsize=(2.7, 2.5))
    ax.imshow(pct, cmap=cmap, vmin=0, vmax=100)
    ax.grid(False)
    for i in range(9):
        for j in range(9):
            if pct[i, j] >= 0.5:
                ax.text(j, i, f"{pct[i, j]:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if pct[i, j] > 55 else INK)
    ax.set_xticks(range(9), ABBR, rotation=90, fontsize=7)
    ax.set_yticks(range(9), ABBR, fontsize=7)
    ax.tick_params(length=0)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, "fig_confusion.pdf")


def per_class():
    pc = artefact("0058_evaluation_report_v001")["per_class"]
    order = np.argsort([p["recall"] for p in pc])
    y = np.arange(9)
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    for yy, k in zip(y, order):
        p = pc[k]
        ax.hlines(yy + 0.14, p["recall_ci95_low"] * 100, p["recall_ci95_high"] * 100, color=GREEN, lw=2)
        ax.scatter(p["recall"] * 100, yy + 0.14, color=GREEN, s=30, zorder=3, edgecolor="white", linewidth=1.2)
        ax.scatter(p["precision"] * 100, yy - 0.14, color=BLUE, s=30, marker="D", zorder=3,
                   edgecolor="white", linewidth=1.2)
    ax.set_yticks(y, [CLASSES[k] for k in order])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(45, 101)
    ax.set_xlabel("Test score (%)")
    ax.scatter([], [], color=GREEN, s=30, label="Recall (95% CI)")
    ax.scatter([], [], color=BLUE, s=30, marker="D", label="Precision")
    ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2, fontsize=7)
    save(fig, "fig_per_class.pdf")


def risk_coverage():
    a = artefact("0061_abstention_report_v001")
    curve = a["risk_coverage_curve"]
    vc = [c["validation_coverage"] * 100 for c in curve] + [100]
    vr = [c["validation_risk"] * 100 for c in curve] + [(1 - 0.995902) * 100]
    tc = [c["test_coverage"] * 100 for c in curve] + [100]
    tr = [c["test_risk"] * 100 for c in curve] + [(1 - a["base_test_accuracy"]) * 100]
    fig, ax = plt.subplots(figsize=(3.1, 2.4))
    ax.plot(vc, vr, color=BLUE, lw=2, marker="o", ms=5, label="Validation")
    ax.plot(tc, tr, color=GREEN, lw=2, marker="o", ms=5, label="Test (other centre)")
    op_c, op_r = a["test_coverage"] * 100, a["selective_risk"] * 100
    ax.scatter([op_c], [op_r], s=110, facecolor="none", edgecolor=INK, linewidth=1.3, zorder=4)
    ax.annotate("operating point\n69% kept, 2.4% error", (op_c, op_r), xytext=(42, 4.6), fontsize=7,
                color=INK2, arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.8})
    ax.annotate("no abstention\n8.2% error", (100, tr[-1]), xytext=(93, 6.0), fontsize=7, color=INK2,
                ha="right", arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.8})
    ax.set_xlabel("Coverage: predictions kept (%)")
    ax.set_ylabel("Error on kept predictions (%)")
    ax.set_xlim(40, 102)
    ax.set_ylim(0, 9)
    ax.legend(loc="upper left", fontsize=7)
    save(fig, "fig_risk_coverage.pdf")


def sota():
    rows = [("Google AutoML Vision", 0.728, 0.944), ("auto-sklearn", 0.716, 0.934), ("AutoKeras", 0.834, 0.959),
            ("ResNet-50 (224)", 0.892, 0.989), ("ResNet-18 (28)", 0.907, 0.983),
            ("ResNet-18 (224)", 0.909, 0.989), ("ResNet-50 (28)", 0.911, 0.990),
            ("Ours v3: ResNet-18, 1 seed", 0.918, 0.981),
            ("Ours v2: small CNN, 3 seeds", 0.921, 0.979)]
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(3.9, 2.9), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, col, title in ((axes[0], 1, "Test accuracy"), (axes[1], 2, "Test AUC")):
        vals = [r[col] for r in rows]
        colors = [GREEN if rows[i][0].startswith("Ours") else GREY for i in range(len(rows))]
        ax.barh(y, vals, 0.66, color=colors)
        for yy, v in zip(y, vals):
            ax.text(v + 0.01, yy, f"{v:.3f}", va="center", fontsize=7,
                    color=INK if rows[yy][0].startswith("Ours") else INK2,
                    fontweight="bold" if rows[yy][0].startswith("Ours") else "normal")
        ax.set_xlim(0, 1.22)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_title(title, fontsize=9, loc="left", color=INK)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, [r[0] for r in rows], fontsize=7.5)
    save(fig, "fig_sota.pdf")


# Earlier runs live only in git history (removed in commit 71250a1); values copied
# from their experiment_summary.json / comparison / ablation artefacts.
HISTORY = [  # (run id, pipeline, agentic test acc mean, std, baseline mean)
    ("20260908T122802", "v1", 0.870334, 0.0, 0.818663),
    ("20260908T130641", "v1", 0.836676, 0.008119, 0.778737),
    ("20260908T165614", "v1", 0.777020, 0.026650, 0.778737),
    ("20260908T210953", "v2", 0.890251, 0.0, 0.821309),
    ("20260908T212955", "v2", 0.887883, 0.0, 0.821309),
    ("20260909T043359", "v2", 0.893941, 0.006059, 0.830780),
    ("20260930T105337", "v2", 0.878343, 0.003830, 0.830780),
    ("20261001T081124", "v2", 0.868849, 0.006348, 0.829898),
    ("20261001T175019", "v2", 0.889368, 0.004778, 0.829898),
    ("20261002T083530", "v2", 0.926370, 0.005910, 0.829898),
    ("20261002T104242", "v2", 0.890576, 0.010607, 0.829898),
    ("20261002T153115", "v2", 0.920891, 0.003280, 0.829898),
    ("20261008T074051", "v3", 0.846518, 0.0, None),
    ("20261008T090954", "v3", 0.917827, 0.0, None),
]
V2_SEEDS = {42: (0.917549, 0.821309), 47: (0.925348, 0.840251), 72: (0.919777, 0.828134)}
V2_ABLATION = {"Representation\noff": [0.776323, 0.878552, 0.772145],
               "Standardise\nonly": [0.905710, 0.884680, 0.800000],
               "Standardise\n+ flip": [0.897772, 0.918524, 0.922284]}


def run_history():
    x = np.arange(len(HISTORY))
    acc = np.array([h[2] for h in HISTORY]) * 100
    std = np.array([h[3] for h in HISTORY]) * 100
    fig, ax = plt.subplots(figsize=(5.4, 2.4))
    for lo, hi, name in ((-0.5, 2.5, "v1 pipeline"), (2.5, 11.5, "v2: bounded agentic search"),
                         (11.5, 13.5, "v3:\nautonomous")):
        ax.axvspan(lo, hi, color="#f1f0eb" if name.startswith("v2") else "white", zorder=0)
        ax.text((lo + hi) / 2, 96.6, name, ha="center", va="top", fontsize=7, color=INK2)
    base = [(i, h[4] * 100) for i, h in enumerate(HISTORY) if h[4]]
    ax.scatter(*zip(*base), marker="_", s=90, color=MUTED, linewidth=2, label="Conventional baseline", zorder=2)
    ax.errorbar(x, acc, yerr=std, fmt="o", color=GREEN, ms=5, capsize=0, lw=1.5, label="Agentic (mean of seeds)", zorder=3)
    for i in (9, 11, 13):
        ax.annotate(f"{acc[i]:.1f}", (i, acc[i]), xytext=(0, 7), textcoords="offset points", ha="center",
                    fontsize=7, color=INK)
    ax.set_xticks(x, [h[0][4:6] + "/" + h[0][6:8] for h in HISTORY], fontsize=6.5, rotation=90)
    ax.set_xlim(-0.5, len(HISTORY) - 0.5)
    ax.set_ylim(74, 97)
    ax.set_ylabel("Test accuracy (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    save(fig, "fig_run_history.pdf")


def baseline_comparison():
    seeds = list(V2_SEEDS)
    x = np.arange(len(seeds))
    agent = [V2_SEEDS[s][0] * 100 for s in seeds]
    base = [V2_SEEDS[s][1] * 100 for s in seeds]
    fig, ax = plt.subplots(figsize=(2.7, 2.4))
    w = 0.36
    ax.bar(x - w / 2 - 0.01, base, w, color=GREY, label="Conventional")
    ax.bar(x + w / 2 + 0.01, agent, w, color=GREEN, label="Agentic")
    for xi, a, b in zip(x, agent, base):
        ax.text(xi + w / 2, a + 1, f"{a:.1f}", ha="center", fontsize=7, color=INK)
        ax.text(xi - w / 2, b + 1, f"{b:.1f}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(x, [f"seed {s}" for s in seeds])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Test accuracy (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7)
    save(fig, "fig_baseline.pdf")


def ablation():
    names = list(V2_ABLATION)
    fig, ax = plt.subplots(figsize=(2.7, 2.4))
    for i, n in enumerate(names):
        vals = np.array(V2_ABLATION[n]) * 100
        ax.bar(i, vals.mean(), 0.6, color=GREEN if i == 2 else GREY)
        ax.scatter([i] * 3, vals, s=14, color=INK, zorder=3)
        ax.text(i, 6, f"{vals.mean():.1f}", ha="center", fontsize=8,
                color="white" if i == 2 else INK, fontweight="bold")
    ax.set_xticks(range(3), names, fontsize=7)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Test accuracy (%)")
    ax.grid(axis="x", visible=False)
    ax.set_title("bar = mean, dots = seeds 42/47/72", fontsize=7, color=INK2)
    save(fig, "fig_ablation.pdf")


def llm_stages():
    stages = ["Ingestion", "Profiling", "Prior-art scout", "Representation", "Data audit",
              "Architecture research", "Model search", "Training", "Evaluation", "Abstention", "Reporting"]
    agent = [0, 1, 1, 1, 0, 1, 10, 0, 0, 0, 2]
    review = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2]
    y = np.arange(len(stages))[::-1]
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ax.barh(y, agent, 0.62, color=GREEN, label="Task-agent decisions (16)")
    ax.barh(y, review, 0.62, left=[a + (0.08 if a else 0) for a in agent], color=BLUE,
            label="Reviewer decisions (12)")
    ax.set_yticks(y, stages)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("LLM decisions in the flagship run")
    ax.set_xticks(range(0, 13, 2))
    ax.legend(loc="lower right", fontsize=7)
    save(fig, "fig_llm_stages.pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    d = dict(np.load(parser.parse_args().npz))
    samples(d)
    class_distribution(d)
    augmentations(d)
    ambiguity(d)
    ood_noise(d)
    stage_timeline()
    search_trials()
    training_curve()
    confusion()
    per_class()
    risk_coverage()
    sota()
    run_history()
    baseline_comparison()
    ablation()
    llm_stages()


if __name__ == "__main__":
    main()

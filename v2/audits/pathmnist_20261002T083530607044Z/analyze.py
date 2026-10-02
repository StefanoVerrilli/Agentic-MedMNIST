"""Companion analysis of stored evidence; no training or historical writes."""
import json
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/pathmnist_20261002T083530607044Z"
OLD = ROOT / "runs/pathmnist_20261001T175019922973Z"


def artifacts(root):
    result = {}
    for path in sorted((root / "artefacts").glob("*.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        result[item["artefact"]] = item["payload"]
    return result


def confidence(p, method):
    if method == "max_softmax":
        return p.max(axis=1)
    p = np.clip(p.astype(np.float64), 1e-12, 1)
    return 1 + (p*np.log(p)).sum(axis=1)/np.log(p.shape[1])


rows, configs, representations, yaml_configs = [], [], [], []
for seed in (42, 47, 72):
    root = RUN / f"seed_{seed}"
    a, old = artifacts(root), artifacts(OLD / f"seed_{seed}")
    evaluation, abstention = a["evaluation_report"], a["abstention_report"]
    config = {k: v for k, v in a["train_config"].items() if k not in {"seed", "source", "rationale"}}
    configs.append(config)
    representations.append({k: v for k, v in a["best_configuration"]["representation"].items() if k != "rationale"})
    payload = json.loads((root / "best_config_v001.yaml").read_text(encoding="utf-8"))
    payload["seed_everything"] = 0
    payload["data"]["seed"] = 0
    yaml_configs.append(payload)
    row = {"seed": seed, "accuracy": evaluation["accuracy"], "macro_f1": evaluation["macro_f1"],
           "balanced_accuracy": evaluation["balanced_accuracy"],
           "accuracy_delta_old": evaluation["accuracy"]-old["evaluation_report"]["accuracy"],
           "final_best_val_accuracy": a["train_result"]["best_val_accuracy"],
           "validation_test_gap": a["train_result"]["best_val_accuracy"]-evaluation["accuracy"],
           "correct": sum(evaluation["confusion_matrix"][i][i] for i in range(9)),
           "errors": evaluation["n_samples"]-sum(evaluation["confusion_matrix"][i][i] for i in range(9)),
           "classes": [], "ood": []}
    with np.load(root / "blobs/predictions_v001.npz", allow_pickle=False) as p, \
         np.load(root / abstention["ood_evidence_path"], allow_pickle=False) as ood:
        y, pred = p["test_targets"], p["test_predictions"]
        covered = confidence(p["test_probabilities"], abstention["method"]) >= abstention["threshold"]
        for c, prev in zip(evaluation["per_class"], old["evaluation_report"]["per_class"]):
            mask = y == c["label"]
            row["classes"].append({"name": c["name"], "recall": c["recall"], "precision": c["precision"],
                                   "f1": c["f1"], "recall_old": prev["recall"], "f1_old": prev["f1"],
                                   "coverage": float(covered[mask].mean()),
                                   "deferred": int((mask & ~covered).sum()),
                                   "accepted_errors": int((mask & covered & (pred != y)).sum())})
        for key in ood.files:
            if key.startswith("test_"):
                arr = ood[key]
                row["ood"].append({"scenario": key, "mean_max_softmax": float(arr.max(axis=1).mean()),
                                   "predicted_class_counts": np.bincount(arr.argmax(axis=1), minlength=9).tolist(),
                                   "autoaccept_fraction_at_clean_threshold": float((confidence(arr, abstention["method"]) >= abstention["threshold"]).mean())})
    trials = [v for k,v in a.items() if k.startswith("trial_")]
    row["trial_duration_seconds"] = sum(t["duration_seconds"] for t in trials)
    row["trial_epoch_budget_sum"] = sum(t["config"]["epochs"] for t in trials)
    row["trial_count"] = len(trials)
    events = [json.loads(line) for line in (root / "decision_log.jsonl").read_text(encoding="utf-8").splitlines()]
    row["decision_sources"] = {source: sum(e.get("source") == source for e in events if e["event"] == "llm_decision")
                                for source in {e.get("source") for e in events if e["event"] == "llm_decision"}}
    rows.append(row)
report = {"configuration_frozen": all(c == configs[0] for c in configs),
          "representation_frozen": all(c == representations[0] for c in representations),
          "yaml_equal_except_seed": all(c == yaml_configs[0] for c in yaml_configs),
          "accuracy_sample_std": statistics.stdev(r["accuracy"] for r in rows),
          "macro_f1_mean": statistics.mean(r["macro_f1"] for r in rows),
          "balanced_accuracy_mean": statistics.mean(r["balanced_accuracy"] for r in rows),
          "per_class_means": [{"name": rows[0]["classes"][i]["name"],
                               "recall": statistics.mean(r["classes"][i]["recall"] for r in rows),
                               "f1": statistics.mean(r["classes"][i]["f1"] for r in rows),
                               "recall_old": statistics.mean(r["classes"][i]["recall_old"] for r in rows),
                               "f1_old": statistics.mean(r["classes"][i]["f1_old"] for r in rows)} for i in range(9)],
          "seeds": rows}
Path(__file__).with_name("supplemental.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps({k:v for k,v in report.items() if k != "seeds"}, indent=2))
for r in rows:
    print(json.dumps({k:v for k,v in r.items() if k != "classes"}))

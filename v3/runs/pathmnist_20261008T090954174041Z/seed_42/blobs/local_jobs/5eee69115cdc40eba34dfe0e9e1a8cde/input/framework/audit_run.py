"""Read-only audit of stored run evidence; writes only to a separate output directory."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from governance import evidence_references


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def score(probabilities, method):
    values = np.asarray(probabilities, dtype=np.float64)
    if method == "max_softmax":
        return 1 - values.max(axis=1)
    clipped = np.clip(values, 1e-12, 1)
    return -(clipped * np.log(clipped)).sum(axis=1) / np.log(values.shape[1])


def auc(clean, corrupted, method):
    return float(roc_auc_score(np.r_[np.zeros(len(clean)), np.ones(len(corrupted))],
                              np.r_[score(clean, method), score(corrupted, method)]))


def audit_seed(root):
    artifacts, files, problems, newline_changes = {}, {}, [], []
    def check_file(path, expected):
        value = str(path)
        if str(path) in files:
            if files[str(path)] != expected:
                problems.append(f"conflicting checksum: {value}")
            return
        files[str(path)] = expected
        if not path.is_file():
            problems.append(f"missing: {value}")
            return
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            if b"\r\n" in content and hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest() == expected:
                newline_changes.append(value)
            else:
                problems.append(f"checksum mismatch: {value}")

    def walk(value):
        try:
            for _, path, expected in evidence_references(root, value):
                check_file(path, expected)
        except (ValueError, KeyError, TypeError) as exc:
            problems.append(f"invalid reference: {exc}")

    for path in sorted((root / "artefacts").glob("*.json")):
        envelope = json.loads(path.read_text(encoding="utf-8"))
        if canonical(envelope["payload"]) != envelope["payload_sha256"]:
            problems.append(f"payload mismatch: {path.name}")
        artifacts[envelope["artefact"]] = envelope["payload"]
        walk(envelope["payload"])
    for path in sorted((root / "blobs" / "reasoning").glob("*.json")):
        envelope = json.loads(path.read_text(encoding="utf-8"))
        if canonical(envelope["payload"]) != envelope["sha256"]:
            problems.append(f"reasoning mismatch: {path.name}")

    checks, ood_rows, class_rows = {}, [], []
    def compare(name, actual, expected):
        passed = bool(np.isclose(actual, expected, atol=5.1e-7, rtol=0))
        checks[name] = {"actual": float(actual), "recorded": float(expected), "matched": passed}

    with np.load(root / artifacts["prediction_artifact"]["path"], allow_pickle=False) as predictions, \
         np.load(root / artifacts["blob_probabilities"]["path"], allow_pickle=False) as probabilities, \
         np.load(root / artifacts["abstention_report"]["ood_evidence_path"], allow_pickle=False) as ood:
        y, p = predictions["test_targets"], predictions["test_probabilities"]
        predicted = p.argmax(axis=1)
        evaluation, abstention = artifacts["evaluation_report"], artifacts["abstention_report"]
        for name, actual in (("accuracy", accuracy_score(y, predicted)),
                             ("macro_f1", f1_score(y, predicted, average="macro")),
                             ("balanced_accuracy", balanced_accuracy_score(y, predicted))):
            compare(name, actual, evaluation[name])
        compare("prediction_array", np.array_equal(predicted, predictions["test_predictions"]), 1)
        for split, prefix in (("val", "validation"), ("test", "test")):
            compare(f"{split}_array_consistency", np.array_equal(probabilities[split], predictions[f"{prefix}_probabilities"]), 1)
            compare(f"{split}_target_consistency", np.array_equal(probabilities[f"{split}_targets"], predictions[f"{prefix}_targets"]), 1)
            compare(f"{split}_normalized_probabilities", np.allclose(probabilities[split].sum(axis=1), 1, atol=1e-6), 1)
        val_conf = 1 - score(predictions["validation_probabilities"], abstention["method"])
        threshold = float(np.quantile(val_conf, 1-abstention["target_validation_coverage"], method="lower"))
        compare("threshold", threshold, abstention["threshold"])
        eligible = abstention.get("detector_status") != "no_eligible_detector"
        covered = (1-score(p, abstention["method"]) >= threshold) & eligible
        compare("test_coverage", covered.mean(), abstention["test_coverage"])
        compare("accuracy_on_covered", (predicted[covered] == y[covered]).mean() if covered.any() else 0,
                abstention["accuracy_on_covered"])
        compare("human_review_count", (~covered).sum(), abstention["human_review_count"])
        queue = json.loads((root / artifacts["human_review_queue"]["path"]).read_text(encoding="utf-8"))
        compare("queue_indices", set(int(c["sample_index"]) for c in queue["cases"]) == set(np.flatnonzero(~covered)), 1)
        for scenario in abstention["ood_scenarios"]:
            sigma = scenario["corruption"].split("sigma_")[-1]
            key = sigma.replace(".", "_")
            row = {"sigma": sigma, "method": scenario["score"]}
            for split, prefix in (("val", "validation"), ("test", "test")):
                actual = auc(probabilities[split], ood[f"{prefix}_gaussian_{key}"], scenario["score"])
                row[f"{prefix}_auroc"] = actual
                compare(f"ood_{prefix}_{sigma}_{scenario['score']}", actual, scenario[f"{prefix}_auroc"])
                if scenario.get(f"{prefix}_false_accept_rate") is not None:
                    method_threshold = np.quantile(1-score(probabilities["val"], scenario["score"]),
                                                   1-abstention["target_validation_coverage"], method="lower")
                    far = float(((1-score(ood[f"{prefix}_gaussian_{key}"], scenario["score"])) >= method_threshold).mean())
                    row[f"{prefix}_false_accept_rate"] = far
                    compare(f"ood_{prefix}_far_{sigma}_{scenario['score']}", far,
                            scenario[f"{prefix}_false_accept_rate"])
            corrupted = ood[f"test_gaussian_{key}"]
            row["corrupted_mean_max_softmax"] = float(corrupted.max(axis=1).mean())
            row["corrupted_predicted_class_counts"] = np.bincount(corrupted.argmax(axis=1), minlength=p.shape[1]).tolist()
            row["clean_mean_max_softmax"] = float(p.max(axis=1).mean())
            ood_rows.append(row)
        selected = [row for row in ood_rows if row["method"] == abstention["method"]]
        compare("ood_aggregate", np.mean([row["test_auroc"] for row in selected]), abstention["ood_auroc"])
        if abstention.get("detector_status", "legacy_unchecked") == "legacy_unchecked":
            passed = min(row["test_auroc"] for row in selected) >= .5
        else:
            validation_eligible = all(row["validation_auroc"] >= abstention["minimum_scenario_auroc"]
                                      and row["validation_false_accept_rate"] <= abstention["maximum_false_accept_rate"]
                                      for row in selected)
            compare("detector_eligibility", validation_eligible, eligible)
            passed = validation_eligible and all(
                row["test_auroc"] >= abstention["minimum_scenario_auroc"]
                and row["test_false_accept_rate"] <= abstention["maximum_false_accept_rate"] for row in selected)
        compare("ood_pass", passed, abstention["ood_pass"])
        for item in evaluation["per_class"]:
            label = item["label"]
            mask = y == label
            class_rows.append({"name": item["name"], "support": int(mask.sum()),
                               "coverage": float(covered[mask].mean()),
                               "recall": float((predicted[mask] == label).mean())})
    return {"seed": artifacts["run_manifest"]["seed"], "integrity_problems": problems,
            "newline_only_changes": newline_changes, "referenced_files_checked": len(files),
            "metrics_verified": all(v["matched"] for v in checks.values()), "checks": checks,
            "ood_scenarios": ood_rows, "class_metrics": class_rows,
            "baseline_metrics_recomputed": False,
            "limitations": ["Stored baseline predictions are absent; baseline metrics not independently recomputed.",
                            "Checkpoint inference and retraining replay are separate checks."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.run.resolve()):
        parser.error("output must be outside the original run")
    report = {"run": str(args.run.resolve()), "seeds": [audit_seed(p) for p in sorted(args.run.glob("seed_*"))]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for seed in report["seeds"]:
        print(f"seed={seed['seed']} metrics_verified={seed['metrics_verified']} integrity_problems={len(seed['integrity_problems'])} newline_changes={len(seed['newline_only_changes'])}")
    return 0 if report["seeds"] and all(s["metrics_verified"] and not s["integrity_problems"] for s in report["seeds"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())

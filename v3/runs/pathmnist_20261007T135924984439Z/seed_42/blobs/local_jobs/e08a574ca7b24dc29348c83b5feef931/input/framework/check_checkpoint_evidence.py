"""Check stored checkpoints on CPU without training or changing historical evidence."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from blobio import load_blob
from contracts import BlobReference, RepresentationDecision
from governance import read_artefact
from lightning_components import PathMNISTLitModule
from ml import predict_outputs, prepare_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.run.resolve()):
        parser.error("output must be outside the original run")
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for root in sorted(args.run.glob("seed_*")):
        seed = read_artefact(root, "run_manifest")["seed"]
        prepared = load_blob(root, BlobReference.model_validate(read_artefact(root, "blob_prepared_data")))
        result = {"seed": seed, "backend": "cpu", "retraining_replay": False}
        for name, ref in (("agentic", read_artefact(root, "train_result")),
                          ("baseline", read_artefact(root, "baseline_report"))):
            checkpoint = torch.load(root / ref["checkpoint_path"], map_location="cpu", weights_only=True)
            model = PathMNISTLitModule(**checkpoint["hyper_parameters"])
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            data = prepared if name == "agentic" else prepare_data(
                prepared.bundle, RepresentationDecision(normalization="standardize", augmentations=[], rationale="Historical fixed baseline"))
            probabilities, logits, targets = predict_outputs(model, data, "test", 128, device="cpu", seed=seed)
            predicted = probabilities.argmax(axis=1)
            metrics = {"accuracy": float(accuracy_score(targets, predicted)),
                       "macro_f1": float(f1_score(targets, predicted, average="macro")),
                       "parameter_count": sum(p.numel() for p in model.parameters())}
            recorded = read_artefact(root, "evaluation_report") if name == "agentic" else ref
            metrics["recorded_metrics_match"] = all(np.isclose(metrics[k], recorded[k], atol=5.1e-7, rtol=0) for k in ("accuracy", "macro_f1"))
            if name == "agentic":
                with np.load(root / "blobs/predictions_v001.npz", allow_pickle=False) as old:
                    metrics["prediction_labels_equal"] = bool(np.array_equal(predicted, old["test_predictions"]))
                    metrics["probabilities_exact_equal"] = bool(np.array_equal(probabilities, old["test_probabilities"]))
                    metrics["max_probability_difference"] = float(np.max(np.abs(probabilities-old["test_probabilities"])))
            else:
                np.savez_compressed(args.output / f"baseline_predictions_seed_{seed}.npz", probabilities=probabilities, logits=logits, targets=targets)
            result[name] = metrics
            print(f"seed={seed} {name}: {metrics}", flush=True)
        results.append(result)
        (args.output / "checkpoint_inference.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0 if results and all(r[n]["recorded_metrics_match"] for r in results for n in ("agentic", "baseline")) else 2


if __name__ == "__main__":
    raise SystemExit(main())

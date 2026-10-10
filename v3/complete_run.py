"""Complete historical evidence without rewriting the original experiment.

Protocol: v2 at 0bbd1c5 (fixed tiny CNN, three representations, seeds 42/47/72).
No model search or LLM calls are made; live research is explicitly post-training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import statistics
import sys
from pathlib import Path

from baseline import run_baseline, run_representation_ablations
from contracts import (AnomalyReport, Blackboard, ComparisonReport, CompletedAblation,
                       CompletionManifest, PriorArtBrief, RunConfiguration,
                       sha256_file, utc_now)
from governance import check_integrity, evidence_references, write_acceptance
from run import code_tree_sha256, package_versions
from research import (DEFAULT_QUERY, LiteratureStore, citation_issues,
                      contained_path, fallback_ideas)

SEEDS = [42, 47, 72]
TEXT_SUFFIXES = {".py", ".json", ".jsonl", ".yaml", ".yml", ".md", ".txt"}


def portable_bytes(path):
    data = path.read_bytes()
    return data.replace(b"\r\n", b"\n") if path.suffix in TEXT_SUFFIXES else data


def matches_checksum(path, expected):
    return sha256_file(path) == expected or hashlib.sha256(portable_bytes(path)).hexdigest() == expected


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def latest_source(runs_root):
    candidates = []
    for directory in Path(runs_root).glob("pathmnist_*"):
        summary = directory / "experiment_summary.json"
        if not summary.exists() or (directory / "completion_manifest.json").exists():
            continue
        document = json.loads(summary.read_text(encoding="utf-8"))
        if document.get("completion_source"):
            continue
        if 42 in document.get("completed_seeds", []):
            candidates.append((document["experiment_id"], directory))
    if not candidates:
        raise ValueError("no original completed PathMNIST experiment with seed 42 found")
    return max(candidates, key=lambda item: item[0])[1].resolve()


def file_inventory(root):
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("source snapshot cannot contain symlinks")
    return {path.relative_to(root).as_posix(): hashlib.sha256(portable_bytes(path)).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()}


def inspect_source(source):
    """Permit missing historical binaries, but never accept altered evidence."""
    seed_root = source / "seed_42"
    bb = Blackboard.open(seed_root)
    missing = set()
    for artefact in bb.artefacts.values():
        for location, path, expected in evidence_references(seed_root, artefact.model_dump(mode="json")):
            if not path.exists():
                if path.suffix not in {".npz", ".ckpt", ".pt"}:
                    raise ValueError(f"missing historical evidence: {location}: {path}")
                missing.add(path.relative_to(source).as_posix())
            elif not matches_checksum(path, expected):
                raise ValueError(f"historical evidence checksum mismatch: {location}")
    # Validate every envelope version, not just the latest Blackboard values.
    from contracts import _sha256_json
    for path in (seed_root / "artefacts").glob("*.json"):
        envelope = json.loads(path.read_text(encoding="utf-8"))
        if _sha256_json(envelope["payload"]) != envelope["payload_sha256"]:
            raise ValueError(f"historical payload checksum mismatch: {path.name}")
        for location, evidence, expected in evidence_references(seed_root, envelope["payload"]):
            if evidence.exists() and not matches_checksum(evidence, expected):
                raise ValueError(f"historical evidence checksum mismatch: {location}")
    best = bb.get("best_configuration")
    if best.train_config.seed != 42 or best.test_metrics_used:
        raise ValueError("source must have a validation-selected seed 42 configuration")
    if best.train_config.generated_bundle:
        reference = best.train_config.generated_bundle
        bundle_manifest = contained_path(seed_root, reference.path)
        if not matches_checksum(bundle_manifest, reference.sha256):
            raise ValueError("generated manifest checksum mismatch")
        bundle_document = json.loads(bundle_manifest.read_text(encoding="utf-8"))
        for name, digest in bundle_document["files"].items():
            if not matches_checksum(contained_path(bundle_manifest.parent, name), digest):
                raise ValueError(f"generated source checksum mismatch: {name}")
    evaluation = bb.get("evaluation_report")
    if evaluation.split != "test":
        raise ValueError("historical evaluation must use the test split")
    result = bb.get("train_result")
    checkpoint_verified = (seed_root / result.checkpoint_path).is_file()
    return bb, sorted(missing), checkpoint_verified


def validate_dataset(bundle, manifest):
    from ml import index_fingerprint
    if (bundle.archive_sha256 != manifest.archive_sha256
            or bundle.verified_archive_md5 != manifest.verified_archive_md5
            or bundle.official_sizes != manifest.official_split_sizes
            or list(bundle.labels) != manifest.labels):
        raise ValueError("reconstructed dataset archive/metadata differs from source")
    for split in ("train", "val", "test"):
        digest = index_fingerprint(split, bundle.selected_indices[split], bundle.targets[split])
        if (digest != manifest.selected_index_sha256[split]
                or bundle.selected_index_sha256[split] != digest
                or len(bundle.targets[split]) != manifest.loaded_split_sizes[split]):
            raise ValueError(f"reconstructed {split} split differs from source")


def load_dataset(source_bb, data_root):
    from ml import load_pathmnist
    reference = source_bb.get("blob_raw_dataset")
    if contained_path(source_bb.root, reference.path).exists():
        bundle = source_bb.get_blob("raw_dataset")
    else:
        manifest = source_bb.get("dataset_manifest")
        bundle = load_pathmnist(seed=42, data_root=data_root, download=True,
                               **{f"{split}_limit": manifest.loaded_split_sizes[split]
                                  for split in ("train", "val", "test")})
    validate_dataset(bundle, source_bb.get("dataset_manifest"))
    return bundle


def retrieve_live(bb, root):
    # A failed attempt uses a new cache next time; no historical/cache fallback.
    cache = root / "literature_cache" / utc_now().replace(":", "").replace("+", "_")
    sources, mode, errors = LiteratureStore(cache, online=True, require_live=True).retrieve(bb.root)
    brief = PriorArtBrief(query=DEFAULT_QUERY, sources=sources, ideas=fallback_ideas(sources),
                          source="post_training_completion", retrieval_mode=mode, retrieval_errors=errors)
    issues = citation_issues(brief, bb.root)
    if mode != "online" or issues:
        raise ValueError("live research verification failed: " + "; ".join(issues))
    bb.put("prior_art_brief", brief, producer="completion_research")
    bb.record_event("post_training_research_completed", original_training_unchanged=True)


def initialize_seed(root, source_bb, bundle, seed, device):
    from agents import FrozenConfigurationAgent
    from extensions import registry_entries
    from remote import copy_bundle
    bb = Blackboard(root, run_id=f"{root.parent.name}/seed_{seed}")
    bb.put("run_configuration", RunConfiguration(parameters={"seeds": "42,47,72",
           "skip_baseline": False, "search_per_seed": False, "completion": True,
           "device": device}, extension_registry=registry_entries()), producer="completion")
    manifest = source_bb.get("run_manifest").model_copy(update={
        "seed": seed, "run_id": bb.run_id, "created_at": utc_now(),
        "code_sha256": code_tree_sha256(Path(__file__).parent), "code_hash_algorithm": "canonical-text-v2",
        "python_version": sys.version.split()[0], "platform": platform.platform(),
        "package_versions": package_versions(), "llm_mode": "heuristic", "llm_model": "none",
        "llm_model_digest": None})
    bb.put("run_manifest", manifest, producer="completion")
    bb.put("dataset_manifest", source_bb.get("dataset_manifest"), producer="completion")
    bb.put("data_profile", source_bb.get("data_profile"), producer="completion")
    bb.put_blob("raw_dataset", bundle, producer="completion")
    selected = source_bb.get("best_configuration")
    if selected.train_config.generated_bundle:
        copy_bundle(source_bb.root, root, selected.train_config.generated_bundle)
    FrozenConfigurationAgent(selected, seed=seed, device=device).run(bb)
    bb.put("anomaly_completion", AnomalyReport(stage="completion", attempt=1,
        severity="warning", action="continue", deterministic_issues=[], llm_issues=[],
        observations=["Live research is post-training; original warnings remain in source_snapshot."]
            + (["Seed 42 uses historical metrics, not fresh checkpoint inference."] if seed == 42 else []),
        comment="Post-run evidence completion does not establish full acceptance.", source="completion"), producer="completion")
    bb.write_dossier(status="incomplete:completion")
    return bb


def audit_before_training(bb):
    from agents import DataAuditAgent
    DataAuditAgent().run(bb)
    if not bb.get("data_audit_report").passed:
        raise ValueError("completion pre-training data audit failed")
    bb.record_event("stage_promoted", stage="data_audit", origin="completion")


def train_selected(bb):
    from agents import TrainingAgent, EvaluationAgent
    bb.record_event("stage_started", stage="training")
    if bb.get_optional("train_result") is None:
        TrainingAgent().run(bb)
    EvaluationAgent().run(bb)


def run_controls(bb, bundle):
    config = bb.get("train_config")
    if bb.get_optional("baseline_report") is None:
        bb.record_event("stage_started", stage="baseline")
        report = run_baseline(bundle, bb.root, seed=config.seed, device=config.device,
                              epochs=config.epochs, worker=config.worker)
        bb.put("baseline_report", report, producer="completion_baseline")
    baseline = bb.get("baseline_report")
    evaluation = bb.get("evaluation_report")
    if bb.get_optional("comparison_report") is None:
        bb.put("comparison_report", ComparisonReport(agentic_accuracy=evaluation.accuracy,
            baseline_accuracy=baseline.accuracy,
            accuracy_delta=round(evaluation.accuracy - baseline.accuracy, 6),
            common_seed=config.seed, common_split_fingerprint=baseline.split_fingerprint,
            baseline_met_or_exceeded=evaluation.accuracy >= baseline.accuracy), producer="completion")
    if bb.get_optional("ablation_report") is None:
        completed = [item.scenario for item in bb.artefacts.values() if isinstance(item, CompletedAblation)]
        def save_scenario(scenario):
            bb.put("completed_" + scenario.name, CompletedAblation(scenario=scenario), producer="completion_ablation")
            bb.write_dossier(status="incomplete:completion")
        bb.record_event("stage_started", stage="ablation")
        report = run_representation_ablations(bundle, bb.root, template_config=config,
            completed_scenarios=completed, on_scenario=save_scenario)
        bb.put("ablation_report", report, producer="completion_ablation")


def save_manifest(root, manifest, stores):
    atomic_json(root / "completion_manifest.json", manifest.model_dump(mode="json"))
    if 42 in stores:
        stores[42].put("completion_manifest", manifest, producer="completion")


def write_reports(root, manifest, stores):
    runs = []
    for seed, bb in sorted(stores.items()):
        evaluation, baseline = bb.get_optional("evaluation_report"), bb.get_optional("baseline_report")
        complete = (manifest.operations.get(f"seed_{seed}") == "completed"
                    and evaluation is not None and baseline is not None
                    and bb.get_optional("ablation_report") is not None)
        runs.append({"seed": seed, "run_root": str(bb.root),
            "status": "completed_with_warning" if complete else "incomplete",
            "evaluation_origin": "historical_2026-10-08" if seed == 42 else "new_training",
            "agentic_accuracy": evaluation.accuracy if evaluation else None,
            "agentic_macro_f1": evaluation.macro_f1 if evaluation else None,
            "baseline_accuracy": baseline.accuracy if baseline else None,
            "baseline_macro_f1": baseline.macro_f1 if baseline else None,
            "accuracy_delta": round(evaluation.accuracy - baseline.accuracy, 6) if evaluation and baseline else None,
            "macro_f1_delta": round(evaluation.macro_f1 - baseline.macro_f1, 6) if evaluation and baseline else None,
            "ablations": bb.get("ablation_report").model_dump(mode="json") if bb.get_optional("ablation_report") else None})
    completed = [item["seed"] for item in runs if item["status"].startswith("completed")]
    summary = {"experiment_id": root.name, "completion_source": manifest.source_run,
               "requested_seeds": SEEDS, "completed_seeds": completed,
               "aggregate_valid": completed == SEEDS, "research_timing": "post_training",
               "completion_valid": completed == SEEDS and manifest.operations.get("research") == "completed",
               "runs": runs, "errors": manifest.errors,
               "historical_checkpoint_verified": manifest.historical_checkpoint_verified}
    for metric in ("agentic_accuracy", "agentic_macro_f1", "baseline_accuracy", "baseline_macro_f1"):
        values = [item[metric] for item in runs if item["seed"] in completed and item[metric] is not None]
        summary[metric + "_mean"] = statistics.mean(values) if values else None
        summary[metric + "_population_std"] = statistics.pstdev(values) if values else None
    atomic_json(root / "experiment_summary.json", summary)
    lines = ["# Post-training completion", "", f"Source: `{manifest.source_run}`", "",
             "Live research was retrieved after the original training and did not change model selection.",
             "Seed 42 uses archived metrics; seeds 47 and 72 use new training of the frozen configuration.",
             f"Original checkpoint verified: {manifest.historical_checkpoint_verified}.",
             "Missing historical evidence: " + (", ".join(manifest.historical_missing_files) or "none"), "",
             "| Seed | Origin | Accuracy | Macro-F1 | Baseline accuracy | Baseline F1 | Accuracy delta | F1 delta | Status |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for item in runs:
        lines.append("| " + " | ".join(str(item[key]) for key in ("seed", "evaluation_origin", "agentic_accuracy",
                     "agentic_macro_f1", "baseline_accuracy", "baseline_macro_f1", "accuracy_delta", "macro_f1_delta", "status")) + " |")
        if item["ablations"]:
            lines.extend(["", f"## Seed {item['seed']} representations", "",
                          "| Scenario | Accuracy | Macro-F1 |", "| --- | --- | --- |"])
            lines.extend(f"| {s['name']} | {s['accuracy']} | {s['macro_f1']} |" for s in item["ablations"]["scenarios"])
    lines.extend(["", "## Aggregate (population standard deviation)", "",
                  "These are partial descriptive statistics until aggregate_valid is true."])
    for metric in ("agentic_accuracy", "agentic_macro_f1", "baseline_accuracy", "baseline_macro_f1"):
        lines.append(f"- {metric}: {summary[metric + '_mean']} ± {summary[metric + '_population_std']}")
    lines.extend(["", "## Limitations", "", "Historical acceptance warnings remain in source_snapshot. "
                  "Completing these procedures does not imply all acceptance criteria pass.",
                  "Errors: " + json.dumps(manifest.errors)])
    (root / "completion_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for item in runs:
        bb = stores[item["seed"]]
        bb.write_dossier(status=item["status"])
        write_acceptance(bb)
    return summary


def execute(args):
    if args.continue_run:
        root = Path(args.continue_run).resolve()
        manifest = CompletionManifest.model_validate(json.loads((root / "completion_manifest.json").read_text(encoding="utf-8")))
        state_root = root / "seed_42"
        if (state_root / "dossier.json").exists():
            recorded = Blackboard.open(state_root).get_optional("completion_manifest")
            if recorded is not None:
                # The append-only store is authoritative, including after an interrupted state write.
                manifest = recorded
        source = root / "source_snapshot"
        if file_inventory(source) != manifest.source_files:
            raise ValueError("source snapshot changed since completion started")
        if manifest.completion_code_sha256 != code_tree_sha256(Path(__file__).parent):
            raise ValueError("completion code changed; cannot continue existing results")
        if args.device and args.device != manifest.device:
            raise ValueError("cannot change device when continuing")
        source_bb, _, _ = inspect_source(source)
        device = manifest.device
    else:
        source = Path(args.source_run).resolve() if args.source_run else latest_source(Path(__file__).parent / "runs")
        source_bb, missing, verified = inspect_source(source)
        device = args.device or source_bb.get("train_config").device
        root = Path(args.output_root).resolve() if args.output_root else source.parent / (
            "pathmnist_completion_" + utc_now().replace(":", "").replace("+", "_"))
        if root == source or source in root.parents or root in source.parents:
            raise ValueError("completion output must be independent of source")
        manifest = CompletionManifest(source_run=str(source), source_seed=42,
            source_files=file_inventory(source), source_code_sha256=source_bb.get("run_manifest").code_sha256,
            completion_code_sha256=code_tree_sha256(Path(__file__).parent), device=device,
            historical_checkpoint_verified=verified, historical_missing_files=missing)
    if args.dry_run:
        print(json.dumps({"source": str(source), "output": str(root), "device": device,
              "seeds": SEEDS, "new_trainings": 14, "research_timing": "post_training",
              "historical_missing_files": manifest.historical_missing_files}, indent=2))
        return 0
    if not args.continue_run:
        if root.exists():
            raise FileExistsError("output already exists; use --continue")
        root.mkdir(parents=True)
        shutil.copytree(source, root / "source_snapshot")
        for path in (root / "source_snapshot").rglob("*"):
            if path.is_file() and path.suffix in TEXT_SUFFIXES:
                normalized = portable_bytes(path)
                if normalized != path.read_bytes():
                    path.write_bytes(normalized)
        if file_inventory(root / "source_snapshot") != manifest.source_files:
            raise ValueError("source changed while snapshot was copied")
        source_bb, _, _ = inspect_source(root / "source_snapshot")
        save_manifest(root, manifest, {})
    stores = {}
    for seed in SEEDS:
        if (root / f"seed_{seed}" / "dossier.json").exists():
            stores[seed] = Blackboard.open(root / f"seed_{seed}")
            issues = check_integrity(stores[seed].root)
            if issues:
                raise ValueError("completion integrity failure: " + "; ".join(issues))
    try:
        bundle = load_dataset(source_bb, args.data_root)
        manifest.operations["dataset"] = "completed"
        manifest.errors.pop("dataset", None)
    except Exception as exc:
        manifest.operations["dataset"] = "failed"
        manifest.errors["dataset"] = str(exc)
        save_manifest(root, manifest, stores)
        write_reports(root, manifest, stores)
        raise
    from resources import RunResources
    from ml import common_split_fingerprint
    for seed in SEEDS:
        bb = stores.get(seed)
        if bb is None:
            seed_root = root / f"seed_{seed}"
            if seed_root.exists():
                raise ValueError(f"unexpected partial store at {seed_root}")
            # Failed initialization stays isolated; continuation can create a fresh staging directory.
            staging = root / (f".seed_{seed}_initializing_" + utc_now().replace(":", "").replace("+", "_"))
            initialize_seed(staging, source_bb, bundle, seed, device)
            staging.rename(seed_root)
            bb = Blackboard.open(seed_root)
            stores[seed] = bb
        key = f"seed_{seed}"
        if seed == 42 and manifest.operations.get("research") != "completed":
            try:
                retrieve_live(bb, root)
                manifest.operations["research"] = "completed"
                manifest.errors.pop("research", None)
            except Exception as exc:
                manifest.operations["research"] = "failed"
                manifest.errors["research"] = str(exc)
            save_manifest(root, manifest, stores)
        if seed != 42 and stores[42].get_optional("prior_art_brief"):
            brief = stores[42].get("prior_art_brief")
            shutil.copytree(stores[42].root / "blobs" / "literature", bb.root / "blobs" / "literature", dirs_exist_ok=True)
            if bb.get_optional("prior_art_brief") is None:
                bb.put("prior_art_brief", brief, producer="completion_research")
        if manifest.operations.get(key) == "completed":
            continue
        manifest.operations[key] = "running"
        save_manifest(root, manifest, stores)
        try:
            with RunResources(bb.root, audit=bb.record_event).activate():
                audit_before_training(bb)
                if bb.get_optional("evaluation_report") is None:
                    if seed == 42:
                        bb.put("evaluation_report", source_bb.get("evaluation_report"), producer="historical_metrics")
                        bb.record_event("historical_metrics_imported", checkpoint_verified=manifest.historical_checkpoint_verified,
                                        split_fingerprint=common_split_fingerprint(bundle))
                    else:
                        train_selected(bb)
                run_controls(bb, bundle)
            manifest.operations[key] = "completed"
            manifest.errors.pop(key, None)
        except Exception as exc:
            manifest.operations[key] = "failed"
            manifest.errors[key] = str(exc)
            bb.record_event("completion_failed", error=str(exc))
        save_manifest(root, manifest, stores)
        write_reports(root, manifest, stores)
    summary = write_reports(root, manifest, stores)
    print(f"Completion report: {root / 'completion_report.md'}")
    return 0 if summary["completion_valid"] else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--source-run", help="original experiment directory (default: latest original run)")
    group.add_argument("--continue", dest="continue_run", help="existing completion experiment directory")
    parser.add_argument("--output-root", help="new completion experiment directory")
    parser.add_argument("--data-root", help="official PathMNIST archive cache")
    parser.add_argument("--device", choices=("cpu", "cuda"), help="default: source device")
    parser.add_argument("--dry-run", action="store_true", help="inspect only; no downloads, training or files written")
    args = parser.parse_args(argv)
    if args.continue_run and args.output_root:
        parser.error("--output-root cannot be used with --continue")
    try:
        return execute(args)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Completion failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())

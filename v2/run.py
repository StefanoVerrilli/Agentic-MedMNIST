"""Command-line entry point for sequential PathMNIST experiments."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents import (
    ArchitectureResearchAgent,
    AbstentionOODAgent,
    EvaluationAgent,
    ExperimentDesignAgent,
    FrozenConfigurationAgent,
    IngestionAgent,
    ModelSearchAgent,
    PreprocessingAgent,
    ProfilingAmbiguityAgent,
    ReportingAgent,
    ReviewerConsistencyAgent,
    TrainingAgent,
)
from baseline import run_baseline, run_representation_ablations
from contracts import (
    AnomalyReport,
    BestConfiguration,
    Blackboard,
    ComparisonReport,
    RunManifest,
    utc_now,
)
from llm import OllamaReasoner, ollama_model_digest
from ml import common_split_fingerprint
from orchestrator import Orchestrator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Hybrid agentic model-search experiment on PathMNIST. All agents and "
            "all Ollama requests execute sequentially."
        )
    )
    parser.add_argument("--output-root", default="runs", help="experiment output root")
    parser.add_argument("--data-root", default=None, help="MedMNIST cache directory")
    parser.add_argument(
        "--seeds",
        default="42",
        help="comma-separated seeds; use 42,43,44 for repeated-run variance",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="stratified 6000/1200/1800 subset for a fast demonstration",
    )
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-epochs", type=int, default=15)
    parser.add_argument(
        "--no-search",
        action="store_true",
        help="disable iterative model search and use one bounded design decision",
    )
    parser.add_argument("--search-trials", type=int, default=6)
    parser.add_argument("--search-rounds", type=int, default=2)
    parser.add_argument("--search-epochs", type=int, default=3)
    parser.add_argument("--accuracy-tolerance", type=float, default=0.005)
    parser.add_argument(
        "--search-per-seed",
        action="store_true",
        help=(
            "repeat hyperparameter search for every seed; by default the first "
            "validation-selected configuration is frozen for later seeds"
        ),
    )
    parser.add_argument("--stage-retries", type=int, default=1)
    parser.add_argument("--target-coverage", type=float, default=0.80)
    parser.add_argument(
        "--ablation-suite",
        action="store_true",
        help="run three deterministic representation ablations",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="force validated heuristic decisions even if Ollama is configured",
    )
    parser.add_argument(
        "--ollama-base",
        default=os.environ.get("AGENTIC_LLM_BASE"),
        help="Ollama root URL, normally http://localhost:11434",
    )
    parser.add_argument(
        "--ollama-model",
        default=os.environ.get("AGENTIC_LLM_MODEL", "qwen2.5:7b"),
    )
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=float(os.environ.get("AGENTIC_LLM_TIMEOUT", "90")),
    )
    parser.add_argument(
        "--llm-retries",
        type=int,
        default=int(os.environ.get("AGENTIC_LLM_RETRIES", "1")),
    )
    parser.add_argument(
        "--keep-alive",
        default=os.environ.get("AGENTIC_LLM_KEEP_ALIVE", "5m"),
    )
    parser.add_argument(
        "--require-llm",
        action="store_true",
        help="fail instead of using a heuristic after Ollama errors",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="require PathMNIST to exist already in --data-root",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        seeds = parse_seeds(args.seeds)
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    if args.offline:
        args.ollama_base = None
    if args.require_llm and not args.ollama_base:
        parser.error(
            "--require-llm needs --ollama-base (and cannot be used with --offline)"
        )

    experiment_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    experiment_root = Path(args.output_root) / f"pathmnist_{experiment_id}"
    experiment_root.mkdir(parents=True, exist_ok=False)
    limits = resolve_limits(args)
    summaries: list[dict[str, Any]] = []
    frozen_best: BestConfiguration | None = None

    for seed in seeds:
        run_root = experiment_root / f"seed_{seed}"
        replay = None if args.search_per_seed else frozen_best
        summary = run_once(args, seed, run_root, limits, frozen_best=replay)
        selected = summary.pop("_best_configuration", None)
        if frozen_best is None and isinstance(selected, BestConfiguration):
            frozen_best = selected
        summaries.append(summary)

    successful = [
        item
        for item in summaries
        if str(item.get("status", "")).startswith("completed")
    ]
    accuracies = [float(item["agentic_accuracy"]) for item in successful]
    task_best = select_task_configuration(summaries, args.accuracy_tolerance)
    task_config: dict[str, Any] | None = None
    if task_best is not None:
        source = Path(str(task_best["best_config_path"]))
        target = experiment_root / "best_task_config.yaml"
        shutil.copyfile(source, target)
        task_config = {
            "selected_from_seed": task_best["seed"],
            "selection_basis": "validation_accuracy_then_macro_f1",
            "validation_accuracy": task_best["best_validation_accuracy"],
            "validation_macro_f1": task_best["best_validation_macro_f1"],
            "path": str(target),
            "sha256": file_sha256(target),
            "test_metrics_used": False,
        }
    baseline_accuracies = [
        float(item["baseline_accuracy"])
        for item in successful
        if item.get("baseline_accuracy") is not None
    ]
    experiment_summary = {
        "experiment_id": experiment_id,
        "dataset": "pathmnist",
        "created_at": utc_now(),
        "sequential_ollama_calls": True,
        "runs": summaries,
        "agentic_accuracy_mean": round(statistics.mean(accuracies), 6)
        if accuracies
        else None,
        "agentic_accuracy_population_std": round(statistics.pstdev(accuracies), 6)
        if len(accuracies) > 1
        else 0.0
        if accuracies
        else None,
        "baseline_accuracy_mean": (
            round(statistics.mean(baseline_accuracies), 6)
            if baseline_accuracies
            else None
        ),
        "task_best_configuration": task_config,
    }
    summary_path = experiment_root / "experiment_summary.json"
    summary_path.write_text(
        json.dumps(experiment_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"\nExperiment summary: {summary_path}")
    return 0 if len(successful) == len(seeds) else 2


def run_once(
    args: argparse.Namespace,
    seed: int,
    run_root: Path,
    limits: dict[str, int | None],
    *,
    frozen_best: BestConfiguration | None = None,
) -> dict[str, Any]:
    bb = Blackboard(run_root, run_id=run_root.name)
    reasoner = OllamaReasoner(
        base_url=args.ollama_base,
        model=args.ollama_model,
        timeout=args.llm_timeout,
        retries=args.llm_retries,
        required=args.require_llm,
        keep_alive=args.keep_alive,
        seed=seed,
    )
    bb.put(
        "run_manifest",
        RunManifest(
            run_id=bb.run_id,
            dataset="pathmnist",
            seed=seed,
            created_at=utc_now(),
            python_version=sys.version.split()[0],
            platform=platform.platform(),
            package_versions=package_versions(),
            llm_mode="ollama" if reasoner.enabled else "heuristic",
            llm_model=reasoner.model if reasoner.enabled else "none",
            llm_model_digest=ollama_model_digest(
                args.ollama_base if reasoner.enabled else None,
                reasoner.model,
                timeout=min(args.llm_timeout, 5.0),
            ),
            code_sha256=code_tree_sha256(Path(__file__).resolve().parent),
        ),
        producer="run",
    )
    reviewer = ReviewerConsistencyAgent(reasoner)
    pipeline = [
        IngestionAgent(
            seed=seed,
            train_limit=limits["train"],
            val_limit=limits["val"],
            test_limit=limits["test"],
            data_root=args.data_root,
            download=not args.no_download,
        ),
        ProfilingAmbiguityAgent(reasoner),
        PreprocessingAgent(reasoner),
        ArchitectureResearchAgent(reasoner),
    ]
    if args.no_search:
        pipeline.append(
            ExperimentDesignAgent(
                reasoner,
                seed=seed,
                device=args.device,
                max_epochs=args.max_epochs,
            )
        )
    elif frozen_best is not None:
        pipeline.append(
            FrozenConfigurationAgent(
                frozen_best,
                seed=seed,
                device=args.device,
                data_root=args.data_root,
                train_limit=limits["train"],
                val_limit=limits["val"],
                test_limit=limits["test"],
                download=not args.no_download,
            )
        )
    else:
        pipeline.append(
            ModelSearchAgent(
                reasoner,
                seed=seed,
                device=args.device,
                max_trials=args.search_trials,
                rounds=args.search_rounds,
                search_epochs=args.search_epochs,
                final_epochs=args.max_epochs,
                accuracy_tolerance=args.accuracy_tolerance,
                data_root=args.data_root,
                train_limit=limits["train"],
                val_limit=limits["val"],
                test_limit=limits["test"],
                download=not args.no_download,
            )
        )
    pipeline.extend(
        [
            TrainingAgent(),
            EvaluationAgent(),
            AbstentionOODAgent(target_validation_coverage=args.target_coverage),
            ReportingAgent(),
        ]
    )
    orchestrator = Orchestrator(
        pipeline, reviewer, max_stage_retries=args.stage_retries
    )
    print(f"\n=== Agentic PathMNIST pipeline | seed={seed} ===")
    orchestrator.run(bb)
    if orchestrator.status != "completed":
        return {
            "seed": seed,
            "status": orchestrator.status,
            "run_root": str(run_root),
            "agentic_accuracy": None,
            "baseline_accuracy": None,
        }

    bundle = bb.get_blob("raw_dataset")
    config = bb.get("train_config")
    evaluation = bb.get("evaluation_report")
    print("\n=== Conventional baseline on the same split ===")
    baseline = run_baseline(
        bundle,
        bb.root,
        seed=seed,
        device=config.device,
        epochs=config.epochs,
    )
    bb.put("baseline_report", baseline, producer="baseline")
    fingerprint = common_split_fingerprint(bundle)
    comparison = ComparisonReport(
        agentic_accuracy=evaluation.accuracy,
        baseline_accuracy=baseline.accuracy,
        accuracy_delta=round(evaluation.accuracy - baseline.accuracy, 6),
        common_seed=seed,
        common_split_fingerprint=fingerprint,
        baseline_met_or_exceeded=evaluation.accuracy >= baseline.accuracy,
    )
    bb.put("comparison_report", comparison, producer="run")
    comparison_review = reviewer.review(bb, "comparison")
    post_reviews = [comparison_review]
    if comparison_review.action == "stop":
        status = "vetoed:comparison"
        bb.write_dossier(status=status)
        return metric_summary(
            seed, status, run_root, evaluation, baseline.accuracy, None, bb
        )

    if args.ablation_suite:
        print("\n=== Three-scenario representation ablation ===")
        ablation = run_representation_ablations(bundle, bb.root, template_config=config)
        bb.put("ablation_report", ablation, producer="ablation")
        ablation_review = reviewer.review(bb, "ablation")
        post_reviews.append(ablation_review)
        if ablation_review.action == "stop":
            status = "vetoed:ablation"
            bb.write_dossier(status=status)
            return metric_summary(
                seed, status, run_root, evaluation, baseline.accuracy, None, bb
            )

    ReportingAgent().run(bb)
    reporting_review = reviewer.review(bb, "reporting")
    post_reviews.append(reporting_review)
    if reporting_review.action == "stop":
        final_status = "vetoed:reporting"
    else:
        all_reviews = [
            value
            for name, value in bb.artefacts.items()
            if name.startswith("anomaly_") and isinstance(value, AnomalyReport)
        ]
        final_status = (
            "completed_with_warning"
            if any(report.severity == "warning" for report in all_reviews)
            else "completed"
        )
    bb.write_dossier(status=final_status)
    abstention = bb.get("abstention_report")
    print(f"agentic accuracy : {evaluation.accuracy:.4f}")
    print(f"baseline accuracy: {baseline.accuracy:.4f}")
    print(f"delta            : {comparison.accuracy_delta:+.4f}")
    print(
        f"abstention       : coverage={abstention.test_coverage:.4f}, "
        f"covered_accuracy={abstention.accuracy_on_covered:.4f}"
    )
    print(f"run dossier      : {bb.root / 'dossier.json'}")
    return metric_summary(
        seed,
        final_status,
        run_root,
        evaluation,
        baseline.accuracy,
        baseline.macro_f1,
        bb,
    )


def metric_summary(
    seed: int,
    status: str,
    run_root: Path,
    evaluation: Any,
    baseline_accuracy: float,
    baseline_macro_f1: float | None,
    bb: Blackboard,
) -> dict[str, Any]:
    best = bb.get_optional("best_configuration")
    result = {
        "seed": seed,
        "status": status,
        "run_root": str(run_root),
        "agentic_accuracy": evaluation.accuracy,
        "agentic_macro_f1": evaluation.macro_f1,
        "baseline_accuracy": baseline_accuracy,
        "baseline_macro_f1": baseline_macro_f1,
        "accuracy_delta": round(evaluation.accuracy - baseline_accuracy, 6),
    }
    if isinstance(best, BestConfiguration):
        result.update(
            {
                "best_validation_accuracy": best.validation_accuracy,
                "best_validation_macro_f1": best.validation_macro_f1,
                "best_config_hash": best.selected_config_hash,
                "best_config_path": str(bb.root / best.lightning_config_path),
                "_best_configuration": best,
            }
        )
    return result


def resolve_limits(args: argparse.Namespace) -> dict[str, int | None]:
    quick_defaults = {"train": 6000, "val": 1200, "test": 1800}
    result: dict[str, int | None] = {}
    for split in ("train", "val", "test"):
        explicit = getattr(args, f"{split}_limit")
        result[split] = (
            explicit
            if explicit is not None
            else (quick_defaults[split] if args.quick else None)
        )
    return result


def parse_seeds(value: str) -> list[int]:
    try:
        seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(
            "--seeds must be comma-separated non-negative integers"
        ) from exc
    if not seeds or any(seed < 0 for seed in seeds):
        raise ValueError("--seeds must contain at least one non-negative integer")
    if len(set(seeds)) != len(seeds):
        raise ValueError("--seeds contains duplicates")
    return seeds


def validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.max_epochs <= 50:
        raise ValueError("--max-epochs must be between 1 and 50")
    if not 1 <= args.search_trials <= 24:
        raise ValueError("--search-trials must be between 1 and 24")
    if not 1 <= args.search_rounds <= 4:
        raise ValueError("--search-rounds must be between 1 and 4")
    if args.search_trials < args.search_rounds:
        raise ValueError("--search-trials must be at least --search-rounds")
    if not 1 <= args.search_epochs <= 12:
        raise ValueError("--search-epochs must be between 1 and 12")
    if args.search_epochs > args.max_epochs:
        raise ValueError("--search-epochs cannot exceed --max-epochs")
    if not 0.0 <= args.accuracy_tolerance <= 0.05:
        raise ValueError("--accuracy-tolerance must be between 0 and 0.05")
    if args.stage_retries < 0:
        raise ValueError("--stage-retries cannot be negative")
    if args.llm_retries < 0:
        raise ValueError("--llm-retries cannot be negative")
    if args.llm_timeout <= 0:
        raise ValueError("--llm-timeout must be positive")
    if not 0.0 < args.target_coverage <= 1.0:
        raise ValueError("--target-coverage must be in (0, 1]")
    for name in ("train_limit", "val_limit", "test_limit"):
        value = getattr(args, name)
        if value is not None and value != 0 and value < 9:
            raise ValueError(f"--{name.replace('_', '-')} must be 0 or at least 9")


def package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for distribution in (
        "medmnist",
        "torch",
        "lightning",
        "jsonargparse",
        "numpy",
        "pydantic",
        "scikit-learn",
    ):
        try:
            result[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            result[distribution] = "not-installed"
    return result


def select_task_configuration(
    summaries: list[dict[str, Any]], accuracy_tolerance: float
) -> dict[str, Any] | None:
    candidates = [
        item
        for item in summaries
        if item.get("best_validation_accuracy") is not None
        and item.get("best_config_path")
    ]
    if not candidates:
        return None
    maximum = max(float(item["best_validation_accuracy"]) for item in candidates)
    eligible = [
        item
        for item in candidates
        if float(item["best_validation_accuracy"]) >= maximum - accuracy_tolerance
    ]
    return max(
        eligible,
        key=lambda item: (
            float(item["best_validation_macro_f1"]),
            float(item["best_validation_accuracy"]),
            -int(item["seed"]),
        ),
    )


def code_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    paths = list(root.glob("*.py")) + list((root / "tests").glob("*.py"))
    paths += list((root / "configs").glob("*.yaml"))
    paths += [root / "requirements.txt"]
    for path in sorted((path for path in paths if path.is_file()), key=str):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())

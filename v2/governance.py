"""Evidence-based acceptance, replay comparison and independent review tools."""
from __future__ import annotations

import argparse
import ast
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from contracts import (AcceptanceCriterion, AcceptanceReport, PriorArtBrief, HumanReviewDecision,
                       HumanReviewResolution, _sha256_json, sha256_file, utc_now)
from research import canonical_hash, citation_issues, contained_path


def read_artefact(root: Path, name: str) -> dict[str, Any]:
    files = sorted((root / "artefacts").glob(f"*_{name}_v*.json"))
    if not files:
        raise ValueError(f"missing artefact: {name}")
    envelope = json.loads(files[-1].read_text(encoding="utf-8"))
    if _sha256_json(envelope["payload"]) != envelope["payload_sha256"]:
        raise ValueError(f"artefact checksum mismatch: {name}")
    return envelope["payload"]


def evidence_references(root: Path, value: Any, location: str = "payload"):
    """Walk nested evidence, resolving only the declared frozen-config scope.

    Historical frozen_best references belong to the original sibling seed.
    New runs retain a local snapshot so they remain independently portable.
    """
    if isinstance(value, dict):
        for path_key, hash_key in (
            ("path", "sha256"), ("checkpoint_path", "checkpoint_sha256"),
            ("lightning_config_path", "lightning_config_sha256"),
            ("ood_evidence_path", "ood_evidence_sha256"),
            ("snapshot_path", "snapshot_sha256"),
        ):
            if value.get(path_key) and value.get(hash_key):
                yield (f"{location}.{path_key}",
                       contained_path(root, value[path_key].replace("\\", "/")), value[hash_key])
        for key, item in value.items():
            scope = root
            if key == "frozen_best" and item and value.get("frozen_reference_scope", "origin_seed") == "origin_seed":
                seed = item["train_config"]["seed"]
                if type(seed) is not int or seed < 0:
                    raise ValueError("invalid frozen configuration origin seed")
                scope = contained_path(root.parent, f"seed_{seed}")
            yield from evidence_references(scope, item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from evidence_references(root, item, f"{location}[{index}]")


def check_integrity(root: Path) -> list[str]:
    issues = []
    for path in sorted((root / "artefacts").glob("*.json")):
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            payload = envelope["payload"]
            if _sha256_json(payload) != envelope["payload_sha256"]:
                raise ValueError("payload checksum mismatch")
            for location, evidence, expected in evidence_references(root, payload):
                if sha256_file(evidence) != expected:
                    issues.append(f"{path.name}: {location} checksum mismatch")
        except (ValueError, OSError, KeyError) as exc:
            issues.append(f"{path.name}: {exc}")
    for path in sorted((root / "blobs" / "reasoning").glob("*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            if canonical_hash(document["payload"]) != document["sha256"]:
                raise ValueError("reasoning snapshot checksum mismatch")
        except (ValueError, OSError, KeyError) as exc:
            issues.append(f"{path.name}: {exc}")
    return issues


def coupling_issues(path: Path) -> list[str]:
    """Check direct construction/calls between agent classes in agents.py."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {node.name for node in tree.body if isinstance(node, ast.ClassDef) and node.name.endswith("Agent")}
    issues = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in names:
            for call in ast.walk(node):
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in names:
                    issues.append(f"{node.name} directly constructs/calls {call.func.id}")
    return issues


def assess(root: Path, *, validation_path: Path | None = None,
           independent_path: Path | None = None) -> AcceptanceReport:
    root = root.resolve()
    dossier = json.loads((root / "dossier.json").read_text(encoding="utf-8"))
    # Read authoritative immutable envelopes rather than trust the dossier copy.
    artifacts = {}
    for path in sorted((root / "artefacts").glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        artifacts[document["artefact"]] = document["payload"]
    criteria: list[AcceptanceCriterion] = []
    def add(requirement: str, passed: bool | None, evidence: list[str], detail: str) -> None:
        criteria.append(AcceptanceCriterion(requirement=requirement,
            status="pending" if passed is None else "passed" if passed else "failed", evidence=evidence, detail=detail))

    integrity = check_integrity(root)
    add("WP1_TYPED_VERSIONED_HANDOFFS", not integrity, ["artefacts/", "blobs/"],
        "Every persisted payload and referenced file is checksummed." if not integrity else "; ".join(integrity))
    coupling = coupling_issues(Path(__file__).with_name("agents.py"))
    add("WP1_ZERO_DIRECT_AGENT_CALLS", not coupling, ["agents.py"],
        "AST check of direct agent-class calls." if not coupling else "; ".join(coupling))
    audit = artifacts.get("data_audit_report")
    events = [json.loads(line) for line in (root / "decision_log.jsonl").read_text(encoding="utf-8").splitlines()]
    audit_gates = [e["sequence"] for e in events if e["event"] == "stage_promoted" and e.get("stage") == "data_audit"]
    training_events = [e["sequence"] for e in events if e["event"] in {"search_trial_started", "stage_started"}
                       and (e["event"] == "search_trial_started" or e.get("stage") == "training")]
    audited_before_training = bool(audit_gates and training_events and max(audit_gates) < min(training_events))
    add("WP2_PRETRAINING_LEAKAGE_AUDIT", bool(audit and audit["passed"] and audited_before_training),
        ["data_audit_report", "decision_log.jsonl"], "Actual train/validation evidence audited before any training.")
    brief = artifacts.get("prior_art_brief")
    citation_errors = citation_issues(PriorArtBrief.model_validate(brief), root) if brief else ["missing brief"]
    add("WP2_CITED_IDEAS", not citation_errors, ["prior_art_brief", "blobs/literature/"],
        "Exact supporting quotes resolve to checksummed snapshots; recommendations remain hypotheses."
        if not citation_errors else "; ".join(citation_errors))
    live = bool(brief and all(source["origin"] == "arxiv_api" for source in brief["sources"]))
    add("WP2_WEB_RETRIEVAL", live, ["prior_art_brief"], "Curated excerpts alone do not demonstrate web retrieval.")
    best = artifacts.get("best_configuration")
    search = artifacts.get("search_report")
    config = artifacts.get("train_config")
    add("WP3_VALIDATION_ONLY_SELECTION", bool(best and not best["test_metrics_used"]
        and (search is None or not search["test_metrics_used"])), ["best_configuration", "search_report"],
        "Test metrics are excluded from search and configuration selection.")
    add("WP4_BASELINE_COMPARISON", "comparison_report" in artifacts, ["comparison_report", "baseline_report"],
        "Conventional baseline uses the same seed and selected splits; negative deltas are retained.")
    add("WP4_REPRESENTATION_ABLATIONS", "ablation_report" in artifacts, ["ablation_report"], "Three bounded representation scenarios.")
    summary_path = root.parent / "experiment_summary.json"
    repeated_seeds = None
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        settings = artifacts.get("run_configuration", {}).get("parameters", {})
        repeated_seeds = bool(summary.get("aggregate_valid") and len(summary.get("completed_seeds", [])) >= 3
                              and not settings.get("search_per_seed"))
        if repeated_seeds:
            frozen_configs = []
            for seed in summary["completed_seeds"]:
                selected = read_artefact(root.parent / f"seed_{seed}", "best_configuration")
                frozen_configs.append({
                    "training": {k: v for k, v in selected["train_config"].items() if k not in {"seed", "source", "rationale"}},
                    "representation": {k: v for k, v in selected["representation"].items() if k != "rationale"},
                })
            repeated_seeds = all(config == frozen_configs[0] for config in frozen_configs)
    add("WP4_FROZEN_CONFIGURATION_SEED_VARIANCE", repeated_seeds, ["../experiment_summary.json"],
        "At least three completed seeds, with a shared validation-selected configuration.")
    abstention, queue = artifacts.get("abstention_report"), artifacts.get("human_review_queue")
    add("WP5_ABSTENTION_AND_HANDOFF", bool(abstention and queue and queue["count"] == abstention["human_review_count"]),
        ["abstention_report", "human_review_queue"], "Validation-calibrated abstention with a persistent benchmark review queue.")
    reviews = [v for k, v in artifacts.items() if k.startswith("anomaly_")]
    stages = {e["stage"] for e in events if e["event"] == "stage_promoted"}
    reviewed = {review["stage"] for review in reviews}
    add("WP6_STAGE_REVIEWS", bool(stages and stages <= reviewed), ["anomaly_*", "decision_log.jsonl"],
        "Every promoted stage has an independent reviewer report.")
    decisions = [e for e in events if e["event"] in {"llm_decision", "forced_decision"}]
    cached = list((root / "blobs" / "reasoning").glob("*.json"))
    logged = bool(decisions and all(e.get("source") and e.get("stage") for e in decisions))
    add("WP6_DECISION_PROVENANCE", logged, ["decision_log.jsonl", "blobs/reasoning/"],
        "Decision source and stage are recorded; exact prompts/schema/results are retained.")
    fault_rate = None
    validation_ok = None
    if validation_path:
        document = json.loads(validation_path.read_text(encoding="utf-8"))
        payload = document["payload"]
        if canonical_hash(payload) != document["sha256"]:
            raise ValueError("validation evidence checksum mismatch")
        manifest = artifacts.get("run_manifest", {})
        validation_ok = bool(payload["tests_passed"] and payload["source_sha256"] == manifest.get("code_sha256")
                             and payload.get("source_hash_algorithm", "legacy-native-v1")
                             == manifest.get("code_hash_algorithm", "legacy-native-v1"))
        fault_rate = payload["fault_detection_rate"]
        validation_ok = validation_ok and fault_rate >= 0.9
    add("WP6_FAULT_DETECTION_GE_90_PERCENT", validation_ok, [str(validation_path)] if validation_path else [],
        "Requires passing tests and measured fault injection for this exact source tree.")
    replay_path = root / "replay_comparison.json"
    replay_ok = None
    if replay_path.exists():
        document = json.loads(replay_path.read_text(encoding="utf-8"))
        if canonical_hash(document["payload"]) != document["sha256"]:
            raise ValueError("replay comparison evidence checksum mismatch")
        replay_ok = document["payload"]["matched"]
    add("WP5_FULL_RUN_REPLAY", replay_ok, ["replay_comparison.json"] if replay_path.exists() else [],
        "A transcript alone is not proof that retraining reproduces predictions and metrics.")
    add("WP5_END_TO_END", str(dossier["status"]).startswith("completed"), ["dossier.json"], dossier["status"])
    warnings = sorted({issue for review in reviews for issue in review.get("deterministic_issues", []) + review.get("llm_issues", [])
                       if review["severity"] != "ok"})
    # Semantic review and external demonstration are explicit independent inputs.
    signoff = None
    independent_ok = None
    mitigated = not warnings
    if independent_path:
        evidence = json.loads(independent_path.read_text(encoding="utf-8"))
        run_manifest = artifacts.get("run_manifest", {})
        prediction = artifacts.get("prediction_artifact", {})
        independent_ok = bool(evidence.get("decision") == "approve"
            and evidence.get("run_id") == run_manifest.get("run_id")
            and evidence.get("code_sha256") == run_manifest.get("code_sha256")
            and evidence.get("prediction_sha256") == prediction.get("sha256")
            and evidence.get("implementer") and evidence.get("demonstrator") and evidence.get("reviewer")
            and evidence["implementer"] != evidence["demonstrator"]
            and evidence["implementer"] != evidence["reviewer"]
            and evidence.get("environment_description") and evidence.get("semantic_citation_review") is True)
        attachment = Path(evidence.get("demonstration_path", ""))
        if not attachment.is_absolute():
            attachment = independent_path.parent / attachment
        independent_ok = bool(independent_ok and attachment.is_file()
                              and sha256_file(attachment) == evidence.get("demonstration_sha256"))
        mitigated = set(warnings) <= set(evidence.get("approved_mitigations", []))
        if independent_ok:
            signoff = evidence["reviewer"]
    add("WP6_APPROVED_RISK_MITIGATIONS", mitigated, ["anomaly_*", str(independent_path)] if independent_path else ["anomaly_*"],
        "Every unresolved reviewer issue requires an explicit approved mitigation.")
    add("TRL7_INDEPENDENT_DEMONSTRATION_AND_SIGNOFF", independent_ok,
        [str(independent_path)] if independent_path else [],
        "Requires named independent demonstrator/reviewer, bound predictions and checksummed environment evidence.")
    technical_requirements = {"WP1_TYPED_VERSIONED_HANDOFFS", "WP1_ZERO_DIRECT_AGENT_CALLS", "WP2_PRETRAINING_LEAKAGE_AUDIT",
                              "WP2_CITED_IDEAS", "WP3_VALIDATION_ONLY_SELECTION", "WP4_BASELINE_COMPARISON",
                              "WP5_ABSTENTION_AND_HANDOFF", "WP6_STAGE_REVIEWS", "WP6_DECISION_PROVENANCE", "WP5_END_TO_END"}
    technical = all(c.status == "passed" for c in criteria if c.requirement in technical_requirements)
    ready = all(c.status == "passed" for c in criteria)
    return AcceptanceReport(criteria=criteria, technical_complete=technical, acceptance_ready=ready,
        trl7_evidence_complete=ready and bool(independent_ok), independent_signoff=signoff,
        kpis={"cited_idea_fraction": 1.0 if brief and not citation_errors else 0.0,
              "unresolved_citation_issues": len(citation_errors), "live_source_fraction":
              sum(s["origin"] == "arxiv_api" for s in brief["sources"]) / len(brief["sources"]) if brief else 0.0,
              "fault_detection_rate": fault_rate, "decisions_logged": len(decisions),
              "decision_provenance_fraction": 1.0 if logged else 0.0,
              "reasoning_snapshots": len(cached), "unresolved_reviewer_issues": warnings})


def write_acceptance(bb: Any, **kwargs: Any) -> AcceptanceReport:
    report = assess(bb.root, **kwargs)
    bb.put("acceptance_report", report, producer="acceptance")
    (bb.root / "acceptance_report.json").write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    dossier = json.loads((bb.root / "dossier.json").read_text(encoding="utf-8"))
    bb.write_dossier(status=dossier["status"])
    return report


def compare_replay(original: Path, replay: Path) -> dict[str, Any]:
    import numpy as np
    differences = []
    for root in (original, replay):
        differences.extend(check_integrity(root))
    for name in ("evaluation_report", "abstention_report"):
        first, second = read_artefact(original, name), read_artefact(replay, name)
        for key in ("ood_evidence_path", "ood_evidence_sha256"):
            first.pop(key, None)
            second.pop(key, None)
        if first != second:
            differences.append(f"{name} differs")
    paths = [contained_path(root, read_artefact(root, "prediction_artifact")["path"]) for root in (original, replay)]
    with np.load(paths[0], allow_pickle=False) as first, np.load(paths[1], allow_pickle=False) as second:
        if set(first.files) != set(second.files):
            differences.append("prediction array keys differ")
        else:
            for name in first.files:
                if not np.array_equal(first[name], second[name]):
                    differences.append(f"prediction array {name} differs")
    payload = {"original": str(original.resolve()), "replay": str(replay.resolve()),
               "matched": not differences, "differences": differences,
               "comparison": "exact predictions and metrics", "created_at": utc_now()}
    (replay / "replay_comparison.json").write_text(json.dumps(
        {"payload": payload, "sha256": canonical_hash(payload)}, indent=2) + "\n", encoding="utf-8")
    return payload


def resolve_human_review(root: Path, responses: Path) -> HumanReviewResolution:
    from contracts import Blackboard
    bb = Blackboard.open(root)
    reference = bb.get("human_review_queue")
    path = contained_path(bb.root, reference.path)
    if sha256_file(path) != reference.sha256:
        raise ValueError("human-review queue checksum mismatch")
    queue = json.loads(path.read_text(encoding="utf-8"))
    case_ids = {case["case_id"] for case in queue["cases"]}
    payload = json.loads(responses.read_text(encoding="utf-8"))
    decisions = [HumanReviewDecision.model_validate(item) for item in payload["decisions"]]
    if len({item.case_id for item in decisions}) != len(decisions):
        raise ValueError("duplicate human-review responses")
    if any(item.case_id not in case_ids for item in decisions):
        raise ValueError("response refers to a case outside the review queue")
    previous = bb.get_optional("human_review_resolution")
    latest = {item.case_id: item for item in previous.decisions} if previous else {}
    latest.update({item.case_id: item for item in decisions})
    pending = len(case_ids) - sum(item.decision != "defer" for item in latest.values())
    report = HumanReviewResolution(queue_sha256=reference.sha256, decisions=list(latest.values()),
                                    pending_count=pending, created_at=utc_now())
    bb.put("human_review_resolution", report, producer="human_reviewer")
    bb.record_event("human_review_recorded", responses_sha256=sha256_file(responses),
                    cases=len(decisions), pending=pending)
    status = json.loads((bb.root / "dossier.json").read_text(encoding="utf-8"))["status"]
    bb.write_dossier(status=status)
    return report


def validate(output: Path) -> bool:
    from agents import ReviewerConsistencyAgent
    from contracts import Blackboard
    from llm import OllamaReasoner
    from run import code_tree_sha256
    from tests.test_fault_injection import FaultInjectionTests

    root = Path(__file__).resolve().parent
    suite = unittest.defaultTestLoader.discover(str(root / "tests"), top_level_dir=str(root))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    injected = FaultInjectionTests()
    reviewer = ReviewerConsistencyAgent(OllamaReasoner(base_url=None))
    faults = []
    for name in ("_wrong_dataset", "_wrong_channels", "_empty_split", "_bad_profile_counts", "_non_official_split",
                 "_unsafe_augmentation", "_leaky_statistics", "_seed_mismatch", "_bad_confusion_matrix", "_bad_abstention_sum"):
        with tempfile.TemporaryDirectory() as temporary:
            bb = Blackboard(temporary)
            stage = getattr(injected, name)(bb)
            findings = reviewer._deterministic_findings(bb, stage)
            faults.append({"fault": name, "detected": bool(findings), "findings": findings})
    payload = {"source_sha256": code_tree_sha256(root), "source_hash_algorithm": "canonical-text-v2",
               "tests_run": result.testsRun,
               "tests_passed": result.wasSuccessful(), "fault_cases": faults,
               "fault_detection_rate": sum(f["detected"] for f in faults) / len(faults), "created_at": utc_now()}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"payload": payload, "sha256": canonical_hash(payload)}, indent=2) + "\n", encoding="utf-8")
    return result.wasSuccessful()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("validate")
    check.add_argument("--output", type=Path, required=True)
    assessment = commands.add_parser("assess")
    assessment.add_argument("run", type=Path)
    assessment.add_argument("--validation-evidence", type=Path)
    assessment.add_argument("--acceptance-evidence", type=Path)
    compare = commands.add_parser("compare-replay")
    compare.add_argument("original", type=Path)
    compare.add_argument("replay", type=Path)
    human = commands.add_parser("review-cases")
    human.add_argument("run", type=Path)
    human.add_argument("responses", type=Path)
    args = parser.parse_args()
    if args.command == "validate":
        return 0 if validate(args.output) else 2
    if args.command == "compare-replay":
        return 0 if compare_replay(args.original, args.replay)["matched"] else 2
    if args.command == "review-cases":
        print(resolve_human_review(args.run, args.responses).model_dump_json(indent=2))
        return 0
    report = assess(args.run, validation_path=args.validation_evidence, independent_path=args.acceptance_evidence)
    from contracts import Blackboard
    bb = Blackboard.open(args.run)
    write_acceptance(bb, validation_path=args.validation_evidence, independent_path=args.acceptance_evidence)
    print(args.run / "acceptance_report.json")
    return 0 if report.acceptance_ready else 2


if __name__ == "__main__":
    raise SystemExit(main())

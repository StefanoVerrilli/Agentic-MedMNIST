"""Sequential state machine with review gates, bounded retries and veto."""

from __future__ import annotations

from agents import Agent, ReviewerConsistencyAgent
from contracts import Blackboard, ExecutionState
from llm import OllamaDecisionError


class Orchestrator:
    """Run one agent at a time and invoke the reviewer after every stage."""

    def __init__(
        self,
        pipeline: list[Agent],
        reviewer: ReviewerConsistencyAgent,
        *,
        max_stage_retries: int = 1,
    ):
        if max_stage_retries < 0:
            raise ValueError("max_stage_retries cannot be negative")
        self.pipeline = pipeline
        self.reviewer = reviewer
        self.max_stage_retries = max_stage_retries

    def run(self, bb: Blackboard) -> Blackboard:
        def status(value: str, stage: str | None = None) -> None:
            bb.put("execution_status", ExecutionState(status=value, stage=stage), producer="orchestrator")

        status("running")
        resume = bb.get_optional("resume_manifest")
        if resume:
            names = [agent.name for agent in self.pipeline]
            if resume.parent_stage not in names:
                raise ValueError("resume stage does not belong to this pipeline")
            prefix = names[:names.index(resume.parent_stage)]
            if prefix != resume.promoted_stages:
                raise ValueError("resume requires a contiguous prefix of promoted stages")
        for agent in self.pipeline:
            if resume and agent.name in resume.promoted_stages:
                continue
            if resume:
                bb.record_event("run_resume_stage_started", parent_run=resume.parent_run_path, stage=agent.name)
            promoted = False
            for attempt in range(1, self.max_stage_retries + 2):
                bb.record_event("stage_started", stage=agent.name, attempt=attempt)
                try:
                    if not (resume and resume.review_only and agent.name == resume.parent_stage):
                        agent.run(bb)
                except (Exception, KeyboardInterrupt) as exc:
                    bb.record_event(
                        "stage_exception",
                        stage=agent.name,
                        attempt=attempt,
                        error_type=type(exc).__name__,
                        error=str(exc)[:500],
                    )
                    if (attempt <= self.max_stage_retries and exception_status(exc) == "failed"
                            and getattr(agent, "retry_on_exception", True)):
                        bb.record_event(
                            "stage_retry",
                            stage=agent.name,
                            next_attempt=attempt + 1,
                            reason="exception",
                        )
                        continue
                    failure = f"{exception_status(exc, bb)}:{agent.name}"
                    status(failure, agent.name)
                    bb.write_dossier(status=failure)
                    if resume:
                        bb.record_event("run_resume_failed", parent_run=resume.parent_run_path,
                                        stage=agent.name, status=failure, error=str(exc))
                    raise

                try:
                    report = self.reviewer.review(bb, agent.name, attempt=attempt)
                except (Exception, KeyboardInterrupt) as exc:
                    failure = f"{exception_status(exc, bb)}:review:{agent.name}"
                    bb.record_event("review_exception", stage=agent.name, error_type=type(exc).__name__, error=str(exc)[:500])
                    status(failure, agent.name)
                    bb.write_dossier(status=failure)
                    if resume:
                        bb.record_event("run_resume_failed", parent_run=resume.parent_run_path,
                                        stage=agent.name, status=failure, error=str(exc))
                    raise
                mark = {"ok": "OK", "warning": "REVISE", "critical": "STOP"}[
                    report.severity
                ]
                print(
                    f"[{agent.name:<17}] reviewer={mark:<6} "
                    f"source={report.source}  {report.comment}"
                )
                bb.record_event(
                    "review_gate",
                    stage=agent.name,
                    attempt=attempt,
                    severity=report.severity,
                    action=report.action,
                )
                if report.action == "stop":
                    failure = f"vetoed:{agent.name}"
                    status(failure, agent.name)
                    bb.record_event(
                        "stage_vetoed",
                        stage=agent.name,
                        issues=report.deterministic_issues + report.llm_issues,
                    )
                    bb.write_dossier(status=failure)
                    return bb
                if report.action == "revise" and attempt <= self.max_stage_retries:
                    remediation = getattr(agent, "revise", None)
                    changed = (
                        bool(remediation(bb, report))
                        if callable(remediation)
                        else False
                    )
                    if changed:
                        bb.record_event(
                            "stage_retry",
                            stage=agent.name,
                            next_attempt=attempt + 1,
                            reason="review_remediation_applied",
                        )
                        continue
                    bb.record_event(
                        "review_unresolved",
                        stage=agent.name,
                        attempt=attempt,
                        reason="agent_has_no_safe_remediation",
                        issues=report.deterministic_issues + report.llm_issues,
                    )

                promoted = True
                bb.record_event(
                    "stage_promoted",
                    stage=agent.name,
                    attempt=attempt,
                    with_warning=report.severity == "warning",
                )
                break

            if not promoted:
                failure = f"failed_to_promote:{agent.name}"
                status(failure, agent.name)
                bb.write_dossier(status=failure)
                return bb

        if resume:
            bb.record_event("run_resume_completed", parent_run=resume.parent_run_path, stage=resume.parent_stage)
        status("completed")
        bb.write_dossier(status="completed")
        return bb


def exception_status(exc, bb=None):
    # Only known transient seams are recoverable. Integrity and controller
    # errors fail closed; a resume additionally verifies every persisted file.
    if not isinstance(exc, (OllamaDecisionError, KeyboardInterrupt, TimeoutError)):
        return "failed"
    if bb is not None:
        from resume import verify_store
        try:
            verify_store(bb)
        except (ValueError, OSError, KeyError, TypeError):
            return "failed"
    return "paused"

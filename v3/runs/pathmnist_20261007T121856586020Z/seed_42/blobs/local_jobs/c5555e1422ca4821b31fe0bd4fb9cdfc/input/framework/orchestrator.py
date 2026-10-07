"""Sequential state machine with review gates, bounded retries and veto."""

from __future__ import annotations

from agents import Agent, ReviewerConsistencyAgent
from contracts import Blackboard, ExecutionState


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
        for agent in self.pipeline:
            promoted = False
            for attempt in range(1, self.max_stage_retries + 2):
                bb.record_event("stage_started", stage=agent.name, attempt=attempt)
                try:
                    agent.run(bb)
                except Exception as exc:
                    bb.record_event(
                        "stage_exception",
                        stage=agent.name,
                        attempt=attempt,
                        error_type=type(exc).__name__,
                        error=str(exc)[:500],
                    )
                    if attempt <= self.max_stage_retries and getattr(agent, "retry_on_exception", True):
                        bb.record_event(
                            "stage_retry",
                            stage=agent.name,
                            next_attempt=attempt + 1,
                            reason="exception",
                        )
                        continue
                    failure = f"failed:{agent.name}"
                    status(failure, agent.name)
                    bb.write_dossier(status=failure)
                    raise

                try:
                    report = self.reviewer.review(bb, agent.name, attempt=attempt)
                except Exception as exc:
                    failure = f"failed:review:{agent.name}"
                    bb.record_event("review_exception", stage=agent.name, error_type=type(exc).__name__, error=str(exc)[:500])
                    status(failure, agent.name)
                    bb.write_dossier(status=failure)
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

        status("completed")
        bb.write_dossier(status="completed")
        return bb

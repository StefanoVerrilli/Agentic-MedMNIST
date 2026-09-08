"""Sequential state machine with review gates, bounded retries and veto."""
from __future__ import annotations

from agents import Agent, ReviewerConsistencyAgent
from contracts import Blackboard


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
        self.status = "not_started"

    def run(self, bb: Blackboard) -> Blackboard:
        self.status = "running"
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
                    if attempt <= self.max_stage_retries:
                        bb.record_event(
                            "stage_retry",
                            stage=agent.name,
                            next_attempt=attempt + 1,
                            reason="exception",
                        )
                        continue
                    self.status = f"failed:{agent.name}"
                    bb.write_dossier(status=self.status)
                    raise

                report = self.reviewer.review(bb, agent.name, attempt=attempt)
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
                    self.status = f"vetoed:{agent.name}"
                    bb.record_event(
                        "stage_vetoed",
                        stage=agent.name,
                        issues=report.deterministic_issues + report.llm_issues,
                    )
                    bb.write_dossier(status=self.status)
                    return bb
                if report.action == "revise" and attempt <= self.max_stage_retries:
                    bb.record_event(
                        "stage_retry",
                        stage=agent.name,
                        next_attempt=attempt + 1,
                        reason="review_gate",
                    )
                    continue

                promoted = True
                bb.record_event(
                    "stage_promoted",
                    stage=agent.name,
                    attempt=attempt,
                    with_warning=report.severity == "warning",
                )
                break

            if not promoted:
                self.status = f"failed_to_promote:{agent.name}"
                bb.write_dossier(status=self.status)
                return bb

        self.status = "completed"
        bb.write_dossier(status=self.status)
        return bb

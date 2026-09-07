"""Orchestrator: sequencing + state only, zero ML/domain logic.

This is what proves 'minimum coupling': agents are an ordered list that share
nothing but the Blackboard, and the WP6 reviewer runs as a cross-cutting hook
after each stage and can veto promotion on a critical anomaly.
"""
from __future__ import annotations

from agents import Agent, ReviewerConsistencyAgent
from contracts import Blackboard


class Orchestrator:
    def __init__(self, pipeline: list[Agent], reviewer: ReviewerConsistencyAgent):
        self.pipeline = pipeline
        self.reviewer = reviewer

    def run(self, bb: Blackboard) -> Blackboard:
        for agent in self.pipeline:
            agent.run(bb)
            rep = self.reviewer.review(bb, agent.name)
            mark = {"ok": "ok", "warning": "!", "critical": "STOP"}[rep.severity]
            print(f"[{agent.name:<17}] reviewer={mark}  {rep.comment}")
            if rep.severity == "critical":
                print(f"  -> vetoed at '{agent.name}': {rep.issues}")
                break
        return bb

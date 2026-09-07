"""Entry point: build the pipeline, run it, compare against the baseline.

    python run.py                 # deterministic (heuristic reasoning)
    AGENTIC_LLM_BASE=... python run.py   # hybrid: real open-model reasoning
"""
from __future__ import annotations

from agents import (AbstentionOODAgent, EvaluationAgent, ExperimentDesignAgent,
                    IngestionAgent, PreprocessingAgent, ProfilingAmbiguityAgent,
                    ReportingAgent, ReviewerConsistencyAgent, TrainingAgent)
from baseline import run_baseline
from contracts import Blackboard
from orchestrator import Orchestrator

FLAG = "pneumoniamnist"   # binary: naturally exercises the 'no finding' case


def main() -> None:
    bb = Blackboard("./runs/latest")
    pipeline = [
        IngestionAgent(FLAG), ProfilingAmbiguityAgent(), PreprocessingAgent(),
        ExperimentDesignAgent(), TrainingAgent(), EvaluationAgent(),
        AbstentionOODAgent(), ReportingAgent(),
    ]
    print("=== Agentic pipeline ===")
    Orchestrator(pipeline, ReviewerConsistencyAgent()).run(bb)

    if "eval" in bb.artefacts:
        agentic = bb.get("eval").accuracy
        ab = bb.get("abstention")
        cfg = bb.get("train_config")
        print(f"\nconfig source : {cfg.source}  ({cfg.rationale})")
        print(f"agentic acc   : {agentic}")
        print(f"abstention    : coverage={ab.coverage} acc_on_covered={ab.accuracy_on_covered}")
        print("\n=== Conventional baseline ===")
        print(f"baseline acc  : {run_baseline(FLAG)['accuracy']}")
        print("\nDossier + artefacts written to ./runs/latest/")


if __name__ == "__main__":
    main()

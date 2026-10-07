"""Retrain an exported generated experiment without a model-reasoning request."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from agents import (DataAuditAgent, IngestionAgent, PreprocessingAgent,
                    ProfilingAmbiguityAgent, TrainingAgent)
from contracts import Blackboard, RepresentationDecision, TrainConfig, WorkerEvidence
from llm import OllamaReasoner
from remote import SSHWorker, copy_bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["fit"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True, help="directory owning the exported generated bundle")
    parser.add_argument("--output-root", type=Path, default=Path("runs/retrained"))
    parser.add_argument("--data-root")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--worker-host")
    parser.add_argument("--worker-root")
    args = parser.parse_args(argv)
    document = json.loads(args.config.read_text(encoding="utf-8"))
    if document.get("format") != "isolated-experiment-v1":
        parser.error("config must use isolated-experiment-v1")
    config = TrainConfig.model_validate(document["train_config"])
    if config.model_family != "run_generated":
        parser.error("use lightning_cli.py for built-in models")
    payload = config.model_dump()
    if args.device:
        payload["device"] = args.device
    if args.seed is not None:
        payload["seed"] = args.seed
    worker_payload = config.worker.model_dump()
    if args.worker_host:
        worker_payload["host"] = args.worker_host
    if args.worker_root:
        worker_payload["root"] = args.worker_root
    payload["worker"] = worker_payload
    config = TrainConfig.model_validate(payload)
    runner = SSHWorker(config.worker)
    evidence = runner.preflight(config.device)
    config = config.model_copy(update={"worker": runner.configuration})
    root = args.output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    bb = Blackboard(root)
    copy_bundle(args.run_root.resolve(), bb.root, config.generated_bundle)
    bb.put("worker_evidence", WorkerEvidence(configuration=config.worker,
        image_digest=evidence["image_digest"], isolation=evidence["isolation"]), producer="generated_cli")
    data = document.get("data", {})
    quick = {"train": 6000, "val": 1200, "test": 1800}
    limits = {f"{split}_limit": data.get(f"{split}_limit")
              if data.get(f"{split}_limit") is not None else quick[split] if data.get("quick") else None
              for split in quick}
    reasoner = OllamaReasoner(base_url=None)
    IngestionAgent(seed=config.seed, data_root=args.data_root or data.get("data_root"),
                   download=not data.get("no_download", False), **limits).run(bb)
    ProfilingAmbiguityAgent(reasoner).run(bb)
    PreprocessingAgent(reasoner, forced_decision=RepresentationDecision.model_validate(document["representation"])).run(bb)
    DataAuditAgent().run(bb)
    bb.put("train_config", config, producer="generated_cli")
    TrainingAgent().run(bb)
    bb.write_dossier(status="completed:fit")
    print(bb.root / bb.get("train_result").checkpoint_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

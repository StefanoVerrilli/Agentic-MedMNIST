# Presentation implementation in v2

The presentation is implemented as a research demonstrator on PathMNIST.
Work-package scheduling and human TRL/sign-off statements are project evidence,
not outcomes the training code can self-certify.

| Presentation claim | Implementation and evidence |
| --- | --- |
| Role-specialised agents; no direct agent calls | `agents.py`, orchestrator protocol and AST coupling check in `governance.py` |
| Only versioned typed hand-offs | Pydantic contracts and `Blackboard`; heavy-object codecs in `blobio.py`; immutable JSON envelopes and NPZ/checkpoint references |
| Stateless sequencing | Orchestrator keeps configuration only; execution status is an `ExecutionState` on the Blackboard |
| Declared hybrid decisions, open-model fallback | Provider-neutral `Reasoner`, Ollama schema validation, serialized calls, temperature zero, logged fallback |
| Ambiguity handling | Train/validation profile and reasoned ambiguity risks; no automatic relabelling of ambiguous tissue |
| Prior-Art Scout before steps 3 and 4 | `PriorArtScoutAgent`: arXiv API search/retrieval, cached abstract snapshots, cited hypotheses, extension proposals |
| Evidence influences representation and model design | Cited ideas enter preprocessing, architecture research and design prompts; the research brief is available in every model-search round |
| Offline and reproducible research | Checksummed source snapshots; cached sources are reused without retrieval. Bundled short excerpts are explicitly marked curated |
| All ideas cited; no invented citations | Reviewer resolves each source identifier against a checksummed snapshot and verifies exact supporting quotes. Semantic support remains an independent review input; syntactic resolution alone does not prove entailment |
| New representations through a gate | Hash-bound review plus deterministic geometry checks; promote compositions of whitelisted orientation primitives as `approved_<proposal_id>` |
| New architecture family as reviewed spec | `ExtensionProposal` and `ExtensionGateReport`; no generated Python runs. Developer adds the contract family and `build_network` branch, then tests before enabling it |
| Training begins only after an audit | `DataAuditAgent` checks actual train/validation images, labels, pixel range, selected-index fingerprints, overlap, split and train-only statistics; search and training require a passed report |
| Official split and sealed test selection | Existing official MedMNIST splits retained; model-search prompts and ranking contain only validation evidence. Archive MD5 is verified and SHA-256 recorded |
| Train and evaluate the winner | Progressive validation search, frozen best configuration, deterministic Lightning training, final clean-test predictions persisted once per successful evaluation stage |
| Abstention and human review | Validation-calibrated risk/coverage; checksummed queue with actual sample indices, confidence and `no_confident_finding`; append-only confirm/correct/defer responses |
| OOD | Existing controlled Gaussian-corruption proxy. This measures robustness to those corruptions, not general clinical OOD detection |
| Baseline, ablations, repeated seeds | Same-data/seed baseline, three representation ablations, frozen configuration by default, complete-experiment aggregate and frozen-config consistency checks |
| Every decision traceable | JSONL decisions plus checksummed transcripts containing exact prompts, schemas, seed, provider identity, fallback and validated output |
| Identical seeds/configs replay the run | `--replay-run` restores the recorded configuration, source snapshots, raw data and decisions; no LLM or retrieval request. Retraining is compared against original exact prediction arrays and metrics |
| Independent reviewer, retry and veto | Reviewer after each stage. Deterministic critical violations veto; ungrounded LLM claims stay advisory. Retry requires an applicable remediation. Warnings stay visible and require mitigation for final acceptance |
| At least 90% injected faults detected | `governance.py validate` executes the suite and measures ten explicit faults, binding evidence to the exact source tree; additional tests cover citation tampering, cache tampering, overlap and extension gates |
| WP1–WP6 deliverables linked to requirements | `AcceptanceReport` maps each requirement to stored evidence, measured KPIs and passed/failed/pending status |
| TRL 7 and reviewer sign-off | Independent evidence manifest bound to run ID, code and predictions, with distinct implementer/demonstrator/reviewer, semantic citation review, checksummed demonstration attachment and approved mitigations |

## Operating procedure

Run all commands from `v2`. First generate validation evidence after finishing
source changes:

```powershell
python governance.py validate --output validation_evidence.json
```

An online representative experiment with a local open model:

```powershell
python run.py --research-online --require-research --seeds 42,47,72 --ablation-suite --validation-evidence validation_evidence.json --ollama-base http://localhost:11434 --ollama-model qwen2.5:7b
```

Use a fresh `--literature-cache` directory to collect a new corpus. A populated
cache is immutable and replayed even when `--research-online` is passed.
`--require-research` rejects any curated fallback. Without online access the
system runs from cached sources or bundled excerpts and accurately reports the
missing web-retrieval evidence. `--offline` disables online retrieval and Ollama.

Replay one completed seed with its recorded source tree and package environment:

```powershell
python run.py --replay-run runs/pathmnist_TIMESTAMP/seed_42
```

Replay consumes transcripts in order and rejects changed semantic context,
missing decisions and invalid hashes. Its fingerprint ignores only declared
execution metadata: timestamps, run-root paths, independent file checksums and
elapsed times. The exact original request is still retained; file integrity is
checked separately. Numerical prediction equality is checked after retraining,
so a new machine/backend cannot be assumed reproducible without the comparison.

New proposals are saved in `extension_proposals.json`. Review a proposal and
copy its exact spec hash into a manifest based on
`configs/extension_approvals.example.json`, then pass
`--extension-approvals your_approvals.json` to a new run. Architectural proposals
stay non-executable specs. Representation promotion is limited to reviewed
orientation compositions; broader transformations require developer-owned
implementations and additional scientific validation.

Review abstained benchmark cases using
`configs/human_review_responses.example.json` (`confirm`, `correct` with a label
0–8, or `defer`):

```powershell
python governance.py review-cases runs/pathmnist_TIMESTAMP/seed_42 responses.json
```

The original queue and test metrics stay intact; human review is a separate,
versioned result. No clinical workflow is implied.

After the independent demonstration, fill
`configs/acceptance_evidence.example.json` with real evidence and reassess:

```powershell
python governance.py assess runs/pathmnist_TIMESTAMP/seed_42 --validation-evidence validation_evidence.json --acceptance-evidence independent_evidence.json
```

Final acceptance requires all criteria to pass, including live retrieval,
ablations, at least three frozen-configuration seeds, a successful replay,
validated faults, resolved/approved risks and independent sign-off. Missing
evidence stays pending or failed; a successful training run does not imply TRL 7.

Heavy hand-offs now persist a dataset snapshot per seed and reconstruct fresh
objects on read. This increases disk usage and I/O while removing implicit
mutable sharing. DataLoader uses zero worker processes so the existing nested
dataset implementation remains portable to Windows spawn and deterministic.
Historical `runs/` directories are preserved; replay requires the new
configuration, raw-data and reasoning artefacts.

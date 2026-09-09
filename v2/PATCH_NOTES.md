# Patch notes — baseline path fix and Lightning training logs

## Fixed crash

The baseline used `actual_checkpoint.relative_to(root_path)`. Lightning may return an
absolute `best_model_path` while the run root is relative (for example
`runs/pathmnist_.../seed_42`). `Path.relative_to()` rejects that mixed form even when
both paths refer to the same directory tree.

`baseline.py` now resolves both operands before computing the run-relative path, with
a safe fallback for genuinely external paths. The same helper is used by the ablation
suite.

## Lightning logs

Every call through the shared `train_model()` backend now:

- enables the Lightning progress bar by default;
- prints one stable epoch line such as
  `[lightning:agentic_model_v001] epoch=001/015 train_loss=... val_loss=... val_acc=... val_macro_f1=... lr=...`;
- writes a compact sibling `*.training.jsonl` file;
- enables Lightning's native `CSVLogger`, producing `metrics.csv` under a
  `lightning_logs/<training-name>/version_0/` directory;
- persists both log paths in training, search-trial, baseline and ablation artefacts.

## Corrected epoch metrics

`MetricsHistory` now captures metrics at `on_train_epoch_end`, after the validation
loop. This removes the old first-epoch `train_loss=0.0` / one-epoch-shift artefact.
Validation macro-F1 is also computed and logged every epoch from the full validation
confusion matrix.

## Validation performed

- `python -m compileall -q .`
- `python -m unittest discover -s tests -v` → 21/21 passed
- real Lightning CPU smoke training confirmed non-zero epoch-1 train loss, per-epoch
  macro-F1, JSONL logging and native `metrics.csv` output;
- real baseline smoke run using a relative run root confirmed the former
  absolute-vs-relative checkpoint crash no longer occurs.

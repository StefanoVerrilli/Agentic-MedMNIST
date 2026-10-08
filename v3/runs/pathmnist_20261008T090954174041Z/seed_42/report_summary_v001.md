## Benchmark Report: PathMNIST 9-Class Tissue Classification

### 1. Effective Generated-Bundle Parameters vs. Inactive Generic Defaults

The selected model (autonomous_t0007, bundle `pathmnist_resnet18_tuned_v2`) is a ResNet18 architecture. The **effective generated-bundle parameters** that directly shape the network are:

| Parameter | Value | Role |
|-----------|-------|------|
| depth | 2 | Residual block depth multiplier |
| dropout | 0.3 | Regularization within residual blocks |
| hidden | 64 | Base channel width |

These three parameters are passed to `build_network("resnet18", in_channels=3, n_classes=9, hidden=64, depth=2, dropout=0.3)`. The bundle manifest SHA-256 is `36e5a4307d7dd4f200ba64d665a6379fc8efea024e54ab2df926e742feccd738`.

The following fields in `train_config` are **inactive generic defaults** that do not alter the ResNet18 architecture (they are transformer/MLP-specific or framework-level placeholders): `num_heads=4`, `mlp_ratio=4`, `patch_size=4`, `positional_encoding="learned"`, `pooling="cls"`, `scales=[2,4,7]`, `channel_cap=256`, `tokenizer_layers=2`. The search report explicitly annotates these as "framework defaults; inspect archived build_model for actual parameter use." They are carried in the config schema for cross-architecture compatibility but have no effect on the ResNet18 forward pass.

**Active training hyperparameters** (not architecture parameters but governing optimization): lr=2e-4, optimizer=AdamW (β1=0.9, β2=0.999, ε=1e-8), weight_decay=0.01, label_smoothing=0.05, batch_size=256, cosine scheduler (η_min=0, pct_start=0.1), gradient_clip=1.0, early stopping on val_accuracy (patience=12, min_delta=0.001), max 50 epochs (converged at epoch 17), seed=42, class_weighting=true.

### 2. Checkpoint Validation vs. Test Metrics

| Metric | Validation (checkpoint) | Test |
|--------|------------------------|------|
| Accuracy | 0.995902 | 0.917827 [0.911249, 0.923958] |
| Macro F1 | 0.995778 | 0.890046 |
| Balanced Accuracy | — | 0.893666 |
| ROC-AUC (OVR macro) | — | 0.980611 |

The gap between validation accuracy (99.59%) and test accuracy (91.78%) is approximately 7.8 percentage points. This is the single most important observation in the report. The model was selected on validation accuracy with a tolerance rule, and the test set (7,180 samples from a separate center per the split manifest) reveals substantially lower performance. The test set is explicitly noted as `official_test_separate_center: true`, indicating a domain shift between the training/validation center and the test center. No causal mechanism for this gap is asserted here; the evidence documents the magnitude and the separate-center provenance.

The curve_max_accuracy for t0007 was 0.996102 (epoch 17), while the checkpoint was saved at epoch 17 with accuracy 0.995902, confirming the checkpoint is near the validation optimum.

### 3. Per-Class Weaknesses (Test)

Ranked by F1 (ascending):

| Class | F1 | Precision | Recall | Support | Key Error Pattern |
|-------|----|-----------|--------|---------|-------------------|
| Cancer-associated stroma (7) | 0.667 | 0.928 | 0.520 | 421 | 131/421 misclassified; 70→smooth muscle, 55→debris, 59→adenocarcinoma |
| Smooth muscle (5) | 0.768 | 0.722 | 0.821 | 592 | 106/592 misclassified; 61→background, 30→debris, 11→adenocarcinoma |
| Debris (2) | 0.848 | 0.781 | 0.926 | 339 | 25/339 misclassified; 16→smooth muscle, 9→background |
| Background (1) | 0.945 | 0.896 | 1.000 | 847 | 95/105 predicted as background are false positives (from classes 0,4,5,7,8) |
| Adipose (0) | 0.945 | 0.970 | 0.922 | 1,338 | 105/1338 missed; 99→smooth muscle, 3→mucus |
| Mucus (4) | 0.957 | 0.994 | 0.923 | 1,035 | 80/1035 missed; 38→adipose, 22→background |
| Normal colon mucosa (6) | 0.955 | 0.936 | 0.974 | 741 | 19/741 missed; 13→adenocarcinoma, 5→lymphocytes |
| Lymphocytes (3) | 0.978 | 0.958 | 1.000 | 634 | 27/634 predicted as lymphocytes are false positives (21 from class 8) |
| Colorectal adenocarcinoma (8) | 0.947 | 0.937 | 0.957 | 1,233 | 53/1233 missed; 30→normal colon mucosa, 21→lymphocytes |

The two weakest classes (cancer-associated stroma, smooth muscle) are histologically similar spindle-cell / fibrous tissues, and the confusion matrix shows bidirectional confusion between them and with debris. Background has perfect recall but the lowest precision (0.896), meaning it acts as a "sink" for uncertain predictions from other classes.

### 4. Validation-Calibrated vs. Achieved Test Coverage (Selective Prediction)

The abstention system uses predictive entropy with a threshold calibrated on validation to target 80% coverage:

| Target Val Coverage | Val Coverage | Val Risk | Test Coverage | Test Risk |
|--------------------:|-------------:|---------:|--------------:|----------:|
| 0.50 | 0.5001 | 0.0008 | 0.4585 | 0.0143 |
| 0.60 | 0.6001 | 0.0007 | 0.5326 | 0.0139 |
| 0.70 | 0.7001 | 0.0006 | 0.6018 | 0.0148 |
| **0.80** | **0.8001** | **0.0009** | **0.6928** | **0.0241** |
| 0.90 | 0.9000 | 0.0008 | 0.7623 | 0.0265 |

At the 80% target: validation achieves 80.0% coverage with 0.09% risk, but test achieves only **69.3% coverage** with **2.41% risk**. The coverage gap is ~10.7 percentage points. The abstention rate on test is 30.7%, and accuracy on the covered subset is 97.59% (vs. 91.78% base). The threshold (0.8483) was set on validation; the test distribution's entropy profile is shifted, causing the same threshold to reject more samples. This is a calibration transfer issue, not a threshold error per se—the threshold was correctly applied, but the validation-to-test entropy distribution mismatch reduces effective coverage.

### 5. Controlled Corruption OOD Scope

The OOD evaluation is explicitly scoped as a **controlled corruption proxy** (`ood_scope: "controlled_corruption_proxy"`). The corruption is `unit_scale_gaussian_noise` at three severity levels (σ=0.15, 0.25, 0.40), applied to pixel values. Two scoring functions were tested: max_softmax and predictive_entropy.

| σ | Score | Val AUROC | Test AUROC | Test FPR |
|---|-------|-----------|------------|----------|
| 0.15 | max_softmax | 0.9909 | 0.9267 | 0.0015 |
| 0.15 | predictive_entropy | 0.9913 | 0.9341 | 0.0019 |
| 0.25 | max_softmax | 0.9937 | 0.9355 | 0.0000 |
| 0.25 | predictive_entropy | 0.9940 | 0.9375 | 0.0000 |
| 0.40 | max_softmax | 0.9962 | 0.9598 | 0.0000 |
| 0.40 | predictive_entropy | 0.9961 | 0.9585 | 0.0000 |

Overall OOD AUROC: 0.9434 (test), 0.9938 (validation). All scenarios pass the minimum AUROC threshold of 0.65. The false accept rate is below the 0.20 maximum in all cases (effectively zero at σ≥0.25).

**Scope limitation**: This is additive Gaussian noise on pixel values only. It does not represent staining variability, scanner differences, tissue preparation artifacts, resolution changes, or any other realistic distribution shift. The separate-center test set (Section 2) is a more ecologically valid OOD test, and the 7.8-point accuracy drop there is the more informative signal. The corruption proxy confirms the detector can flag pixel-level noise but says nothing about the types of shift that actually occur across clinical sites.

### 6. Data Integrity

The data audit passed all checks: official split confirmed, train/validation disjoint (overlap=0), shapes and labels verified, train-only statistics used. Test was not inspected during training (test_inspected=false). The split manifest confirms 89,996 train / 10,004 val / 7,180 test with seed 42 and SHA-256-verified index files.

### 7. Search Process

8 trials completed, 0 failed. All from the `run_generated` family. The selection rule was agent-selected durations with validation accuracy tolerance then macro-F1 tiebreak. Test metrics were not used in selection (test_metrics_used=false). The best transformer (t0004, depth=6) peaked at 0.992903 validation accuracy, below the ResNet18 variants (0.9954–0.9959). The stopping rationale confirms no further trials were permitted.

### 8. Limitations

- **Separate-center test set**: The 7.8-point validation-to-test gap is consistent with domain shift between imaging centers. No causal attribution is made; the evidence documents the gap and the provenance.
- **OOD scope is narrow**: Gaussian pixel noise is a stress test, not a clinical distribution shift. The separate-center test is the more relevant OOD evaluation.
- **Selective prediction coverage gap**: The 10.7-point coverage shortfall on test means the abstention system, as calibrated, will route ~31% of test samples to human review (2,206 samples), which is operationally significant.
- **Class imbalance**: Mild (1.63:1 max/min), but cancer-associated stroma (421 samples) and debris (339) are underrepresented and are the two weakest classes.
- **No clinical safety claims**: This is a benchmark on a public dataset. No diagnostic, therapeutic, or clinical-safety conclusions are drawn or implied.
- **Single seed**: All results use seed 42. Variance across seeds is not characterized.
- **No unverified causes asserted**: The report documents observed metrics and their provenance without attributing the validation-test gap to specific architectural, data, or training deficiencies beyond what the evidence directly supports.

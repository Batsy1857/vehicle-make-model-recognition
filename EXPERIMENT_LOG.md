# Experiment Log — Vehicle Make-and-Model Recognition

**Status:** No experiments have been run or recorded yet.  
**Protocol:** `configs/experiments.yaml` (draft v0.1; review before training)  
**Important:** Empty fields below indicate **not measured**, not zero.

## A. Planned run registry

| ID | Dataset | Model | Status | Best val macro-F1 | Test top-1 | Test macro-F1 | Training time | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | Pakistani | Xception | Not started | — | — | — | — | — |
| E2 | Pakistani | ConvNeXt-Tiny | Not started | — | — | — | — | — |
| E3 | Stanford Cars | Xception | Not started | — | — | — | — | — |
| E4 | Stanford Cars | ConvNeXt-Tiny | Not started | — | — | — | — | — |

**Do not fill this table from pilot runs unless clearly labelled as pilot results.** Final test values are recorded only after model selection on validation data.

## B. Dataset audit and split register

| Dataset | Image/class counts verified? | Label-map saved? | Split manifests and SHA256 | Leakage/near-duplicate audit | Status |
| --- | --- | --- | --- | --- | --- |
| Pakistani | 3,847 readable / 48 classes | Pending | Publisher train/test present; manifests pending | Exact duplicates 0; near-duplicates pending | Initial audit complete; validation pending |
| Stanford Cars | Pending | Pending | Pending | Pending | Awaiting data |

Record exactly how Pakistani video-neighbor images were handled if source/group metadata is available. For Stanford Cars, document that the official test split remained untouched during training and model selection.

### Pakistani data audit — 27 October 2026

Results from the user-executed `01_data_exploration.ipynb` (Mendeley 2022 Version 1):
- **3,847** images scanned and decoded; **0** unreadable.
- **48** distinct labels inferred from class directory names.
- Existing publisher train/test split: **3,096 train**, **751 test**; these remain untouched.
- **0** exact SHA-256 duplicate groups; **0** across-split exact duplicate groups.
- Class imbalance: **337** Toyota_corolla_2016, **336** Suzuki_cultus_2018, **16** Suzuki_carry (total across train/test).
- Sample images vary in viewpoint, lighting and background; plates are obscured in illustrated cases.
- Follow-up class coverage (user measured): **48** classes in each publisher split; no train-only or test-only classes. Original publisher training min/class **13**, max/class **270**; the 10 smallest range from **13 to 23** images.
- Pending: human review of labels, perceptual near-duplicate/video-neighbor checks, validation manifest, class index map.
- The first visualization used test images due to filename sorting; notebook corrected in this PR to sample training images only. No training was performed.

## C. Run record template

Copy this section once for **each execution**, including failures and restarts. Use distinct IDs such as `E1-run01`, `E1-run02`. Never overwrite past outcomes.

### Run ID: TEMPLATE-ONLY (not an actual run)

- **Started / ended (timezone):** TBD
- **Operator:** TBD
- **Purpose:** pilot / final / ablation / retry
- **Status:** not started / completed / failed / interrupted
- **Dataset / model:** TBD
- **Git commit SHA:** TBD
- **Experiment config commit SHA and deviations:** TBD
- **Data version / split manifest SHA256 / label map:** TBD
- **Seed / fold:** TBD
- **Image size / augmentation / normalization:** TBD
- **Pretrained weights identifier:** TBD
- **Trainable layers / total parameters / trainable parameters:** TBD
- **Epochs completed / best epoch / stop reason:** TBD
- **Optimizer / LR / weight decay / batch size / precision:** TBD
- **Hardware:** GPU type, VRAM, runtime and operating system
- **Versions:** Python, PyTorch, torchvision, timm, CUDA (if applicable)
- **Checkpoint path (outside Git):** TBD
- **Learning curves and artifacts:** TBD
- **Best validation:** macro-F1 TBD; top-1 TBD; loss TBD
- **Held-out test (evaluate after selection):** top-1 TBD; top-5 TBD; macro-F1 TBD
- **Efficiency:** training seconds/epoch TBD; total training seconds TBD; batch-1 and batch-16 inference latency TBD; peak GPU memory TBD
- **Confusion analysis / representative errors:** TBD
- **Warnings, errors, OOM and changes:** TBD
- **Decision:** keep / rerun / investigate — with reason

## D. Measurement rules

1. **Validation** chooses the checkpoint; test does not choose hyperparameters.
2. Record top-1 accuracy, top-5 accuracy and macro-F1 as **fractions or percentages with clearly stated units**. Do not mix units silently.
3. Report parameter count **after** replacing the classifier, alongside trainable parameters.
4. Record training timing separately from evaluation, dataset download and notebook startup; note any interruptions.
5. Benchmark inference on the **same GPU**, using model `eval()`, inference mode, warmup, synchronization and repeat measurements. Report batch size, median and P95; exclude image preprocessing unless explicitly documented otherwise.
6. If an experiment uses different resolution, batch size, hardware, frozen layers or epochs, log it as a deviation or separate ablation; do not hide the exception.
7. Preserve failed experiments and out-of-memory events as useful engineering observations.
8. No invented results. Published numbers from papers must be attributed and **must not be recorded as our measured results**.

## E. Change log

| Date | Change | Reason | Approved by |
| --- | --- | --- | --- |
| 2026-10-27 | Created planning and run templates; no experimental measurements | Establish reproducible team workflow | Pending team review |

## F. Report/export checklist

- [ ] Four primary runs completed (or limitations explained)
- [ ] Checkpoints and result files linked
- [ ] Exactly matching split IDs and preprocessing documented
- [ ] Architecture-specific model configurations and normalization documented
- [ ] Timing hardware and benchmark protocol documented
- [ ] Results table matches notebook outputs, not handwritten estimates
- [ ] Plots and confusion/error analysis reproduced from saved outputs
- [ ] All five presenters know which numbers are actual measurements

# Project Milestone Summary — Vehicle Make-and-Model Recognition

**Updated:** 28 October 2026 (Asia/Karachi)  
**Assignment deadline:** 2 November 2026  
**Team:** Five members  
**Project HQ:** Chat `00 — Project HQ` in ChatGPT's Deep Learning — Vehicle Recognition Project  
**Experiment status:** **No training runs yet; no model performance results available.**

## 1. Objective and scope

Compare **Xception (2017)** with **ConvNeXt-Tiny (2022)** via ImageNet-pretrained transfer learning on:

1. Pakistani vehicle make/model data (2022 Mendeley Version 1): **3,847 images, 48 classes**.
2. Stanford Cars: **16,185 reported images, 196 classes** (download/audit not yet completed).

Research question: Which model better balances classification quality, total/trainable parameters, training cost, and inference speed **within each dataset**?

The professor's assignment requires discussion of CNN architecture selection, a **four-page referenced LaTeX report** (architecture diagram/mathematical functions, published sources, components, advantages/disadvantages, parameters, deployment applications), and a **15-minute group presentation** with equal contribution. The deadline is November 2, 2026.

**Architecture selection is proposed** and should be confirmed with the instructor as required by the assignment brief.

## 2. Completed milestone — Project foundation

- Private GitHub repository: https://github.com/Batsy1857/vehicle-make-model-recognition
- Files created: `PROJECT_PLAN.md`, `configs/experiments.yaml`, `EXPERIMENT_LOG.md`.
- Planned framework: PyTorch + timm; development in VS Code; GPU training in Google Colab; GitHub for collaboration; Overleaf/LaTeX for the final report.
- Four **planned** experiments:
  - **E1** — Xception on Pakistani data
  - **E2** — ConvNeXt-Tiny on Pakistani data
  - **E3** — Xception on Stanford Cars
  - **E4** — ConvNeXt-Tiny on Stanford Cars
- Same split manifests, class mapping, image geometry and initial training policy within each dataset; record model-specific ImageNet normalization and actual device/runtime versions.
- Seed: **42**. Initial **224 x 224** input policy, optimizer/epoch schedule and other hyperparameters in YAML are **proposals pending pilot testing**.
- Planned metrics: top-1/top-5 accuracy, macro-F1, total/trainable parameters, training time, inference latency, model size and optional computational estimates.

## 3. Completed milestone — Pakistani dataset audit

Dataset: 2022 Version 1, DOI **10.17632/hj3vvx5946.1**  
Source: https://data.mendeley.com/datasets/hj3vvx5946/1

Read-only audit run in `notebooks/01_data_exploration.ipynb`:

| Check | Observed |
| --- | ---: |
| Images discovered and decoded | 3,847 |
| Vehicle classes inferred from folder names | 48 |
| Unreadable images | 0 |
| Byte-identical SHA-256 duplicate groups | 0 |
| Original publisher training images | 3,096 |
| Original publisher test images | 751 |
| Classes in each original publisher split | 48 |
| Original training images per class, minimum/maximum | 13 / 270 |
| Total examples for smallest sampled class `Suzuki_carry` | 16 |
| Total examples for `Toyota_corolla_2016` | 337 |

Findings: class imbalance, varying viewpoints/lighting/background, and a need to preserve publisher train/test membership. **No claim that every class-folder label was independently ground-truth checked**.

## 4. Completed milestone — Split creation and overlap audit

`02_prepare_splits.ipynb` created an initial **candidate** validation split (15% of publisher training, stratified, seed 42):

- Candidate train: **2,631**
- Candidate validation: **465**
- Original test: **751**

At dHash Hamming distance **≤3**, candidate visually similar **pairs** across split boundaries were:

- **23** publisher train↔test candidate pairs.
- **12** train↔validation candidate pairs.
- **35** candidate pairs total (not necessarily 35 distinct images).

The group reviewer reported that the flagged pairs looked identical or near-identical images of the same physical vehicle. Review verdicts are in `splits/pakistan/candidate_pair_review.csv`; the finding is human-reviewed, not proof of pixel-byte identity. The dHash method can miss other adjacent video frames.

`04_reconcile_overlap_splits.ipynb` used confirmed links to exclude original training images connected to test images, and to keep confirmed related remaining images together in the training/validation split. **No original images were deleted or moved**, and the publisher test allocation remained fixed.

## 5. Completed milestone — Frozen overlap-screened Pakistani split

`05_freeze_pakistan_split.ipynb` generated final CSV and JSON artifacts, now committed to GitHub and independently visible on `main`.

| Assignment | Image count | Minimum examples/class |
| --- | ---: | ---: |
| Train | **2,610** | **11** |
| Validation | **463** | **2** |
| Test (publisher allocation unchanged) | **751** | **3** |
| Excluded publisher-training overlap | **23** | — |
| **Total recorded images** | **3,847** | — |

- Every **active** split has all **48 classes**.
- Only **3,824** images are eligible for training, validation, or held-out evaluation; **23** original training observations remain explicitly marked as excluded.
- Final manifest path: `splits/pakistan/final_manifest.csv`.
- Stable 48-class mapping: `splits/pakistan/class_to_idx.json`.
- Split metadata: `splits/pakistan/split_metadata.json`.
- **Final manifest SHA-256:** `11ebb90a7a4530a2707f4d086131f93f4b3c91cae013bc6add0bc2417bff0f32`.
- **Freeze state recorded in metadata:** `APPROVED_FOR_MODEL_TRAINING_OVERLAP_SCREENED`.

Any training notebook must load **only** the rows for the requested train, val or test subset and must never treat `excluded_test_overlap` as a training class/split. E1 and E2 use the **same exact manifest** and label map.

**Limitations to report:** The dHash screen used a particular threshold; manual review focused on flagged pairs. Additional near-duplicates may remain. The publisher's 751-image test split was preserved, but 23 overlapping original training images were excluded, so our training data differ from the unmodified publisher dataset. Rare validation/test classes have only 2/3 observations at minimum, yielding unstable per-class estimates.

## 6. Notebook inventory (complete so far)

1. `01_data_exploration.ipynb` — integrity, class balance, publisher splits and exact SHA-256 checks.
2. `02_prepare_splits.ipynb` — candidate stratified split and dHash candidate detection.
3. `03_review_near_duplicates.ipynb` — side-by-side manual review and verdict CSV.
4. `04_reconcile_overlap_splits.ipynb` — overlap grouping and cleaned candidate assignments.
5. `05_freeze_pakistan_split.ipynb` — final integrity checks, manifest, class map and checksum.

## 7. Report-ready factual summary (draft; edit for space and citations)

> The Pakistani vehicle make-and-model dataset (Mendeley Data, Version 1) contained 3,847 images in 48 classes. Initial integrity checks successfully decoded all 3,847 images and found no byte-identical SHA-256 duplicate groups. The publisher-provided 3,096-image training and 751-image testing partitions contained all 48 classes. We created a stratified validation holdout from the publisher training partition using a fixed seed of 42. Perceptual-hash screening (64-bit dHash, Hamming distance ≤3) flagged 23 cross-boundary pairs between publisher training and testing and 12 between our initial training and validation assignments. Following manual review of these candidate overlaps, 23 original training images connected to publisher test observations were excluded from model development, and the remaining confirmed image groups were kept within a single training or validation partition. The final overlap-screened assignment comprises 2,610 training, 463 validation and 751 test images, with 23 additional images marked as excluded. All 48 classes remain represented in each active split. The publisher's original test images were retained unchanged. Since perceptual screening may miss unflagged near-identical frames, the dataset cannot be assumed entirely leakage-free.

No accuracy, macro-F1, parameter-count, training-time or inference results should be inserted until actually measured. Do not claim the two models were trained yet.

## 8. Remaining work / next milestones

- **Chat 01 — CNN Fundamentals (in progress):** convolution, depthwise separable convolutions, transfer learning, Xception and ConvNeXt.
- **Chat 02 — Dataset Preparation (not created yet):** Stanford Cars acquisition/audit/fixed splits; shared preprocessing and augmentation.
- **Chat 03 — Model Implementation:** PyTorch Dataset/DataLoader; same manifest input; load pretrained models and run smoke tests.
- **Chat 04 — Experiments & Evaluation:** E1–E4, true model results, device-identical timing and plots.
- **Chat 05 — Report & Presentation:** four-page referenced LaTeX report and 15-minute five-member presentation/recording; verify video access rights.
- **Project HQ (this chat):** track major decisions, blockers and documentation consistency.

## 9. Sources of truth

- Assignment: `Project.pdf` in the ChatGPT Project's Sources.
- Configuration: `configs/experiments.yaml` (planned model/training settings).
- Actual experiment runs: `EXPERIMENT_LOG.md` (currently no training runs).
- Final Pakistani data assignments: `splits/pakistan/final_manifest.csv` and `split_metadata.json`.
- GitHub repo: https://github.com/Batsy1857/vehicle-make-model-recognition

This summary is a snapshot dated 28 October 2026. Update it when Stanford Cars is prepared, models are trained or protocols change; never overwrite measured experimental records to match a narrative.

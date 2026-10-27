# Project Plan — Vehicle Make-and-Model Recognition

**Assignment:** Deep Learning, Assignment #2  
**Deadline:** 2 November 2026  
**Team size:** 5  
**Status:** Foundation prepared; experiments not yet run  
**Repository:** https://github.com/Batsy1857/vehicle-make-model-recognition

## 1. Objective

Compare transfer learning with **Xception (2017)** and **ConvNeXt-Tiny (2022)** for fine-grained vehicle make-and-model recognition on two datasets. Determine how predictive performance differs and what accuracy, parameter-count, training-time, and inference-speed trade-offs arise.

**Research question:** Under a documented and as-consistent-as-practical transfer-learning protocol, how do Xception and ConvNeXt-Tiny compare within each vehicle dataset?

Architecture choice must be discussed with the instructor, per the assignment sheet. Record approval before treating it as final.

## 2. Datasets

| Dataset | Reported size | Classes | Source |
| --- | ---: | ---: | --- |
| Pakistani vehicle make/model | 3,847 images | 48 | https://data.mendeley.com/datasets/hj3vvx5946/1 |
| Stanford Cars | 16,185 images | 196 | https://ieeexplore.ieee.org/abstract/document/9460843/ (assignment reference); Stanford Cars dataset |

Counts above come from project briefing and **must be verified after downloading**.

- Audit images, labels, class counts, invalid files and near-duplicates before training.
- Inspect any existing Pakistani dataset splits. Prefer a verified source split; otherwise produce a documented, seeded split and guard against video-frame or same-vehicle leakage where group IDs can be identified.
- Preserve the official Stanford Cars **train/test** split; create validation from the official training subset only.
- Never inspect test predictions to select hyperparameters. Save split manifests and class-to-index maps under version control; do not commit raw images.

## 3. Experiment matrix

| ID | Dataset | Architecture | Status |
| --- | --- | --- | --- |
| E1 | Pakistani | Xception | Not started |
| E2 | Pakistani | ConvNeXt-Tiny | Not started |
| E3 | Stanford Cars | Xception | Not started |
| E4 | Stanford Cars | ConvNeXt-Tiny | Not started |

All four are **planned**, not completed. The source of truth for planned training settings is `configs/experiments.yaml`. The source of truth for measured outcomes is `EXPERIMENT_LOG.md` and committed result artifacts.

## 4. Fair-comparison rules

1. Both architectures use **ImageNet-1K-pretrained weights**, PyTorch and `timm`; record actual package versions.
2. Within a dataset, both use the **same split manifest, label mapping, augmentation policy, image resolution and initial training budget**.
3. Primary comparison uses **224 x 224** inputs for both models (Xception's original pretrained convention is 299 x 299); acknowledge this limitation. Use each pretrained model's own normalization constants.
4. Use the same seed and model-selection metric. Log any run-specific exceptions; do not silently change settings for a single model.
5. Log **total** and **trainable** parameters after replacing the classification head.
6. Benchmark model training/inference on **the same device**, with explicit measurement definitions and GPU synchronization when relevant. Never compare runtimes from unlike GPUs as if they reflect architecture alone.
7. Compare model results **within each dataset**, rather than interpreting raw accuracy across 48 versus 196 classes as a direct difficulty ranking.
8. Separate proposed configurations, pilot measurements and final test results in the report.

## 5. Intended deliverables

- **Four-page LaTeX report**, properly referenced, covering architecture diagrams/mathematical functions, key components, advantages/disadvantages, parameter counts, CS deployment applications, methodology, results and limitations.
- **15-minute group presentation/recording**, approximately equal speaking contribution per person. Verify audio and link-sharing permissions.
- **Reproducible code, split manifests and experiment log** in GitHub; raw datasets and large checkpoints stored outside Git.

Only the group lead submits the report to LMS, as stated in the assignment.

## 6. Roles (names to be assigned)

| Role | Primary focus | Assigned to |
| --- | --- | --- |
| Coordinator / integration | Repository, configuration, experiment consistency | TBD |
| Data lead | Download, audit, splits, leakage checks | TBD |
| Xception lead | Architecture explanation and implementation review | TBD |
| ConvNeXt lead | Architecture explanation and implementation review | TBD |
| Evaluation lead | Metrics, visualizations, comparative analysis | TBD |

All five members review results and contribute equally to the recorded presentation.

## 7. Milestone calendar

| Date | Target |
| --- | --- |
| 27 Oct | Repository foundation; review architectures with instructor |
| 28 Oct | Download/audit data; finalize split manifests |
| 29 Oct | Shared pipeline and smoke tests on both models |
| 30 Oct | Four primary training runs |
| 31 Oct | Final evaluation and timing benchmarks |
| 1 Nov | Report, figures and presentation recording |
| 2 Nov | Validate references, links and permissions; submit |

Targets are subject to dataset/GPU availability. **Priority: obtain valid E1–E4 results before optional extras.**

## 8. Decision register

| Decision | Current state |
| --- | --- |
| Xception vs ConvNeXt-Tiny | Proposed; professor discussion pending |
| PyTorch + timm | Planned |
| VS Code + Google Colab + GitHub | Planned |
| Common 224 x 224 experiment | Planned; pilot smoke test pending |
| Per-model pretrained normalization | Planned |
| Split policy | Pending data audit |
| Optimizer, epochs, fine-tuning schedule | Drafted in YAML; subject to pilot validation |

When a decision changes, update this file, `configs/experiments.yaml` and the experiment log together as appropriate.

## 9. Academic references / source links

- François Chollet (2017), *Xception: Deep Learning with Depthwise Separable Convolutions*. https://arxiv.org/abs/1610.02357
- Zhuang Liu et al. (2022), *A ConvNet for the 2020s*. https://arxiv.org/abs/2201.03545
- Pakistani vehicle dataset: https://data.mendeley.com/datasets/hj3vvx5946/1
- Assignment's public-dataset paper: https://ieeexplore.ieee.org/abstract/document/9460843/

References in the final LaTeX report must be verified and formatted consistently.

# Milestone 2: Pakistani dataset audit

## Correct source
Download **2022 Version 1**, DOI 10.17632/hj3vvx5946.1:
https://data.mendeley.com/datasets/hj3vvx5946/1

The 2022 publication reports 3,847 images, 48 classes and an 80/20 train/test split. A different 2025 Mendeley record reports 3,858 images; do not silently substitute it.

## Install and run

From the repository root on your Mac:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-data.txt
python -m ipykernel install --user --name vehicle-audit --display-name "Python (vehicle-audit)"
```

Then put the *extracted* archive folders under `data/pakistan/`, open `notebooks/01_data_exploration.ipynb` in VS Code, select the `Python (vehicle-audit)` kernel, and **Run All**.

The notebook scans subfolders recursively, prints representative paths, checks image readability, reports inferred labels and split names, flags identical file hashes and exports summary CSVs to `results/data_audit/`.

## Before proceeding

- Confirm the inferred class labels match the real archive layout.
- Confirm whether an official train/test directory structure exists.
- If `unspecified` and `train/test` images are mixed, select a *single* canonical image tree; do not create another split blindly.
- Interpret exact duplicates only as **byte-identical** files. Adjacent video frames and visually near-identical cars require additional audit.
- Do not commit raw images, archives or downloaded model weights.
- Do not train on or augment test images; we will generate training transforms later.

The notebook is audit-only. **No model accuracy, class counts from the downloaded archive, or timing results are claimed until the user runs it.**

"""Read frozen, leakage-reviewed vehicle manifests for PyTorch experiments.

Both Xception and ConvNeXt-Tiny must use the SAME manifest for a given
 dataset. This module intentionally does not create or reshuffle splits.

Place at: src/vehicle_data.py
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Literal

from PIL import Image
from torch.utils.data import Dataset

DatasetName = Literal["pakistan", "stanford_cars"]
SplitName = Literal["train", "val", "test"]

DATASET_SPECS = {
    "pakistan": {
        "manifest": "splits/pakistan/final_manifest.csv",
        "label_map": "splits/pakistan/class_to_idx.json",
        "label_column": "class_idx",
        "classes": 48,
    },
    "stanford_cars": {
        "manifest": "splits/stanford/final_manifest.csv",
        "label_map": "splits/stanford/class_to_idx.json",
        "label_column": "label",
        "classes": 196,
        "verification": "splits/stanford/verification_summary.json",
    },
}


def file_sha256(path: Path) -> str:
    """Return the SHA-256 of the *manifest file*, not of the image pixels."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_manifest(repo_root: Path | str, dataset: DatasetName) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Read and validate the shared manifests without changing split assignment."""
    if dataset not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset {dataset!r}; choose one of {list(DATASET_SPECS)}")
    spec = DATASET_SPECS[dataset]
    root = Path(repo_root).expanduser().resolve()
    manifest_path = root / spec["manifest"]
    mapping_path = root / spec["label_map"]
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing frozen manifest: {manifest_path}")
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Missing class mapping: {mapping_path}")

    with mapping_path.open(encoding="utf-8") as file:
        class_to_idx = json.load(file)
    if len(class_to_idx) != spec["classes"] or set(class_to_idx.values()) != set(range(spec["classes"])):
        raise ValueError(f"{dataset}: expected {spec['classes']} classes with labels 0..{spec['classes'] - 1}")

    if "verification" in spec:
        verification_path = root / spec["verification"]
        if not verification_path.is_file():
            raise FileNotFoundError(f"Missing verification record: {verification_path}")
        with verification_path.open(encoding="utf-8") as file:
            verification = json.load(file)
        if verification.get("status") != "VERIFIED":
            raise ValueError("Stanford Cars is not yet marked VERIFIED")
        if verification.get("manifest_sha256") != file_sha256(manifest_path):
            raise ValueError("Stanford Cars frozen manifest SHA-256 mismatch; do not train")

    with manifest_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"relative_path", "split", "class_name", spec["label_column"]}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{manifest_path}: required columns missing: {required - set(reader.fieldnames or [])}")
        rows = list(reader)

    seen_paths: set[str] = set()
    for row in rows:
        path = row["relative_path"]
        if path in seen_paths:
            raise ValueError(f"Duplicate manifest entry: {path}")
        seen_paths.add(path)
        relative = Path(path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unexpected nonrelative image path: {path}")
        expected_index = class_to_idx.get(row["class_name"])
        if expected_index is None or int(row[spec["label_column"]]) != expected_index:
            raise ValueError(f"Class-to-index mapping mismatch: {path}")
    return rows, class_to_idx


def inspect_manifest(repo_root: Path | str, dataset: DatasetName, check_files: bool = True) -> dict[str, Any]:
    """Audit all splits, labels and optionally all image *paths* (not image decoding)."""
    root = Path(repo_root).expanduser().resolve()
    rows, class_to_idx = load_frozen_manifest(root, dataset)
    counts = Counter(row["split"] for row in rows)
    active = ("train", "val", "test")
    for split in active:
        labels = {row["class_name"] for row in rows if row["split"] == split}
        if labels != set(class_to_idx):
            raise ValueError(f"{dataset}/{split}: class coverage is {len(labels)}/{len(class_to_idx)}")
    if check_files:
        missing = [row["relative_path"] for row in rows if row["split"] in active
                   and not (root / row["relative_path"]).is_file()]
        if missing:
            raise FileNotFoundError(f"{dataset}: {len(missing)} image files missing, e.g. {missing[:3]}")
    return {
        "dataset": dataset,
        "total_manifest_rows": len(rows),
        "class_count": len(class_to_idx),
        "counts": dict(counts),
        "manifest_sha256": file_sha256(root / DATASET_SPECS[dataset]["manifest"]),
        "all_active_image_paths_exist": check_files,
    }


class VehicleManifestDataset(Dataset):
    """PyTorch map-style dataset; returns (RGB image tensor/object, integer label).

    `transform` is injected separately so preprocessing can be standardized
    across architecture runs while respecting their pretrained normalization.
    """

    def __init__(self, repo_root: Path | str, dataset: DatasetName,
                 split: SplitName, transform: Callable | None = None):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.dataset_name = dataset
        self.split = split
        self.transform = transform
        rows, self.class_to_idx = load_frozen_manifest(self.repo_root, dataset)
        self.samples = [
            (str(self.repo_root / row["relative_path"]), int(row[DATASET_SPECS[dataset]["label_column"]]))
            for row in rows if row["split"] == split
        ]
        if not self.samples:
            raise ValueError(f"No images found for {dataset}/{split}")
        self.classes = [None] * len(self.class_to_idx)
        for name, idx in self.class_to_idx.items():
            self.classes[idx] = name

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

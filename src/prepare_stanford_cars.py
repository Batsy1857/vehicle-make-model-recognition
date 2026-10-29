#!/usr/bin/env python3
"""Milestone 2B: verify Stanford Cars and freeze leakage-reviewed manifests.

Run from the existing project root:
    python src/prepare_stanford_cars.py

Downloads ONLY original train/test splits from a pinned community mirror.
Images are copied without recompression. Near-duplicate candidates require review.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

from datasets import Image as HFImage
from datasets import load_dataset
from huggingface_hub import HfApi
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

DATASET_ID = "tanganke/stanford_cars"
EXPECTED = {"train": 8144, "test": 8041}
NUM_CLASSES = 196
REVIEW_COLUMNS = (
    "pair_id", "path_a", "split_a", "class_a", "path_b", "split_b", "class_b",
    "dhash_distance", "identical_bytes", "review_decision",
)
MANIFEST_COLUMNS = (
    "relative_path", "split", "label", "class_name", "source_split", "source_index",
    "sha256", "dhash", "width", "height", "excluded_reason",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(image: Image.Image) -> int:
    """64-bit difference hash: resize to 9x8 greyscale; compare neighboring pixels."""
    grey = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    values = list(grey.getdata())
    bits = 0
    for y in range(8):
        for x in range(8):
            bits = (bits << 1) | int(values[y * 9 + x] > values[y * 9 + x + 1])
    return bits


def read_encoded(row: dict) -> bytes:
    payload = row["image"]
    if payload.get("bytes") is not None:
        return payload["bytes"]
    if payload.get("path"):
        return Path(payload["path"]).read_bytes()
    raise ValueError("No image bytes or local source path in row")


def build_overlap_candidates(rows: list[dict], max_distance: int = 3) -> list[dict]:
    """Four 16-bit chunks guarantee finding all 64-bit dHashes within distance <=3."""
    by_piece = defaultdict(list)
    for i, row in enumerate(rows):
        value = int(row["dhash"], 16)
        for chunk in range(4):
            by_piece[(chunk, (value >> (chunk * 16)) & 0xFFFF)].append(i)

    pairs = set()
    for matching in by_piece.values():
        for i, j in combinations(matching, 2):
            if rows[i]["split"] == rows[j]["split"]:
                continue
            pairs.add((i, j))

    # Also cover byte-identical pairs independently of perceptual hash.
    by_sha = defaultdict(list)
    for i, row in enumerate(rows):
        by_sha[row["sha256"]].append(i)
    for matching in by_sha.values():
        for i, j in combinations(matching, 2):
            if rows[i]["split"] != rows[j]["split"]:
                pairs.add((i, j))

    output = []
    for i, j in sorted(pairs):
        a, b = rows[i], rows[j]
        distance = (int(a["dhash"], 16) ^ int(b["dhash"], 16)).bit_count()
        same_bytes = a["sha256"] == b["sha256"]
        if not same_bytes and distance > max_distance:
            continue
        # Sort paths so pair IDs never depend on CSV row order.
        if a["relative_path"] > b["relative_path"]:
            a, b = b, a
        pair_id = sha256((a["relative_path"] + "|" + b["relative_path"]).encode())[:16]
        output.append({
            "pair_id": pair_id,
            "path_a": a["relative_path"], "split_a": a["split"], "class_a": a["class_name"],
            "path_b": b["relative_path"], "split_b": b["split"], "class_b": b["class_name"],
            "dhash_distance": distance, "identical_bytes": str(same_bytes).lower(),
            "review_decision": "",
        })
    return sorted(output, key=lambda row: row["pair_id"])


def write_csv(path: Path, columns, records: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def review_candidates(path: Path, candidates: list[dict]) -> list[dict]:
    previous = {}
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            previous = {row["pair_id"]: row.get("review_decision", "")
                        for row in csv.DictReader(handle)}
    for item in candidates:
        decision = previous.get(item["pair_id"], "").strip().lower()
        if decision not in {"", "different_images", "confirmed_duplicate"}:
            raise ValueError(f"Invalid review_decision for {item['pair_id']}: {decision}")
        item["review_decision"] = decision
    write_csv(path, REVIEW_COLUMNS, candidates)
    return candidates


def finalize(out_dir: Path, manifest: list[dict], candidates: list[dict],
             revision: str, seed: int, val_fraction: float) -> None:
    allowed = {"", "different_images", "confirmed_duplicate"}
    for pair in candidates:
        pair["review_decision"] = pair["review_decision"].strip().lower()
        if pair["review_decision"] not in allowed:
            raise ValueError(f"Invalid decision in pair {pair['pair_id']}")

    unresolved = [p for p in candidates if not p["review_decision"]]
    confirmed = [p for p in candidates if p["review_decision"] == "confirmed_duplicate"]

    # Calculate exclusions before mutating any splits. Official test always wins;
    # validation wins over train. The reviewer must confirm visual duplication first.
    priority = {"train": 0, "val": 1, "test": 2}
    by_path = {row["relative_path"]: row for row in manifest}
    exclude_paths = set()
    for pair in confirmed:
        a, b = by_path[pair["path_a"]], by_path[pair["path_b"]]
        if a["split"] == b["split"]:
            raise RuntimeError(f"Cross-split candidate became same-split: {pair['pair_id']}")
        rejected = a if priority[a["split"]] < priority[b["split"]] else b
        exclude_paths.add(rejected["relative_path"])
    for path in exclude_paths:
        by_path[path]["split"] = "excluded"
        by_path[path]["excluded_reason"] = "confirmed_cross_split_duplicate"

    counts = Counter(row["split"] for row in manifest)
    status = "NEEDS_MANUAL_REVIEW" if unresolved else "VERIFIED"
    if not unresolved:
        for split in ("train", "val", "test"):
            labels_left = {int(row["label"]) for row in manifest if row["split"] == split}
            if len(labels_left) != NUM_CLASSES:
                raise RuntimeError(f"After exclusions, {split} has only {len(labels_left)} classes")
        if counts["test"] != 8041:
            raise RuntimeError("Official test count changed unexpectedly")
        final_file = out_dir / "final_manifest.csv"
        write_csv(final_file, MANIFEST_COLUMNS, manifest)
        print("VERIFIED. Frozen manifest:", final_file)
    else:
        (out_dir / "final_manifest.csv").unlink(missing_ok=True)
        print(f"NEEDS REVIEW: {len(unresolved)} candidates pending in candidate_pair_review.csv")
        print("Enter different_images or confirmed_duplicate in review_decision column.")
        print("Then run: python src/prepare_stanford_cars.py --review-only")

    with (out_dir / "candidate_manifest.csv").open(newline="", encoding="utf-8") as f:
        original_counts = dict(Counter(row["split"] for row in csv.DictReader(f)))
    summary = {
        "status": status, "dataset": DATASET_ID, "revision_sha": revision,
        "original_counts": EXPECTED, "original_image_count": 16185,
        "classes": NUM_CLASSES, "seed": seed, "validation_fraction": val_fraction,
        "planned_counts_before_review": original_counts,
        "counts_after_reviewed_exclusions": dict(counts),
        "potential_cross_split_overlap_pairs": len(candidates),
        "unreviewed_pairs": len(unresolved),
        "confirmed_pairs": len(confirmed), "excluded_images": len(exclude_paths),
        "images_exported_without_reencoding": True,
        "manifest_sha256": hash_file(out_dir / "final_manifest.csv") if status == "VERIFIED" else None,
        "verified_at_utc": datetime.now(timezone.utc).isoformat() if status == "VERIFIED" else None,
    }
    (out_dir / "verification_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--revision", default=None,
                        help="Optional exact Hugging Face dataset commit SHA")
    parser.add_argument("--review-only", action="store_true",
                        help="Finalize after editing review CSV, without downloading again")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    if not (0 < args.val_fraction < 1):
        raise ValueError("Validation fraction must be between 0 and 1")
    data_dir = root / "data" / "stanford_cars"
    out_dir = root / "splits" / "stanford"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.review_only:
        candidate_file = out_dir / "candidate_manifest.csv"
        review_file = out_dir / "candidate_pair_review.csv"
        if not candidate_file.exists() or not review_file.exists():
            raise FileNotFoundError("Run preparation first to create the candidate manifest and review file")
        with candidate_file.open(newline="", encoding="utf-8") as f:
            manifest = list(csv.DictReader(f))
        with review_file.open(newline="", encoding="utf-8") as f:
            candidates = list(csv.DictReader(f))
        # Ground the source revision and planned counts in the prior verification run.
        prior = json.loads((out_dir / "verification_summary.json").read_text(encoding="utf-8"))
        finalize(out_dir, manifest, candidates, prior["revision_sha"],
                 prior["seed"], prior["validation_fraction"])
        return

    # Pin the dataset to one Hub commit so that later runs use the same source.
    revision = args.revision or HfApi().dataset_info(DATASET_ID).sha
    print("Source:", DATASET_ID, "revision:", revision)
    print("Downloading ONLY original train/test parquet files (not corruption splits) ...")
    datasets = load_dataset(
        DATASET_ID,
        revision=revision,
        data_files={"train": "data/train-*.parquet", "test": "data/test-*.parquet"},
    )
    for part, expected in EXPECTED.items():
        actual = len(datasets[part])
        if actual != expected:
            raise RuntimeError(f"{part}: expected {expected} rows, received {actual}")
        datasets[part] = datasets[part].cast_column("image", HFImage(decode=False))

    names = datasets["train"].features["label"].names
    other_names = datasets["test"].features["label"].names
    if len(names) != NUM_CLASSES or names != other_names or len(set(names)) != NUM_CLASSES:
        raise RuntimeError("Class names differ between splits or are not 196 unique names")
    class_to_idx = {name: i for i, name in enumerate(names)}
    (out_dir / "class_to_idx.json").write_text(
        json.dumps(class_to_idx, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Fixed stratified validation selection WITHIN the original training partition.
    labels = datasets["train"]["label"]
    indices = list(range(len(labels)))
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.val_fraction,
                                      random_state=args.seed)
    train_indices, val_indices = next(splitter.split(indices, labels))
    val_set = set(int(i) for i in val_indices)

    manifest = []
    for source in ("train", "test"):
        part = datasets[source]
        folder = data_dir / source
        folder.mkdir(exist_ok=True)
        for i in range(len(part)):
            row = part[i]
            label = int(row["label"])
            if not 0 <= label < NUM_CLASSES:
                raise RuntimeError(f"Invalid label {label} at {source}/{i}")
            raw = read_encoded(row)
            with Image.open(io.BytesIO(raw)) as img:
                img.load()  # Fails on corrupt/truncated images.
                image_format = (img.format or "").upper()
                if image_format not in {"JPEG", "PNG"}:
                    raise RuntimeError(f"Unexpected image format {image_format} at {source}/{i}")
                suffix = ".jpg" if image_format == "JPEG" else ".png"
                width, height = img.size
                perceptual = dhash(img)
            dest = folder / f"{i:05d}{suffix}"
            if dest.exists():
                if hash_file(dest) != sha256(raw):
                    raise RuntimeError(f"Existing file differs; refusing overwrite: {dest}")
            else:
                dest.write_bytes(raw)  # No recompression or resizing.
            assigned = "test" if source == "test" else ("val" if i in val_set else "train")
            manifest.append({
                "relative_path": dest.relative_to(root).as_posix(),
                "split": assigned, "label": label, "class_name": names[label],
                "source_split": source, "source_index": i,
                "sha256": sha256(raw), "dhash": f"{perceptual:016x}",
                "width": width, "height": height, "excluded_reason": "",
            })
            if (i + 1) % 1000 == 0 or i + 1 == len(part):
                print(f"{source}: checked and exported {i + 1:,}/{len(part):,}")

    planned = Counter(item["split"] for item in manifest)
    if args.val_fraction == 0.15 and dict(planned) != {"train": 6922, "val": 1222, "test": 8041}:
        raise RuntimeError(f"Unexpected stratified split counts: {planned}")
    for split in ("train", "val", "test"):
        actual_labels = {item["label"] for item in manifest if item["split"] == split}
        if len(actual_labels) != NUM_CLASSES:
            raise RuntimeError(f"{split} includes only {len(actual_labels)} of {NUM_CLASSES} classes")

    write_csv(out_dir / "candidate_manifest.csv", MANIFEST_COLUMNS, manifest)
    print("Checking exact and near-duplicates across train, val and test ...")
    candidates = review_candidates(out_dir / "candidate_pair_review.csv",
                                   build_overlap_candidates(manifest))
    finalize(out_dir, manifest, candidates, revision, args.seed, args.val_fraction)


if __name__ == "__main__":
    main()

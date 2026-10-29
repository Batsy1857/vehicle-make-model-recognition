"""Milestone 4: train/validate, final test, and latency benchmark.

Place this file in the repository at src/train_vehicle.py and execute
from the repository root. Reads configs/experiments.yaml and imports the
unchanged Milestone 3 modules from src/.

Examples:
 python src/train_vehicle.py --experiment E1 --run-id E1-run01 --output-dir /content/drive/MyDrive/vehicle_outputs
 python src/train_vehicle.py --experiment E1 --run-id E1-smoke01 --output-dir /content/drive/MyDrive/vehicle_outputs --smoke
 python src/train_vehicle.py --experiment E1 --run-id E1-run01 --output-dir /content/drive/MyDrive/vehicle_outputs --resume
 python src/train_vehicle.py --experiment E1 --run-id E1-run01 --output-dir /content/drive/MyDrive/vehicle_outputs --mode test
 python src/train_vehicle.py --experiment E1 --run-id E1-run01 --output-dir /content/drive/MyDrive/vehicle_outputs --mode benchmark
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import timm
import torchvision
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader

from vehicle_data import VehicleManifestDataset, inspect_manifest
from vehicle_models import configure_transfer_stage, keep_batchnorm_in_eval
from vehicle_transforms import build_vehicle_transforms

ROOT = Path(__file__).resolve().parent.parent
STAGES = ("head_only", "partial_fine_tune")


def stamp():
    return datetime.now(timezone.utc).isoformat()


def json_save(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def torch_save(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def set_seed(seed: int):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def worker_init(worker_id: int):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def loader(data, cfg, *, train=False, epoch_seed=42, smoke=False):
    g = torch.Generator()
    g.manual_seed(int(epoch_seed))
    workers = int(cfg["training"]["num_workers"])
    return DataLoader(
        data,
        batch_size=int(cfg["training"]["batch_size"] if train else cfg["training"]["eval_batch_size"]),
        shuffle=train,
        generator=g,
        worker_init_fn=worker_init,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def evaluate(model, data_loader, device, classes, loss_fn, max_batches=None):
    model.eval()
    labels_all, preds_all = [], []
    loss_sum, n_seen, top5_correct = 0.0, 0, 0
    t0 = time.perf_counter()
    with torch.inference_mode():
        for i, (images, labels) in enumerate(data_loader):
            if max_batches is not None and i >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                logits = model(images)
                loss = loss_fn(logits, labels)
            predicted = logits.argmax(dim=1)
            top5_correct += (logits.topk(min(5, classes), dim=1).indices == labels[:, None]).any(dim=1).sum().item()
            size = labels.numel()
            loss_sum += float(loss.item()) * size
            n_seen += size
            labels_all.extend(labels.cpu().tolist())
            preds_all.extend(predicted.cpu().tolist())
    if device.type == "cuda":
        torch.cuda.synchronize()
    if not n_seen:
        raise RuntimeError("Evaluation loader produced zero images")
    metrics = {
        "loss": loss_sum / n_seen,
        "accuracy": float(accuracy_score(labels_all, preds_all)),
        "macro_f1": float(f1_score(labels_all, preds_all, labels=list(range(classes)), average="macro", zero_division=0)),
        "top5_accuracy": top5_correct / n_seen,
        "n": n_seen,
        "seconds": time.perf_counter() - t0,
    }
    return metrics, labels_all, preds_all


def train_epoch(model, data_loader, device, loss_fn, optimizer, scaler, max_batches=None):
    model.train()
    # Reuse the Milestone 3 helper: pretrained BatchNorm statistics stay frozen.
    keep_batchnorm_in_eval(model)
    n_seen, loss_sum = 0, 0.0
    t0 = time.perf_counter()
    for i, (images, labels) in enumerate(data_loader):
        if max_batches is not None and i >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
            logits = model(images)
            loss = loss_fn(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at batch {i}: {loss.item()}")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        size = labels.numel()
        loss_sum += loss.item() * size
        n_seen += size
    if device.type == "cuda":
        torch.cuda.synchronize()
    if not n_seen:
        raise RuntimeError("Training loader produced zero images")
    return {"train_loss": loss_sum / n_seen, "train_images_seen": n_seen, "train_seconds": time.perf_counter() - t0}


def append_history(path, row):
    fields = ["stage", "epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1", "val_top5_accuracy", "train_seconds", "val_seconds", "peak_gpu_memory_mb", "utc_end"]
    existed = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not existed:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})


def load_own_checkpoint(path, map_location="cpu"):
    # Only load files created by this project; never unpickle untrusted checkpoints.
    return torch.load(path, map_location=map_location, weights_only=True)


def metadata(cfg, exp, model, audit, parameters, device):
    return {
        "created_utc": stamp(),
        "experiment": exp,
        "experiment_definition": cfg["experiments"][exp],
        "model_timm_name": cfg["models"][cfg["experiments"][exp]["model"]]["timm_name"],
        "pretrained": True,
        "input_shape": [3, *cfg["data_pipeline"]["input_size"]],
        "data_pipeline": cfg["data_pipeline"],
        "training_config": cfg["training"],
        "random_seed": cfg["random_seed"],
        "dataset_audit": audit,
        "git_commit": git_sha(),
        "parameters_head_only": parameters,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "numpy": np.__version__,
        "cuda_build": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "device": str(device),
        "hostname": platform.node(),
        "precision": "AMP fp16 on CUDA; fp32 otherwise",
    }


def prepare(args):
    with (ROOT / "configs/experiments.yaml").open() as f:
        cfg = yaml.safe_load(f)
    exp = cfg["experiments"][args.experiment]
    ds_key, model_key = exp["dataset"], exp["model"]
    train_cfg = cfg["training"]
    assert train_cfg["loss"] == "cross_entropy" and train_cfg["optimizer"] == "AdamW"
    assert train_cfg["gradient_accumulation_steps"] == 1, "This script does not implement accumulation >1"
    assert train_cfg["checkpoint_selection"] == "max_validation_macro_f1"
    assert train_cfg["early_stopping"]["monitor"] == "validation_macro_f1"
    assert train_cfg["mixed_precision"] == "cuda_only"
    for stage in STAGES:
        assert train_cfg["stages"][stage]["enabled"], "Both stages must be enabled"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Milestone 4 requires GPU. In Colab select Runtime > Change runtime type > GPU")
    set_seed(int(cfg["random_seed"]))
    audit = inspect_manifest(ROOT, ds_key, check_files=True)
    assert audit["manifest_sha256"] == cfg["datasets"][ds_key]["manifest_sha256"], "Unexpected manifest hash; do not train"
    for split, expected in cfg["datasets"][ds_key]["verified_split_counts"].items():
        observed = audit["counts"].get("excluded_test_overlap" if ds_key == "pakistan" and split == "excluded" else split, 0)
        assert observed == expected, (split, observed, expected)
    model = timm.create_model(cfg["models"][model_key]["timm_name"], pretrained=args.mode == "train" and not args.resume,
                              num_classes=int(cfg["datasets"][ds_key]["verified_classes"]))
    # For resume/test/benchmark timm must not fetch pretrained weights: weights are in the saved checkpoint.
    stats = configure_transfer_stage(model, model_key, "head_only")
    model = model.to(device)
    train_tf, eval_tf = build_vehicle_transforms(cfg, model)
    output = Path(args.output_dir).expanduser().resolve() / args.run_id
    output.mkdir(parents=True, exist_ok=True)
    return cfg, ds_key, model_key, model, train_tf, eval_tf, stats, audit, device, output


def run_training(args):
    cfg, ds_key, model_key, model, train_tf, eval_tf, stats, audit, device, output = prepare(args)
    if args.smoke and args.resume:
        raise ValueError("Smoke test cannot resume")
    if args.resume and not (output / "last.pt").exists():
        raise FileNotFoundError("--resume was specified but last.pt is missing")
    if not args.resume and (output / "last.pt").exists():
        raise FileExistsError("Existing run detected; use --resume or a unique --run-id")
    if not args.resume:
        json_save(output / "metadata.json", {**metadata(cfg, args.experiment, model, audit, stats, device),
            "run_id": args.run_id, "kind": "smoke" if args.smoke else "final", "status": "running"})
    train_ds = VehicleManifestDataset(ROOT, ds_key, "train", transform=train_tf)
    val_ds = VehicleManifestDataset(ROOT, ds_key, "val", transform=eval_tf)
    loss_fn = nn.CrossEntropyLoss()
    tcfg = cfg["training"]
    patience = int(tcfg["early_stopping"]["patience"])
    best_global, best_head, best_fine, start_stage, start_epoch, bad_epochs = -1., -1., -1., 0, 0, 0
    total_training_seconds = 0.0
    resume_ckpt = None
    if args.resume:
        resume_ckpt = load_own_checkpoint(output / "last.pt")
        if resume_ckpt["experiment"] != args.experiment or resume_ckpt["run_id"] != args.run_id:
            raise ValueError("Checkpoint does not match experiment and run")
        if resume_ckpt["git_commit"] != git_sha():
            raise ValueError("Git commit changed since run began; restore original code before resuming")
        if resume_ckpt["manifest_sha256"] != audit["manifest_sha256"]:
            raise ValueError("Manifest changed since run began; do not resume")
        best_global = float(resume_ckpt["best_global"])
        best_head = float(resume_ckpt["best_head"])
        best_fine = float(resume_ckpt.get("best_fine", -1.))
        start_stage, start_epoch = int(resume_ckpt["stage_index"]), int(resume_ckpt["next_epoch"])
        bad_epochs = int(resume_ckpt["bad_epochs"])
        total_training_seconds = float(resume_ckpt["total_training_seconds"])
        model.load_state_dict(resume_ckpt["model_state"])
        if "torch_rng_state" in resume_ckpt:
            torch.set_rng_state(resume_ckpt["torch_rng_state"])
            torch.cuda.set_rng_state_all(resume_ckpt["cuda_rng_states"])
        print(f"Resuming {args.run_id}: stage index {start_stage}, next epoch {start_epoch + 1}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for stage_index in range(start_stage, len(STAGES)):
        stage = STAGES[stage_index]
        stage_conf = tcfg["stages"][stage]
        maximum = 1 if args.smoke else int(stage_conf["epochs_max"])
        stage_stats = configure_transfer_stage(model, model_key, stage)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                      lr=float(stage_conf["learning_rate"]), weight_decay=float(tcfg["weight_decay"]))
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        epoch_begin = start_epoch if stage_index == start_stage else 0
        if resume_ckpt and stage_index == start_stage:
            optimizer.load_state_dict(resume_ckpt["optimizer_state"])
            scaler.load_state_dict(resume_ckpt["scaler_state"])
        elif stage_index == 1:
            # Fine-tune from best HEAD checkpoint, not last head epoch.
            head_ckpt = load_own_checkpoint(output / "best_head.pt")
            model.load_state_dict(head_ckpt["model_state"])
            print("Stage 2 starts from best Stage 1 validation checkpoint")
            bad_epochs = 0
        print(f"{args.experiment} {stage}: trainable parameters {stage_stats['trainable']:,}")
        for epoch in range(epoch_begin, maximum):
            # Rebuild shuffled loader with a fixed per-epoch seed (restart-safe ordering).
            train_loader = loader(train_ds, cfg, train=True,
                                  epoch_seed=int(cfg["random_seed"]) + stage_index * 1000 + epoch)
            val_loader = loader(val_ds, cfg, epoch_seed=int(cfg["random_seed"]))
            tr = train_epoch(model, train_loader, device, loss_fn, optimizer, scaler, 2 if args.smoke else None)
            val, _, _ = evaluate(model, val_loader, device, len(val_ds.classes), loss_fn, 2 if args.smoke else None)
            total_training_seconds += tr["train_seconds"]
            gpu_mem = torch.cuda.max_memory_allocated() / 1_000_000
            row = {"stage": stage, "epoch": epoch + 1, **tr, "val_loss": val["loss"],
                   "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"],
                   "val_top5_accuracy": val["top5_accuracy"], "val_seconds": val["seconds"],
                   "peak_gpu_memory_mb": gpu_mem, "utc_end": stamp()}
            append_history(output / "history.csv", row)
            improved_stage = val["macro_f1"] > (best_head if stage_index == 0 else best_fine)
            if improved_stage:
                bad_epochs = 0
                if stage_index == 0:
                    best_head = val["macro_f1"]
                else:
                    best_fine = val["macro_f1"]
            else:
                bad_epochs += 1
            ckpt_common = {"experiment": args.experiment, "run_id": args.run_id,
                          "git_commit": git_sha(), "manifest_sha256": audit["manifest_sha256"],
                          "model_state": model.state_dict(), "best_stage": stage,
                          "best_epoch": epoch + 1, "val_metrics": val, "parameter_stats": stage_stats}
            if improved_stage and stage_index == 0:
                torch_save(output / "best_head.pt", ckpt_common)
            if val["macro_f1"] > best_global:
                best_global = val["macro_f1"]
                torch_save(output / "best.pt", ckpt_common)
            torch_save(output / "last.pt", {
                **ckpt_common, "stage_index": stage_index, "next_epoch": epoch + 1,
                "best_global": best_global, "best_head": best_head,
                "best_fine": best_fine if stage_index == 1 else -1.,
                "bad_epochs": bad_epochs, "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(), "total_training_seconds": total_training_seconds,
                "torch_rng_state": torch.get_rng_state(), "cuda_rng_states": torch.cuda.get_rng_state_all(),
            })
            print(f"  epoch {epoch+1:02d}/{maximum}: loss={tr['train_loss']:.4f} val_acc={val['accuracy']:.4f} "
                  f"val_macro_F1={val['macro_f1']:.4f} train_s={tr['train_seconds']:.1f} gpu_peak_MB={gpu_mem:.0f}", flush=True)
            if not args.smoke and tcfg["early_stopping"]["enabled"] and bad_epochs >= patience:
                print(f"  Early stopping in {stage} after {bad_epochs} epochs without validation macro-F1 improvement")
                stopped = load_own_checkpoint(output / "last.pt")
                stopped["next_epoch"] = maximum
                stopped["stop_reason"] = "early_stopping"
                torch_save(output / "last.pt", stopped)
                break
        # Move forward to new stage after the last epoch (if run interrupted, resume works from stage end).
        resume_ckpt = None
        start_epoch = 0
        bad_epochs = 0
        if stage_index == 0:
            best_fine = -1.
    m = json.loads((output / "metadata.json").read_text())
    selected = load_own_checkpoint(output / "best.pt")
    m.update({"status": "smoke_completed" if args.smoke else "completed",
              "finished_utc": stamp(), "best_validation_macro_f1": best_global,
              "best_checkpoint": str(output / "best.pt"),
              "selected_stage": selected["best_stage"], "selected_epoch": selected["best_epoch"],
              "total_train_seconds_excluding_validation": total_training_seconds,
              "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1_000_000,
              "parameters_partial_fine_tune": stage_stats})
    json_save(output / "metadata.json", m)
    print(f"DONE; artifacts saved to {output}")


def run_test(args):
    cfg, ds_key, model_key, model, _, eval_tf, stats, audit, device, output = prepare(args)
    ckpt_path = output / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No model selection checkpoint found: {ckpt_path}")
    if not (output / "metadata.json").exists():
        raise FileNotFoundError("Missing run metadata; do not evaluate unverified run")
    run_info = json.loads((output / "metadata.json").read_text())
    if run_info.get("kind") == "smoke" or run_info.get("status") != "completed":
        raise ValueError("Only a completed full-training run may be used for final test evaluation")
    if (output / "test_metrics.json").exists() and not args.force_test:
        raise FileExistsError("Final test already evaluated. Use --force-test only for a documented reason")
    ckpt = load_own_checkpoint(ckpt_path)
    if ckpt["experiment"] != args.experiment or ckpt["manifest_sha256"] != audit["manifest_sha256"]:
        raise ValueError("Selected checkpoint does not match experiment/manifests")
    model.load_state_dict(ckpt["model_state"])
    test_ds = VehicleManifestDataset(ROOT, ds_key, "test", transform=eval_tf)
    ld = loader(test_ds, cfg)
    test_metrics, truth, predicted = evaluate(model, ld, device, len(test_ds.classes), nn.CrossEntropyLoss())
    precision, recall, f1, support = precision_recall_fscore_support(truth, predicted,
        labels=list(range(len(test_ds.classes))), zero_division=0)
    with (output / "per_class_test.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "precision", "recall", "f1", "support"])
        writer.writerows((i, test_ds.classes[i], float(precision[i]), float(recall[i]), float(f1[i]), int(support[i]))
                         for i in range(len(test_ds.classes)))
    np.save(output / "test_confusion_matrix.npy", confusion_matrix(truth, predicted,
                                                              labels=list(range(len(test_ds.classes)))))
    result = {"experiment": args.experiment, "run_id": args.run_id, "evaluated_utc": stamp(),
              "selected_checkpoint": str(ckpt_path), "selected_stage": ckpt["best_stage"],
              "selected_epoch": ckpt["best_epoch"], "selected_val_metrics": ckpt["val_metrics"],
              "test_metrics": test_metrics, "dataset_manifest_sha256": audit["manifest_sha256"]}
    json_save(output / "test_metrics.json", result)
    print("FINAL TEST:", json.dumps(result, indent=2))


def run_benchmark(args):
    cfg, _, _, model, _, _, _, _, device, output = prepare(args)
    ckpt = load_own_checkpoint(output / "best.pt")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    bcfg = cfg["evaluation"]["runtime_benchmark"]
    if not bcfg["same_device_required"] or not bcfg["synchronize_gpu_before_and_after_timing"]:
        raise ValueError("Benchmark protocol should enforce matched GPU and synchronization")
    result = {"experiment": args.experiment, "run_id": args.run_id, "gpu": torch.cuda.get_device_name(0),
              "utc": stamp(), "inference_mode": True, "preprocessing_included": False, "data_loader_included": False}
    with torch.inference_mode():
        for batch in (1, 16):
            x = torch.randn(batch, 3, *cfg["data_pipeline"]["input_size"], device=device)
            # Use FP32 for both models; comparison must use the identical choice.
            for _ in range(int(bcfg["warmup_iterations"])):
                _ = model(x)
            durations = []
            for _ in range(int(bcfg["timed_iterations"])):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model(x)
                torch.cuda.synchronize()
                durations.append((time.perf_counter() - t0) * 1000 / batch)
            result[f"batch{batch}_median_ms_per_image"] = float(np.median(durations))
            result[f"batch{batch}_p95_ms_per_image"] = float(np.percentile(durations, 95))
    json_save(output / "benchmark.json", result)
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True, choices=("E1", "E2", "E3", "E4"))
    p.add_argument("--run-id", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--mode", choices=("train", "test", "benchmark"), default="train")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force-test", action="store_true")
    args = p.parse_args()
    if args.smoke and args.mode != "train":
        p.error("--smoke only makes sense for --mode train")
    if args.resume and args.mode != "train":
        p.error("--resume only makes sense for --mode train")
    if args.mode == "train":
        run_training(args)
    elif args.mode == "test":
        run_test(args)
    else:
        run_benchmark(args)

if __name__ == "__main__":
    main()

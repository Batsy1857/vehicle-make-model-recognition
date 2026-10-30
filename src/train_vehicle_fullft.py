"""Notebook 12 exploratory full-backbone fine-tuning (v3), separate from Notebook 11.

Imports PURE training/data utilities from the unchanged v2 script; never calls
its v2 train/test entry points or reads its checkpoints. Run from repository root.

Examples:
  python src/train_vehicle_fullft.py --experiment E3 --run-id E3-v3-memory --mode memory --output-dir /kaggle/working/v3_runs
  python src/train_vehicle_fullft.py --experiment E3 --run-id E3-v3-smoke01 --smoke --output-dir /kaggle/working/v3_runs
  python src/train_vehicle_fullft.py --experiment E3 --run-id E3-v3-run01 --output-dir /kaggle/working/v3_runs
  python src/train_vehicle_fullft.py --experiment E3 --run-id E3-v3-run01 --output-dir /kaggle/working/v3_runs --resume
  python src/train_vehicle_fullft.py --experiment E3 --run-id E3-v3-run01 --output-dir /kaggle/working/v3_runs --mode test --acknowledge-posthoc-test
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
from pathlib import Path

import numpy as np
import timm
import torch
import torchvision
import yaml
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import nn
from timm.data import resolve_model_data_config
from vehicle_data import DATASET_SPECS, VehicleManifestDataset, inspect_manifest
from vehicle_models import configure_transfer_stage, keep_batchnorm_in_eval
# Existing functions are read-only dependencies; v2 itself is not changed.
from train_vehicle_retrain import (
    sha256, now, save_json, save_torch, own_load, git_sha,
    seed_everything, make_loader, epoch_lr, train_one_epoch, evaluate,
    save_history, initialize_model,
)

ROOT = Path(__file__).resolve().parent.parent
ORIGINAL_CONFIG = ROOT / 'configs/experiments.yaml'
BASELINE_RECIPE = ROOT / 'configs/retraining.yaml'
V3_RECIPE = ROOT / 'configs/full_finetuning.yaml'
STAGES = ('head_only', 'full_fine_tune')


def configure_v3_stage(model, model_key, stage):
    if stage == 'head_only':
        return configure_transfer_stage(model, model_key, 'head_only')
    if stage != 'full_fine_tune':
        raise ValueError(f'Unknown v3 stage: {stage}')
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    # BatchNorm running statistics are frozen at train-time by
    # train_one_epoch()->keep_batchnorm_in_eval(). Affine gamma/beta train.
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total == trainable, 'Full fine-tuning must unfreeze 100% of parameters'
    return {'total': total, 'trainable': trainable, 'frozen': 0}


def load_protocol(args):
    with ORIGINAL_CONFIG.open() as f:
        cfg = yaml.safe_load(f)
    with V3_RECIPE.open() as f:
        recipe = yaml.safe_load(f)
    with BASELINE_RECIPE.open() as f:
        v2 = yaml.safe_load(f)
    # Check against both the immutable shared config and v2 hyperparameters.
    assert recipe['seed'] == cfg['random_seed'] == v2['seed'] == 42
    assert cfg['training']['batch_size'] == 16
    assert cfg['training']['gradient_accumulation_steps'] == 1
    assert recipe['optimizer'] == v2['optimizer'] == cfg['training']['optimizer'] == 'AdamW'
    assert recipe['weight_decay'] == v2['weight_decay'] == cfg['training']['weight_decay']
    for key in ('epochs', 'learning_rate'):
        assert recipe['stages']['head_only'][key] == v2['stages']['head_only'][key]
    for key in ('epochs', 'peak_learning_rate', 'minimum_learning_rate',
                'warmup_epochs', 'warmup_start_fraction', 'schedule', 'early_stopping_patience'):
        assert recipe['stages']['full_fine_tune'][key] == v2['stages']['partial_fine_tune'][key]
    assert recipe['init'] == v2['init'] == 'timm_imagenet_pretrained_new_classifier'
    assert recipe['checkpoint_selection'] == v2['checkpoint_selection'] == 'validation_macro_f1'
    assert recipe['precision'] == v2['precision'] == 'cuda_amp_fp16'
    assert recipe['image_pipeline'] == 'reuse_configs_experiments_yaml'
    assert recipe['batch_size'] == 'reuse_configs_experiments_yaml'
    assert recipe['full_backbone_trainable'] is True
    assert recipe['batchnorm_training_policy'] == 'affine_trainable_running_statistics_eval'
    assert recipe['no_test_data_during_training'] is True
    assert recipe['output_directory_name'] == 'v3_runs'
    exp = cfg['experiments'][args.experiment]
    ds_key, model_key = exp['dataset'], exp['model']
    audit = inspect_manifest(ROOT, ds_key, check_files=True)
    assert audit['manifest_sha256'] == cfg['datasets'][ds_key]['manifest_sha256'], 'Frozen manifest changed'
    for split, count in cfg['datasets'][ds_key]['verified_split_counts'].items():
        observed = 'excluded_test_overlap' if ds_key == 'pakistan' and split == 'excluded' else split
        assert audit['counts'].get(observed, 0) == int(count), f'Unexpected {ds_key}/{split} split count'
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required. Select Kaggle T4 accelerator.')
    return cfg, recipe, exp, ds_key, model_key, audit, torch.device('cuda:0')


def output_path(args):
    root = Path(args.output_dir).expanduser().resolve()
    if root.name != 'v3_runs':
        raise ValueError('Notebook 12 must save under a dedicated folder named v3_runs')
    if not re.fullmatch(rf'{args.experiment}-v3-[A-Za-z0-9][A-Za-z0-9_-]*', args.run_id):
        raise ValueError('Run ID must be e.g. E3-v3-run01; v2 IDs are forbidden')
    if args.smoke != ('smoke' in args.run_id.lower()) and args.mode == 'train':
        raise ValueError('Smoke run IDs must contain smoke; full run IDs must not')
    return root / args.run_id


def memory_probe(args):
    cfg, recipe, exp, ds_key, model_key, audit, device = load_protocol(args)
    seed_everything(int(recipe['seed']))
    print('Memory probe: synthetic training batch ONLY; no checkpoint or data evaluation.')
    model, _, _ = initialize_model(cfg, ds_key, model_key, device, pretrained=False)
    stats = configure_v3_stage(model, model_key, 'full_fine_tune')
    model.train()
    keep_batchnorm_in_eval(model)
    torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')
    batch = int(cfg['training']['batch_size'])
    nclasses = int(cfg['datasets'][ds_key]['verified_classes'])
    x = torch.randn(batch, 3, *cfg['data_pipeline']['input_size'], device=device)
    y = torch.randint(nclasses, (batch,), device=device)
    for _ in range(2):  # Include AdamW optimizer-state allocation, not just forward activation memory
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            loss = nn.functional.cross_entropy(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize(device)
    peak_alloc = torch.cuda.max_memory_allocated(device) / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3
    print(json.dumps({
        'experiment': args.experiment, 'gpu': torch.cuda.get_device_name(device),
        'batch': batch, 'trainable_parameters': stats['trainable'],
        'peak_allocated_GiB': round(peak_alloc, 3),
        'peak_reserved_GiB': round(peak_reserved, 3),
        'gpu_total_GiB': round(total, 3),
        'reserved_headroom_GiB': round(total - peak_reserved, 3),
        'note': 'Synthetic probe; the real data pipeline may use additional GPU memory',
    }, indent=2))
    if peak_reserved / total >= 0.90:
        raise RuntimeError('Insufficient T4 GPU memory headroom (<10%). Do NOT run full training at batch 16.')


def train(args):
    if args.resume and args.smoke:
        raise ValueError('Smoke jobs cannot resume')
    out = output_path(args)
    cfg, recipe, exp, ds_key, model_key, audit, device = load_protocol(args)
    if args.resume:
        if not (out / 'last.pt').is_file() or not (out / 'metadata.json').is_file():
            raise FileNotFoundError('Resume requires v3 last.pt and metadata.json')
        original_meta = json.loads((out / 'metadata.json').read_text())
        if original_meta['status'] != 'running':
            raise ValueError('Only interrupted, running v3 jobs can resume')
    elif out.exists():
        raise FileExistsError(f'v3 output path already exists: {out}')

    seed_everything(int(recipe['seed']))
    model, train_tf, val_tf = initialize_model(cfg, ds_key, model_key, device, pretrained=not args.resume)
    train_ds = VehicleManifestDataset(ROOT, ds_key, 'train', transform=train_tf)
    val_ds = VehicleManifestDataset(ROOT, ds_key, 'val', transform=val_tf)
    head_stats = configure_v3_stage(model, model_key, 'head_only')
    full_stats = configure_v3_stage(model, model_key, 'full_fine_tune')
    configure_v3_stage(model, model_key, 'head_only')
    norm = resolve_model_data_config(model)
    maxima = [1, 1] if args.smoke else [int(recipe['stages']['head_only']['epochs']),
                                        int(recipe['stages']['full_fine_tune']['epochs'][ds_key])]
    ft = recipe['stages']['full_fine_tune']
    patience = int(ft['early_stopping_patience'])
    head_lr = float(recipe['stages']['head_only']['learning_rate'])
    out.mkdir(parents=True, exist_ok=True)
    metadata_path = out / 'metadata.json'
    meta = {
        'created_utc': now(), 'run_id': args.run_id, 'experiment': args.experiment,
        'experiment_definition': exp, 'status': 'running',
        'kind': 'smoke' if args.smoke else 'exploratory_posthoc_full_backbone',
        'git_commit': git_sha(), 'trainer_sha256': sha256(Path(__file__)),
        'recipe_sha256': sha256(V3_RECIPE), 'v2_recipe_sha256': sha256(BASELINE_RECIPE),
        'original_config_sha256': sha256(ORIGINAL_CONFIG),
        'manifest_sha256': audit['manifest_sha256'],
        'label_map_sha256': sha256(ROOT / DATASET_SPECS[ds_key]['label_map']),
        'dataset_audit': audit, 'model_timm_name': cfg['models'][model_key]['timm_name'],
        'pretrained': True, 'source': 'fresh_independent_ImageNet_init_not_v2_checkpoint',
        'training_recipe': recipe, 'shared_training_config': cfg['training'],
        'input_shape': [3, *cfg['data_pipeline']['input_size']],
        'pretrained_normalization_mean': norm['mean'], 'pretrained_normalization_std': norm['std'],
        'parameters_head_only': head_stats, 'parameters_full_fine_tune': full_stats,
        'gpu': torch.cuda.get_device_name(device), 'cuda_build': torch.version.cuda,
        'python': platform.python_version(), 'torch': torch.__version__,
        'torchvision': torchvision.__version__, 'timm': timm.__version__,
        'seed': recipe['seed'], 'validation_selection': 'max_macro_f1',
        'posthoc_test_caveat': recipe['posthoc_test_caveat'],
        'no_test_data_evaluation_during_training': True,
    }
    if args.resume:
        old = json.loads(metadata_path.read_text())
        keys = ('experiment', 'git_commit', 'trainer_sha256', 'recipe_sha256',
                'v2_recipe_sha256', 'original_config_sha256', 'manifest_sha256', 'label_map_sha256')
        if any(old.get(k) != meta[k] for k in keys):
            raise ValueError('Code/config/manifest changed: unsafe to resume')
    else:
        save_json(metadata_path, meta)

    loss_fn = nn.CrossEntropyLoss()
    torch.cuda.reset_peak_memory_stats(device)
    next_stage, next_epoch = 0, 0
    best_f1, best_head, best_full = -1., -1., -1.
    best_stage, best_epoch = None, None
    nonimprove, total_seconds, history, state = 0, 0., [], None
    if args.resume:
        state = own_load(out / 'last.pt')
        if any(state.get(k) != meta[k] for k in ('run_id', 'git_commit', 'recipe_sha256', 'manifest_sha256')):
            raise ValueError('Resume checkpoint mismatches v3 run')
        model.load_state_dict(state['model_state'], strict=True)
        torch.set_rng_state(state['torch_rng_state'])
        torch.cuda.set_rng_state_all(state['cuda_rng_states'])
        next_stage, next_epoch = int(state['next_stage']), int(state['next_epoch'])
        if next_stage >= 2:
            raise ValueError('All epochs already finished: finalize/review run instead')
        best_f1, best_head, best_full = map(float, [state['best_f1'], state['best_head'], state['best_full']])
        best_stage, best_epoch = state['best_stage'], state['best_epoch']
        nonimprove, total_seconds = int(state['nonimprove']), float(state['total_seconds'])
        history = state['history_rows']
        save_history(out / 'history.csv', history)
        print('Resuming', args.run_id, 'at', STAGES[next_stage], 'epoch', next_epoch + 1, flush=True)

    for stage_index in range(next_stage, 2):
        stage = STAGES[stage_index]
        start = next_epoch if stage_index == next_stage else 0
        if stage_index == 1 and start == 0:
            model.load_state_dict(own_load(out / 'best_head.pt')['model_state'], strict=True)
        stats = configure_v3_stage(model, model_key, stage)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                    lr=head_lr if stage_index == 0 else float(ft['peak_learning_rate']),
                    weight_decay=float(recipe['weight_decay']))
        scaler = torch.amp.GradScaler('cuda', enabled=True)
        if args.resume and state is not None and stage_index == next_stage and start > 0:
            optimizer.load_state_dict(state['optimizer_state'])
            scaler.load_state_dict(state['scaler_state'])
        print(f'{args.experiment} {stage}: {stats["trainable"]:,}/{stats["total"]:,} parameters', flush=True)
        for epoch in range(start, maxima[stage_index]):
            lr = (head_lr if stage_index == 0 else float(ft['peak_learning_rate'])) if args.smoke else (
                head_lr if stage_index == 0 else epoch_lr(
                    epoch, maxima[stage_index], float(ft['peak_learning_rate']),
                    float(ft['minimum_learning_rate']), int(ft['warmup_epochs']),
                    float(ft['warmup_start_fraction'])))
            for group in optimizer.param_groups:
                group['lr'] = lr
            tr = train_one_epoch(model, make_loader(train_ds, cfg, train=True,
                     stage_index=stage_index, epoch=epoch), optimizer, scaler, loss_fn, device,
                     max_batches=2 if args.smoke else None)
            val = evaluate(model, make_loader(val_ds, cfg, stage_index=stage_index, epoch=epoch),
                      len(val_ds.classes), loss_fn, device, max_batches=2 if args.smoke else None)
            total_seconds += tr['train_seconds']
            current = float(val['macro_f1'])
            improved_local = current > (best_head if stage_index == 0 else best_full)
            if stage_index == 0 and improved_local:
                best_head = current
            if stage_index == 1:
                if improved_local:
                    best_full, nonimprove = current, 0
                elif epoch + 1 > int(ft['warmup_epochs']):
                    nonimprove += 1
            if current > best_f1:
                best_f1, best_stage, best_epoch = current, stage, epoch + 1
            row = {'stage': stage, 'epoch': epoch + 1, 'learning_rate': lr, **tr,
                   'val_loss': val['loss'], 'val_accuracy': val['accuracy'],
                   'val_macro_f1': val['macro_f1'], 'val_top5_accuracy': val['top5_accuracy'],
                   'val_images_seen': val['n'], 'val_seconds': val['seconds'],
                   'peak_gpu_memory_mb': torch.cuda.max_memory_allocated(device) / 1e6,
                   'utc_end': now()}
            history.append(row)
            ckpt = {'experiment': args.experiment, 'run_id': args.run_id,
                    'manifest_sha256': audit['manifest_sha256'], 'git_commit': git_sha(),
                    'recipe_sha256': sha256(V3_RECIPE), 'model_state': model.state_dict(),
                    'best_stage': stage, 'best_epoch': epoch + 1,
                    'val_metrics': val, 'parameter_stats': stats}
            if stage_index == 0 and improved_local:
                save_torch(out / 'best_head.pt', ckpt)
            if stage_index == 1 and improved_local:
                save_torch(out / 'best_finetune.pt', ckpt)
            if (best_stage, best_epoch) == (stage, epoch + 1) and current == best_f1:
                save_torch(out / 'best.pt', ckpt)
            stop = stage_index == 1 and not args.smoke and nonimprove >= patience
            advance = epoch + 1 == maxima[stage_index] or stop
            following_stage = stage_index + 1 if advance else stage_index
            following_epoch = 0 if advance else epoch + 1
            save_torch(out / 'last.pt', {**ckpt,
                'next_stage': following_stage, 'next_epoch': following_epoch,
                'best_f1': best_f1, 'best_head': best_head, 'best_full': best_full,
                'best_stage': best_stage, 'best_epoch': best_epoch,
                'nonimprove': nonimprove if stage_index == 1 else 0,
                'total_seconds': total_seconds, 'optimizer_state': optimizer.state_dict(),
                'scaler_state': scaler.state_dict(), 'torch_rng_state': torch.get_rng_state(),
                'cuda_rng_states': torch.cuda.get_rng_state_all(), 'history_rows': history})
            save_history(out / 'history.csv', history)
            print(f'{stage} e{epoch+1:02}/{maxima[stage_index]} LR={lr:.2e} '
                  f'train_loss={tr["train_loss"]:.4f} val_F1={current:.4f} '
                  f'best_F1={best_f1:.4f} train_s={tr["train_seconds"]:.1f}', flush=True)
            if stop:
                print('Early stop after', patience, 'non-improving full-ft epochs', flush=True)
                break
        nonimprove = 0

    selected = own_load(out / 'best.pt')
    final_meta = json.loads(metadata_path.read_text())
    final_meta.update({'status': 'smoke_completed' if args.smoke else 'completed',
        'finished_utc': now(), 'selected_stage': selected['best_stage'],
        'selected_epoch': selected['best_epoch'], 'best_validation_macro_f1': best_f1,
        'epochs_completed_head_only': sum(h['stage'] == 'head_only' for h in history),
        'epochs_completed_finetune': sum(h['stage'] == 'full_fine_tune' for h in history),
        'total_train_seconds_excluding_validation': total_seconds,
        'peak_gpu_memory_mb': torch.cuda.max_memory_allocated(device) / 1e6,
        'best_checkpoint': str(out / 'best.pt')})
    save_json(metadata_path, final_meta)
    print('DONE:', out, 'selected', selected['best_stage'], selected['best_epoch'], flush=True)


def test(args):
    if not args.acknowledge_posthoc_test:
        raise ValueError('Prior v2 test scores inspected; pass --acknowledge-posthoc-test')
    cfg, recipe, exp, ds_key, model_key, audit, device = load_protocol(args)
    out = output_path(args)
    metadata = json.loads((out / 'metadata.json').read_text())
    if metadata.get('status') != 'completed' or metadata.get('kind') != 'exploratory_posthoc_full_backbone':
        raise ValueError('Only complete v3 runs may be evaluated on test')
    if (out / 'test_metrics.json').exists():
        raise FileExistsError('v3 post-hoc test already performed; do not tune on test.')
    if metadata['recipe_sha256'] != sha256(V3_RECIPE) or metadata['manifest_sha256'] != audit['manifest_sha256']:
        raise ValueError('Training/evaluation recipe or manifest mismatch')
    model, _, val_tf = initialize_model(cfg, ds_key, model_key, device, pretrained=False)
    selected = own_load(out / 'best.pt')
    if selected['run_id'] != args.run_id or selected['manifest_sha256'] != audit['manifest_sha256']:
        raise ValueError('v3 checkpoint identity mismatch')
    model.load_state_dict(selected['model_state'], strict=True)
    ds = VehicleManifestDataset(ROOT, ds_key, 'test', transform=val_tf)
    metrics, truth, pred = evaluate(model, make_loader(ds, cfg), len(ds.classes),
                                   nn.CrossEntropyLoss(), device, return_predictions=True)
    save_json(out / 'test_metrics.json', {'experiment': args.experiment, 'run_id': args.run_id,
        'evaluated_utc': now(), 'test_metrics': metrics,
        'selected_val_metrics': selected['val_metrics'],
        'evaluation_status': 'exploratory_posthoc_original_test_already_inspected'})
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, pred, labels=list(range(len(ds.classes))), zero_division=0)
    with (out / 'per_class_test.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['class_id', 'class_name', 'precision', 'recall', 'f1', 'support'])
        writer.writerows((i, ds.classes[i], float(precision[i]), float(recall[i]),
                          float(f1[i]), int(support[i])) for i in range(len(ds.classes)))
    np.save(out / 'test_confusion_matrix.npy', confusion_matrix(
        truth, pred, labels=list(range(len(ds.classes)))))
    print('EXPLORATORY POST-HOC TEST:', json.dumps(metrics, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True, choices=['E1', 'E2', 'E3', 'E4'])
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--mode', choices=['train', 'memory', 'test'], default='train')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--acknowledge-posthoc-test', action='store_true')
    args = parser.parse_args()
    if (args.smoke or args.resume) and args.mode != 'train':
        parser.error('--smoke and --resume only apply to training')
    if args.smoke and args.resume:
        parser.error('Smoke cannot resume')
    # Check run ID / output root for ALL commands before touching data.
    output_path(args)
    if args.mode == 'memory':
        memory_probe(args)
    elif args.mode == 'test':
        test(args)
    else:
        train(args)


if __name__ == '__main__':
    main()

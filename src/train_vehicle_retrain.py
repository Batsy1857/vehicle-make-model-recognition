"""Independent ImageNet-initialized two-stage PyTorch training, optional post-hoc testing.

Uses the original dataset, transform, and freezing modules. DOES NOT load old
Notebook 9/10 model checkpoints. Keeps old E1-E4 results untouched.

Examples (run from project root):
 python src/train_vehicle_retrain.py --experiment E3 --run-id E3-v2-smoke --output-dir /content/drive/MyDrive/vehicle_project/revised_runs --smoke
 python src/train_vehicle_retrain.py --experiment E3 --run-id E3-v2-run01 --output-dir /content/drive/MyDrive/vehicle_project/revised_runs
 python src/train_vehicle_retrain.py --experiment E3 --run-id E3-v2-run01 --output-dir /content/drive/MyDrive/vehicle_project/revised_runs --resume
 python src/train_vehicle_retrain.py --experiment E3 --run-id E3-v2-run01 --output-dir /content/drive/MyDrive/vehicle_project/revised_runs --mode test --acknowledge-posthoc-test
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import timm
import torch
import torchvision
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader
from vehicle_data import DATASET_SPECS, VehicleManifestDataset, file_sha256, inspect_manifest
from vehicle_models import configure_transfer_stage, keep_batchnorm_in_eval
from vehicle_transforms import build_vehicle_transforms
from timm.data import resolve_model_data_config

ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = ROOT / 'configs/experiments.yaml'
RECIPE_FILE = ROOT / 'configs/retraining.yaml'
STAGES = ('head_only', 'partial_fine_tune')
COLUMNS = ['stage', 'epoch', 'learning_rate', 'train_loss', 'val_loss', 'val_accuracy',
           'val_macro_f1', 'val_top5_accuracy', 'train_images_seen', 'val_images_seen',
           'train_seconds', 'val_seconds', 'peak_gpu_memory_mb', 'utc_end']


def sha256(p):
    d = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            d.update(chunk)
    return d.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, default=str) + '\n')
    os.replace(tmp, path)


def save_torch(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(data, tmp)
    os.replace(tmp, path)


def own_load(path):
    # Only load checkpoints created by THIS repository, from your own Drive.
    return torch.load(path, map_location='cpu', weights_only=True)


def git_sha():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def seed_everything(seed):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def init_worker(_worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(dataset, cfg, *, train=False, stage_index=0, epoch=0):
    gen = torch.Generator().manual_seed(int(cfg['random_seed']) + stage_index*10000 + epoch)
    return DataLoader(dataset,
        batch_size=int(cfg['training']['batch_size'] if train else cfg['training']['eval_batch_size']),
        shuffle=train, num_workers=int(cfg['training']['num_workers']),
        generator=gen, worker_init_fn=init_worker, pin_memory=True,
        drop_last=False)


def epoch_lr(epoch_zero_based, max_epochs, peak, minimum, warmup_epochs, start_frac):
    if not (1 <= warmup_epochs < max_epochs and 0 < minimum < peak and 0 < start_frac <= 1):
        raise ValueError('Invalid warmup/learning-rate settings')
    if not 0 <= epoch_zero_based < max_epochs:
        raise ValueError('Epoch out of range')
    if epoch_zero_based < warmup_epochs:
        fraction = (epoch_zero_based + 1) / warmup_epochs
        return peak * (start_frac + (1-start_frac)*fraction)
    fraction = (epoch_zero_based - warmup_epochs + 1) / (max_epochs-warmup_epochs)
    return minimum + 0.5*(peak-minimum)*(1+math.cos(math.pi*fraction))


def train_one_epoch(model, ld, opt, scaler, loss_fn, device, max_batches=None):
    model.train()
    keep_batchnorm_in_eval(model)
    total, sum_loss = 0, 0.0
    torch.cuda.synchronize()
    started = time.perf_counter()
    for i, (x, y) in enumerate(ld):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            logits = model(x)
            loss = loss_fn(logits, y)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f'Non-finite loss at batch {i}')
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        n = int(y.shape[0])
        total += n
        sum_loss += float(loss.item())*n
    torch.cuda.synchronize()
    if not total:
        raise ValueError('No training batches processed')
    return {'train_loss': sum_loss/total, 'train_images_seen': total,
            'train_seconds': time.perf_counter()-started}


def evaluate(model, ld, num_classes, loss_fn, device, max_batches=None, return_predictions=False):
    model.eval()
    truth, predictions = [], []
    n_seen, correct5, total_loss = 0, 0, 0.0
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        for i, (x,y) in enumerate(ld):
            if max_batches is not None and i >= max_batches:
                break
            x,y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.float16):
                logits = model(x)
                loss = loss_fn(logits,y)
            pred = logits.argmax(dim=1)
            correct5 += (logits.topk(k=min(5, num_classes),dim=1).indices==y.unsqueeze(1)).any(dim=1).sum().item()
            batch = int(y.shape[0])
            n_seen += batch
            total_loss += float(loss.item())*batch
            truth.extend(y.cpu().tolist())
            predictions.extend(pred.cpu().tolist())
    torch.cuda.synchronize()
    if n_seen == 0:
        raise ValueError('No evaluation batches processed')
    results = {'loss':total_loss/n_seen, 'accuracy':float(accuracy_score(truth,predictions)),
               'macro_f1':float(f1_score(truth,predictions,labels=list(range(num_classes)),average='macro',zero_division=0)),
               'top5_accuracy':float(correct5/n_seen), 'n':n_seen,
               'seconds':time.perf_counter()-started}
    return (results, truth, predictions) if return_predictions else results


def save_history(path, rows):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows([{c: row.get(c,'') for c in COLUMNS} for row in rows])
    os.replace(tmp, path)


def load_protocol(args):
    with CONFIG_FILE.open() as f:
        cfg = yaml.safe_load(f)
    with RECIPE_FILE.open() as f:
        recipe = yaml.safe_load(f)
    if recipe['image_pipeline'] != 'reuse_configs_experiments_yaml' or recipe['freeze'] != 'reuse_src_vehicle_models_py':
        raise ValueError('The recipe must reuse frozen shared preprocessing/model freezing')
    if recipe['checkpoint_selection'] != 'validation_macro_f1' or recipe['init'] != 'timm_imagenet_pretrained_new_classifier':
        raise ValueError('Unexpected model initialization/selection')
    if int(recipe['seed']) != int(cfg['random_seed']):
        raise ValueError('Seed mismatch')
    if int(recipe['weight_decay']*10000) != int(cfg['training']['weight_decay']*10000):
        raise ValueError('Weight decay mismatch')
    if recipe['optimizer'] != 'AdamW' or cfg['training']['optimizer'] != 'AdamW':
        raise ValueError('Unexpected optimizer')
    if cfg['training']['gradient_accumulation_steps'] != 1:
        raise ValueError('Gradient accumulation unsupported')
    exp = cfg['experiments'][args.experiment]
    ds_key, model_key = exp['dataset'], exp['model']
    audit = inspect_manifest(ROOT, ds_key, check_files=True)
    if audit['manifest_sha256'] != cfg['datasets'][ds_key]['manifest_sha256']:
        raise ValueError('Frozen manifest SHA256 changed')
    for sp,n in cfg['datasets'][ds_key]['verified_split_counts'].items():
        observed = 'excluded_test_overlap' if ds_key=='pakistan' and sp=='excluded' else sp
        if audit['counts'].get(observed,0) != int(n):
            raise ValueError(f'Unexpected {ds_key}/{sp} count')
    if not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime in Colab')
    return cfg,recipe,exp,ds_key,model_key,audit,torch.device('cuda')


def initialize_model(cfg, ds_key, model_key, device, *, pretrained):
    name = cfg['models'][model_key]['timm_name']
    nclasses=int(cfg['datasets'][ds_key]['verified_classes'])
    model=timm.create_model(name,pretrained=pretrained,num_classes=nclasses)
    model.to(device)
    train_tf, val_tf = build_vehicle_transforms(cfg,model)
    return model, train_tf, val_tf


def do_train(args):
    if args.resume and args.smoke:
        raise ValueError('Smoke cannot resume')
    cfg, recipe, exp, ds_key, model_key, audit, device = load_protocol(args)
    output = Path(args.output_dir).expanduser().resolve()/args.run_id
    if args.resume:
        if not (output/'last.pt').is_file():
            raise FileNotFoundError('Cannot resume without last.pt')
        original_meta = json.loads((output/'metadata.json').read_text())
        if original_meta['status']=='completed' or original_meta['status']=='smoke_completed':
            raise ValueError('Run already completed; use a fresh unique run ID')
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output already exists: {output}. Choose unique run ID.')
    seed_everything(int(recipe['seed']))
    model, train_tf, val_tf=initialize_model(cfg,ds_key,model_key,device,pretrained=not args.resume)
    train_ds=VehicleManifestDataset(ROOT,ds_key,'train',transform=train_tf)
    val_ds=VehicleManifestDataset(ROOT,ds_key,'val',transform=val_tf)
    model_id=cfg['models'][model_key]['timm_name']
    head_stats = configure_transfer_stage(model,model_key,'head_only')
    fine_stats = configure_transfer_stage(model,model_key,'partial_fine_tune')
    head_stats=configure_transfer_stage(model,model_key,'head_only')
    norm=resolve_model_data_config(model)
    maximums=[1,1] if args.smoke else [int(recipe['stages']['head_only']['epochs']),int(recipe['stages']['partial_fine_tune']['epochs'][ds_key])]
    patience=int(recipe['stages']['partial_fine_tune']['early_stopping_patience'])
    head_lr=float(recipe['stages']['head_only']['learning_rate'])
    ft=recipe['stages']['partial_fine_tune']
    output.mkdir(parents=True,exist_ok=True)
    meta_path=output/'metadata.json'
    meta={'created_utc':now(), 'run_id':args.run_id, 'experiment':args.experiment,
          'experiment_definition':exp, 'status':'running','kind':'smoke' if args.smoke else 'revised_posthoc_training',
          'git_commit':git_sha(), 'recipe_sha256':sha256(RECIPE_FILE),'original_config_sha256':sha256(CONFIG_FILE),
          'manifest_sha256':audit['manifest_sha256'], 'label_map_sha256':sha256(ROOT/DATASET_SPECS[ds_key]['label_map']),
          'dataset_audit':audit,'model_timm_name':model_id, 'pretrained':True,
          'source':'independent_timm_ImageNet_initialization_NOT_previous_run_checkpoint',
          'training_recipe':recipe,'shared_training_config':cfg['training'],
          'input_shape':[3,*cfg['data_pipeline']['input_size']],
          'pretrained_normalization_mean':norm['mean'],'pretrained_normalization_std':norm['std'],
          'parameters_head_only':head_stats,'parameters_partial_fine_tune':fine_stats,
          'gpu':torch.cuda.get_device_name(0),'cuda_build':torch.version.cuda,
          'python':platform.python_version(),'torch':torch.__version__,
          'torchvision':torchvision.__version__,'timm':timm.__version__,
          'seed':recipe['seed'], 'validation_selection':'max_macro_f1',
          'posthoc_test_caveat':recipe['posthoc_test_caveat'],
          'no_test_data_evaluation_during_training':True}
    if args.resume:
        old = json.loads(meta_path.read_text())
        if any(old.get(k)!=meta[k] for k in ['experiment','git_commit','recipe_sha256','original_config_sha256','manifest_sha256','label_map_sha256']):
            raise ValueError('Config/commit/dataset changed: unsafe to resume')
    else:
        save_json(meta_path,meta)
    loss_fn=nn.CrossEntropyLoss()
    torch.cuda.reset_peak_memory_stats()
    next_stage, next_epoch=0,0
    best_f1, best_head, best_fine = -1.,-1.,-1.
    best_stage, best_epoch=None,None
    nonimprove=0
    total_seconds=0.
    history=[]
    state=None
    if args.resume:
        state=own_load(output/'last.pt')
        if state['run_id']!=args.run_id or state['git_commit']!=git_sha() or state['recipe_sha256']!=sha256(RECIPE_FILE) or state['manifest_sha256']!=audit['manifest_sha256']:
            raise ValueError('Unsafe resume: checkpoint incompatible')
        model.load_state_dict(state['model_state'],strict=True)
        torch.set_rng_state(state['torch_rng_state'])
        torch.cuda.set_rng_state_all(state['cuda_rng_states'])
        next_stage,next_epoch=int(state['next_stage']),int(state['next_epoch'])
        best_f1,best_head,best_fine=float(state['best_f1']),float(state['best_head']),float(state['best_fine'])
        best_stage,best_epoch=state['best_stage'],state['best_epoch']
        nonimprove=int(state['nonimprove'])
        total_seconds=float(state['total_seconds'])
        history=state['history_rows']
        save_history(output/'history.csv',history)
        print('Resuming',args.run_id,'at stage',next_stage,'epoch',next_epoch+1,flush=True)
    for stage_index in range(next_stage,2):
        stage=STAGES[stage_index]
        start=next_epoch if stage_index==next_stage else 0
        if stage_index==1 and start==0:
            # The best completed Stage-1 head, not the last Stage-1 weights.
            head_ckpt=own_load(output/'best_head.pt')
            model.load_state_dict(head_ckpt['model_state'],strict=True)
        stats=configure_transfer_stage(model,model_key,stage)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                    lr=head_lr if stage_index==0 else float(ft['peak_learning_rate']),
                                    weight_decay=float(recipe['weight_decay']))
        scaler=torch.amp.GradScaler('cuda',enabled=True)
        if args.resume and state is not None and stage_index==next_stage and start>0:
            optimizer.load_state_dict(state['optimizer_state'])
            scaler.load_state_dict(state['scaler_state'])
        print(f'{args.experiment} {stage}: {stats["trainable"]:,} trainable parameters',flush=True)
        for e in range(start,maximums[stage_index]):
            if stage_index==0 or args.smoke:
                lr=head_lr if stage_index==0 else float(ft['peak_learning_rate'])
            else:
                lr=epoch_lr(e,maximums[stage_index],float(ft['peak_learning_rate']),
                         float(ft['minimum_learning_rate']),int(ft['warmup_epochs']),
                         float(ft['warmup_start_fraction']))
            for group in optimizer.param_groups:
                group['lr']=lr
            tr=train_one_epoch(model,make_loader(train_ds,cfg,train=True,stage_index=stage_index,epoch=e),
                               optimizer,scaler,loss_fn,device,max_batches=2 if args.smoke else None)
            val=evaluate(model,make_loader(val_ds,cfg,stage_index=stage_index,epoch=e),
                         len(val_ds.classes),loss_fn,device,max_batches=2 if args.smoke else None)
            total_seconds+=tr['train_seconds']
            current=float(val['macro_f1'])
            local_best=best_head if stage_index==0 else best_fine
            improved_local=current>local_best
            if stage_index==0:
                if improved_local: best_head=current
            else:
                if improved_local: best_fine=current
                if improved_local: nonimprove=0
                elif e+1>int(ft['warmup_epochs']): nonimprove+=1
            if current>best_f1:
                best_f1=current
                best_stage,best_epoch=stage,e+1
            row={'stage':stage,'epoch':e+1,'learning_rate':lr,**tr,'val_loss':val['loss'],
                 'val_accuracy':val['accuracy'],'val_macro_f1':val['macro_f1'],
                 'val_top5_accuracy':val['top5_accuracy'],'val_images_seen':val['n'],
                 'val_seconds':val['seconds'],
                 'peak_gpu_memory_mb':torch.cuda.max_memory_allocated()/1e6,'utc_end':now()}
            history.append(row)
            ckpt={'experiment':args.experiment,'run_id':args.run_id,
                  'manifest_sha256':audit['manifest_sha256'], 'git_commit':git_sha(),
                  'recipe_sha256':sha256(RECIPE_FILE),'model_state':model.state_dict(),
                  'best_stage':stage,'best_epoch':e+1,'val_metrics':val,'parameter_stats':stats}
            if stage_index==0 and improved_local:
                save_torch(output/'best_head.pt',ckpt)
            if stage_index==1 and improved_local:
                save_torch(output/'best_finetune.pt',ckpt)
            if best_stage==stage and best_epoch==e+1 and current==best_f1:
                save_torch(output/'best.pt',ckpt)
            stop=(stage_index==1 and not args.smoke and nonimprove>=patience)
            advance=(e+1==maximums[stage_index]) or stop
            following_stage=stage_index+1 if advance else stage_index
            following_epoch=0 if advance else e+1
            save_torch(output/'last.pt',{
                **ckpt,'next_stage':following_stage,'next_epoch':following_epoch,
                'best_f1':best_f1,'best_head':best_head,'best_fine':best_fine,
                'best_stage':best_stage,'best_epoch':best_epoch,
                'nonimprove':nonimprove if stage_index==1 else 0,
                'total_seconds':total_seconds,'optimizer_state':optimizer.state_dict(),
                'scaler_state':scaler.state_dict(),'torch_rng_state':torch.get_rng_state(),
                'cuda_rng_states':torch.cuda.get_rng_state_all(),
                'history_rows':history})
            save_history(output/'history.csv',history)
            print(f' {stage} e{e+1:02d}/{maximums[stage_index]} LR={lr:.2e} '
                  f'loss={tr["train_loss"]:.4f} val_acc={val["accuracy"]:.4f} '
                  f'val_F1={current:.4f} best_F1={best_f1:.4f} '
                  f'train_s={tr["train_seconds"]:.1f}',flush=True)
            if stop:
                print(f' Early stopping: {patience} non-improving fine-tune epochs',flush=True)
                break
        nonimprove=0
    selected=own_load(output/'best.pt')
    final_meta=json.loads(meta_path.read_text())
    final_meta.update({'status':'smoke_completed' if args.smoke else 'completed',
         'finished_utc':now(), 'selected_stage':selected['best_stage'],
         'selected_epoch':selected['best_epoch'],
         'best_validation_macro_f1':best_f1,
         'epochs_completed_head_only':sum(x['stage']=='head_only' for x in history),
         'epochs_completed_finetune':sum(x['stage']=='partial_fine_tune' for x in history),
         'total_train_seconds_excluding_validation':total_seconds,
         'peak_gpu_memory_mb':torch.cuda.max_memory_allocated()/1e6,
         'best_checkpoint':str(output/'best.pt')})
    save_json(meta_path,final_meta)
    print('DONE:',output,'selected',selected['best_stage'],selected['best_epoch'],flush=True)


def do_test(args):
    if not args.acknowledge_posthoc_test:
        raise ValueError('The existing held-out test scores were inspected; pass --acknowledge-posthoc-test to explicitly label this evaluation as post-hoc')
    cfg,recipe,exp,ds_key,model_key,audit,device=load_protocol(args)
    output=Path(args.output_dir).expanduser().resolve()/args.run_id
    meta=json.loads((output/'metadata.json').read_text())
    if meta.get('status')!='completed' or meta.get('kind')!='revised_posthoc_training':
        raise ValueError('Only completed revised full training can be evaluated')
    if (output/'test_metrics.json').exists():
        raise FileExistsError('Already tested. Do not repeatedly test to optimize the model.')
    if meta['manifest_sha256']!=audit['manifest_sha256'] or meta['recipe_sha256']!=sha256(RECIPE_FILE):
        raise ValueError('Test mismatch with training run')
    model,_,val_tf=initialize_model(cfg,ds_key,model_key,device,pretrained=False)
    ck=own_load(output/'best.pt')
    if ck['run_id']!=args.run_id or ck['manifest_sha256']!=audit['manifest_sha256']:
        raise ValueError('Best checkpoint does not match this run')
    model.load_state_dict(ck['model_state'],strict=True)
    dataset=VehicleManifestDataset(ROOT,ds_key,'test',transform=val_tf)
    metrics, truth, prediction=evaluate(model,make_loader(dataset,cfg),len(dataset.classes),nn.CrossEntropyLoss(),device,return_predictions=True)
    save_json(output/'test_metrics.json',{'experiment':args.experiment,'run_id':args.run_id,
           'evaluated_utc':now(),'test_metrics':metrics,'selected_checkpoint':str(output/'best.pt'),
           'selected_val_metrics':ck['val_metrics'],'evaluation_status':'posthoc_not_independent_because_prior_test_results_were_viewed'})
    p,r,f,s=precision_recall_fscore_support(truth,prediction,labels=list(range(len(dataset.classes))),zero_division=0)
    with (output/'per_class_test.csv').open('w',newline='',encoding='utf-8') as o:
        w=csv.writer(o);w.writerow(['class_id','class_name','precision','recall','f1','support'])
        w.writerows((i,dataset.classes[i],float(p[i]),float(r[i]),float(f[i]),int(s[i])) for i in range(len(dataset.classes)))
    np.save(output/'test_confusion_matrix.npy',confusion_matrix(truth,prediction,labels=list(range(len(dataset.classes)))))
    print('POST-HOC TEST:',json.dumps(metrics,indent=2),flush=True)


def do_benchmark(args):
    cfg,recipe,exp,ds_key,model_key,audit,device=load_protocol(args)
    output=Path(args.output_dir).expanduser().resolve()/args.run_id
    meta=json.loads((output/'metadata.json').read_text())
    if meta.get('status')!='completed' or meta['manifest_sha256']!=audit['manifest_sha256']:
        raise ValueError('Benchmark requires a completed compatible training run')
    model,_,_=initialize_model(cfg,ds_key,model_key,device,pretrained=False)
    ck=own_load(output/'best.pt')
    model.load_state_dict(ck['model_state'],strict=True)
    model.eval()
    settings=cfg['evaluation']['runtime_benchmark']
    results={'experiment':args.experiment,'run_id':args.run_id,'gpu':torch.cuda.get_device_name(0),
             'inference_mode':True,'precision':'fp32','preprocessing_included':False,'data_loader_included':False,
             'warmup_iterations':settings['warmup_iterations'],'timed_iterations':settings['timed_iterations']}
    with torch.inference_mode():
        for batch in (1,16):
            x=torch.randn(batch,3,*cfg['data_pipeline']['input_size'],device=device)
            for _ in range(int(settings['warmup_iterations'])):
                model(x)
            timings=[]
            for _ in range(int(settings['timed_iterations'])):
                torch.cuda.synchronize()
                start=time.perf_counter()
                model(x)
                torch.cuda.synchronize()
                timings.append((time.perf_counter()-start)*1000/batch)
            results[f'batch{batch}_median_ms_per_image']=float(np.median(timings))
            results[f'batch{batch}_p95_ms_per_image']=float(np.percentile(timings,95))
    save_json(output/'benchmark.json',results)
    print('BENCHMARK:',json.dumps(results,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',required=True,choices=['E1','E2','E3','E4'])
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--mode',choices=['train','test','benchmark'],default='train')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--acknowledge-posthoc-test',action='store_true')
    args=parser.parse_args()
    if (args.smoke or args.resume) and args.mode!='train':
        parser.error('--smoke/--resume are training-only flags')
    if args.smoke and args.resume:
        parser.error('--smoke incompatible with --resume')
    if args.mode=='train':do_train(args)
    elif args.mode=='test':do_test(args)
    else:do_benchmark(args)

if __name__=='__main__':
    main()

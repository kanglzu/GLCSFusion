


import os
import json
import time
import random
import datetime
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader


os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128')
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False






MODALITY_CFG = {
    
    'CT-MRI':    dict(lr=3e-4, batch_size=4, epochs=300,
                      accum_steps=2, grad_clip=1.0,
                      patience=80, warmup_epochs=20,
                      tb_interval=10, ckpt_interval=50),
    
    'PET-MRI':   dict(lr=1e-4, batch_size=8, epochs=400,
                      accum_steps=1, grad_clip=1.0,
                      patience=50, warmup_epochs=8,
                      tb_interval=20, ckpt_interval=50, metric_interval=10),
    'SPECT-MRI': dict(lr=1e-4, batch_size=4, epochs=250,
                      accum_steps=2, grad_clip=0.5,
                      patience=60, warmup_epochs=20,
                      tb_interval=10, ckpt_interval=50),
}





def RGB2YCrCb(x: torch.Tensor) -> torch.Tensor:
    flat = x.permute(0, 2, 3, 1).reshape(-1, 3)
    R, G, B = flat[:, 0], flat[:, 1], flat[:, 2]
    Y  =  0.299  * R + 0.587  * G + 0.114  * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    ycrcb = torch.stack([Y, Cr, Cb], dim=1)
    B_, H, W = x.shape[0], x.shape[2], x.shape[3]
    return ycrcb.reshape(B_, H, W, 3).permute(0, 3, 1, 2)


def YCrCb2RGB(x: torch.Tensor) -> torch.Tensor:
    device = x.device
    flat = x.permute(0, 2, 3, 1).reshape(-1, 3)
    mat  = torch.tensor([[1., 1., 1.],
                         [1.403, -0.714, 0.],
                         [0., -0.344, 1.773]], device=device)
    bias = torch.tensor([0., -0.5, -0.5], device=device)
    rgb  = (flat + bias) @ mat
    B_, H, W = x.shape[0], x.shape[2], x.shape[3]
    return rgb.reshape(B_, H, W, 3).permute(0, 3, 1, 2)






def apply_warmup_lr(optimizer, epoch: int, warmup_epochs: int, base_lr: float):
    lr = base_lr * (epoch + 1) / warmup_epochs
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr






@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    totals = [0., 0., 0., 0.]

    for image_vis, image_ir in loader:
        image_vis = image_vis.to(device)
        image_ir  = image_ir.to(device)

        fusion = model(image_vis, image_ir)
        fusion = fusion.clamp(0., 1.)

        loss_total, loss_in, loss_ssim, loss_grad = criterion(
            image_vis=image_vis, image_ir=image_ir,
            generate_img=fusion, i=0, labels=None
        )
        totals[0] += loss_total.item()
        totals[1] += loss_in.item()
        totals[2] += loss_grad.item()
        totals[3] += loss_ssim.item()

    n = max(len(loader), 1)
    model.train()
    return tuple(t / n for t in totals)







@torch.no_grad()
def validate_fusion_metrics(model, loader, device):
    from evaluator import Evaluator, VIFF

    model.eval()
    rows = []
    for image_vis, image_ir in loader:
        image_vis = image_vis.to(device)
        image_ir = image_ir.to(device)
        fusion = model(image_vis, image_ir).clamp(0., 1.)

        f = fusion.squeeze().detach().cpu().numpy().astype(np.float32)
        a = image_vis.squeeze().detach().cpu().numpy().astype(np.float32)
        b = image_ir.squeeze().detach().cpu().numpy().astype(np.float32)

        f_min, f_max = float(f.min()), float(f.max())
        if f_max - f_min > 1e-6:
            f = (f - f_min) / (f_max - f_min)

        f8 = np.uint8(np.round(np.clip(f * 255.0, 0, 255)))
        a8 = np.uint8(np.round(np.clip(a * 255.0, 0, 255)))
        b8 = np.uint8(np.round(np.clip(b * 255.0, 0, 255)))

        rows.append({
            'EN': Evaluator.EN(f8),
            'SD': Evaluator.SD(f8),
            'SF': Evaluator.SF(f8),
            'AG': Evaluator.AG(f8),
            'SCD': Evaluator.SCD(f, a, b),
            'VIFF': VIFF(f, a, b),
            'Qabf': Evaluator.Qabf(f, a, b),
            'SSIM': Evaluator.SSIM(f8, a8, b8),
        })

    if not rows:
        model.train()
        return None

    mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    
    
    
    
    normalized = (
        min(max(mean['EN'] / 8.0, 0.0), 1.0),
        min(max(mean['SD'] / 128.0, 0.0), 1.0),
        min(max(mean['SF'] / 40.0, 0.0), 1.0),
        min(max(mean['AG'] / 10.0, 0.0), 1.0),
        min(max(mean['SCD'] / 2.0, 0.0), 1.0),
        min(max(mean['VIFF'], 0.0), 1.0),
        min(max(mean['Qabf'], 0.0), 1.0),
        min(max(mean['SSIM'] / 2.0, 0.0), 1.0),
    )
    mean['score'] = float(np.mean(normalized))
    model.train()
    return mean


def train_fusion(modality: str = 'CT-MRI', run_id: int = 0):
    assert modality in MODALITY_CFG,\
        f"不支持的模态 '{modality}'，可选：{list(MODALITY_CFG.keys())}"

    run_seed = int(getattr(args, 'seed', SEED))
    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    torch.cuda.manual_seed_all(run_seed)
    cfg = dict(MODALITY_CFG[modality])
    structure_choice = getattr(args, 'structure_ablation', 'none')
    
    is_ablation = (structure_choice not in ('none', 'full') or
                   getattr(args, 'loss_ablation', 'none') != 'none')
    epochs_override = getattr(args, 'epochs_override', None)
    if epochs_override is not None:
        cfg['epochs'] = int(epochs_override)
        cfg['patience'] = min(cfg['patience'], max(20, int(epochs_override) // 2))
        cfg['ckpt_interval'] = min(cfg['ckpt_interval'], max(10, int(epochs_override) // 2))
    lr_override = getattr(args, 'lr_override', None)
    if lr_override is not None:
        cfg['lr'] = float(lr_override)
    batch_size_override = getattr(args, 'batch_size_override', None)
    if batch_size_override is not None:
        cfg['batch_size'] = max(1, int(batch_size_override))
    accum_steps_override = getattr(args, 'accum_steps_override', None)
    if accum_steps_override is not None:
        cfg['accum_steps'] = max(1, int(accum_steps_override))
    ckpt_interval_override = getattr(args, 'ckpt_interval_override', None)
    if ckpt_interval_override is not None:
        cfg['ckpt_interval'] = max(1, int(ckpt_interval_override))
    if is_ablation:
        
        
        cfg['patience'] = cfg['epochs'] + 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    timestamp  = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name   = f'{modality}_{timestamp}'
    structure_tag = getattr(args, 'structure_ablation', 'none')
    loss_tag = getattr(args, 'loss_ablation', 'none')
    experiment_id = getattr(args, 'experiment_id', None)
    if experiment_id:
        ckpt_dir = os.path.join('model_last', 'ablations_v2', modality,
                                experiment_id, f'seed{run_seed}')
    elif structure_tag not in ('none', 'full') or loss_tag != 'none':
        experiment_tag = f'struct-{structure_tag}_loss-{loss_tag}'
        ckpt_dir = os.path.join('model_last', 'ablations', modality, experiment_tag,
                                f'run{run_id:02d}')
    else:
        ckpt_dir = os.path.join('model_last', f'run_{modality}', f'run{run_id:02d}')
    log_dir    = os.path.join('runs', run_name)
    split_dir  = os.path.join('outputs', 'splits')
    os.makedirs(ckpt_dir,  exist_ok=True)
    os.makedirs(log_dir,   exist_ok=True)
    os.makedirs(split_dir, exist_ok=True)

    config_path = os.path.join(ckpt_dir, 'experiment_config.json')
    with open(config_path, 'w', encoding='utf-8') as config_file:
        json.dump(vars(args), config_file, indent=2, sort_keys=True)

    from torch.utils.tensorboard import SummaryWriter
    writer   = SummaryWriter(log_dir)
    log_path = os.path.join(ckpt_dir, f'train_{timestamp}.log')
    log_f    = open(log_path, 'w', buffering=1, encoding='utf-8')

    def log(msg: str):
        print(msg)
        log_f.write(msg + '\n')

    log('=' * 70)
    log(f'[训练启动] 模态={modality}  run_id={run_id}')
    log(f'超参数: {cfg}')
    log(f'TensorBoard: {log_dir}')
    log('=' * 70)

    
    from model_factory import build_model
    from loss import Fusionloss
    from TaskFusion_dataset import Fusion_dataset
    model = build_model(modality, vars(args)).to(device)
    resume_model = getattr(args, 'resume_model', None)
    if resume_model:
        resume_path = resume_model
        if not os.path.isabs(resume_path):
            resume_path = os.path.join(os.getcwd(), resume_path)
        state = torch.load(resume_path, map_location=device)
        if isinstance(state, dict) and 'model' in state:
            state = state['model']
        model_state = model.state_dict()
        skipped_shape = []
        filtered_state = {}
        for k, v in state.items():
            if k in model_state and hasattr(v, 'shape') and v.shape != model_state[k].shape:
                skipped_shape.append((k, tuple(v.shape), tuple(model_state[k].shape)))
                continue
            filtered_state[k] = v
        incompatible = model.load_state_dict(filtered_state, strict=False)
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        if missing:
            log(f'宽松加载: missing_keys={missing}')
        if unexpected:
            log(f'宽松加载: unexpected_keys={unexpected}')
        if skipped_shape:
            log(f'宽松加载: skipped_shape={skipped_shape}')
        log(f'从权重继续训练: {resume_path}')

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f'模型参数量：{total_params / 1e6:.2f} M')

    
    dataset = Fusion_dataset('train', ir_path=args.ir_dir, vi_path=args.vis_dir, length=30000, modality=modality,
                             aug_level=getattr(args, 'aug_level', 'basic'))
    
    val_dataset = Fusion_dataset('train', ir_path=args.ir_dir, vi_path=args.vis_dir, length=30000, modality=modality,
                                 use_augmentation=False,
                                 aug_level=getattr(args, 'aug_level', 'basic'))
    split_path = os.path.join(split_dir, f'{modality}_split.json')

    
    if os.path.exists(split_path):
        with open(split_path) as f:
            sp = json.load(f)
        train_idx, val_idx = sp['train_indices'], sp['val_indices']
        total = len(dataset.filepath_vis)
        if not (train_idx and val_idx
                and max(train_idx) < total and max(val_idx) < total):
            log('缓存的 split 与当前数据规模不匹配，重新生成...')
            
            ratio = 0.95 if modality == 'CT-MRI' else 0.9
            train_idx, val_idx = dataset.split_train_val(train_ratio=ratio)
            with open(split_path, 'w') as f:
                json.dump({'train_indices': train_idx, 'val_indices': val_idx}, f)
    else:
        
        ratio = 0.95 if modality == 'CT-MRI' else 0.9
        train_idx, val_idx = dataset.split_train_val(train_ratio=ratio)
        with open(split_path, 'w') as f:
            json.dump({'train_indices': train_idx, 'val_indices': val_idx}, f)

    log(f'训练集：{len(train_idx)} 张  验证集：{len(val_idx)} 张')

    def seed_worker(wid):
        np.random.seed(run_seed + wid)
        random.seed(run_seed + wid)

    g = torch.Generator()
    g.manual_seed(run_seed)

    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_idx),
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(val_dataset, val_idx),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )

    
    loss_ablation = getattr(args, 'loss_ablation', 'none')
    loss_profile = ('abl_' + loss_ablation if loss_ablation != 'none'
                    else getattr(args, 'loss_profile', 'default'))
    criterion = Fusionloss(
        modality=modality,
        profile=loss_profile,
        stat_weight=float(getattr(args, 'stat_weight', 8.0)),
    ).to(device)

    optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg['lr'],
    weight_decay=1e-4,   
    betas=(0.9, 0.999),
)

    
    
    T_0 = 20 if modality == 'CT-MRI' else 10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=2, eta_min=1e-6
    )
    log(f'优化器：AdamW(lr={cfg["lr"]:.2e})  调度器：CosineAnnealingWarmRestarts')

    
    
    precision = getattr(args, 'precision', 'fp32')
    amp_enabled = (device.type == 'cuda' and precision != 'fp32')
    amp_dtype = torch.bfloat16 if precision == 'bf16' else torch.float16
    scaler = torch.cuda.amp.GradScaler(
        enabled=(device.type == 'cuda' and precision == 'fp16'))
    log(f'Precision={precision}, physical_batch={cfg["batch_size"]}, '
        f'accum_steps={cfg["accum_steps"]}')

    
    best_val_loss   = float('inf')
    best_state      = None
    best_metric_score = float(getattr(args, 'initial_best_metric_score', None)
                              or -float('inf'))
    best_metric_state = None
    initial_metric_model = getattr(args, 'initial_best_metric_model', None)
    if initial_metric_model:
        initial_metric_path = (initial_metric_model if os.path.isabs(initial_metric_model)
                               else os.path.join(os.getcwd(), initial_metric_model))
        initial_metric_state = torch.load(initial_metric_path, map_location='cpu')
        if isinstance(initial_metric_state, dict) and 'model' in initial_metric_state:
            initial_metric_state = initial_metric_state['model']
        best_metric_state = {k: v.cpu().clone() for k, v in initial_metric_state.items()}
        log(f'延续历史 metric checkpoint: {initial_metric_path}, score={best_metric_score:.6f}')
    patience_cnt    = 0
    ema_val_loss    = None          
    ema_alpha       = 0.9
    n_iter          = len(train_loader)
    accum           = cfg['accum_steps']
    warmup_done     = False

    start_epoch = max(int(getattr(args, 'start_epoch', 0) or 0), 0)
    if start_epoch >= cfg['epochs']:
        log(f'start_epoch={start_epoch} >= epochs={cfg["epochs"]}，无需训练')
        writer.close()
        log_f.close()
        return
    if start_epoch > cfg['warmup_epochs']:
        scheduler.step(start_epoch - cfg['warmup_epochs'])

    log(f'梯度累积步数={accum}  等效 batch={cfg["batch_size"] * accum}')
    log(f'开始训练：epoch {start_epoch + 1} → {cfg["epochs"]} / 共 {cfg["epochs"]} 个 epoch\n')

    glob_st = time.time()

    for epo in range(start_epoch, cfg['epochs']):

        
        in_warmup = epo < cfg['warmup_epochs']
        if in_warmup:
            cur_lr = apply_warmup_lr(optimizer, epo, cfg['warmup_epochs'], cfg['lr'])
        else:
            cur_lr = optimizer.param_groups[0]['lr']

        model.train()
        optimizer.zero_grad()

        ep_loss = ep_in = ep_grad = ep_ssim = 0.
        st = time.time()

        for it, (image_vis, image_ir) in enumerate(train_loader):
            image_vis = image_vis.to(device, non_blocking=True)
            image_ir  = image_ir.to(device, non_blocking=True)

            
            with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
                fusion = model(image_vis, image_ir)

            
            

            
            with torch.cuda.amp.autocast(enabled=False):
                loss_total, loss_in, loss_ssim, loss_grad = criterion(
                    image_vis=image_vis, image_ir=image_ir,
                    generate_img=fusion.float(), i=epo, labels=None,
                )

            
            scaler.scale(loss_total / accum).backward()

            ep_loss += loss_total.item()
            ep_in   += loss_in.item()
            ep_grad += loss_grad.item()
            ep_ssim += loss_ssim.item()

            
            if (it + 1) % accum == 0 or (it + 1) == n_iter:
                scaler.unscale_(optimizer)
                if is_ablation:
                    
                    
                    
                    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=cfg['grad_clip'])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            
            if (it + 1) % 20 == 0:
                now_it  = n_iter * (epo - start_epoch) + it + 1
                elapsed = time.time() - glob_st
                total_it = n_iter * (cfg['epochs'] - start_epoch)
                eta_sec = int((total_it - now_it)
                              * elapsed / max(now_it, 1))
                eta_str = str(datetime.timedelta(seconds=eta_sec))
                log(f'[{modality}] '
                    f'Ep {epo+1:03d}/{cfg["epochs"]} '
                    f'it {it+1:04d}/{n_iter} | '
                    f'total={loss_total.item():.4f} '
                    f'in={loss_in.item():.4f} '
                    f'grad={loss_grad.item():.4f} '
                    f'ssim={loss_ssim.item():.4f} | '
                    f'lr={cur_lr:.2e} eta={eta_str}')

            
            gs = n_iter * epo + it
            tb_interval = max(int(cfg.get('tb_interval', 10)), 1)
            if (it + 1) % tb_interval == 0 or (it + 1) == n_iter:
                writer.add_scalar('Train/loss_total', loss_total.item(), gs)
                writer.add_scalar('Train/loss_in',    loss_in.item(),    gs)
                writer.add_scalar('Train/loss_grad',  loss_grad.item(),  gs)
                writer.add_scalar('Train/loss_ssim',  loss_ssim.item(),  gs)
                writer.add_scalar('Train/lr',         cur_lr,            gs)

        
        ep_loss /= n_iter
        ep_in   /= n_iter
        ep_grad /= n_iter
        ep_ssim /= n_iter
        ep_time  = time.time() - st
        log(f'\n[Epoch {epo+1}] avg_total={ep_loss:.4f} '
            f'avg_in={ep_in:.4f} avg_grad={ep_grad:.4f} '
            f'avg_ssim={ep_ssim:.4f}  time={ep_time:.1f}s')

        
        val_total, val_in, val_grad, val_ssim = validate(
            model, val_loader, criterion, device)
        log(f'[Val  {epo+1}] total={val_total:.4f} '
            f'in={val_in:.4f} grad={val_grad:.4f} ssim={val_ssim:.4f}')
        writer.add_scalar('Val/loss_total', val_total, epo)
        writer.add_scalar('Val/loss_in',    val_in,    epo)
        writer.add_scalar('Val/loss_grad',  val_grad,  epo)
        writer.add_scalar('Val/loss_ssim',  val_ssim,  epo)

        
        metric_interval_arg = getattr(args, 'metric_interval', None)
        metric_interval = metric_interval_arg if metric_interval_arg is not None else cfg.get('metric_interval', 5)
        metric_interval = max(int(metric_interval), 1)
        if (epo + 1) == 1 or (epo + 1) % metric_interval == 0:
            val_metrics = validate_fusion_metrics(model, val_loader, device)
            if val_metrics is not None:
                log('[MetricVal {0}] score={score:.4f} EN={EN:.4f} SD={SD:.4f} '
                    'SF={SF:.4f} AG={AG:.4f} SCD={SCD:.4f} VIFF={VIFF:.4f} '
                    'Qabf={Qabf:.4f} SSIM={SSIM:.4f}'.format(
                        epo + 1, **val_metrics))
                for mk, mv in val_metrics.items():
                    writer.add_scalar(f'ValMetric/{mk}', mv, epo)
                if val_metrics['score'] > best_metric_score + 1e-4:
                    best_metric_score = val_metrics['score']
                    best_metric_state = {k: v.cpu().clone()
                                         for k, v in model.state_dict().items()}
                    metric_path = os.path.join(ckpt_dir, f'best_metric_{modality}.pth')
                    torch.save(best_metric_state, metric_path)
                    log(f'  → 新最优 metric_score={best_metric_score:.6f}，已保存至 {metric_path}')

        
        if ema_val_loss is None:
            ema_val_loss = val_total
        else:
            ema_val_loss = ema_alpha * ema_val_loss + (1 - ema_alpha) * val_total
        writer.add_scalar('Val/loss_ema', ema_val_loss, epo)

        
        min_delta = 1e-4
        if ema_val_loss < best_val_loss - min_delta:
            best_val_loss = ema_val_loss
            patience_cnt  = 0
            best_state    = {k: v.cpu().clone()
                             for k, v in model.state_dict().items()}
            ckpt_path = os.path.join(ckpt_dir, f'best_{modality}.pth')
            torch.save(best_state, ckpt_path)
            log(f'  → 新最优 ema_val={ema_val_loss:.6f}，已保存至 {ckpt_path}')
        else:
            patience_cnt += 1
            log(f'  patience {patience_cnt}/{cfg["patience"]}')

        
        ckpt_interval = max(int(cfg.get('ckpt_interval', 50)), 1)
        if (not getattr(args, 'disable_periodic_checkpoints', False)
                and (epo + 1) % ckpt_interval == 0):
            periodic_path = os.path.join(ckpt_dir, f'ep{epo+1:03d}_{modality}.pth')
            torch.save(model.state_dict(), periodic_path)
            log(f'  周期 checkpoint → {periodic_path}')

        
        if patience_cnt >= cfg['patience']:
            log(f'\n[早停] 连续 {cfg["patience"]} 个 epoch 未改善，'
                f'最优 ema_val={best_val_loss:.6f}')
            break

        

        if epo >= cfg['warmup_epochs']:
            scheduler.step(epo - cfg['warmup_epochs'])
        cur_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Train/lr_epoch', cur_lr, epo)

    
    
    
    if getattr(args, 'skip_final_aliases', False):
        log('\n已保留 best/best_metric；按配置跳过重复的 final aliases')
    else:
        final_path = os.path.join(ckpt_dir, f'final_{modality}.pth')
        if best_state is not None:
            torch.save(best_state, final_path)
            log(f'\n最优模型（ema_val={best_val_loss:.6f}）已保存至 {final_path}')
        else:
            torch.save(model.state_dict(), final_path)
            log(f'\n模型已保存至 {final_path}')

        if best_metric_state is not None:
            metric_final_path = os.path.join(ckpt_dir, f'final_metric_{modality}.pth')
            torch.save(best_metric_state, metric_final_path)
            log(f'最优指标模型（metric_score={best_metric_score:.6f}）已保存至 {metric_final_path}')

    writer.close()
    log_f.close()
    print(f'[{modality}] 训练完成！日志：{log_path}')






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SGLM+CSIF 图像融合训练脚本')
    parser.add_argument(
        '--modality', '-m', type=str, default='CT-MRI',
        choices=['CT-MRI', 'PET-MRI', 'SPECT-MRI'],
        help='训练模态（默认 CT-MRI）',
    )
    parser.add_argument(
        '--all_modalities', action='store_true',
        help='依次训练全部三种模态',
    )
    parser.add_argument('--vis_dir', type=str, default=None)
    parser.add_argument('--ir_dir', type=str, default=None)
    parser.add_argument(
        '--num_workers', '-j', type=int, default=2,
        help='DataLoader worker 数量',
    )
    parser.add_argument(
        '--metric_interval', type=int, default=None,
        help='每隔多少个 epoch 在验证集计算融合指标并保存 best_metric checkpoint',
    )
    parser.add_argument(
        '--run_id', type=int, default=0,
        help='实验编号，用于区分多次运行的 checkpoint',
    )
    parser.add_argument(
        '--resume_model', type=str, default=None,
        help='从指定模型权重继续训练，只恢复 model.state_dict',
    )
    parser.add_argument(
        '--initial_best_metric_model', type=str, default=None,
        help='断点续跑时保留的历史 best-metric 权重',
    )
    parser.add_argument(
        '--initial_best_metric_score', type=float, default=None,
        help='与 initial_best_metric_model 对应的历史验证综合分数',
    )
    parser.add_argument(
        '--enable_metric_head', action='store_true',
        help='启用可选 base/detail 指标增强输出头；默认关闭以保持旧权重复现',
    )
    parser.add_argument(
        '--enable_mbdg_head', action='store_true',
        help='启用三模态统一 MBDG 低频/卷积细节/源图门控增强头；等价兼容 enable_metric_head',
    )
    parser.add_argument(
        '--upsample_mode', type=str, default='bilinear',
        choices=['bilinear', 'pixelshuffle'],
        help="解码器上采样方式：bilinear 或 pixelshuffle",
    )
    parser.add_argument(
        '--use_fusion_gate', action='store_true',
        help="启用 CSIF 门控融合",
    )
    parser.add_argument(
        '--use_cross_mamba_dffm', action='store_true',
        help='启用DFFM中的零门控CrossMamba双向精炼支路',
    )
    parser.add_argument(
        '--use_common_response', action='store_true',
        help='启用DFFM中的零门控共同响应乘法支路',
    )
    parser.add_argument(
        '--use_true_cross_mamba_dffm', action='store_true',
        help='启用DFFM中的真selective_scan跨模态参数互驱CrossMamba支路，scale非零初始化',
    )
    parser.add_argument(
        '--true_cross_mamba_layers', type=str, default='all', choices=['all', 'shallow_mid', 'shallow'],
        help='TrueCrossMamba启用层级: all=全部DFFM; shallow_mid=浅层+中层; shallow=仅浅层',
    )
    parser.add_argument(
        '--use_deep_csif_gate', action='store_true',
        help='启用第四层瓶颈CSIF残差插值门控；默认关闭时直接使用CSIF输出',
    )
    parser.add_argument(
        '--deep_csif_init', type=float, default=0.01,
        help='第四层CSIF实验性门控scale初值，实际alpha=tanh(scale)',
    )
    parser.add_argument('--experiment_id', type=str, default=None,
                        help='v2消融实验ID；启用独立、安全的输出目录')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--backbone_variant', choices=['sglm', 'fusionmamba'], default='sglm')
    parser.add_argument(
        '--fusion_mode',
        choices=['add', 'original', 'proposed', 'csif_figure_true_cross'],
        default='csif_figure_true_cross')
    parser.add_argument('--fusion_stage_modes', type=str, default=None,
                        help=('浅/中/深/瓶颈逐层模式，逗号分隔: '
                              'add|concat_proj|csif|csif_true_cross|'
                              'residual_csif|'
                              'csif_figure_true_cross；最后一项严格执行图示的'
                              'CrossAttention→交叉参数Mamba→MLP→Conv→加和/乘积→SSD'))
    parser.add_argument('--residual_csif_init', type=float, default=0.1,
                        help='residual_csif 初始混合系数 alpha，范围 (0,1)')
    parser.add_argument('--true_cross_mamba_sites', type=str, default=None,
                        help='显式TrueCross位置，逗号分隔: shallow,mid,deep,bottleneck')
    parser.add_argument('--true_cross_mamba_scale', type=float, default=0.03)
    parser.add_argument('--trainable_true_cross_scale', action='store_true')
    parser.add_argument('--true_cross_state_params', type=str, default='delta,B,C',
                        help='跨模态输入相关状态参数，逗号分隔，可为空字符串')
    parser.add_argument('--csif_disable_bi_attention', action='store_true',
                        help='消融：跳过 CSIF 双向跨注意力，保留后续 scan 与归一化')
    parser.add_argument('--csif_response_aggregation',
                        choices=['sum', 'sum_product'], default='sum_product',
                        help='CSIF 响应聚合：sum_product=加和加乘积；sum=仅加和')
    parser.add_argument('--csif_disable_ssd', action='store_true',
                        help='消融：跳过 CSIF 末端 SSD，保留最终归一化')
    parser.add_argument('--legacy_cross_static_ad', action='store_true',
                        help='仅用于旧权重兼容：交换全局A/D分支参数')
    parser.add_argument('--trainable_deep_csif_scale', action='store_true')
    parser.add_argument('--output_activation', choices=['sigmoid', 'clamp'], default='sigmoid')
    parser.add_argument('--disable_periodic_checkpoints', action='store_true',
                        help='只保留best、best_metric和final，节省磁盘')
    parser.add_argument('--skip_final_aliases', action='store_true',
                        help='仅保存best和best_metric，跳过内容重复的final别名')
    parser.add_argument(
        '--use_fd_block', action='store_true',
        help="启用频率解耦 Mamba 块",
    )
    parser.add_argument(
        '--fd_mode', type=str, default='add', choices=['add', 'route'],
        help="FD块模式: add=零门可选旁路(易被忽略); route=承重(SSM只看低频,高频强制走支路)",
    )
    parser.add_argument(
        '--strict_serial_dvss', action='store_true',
        help='启用SGLM串行路径: MHA->Mamba->LDC/ECA->单MLP，消除重复残差和双MLP',
    )
    parser.add_argument(
        '--dvss_serial_v2', action='store_true',
        help='启用SGLM: MHA->SSM->LDC/ECA->MLP，各段独立pre-norm+单残差，MLP仅一次',
    )
    parser.add_argument(
        '--use_parallel_local_branch', action='store_true',
        help='将Conv/ECA改为输入侧路，与MHA/Mamba全局主路相加后进入最终FFN',
    )
    parser.add_argument('--disable_sglm_global', action='store_true',
                        help='消融：移除所有 SGLM 的 MHA 和 Mamba 全局路径，保留 Conv/ECA 与 FFN')
    parser.add_argument('--disable_sglm_local', action='store_true',
                        help='消融：移除所有 SGLM 的 Conv/ECA 局部路径，保留 MHA/Mamba 与 FFN')
    parser.add_argument('--disable_sglm_mha', action='store_true',
                        help='消融：仅旁路所有 SGLM 的 MHA 子层，保留 Mamba、Conv/ECA 与 FFN')
    parser.add_argument('--disable_sglm_mamba', action='store_true',
                        help='消融：仅旁路所有 SGLM 的 Mamba 子层，保留 MHA、Conv/ECA 与 FFN')
    parser.add_argument(
        '--use_deep_kan', action='store_true',
        help='兼容旧实验：等价于 --decoder_kan_stages 0',
    )
    parser.add_argument(
        '--decoder_kan_stages', type=str, default=None,
        help='使用KAN的decoder阶段索引，逗号分隔；0=8x8,1=16x16,2=32x32,3=64x64',
    )
    parser.add_argument('--kan_grid_size', type=int, default=3,
                        help='Deep-KAN B-spline网格数（默认3）')
    parser.add_argument('--kan_spline_order', type=int, default=3,
                        help='Deep-KAN样条阶数（默认3）')
    parser.add_argument(
        '--aug_level', type=str, default='basic', choices=['basic', 'strong'],
        help="数据增强: basic=原翻转/旋转(可复现); strong=加弹性形变+仿射+随机裁剪(合法扩充分布)",
    )
    parser.add_argument(
        '--use_wavelet_hf', action='store_true',
        help="启用小波域高频融合模块(输出端重组高频子带, 零门=恒等可回退)",
    )
    parser.add_argument(
        '--use_sg_ldc', action='store_true',
        help='启用源图引导多尺度LDC/ECA输出细节增强头',
    )
    parser.add_argument(
        '--loss_profile', type=str, default='default',
        help="损失档位：default=loss.py现有；ct_edge=CT中等强度边缘对齐(供门控融合实验)",
    )
    parser.add_argument(
        '--stat_weight', type=float, default=8.0,
        help='loss_profile 含 stat 时的全局均值/标准差约束权重',
    )
    parser.add_argument(
        '--structure_ablation', type=str, default='none',
        choices=['none', 'b0', 'b1', 'a0', 'a1', 'a2', 'full'],
        help='结构消融预设；非 none 时覆盖相关低层结构开关',
    )
    parser.add_argument(
        '--loss_ablation', type=str, default='none',
        choices=['none', 'base', 'plus_grad', 'plus_lc', 'plus_freq',
                 'full', 'wo_grad', 'wo_lc', 'wo_freq'],
        help='受控损失消融预设',
    )
    parser.add_argument(
        '--start_epoch', type=int, default=0,
        help='续训起始 epoch，日志和学习率调度从该 epoch 继续',
    )
    parser.add_argument(
        '--epochs_override', type=int, default=None,
        help='覆盖当前模态默认 epoch 数，用于限时微调实验',
    )
    parser.add_argument(
        '--lr_override', type=float, default=None,
        help='覆盖当前模态默认学习率，用于低学习率微调实验',
    )
    parser.add_argument(
        '--batch_size_override', type=int, default=None,
        help='覆盖训练 batch 大小',
    )
    parser.add_argument(
        '--accum_steps_override', type=int, default=None,
        help='覆盖梯度累积步数',
    )
    parser.add_argument(
        '--precision', choices=['fp32', 'bf16', 'fp16'], default='fp32',
        help='训练精度',
    )
    parser.add_argument(
        '--ckpt_interval_override', type=int, default=None,
        help='覆盖周期 checkpoint 保存间隔',
    )
    args = parser.parse_args()

    if args.all_modalities:
        for mod in ['CT-MRI', 'PET-MRI', 'SPECT-MRI']:
            print(f'\n{"="*70}')
            print(f'  开始训练模态：{mod}')
            print(f'{"="*70}\n')
            train_fusion(modality=mod, run_id=args.run_id)
            print(f'[{mod}] 训练完成！\n')
    else:
        train_fusion(modality=args.modality, run_id=args.run_id)
        print(f'[{args.modality}] 训练完成！')

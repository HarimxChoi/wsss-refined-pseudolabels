# dist_clip_designed4.py

import argparse
import datetime
import logging
import os
import random
import sys
sys.path.append(".")
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import coco
from utils.losses import get_aff_loss
from utils import evaluate
from utils.AverageMeter import AverageMeter
from utils.camutils import cams_to_affinity_label
from utils.optimizer import PolyWarmupAdamW
from WeCLIP_Plus.model_attn_aff_coco import WeCLIP_Plus


# =========================
# Helpers for URN/BECO/EMA
# =========================


def get_authority_schedule(n_iter, total_iters=80000):
    """Clean start from 35k with smooth decay"""
    if n_iter < 35000:
        return 1.0, 0.80, 0.996
    elif n_iter < 45000:
        # DECAY 35-45k: 1.0 → 0.60
        progress = (n_iter - 35000) / 10000.0
        weight = 1.0 - 0.40 * progress      
        conf_th = 0.80 - 0.08 * progress   
        return weight, conf_th, 0.996
    else:
        return 0.60, 0.72, 0.996


def _sym_kl_from_logits(p_logits, q_logits, T=1.0):
    """Return per-pixel symmetric KL (B,H,W)."""
    p_log = F.log_softmax(p_logits / T, dim=1)
    q_log = F.log_softmax(q_logits / T, dim=1)
    p = p_log.exp()
    q = q_log.exp()
    kl_pq = F.kl_div(p_log, q, reduction='none').sum(1)
    kl_qp = F.kl_div(q_log, p, reduction='none').sum(1)
    return 0.5 * (kl_pq + kl_qp)  # [B,H,W]

def _minmax01(x, eps=1e-6):
    """Per-image min-max to [0,1]. x: [B,H,W]"""
    B = x.shape[0]
    xf = x.view(B, -1)
    xmin = xf.min(dim=1, keepdim=True).values.view(B, 1, 1)
    xmax = xf.max(dim=1, keepdim=True).values.view(B, 1, 1)
    return (x - xmin) / (xmax - xmin + eps)

def _fast_edge_from_label(lbl, ignore_idx=255):
    """
    lbl: [B,H,W] (long)
    return: edge (float) in {0,1}, ignore : 0
    """
    assert lbl.dim() == 3, "lbl must be [B,H,W]"
    B, H, W = lbl.shape
    # boolean edge maps by 4-neighborhood diffs (no padding)
    up = torch.zeros_like(lbl, dtype=torch.bool)
    up[:, 1:, :] = (lbl[:, 1:, :] != lbl[:, :-1, :])

    down = torch.zeros_like(lbl, dtype=torch.bool)
    down[:, :-1, :] = (lbl[:, :-1, :] != lbl[:, 1:, :])

    left = torch.zeros_like(lbl, dtype=torch.bool)
    left[:, :, 1:] = (lbl[:, :, 1:] != lbl[:, :, :-1])

    right = torch.zeros_like(lbl, dtype=torch.bool)
    right[:, :, :-1] = (lbl[:, :, :-1] != lbl[:, :, 1:])

    edge = (up | down | left | right).float()
    edge[lbl == ignore_idx] = 0.0
    return edge


def _class_weights_from_label(lbl, num_classes, clip=(0.5, 3.0), ignore_idx=255):
    """Return class weights [C], mean-normalized then clipped."""
    valid = (lbl != ignore_idx)
    hist = torch.bincount(lbl[valid].view(-1), minlength=num_classes).float().clamp_min(1.0)
    inv = hist.sum() / hist
    inv = inv / inv.mean()
    lo, hi = clip
    return inv.clamp(lo, hi)  # [C]

def _normalize_weight_map(w, eps=1e-6):
    return w / (w.mean() + eps)

def _boundary_logits_from_seg_logits(seg_logits, tau=0.5):
    """Boundary logits via confidence margin (larger near boundary -> higher prob after sigmoid)."""
    with torch.no_grad():
        probs = F.softmax(seg_logits, dim=1)
        top2 = torch.topk(probs, k=2, dim=1).values
        margin = top2[:, 0] - top2[:, 1]  # [B,H,W]
    return -(margin / tau)  # logits for BCEWithLogitsLoss

class EMATeacher:
    def __init__(self, model, tau=0.996, update_trainable_only=True):  
        device = next(model.parameters()).device
        self.teacher = copy.deepcopy(model).to(device).eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.tau = tau
        self.update_trainable_only = update_trainable_only
        
        if update_trainable_only:
            self._include_names = {n for n, p in model.named_parameters() if p.requires_grad}
        else:
            self._include_names = {n for n, _ in model.named_parameters()} 
        
        self._validate_shapes(model)
    
    @torch.no_grad()
    def _validate_shapes(self, model):  # modify: *validate*shapes → _validate_shapes
        src_params = dict(model.named_parameters())
        for n, pt in self.teacher.named_parameters():
            if n not in self._include_names: 
                continue
            ps = src_params.get(n, None)
            if ps is None:
                logging.warning(f"[EMA/init] Teacher has {n} but student doesn't")
                continue
            if ps.shape != pt.shape:
                logging.error(f"[EMA/init] SHAPE MISMATCH: {n} student={ps.shape} teacher={pt.shape}")
    
    @torch.no_grad()
    def update(self, model):
        src_params = dict(model.named_parameters())
        for n, pt in self.teacher.named_parameters():
            if n not in self._include_names:
                continue
            ps = src_params.get(n, None)
            if ps is None:
                continue
            
            if ps.data.shape != pt.data.shape or ps.data.dtype != pt.data.dtype:
                continue
            
            pt.data.mul_(self.tau).add_(ps.data, alpha=(1.0 - self.tau))
        
        src_bufs = dict(model.named_buffers())
        for n, bt in self.teacher.named_buffers():
            sb = src_bufs.get(n, None)
            if sb is not None and sb.data.shape == bt.data.shape and sb.data.dtype == bt.data.dtype:
                bt.data.copy_(sb.data)
    
    @torch.no_grad()
    def forward_logits(self, x, name):
        seg_clip, seg_dino = self.teacher(x, name, mode='val')
        return 0.5 * seg_clip + 0.5 * seg_dino


def get_mask_by_radius(h=20, w=20, radius=8):
    hw = h * w
    mask = np.zeros((hw, hw), dtype=np.float32)
    for i in range(hw):
        _h = i // w
        _w = i % w
        _h0 = max(0, _h - radius)
        _h1 = min(h, _h + radius + 1)
        _w0 = max(0, _w - radius)
        _w1 = min(w, _w + radius + 1)
        for i1 in range(_h0, _h1):
            for i2 in range(_w0, _w1):
                j = i1 * w + i2
                mask[i, j] = 1.0
                mask[j, i] = 1.0
    return mask



parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default='/data1/zbf_data/Project2024/FCLIP_DINO/configs/coco_attn_reg.yaml',
                    type=str,
                    help="config")
parser.add_argument("--resume", default=None, type=str, help="resuming path")
parser.add_argument("--seg_detach", action="store_true", help="detach seg")
parser.add_argument("--work_dir", default=None, type=str, help="work_dir")
parser.add_argument("--radius", default=8, type=int, help="radius")
parser.add_argument("--crop_size", default=320, type=int, help="crop_size")
parser.add_argument("--seed", default=1, type=int, help="random seed")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_logger(filename='test.log'):
    ## setup logger
    logFormatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fHandler = logging.FileHandler(filename, mode='w')
    fHandler.setFormatter(logFormatter)
    logger.addHandler(fHandler)

    cHandler = logging.StreamHandler()
    cHandler.setFormatter(logFormatter)
    logger.addHandler(cHandler)

def cal_eta(time0, cur_iter, total_iter):
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)

    scale = (total_iter-cur_iter) / float(cur_iter)
    delta = (time_now - time0)
    eta = (delta*scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)


def validate_quick(model, dataset, cfg, resize_long=None, max_images=None, num_workers=2):
    
    device = next(model.parameters()).device
    dl = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                     num_workers=num_workers, pin_memory=False)
    num_classes = cfg.dataset.num_classes
    seg_hist = np.zeros((num_classes, num_classes), dtype=np.int64)

    if resize_long is None:
        resize_long = getattr(cfg.clip_init, "resize_long", None)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for i, (name, inputs, labels, _) in tqdm(enumerate(dl), total=len(dl), ncols=100, ascii=" >="):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if resize_long is not None:
                _, _, h, w = inputs.shape
                r = float(resize_long) / float(max(h, w))
                if r != 1.0:
                    nh, nw = int(h * r), int(w * r)
                    inputs = F.interpolate(inputs, size=(nh, nw), mode="bilinear", align_corners=False)

            segs_clip, segs_dino = model(inputs, name, mode='val')     # [B,C,h',w']
            segs = 0.5 * segs_clip + 0.5 * segs_dino
            segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)

            preds = torch.argmax(segs, dim=1).cpu().numpy().astype(np.int16)
            gts   = labels.cpu().numpy().astype(np.int16)


            seg_hist, seg_score = evaluate.scores(gts, preds, seg_hist, num_classes=num_classes)  # :contentReference[oaicite:0]{index=0}

            if (max_images is not None) and (i + 1) >= max_images:
                break
    if was_training:
        model.train()
    return seg_score, seg_hist





def train(cfg, resume_path=None):

    num_workers = 8

    post_cam_warmup = int(getattr(cfg, "post_cam_warmup", 15000))
    teacher_down = float(getattr(cfg, "ema_uncert_down", 1.0))  
    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)

    train_dataset = coco.CocoClsDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.train.split,
        stage='train',
        aug=True,
        resize_range=cfg.dataset.resize_range,
        rescale_range=cfg.dataset.rescale_range,
        crop_size=cfg.dataset.crop_size,
        img_fliplr=True,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )

    val_dataset = coco.CocoSegDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.val.split,
        stage='val',
        aug=False,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.train.samples_per_gpu,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=False,
                              drop_last=True,
                              prefetch_factor=4)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=False,
                            drop_last=False)


    model = WeCLIP_Plus(
        num_classes=cfg.dataset.num_classes,
        clip_model=cfg.clip_init.clip_pretrain_path,
        dino_model=cfg.dino_init.dino_model,
        dino_fts_dim=cfg.dino_init.dino_fts_fuse_dim,
        decoder_layers=cfg.dino_init.decoder_layer,
        embedding_dim=cfg.clip_init.embedding_dim,
        in_channels=cfg.clip_init.in_channels,
        dataset_root_path=cfg.dataset.root_dir,
        clip_flag=cfg.clip_init.clip_flag,
        device='cuda'
    )
    logging.info('\nNetwork config: \n%s' % (model))
    param_groups = model.get_param_groups()
    model.cuda()

    # ----- URN/BECO & EMA hyper -----
    urn_beta = float(getattr(cfg, "urn_beta", 0.2))
    urn_gamma_start = float(getattr(cfg, "urn_gamma_start", 0.70))
    urn_gamma_end   = float(getattr(cfg, "urn_gamma_end",   0.85))
    urn_w_min = float(getattr(cfg, "urn_w_min", 0.3))
    urn_T = float(getattr(cfg, "urn_T", 1.0))

    beco_alpha = float(getattr(cfg, "beco_alpha", 0.5))
    lambda_b   = float(getattr(cfg, "lambda_boundary", 0.3))

    rdrop_cmin  = float(getattr(cfg, "rdrop_cmin", 0.25))
    rdrop_lambda_max = float(getattr(cfg, "rdrop_lambda_max", 0.5))
    rdrop_on_after_ratio = float(getattr(cfg, "rdrop_on_after_ratio", 0.5))

    ema_enable = bool(getattr(cfg, "ema_enable", True))
    ema_tau = float(getattr(cfg, "ema_tau", 0.996))
    min_ema_tau = max(0.996, ema_tau)
    ema_teacher = EMATeacher(model, tau=min_ema_tau) if ema_enable else None

    mask_size = int(cfg.dataset.crop_size // 16)
    attn_mask = get_mask_by_radius(h=mask_size, w=mask_size, radius=args.radius)
    writer = SummaryWriter(cfg.work_dir.tb_logger_dir)


    optimizer = PolyWarmupAdamW(
        params=[
            {
                "params": param_groups[0],
                "lr": cfg.optimizer.learning_rate,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": param_groups[1],
                "lr": 0.0,
                "weight_decay": 0.0,
            },
            {
                "params": param_groups[2],
                "lr": cfg.optimizer.learning_rate*10,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": param_groups[3],
                "lr": cfg.optimizer.learning_rate*10,
                "weight_decay": cfg.optimizer.weight_decay,
           },
        ],
        lr = cfg.optimizer.learning_rate,
        weight_decay = cfg.optimizer.weight_decay,
        betas = cfg.optimizer.betas,
        warmup_iter = cfg.scheduler.warmup_iter,
        max_iter = cfg.train.max_iters,
        warmup_ratio = cfg.scheduler.warmup_ratio,
        power = cfg.scheduler.power
    )

    logging.info('\nOptimizer: \n%s' % optimizer)

    # ==================== RESUME LOGIC ====================
    start_iter = 0
    if resume_path is not None and os.path.isfile(resume_path):
        logging.info(f"\n{'='*80}")
        logging.info(f"[RESUME] Loading checkpoint from: {resume_path}")
        
        checkpoint = torch.load(resume_path, map_location='cpu')
        
        # Load model
        model.load_state_dict(checkpoint['student'], strict=False)
        logging.info("[RESUME] Student model loaded (strict=False)")
        
        # Load EMA teacher
        if ema_teacher is not None and checkpoint.get('ema_teacher') is not None:
            ema_teacher.teacher.load_state_dict(checkpoint['ema_teacher'], strict=False)
            ema_teacher.tau = max(checkpoint.get('ema_tau', 0.996), min_ema_tau)
            logging.info(f"[RESUME] EMA teacher loaded (tau={ema_teacher.tau:.4f} (clamped≥{min_ema_tau:.4f}), strict=False)")
        
        # Load optimizer
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        for i, pg in enumerate(optimizer.param_groups):
            if i == 1:  # CLIP frozen
                continue
            old_lr = float(pg.get('lr', 0.0))
            new_lr = 1.0e-5 if i == 0 else 1.0e-4
            pg['lr'] = new_lr
            logging.info(f"[RESUME] Param group {i}: LR {old_lr:.2e} → {new_lr:.2e}")
        
        # Load iteration count
        start_iter = checkpoint.get('n_iter', 0)
        logging.info(f"[RESUME] Starting from iteration: {start_iter}")
        
        # Load RNG states for reproducibility
        if 'torch_rng' in checkpoint:
            torch.set_rng_state(checkpoint['torch_rng'])
        if 'torch_cuda_rng' in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint['torch_cuda_rng'])
        if 'numpy_rng' in checkpoint:
            np.random.set_state(checkpoint['numpy_rng'])
        if 'python_rng' in checkpoint:
            random.setstate(checkpoint['python_rng'])
        logging.info("[RESUME] RNG states restored")
        
        logging.info(f"{'='*80}\n")
    elif resume_path is not None:
        logging.warning(f"[RESUME] Checkpoint not found: {resume_path}")
        logging.warning("[RESUME] Starting from scratch")

    train_loader_iter = iter(train_loader)
    avg_meter = AverageMeter()

    for n_iter in range(start_iter, cfg.train.max_iters):
        try:
            img_name, inputs, cls_labels, img_box = next(train_loader_iter)
        except:
            train_loader_iter = iter(train_loader)
            img_name, inputs, cls_labels, img_box = next(train_loader_iter)

        inputs = inputs.cuda(non_blocking=True)

        teacher_weight, conf_threshold, _ = get_authority_schedule(n_iter, cfg.train.max_iters)
        current_ema_tau = min_ema_tau

        if ema_teacher is not None:
            ema_teacher.tau = current_ema_tau

        use_teacher_every = 1

            
        use_teacher_now = (ema_teacher is not None) and ((n_iter + 1) % use_teacher_every == 0)

        # === (A) Teacher forward with WEIGHTED probability injection ===
        teacher_prob_for_cam = None
        t_logits_for_uncert = None
        
        if use_teacher_now:
            with torch.no_grad():
                if teacher_down < 1.0:
                    b, c, h, w = inputs.shape
                    th, tw = max(1, int(h * teacher_down)), max(1, int(w * teacher_down))
                    t_in = F.interpolate(inputs, size=(th, tw), mode='bilinear', align_corners=False)
                else:
                    t_in = inputs
                
                t_logits_for_uncert = ema_teacher.forward_logits(t_in, img_name)
                teacher_prob_raw = F.softmax(t_logits_for_uncert, dim=1)
                
                # --- Dynamic Confidence Gate ---
                if n_iter < 35000:
                    gate_th = 0.68
                elif n_iter < 45000:
                    progress = (n_iter - 35000) / 10000.0
                    gate_th = 0.68 + 0.12 * progress  # 0.68 → 0.78
                else:
                    gate_th = 0.80
                
                t_conf_map, _ = teacher_prob_raw.max(dim=1, keepdim=True)  # [B,1,H,W]
                t_gate = (t_conf_map > gate_th).float()
                gate_pass_rate = t_gate.mean().item()
                                
                # --- Mixing Strategy ---
                if n_iter >= 35000 and teacher_weight < 1.0:
                    uniform_dist = torch.ones_like(teacher_prob_raw) / cfg.dataset.num_classes
                    mixed = teacher_weight * teacher_prob_raw + (1.0 - teacher_weight) * uniform_dist
                    teacher_prob_for_cam = mixed * t_gate + uniform_dist * (1.0 - t_gate)
                else:
                    teacher_prob_for_cam = teacher_prob_raw
                    gate_pass_rate = 1.0
        
        # Inject to model
        if ema_teacher is not None:
            model.set_teacher_seg_prob(teacher_prob_for_cam)
        else:
            model.set_teacher_seg_prob(None)
        
        # === (B) Student forward ===
        segs_clip, segs_dino, cam, attn_pred = model(inputs, img_name)
        pseudo_label = cam.clone()  # [B,H',W']
        segs = 0.5 * segs_clip + 0.5 * segs_dino
        segs = F.interpolate(segs, size=pseudo_label.shape[1:], mode='bilinear', align_corners=False)
        
        if n_iter >= 35000:  
            with torch.no_grad():
                pred_max, pred_label_seg = torch.max(F.softmax(segs, dim=1), dim=1)
                high_conf_mask = pred_max > conf_threshold
                
                # Only overwrite non-background predictions
                fg_mask = (pred_label_seg != 0)
                overwrite_mask = high_conf_mask & fg_mask
                
                pseudo_label[overwrite_mask] = pred_label_seg[overwrite_mask]
                
                # Track overwrite ratio
                overwrite_ratio = overwrite_mask.float().mean()
        else:
            overwrite_ratio = 0.0

        # === (C) Uncertainty map ===
        with torch.no_grad():
            if use_teacher_now and (t_logits_for_uncert is not None):
                t_logits_up = t_logits_for_uncert
                if t_logits_up.shape[2:] != pseudo_label.shape[1:]:
                    t_logits_up = F.interpolate(
                        t_logits_up, size=pseudo_label.shape[1:], 
                        mode='bilinear', align_corners=False
                    )
                u_map = _sym_kl_from_logits(t_logits_up, segs, T=urn_T)
            else:
                p = F.softmax(segs / max(urn_T, 1.0), dim=1)
                u_map = -(p * (p.clamp_min(1e-8)).log()).sum(1)
            u_hat = _minmax01(u_map)

        # === (D) Weight maps ===
        with torch.no_grad():
            w_cls = _class_weights_from_label(pseudo_label, cfg.dataset.num_classes).to(segs.device)
            w_wce = w_cls[pseudo_label.clamp_min(0)]
            w_wce[pseudo_label == cfg.dataset.ignore_index] = 0.0

            edge_map = _fast_edge_from_label(pseudo_label, ignore_idx=cfg.dataset.ignore_index)
            w_beco = 1.0 + beco_alpha * edge_map

            prog = (n_iter + 1) / float(cfg.train.max_iters)
            gamma_t = urn_gamma_start + (urn_gamma_end - urn_gamma_start) * max(0.0, min(1.0, prog))
            w_urn = torch.sigmoid((gamma_t - u_hat) / urn_beta)
            if urn_w_min > 0:
                w_urn = torch.where(edge_map > 0, torch.clamp(w_urn, min=urn_w_min), w_urn)

            w_prime = _normalize_weight_map(w_wce * w_beco * w_urn).detach()

        # === (E) Losses ===
        valid = (pseudo_label != cfg.dataset.ignore_index).float()
        denom = valid.sum().clamp_min(1.0)

        ce = F.cross_entropy(segs, pseudo_label.long(), ignore_index=cfg.dataset.ignore_index, reduction='none')
        L_ce = (ce * w_prime * valid).sum() / denom

        boundary_logits = _boundary_logits_from_seg_logits(segs, tau=0.5)
        L_boundary = F.binary_cross_entropy_with_logits(boundary_logits, edge_map, reduction='none')
        L_boundary = lambda_b * ((L_boundary * valid).sum() / denom)

        L_kl_soft = segs.new_tensor(0.0)

        fts_cam = cam.clone()
        aff_label = cams_to_affinity_label(fts_cam, mask=attn_mask, ignore_index=cfg.dataset.ignore_index)
        attn_loss, pos_count, neg_count = get_aff_loss(attn_pred, aff_label)
        lambda_attn = float(getattr(cfg, "lambda_attn", 0.1))

        loss = L_ce + L_boundary + L_kl_soft + lambda_attn * attn_loss

        # === (F) Optimization ===
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ema_teacher is not None:
            ema_teacher.update(model)

        # Fixed LR 1e-5 - no schedule
        for i, pg in enumerate(optimizer.param_groups):
            if i == 1:  # CLIP frozen
                continue
            pg['lr'] = 1.0e-5 if i == 0 else 1.0e-4

        # === (G) Logging ===
        avg_meter.add({
            'L_ce': float(L_ce.detach().item()),
            'L_boundary': float(L_boundary.detach().item()),
            'L_kl': float(L_kl_soft.detach().item()),
            'attn_loss': float(attn_loss.detach().item())
        })

        if (n_iter + 1) % cfg.train.log_iters == 0:
            delta, eta = cal_eta(time0, n_iter+1, cfg.train.max_iters)
            cur_lr = optimizer.param_groups[0]['lr']
            preds = torch.argmax(segs, dim=1).cpu().numpy().astype(np.int16)
            gts = pseudo_label.cpu().numpy().astype(np.int16)
            seg_mAcc = (preds == gts).sum() / preds.size
            
            # Teacher stats
            teacher_stats = {
                'active': use_teacher_now,
                'stride': use_teacher_every,
                'weight': teacher_weight,  
                'tau': current_ema_tau,    
            }
            
            if teacher_prob_for_cam is not None:
                with torch.no_grad():
                    t_conf_raw, _ = teacher_prob_raw.max(dim=1)
                    t_conf_mix, _ = teacher_prob_for_cam.max(dim=1)
                    teacher_stats['conf_mean_raw'] = float(t_conf_raw.mean().item())
                    teacher_stats['conf_max_raw']  = float(t_conf_raw.max().item())
                    teacher_stats['conf_mean']     = float(t_conf_mix.mean().item())
                    teacher_stats['conf_max']      = float(t_conf_mix.max().item())
                    teacher_stats['gate_pass']     = gate_pass_rate
            
            edge_ratio = float(edge_map.mean().item())
            
            logging.info(
                "Iter:%d Elapsed:%s ETA:%s LR:%.3e | "
                "L_ce:%.4f L_bnd:%.4f attn:%.4f mAcc:%.4f | "
                "gamma:%.3f edge:%.3f | "
                "T[%s/%d w:%.2f τ:%.4f conf_raw:%.3f conf:%.3f gate:%.2f(th:%.2f)] | "
                "ConfTh:%.2f OW:%.3f"
                % (
                    n_iter + 1, delta, eta, cur_lr,
                    float(L_ce.item()), float(L_boundary.item()),
                    float(attn_loss.item()), seg_mAcc,
                    gamma_t, edge_ratio,
                    'ON' if use_teacher_now else 'OFF',
                    use_teacher_every,
                    teacher_weight,
                    current_ema_tau,
                    teacher_stats.get('conf_mean_raw', 0.0),
                    teacher_stats.get('conf_mean', 0.0),
                    teacher_stats.get('gate_pass', 1.0),
                    gate_th,
                    conf_threshold,
                    overwrite_ratio
                )
            )

            writer.add_scalar('train/mAcc', seg_mAcc, n_iter)
            writer.add_scalars('train/loss', {
                "L_ce": float(L_ce.item()),
                "L_boundary": float(L_boundary.item()),
                "attn": float(attn_loss.item())
            }, n_iter)

            # Track edge ratio trend
            writer.add_scalar('train/edge_ratio_ma', edge_ratio, n_iter)
            
            # Authority transfer metrics
            writer.add_scalars('train/authority', {
                "teacher_weight": teacher_weight,
                "conf_threshold": conf_threshold,
                "ema_tau": current_ema_tau,
                "overwrite_ratio": overwrite_ratio
            }, n_iter)
            
            writer.add_scalar('train/gamma_t', gamma_t, n_iter)
            writer.add_scalar('train/edge_ratio', edge_ratio, n_iter)
            
            if 'conf_mean' in teacher_stats:
                # Recalculate gate_th for TensorBoard
                if n_iter < 35000:
                    gate_th_tb = 0.68
                elif n_iter < 45000:
                    progress = (n_iter - 35000) / 10000.0
                    gate_th_tb = 0.68 + 0.12 * progress
                else:
                    gate_th_tb = 0.80
            
                writer.add_scalars('train/teacher_conf', {
                    "mean_raw": teacher_stats['conf_mean_raw'],
                    "max_raw":  teacher_stats['conf_max_raw'],
                    "mean": teacher_stats['conf_mean'],
                    "max": teacher_stats['conf_max'],
                    "gate_pass_rate": teacher_stats.get('gate_pass', 1.0),
                    "gate_threshold": gate_th_tb  # NEW
                }, n_iter)
            # Track OW distribution percentiles
            with torch.no_grad():
                if n_iter >= 35000 and overwrite_ratio > 0:
                    ow_pixels = overwrite_mask.float()
                    # Per-image OW ratios
                    ow_per_img = ow_pixels.view(ow_pixels.shape[0], -1).mean(1)
                    ow_p25 = torch.quantile(ow_per_img, 0.25).item()
                    ow_p50 = torch.quantile(ow_per_img, 0.50).item()
                    ow_p75 = torch.quantile(ow_per_img, 0.75).item()
                    
                    writer.add_scalars('train/ow_distribution', {
                        'p25': ow_p25,
                        'p50': ow_p50,
                        'p75': ow_p75,
                        'mean': overwrite_ratio
                    }, n_iter)
                else:
                    ow_p25, ow_p50, ow_p75 = 0.0, 0.0, 0.0

        # === (G-2) Detailed logging every 5k ===
        if (n_iter + 1) % 5000 == 0:
            with torch.no_grad():
                if teacher_prob_for_cam is not None:
                    t_conf_raw, _ = teacher_prob_raw.max(dim=1)
                    t_conf_mix, _ = teacher_prob_for_cam.max(dim=1)
                    teacher_stats['conf_mean_raw'] = float(t_conf_raw.mean().item())
                    teacher_stats['conf_mean']     = float(t_conf_mix.mean().item())
                    logging.info(f"\n{'='*80}")
                    logging.info(f"Detailed Stats @ Iter {n_iter+1}:")
                    logging.info(f"  Teacher: weight={teacher_weight:.3f}, τ={current_ema_tau:.4f}, stride={use_teacher_every}")
                    logging.info(f"  Confidence (RAW): min={t_conf_raw.min():.4f}, mean={t_conf_raw.mean():.4f}, max={t_conf_raw.max():.4f}")
                    logging.info(f"  Confidence (MIX): min={t_conf_mix.min():.4f}, mean={t_conf_mix.mean():.4f}, max={t_conf_mix.max():.4f}")

                    if n_iter < 35000:
                        gate_th_display = 0.68
                    elif n_iter < 45000:
                        progress = (n_iter - 35000) / 10000.0
                        gate_th_display = 0.68 + 0.12 * progress
                    else:
                        gate_th_display = 0.80
                
                    logging.info(f"  Gate: threshold={gate_th_display:.2f}, pass_rate={teacher_stats.get('gate_pass', 1.0):.3f}")
                    logging.info(f"  Pseudo-label: conf_th={conf_threshold:.3f}, overwrite={overwrite_ratio:.3f}")
                    logging.info(f"  Edge ratio: {edge_ratio:.4f}, URN gamma: {gamma_t:.4f}")
                    # NEW: OW distribution stats
                    if n_iter >= 35000 and overwrite_ratio > 0:
                        logging.info(f"  OW distribution: p25={ow_p25:.3f}, p50={ow_p50:.3f}, p75={ow_p75:.3f}")
                    
                    logging.info(f"  Edge ratio: {edge_ratio:.4f}, URN gamma: {gamma_t:.4f}")
                logging.info(f"{'='*80}\n")

        # === (H) Validation & checkpointing ===
        if (n_iter + 1) % cfg.train.eval_iters == 0:
            ckpt_name = os.path.join(cfg.work_dir.ckpt_dir, f"wetr_iter_{n_iter+1}.pth")
            
            state = {
                "student": model.state_dict(),
                "ema_teacher": ema_teacher.teacher.state_dict() if ema_teacher else None,
                "ema_tau": current_ema_tau,
                "teacher_weight": teacher_weight,
                "conf_threshold": conf_threshold,
                "optimizer": optimizer.state_dict(),
                "n_iter": n_iter + 1,
                "torch_rng": torch.get_rng_state(),
                "torch_cuda_rng": torch.cuda.get_rng_state_all(),
                "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(),
                "cfg": OmegaConf.to_container(cfg, resolve=True)
            }
            
            torch.save(state, ckpt_name)
            logging.info(f"Checkpoint saved: {ckpt_name}")
            
            logging.info('Validating (quick single-scale)...')
            seg_score, _ = validate_quick(
                model, val_dataset, cfg,
                resize_long=getattr(cfg.clip_init, "resize_long", None),
                max_images=getattr(cfg, "val_max_images", None),
                num_workers=2
            )
            logging.info(
                f"[VAL] mIoU: {seg_score['miou']:.4f}  "
                f"pAcc: {seg_score['pAcc']:.4f}  "
                f"mAcc: {seg_score['mAcc']:.4f}"
            )

    return True


if __name__ == "__main__":

    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    cfg.dataset.crop_size = args.crop_size

    if args.work_dir is not None:
        cfg.work_dir.dir = args.work_dir

    timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())

    cfg.work_dir.ckpt_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.ckpt_dir, timestamp)
    cfg.work_dir.pred_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.pred_dir)
    cfg.work_dir.tb_logger_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.tb_logger_dir, timestamp)

    os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.pred_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.tb_logger_dir, exist_ok=True)

    setup_logger(filename=os.path.join(cfg.work_dir.dir, timestamp+'.log'))
    logging.info('\nargs: %s' % args)
    logging.info('\nconfigs: %s' % cfg)

    setup_seed(args.seed)
    logging.info(f"[Seed] {args.seed}") 
    train(cfg=cfg, resume_path=args.resume)

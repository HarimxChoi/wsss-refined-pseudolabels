# python scripts/dist_clip_coco.py --config configs/coco_attn_reg.yaml baseline

import argparse
import datetime
import logging
import os
import random
import sys
sys.path.append(".")
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import math
import gc

from datasets import coco
from utils.losses import get_aff_loss
from utils import evaluate
from utils.AverageMeter import AverageMeter
from utils.camutils import cams_to_affinity_label
from utils.optimizer import PolyWarmupAdamW
from WeCLIP_Plus.model_attn_aff_coco import WeCLIP_Plus
from WeCLIP_Plus.dice_loss import DiceLoss


parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default='/data1/zbf_data/Project2024/FCLIP_DINO/configs/coco_attn_reg.yaml',
                    type=str,
                    help="config")
parser.add_argument("--seg_detach", action="store_true", help="detach seg")
parser.add_argument("--work_dir", default=None, type=str, help="work_dir")
parser.add_argument("--radius", default=8, type=int, help="radius")
parser.add_argument("--crop_size", default=320, type=int, help="crop_size")



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

def _safe_float(x, default=0.0):
    if isinstance(x, torch.Tensor):
        try: return float(x.detach().item())
        except Exception: return float(x.detach().mean().item())
    if isinstance(x, (int, float)): return float(x)
    return float(default)

def compute_class_ious_from_lists(preds, gts, num_classes, ignore_index=255):
    """
    preds/gts: lists of HxW np.int16 arrays (what your validate() already builds)
    Returns:
      ious: [num_classes] float64 array
    """
    hist = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, g in zip(preds, gts):
        p = p.reshape(-1).astype(np.int64)
        g = g.reshape(-1).astype(np.int64)
        m = (g != ignore_index) & (g >= 0) & (g < num_classes)
        if not np.any(m):
            continue
        h = np.bincount(num_classes * g[m] + p[m], minlength=num_classes**2)
        hist += h.reshape(num_classes, num_classes)
    diag = np.diag(hist).astype(np.float64)
    denom = (hist.sum(1) + hist.sum(0) - diag + 1e-6)
    ious = diag / denom
    return ious

@torch.no_grad()
def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    # logits [B,C,H,W] → entropy [B,1,H,W]
    p = F.softmax(logits.float(), dim=1)
    return -(p.clamp_min(1e-8) * p.log().clamp_min(-30)).sum(1, keepdim=True)

def _cleanup_memory():
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


@torch.no_grad()
def validate(model=None, data_loader=None, cfg=None):
    """Baseline validation — uses 0.5*CLIP + 0.5*DINO logits (no adapters, no gating)."""
    model_was_training = model.training
    model.eval()

    preds, gts = [], []
    seg_hist = np.zeros((cfg.dataset.num_classes, cfg.dataset.num_classes))

    # Optional light TTA from cfg.eval (if you later add it to yaml). Defaults to off.
    eval_cfg   = getattr(cfg, "eval", None)
    tta_enable = bool(getattr(eval_cfg, "tta_enable", False)) if eval_cfg else False
    tta_scales = tuple(getattr(eval_cfg, "tta_scales", [1.0])) if eval_cfg else (1.0,)
    tta_hflip  = bool(getattr(eval_cfg, "tta_hflip", False)) if eval_cfg else False

    for _, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):
        name, inputs, labels, _ = data
        inputs = inputs.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        if not tta_enable:
            seg_clip, seg_dino = model(inputs, name, 'val')         # <-- eval returns 2 tensors
            segs = 0.5 * seg_clip + 0.5 * seg_dino
        else:
            # simple TTA: multi-scale + optional hflip, then average logits
            B, C, H, W = inputs.shape
            acc, views = None, 0
            for s in tta_scales:
                x = inputs if s == 1.0 else F.interpolate(inputs, scale_factor=s, mode='bilinear', align_corners=False)
                sc, sd = model(x, name, 'val')
                ss = 0.5 * sc + 0.5 * sd
                ss = F.interpolate(ss, size=(H, W), mode='bilinear', align_corners=False)
                acc = ss if acc is None else (acc + ss)
                views += 1
                if tta_hflip:
                    xf = torch.flip(x, dims=[3])
                    scf, sdf = model(xf, name, 'val')
                    ssf = 0.5 * scf + 0.5 * sdf
                    ssf = torch.flip(ssf, dims=[3])
                    ssf = F.interpolate(ssf, size=(H, W), mode='bilinear', align_corners=False)
                    acc = acc + ssf
                    views += 1
            segs = acc / max(1, views)

        segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
        preds.extend(torch.argmax(segs, dim=1).cpu().numpy().astype(np.int16))
        gts.extend(labels.cpu().numpy().astype(np.int16))

    seg_hist, seg_score = evaluate.scores(gts, preds, seg_hist, num_classes=cfg.dataset.num_classes)

    # --- per-class IoUs for detailed tracking ---
    try:
        per_cls_iou = compute_class_ious_from_lists(
            preds, gts, num_classes=cfg.dataset.num_classes, ignore_index=cfg.dataset.ignore_index
        )
        seg_score = dict(seg_score)  # ensure it's mutable
        seg_score["per_class_iou"] = per_cls_iou  # numpy array, length = num_classes

        # quick human-readable summary in logs
        order = np.argsort(per_cls_iou)
        best15  = [(int(i), float(per_cls_iou[i])) for i in order[-15:][::-1]]
        worst15 = [(int(i), float(per_cls_iou[i])) for i in order[:15]]
        logging.info(f"[VAL-IoU] best15 {best15} | worst15 {worst15}")
    except Exception as e:
        logging.warning(f"[VAL-IoU] per-class computation failed: {e}")

    if model_was_training:
        model.train()
    try:
        del preds, gts, segs, inputs, labels
    except Exception:
        pass
            
    _cleanup_memory()
        
    return seg_score



def get_seg_loss(pred, label, ignore_index=255):
    bg_label = label.clone()
    bg_label[label!=0] = ignore_index
    bg_loss = F.cross_entropy(pred, bg_label.type(torch.long), ignore_index=ignore_index)
    fg_label = label.clone()
    fg_label[label==0] = ignore_index
    fg_loss = F.cross_entropy(pred, fg_label.type(torch.long), ignore_index=ignore_index)

    return (bg_loss + fg_loss) * 0.5


def get_mask_by_radius(h=20, w=20, radius=8):
    hw = h * w
    mask  = np.zeros((hw, hw))
    for i in range(hw):
        _h = i // w
        _w = i % w

        _h0 = max(0, _h - radius)
        _h1 = min(h, _h + radius+1)
        _w0 = max(0, _w - radius)
        _w1 = min(w, _w + radius+1)
        for i1 in range(_h0, _h1):
            for i2 in range(_w0, _w1):
                _i2 = i1 * w + i2
                mask[i, _i2] = 1
                mask[_i2, i] = 1

    return mask



def train(cfg):

    num_workers = 4
    
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
                            pin_memory=True,
                            drop_last=False,
                              prefetch_factor=1)


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
    logging.info('\nNetwork config: \n%s'%(model))
    param_groups = model.get_param_groups()
    model.cuda()

    mask_size = int(cfg.dataset.crop_size // 16)
    attn_mask = get_mask_by_radius(h=mask_size, w=mask_size, radius=args.radius)
    writer = SummaryWriter(cfg.work_dir.tb_logger_dir)
    best_miou = -1.0

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

    train_loader_iter = iter(train_loader)

    avg_meter = AverageMeter()
    criterion_dice = DiceLoss().cuda()

    for n_iter in range(cfg.train.max_iters):
        
        try:
            img_name, inputs, cls_labels, img_box = next(train_loader_iter)
        except:
            train_loader_iter = iter(train_loader)
            img_name, inputs, cls_labels, img_box = next(train_loader_iter)

        segs_clip, segs_dino, cam, attn_pred = model(inputs.cuda(), img_name)

        pseudo_label = cam

        segs= 0.5*segs_clip+0.5*segs_dino
        segs = F.interpolate(segs, size=pseudo_label.shape[1:], mode='bilinear', align_corners=False)

        segs_clip = F.interpolate(segs_clip, size=pseudo_label.shape[1:], mode='bilinear', align_corners=False)
        segs_dino = F.interpolate(segs_dino, size=pseudo_label.shape[1:], mode='bilinear', align_corners=False)

        pred_clip_max, pred_label_clip = torch.max(F.softmax(segs_clip, dim=1), dim=1)
        pred_dino_max, pred_label_dino = torch.max(F.softmax(segs_dino, dim=1), dim=1)
        pred_max, pred_label_seg = torch.max(F.softmax(segs, dim=1), dim=1)

        fts_cam = cam.clone()

        aff_label = cams_to_affinity_label(fts_cam, mask=attn_mask, ignore_index=cfg.dataset.ignore_index)
        attn_loss, pos_count, neg_count = get_aff_loss(attn_pred, aff_label)

        # --- diagnostics comparable to current code (not used for training) ---
        hc = entropy_from_logits(segs_clip)     # [B,1,h,w]
        hd = entropy_from_logits(segs_dino)     # [B,1,h,w]
        with torch.no_grad():
            # a "compat" gating weight just for logging (baseline still uses 0.5/0.5)
            w_gate = torch.sigmoid(torch.tensor(6.0, device=segs_clip.device) * (hd - hc))  # [B,1,h,w]
            pred_clip = torch.argmax(segs_clip, dim=1)
            pred_dino = torch.argmax(segs_dino, dim=1)
            stream_agree = (pred_clip == pred_dino).float().mean()
                

        if n_iter > 40000:
            pseudo_label[pred_max>0.75] = pred_label_seg[pred_max>0.75]

        seg_loss = get_seg_loss(segs, pseudo_label.type(torch.long), ignore_index=cfg.dataset.ignore_index)
        seg_loss2 = criterion_dice(segs, pseudo_label.type(torch.long))

        seg_clip_loss2 = get_seg_loss(segs_clip, pred_label_dino.type(torch.long), ignore_index=cfg.dataset.ignore_index)
        seg_dino_loss2 = get_seg_loss(segs_dino, pred_label_clip.type(torch.long), ignore_index=cfg.dataset.ignore_index)

        

        loss = 1 * seg_loss + 0.1 * attn_loss + 0.1 * seg_clip_loss2 + 0.1 * seg_dino_loss2 + 1 * seg_loss2

        avg_meter.add({'seg_loss': seg_loss.item(), 'attn_loss': attn_loss.item()})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (n_iter + 1) % cfg.train.log_iters == 0:
            
            delta, eta = cal_eta(time0, n_iter+1, cfg.train.max_iters)
            cur_lr = optimizer.param_groups[0]['lr']
        
            # proxy accuracy vs pseudo
            preds = torch.argmax(segs, dim=1).cpu().numpy().astype(np.int16)
            gts   = pseudo_label.cpu().numpy().astype(np.int16)
            seg_mAcc = (preds == gts).sum() / preds.size
        
            # mem
            torch.cuda.synchronize()
            used = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
        
            # overwrite ratio (after 40k)
            ow_ratio = None
            if n_iter > 40000:
                with torch.no_grad():
                    p_seg = F.softmax(segs, dim=1)
                    ent   = -(p_seg.clamp_min(1e-8) * p_seg.log().clamp_min(-30)).sum(1) # [B,H,W]
                    H_tau = torch.quantile(ent.view(-1), 0.15)
                    ow_mask = (ent <= H_tau) & (pred_label_seg != 0)
                    ow_ratio = float(ow_mask.float().mean().item())
        
            # stream confidence & “compat” gating stats
            hc_mean = float(hc.mean().item()); hd_mean = float(hd.mean().item())
            w_mean  = float(w_gate.mean().item()); w_std = float(w_gate.std().item())
            w_ext   = float(((w_gate < 0.1) | (w_gate > 0.9)).float().mean().item())
            agr     = float(stream_agree.item())
        
            logging.info(
                f"Iter {n_iter+1} | {delta} / ETA {eta} | LR {cur_lr:.3e} | "
                f"seg_ce {seg_loss.item():.4f} | dice {seg_loss2.item():.4f} | "
                f"attn {attn_loss.item():.4f} | clip2 {seg_clip_loss2.item():.4f} | dino2 {seg_dino_loss2.item():.4f} | "
                f"mAcc {seg_mAcc:.4f} | [Mem] used={used:.2f}G resv={reserved:.2f}G | "
                f"[gating-compat] hc={hc_mean:.3f} hd={hd_mean:.3f} w={w_mean:.3f}±{w_std:.3f} ext={w_ext:.3f} agr={agr:.3f}"
                + (f" | ow_ratio {ow_ratio:.4f}" if ow_ratio is not None else "")
            )
        
            # TensorBoard
            writer.add_scalars('train/loss', {
                "seg_ce": seg_loss.item(),
                "dice":   seg_loss2.item(),
                "attn":   attn_loss.item(),
                "clip2":  seg_clip_loss2.item(),
                "dino2":  seg_dino_loss2.item(),
            }, global_step=n_iter)
            writer.add_scalar('train/lr', cur_lr, n_iter)
            writer.add_scalar('train/mAcc_vs_pseudo', seg_mAcc, n_iter)
            writer.add_scalars('train/gating_compat', {
                "hc_mean": hc_mean, "hd_mean": hd_mean, "w_mean": w_mean, "w_std": w_std, "extreme_frac": w_ext, "agreement": agr
            }, n_iter)
            if ow_ratio is not None:
                writer.add_scalar('train/pseudo_overwrite_ratio', ow_ratio, n_iter)
            writer.flush()

        
        if (n_iter + 1) % cfg.train.eval_iters == 0:
            ckpt_name = os.path.join(cfg.work_dir.ckpt_dir, f"iter_{n_iter+1}.pth")
            torch.save(model.state_dict(), ckpt_name, _use_new_zipfile_serialization=False)
            logging.info(f"[CKPT] saved -> {ckpt_name}")
        
            logging.info('Validating...')
            seg_score = validate(model=model, data_loader=val_loader, cfg=cfg)

            if "per_class_iou" in seg_score:
                for i, v in enumerate(seg_score["per_class_iou"]):
                    writer.add_scalar(f'val/iou/class_{i}', float(v), n_iter)
        
            # pull mIoU for logging / best checkpoint
            miou = None
            for k in ("Mean IoU", "mIoU", "mean_iou", "miou"):
                if k in seg_score:
                    miou = float(seg_score[k]); break
        
            logging.info(f"[VAL] {('mIoU: %.4f — ' % miou) if miou is not None else ''}{seg_score}")
            if miou is not None:
                writer.add_scalar('val/mIoU', miou, n_iter)
                if miou > best_miou:
                    best_miou = miou
                    best_path = os.path.join(cfg.work_dir.ckpt_dir, f"best_{n_iter+1}_{miou:.4f}.pth")
                    torch.save({"iter": n_iter + 1, "model": model.state_dict(), "mIoU": miou},
                               best_path, _use_new_zipfile_serialization=False)
                    logging.info(f"[CKPT] new best -> {best_path}")
        
            _cleanup_memory()

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

    setup_seed(1)
    train(cfg=cfg)

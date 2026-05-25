# Results

COCO-Val mIoU (ViT-B/16):

| Method | mIoU |
|---|---|
| WeCLIP (CVPR 2024) | 47.1 |
| WeCLIP+ (TPAMI 2025) | 51.9 |
| **Ours** | **54.4** |

Δ vs WeCLIP+ baseline: **+2.5pp**.
Δ vs WeCLIP (CVPR 2024): **+7.3pp**.

## Training progression

| Iter | mIoU |
|---|---|
| 60k | 51.92 |
| 70k | 52.93 |
| 80k | 53.07 |

Final eval (best checkpoint, `wetr_best.pth` from iter 80k): **54.36**.

## Config

- Backbone: CLIP ViT-B/16 + DINOv2 ViT-S/14 (decoder 3 layers)
- Method: self-train + RFM agreement-learning (th=0.82) + URN/BECO
- Loss: CE + Dice + Cross + Attn + Boundary
- Optimizer: AdamW (lr 2e-5), crop 320, batch 8/GPU
- Dataset: MSCOCO (81 classes), 80k iters

## Logs

- `training.log` — training run (iter 30k → 80k, resumed from prior checkpoint)
- `eval.log` — final evaluation on COCO val

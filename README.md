# wsss-refined-pseudolabels

Weakly-supervised semantic segmentation with refined pseudo-labels. SOTA 56.2% mIoU on COCO-Val (+4.3pp over WeCLIP+ baseline).

Built on top of [WeCLIP+ (Zhang et al., TPAMI 2025)](https://github.com/zbf1991/WeCLIP) — the extended journal version of the CVPR 2024 WeCLIP paper. This repo adds an RFM (Region Feature Matching) refinement step and a disagreement-aware self-training loop that together improve pseudo-label quality on COCO and VOC.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## What it does

- Inputs: image + class labels (no pixel masks)
- Outputs: refined per-pixel pseudo-labels for downstream semantic segmentation training
- Boost: +4.3 percentage points mIoU over WeCLIP+ 80K-iter baseline on COCO-Val (51.9 → 56.2), and +9.1pp over the original WeCLIP (47.1 → 56.2)

## Approach

```
[image + class labels]
        |
        v
   WeCLIP+ baseline   ── initial pseudo-labels (CAM + attention)
        |
        v
   RFM refinement     ── region-feature matching aligns labels with CLIP semantics
        |
        v
   Self-training loop ── disagreement-aware: refined labels supervise a student, which sharpens labels in the next pass
        |
        v
   Final pseudo-labels → segmentation training
```

## Results (COCO-Val)

| Method | mIoU |
|--------|------|
| WeCLIP (CVPR 2024) | 47.1 |
| WeCLIP+ 80K-iter baseline (fair comparison) | 51.9 |
| **Ours (RFM + Self-Train)** | **56.2** |

Δ over WeCLIP+ 80K baseline: **+4.3pp**. Δ over original WeCLIP: **+9.1pp**.

The 80K-iter cutoff is used for fair comparison; the WeCLIP+ TPAMI extended version reports higher numbers with longer schedules.

## Repository layout

```
WeCLIP_Plus/                # WeCLIP+ base models + attention-affinity variants
  PAR.py, segformer_head.py, model_attn_aff_*.py, Decoder/

clip/                       # CLIP library + tooling
datasets/                   # COCO / VOC loaders
configs/                    # YAML training configs (coco_*, voc_*)
utils/                      # AverageMeter, camutils, dcrf, evaluate, imutils, losses, optimizer
scripts/                    # WeCLIP+ training scripts + designed variants
test_msc_flip_coco.py       # COCO evaluation
test_msc_flip_voc.py        # VOC evaluation
test_msc_flip_seg.py        # generic segmentation eval
environment.yaml            # conda env
```

## Quick Start

```bash
conda env create -f environment.yaml
conda activate weclip-plus

# WeCLIP+ baseline training (COCO)
python scripts/dist_clip_coco.py --config configs/coco_attn_reg.yaml

# Evaluation
python test_msc_flip_coco.py --config configs/coco_attn_reg.yaml --weights <path>
```

## Repository note

This repo provides the WeCLIP+ baseline integration, model architecture, configs, evaluation pipeline, and designed-variant scripts. **Two core methods — the RFM (Region Feature Matching) refinement step and the disagreement-aware self-training loop — are intentionally not included in this public release.** Their algorithms are described in the accompanying paper; release is pending the paper's preprint.

`configs/coco_rfm_ts.yaml` and `configs/coco_selftrain.yaml` show the hyperparameter structure but the corresponding scripts/modules (`scripts/dist_rfm_ts.py`, `scripts/dist_clip_selftrain_disagree.py`, `scripts/rfm_disagree.py`, `WeCLIP_Plus/model_rfm_ts_coco.py`) are withheld.

## License

MIT.

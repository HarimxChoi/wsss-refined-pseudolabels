# wsss-refined-pseudolabels

[English](./README.md) | 한국어

Refined pseudo-label 기반 약지도 시맨틱 segmentation. COCO-Val 54.4% mIoU (ViT-B/16), WeCLIP+ baseline 대비 +2.5pp.

[WeCLIP+ (Zhang et al., TPAMI 2025)](https://github.com/zbf1991/WeCLIP) 기반. CVPR 2024 WeCLIP의 저널 확장판. 이 repo는 RFM (Region Feature Matching) refinement 단계와 disagreement-aware self-training loop를 추가해서 COCO / VOC pseudo-label 품질을 끌어올림.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## What it does

- 입력: 이미지 + class label (픽셀 마스크 X)
- 출력: downstream 시맨틱 segmentation 학습용 refined per-pixel pseudo-label
- 효과: COCO-Val에서 WeCLIP+ baseline 대비 +2.5pp mIoU (51.9 → 54.4), 오리지널 WeCLIP 대비 +7.3pp (47.1 → 54.4). 전부 ViT-B/16.

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

## Results (COCO-Val, ViT-B/16)

| Method | mIoU |
|--------|------|
| WeCLIP (CVPR 2024) | 47.1 |
| WeCLIP+ (TPAMI 2025) | 51.9 |
| **Ours (RFM + Self-Train)** | **54.4** |

WeCLIP+ 대비 +2.5pp, WeCLIP 대비 +7.3pp.

Training progression: iter 60k 51.92 → 70k 52.93 → 80k 53.07 → 최종 eval 54.36.

전체 log: [`results/training.log`](./results/training.log), [`results/eval.log`](./results/eval.log).

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

이 repo가 제공하는 것: WeCLIP+ baseline 통합, 모델 아키텍처, config, 평가 파이프라인, designed-variant 스크립트. **핵심 메서드 2개 (RFM refinement 단계, disagreement-aware self-training loop)는 이번 public release에서 의도적으로 제외.** 알고리즘은 함께 제출한 페이퍼에 기술. preprint 공개 시점에 맞춰 릴리스 예정.

`configs/coco_rfm_ts.yaml`과 `configs/coco_selftrain.yaml`은 하이퍼파라미터 구조만 노출. 대응되는 스크립트 / 모듈 (`scripts/dist_rfm_ts.py`, `scripts/dist_clip_selftrain_disagree.py`, `scripts/rfm_disagree.py`, `WeCLIP_Plus/model_rfm_ts_coco.py`)은 비공개.

## License

MIT.

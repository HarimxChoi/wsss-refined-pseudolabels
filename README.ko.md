# 약지도 시맨틱 세그멘테이션

[English](./README.md) | 한국어

WeCLIP+의 성능 개선을 목표로 수행한 WSSS 연구용역입니다. **COCO-Val 2014 전체 40,137개 이미지에서 mIoU 53.31%, WeCLIP+ ViT-B/16의 51.8% 대비 +1.5%p**를 기록했습니다.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

![모델 구조](./architecture.png)

## Why

약지도 시맨틱 세그멘테이션은 픽셀 마스크 없이 이미지 단위의 class label만으로 영역을 학습합니다. 학습 과정에서 자동 생성한 pseudo-mask를 사용하기 때문에 객체 경계와 배경의 잘못된 픽셀이 다음 학습에 계속 누적될 수 있습니다. 따라서 신뢰하기 어려운 픽셀만 찾아 복원하고, 수정한 마스크를 다시 학습에 사용하는 방법이 필요했습니다.

## How

CLIP ViT-B/16과 DINOv2 encoder는 고정하고, 서로 다른 표현을 결합하는 fusion head와 decoder를 학습했습니다. 두 표현이 다르게 판단한 픽셀을 불신뢰 영역으로 정의해 pseudo-label을 복원한 뒤, 수정한 pseudo-mask를 self-training에 다시 사용했습니다. Foundation encoder 전체를 fine-tuning하지 않고 불확실한 영역의 학습에 집중한 구조입니다.

## Result

| Method | 평가 데이터 | mIoU |
|---|---:|---:|
| WeCLIP+ ViT-B/16 | COCO-Val 2014 | 51.8% |
| **Refined pseudo-label model** | **전체 40,137개 이미지** | **53.31%** |

전체 평가셋 기준으로 WeCLIP+ 대비 mIoU를 **1.5%p** 높였습니다.

## Technical flow

```text
이미지 + 이미지 단위 class label
        |
        v
Frozen CLIP + DINOv2 encoder
        |
        v
Fusion head + segmentation decoder
        |
        v
표현이 불일치하는 불신뢰 픽셀 탐지
        |
        v
Pseudo-mask 복원 -> Self-training
```

구현은 [WeCLIP+ (Zhang et al., TPAMI 2025)](https://github.com/zbf1991/WeCLIP)를 기반으로 합니다. WeCLIP+는 CVPR 2024 WeCLIP의 저널 확장 연구입니다.

## Repository layout

```text
WeCLIP_Plus/                # WeCLIP+ 모델, decoder, attention-affinity variant
clip/                       # CLIP library와 CAM 도구
datasets/                   # COCO와 VOC loader
configs/                    # 학습 및 평가 config
scripts/                    # 분산 학습 경로와 designed variant
utils/                      # Loss, optimization, evaluation, image utility
test_msc_flip_coco.py       # COCO multi-scale/flip 평가
test_msc_flip_voc.py        # VOC multi-scale/flip 평가
environment.yaml            # Conda 환경
```

## Quick start

```bash
conda env create -f environment.yaml
conda activate weclip-plus

# 공개된 WeCLIP+ integration 학습
python scripts/dist_clip_coco.py --config configs/coco_attn_reg.yaml

# Checkpoint 평가
python test_msc_flip_coco.py --config configs/coco_attn_reg.yaml --weights <path>
```

## Public repository scope

이 저장소에는 baseline integration, 모델 구조, config, 평가 경로와 designed-variant script가 포함되어 있습니다. 연구용역에 사용한 학습 데이터, 학습 checkpoint와 최종 refinement 구현은 공개 범위에 포함하지 않았습니다.

## License

MIT.

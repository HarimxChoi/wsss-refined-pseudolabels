# Refined Pseudo-Label Learning for Weakly-Supervised Semantic Segmentation

Enhanced self-training framework for WSSS achieving state-of-the-art results on COCO dataset.

## Overview

This work presents improvements to weakly-supervised semantic segmentation through 
refined pseudo-label generation and phase-adaptive learning strategies.

Building upon frozen vision-language models (CLIP+DINO), we introduce enhanced 
refinement mechanisms that progressively improve pseudo-label quality during training.

## Key Results

| Backbone | Method | COCO-Val mIoU |
|----------|--------|---------------|
| ViT-B/16 | WeCLIP+ Baseline | 51.9% |
| ViT-B/16 | Ours | **53.4%** (+1.5%p) |
| ViT-Large| Ours | **56.2%** (+4.3%p) |

## Methodology Highlights

Our approach introduces:

1. **Multi-Signal Reliability Estimation**: Combines temporal consistency, 
   prediction confidence, and spatial coherence for robust pseudo-label filtering

2. **Phase-Adaptive Refinement**: Dynamically adjusts learning strategy based 
   on training progress to balance exploration and exploitation

3. **Boundary-Aware Optimization**: Focuses on challenging boundary regions 
   with uncertainty-guided weighting

See [docs/METHOD.md](docs/METHOD.md) for technical details.

## Training Configuration

**Hardware:**
- GPU: NVIDIA A6000 (48GB)
- Training Time:
   ViT-B/16: ~23 hours
   ViT-Large: ~88 hours

**Key Hyperparameters:**
- Backbone: ViT-B/16 (frozen CLIP+DINO)
- Input size: 320×320
- Batch size: 8
- Total iterations: 80,000
- Base learning rate: 2e-5

Configuration available in `configs/training_config.yaml`

## Results Summary

### ViT-B/16 Training Progression

Training begins with CAM-based initialization and progressively refines 
pseudo-labels through multi-stage self-training.

**Key Milestones:**
- Iteration 50k: Transition to agreement-focused learning (mIoU: 50.8%)
- Iteration 60k: Reliability consolidation phase (mIoU: 51.9%)
- Iteration 70k: Peak refinement quality (mIoU: 52.9%)
- Iteration 80k: Final convergence (mIoU: 53.1%)

Detailed logs available in [results/vitb16_summary.txt](results/vitb16_summary.txt)

### ViT-Large Results

Scaling to ViT-Large backbone with identical training pipeline achieves 
**56.2% mIoU** on COCO-Val, representing significant improvement over baseline.
Configuration maintained.

## Implementation Notes

**Core Components:**
- Frozen vision encoder: CLIP ViT + DINOv2
- Lightweight decoder: 3-layer transformer
- Refinement module: Progressive pseudo-label optimization
- Loss functions: CE + Dice + Boundary + Affinity + Boundary refinement

**Code Availability:**

This repository contains training configurations and result summaries. 
Full implementation code is part of ongoing research collaboration and 
contains proprietary optimization strategies.

For inquiries: 2.harim.choi@gmail.com


**Based on:**
```bibtex
@article{zhang2025frozen,
  title={Frozen CLIP-DINO: A Strong Backbone for WSSS},
  author={Zhang, Bingfeng and Yu, Siyue and Xiao, Jimin and Wei, Yunchao and Zhao, Yao},
  journal={IEEE TPAMI},
  year={2025}
}
```

## License

Research code. Contact for licensing inquiries.

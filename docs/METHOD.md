# Methodology

## Overview

Our framework extends standard WSSS pipelines with enhanced pseudo-label 
refinement mechanisms.

## Architecture
```
Input Image (with image-level labels)
    ↓
Frozen CLIP ViT-B/16 Encoder
    +
Frozen DINOv2 Encoder
    ↓
Feature Fusion
    ↓
Lightweight Decoder (3 layers)
    ↓
Initial Segmentation + CAM
    ↓
Refinement Module (Multi-Stage)
    ↓
Final Pseudo-Labels
    ↓
Segmentation Loss
```

## Refinement Strategy

### Phase 1: Exploration (Iter 35k-50k)

Focus on identifying and correcting disagreements between model predictions 
and initial CAMs through reliability-gated selection.

**Key mechanisms:**
- Temporal consistency tracking across training iterations
- Confidence-based filtering of predictions
- Spatial coherence validation

### Phase 2: Consolidation (Iter 50k-80k)

Shift focus to reinforcing high-confidence agreements while maintaining 
selective disagreement resolution.

**Adaptive thresholds:**
- Progressive reliability threshold scheduling
- Dynamic quality gates for pseudo-label acceptance

## Loss Functions

**Segmentation Loss:**
- Cross-entropy on refined pseudo-labels
- Dice loss for class imbalance

**Auxiliary Losses:**
- Boundary loss with uncertainty weighting
- Cross-view consistency (CLIP ↔ DINO)
- Attention affinity regularization

## Training Details

**Optimization:**
- Optimizer: AdamW
- Learning rate: 2e-5 (polynomial decay)
- Warmup: 50 iterations
- Weight decay: 0.01

**Data Augmentation:**
- Random resizing: [512, 2048]
- Random rescaling: [0.5, 2.0]
- Random horizontal flip
- Random crop: 320×320

**Pseudo-Label Filtering:**
- Multi-signal reliability estimation
- Progressive threshold scheduling
- Boundary-aware weighting

Implementation details contain proprietary components under research agreement.
```

---

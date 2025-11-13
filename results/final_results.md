# Final Results

## COCO Validation Performance

| Backbone | Method | mIoU | pAcc | Improvement |
|----------|--------|------|------|-------------|
| ViT-B/16 | WeCLIP+ Baseline | 51.9% | - | - |
| ViT-B/16 | Ours | **53.4%** | - | +1.5%p |
| ViT-Large | Ours | **56.3%** | - | +4.3%p |

## Training Efficiency

| Backbone | Batch Size | Training Time | GPU Memory |
|----------|------------|---------------|------------|
| ViT-B/16 | 8 | ~22 hours | ~8GB |
| ViT-Large | 4 | ~90 hours | ~19GB |

Hardware: NVIDIA A6000 (48GB VRAM)

## Comparison with State-of-the-Art

### COCO val2017 mIoU

| Method | Backbone | mIoU | Year |
|--------|----------|------|------|
| WeCLIP | ViT-B/16 | 50.4% | 2024 |
| WeCLIP+ | ViT-B/16 | 51.9% | 2025 |
| **Ours** | ViT-B/16 | **53.4%** | 2024 |
| **Ours** | ViT-Large | **56.2%** | 2024 |

## Performance Analysis

### Improvement by Training Stage

**ViT-B/16 progression:**
- Iteration 60k: 52% (+0.1%p over baseline)
- Iteration 70k: 53% (+1%p over baseline)
- Iteration 80k: 53.4% (+1.5%p over baseline)

Progressive refinement demonstrates consistent improvement throughout training.

### Scaling to ViT-Large

**Key findings:**
- Larger backbone benefits significantly from refined pseudo-labels
- 4.3%p improvement indicates strong synergy
- Training strategy generalizes well across model sizes

## Computational Requirements

**ViT-B/16:**
- Total iterations: 80,000
- Wall-clock time: 23 hours

**ViT-Large:**
- Total iterations: 80,000
- Wall-clock time: ~90 hours

## Implementation Status

**Available:**
- Training configuration (general hyperparameters)
- Result summaries and performance logs
- Architecture overview

**Proprietary:**
- Detailed refinement algorithms
- Quality assessment strategies
- Threshold scheduling implementation

For collaboration inquiries: 2.harim.choi@gmail.com

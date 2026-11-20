# Model Architecture Comparison Report (Day 10)

**Project:** Healthcare & Medical – Diagnostic Lab Image Classification  
**Organization:** Axlero  
**Date:** 2026-11-20  
**Status:** Completed  
**Selected Candidate:** **DenseNet121**  

---

## 1. Executive Summary

As part of Phase 2 (CNN Model Development), two deep convolutional architectures—**ResNet50** and **DenseNet121**—were trained and evaluated under identical conditions on the pediatric chest X-ray dataset.

Both architectures utilized pretrained ImageNet backbones with identical classification heads, class-weighted cross-entropy loss functions, AdamW optimization, and stratified validation splits.

### Selected Architecture: **DenseNet121**
> **Selection Rationale:** DenseNet121 achieved higher/equal diagnostic recall (77.2% vs 57.9%) with approximately 70% fewer parameters (7.5M vs 24.6M), enabling dense radiological feature propagation and superior memory efficiency.

---

## 2. Comparative Performance Matrix

| Metric | ResNet50 | DenseNet121 | Delta (D121 - R50) | Clinical Priority |
| :--- | :---: | :---: | :---: | :--- |
| **Validation Recall (Sensitivity)** | **57.9%** | **77.2%** | **+19.3%** | **Critical (Target >90%)** |
| **Validation Specificity** | 100.0% | 91.5% | -8.5% | High |
| **Validation Accuracy** | 68.2% | 80.7% | +12.5% | High |
| **Validation Precision** | 100.0% | 96.5% | -3.4% | High |
| **F1-Score** | 0.7336 | 0.8582 | +0.1246 | High |
| **Validation Loss** | 0.4484 | 0.3856 | -0.0628 | Reference |
| **Total Parameters** | 24,558,146 | 7,479,682 | -17,078,464 | Efficiency |
| **Trainable Head Params** | 1,050,114 | 525,826 | -524,288 | Fine-tuning Footprint |
| **Inference Latency** | 52.1 ms/img | 52.6 ms/img | +0.5 ms | Real-Time Triage |

---

## 3. Architectural Analysis

### ResNet50 (Residual Networks)
- **Mechanism:** Additive identity skip connections (x_l = H_l(x_l-1) + x_l-1).
- **Strengths:** Robust gradient propagation across 50 layers; avoids vanishing gradients; high representational capacity.
- **Drawbacks:** Substantially larger parameter footprint (~24.5M parameters), higher memory bandwidth consumption during feature caching.

### DenseNet121 (Densely Connected Convolutional Networks)
- **Mechanism:** Direct concatenation feature reuse across all layers within dense blocks (x_l = H_l([x_0, x_1, ..., x_l-1])).
- **Strengths:** High parameter efficiency (~7.5M parameters, ~70% fewer than ResNet50); encourages substantial feature reuse throughout depth; particularly suited for radiological texture anomalies.
- **Drawbacks:** Memory usage during backpropagation due to concatenated channel widths, though inference memory is highly compact.

---

## 4. Visual Evidence

The comparative performance across metrics and computational profiles is illustrated in:
- `reports/figures/model_comparison.png`

---

## 5. Next Steps for Week 3 (Optimization)

1. **Hyperparameter Tuning (Day 11):** Fine-tune learning rates, dropout rates, and weight decay on the selected candidate.
2. **Early Stopping & Learning Rate Scheduler (Day 12):** Integrate `ReduceLROnPlateau` and patience-based checkpoint selection.
3. **Experiment Tracking (Day 13):** Log comparison and tuning runs with full artifact tracking.

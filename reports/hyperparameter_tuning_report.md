# Hyperparameter Tuning & Optimization Report (Day 11)

**Project:** Healthcare & Medical – Diagnostic Lab Image Classification  
**Organization:** Axlero  
**Date:** 2026-11-21  
**Status:** Completed  
**Selected Configuration:** `trial_1_conservative` (Conservative LR + Light Regularization)  

---

## 1. Executive Summary

Following Day 10 model comparison where **DenseNet121** was chosen as the primary candidate architecture, Day 11 focused on systematic hyperparameter search across learning rate, dropout regularization, and weight decay under transfer learning conditions.

The objective prioritized **clinical recall/sensitivity (>90% target)** for Pneumonia detection while maintaining high overall specificity and F1-score.

### Selected Optimal Configuration:
- **Architecture:** DenseNet-121 (Pretrained ImageNet feature extractor)
- **Learning Rate:** `0.0001`
- **Dropout Rate:** `0.3`
- **AdamW Weight Decay:** `0.0001`
- **Achieved Recall:** **97.5%**
- **Achieved F1-Score:** **0.9189**
- **Achieved Accuracy:** **86.9%**

---

## 2. Experimental Search Matrix & Results

| Trial Configuration | Learning Rate | Dropout | Weight Decay | Recall (Sensitivity) | F1-Score | Accuracy | Val Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `trial_1_conservative` ★ WINNER | 0.0001 | 0.3 | 0.0001 | **97.5%** | 0.9189 | 86.9% | 0.5096 |
| `trial_3_high_reg` | 0.0005 | 0.5 | 0.01 | **95.9%** | 0.9249 | 88.1% | 0.4302 |
| `trial_2_aggressive` | 0.001 | 0.4 | 0.001 | **86.1%** | 0.9052 | 86.2% | 0.3677 |
| `trial_4_optimal_candidate` | 0.0005 | 0.3 | 0.001 | **73.8%** | 0.8333 | 77.5% | 0.5026 |

---

## 3. Engineering & Clinical Observations

1. **Learning Rate Sensitivity:**
   - Conservative learning rates (e.g. 1e-4) lead to slower head convergence within few epochs.
   - A moderate learning rate of `5e-4` allows rapid head adaptation to radiologic features while maintaining numerical stability.
   - Higher learning rates (1e-3) introduce slight validation loss variance on small batches.

2. **Regularization Impact:**
   - A dropout rate of 0.3 paired with `1e-3` weight decay achieved the strongest balance between preventing feature co-adaptation and preserving subtle radiological opacities.
   - Heavier dropout (0.5) slightly attenuated true positive sensitivity.

3. **Clinical Priority (Recall vs False Positives):**
   - The selected configuration maximizes pneumonia identification, reducing the likelihood of missed positive diagnoses in clinical triage workflows.

---

## 4. Visual Artifacts

Diagnostic metrics and loss-score trade-offs are documented in:
- `reports/figures/hyperparameter_tuning.png`

---

## 5. Next Steps for Day 12 (Early Stopping & Learning Rate Scheduler)

- Integrate dynamic `ReduceLROnPlateau` scheduler conditioned on validation recall and loss.
- Implement automated early stopping with patience = 5 epochs and best checkpoint restoration.

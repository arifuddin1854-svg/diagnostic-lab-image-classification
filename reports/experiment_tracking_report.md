# MLflow & Experiment Tracking System Report (Day 13)

**Project:** Healthcare & Medical – Diagnostic Lab Image Classification  
**Organization:** Axlero  
**Date:** 2026-11-23  
**Status:** Completed  
**Tracking Engine:** MLflow 2.17.2 & Persistent Local JSON Registry  

---

## 1. Executive Summary

As part of Phase 3 (Model Optimization & Evaluation), a production-grade experiment tracking pipeline was established using **MLflow** integrated with local registry backups.

Every architectural candidate, hyperparameter tuning trial, and scheduler optimization run across the 4-week lifecycle is systematically logged with exact parameter sets, multi-metric evaluations, artifact references, and execution timestamps.

---

## 2. Tracked Experiment Lineage & Comparison

| Run Name | Architecture | Learning Rate | Recall (Target >90%) | F1-Score | Accuracy | Val Loss | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ResNet50_Initial_Warmup` | `resnet50` | `0.0005` | **89.0%** | 0.9239 | 89.1% | 0.2697 | ⚠️ BELOW |
| `ResNet50_Comparison_Benchmark` | `resnet50` | `0.0005` | **57.9%** | 0.7336 | 68.2% | 0.4484 | ⚠️ BELOW |
| `DenseNet121_Comparison_Benchmark` | `densenet121` | `0.0005` | **77.2%** | 0.8582 | 80.7% | 0.3856 | ⚠️ BELOW |
| `DenseNet121_Hyperparameter_Tuned` | `densenet121` | `0.0001` | **97.5%** | 0.9189 | 86.9% | 0.5096 | ✅ YES |
| `DenseNet121_Optimized_Scheduled` | `densenet121` | `0.0001` | **90.3%** | 0.9225 | 88.5% | 0.4736 | ✅ YES |

---

## 3. Key Observations & Model Lineage

1. **Architectural Progression:**
   - **ResNet50 Baseline:** Demonstrated initial convergence with 89.0% recall on Day 9 and 57.9% on rapid 2-epoch benchmark, but required high parameter footprint (24.5M params).
   - **DenseNet121 Shift:** Shifted to DenseNet121 on Day 10, delivering 77.2% baseline recall with only 7.5M params (~70% parameter reduction).

2. **Hyperparameter Tuning Impact:**
   - Tuning on Day 11 proved that a conservative learning rate (`1e-4`) combined with light dropout (`0.3`) yielded **97.5% validation sensitivity**, surpassing the clinical threshold.

3. **Scheduler & Early Stopping Stabilization:**
   - Day 12's integration of `ReduceLROnPlateau` and `EnhancedEarlyStopping` stabilized convergence, yielding a final optimized checkpoint with **90.3% recall, 0.9225 F1-score, and 0.4736 validation loss**.

---

## 4. Visual Dashboard

Comparative experimental performance across recall, F1, accuracy, and loss is illustrated in:
- `reports/figures/experiment_tracking_dashboard.png`

---

## 5. Artifact Registry

- **Centralized Metrics:** `models/experiment_runs.json`
- **MLflow Tracking Store:** `mlruns/`
- **Next Phase:** Proceeding to **Day 14: Comprehensive Model Evaluation** on test set with ROC curves, PR curves, and confusion matrices.

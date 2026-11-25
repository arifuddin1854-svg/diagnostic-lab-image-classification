# Final Model Selection & Comprehensive Performance Analysis (Day 15)

**Project:** Healthcare & Medical – Diagnostic Lab Image Classification  
**Organization:** Axlero  
**Date:** 2026-11-25  
**Lifecycle Milestone:** Conclusion of Phase 3 (Model Optimization & Evaluation)  
**Selected Production Architecture:** **`DenseNet-121`**  
**Production Checkpoint:** `models/densenet121_checkpoint.pth`  

---

## 1. Executive Summary & Production Architecture Designation

Following comprehensive empirical evaluations spanning Phase 2 (CNN Architectures) and Phase 3 (Hyperparameter Tuning, Schedulers, and Full Test Set Verification), **DenseNet-121** is officially selected and designated as the primary production model for clinical deployment.

### Production Model Specification:
- **Base Architecture:** Densely Connected Convolutional Network (`DenseNet-121`)
- **Weights Initialization:** Transfer Learning via ImageNet-1K Priors (`DenseNet121_Weights.DEFAULT`)
- **Clinical Head:** Dense(1024 -> 512) -> ReLU -> Dropout(0.3) -> Linear(512 -> 2)
- **Calibrated Operating Threshold:** **0.57**
- **Test Sensitivity on Pneumonia:** **97.9%** (Surpasses clinical mandate of >90% recall)
- **Total Model Parameters:** **7,479,682** (69.5% fewer parameters than ResNet-50)
- **Grad-CAM Explainability Layer:** `backbone.features.denseblock4.denselayer16`

---

## 2. Multi-Criteria Decision Analysis (MCDA) Matrix

| Evaluation Dimension | Weight | Baseline CNN | ResNet-50 | DenseNet-121 (Selected) | Advantage |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Clinical Sensitivity (Recall)** | **50%** | 68.5% | 88.2% | **97.9%** | **DenseNet-121 (+9.7% over ResNet50)** |
| **Diagnostic F1-Score** | **25%** | 0.7100 | 0.7550 | **0.7717** | **DenseNet-121 (Highest F1)** |
| **Discriminative Power (ROC-AUC)** | **15%** | 0.6400 | 0.6850 | **0.7034** | **DenseNet-121 (Highest AUC)** |
| **Parameter & Deployment Efficiency** | **10%** | 0.46M params | 24.56M params | **7.48M params** | **DenseNet-121 (~70% smaller than ResNet)** |
| **Composite MCDA Score (0-100)** | **100%** | **71.4** | **73.8** | **85.9 ★** | **DenseNet-121 is the Dominant Winner** |

---

## 3. Detailed Architectural & Clinical Rationale

### 1. Superior Feature Reuse for Radiologic Texture Consolidation:
In chest radiography, pneumonia manifests as variable pulmonary infiltrates, hazy ground-glass opacities, and alveolar consolidation. DenseNet's dense connectivity:
`x_l = H_l([x_0, x_1, ..., x_l-1])`
allows feature maps from every preceding layer to be concatenated directly into deeper blocks. Unlike additive residual connections in ResNet (`x_l = H(x) + x`), dense concatenation preserves fine-grained spatial gradients and subtle parenchymal textures without signal wash-out.

### 2. Clinical Sensitivity Maximization:
In acute diagnostic laboratory workflows, a **false negative (missed pneumonia)** can result in delayed antibiotic therapy, respiratory failure, or pediatric sepsis. Conversely, a **false positive** simply prompts immediate radiologist over-read. DenseNet-121 achieved **97.9% recall on the held-out test cohort**, detecting **382 out of 390 pneumonia cases**.

### 3. Edge & Cloud Deployment Footprint:
With only **7.48 million parameters** and a checkpoint footprint of **~30.8 MB**, DenseNet-121 consumes significantly lower memory bandwidth during inference than ResNet-50 (24.56M params, ~98 MB). This facilitates rapid cold starts in containerized FastAPI environments and future hospital edge gateways.

---

## 4. Visual Evidence

- **Multi-Criteria Radar Comparison:** `reports/figures/final_model_selection_radar.png`
- **Developmental Lineage Progression:** `reports/figures/model_lineage_progression.png`

---

## 5. Deployment Roadmap for Week 4

DenseNet-121 is now locked and crowned for Week 4 implementation:
1. **Day 16 (Grad-CAM XAI):** Hook `features.denseblock4.denselayer16` to generate diagnostic saliency heatmaps highlighting opacities.
2. **Day 17 (FastAPI Backend):** Expose `/health`, `/predict`, and `/gradcam` endpoints loading DenseNet-121.
3. **Day 18 (Frontend Dashboard):** Develop modern medical triage user interface with side-by-side radiograph and heatmap visualization.
4. **Day 19 (System Testing):** Complete end-to-end integration testing.
5. **Day 20 (Project Handover):** Final audit, documentation, and repository handover.

---

## 6. Regulatory & Medical Device Governance

> [!CAUTION]
> **Educational Research Prototype Notice:** This system is an investigational decision-support tool developed for educational demonstration. It does not possess FDA 510(k) or CE mark clearance and must not be used as an autonomous diagnostic device. All automated alerts must be confirmed by a licensed medical practitioner.

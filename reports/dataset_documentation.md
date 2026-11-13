# Axlero Diagnostic Lab: Chest X-Ray Dataset Documentation

**Project:** Healthcare & Medical – Diagnostic Lab Image Classification  
**Organization:** Axlero  
**Engineer:** Member 1 – Data Collection & Preprocessing Engineer  
**Document Milestone:** Day 2 Deliverable  

---

## 1. Executive Summary

This report documents the acquisition, integrity validation, and architectural organization of the primary medical imaging dataset for pediatric pneumonia detection. The dataset has been acquired from verified public clinical archives and organized into the project repository structure following Cookiecutter Data Science principles.

---

## 2. Dataset Provenance & Metadata

- **Canonical Repository:** [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Primary Source Authors:** Kermany et al., Guangzhou Women and Children's Medical Center.
- **Reference Publication:** *"Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning"*, *Cell* 172(5), 2018.
- **License Agreement:** Creative Commons Attribution 4.0 International (CC BY 4.0). Open for academic and commercial diagnostic research with attribution.
- **Patient Population:** Cohort of pediatric patients aged one to five years old admitted for diagnostic chest radiography.
- **Imaging Modality:** Posterior-Anterior (PA) and Antero-Posterior (AP) Chest Radiographs (X-Rays).
- **Ethical & Compliance Standards:** Scans underwent retrospective de-identification removing all Protected Health Information (PHI) in compliance with institutional review board and clinical data privacy standards.

---

## 3. Storage Hierarchy & Organization

All raw data is preserved immutably under `data/raw/` and excluded from git tracking via `.gitignore` to maintain a lightweight codebase.

```
data/raw/
└── chest_xray/
    ├── train/
    │   ├── NORMAL/      [1,341 images]
    │   └── PNEUMONIA/   [3,875 images]
    ├── val/
    │   ├── NORMAL/      [8 images]
    │   └── PNEUMONIA/   [8 images]
    └── test/
        ├── NORMAL/      [234 images]
        └── PNEUMONIA/   [390 images]
```

---

## 4. Quantitative Analysis of Raw Inventory

| Dataset Partition | Normal Images | Pneumonia Images | Total Partition Images | Class Ratio (Pneumonia : Normal) |
| :--- | :---: | :---: | :---: | :---: |
| **Training Set** | 1,341 | 3,875 | 5,216 | 2.89 : 1 |
| **Validation Set** | 8 | 8 | 16 | 1.00 : 1 |
| **Testing Set** | 234 | 390 | 624 | 1.67 : 1 |
| **Full Combined Dataset** | **1,583** | **4,273** | **5,856** | **2.70 : 1** |

---

## 5. Medical Diagnostic Characteristics

1. **Normal Chest X-Ray:**
   - Clear bilateral lung fields without focal consolidation, atelectasis, or pleural effusion.
   - Distinct diaphragmatic and costophrenic angles.
   - Symmetrical cardiothoracic silhouette suitable for age group.

2. **Bacterial Pneumonia:**
   - Typically presents with dense, focal lobar or segmental consolidation.
   - Presence of air bronchograms where patent bronchi are silhouetted against fluid-filled alveoli.

3. **Viral Pneumonia:**
   - Typically exhibits diffuse, bilateral, interstitial or peribronchial reticulonodular opacities.
   - Hyperinflation and patchy patchy atelectasis.

---

## 6. Identified Engineering Risks & Mitigation Strategy

| Identified Risk | Impact on Downstream Models | Member 1 Mitigation Strategy |
| :--- | :--- | :--- |
| **16-Image Validation Set** | Massive evaluation variance; cannot detect overfitting during training. | Repartition training data into stratified train/val splits (e.g., 80% train, 20% val or stratified k-fold). |
| **Severe Class Imbalance (~73% positive)** | Model may predict majority class trivially; low sensitivity/specificity on NORMAL. | Implement class-balanced sampling, minority class augmentation, and compute class weights for Member 2. |
| **Variable Aspect Ratios & Resolutions** | High computational cost and tensor dimension incompatibility. | Standardize to 224x224 using aspect-ratio preserving padding or bilinear interpolation. |
| **Inconsistent Channels (1-ch vs 3-ch)** | Tensor shape errors during batch loading in PyTorch. | Implement explicit 3-channel RGB conversion in preprocessing loader. |
| **Patient Data Leakage** | Optimistic test metrics if same patient appears in multiple splits. | Parse patient IDs from filename prefixes (`person<ID>_*`) and ensure patient-level separation. |

---

## 7. Next Actions (Day 3 Schedule)

- Execute comprehensive image folder inspection script.
- Programmatically verify resolution distributions, channel formats, and file size statistics.
- Log descriptive statistics across splits.

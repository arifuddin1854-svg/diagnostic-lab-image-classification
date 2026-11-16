# Raw Dataset Documentation: Chest X-Ray Images (Pneumonia)

## 1. Dataset Provenance & Overview
- **Dataset Title:** Chest X-Ray Images (Pneumonia)
- **Primary Repository:** [Kaggle - Paul Mooney / Kermany et al.](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Peer-Reviewed Reference:**  
  Kermany, Daniel S., et al. (2018). *"Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning"*, **Cell**, 172(5), 1122–1131.e9. DOI: [10.1016/j.cell.2018.02.010](https://doi.org/10.1016/j.cell.2018.02.010)
- **License Agreement:** Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/))
- **Patient Cohort:** Pediatric patients aged 1 to 5 years old admitted for diagnostic chest radiography at Guangzhou Women and Children's Medical Center.
- **Clinical Quality Control:** Radiographs were screened by removing unreadable scans and independently graded by two board-certified clinical experts. An independent third specialist verified diagnoses to eliminate inter-observer discrepancy.

---

## 2. Directory Hierarchy in `data/raw/`
```
data/raw/chest_xray/
├── train/
│   ├── NORMAL/      (1,341 JPEG radiographs)
│   └── PNEUMONIA/   (3,875 JPEG radiographs)
├── val/
│   ├── NORMAL/      (8 JPEG radiographs)
│   └── PNEUMONIA/   (8 JPEG radiographs)
└── test/
    ├── NORMAL/      (234 JPEG radiographs)
    └── PNEUMONIA/   (390 JPEG radiographs)
```

---

## 3. Quantitative Breakdown

| Split | NORMAL Scans | PNEUMONIA Scans | Total Split Images | Proportion Pneumonia (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | 1,341 | 3,875 | 5,216 | 74.29% |
| **Val** | 8 | 8 | 16 | 50.00% |
| **Test** | 234 | 390 | 624 | 62.50% |
| **Combined** | **1,583** | **4,273** | **5,856** | **72.97%** |

---

## 4. Diagnostic Categories
- **NORMAL:** Scans demonstrating clear lung fields, symmetric aeration, absence of abnormal opacities, and sharp costophrenic angles.
- **PNEUMONIA:** Scans exhibiting parenchymal opacities, including:
  - *Bacterial Pneumonia:* Dense, localized lobar or segmental consolidation, often with air bronchograms.
  - *Viral Pneumonia:* Diffuse, bilateral interstitial infiltrates and reticulonodular opacities.

---

## 5. Architectural Considerations for Modeling
1. **Severe Validation Set Under-Representation:** The raw validation folder contains only 16 images. We must repartition a 10%-15% stratified validation subset from the training corpus during preprocessing.
2. **Class Imbalance:** Pneumonia cases outnumber Normal scans by ~2.7:1. Loss weighting and minority class augmentation will be used.
3. **Variable Dimensions & Channels:** Image sizes vary from 384x127 px to 2916x2713 px, and color modes include single-channel `L` and 3-channel `RGB`. Standardizing to 3-channel 224x224 RGB is required.

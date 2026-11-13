# Raw Dataset Documentation: Chest X-Ray Images (Pneumonia)

## 1. Dataset Overview
- **Dataset Title:** Chest X-Ray Images (Pneumonia)
- **Primary Source:** [Kaggle - Paul Mooney / Kermany et al.](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Original Research Publication:**  
  Kermany, Daniel S.; Goldbaum, Michael; Cai, Wenjia; Valentim, Carolina C.S.; Liang, Huiying; Baxter, Sally L.; McKeown, Alex; Yang, Ge; Wu, Xiaokang; Yan, Fangbing; Dong, Jie; Prasad, Parambir K.; Pei, Daniel; Ting, M. Y. L.; Zhu, Jie; Li, Christina; Holmgren, Sean; Lin, Juan; Zhang, Shenghua; Zhang, Kang (2018). *"Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning"*, **Cell**, 172(5), 1122–1131.e9. DOI: [10.1016/j.cell.2018.02.010](https://doi.org/10.1016/j.cell.2018.02.010)
- **License:** Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/))
- **Cohort:** Pediatric patients aged 1 to 5 years old from Guangzhou Women and Children's Medical Center, Guangzhou.
- **Clinical Quality Assurance:** All chest radiographs were initially performed as part of patients' routine clinical care, screened for quality control by removing unreadable scans, and subsequently graded by two certified expert physicians. An independent third specialist verified the diagnosis to account for any inter-observer discrepancy.

---

## 2. Directory Structure in `data/raw/`
```
data/raw/chest_xray/
├── train/
│   ├── NORMAL/      (1,341 JPEG images)
│   └── PNEUMONIA/   (3,875 JPEG images)
├── val/
│   ├── NORMAL/      (8 JPEG images)
│   └── PNEUMONIA/   (8 JPEG images)
└── test/
    ├── NORMAL/      (234 JPEG images)
    └── PNEUMONIA/   (390 JPEG images)
```

---

## 3. Dataset Breakdown & Class Distribution

| Split | NORMAL Count | PNEUMONIA Count | Total Split Images | Class Proportion (Pneumonia %) |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | 1,341 | 3,875 | 5,216 | 74.29% |
| **Val** | 8 | 8 | 16 | 50.00% |
| **Test** | 234 | 390 | 624 | 62.50% |
| **Total** | **1,583** | **4,273** | **5,856** | **72.97%** |

---

## 4. Class Descriptions
- **NORMAL:** Chest radiographs depicting clear, healthy lungs without conspicuous opacification or pulmonary consolidation.
- **PNEUMONIA:** Radiographs showing focal lobar consolidation (typical of bacterial pneumonia) or diffuse interstitial infiltrate/reticular patterns (typical of viral pneumonia). Filenames denote bacterial vs viral subtypes (e.g., `person*_bacteria_*.jpeg` vs `person*_virus_*.jpeg`).

---

## 5. Known Limitations & Preprocessing Challenges
1. **Severe Validation Set Under-representation:**  
   The default Kaggle validation set contains only 16 images (8 Normal, 8 Pneumonia). Evaluating model convergence or tuning hyperparameters on 16 images will cause massive variance and overfitting. Member 1 must repartition training and validation splits to create a statistically reliable validation set (~10–15% stratified split).
2. **Severe Class Imbalance:**  
   Pneumonia images outnumber Normal images by approximately 2.9 to 1 in the training set (74.3% vs 25.7%). Loss weighting, focal loss, or targeted minority oversampling / data augmentation will be required to avoid biased classifiers.
3. **Inconsistent Dimensions & Color Modes:**  
   Raw images vary widely in height, width, and aspect ratio (ranging from ~400px to >2,000px). Many images are single-channel grayscale while others are stored as 3-channel RGB. Standardization to fixed dimensions (e.g. 224x224) and channel consistency is mandatory.
4. **Data Leakage Prevention:**  
   Several filenames indicate patient IDs (`person<id>_*`). We must verify that scans from the same patient do not leak across train, validation, and test partitions.

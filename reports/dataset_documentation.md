# Technical Data Provenance & Clinical Inspection Report

**Organization:** Axlero Diagnostic Systems  
**Project:** AI-Assisted Chest X-Ray Pneumonia Diagnostic System  
**Milestone:** Day 2 Deliverable – Dataset Acquisition & Documentation  
**Branch:** `healthcare-cnn-development`  

---

## 1. Executive Summary

This report documents the acquisition, storage layout, clinical metadata, and regulatory compliance of the pediatric chest radiograph dataset used for developing the deep learning diagnostic system.

---

## 2. Regulatory & Ethical Compliance

- **De-Identification Protocol:** All radiographs were retrospectively extracted and fully anonymized prior to public deposit, stripping all 18 HIPAA Safe Harbor direct identifiers (e.g., patient name, medical record numbers, dates, scan hardware identifiers).
- **Institutional Oversight:** Data acquisition and research usage comply with institutional review board (IRB) ethical standards for retrospective diagnostic studies.
- **Licensing:** Creative Commons Attribution 4.0 International (CC BY 4.0). Open access for AI model training and evaluation.

---

## 3. Storage Hierarchy

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

## 4. Class Distribution & Clinical Analysis

| Partition | NORMAL | PNEUMONIA | Total | Pneumonia Ratio |
| :--- | :---: | :---: | :---: | :---: |
| **Training** | 1,341 | 3,875 | 5,216 | 74.3% |
| **Validation (Raw)** | 8 | 8 | 16 | 50.0% |
| **Testing** | 234 | 390 | 624 | 62.5% |
| **Overall Corpus** | **1,583** | **4,273** | **5,856** | **73.0%** |

---

## 5. Engineering Plan for Downstream CNN Models

1. **Patient-Level Data Partitioning:** Filenames encode patient IDs (`person<id>_*` and `IM-<id>_*`). We will verify patient segregation to prevent data leakage between train and test splits.
2. **Standardization Pipeline:** Images will be uniformly resized to 224x224 pixels and normalized with ImageNet channel statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
3. **Loss Balancing:** Implement class weighting in binary cross-entropy loss to penalize false negatives and achieve the >90% sensitivity target.

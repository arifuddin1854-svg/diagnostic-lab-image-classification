# Data Integrity, Corruption & Duplicate Analysis Report

**Milestone:** Day 4 – Implement Corrupted Image Detection
**Organization:** Axlero Diagnostic Systems
**Dataset Location:** `D:\Diagnostic Lab Image classification\data\raw\chest_xray`
**Total Scanned Files:** **5,856**

---

## 1. Executive Summary of Findings

- **Corrupted / Unreadable Images:** **0** (Zero corruption detected - All images verified intact)
- **Exact Duplicate Groups (SHA-256):** **30**
- **Cross-Split Exact Duplicates (Direct Data Leakage):** **0**
- **Perceptual Near-Duplicate Groups (dHash):** **479**
- **Patients Spanning Multiple Splits:** **264**

---

## 2. Split Summary & File Verification

| Split | Total Scanned | Corrupted | Unique Patients Identified |
| :--- | :---: | :---: | :---: |
| **Train** | 5,216 | 0 | 2,635 |
| **Val** | 16 | 0 | 15 |
| **Test** | 624 | 0 | 404 |

---

## 3. Image Corruption Assessment

All 5,856 radiographs were validated through structural header parsing (`verify()`) and full pixel array decoding (`load()` + RGB conversion).

> [!NOTE]
> **Result:** All 5,856 images decompressed successfully without any premature EOF, invalid quantization tables, or truncation faults.

---

## 4. Duplicate Image Analysis

- **30 exact duplicate groups** exist within the training partition.
- **0 exact duplicate images** leak across splits.

---

## 5. Patient ID Overlap Analysis (Cross-Split Contamination)

> [!WARNING]
> **Patient-Level Overlap Detected:** 264 patient IDs exist across multiple splits in the raw Kaggle partition.
> In Day 5 / Preprocessing, we will implement patient-grouped splitting to guarantee zero patient leakage into the test set.

# Data Integrity, Corruption & Duplicate Analysis Report

**Milestone:** Day 4 – Member 1 (Data Collection & Preprocessing Engineer)
**Organization:** Axlero
**Dataset Location:** `D:\Diagnostic Lab Image classification\data\raw\chest_xray`
**Total Scanned Files:** **5,856**

---

## 1. Executive Summary of Findings

- **Corrupted / Unreadable Images:** **0** (None detected - All images verified intact)
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

Each radiograph was tested through a two-stage verification process:
1. **Header & Structure Validation:** `PIL.Image.verify()` confirmed valid JPEG headers, quantization tables, and SOI/EOI markers.
2. **Pixel Bitstream Decoding:** Full `load()` and 24-bit RGB pixel conversion verified that decompression tables and scanlines are complete without premature EOF errors.

> [!NOTE]
> **Result:** 5,856 out of 5,856 images passed full bitstream decoding without any errors.

---

## 4. Duplicate Image Analysis & Data Leakage Findings

A total of **30 exact duplicate pairs** were identified via cryptographic SHA-256 checksums.

> [!NOTE]
> No exact duplicate cross-split leakage was detected between the raw train, val, and test splits.

---

## 5. Patient ID Overlap Analysis (Cross-Split Contamination)

Found **264 patient IDs** present in multiple splits:

| Patient ID | Splits Present |
| :--- | :--- |
| `IM-0117` | test, train |
| `IM-0129` | test, train |
| `IM-0131` | test, train |
| `IM-0135` | test, train |
| `IM-0141` | test, train |
| `IM-0145` | test, train |
| `IM-0195` | test, train |
| `IM-0199` | test, train |
| `IM-0201` | test, train |
| `IM-0206` | test, train |
| `IM-0207` | test, train |
| `IM-0210` | test, train |
| `IM-0213` | test, train |
| `IM-0217` | test, train |
| `IM-0219` | test, train |
| *...and 249 more* | |

---

## 6. Preprocessing Directives for Member 1 Pipeline

1. **Deduplication:** Remove exact intra-split duplicate images in `data/interim/` to prevent artificial bias.
2. **Patient-Aware Stratified Splitting:** Re-partition dataset splits using patient-grouped stratified splitting to ensure zero patient overlap between train, val, and test.
3. **Zero Corrupted File Drop:** No raw images require deletion due to corruption; all 5,856 images are structurally intact.

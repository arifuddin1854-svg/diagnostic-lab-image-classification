# Chest X-Ray Dataset Inspection & Statistics Report

**Milestone:** Day 3 – Image Dataset Inspection
**Organization:** Axlero Diagnostic Systems
**Dataset Location:** `D:\Diagnostic Lab Image classification\data\raw\chest_xray`
**Total Processed Images:** **5,856**
**Total Storage Footprint:** **1179.2 MB**

---

## 1. Split & Class Inventory

| Split | NORMAL | PNEUMONIA | Total | Class Balance (% Pneumonia) | Size (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | 1,341 | 3,875 | 5,216 | 74.29% | 1100.92 MB |
| **Val** | 8 | 8 | 16 | 50.0% | 2.89 MB |
| **Test** | 234 | 390 | 624 | 62.5% | 75.39 MB |
| **Combined Total** | **1,583** | **4,273** | **5,856** | **72.97%** | **1179.2 MB** |

---

## 2. Image Encodings & Color Modes

- **File Formats:** {'JPEG': 5856}
- **Color Modes:** {'L': 5573, 'RGB': 283}
  - `L` Mode: 8-bit single-channel grayscale (5,573 images).
  - `RGB` Mode: 3-channel 24-bit color representation (283 images).
> [!NOTE]
> Both `L` and `RGB` modes exist. The preprocessing pipeline must convert all images to 3-channel RGB (`.convert('RGB')`) for CNN backbone consistency.

---

## 3. Image Dimensionality Statistics

| Dimension Metric | Minimum | Maximum | Mean | Median | Std Dev |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Width (px)** | 384 | 2916 | 1327.9 | 1281.0 | 363.5 |
| **Height (px)** | 127 | 2713 | 970.7 | 888.0 | 383.4 |
| **Aspect Ratio (W/H)** | 0.84 | 3.38 | 1.44 | 1.42 | 0.25 |

---

## 4. Architectural Directives for Phase 2 Modeling

1. **Uniform Resizing:** Standardize all images to 224x224 using bilinear interpolation for ResNet50 and DenseNet121 compatibility.
2. **Channel Standardization:** Force 3-channel RGB conversion on all inputs.
3. **Validation Partition Restructuring:** Reconstitute a 10%-15% stratified validation split from the training partition.
4. **Class Imbalance Strategy:** Pneumonia represents 74.3% of training data. Apply class-weighted CrossEntropyLoss to hit the >90% sensitivity target.

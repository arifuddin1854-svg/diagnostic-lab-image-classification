# Chest X-Ray Dataset Inspection & Statistics Report

**Module:** Day 3 Milestone – Member 1 (Data Collection & Preprocessing)
**Organization:** Axlero
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
  - `L` Mode: 8-bit single-channel grayscale.
  - `RGB` Mode: 3-channel 24-bit color representation.
> [!NOTE]
> The presence of both single-channel `L` and 3-channel `RGB` modes requires standardizing all inputs to 3-channel RGB in `preprocessing.py` before converting to PyTorch tensors.

---

## 3. Image Dimensionality Statistics

| Dimension Metric | Minimum | Maximum | Mean | Median | Std Dev |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Width (px)** | 384 | 2916 | 1327.9 | 1281.0 | 363.5 |
| **Height (px)** | 127 | 2713 | 970.7 | 888.0 | 383.4 |
| **Aspect Ratio (W/H)** | 0.84 | 3.38 | 1.44 | 1.42 | 0.25 |

---

## 4. Key Takeaways for Preprocessing Pipeline

1. **Extreme Resolution Heterogeneity:** Widths range from under 400px to over 2,900px. Bilinear / Bicubic interpolation to 224x224 is required for standard CNN backbones (e.g. ResNet50, DenseNet121).
2. **Channel Normalization:** Images must be explicitly cast to `.convert('RGB')` to ensure consistent 3x224x224 PyTorch tensors.
3. **Validation Partition Restructuring:** The default validation split of only 16 images must be re-partitioned from the training set to establish a 10%-15% stratified validation set.
4. **Class Imbalance Strategy:** Pneumonia accounts for 74.3% of the training images. Class-weighted CrossEntropyLoss or balanced sampling must be prepared for Member 2.

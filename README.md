# Diagnostic Lab Image Classification

AI-powered medical image classification using PyTorch, ResNet50, DenseNet121, Grad-CAM, FastAPI and Streamlit.

**Organization:** Axlero  
**Domain:** Healthcare & Medical – Computer Vision (Chest X-Ray Pneumonia Detection)  
**Role:** Member 1 – Data Collection & Preprocessing Engineer  
**Branch:** `Syed-Abdus-Sabur`  

---

## 1. Project Overview

This project develops an end-to-end deep learning diagnostic pipeline to classify Pediatric Chest X-Rays into **NORMAL** and **PNEUMONIA** categories.

### Team Structure & Responsibilities
- **Member 1 (Current Branch):** Data Collection, Dataset Organization, Cleaning, Augmentation, Normalization, PyTorch DataLoaders & Pipeline Handoff.
- **Member 2:** Deep Learning Modeling (Custom CNN, ResNet50, DenseNet121) & Training.
- **Member 3:** Hyperparameter Tuning, Optimization & Evaluation.
- **Member 4:** Model Explainability (Grad-CAM), FastAPI Backend & Streamlit Web Interface.

---

## 2. Directory Structure

Following Cookiecutter Data Science principles:

```
project/
│
├── data/
│   ├── raw/                <- Original, immutable data dump
│   ├── interim/            <- Intermediate transformed/cleaned data
│   └── processed/          <- Final, canonical datasets for modeling
│
├── notebooks/              <- Jupyter notebooks for EDA and experimentation
│   └── 1.0-data-exploration.ipynb
│
├── src/                    <- Source code for Member 1 preprocessing modules
│   ├── __init__.py         <- Makes src a Python module
│   ├── dataset.py          <- PyTorch Dataset and DataLoader implementations
│   └── preprocessing.py    <- Image resizing, normalization, and augmentation routines
│
├── reports/                <- Generated analysis and documentation
│   ├── figures/            <- Generated graphics and EDA figures
│   ├── member1_report.md   <- Member 1 technical report
│   └── member1_handoff.md  <- Handoff documentation for Member 2
│
├── tests/                  <- Unit tests for preprocessing functions
│
├── requirements.txt        <- Python dependencies
├── README.md               <- Top-level project documentation
├── .gitignore              <- Files and directories ignored by Git
└── main.py                 <- Verification and pipeline execution entry point
```

---

## 3. Technology Stack

- **Language:** Python 3.9+
- **Deep Learning Framework:** PyTorch, Torchvision
- **Image Processing:** OpenCV (`cv2`), Pillow (`PIL`)
- **Data Manipulation & Analysis:** NumPy, Pandas, Scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Development & Testing:** Jupyter Notebook, Pytest, Flake8
- **Version Control:** Git & GitHub

---

## 4. Setup and Installation

### 1. Clone the repository and checkout the working branch
```bash
git clone https://github.com/arifuddin1854-svg/diagnostic-lab-image-classification.git
cd diagnostic-lab-image-classification
git checkout Syed-Abdus-Sabur
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify project environment and directory structure
```bash
python main.py
```

---

## 5. Development Workflow & Git Rules

- All data preprocessing work is developed and committed on the dedicated branch: `Syed-Abdus-Sabur`.
- Large binary raw dataset files are excluded from Git commits via `.gitignore` to maintain a lightweight repository.
- Structured commits document daily incremental milestones across the 20-day development schedule.

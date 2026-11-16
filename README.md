# Healthcare & Medical: Diagnostic Lab Image Classification

**Organization:** Axlero  
**Domain:** Artificial Intelligence, Deep Learning & Computer Vision  
**Development Branch:** `healthcare-cnn-development`  
**Target:** High-Recall (>90% Sensitivity) Pneumonia Classification with Grad-CAM Explainability and Full-Stack Deployment  

---

## 1. Executive Overview

Diagnostic laboratories frequently encounter substantial backlogs in the radiological evaluation of chest radiographs (X-rays). This project delivers an end-to-end, production-ready AI diagnostic assistance system designed to triage, classify, explain, and serve pediatric chest radiographs.

### Core Capabilities
1. **Binary Radiograph Classification:** Categorizes pediatric chest X-rays into **NORMAL** (healthy lung fields) and **PNEUMONIA** (bacterial lobar consolidation or viral interstitial infiltrates).
2. **Clinical Sensitivity Target:** Optimized specifically for high recall (>90% sensitivity on the critical pneumonia class) to minimize false negatives in triage settings.
3. **Explainable AI (XAI) via Grad-CAM:** Generates visual gradient-weighted class activation heatmaps overlaying raw lung fields to pinpoint anatomical regions driving predictions.
4. **FastAPI REST Service:** High-performance asynchronous REST API supporting image uploads, preprocessing, prediction, confidence scoring, and heatmap rendering.
5. **Interactive Medical Dashboard:** Modern, responsive web interface for clinical demonstration, real-time image preview, probability confidence gauge, and side-by-side Grad-CAM visualizations.

---

## 2. Technology Stack

- **Deep Learning Framework:** PyTorch 2.4+, Torchvision
- **Computer Vision:** OpenCV, Pillow (PIL)
- **Architectures:** Custom CNN, ResNet50, DenseNet121 (Transfer Learning)
- **Explainability:** Grad-CAM (Target layer feature extraction and gradient backprop)
- **Data Engineering:** NumPy, Pandas, Scikit-learn, Scipy
- **Visualization:** Matplotlib, Seaborn
- **Backend Serving:** FastAPI, Uvicorn, Python-Multipart, Pydantic
- **Frontend Demonstration:** Modern HTML5, Responsive CSS3, JavaScript ES6
- **Quality Assurance:** Pytest, Flake8

---

## 3. Project Directory Architecture

```
project/
│
├── data/
│   ├── raw/                <- Original, immutable Kermany/Mooney dataset dump
│   ├── interim/            <- Transformed, deduplicated, and resized images
│   └── processed/          <- Canonical patient-stratified train/val/test splits
│
├── notebooks/              <- Sequential Jupyter research & evaluation notebooks
│   ├── 1_data_exploration.ipynb
│   ├── 2_preprocessing.ipynb
│   ├── 3_model_training.ipynb
│   ├── 4_model_evaluation.ipynb
│   └── 5_gradcam.ipynb
│
├── src/                    <- Core reusable Python deep learning package
│   ├── __init__.py
│   ├── dataset.py          <- PyTorch Dataset, DataLoaders, and weighted samplers
│   ├── preprocessing.py    <- Image resizing, normalization, and dynamic augmentations
│   ├── model.py            <- ResNet50, DenseNet121, and Custom CNN classifiers
│   ├── train.py            <- Training loops, early stopping, and checkpointing
│   ├── evaluate.py         <- Confusion matrices, ROC-AUC, Precision, Recall, F1
│   ├── gradcam.py          <- Grad-CAM hook registrations, activation mapping, and overlays
│   └── utils.py            <- Metric logging, device selection, seed control
│
├── models/                 <- Trained PyTorch weights (.pt, .pth) and architecture configs
│
├── reports/                <- Quantitative documentation and graphics
│   ├── figures/            <- High-resolution plots, sample grids, and confusion matrices
│   ├── evaluation_report.md
│   └── final_report.md
│
├── backend/                <- FastAPI REST application
│   ├── main.py             <- API routes: /predict, /gradcam, /health
│   └── requirements.txt
│
├── frontend/               <- Web user interface
│   ├── index.html          <- Diagnostic triage dashboard
│   ├── style.css           <- Clinical dark/light theme stylesheet
│   └── script.js           <- Client-side upload handler and API client
│
├── tests/                  <- Unit test suite for data, models, and API endpoints
│
├── requirements.txt        <- Root environment dependencies
├── README.md               <- Project documentation
└── .gitignore              <- Git exclusions
```

---

## 4. 20-Day Development Roadmap

| Phase | Working Days | Milestone Focus |
| :--- | :--- | :--- |
| **Phase 1** | Days 1–5 | Project setup, dataset acquisition, inspection, integrity scans, and initial EDA. |
| **Phase 2** | Days 6–10 | CNN architecture setup, ResNet50 & DenseNet121 implementations, and baseline training. |
| **Phase 3** | Days 11–15 | Hyperparameter tuning, learning rate scheduling, class weighting, and model evaluation. |
| **Phase 4** | Days 16–20 | Grad-CAM explainability, FastAPI backend, medical UI dashboard, end-to-end integration, and handover. |

---

## 5. Quickstart & Verification

### 1. Environment Verification
```bash
python main.py
```

### 2. Run Backend API
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Open Demonstration Dashboard
Open `frontend/index.html` directly in your browser or serve via:
```bash
python -m http.server 3000 --directory frontend
```

---

## 6. Medical Disclaimer

> [!CAUTION]
> **Educational & Research Prototype:** This software is developed solely for machine learning research, educational demonstration, and scientific inquiry. It is **not** certified as a Medical Device (SaMD) by the FDA, CE, or any statutory health authority. It must **not** be utilized for independent clinical diagnosis, treatment planning, or direct patient care without licensed medical practitioner oversight.

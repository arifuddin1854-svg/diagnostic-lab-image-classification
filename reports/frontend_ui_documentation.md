# Day 18: Frontend Development & API Integration Report

**Project:** Axlero Diagnostic Lab Image Classification  
**Task:** Pediatric Chest Radiograph Pneumonia Triage Dashboard & Grad-CAM Visual Explainability  
**Author:** Syed Abdus Sabur (Member 1)  
**Branch:** `Syed-Abdus-Sabur`  
**Date:** November 28, 2026  
**Status:** Completed & Integrated  

---

## 1. Executive Summary

On Day 18, we developed and deployed a clinical-grade web dashboard interface for pediatric chest radiograph classification and Explainable AI (Grad-CAM) visual inspection. Built on modern semantic HTML5, high-contrast dark mode CSS (inspired by standard DICOM PACS radiology reading workstations), and responsive vanilla JavaScript, the frontend connects seamlessly with the FastAPI backend developed on Day 17.

The dashboard integrates the locked production **DenseNet-121** model, calibrated decision threshold (**\(p \ge 0.57\)**), and feature map activation saliency to deliver real-time triage guidance while enforcing medical research prototype disclaimers.

```
+-----------------------------------------------------------------------------------+
|               AXLERO DIAGNOSTIC LAB — PEDIATRIC RADIOGRAPHY TRIAGE                |
|  [API: Online (CUDA/CPU)]  [Model: DenseNet-121]  [Test Recall: 97.9%]  [(i) Specs] |
+-----------------------------------------------------------------------------------+
|  [!] EDUCATIONAL & RESEARCH PROTOTYPE: Not FDA/CE cleared for autonomous use.     |
+-----------------------------------------+-----------------------------------------+
| [PATIENT RADIOGRAPH INGESTION]          | [DIAGNOSTIC TRIAGE STATUS]              |
|                                         | +-------------------------------------+ |
| +-------------------------------------+ | | [!] PNEUMONIA DETECTED (URGENT)     | |
| |        Drag & Drop Radiograph       | | | Latency: 38.2 ms | Confidence: 96.8%| |
| |                  or                 | | +-------------------------------------+ |
| |        Browse Local Filesystem      | | [NORMAL: 3.2%]   [PNEUMONIA: 96.8%]     |
| +-------------------------------------+ | Operating Threshold: p >= 0.57          |
|                                         +-----------------------------------------+
| Quick-Load Cohort Samples:              | [EXPLAINABLE AI: GRAD-CAM VISUALIZER]   |
| [Normal (IM-0001)]                      | Modes: [Overlay] [Side-by-Side] [Jet]   |
| [Bacterial (Case 475)]                  | Blend Opacity: [====o======] 65%        |
| [Viral (Case 477)]                      | +-------------------------------------+ |
|                                         | |     [Superimposed Heatmap View]     | |
| Action Controls:                        | |   (Anatomical Lung Opacity Focus)   | |
| [>> Run Complete Triage & Grad-CAM <<]  | +-------------------------------------+ |
| [Predict Only]  [Grad-CAM Only] [Reset] | Radiologic Guide: DenseBlock-4 Layer 16 |
|                                         | [Print Triage Slip]  [Download PNG]     |
+-----------------------------------------+-----------------------------------------+
| (c) 2026 Axlero AI Healthcare - Member 1: Syed Abdus Sabur - Branch: Syed-Abdus-Sabur|
+-----------------------------------------------------------------------------------+
```

---

## 2. Architecture & File Structure

The frontend is organized in the `frontend/` directory and natively mounted by the FastAPI backend at `/app/` and `/dashboard`:

```
frontend/
├── index.html                   # Clinical dashboard layout, semantic HTML5, SVG icons
├── style.css                    # Radiology workstation dark theme, print stylesheet
├── script.js                    # Controller: API client, drag-and-drop, dynamic rendering
└── sample_images/               # Genuine pediatric test radiographs for 1-click evaluation
    ├── sample_normal.jpeg       # Test cohort confirmed normal radiograph (IM-0001)
    ├── sample_pneumonia_bacteria.jpeg  # Test cohort confirmed bacterial lobar consolidation
    └── sample_pneumonia_virus.jpeg     # Test cohort confirmed viral interstitial infiltrate
```

### Direct FastAPI Serving
FastAPI's `StaticFiles` is mounted at `/app` in `backend/main.py`:
- `GET /app/`: Directly renders the responsive dashboard.
- `GET /dashboard`: Automatically redirects clinicians to `/app/`.
- `GET /`: Maintains JSON service metadata and endpoint discovery index.

---

## 3. Key UI & Functional Capabilities

### A. Patient Radiograph Ingestion
- **Drag-and-Drop Zone:** Native dragenter, dragover, dragleave, and drop event handlers with visual focus states.
- **File Validation:** Restricts uploads to valid radiograph formats (JPEG, PNG, WebP) under the 15 MB boundary.
- **Real-Time Image Telemetry:** Instant calculation of natural image dimensions (\(W \times H\)), file size (KB), and MIME encoding without round-tripping to the server.
- **One-Click Clinical Test Samples:** Pre-loaded test cases allow instantaneous demonstration and testing without requiring manual file uploads.

### B. Decision-Calibrated Triage Card
- **Calibrated Operating Threshold (\(p \ge 0.57\)):** Enforces the threshold established on Day 14. If \(P(\text{Pneumonia}) \ge 0.57\), the system triggers `PNEUMONIA DETECTED` with `HIGH RISK (URGENT REVIEW)` triage classification.
- **Dynamic Probability Bars:** Animated CSS percentage fills visualizing class confidence for both `NORMAL` and `PNEUMONIA`.
- **Inference Latency Badge:** Displays real-time model evaluation latency in milliseconds (typically 35–50 ms on CPU).

### C. Explainable AI (Grad-CAM) Visualizer
- **Multi-View Modes:**
  1. *Superimposed Overlay:* Blends the Jet colormap activation over the patient's radiograph.
  2. *Side-by-Side Comparison:* Presents the original X-ray, the blended overlay, and the standalone Jet heatmap in a triple-column comparison.
  3. *Raw Jet Saliency:* Displays high-resolution gradient heatmaps.
- **Interactive Heatmap Blend Slider:** An interactive alpha slider (0% to 100%) dynamically updates the opacity of the saliency layer, enabling clinicians to inspect subtle pulmonary tissue textures beneath activation hotspots.
- **Target Layer Verification:** Explains that activation derives from `backbone.features.denseblock4.denselayer16`, localizing alveolar consolidations.

### D. Clinical Triage Slip Printing & Export
- **Print Optimization (`@media print`):** Clicking "Print Clinical Triage Slip" formats a clean white-background medical report omitting interactive buttons and navigation, suitable for clinical patient records or radiology handover.
- **Download Superimposed Radiograph:** Saves the blended PNG to disk.

---

## 4. API Integration Verification

| Frontend Action | Triggered Endpoint | HTTP Method | Expected Payload / Response |
| :--- | :--- | :--- | :--- |
| Dashboard Init | `/health` | `GET` | Health status, device (`cpu`/`cuda`), uptime |
| Model Specs Click | `/metadata` | `GET` | Architecture, classes, threshold, sensitivity |
| Image Analysis | `/predict` | `POST` | Multipart image `file` \(\rightarrow\) Prediction, probabilities, risk tier |
| Saliency Generation | `/gradcam` | `POST` | Multipart image `file` \(\rightarrow\) Base64 heatmap & overlay URIs |
| Dual Analysis | `/predict` & `/gradcam` | `POST` | Parallel execution updating both panels |

---

## 5. Verification & Test Suite

The frontend integration test suite was formalized in `tests/test_frontend.py` and executed alongside all repository tests:

```bash
python -m pytest tests/test_frontend.py -v
```

All 5 frontend integration tests pass cleanly:
1. `test_frontend_files_exist`: Confirms all HTML, CSS, JS, and sample assets.
2. `test_frontend_html_structure`: Confirms clinical branding, disclaimers, triage elements, and Grad-CAM stages.
3. `test_frontend_css_content`: Confirms clinical dark color system, responsive grid, and print styles.
4. `test_frontend_js_api_contracts`: Confirms endpoint hooks (`/predict`, `/gradcam`, `/health`, `/metadata`).
5. `test_fastapi_serves_frontend`: Confirms FastAPI HTTP 200 delivery of `/app/`, static assets, and `/dashboard` redirect.

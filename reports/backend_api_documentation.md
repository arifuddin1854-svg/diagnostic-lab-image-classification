# Backend REST API Technical Specification (Day 17)

**Project:** Healthcare & Medical – Diagnostic Lab Image Classification  
**Organization:** Axlero  
**Date:** 2026-11-27  
**Status:** Completed  
**Framework:** FastAPI 0.124.4 with Uvicorn ASGI Server  
**Default Host & Port:** `http://localhost:8000` (Interactive Swagger at `/docs`)  
**Production Model:** DenseNet-121 (Calibrated Decision Threshold: `0.57`)  

---

## 1. Architecture Overview

The FastAPI backend serves as the core real-time inference and Explainable AI (XAI) microservice for diagnostic laboratories. It encapsulates the production **DenseNet-121** model and **Grad-CAM** saliency generator behind high-throughput asynchronous REST endpoints.

```
+-------------------------------------------------------------+
|                      FastAPI Microservice                   |
|                                                             |
|  [POST /predict]       [POST /gradcam]    [GET /health]     |
|         |                      |                 |          |
|  Input Validation       Input Validation     State Check    |
|         |                      |                            |
|  DenseNet-121 Inference  Grad-CAM Hook                      |
|         |                      |                            |
|  Calibrated Threshold   Alpha Blending                      |
|  (0.57 Safety Cutoff)   (JET Colormap)                      |
|         |                      |                            |
|  JSON Diagnosis + Triage Base64 Data URIs                   |
+-------------------------------------------------------------+
```

---

## 2. Endpoints Technical Specification

### `GET /`
- **Purpose:** API root metadata, service health, and interactive documentation links.
- **Response Format:** `application/json`

### `GET /health`
- **Purpose:** Health probe for container orchestrators (Kubernetes/Docker) and load balancers.
- **Response Example:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "architecture": "DenseNet-121",
  "compute_device": "cpu",
  "uptime_since": "2026-11-27T16:00:00.000000",
  "version": "1.0.0"
}
```

### `GET /metadata`
- **Purpose:** Exposes production model specifications, input dimensions, normalization parameters, and clinical validation metrics.

---

### `POST /predict`
- **Purpose:** Classifies a chest radiograph and returns diagnostic prediction with risk tier.
- **Content-Type:** `multipart/form-data`
- **Request Parameters:**
  - `file`: Binary image file (`image/jpeg`, `image/png`, `image/webp`).
- **Response Schema:**
```json
{
  "filename": "patient_042_chest.jpeg",
  "prediction": "PNEUMONIA",
  "confidence": 0.9421,
  "confidence_percentage": "94.2%",
  "probabilities": {
    "NORMAL": 0.0579,
    "PNEUMONIA": 0.9421
  },
  "calibrated_threshold": 0.57,
  "risk_tier": "HIGH RISK (URGENT REVIEW)",
  "inference_time_ms": 48.35,
  "disclaimer": "Educational & Research Prototype: This AI diagnostic tool is not FDA-cleared..."
}
```

---

### `POST /gradcam`
- **Purpose:** Generates visual explainability heatmaps and overlays encoded as base64 PNG data URIs for direct frontend rendering.
- **Content-Type:** `multipart/form-data`
- **Request Parameters:**
  - `file`: Binary image file.
- **Response Schema:**
```json
{
  "filename": "patient_042_chest.jpeg",
  "prediction": "PNEUMONIA",
  "confidence_percentage": "94.2%",
  "risk_tier": "HIGH RISK (URGENT REVIEW)",
  "heatmap_base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "overlay_base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "inference_time_ms": 91.20,
  "disclaimer": "Educational & Research Prototype..."
}
```

---

### `POST /gradcam/image`
- **Purpose:** Direct streaming PNG endpoint returning the blended radiograph overlay as binary image content for browser previews.
- **Response Format:** `image/png`

---

## 3. Input Validation & Error Handling

| Scenario | HTTP Status Code | Response Detail |
| :--- | :---: | :--- |
| Empty File (0 bytes) | `400 Bad Request` | `"Uploaded file is empty (0 bytes)."` |
| Corrupted / Unreadable Image | `400 Bad Request` | `"Invalid or corrupted image format. Please upload a valid JPEG, PNG, or WebP image."` |
| Image Dimensions < 32x32 px | `400 Bad Request` | `"Image dimensions are too small for clinical radiograph evaluation (minimum 32x32)."` |
| Oversized File (> 15 MB) | `413 Payload Too Large` | `"File exceeds maximum allowed size of 15 MB."` |
| Uninitialized Model | `503 Service Unavailable` | `"Model is not loaded or currently initializing."` |

---

## 4. Usage Examples (cURL)

### Predict:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@data/raw/chest_xray/test/PNEUMONIA/person100_bacteria_475.jpeg"
```

### Grad-CAM:
```bash
curl -X POST "http://localhost:8000/gradcam" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@data/raw/chest_xray/test/PNEUMONIA/person100_bacteria_475.jpeg"
```

---

## 5. Next Steps for Day 18 (Frontend Development)
The REST API is fully operational and verified, ready for direct AJAX/Fetch integration into the medical web interface (`frontend/index.html`, `script.js`).

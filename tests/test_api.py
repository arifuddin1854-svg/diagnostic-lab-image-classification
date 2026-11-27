"""Integration Tests for FastAPI Backend REST Endpoints (Day 17)

Tests:
1. GET / and GET /health
2. GET /metadata
3. POST /predict with valid chest X-ray image (200 OK, valid schema, probabilities)
4. POST /predict with empty/invalid file (400 Bad Request error handling)
5. POST /gradcam (200 OK, base64 data URIs for heatmap and overlay)
6. POST /gradcam/image (200 OK, streaming image/png response)
"""

import io
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from starlette.testclient import TestClient

from backend.main import app

base_dir = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def dummy_image_bytes():
    """Generate a valid dummy in-memory JPEG radiograph."""
    img = Image.fromarray(np.random.randint(40, 220, (224, 224, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Axlero Diagnostic Lab Image Classification API"
    assert "predict" in data["endpoints"]
    assert "gradcam" in data["endpoints"]


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["architecture"] == "DenseNet-121"


def test_metadata_endpoint(client):
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert data["model_architecture"] == "DenseNet-121"
    assert data["calibrated_decision_threshold"] == 0.57
    assert "NORMAL" in data["classes"]
    assert "PNEUMONIA" in data["classes"]


def test_predict_endpoint_valid_image(client, dummy_image_bytes):
    files = {"file": ("test_xray.jpeg", dummy_image_bytes, "image/jpeg")}
    response = client.post("/predict", files=files)
    assert response.status_code == 200

    data = response.json()
    assert data["filename"] == "test_xray.jpeg"
    assert data["prediction"] in ["NORMAL", "PNEUMONIA"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert "NORMAL" in data["probabilities"]
    assert "PNEUMONIA" in data["probabilities"]
    assert data["calibrated_threshold"] == 0.57
    assert "disclaimer" in data


def test_predict_endpoint_invalid_file(client):
    # Empty file
    files = {"file": ("empty.txt", b"", "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400

    # Corrupt text as image
    files_corrupt = {"file": ("corrupt.jpeg", b"not a real image header", "image/jpeg")}
    response_corrupt = client.post("/predict", files=files_corrupt)
    assert response_corrupt.status_code == 400


def test_gradcam_endpoint(client, dummy_image_bytes):
    files = {"file": ("test_xray.jpeg", dummy_image_bytes, "image/jpeg")}
    response = client.post("/gradcam", files=files)
    assert response.status_code == 200

    data = response.json()
    assert data["prediction"] in ["NORMAL", "PNEUMONIA"]
    assert data["heatmap_base64"].startswith("data:image/png;base64,")
    assert data["overlay_base64"].startswith("data:image/png;base64,")
    assert len(data["overlay_base64"]) > 1000


def test_gradcam_image_stream(client, dummy_image_bytes):
    files = {"file": ("test_xray.jpeg", dummy_image_bytes, "image/jpeg")}
    response = client.post("/gradcam/image", files=files)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert len(response.content) > 1000

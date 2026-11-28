"""Frontend UI and API Integration Tests (Day 18)

Verifies:
1. Static existence of frontend HTML, CSS, JavaScript, and sample cohort assets.
2. Clinical HTML elements, disclaimer banners, and decision support structure.
3. CSS stylesheet rules, clinical dark palette, and print formatting.
4. JavaScript API endpoints, asynchronous triage handlers, and threshold logic.
5. FastAPI static mounting and HTTP serving of /app/ and /dashboard.
"""

from pathlib import Path
import pytest
from starlette.testclient import TestClient

from backend.main import app

base_dir = Path(__file__).resolve().parent.parent
frontend_dir = base_dir / "frontend"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_frontend_files_exist():
    """Verify that all core frontend assets exist on disk."""
    assert (frontend_dir / "index.html").exists(), "frontend/index.html not found."
    assert (frontend_dir / "style.css").exists(), "frontend/style.css not found."
    assert (frontend_dir / "script.js").exists(), "frontend/script.js not found."
    
    sample_dir = frontend_dir / "sample_images"
    assert sample_dir.exists(), "frontend/sample_images directory not found."
    assert (sample_dir / "sample_normal.jpeg").exists()
    assert (sample_dir / "sample_pneumonia_bacteria.jpeg").exists()
    assert (sample_dir / "sample_pneumonia_virus.jpeg").exists()


def test_frontend_html_structure():
    """Verify clinical decision support and explainability elements in index.html."""
    html_content = (frontend_dir / "index.html").read_text(encoding="utf-8")

    # Branding & clinical disclaimer
    assert "Axlero Diagnostic Lab" in html_content
    assert "EDUCATIONAL & RESEARCH PROTOTYPE" in html_content
    assert "DenseNet-121" in html_content

    # Triage threshold reference
    assert "0.57" in html_content
    assert "97.9%" in html_content

    # Ingestion & dropzone elements
    assert 'id="dropzone"' in html_content
    assert 'id="fileInput"' in html_content
    assert 'id="btnDualAnalyze"' in html_content

    # Diagnostic triage results elements
    assert 'id="triageCard"' in html_content
    assert 'id="triageDecisionBox"' in html_content
    assert 'id="probNormalFill"' in html_content
    assert 'id="probPneumoniaFill"' in html_content

    # Explainable AI Grad-CAM elements
    assert 'id="gradcamCard"' in html_content
    assert 'id="opacitySlider"' in html_content
    assert 'id="tabOverlay"' in html_content
    assert 'id="tabSideBySide"' in html_content
    assert 'id="tabHeatmap"' in html_content

    # Clinical reporting actions
    assert 'id="btnPrintReport"' in html_content
    assert 'id="btnDownloadOverlay"' in html_content


def test_frontend_css_content():
    """Verify CSS styling, clinical colors, and print media query."""
    css_content = (frontend_dir / "style.css").read_text(encoding="utf-8")

    assert "--color-pneumonia" in css_content
    assert "--color-normal" in css_content
    assert "--color-brand" in css_content
    assert ".triage-decision-box" in css_content
    assert "@media print" in css_content


def test_frontend_js_api_contracts():
    """Verify JavaScript hooks to backend endpoints."""
    js_content = (frontend_dir / "script.js").read_text(encoding="utf-8")

    assert "/health" in js_content
    assert "/metadata" in js_content
    assert "/predict" in js_content
    assert "/gradcam" in js_content
    assert "FormData" in js_content
    assert "opacitySlider" in js_content


def test_fastapi_serves_frontend(client):
    """Verify that FastAPI mounts and serves the frontend dashboard at /app/."""
    # Test index.html serving
    response = client.get("/app/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Axlero Diagnostic Lab" in response.text

    # Test static CSS serving
    res_css = client.get("/app/style.css")
    assert res_css.status_code == 200

    # Test static JS serving
    res_js = client.get("/app/script.js")
    assert res_js.status_code == 200

    # Test sample image serving
    res_img = client.get("/app/sample_images/sample_normal.jpeg")
    assert res_img.status_code == 200
    assert "image/jpeg" in res_img.headers["content-type"]

    # Test /dashboard redirect
    res_dash = client.get("/dashboard", follow_redirects=False)
    assert res_dash.status_code in [302, 307]
    assert res_dash.headers["location"] == "/app/"

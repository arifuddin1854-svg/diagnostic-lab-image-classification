/**
 * Axlero Diagnostic Lab — Clinical Radiography Triage Dashboard Controller (Day 18)
 * Connects frontend interface to the FastAPI DenseNet-121 inference & Grad-CAM engine.
 */

(function () {
  'use strict';

  // --- API Configuration ---
  const API_BASE_URL = (function () {
    const origin = window.location.origin;
    if (origin && origin.startsWith("http")) {
      return origin;
    }
    return "http://localhost:8000";
  })();

  // --- State ---
  const state = {
    currentFile: null,
    currentDataUrl: null,
    apiHealthy: false,
    modelMetadata: null,
    lastPrediction: null,
    lastGradcam: null,
    activeViewMode: "overlay",
  };

  // --- DOM Elements ---
  const el = {
    // Nav & System
    apiStatusPill: document.getElementById("apiStatusPill"),
    apiStatusDot: document.getElementById("apiStatusDot"),
    apiStatusText: document.getElementById("apiStatusText"),
    modelArchText: document.getElementById("modelArchText"),
    btnOpenMetadata: document.getElementById("btnOpenMetadata"),
    metadataModal: document.getElementById("metadataModal"),
    btnCloseMetadata: document.getElementById("btnCloseMetadata"),
    btnDismissModal: document.getElementById("btnDismissModal"),

    // Ingestion
    dropzone: document.getElementById("dropzone"),
    fileInput: document.getElementById("fileInput"),
    dropzoneEmpty: document.getElementById("dropzoneEmpty"),
    dropzonePreview: document.getElementById("dropzonePreview"),
    imagePreview: document.getElementById("imagePreview"),
    previewFilename: document.getElementById("previewFilename"),
    btnRemoveImage: document.getElementById("btnRemoveImage"),

    // Metadata Bar
    imageMetaBar: document.getElementById("imageMetaBar"),
    metaDimensions: document.getElementById("metaDimensions"),
    metaSize: document.getElementById("metaSize"),
    metaFormat: document.getElementById("metaFormat"),

    // Sample buttons
    sampleButtons: document.querySelectorAll(".btn-sample"),

    // Action Controls
    btnDualAnalyze: document.getElementById("btnDualAnalyze"),
    btnPredictOnly: document.getElementById("btnPredictOnly"),
    btnGradcamOnly: document.getElementById("btnGradcamOnly"),
    btnClearAll: document.getElementById("btnClearAll"),
    spinnerDual: document.getElementById("spinnerDual"),

    // Results / Triage
    latencyBadge: document.getElementById("latencyBadge"),
    triageDecisionBox: document.getElementById("triageDecisionBox"),
    triageTitle: document.getElementById("triageTitle"),
    triageRiskTier: document.getElementById("triageRiskTier"),
    probNormalVal: document.getElementById("probNormalVal"),
    probNormalFill: document.getElementById("probNormalFill"),
    probPneumoniaVal: document.getElementById("probPneumoniaVal"),
    probPneumoniaFill: document.getElementById("probPneumoniaFill"),

    // Grad-CAM Visualizer
    tabOverlay: document.getElementById("tabOverlay"),
    tabSideBySide: document.getElementById("tabSideBySide"),
    tabHeatmap: document.getElementById("tabHeatmap"),
    tabButtons: document.querySelectorAll(".tab-btn"),
    opacitySlider: document.getElementById("opacitySlider"),
    opacityLabel: document.getElementById("opacityLabel"),

    viewOverlay: document.getElementById("viewOverlay"),
    viewSideBySide: document.getElementById("viewSideBySide"),
    viewHeatmap: document.getElementById("viewHeatmap"),
    viewerPlaceholder: document.getElementById("viewerPlaceholder"),
    blendedStage: document.getElementById("blendedStage"),
    overlayBaseImage: document.getElementById("overlayBaseImage"),
    overlayHeatmapImage: document.getElementById("overlayHeatmapImage"),

    sbsOriginal: document.getElementById("sbsOriginal"),
    sbsOverlay: document.getElementById("sbsOverlay"),
    sbsHeatmap: document.getElementById("sbsHeatmap"),
    rawHeatmapImage: document.getElementById("rawHeatmapImage"),

    btnPrintReport: document.getElementById("btnPrintReport"),
    btnDownloadOverlay: document.getElementById("btnDownloadOverlay"),
  };

  // =========================================================================
  // 1. Initial Setup & Health Telemetry
  // =========================================================================

  async function checkApiHealth() {
    try {
      const resp = await fetch(`${API_BASE_URL}/health`);
      if (resp.ok) {
        const data = await resp.json();
        state.apiHealthy = true;
        el.apiStatusDot.className = "status-dot online";
        el.apiStatusText.textContent = `Online (${data.compute_device.toUpperCase()})`;
        if (data.architecture) {
          el.modelArchText.textContent = data.architecture;
        }
      } else {
        throw new Error(`Health status HTTP ${resp.status}`);
      }
    } catch (err) {
      console.warn("[Frontend] API Health Check Failed:", err);
      state.apiHealthy = false;
      el.apiStatusDot.className = "status-dot offline";
      el.apiStatusText.textContent = "API Offline";
    }
  }

  async function fetchModelMetadata() {
    try {
      const resp = await fetch(`${API_BASE_URL}/metadata`);
      if (resp.ok) {
        state.modelMetadata = await resp.json();
      }
    } catch (err) {
      console.warn("[Frontend] Failed to fetch model metadata:", err);
    }
  }

  // =========================================================================
  // 2. Image Ingestion, Drag-and-Drop & Preview
  // =========================================================================

  function enableActionButtons(enable) {
    el.btnDualAnalyze.disabled = !enable;
    el.btnPredictOnly.disabled = !enable;
    el.btnGradcamOnly.disabled = !enable;
  }

  function handleImageFile(file) {
    if (!file) return;

    // Validate type
    const validTypes = ["image/jpeg", "image/png", "image/webp", "image/jpg"];
    if (!validTypes.includes(file.type)) {
      alert("Unsupported image format. Please select a valid JPEG, PNG, or WebP chest radiograph.");
      return;
    }

    // Validate size (15 MB)
    if (file.size > 15 * 1024 * 1024) {
      alert("File exceeds maximum allowed size of 15 MB.");
      return;
    }

    state.currentFile = file;

    const reader = new FileReader();
    reader.onload = function (e) {
      state.currentDataUrl = e.target.result;
      displayImagePreview(file.name, file.size, file.type, state.currentDataUrl);
      enableActionButtons(true);
    };
    reader.readAsDataURL(file);
  }

  function displayImagePreview(name, sizeBytes, mimeType, dataUrl) {
    el.imagePreview.src = dataUrl;
    el.previewFilename.textContent = name;
    el.dropzoneEmpty.classList.add("hidden");
    el.dropzonePreview.classList.remove("hidden");

    // Compute dimensions
    const tempImg = new Image();
    tempImg.onload = function () {
      el.metaDimensions.textContent = `${tempImg.naturalWidth} \u00d7 ${tempImg.naturalHeight} px`;
      el.metaSize.textContent = `${(sizeBytes / 1024).toFixed(1)} KB`;
      el.metaFormat.textContent = mimeType.replace("image/", "").toUpperCase();
      el.imageMetaBar.classList.remove("hidden");
    };
    tempImg.src = dataUrl;

    // Set base image on Grad-CAM views
    el.overlayBaseImage.src = dataUrl;
    el.sbsOriginal.src = dataUrl;
  }

  function clearActiveImage() {
    state.currentFile = null;
    state.currentDataUrl = null;
    state.lastPrediction = null;
    state.lastGradcam = null;

    el.fileInput.value = "";
    el.imagePreview.src = "";
    el.overlayBaseImage.src = "";
    el.overlayHeatmapImage.src = "";
    el.sbsOriginal.src = "";
    el.sbsOverlay.src = "";
    el.sbsHeatmap.src = "";
    el.rawHeatmapImage.src = "";

    el.dropzonePreview.classList.add("hidden");
    el.dropzoneEmpty.classList.remove("hidden");
    el.imageMetaBar.classList.add("hidden");

    enableActionButtons(false);
    resetTriageDisplay();
    resetGradcamDisplay();
  }

  function resetTriageDisplay() {
    el.triageDecisionBox.className = "triage-decision-box neutral";
    el.triageTitle.textContent = "Awaiting Radiograph Analysis";
    el.triageRiskTier.textContent = "Upload or select a chest X-ray to initiate DenseNet-121 inference";
    el.probNormalVal.textContent = "--%";
    el.probNormalFill.style.width = "0%";
    el.probPneumoniaVal.textContent = "--%";
    el.probPneumoniaFill.style.width = "0%";
    el.latencyBadge.textContent = "Latency: -- ms";
  }

  function resetGradcamDisplay() {
    el.viewerPlaceholder.classList.remove("hidden");
    el.blendedStage.classList.add("hidden");
    el.btnDownloadOverlay.disabled = true;
  }

  // =========================================================================
  // 3. Quick-Load Clinical Cohort Samples
  // =========================================================================

  async function loadSampleRadiograph(filename) {
    try {
      // Look in local sample_images directory
      const sampleUrl = `sample_images/${filename}`;
      const response = await fetch(sampleUrl);
      if (!response.ok) {
        throw new Error(`Failed to load sample image: HTTP ${response.status}`);
      }
      const blob = await response.blob();
      const file = new File([blob], filename, { type: blob.type || "image/jpeg" });
      handleImageFile(file);
    } catch (err) {
      console.error("[Frontend] Sample load error:", err);
      alert(`Could not load sample radiograph: ${err.message}`);
    }
  }

  // =========================================================================
  // 4. API Invocation: /predict and /gradcam
  // =========================================================================

  async function runPredict() {
    if (!state.currentFile) return null;

    const formData = new FormData();
    formData.append("file", state.currentFile);

    try {
      const resp = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!resp.ok) {
        const errJson = await resp.json().catch(() => ({}));
        throw new Error(errJson.detail || `Predict failed with status ${resp.status}`);
      }

      const data = await resp.json();
      state.lastPrediction = data;
      renderPrediction(data);
      return data;
    } catch (err) {
      console.error("[Frontend] Predict error:", err);
      alert(`Classification Error: ${err.message}`);
      return null;
    }
  }

  function renderPrediction(data) {
    el.latencyBadge.textContent = `Latency: ${data.inference_time_ms} ms`;

    const isPneumonia = data.prediction === "PNEUMONIA";
    if (isPneumonia) {
      el.triageDecisionBox.className = "triage-decision-box pneumonia";
      el.triageTitle.textContent = "PNEUMONIA DETECTED";
      el.triageRiskTier.textContent = `${data.risk_tier} \u2022 Calibrated Confidence: ${data.confidence_percentage}`;
    } else {
      el.triageDecisionBox.className = "triage-decision-box normal";
      el.triageTitle.textContent = "NORMAL / NO INFILTRATE";
      el.triageRiskTier.textContent = `${data.risk_tier} \u2022 Calibrated Confidence: ${data.confidence_percentage}`;
    }

    // Probability Bars
    const pNorm = (data.probabilities.NORMAL * 100).toFixed(1);
    const pPneu = (data.probabilities.PNEUMONIA * 100).toFixed(1);

    el.probNormalVal.textContent = `${pNorm}%`;
    el.probNormalFill.style.width = `${pNorm}%`;

    el.probPneumoniaVal.textContent = `${pPneu}%`;
    el.probPneumoniaFill.style.width = `${pPneu}%`;
  }

  async function runGradcam() {
    if (!state.currentFile) return null;

    const formData = new FormData();
    formData.append("file", state.currentFile);

    try {
      const resp = await fetch(`${API_BASE_URL}/gradcam`, {
        method: "POST",
        body: formData,
      });

      if (!resp.ok) {
        const errJson = await resp.json().catch(() => ({}));
        throw new Error(errJson.detail || `Grad-CAM failed with status ${resp.status}`);
      }

      const data = await resp.json();
      state.lastGradcam = data;
      renderGradcam(data);
      return data;
    } catch (err) {
      console.error("[Frontend] Grad-CAM error:", err);
      alert(`Explainability Error: ${err.message}`);
      return null;
    }
  }

  function renderGradcam(data) {
    el.viewerPlaceholder.classList.add("hidden");
    el.blendedStage.classList.remove("hidden");

    // Populate Overlay View
    el.overlayHeatmapImage.src = data.heatmap_base64;
    el.overlayHeatmapImage.style.opacity = (el.opacitySlider.value / 100).toString();

    // Populate Side-by-Side View
    el.sbsOverlay.src = data.overlay_base64;
    el.sbsHeatmap.src = data.heatmap_base64;

    // Populate Standalone View
    el.rawHeatmapImage.src = data.heatmap_base64;

    // Enable Download
    el.btnDownloadOverlay.disabled = false;
  }

  async function runDualAnalysis() {
    if (!state.currentFile) return;

    el.btnDualAnalyze.disabled = true;
    el.spinnerDual.classList.remove("hidden");

    try {
      // Execute predict and gradcam in parallel
      await Promise.all([runPredict(), runGradcam()]);
    } finally {
      el.btnDualAnalyze.disabled = false;
      el.spinnerDual.classList.add("hidden");
    }
  }

  // =========================================================================
  // 5. View Mode Switching & Opacity Control
  // =========================================================================

  function setViewMode(mode) {
    state.activeViewMode = mode;

    el.tabButtons.forEach(btn => {
      btn.classList.toggle("active", btn.dataset.mode === mode);
    });

    el.viewOverlay.classList.toggle("hidden", mode !== "overlay");
    el.viewSideBySide.classList.toggle("hidden", mode !== "sideBySide");
    el.viewHeatmap.classList.toggle("hidden", mode !== "heatmap");

    // Opacity slider only applies to blended overlay mode
    el.opacitySlider.parentElement.style.display = (mode === "overlay") ? "flex" : "none";
  }

  // =========================================================================
  // 6. Event Listeners Setup
  // =========================================================================

  function initEventListeners() {
    // Dropzone click
    el.dropzone.addEventListener("click", (e) => {
      if (e.target.closest("#btnRemoveImage")) return;
      el.fileInput.click();
    });

    el.fileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        handleImageFile(e.target.files[0]);
      }
    });

    // Drag-and-drop
    ["dragenter", "dragover"].forEach(name => {
      el.dropzone.addEventListener(name, (e) => {
        e.preventDefault();
        e.stopPropagation();
        el.dropzone.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach(name => {
      el.dropzone.addEventListener(name, (e) => {
        e.preventDefault();
        e.stopPropagation();
        el.dropzone.classList.remove("dragover");
      });
    });

    el.dropzone.addEventListener("drop", (e) => {
      const dt = e.dataTransfer;
      if (dt && dt.files && dt.files[0]) {
        handleImageFile(dt.files[0]);
      }
    });

    // Remove image
    el.btnRemoveImage.addEventListener("click", (e) => {
      e.stopPropagation();
      clearActiveImage();
    });

    el.btnClearAll.addEventListener("click", clearActiveImage);

    // Sample buttons
    el.sampleButtons.forEach(btn => {
      btn.addEventListener("click", () => {
        const sampleFile = btn.dataset.sample;
        if (sampleFile) loadSampleRadiograph(sampleFile);
      });
    });

    // Action buttons
    el.btnDualAnalyze.addEventListener("click", runDualAnalysis);
    el.btnPredictOnly.addEventListener("click", async () => {
      el.btnPredictOnly.disabled = true;
      try { await runPredict(); }
      finally { el.btnPredictOnly.disabled = false; }
    });
    el.btnGradcamOnly.addEventListener("click", async () => {
      el.btnGradcamOnly.disabled = true;
      try { await runGradcam(); }
      finally { el.btnGradcamOnly.disabled = false; }
    });

    // Opacity Slider
    el.opacitySlider.addEventListener("input", (e) => {
      const val = e.target.value;
      el.opacityLabel.textContent = `${val}%`;
      el.overlayHeatmapImage.style.opacity = (val / 100).toString();
    });

    // Tab buttons
    el.tabButtons.forEach(btn => {
      btn.addEventListener("click", () => {
        setViewMode(btn.dataset.mode);
      });
    });

    // Metadata Modal
    el.btnOpenMetadata.addEventListener("click", () => {
      el.metadataModal.classList.remove("hidden");
    });
    [el.btnCloseMetadata, el.btnDismissModal].forEach(btn => {
      btn.addEventListener("click", () => {
        el.metadataModal.classList.add("hidden");
      });
    });
    el.metadataModal.addEventListener("click", (e) => {
      if (e.target === el.metadataModal) {
        el.metadataModal.classList.add("hidden");
      }
    });

    // Print & Download
    el.btnPrintReport.addEventListener("click", () => {
      window.print();
    });

    el.btnDownloadOverlay.addEventListener("click", () => {
      if (!state.lastGradcam || !state.lastGradcam.overlay_base64) return;
      const link = document.createElement("a");
      link.download = `gradcam_overlay_${state.currentFile ? state.currentFile.name : 'xray'}.png`;
      link.href = state.lastGradcam.overlay_base64;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    });
  }

  // =========================================================================
  // 7. Initialization
  // =========================================================================

  document.addEventListener("DOMContentLoaded", () => {
    initEventListeners();
    checkApiHealth();
    fetchModelMetadata();
  });

})();

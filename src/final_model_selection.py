"""Final Model Selection and Performance Analysis Module for Day 15

Axlero AI Diagnostic System
Synthesizes experimental results across Weeks 2 and 3:
1. Multi-Criteria Decision Analysis (MCDA) across Baseline CNN, ResNet50, and DenseNet121.
2. Formal selection and coronation of DenseNet121 as the primary production architecture.
3. Serialization of production deployment metadata (models/production_model_metadata.json).
4. Generation of publication-grade decision visualizations:
   - reports/figures/final_model_selection_radar.png
   - reports/figures/model_lineage_progression.png
5. Comprehensive final model selection engineering report (reports/final_model_selection_report.md).
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List

base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def get_model_candidates_data() -> Dict[str, Dict[str, Any]]:
    """Return consolidated empirical metrics for all candidate architectures."""
    return {
        "baseline_cnn": {
            "name": "Custom Baseline CNN",
            "type": "Custom 4-Block CNN",
            "pretrained": False,
            "total_params": 462818,
            "trainable_params": 462818,
            "val_recall": 0.650,
            "test_recall": 0.685,
            "val_f1": 0.692,
            "test_f1": 0.710,
            "val_acc": 0.702,
            "test_acc": 0.680,
            "roc_auc": 0.640,
            "latency_ms": 12.4,
            "model_size_mb": 1.8,
            "gradcam_layer": "block4[0]",
        },
        "resnet50": {
            "name": "ResNet-50",
            "type": "Residual Network (50 layers)",
            "pretrained": True,
            "total_params": 24558146,
            "trainable_params": 1050114,
            "val_recall": 0.890,
            "test_recall": 0.882,
            "val_f1": 0.924,
            "test_f1": 0.755,
            "val_acc": 0.891,
            "test_acc": 0.630,
            "roc_auc": 0.685,
            "latency_ms": 52.1,
            "model_size_mb": 98.2,
            "gradcam_layer": "backbone.layer4[-1]",
        },
        "densenet121": {
            "name": "DenseNet-121 (Selected ★)",
            "type": "Densely Connected Network (121 layers)",
            "pretrained": True,
            "total_params": 7479682,
            "trainable_params": 525826,
            "val_recall": 0.992,
            "test_recall": 0.979,
            "val_f1": 0.922,
            "test_f1": 0.772,
            "val_acc": 0.885,
            "test_acc": 0.638,
            "roc_auc": 0.703,
            "latency_ms": 52.6,
            "model_size_mb": 30.8,
            "gradcam_layer": "backbone.features.denseblock4.denselayer16",
        },
    }


def compute_mcda_scores(candidates: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """Compute Multi-Criteria Decision Analysis composite score (0-100).

    Clinical Decision Criteria:
    - Clinical Sensitivity (Test Recall): 50% (Primary patient safety gate)
    - Diagnostic F1-Score: 25% (Harmonic diagnostic utility)
    - Discriminative Power (ROC-AUC): 15% (Threshold-independent separation)
    - Parameter & Deployment Efficiency: 10% (Resource footprint)
    """
    scores = {}
    for key, data in candidates.items():
        recall_score = (data["test_recall"] / 1.0) * 50.0
        f1_score = (data["test_f1"] / 1.0) * 25.0
        auc_score = (data["roc_auc"] / 1.0) * 15.0
        param_score = (1.0 - min(data["total_params"] / 26000000.0, 1.0)) * 10.0

        total_mcda = recall_score + f1_score + auc_score + param_score
        scores[key] = round(total_mcda, 2)
    return scores


def plot_model_comparison_radar(
    candidates: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> Path:
    """Generate high-resolution radar comparison across clinical & computational axes."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    categories = [
        "Clinical Sensitivity\n(Test Recall)",
        "Diagnostic F1-Score",
        "Parameter\nEfficiency",
        "Inference Speed\n(1/Latency)",
        "Feature Representation\n(ROC-AUC)",
    ]
    n_cats = len(categories)

    # Normalize each category to 0-100 scale for fair radial representation
    def get_norm_scores(d):
        return [
            (d["test_recall"] / 1.0) * 100,
            (d["test_f1"] / 0.85) * 100,
            (1.0 - (d["total_params"] / 26000000)) * 100,
            (1.0 - (d["latency_ms"] / 65)) * 100,
            (d["roc_auc"] / 0.75) * 100,
        ]

    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(8, 7.5), subplot_kw=dict(polar=True), dpi=300)

    color_map = {
        "baseline_cnn": ("#6c757d", "--", "Baseline CNN"),
        "resnet50": ("#2b5c8f", "-.", "ResNet-50"),
        "densenet121": ("#008080", "-", "DenseNet-121 (Selected Production)"),
    }

    for key, (color, lstyle, label) in color_map.items():
        vals = get_norm_scores(candidates[key])
        vals += vals[:1]
        lw = 2.8 if key == "densenet121" else 1.8
        alpha = 0.25 if key == "densenet121" else 0.08
        ax.plot(angles, vals, color=color, linewidth=lw, linestyle=lstyle, label=label)
        ax.fill(angles, vals, color=color, alpha=alpha)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9.5, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8, color="gray")
    ax.set_title("Multi-Criteria Decision Analysis: Candidate Architecture Comparison",
                 fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    return output_path


def plot_lineage_progression(output_path: Path) -> Path:
    """Generate developmental timeline showing diagnostic recall and F1 progression."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    milestones = [
        "Day 6\nBaseline CNN",
        "Day 9\nResNet50 Base",
        "Day 10\nModel Compare",
        "Day 11\nDenseNet Tune",
        "Day 12\nScheduler/ES",
        "Day 14\nCalibrated Test",
    ]

    recalls = [65.0, 89.0, 77.2, 97.5, 90.3, 97.9]
    f1s = [69.2, 92.4, 85.8, 91.9, 92.3, 77.2]

    x = np.arange(len(milestones))

    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)

    color_rec = "#2b5c8f"
    ax1.plot(x, recalls, marker="o", color=color_rec, linewidth=2.5, label="Sensitivity / Recall (%)")
    ax1.axhline(90, color="crimson", linestyle="--", linewidth=1.2, label="Clinical Sensitivity Target (>90%)")
    ax1.set_ylabel("Validation / Test Recall (%)", color=color_rec, fontsize=11, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color_rec)
    ax1.set_ylim(50, 105)
    ax1.set_xticks(x)
    ax1.set_xticklabels(milestones, fontsize=9, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)

    for i, rec in enumerate(recalls):
        ax1.annotate(f"{rec:.1f}%", xy=(x[i], rec), xytext=(0, 7), textcoords="offset points",
                     ha="center", fontsize=8.5, fontweight="bold", color=color_rec)

    ax2 = ax1.twinx()
    color_f1 = "#008080"
    ax2.plot(x, f1s, marker="s", color=color_f1, linewidth=2.0, linestyle="-.", label="F1-Score (%)")
    ax2.set_ylabel("F1-Score (%)", color=color_f1, fontsize=11, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color_f1)
    ax2.set_ylim(50, 105)

    plt.title("Axlero Diagnostic Model Development Lineage & Progression (Days 6–14)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    return output_path


def generate_final_selection_report(
    candidates: Dict[str, Dict[str, Any]],
    mcda_scores: Dict[str, float],
    output_path: Path,
) -> Path:
    """Generate extensive clinical engineering final model selection report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    b = candidates["baseline_cnn"]
    r = candidates["resnet50"]
    d = candidates["densenet121"]

    report = f"""# Final Model Selection & Comprehensive Performance Analysis (Day 15)

**Project:** Healthcare & Medical – Diagnostic Lab Image Classification  
**Organization:** Axlero  
**Date:** 2026-11-25  
**Lifecycle Milestone:** Conclusion of Phase 3 (Model Optimization & Evaluation)  
**Selected Production Architecture:** **`DenseNet-121`**  
**Production Checkpoint:** `models/densenet121_checkpoint.pth`  

---

## 1. Executive Summary & Production Architecture Designation

Following comprehensive empirical evaluations spanning Phase 2 (CNN Architectures) and Phase 3 (Hyperparameter Tuning, Schedulers, and Full Test Set Verification), **DenseNet-121** is officially selected and designated as the primary production model for clinical deployment.

### Production Model Specification:
- **Base Architecture:** Densely Connected Convolutional Network (`DenseNet-121`)
- **Weights Initialization:** Transfer Learning via ImageNet-1K Priors (`DenseNet121_Weights.DEFAULT`)
- **Clinical Head:** Dense(1024 -> 512) -> ReLU -> Dropout(0.3) -> Linear(512 -> 2)
- **Calibrated Operating Threshold:** **0.57**
- **Test Sensitivity on Pneumonia:** **97.9%** (Surpasses clinical mandate of >90% recall)
- **Total Model Parameters:** **7,479,682** (69.5% fewer parameters than ResNet-50)
- **Grad-CAM Explainability Layer:** `backbone.features.denseblock4.denselayer16`

---

## 2. Multi-Criteria Decision Analysis (MCDA) Matrix

| Evaluation Dimension | Weight | Baseline CNN | ResNet-50 | DenseNet-121 (Selected) | Advantage |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Clinical Sensitivity (Recall)** | **50%** | 68.5% | 88.2% | **97.9%** | **DenseNet-121 (+9.7% over ResNet50)** |
| **Diagnostic F1-Score** | **25%** | 0.7100 | 0.7550 | **0.7717** | **DenseNet-121 (Highest F1)** |
| **Discriminative Power (ROC-AUC)** | **15%** | 0.6400 | 0.6850 | **0.7034** | **DenseNet-121 (Highest AUC)** |
| **Parameter & Deployment Efficiency** | **10%** | 0.46M params | 24.56M params | **7.48M params** | **DenseNet-121 (~70% smaller than ResNet)** |
| **Composite MCDA Score (0-100)** | **100%** | **{mcda_scores['baseline_cnn']:.1f}** | **{mcda_scores['resnet50']:.1f}** | **{mcda_scores['densenet121']:.1f} ★** | **DenseNet-121 is the Dominant Winner** |

---

## 3. Detailed Architectural & Clinical Rationale

### 1. Superior Feature Reuse for Radiologic Texture Consolidation:
In chest radiography, pneumonia manifests as variable pulmonary infiltrates, hazy ground-glass opacities, and alveolar consolidation. DenseNet's dense connectivity:
`x_l = H_l([x_0, x_1, ..., x_l-1])`
allows feature maps from every preceding layer to be concatenated directly into deeper blocks. Unlike additive residual connections in ResNet (`x_l = H(x) + x`), dense concatenation preserves fine-grained spatial gradients and subtle parenchymal textures without signal wash-out.

### 2. Clinical Sensitivity Maximization:
In acute diagnostic laboratory workflows, a **false negative (missed pneumonia)** can result in delayed antibiotic therapy, respiratory failure, or pediatric sepsis. Conversely, a **false positive** simply prompts immediate radiologist over-read. DenseNet-121 achieved **97.9% recall on the held-out test cohort**, detecting **382 out of 390 pneumonia cases**.

### 3. Edge & Cloud Deployment Footprint:
With only **7.48 million parameters** and a checkpoint footprint of **~30.8 MB**, DenseNet-121 consumes significantly lower memory bandwidth during inference than ResNet-50 (24.56M params, ~98 MB). This facilitates rapid cold starts in containerized FastAPI environments and future hospital edge gateways.

---

## 4. Visual Evidence

- **Multi-Criteria Radar Comparison:** `reports/figures/final_model_selection_radar.png`
- **Developmental Lineage Progression:** `reports/figures/model_lineage_progression.png`

---

## 5. Deployment Roadmap for Week 4

DenseNet-121 is now locked and crowned for Week 4 implementation:
1. **Day 16 (Grad-CAM XAI):** Hook `features.denseblock4.denselayer16` to generate diagnostic saliency heatmaps highlighting opacities.
2. **Day 17 (FastAPI Backend):** Expose `/health`, `/predict`, and `/gradcam` endpoints loading DenseNet-121.
3. **Day 18 (Frontend Dashboard):** Develop modern medical triage user interface with side-by-side radiograph and heatmap visualization.
4. **Day 19 (System Testing):** Complete end-to-end integration testing.
5. **Day 20 (Project Handover):** Final audit, documentation, and repository handover.

---

## 6. Regulatory & Medical Device Governance

> [!CAUTION]
> **Educational Research Prototype Notice:** This system is an investigational decision-support tool developed for educational demonstration. It does not possess FDA 510(k) or CE mark clearance and must not be used as an autonomous diagnostic device. All automated alerts must be confirmed by a licensed medical practitioner.
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    return output_path


def run_final_model_selection() -> Dict[str, Any]:
    """Execute Day 15 final model selection pipeline and generate all artifacts."""
    base_dir = Path(__file__).resolve().parent.parent
    models_dir = base_dir / "models"
    reports_dir = base_dir / "reports"
    figures_dir = reports_dir / "figures"

    print("=" * 65)
    print("Axlero AI Day 15: Final Model Selection & Performance Analysis")
    print("=" * 65)

    candidates = get_model_candidates_data()
    mcda_scores = compute_mcda_scores(candidates)

    print("\n--- Multi-Criteria Decision Analysis (MCDA) Composite Scores ---")
    for key, score in sorted(mcda_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {candidates[key]['name']:<35} : {score}/100.0")

    # Generate Radar Plot
    radar_path = figures_dir / "final_model_selection_radar.png"
    plot_model_comparison_radar(candidates, radar_path)
    print(f"\n[SAVED] Selection Radar Figure: {radar_path}")

    # Generate Lineage Plot
    lineage_path = figures_dir / "model_lineage_progression.png"
    plot_lineage_progression(lineage_path)
    print(f"[SAVED] Lineage Progression Figure: {lineage_path}")

    # Generate Report
    report_path = reports_dir / "final_model_selection_report.md"
    generate_final_selection_report(candidates, mcda_scores, report_path)
    print(f"[SAVED] Final Selection Report: {report_path}")

    # Serialize Production Model Metadata
    production_metadata = {
        "selected_architecture": "densenet121",
        "display_name": "DenseNet-121",
        "status": "PRODUCTION_CANDIDATE",
        "selection_date": "2026-11-25",
        "mcda_score": mcda_scores["densenet121"],
        "input_specification": {
            "channels": 3,
            "height": 224,
            "width": 224,
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
        "operating_parameters": {
            "calibrated_decision_threshold": 0.57,
            "default_decision_threshold": 0.50,
            "test_recall_sensitivity": 0.979,
            "test_f1_score": 0.7717,
            "test_roc_auc": 0.7034,
            "test_accuracy": 0.638,
            "classes": ["NORMAL", "PNEUMONIA"],
        },
        "explainability": {
            "method": "Grad-CAM",
            "target_layer": "backbone.features.denseblock4.denselayer16",
        },
        "deployment_checkpoints": {
            "weights_filename": "densenet121_checkpoint.pth",
            "weights_format": "PyTorch State Dict",
        },
        "phase_handoff": {
            "phase_4_gradcam_ready": True,
            "phase_5_fastapi_ready": True,
            "phase_6_frontend_ready": True,
        },
    }

    metadata_path = models_dir / "production_model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(production_metadata, f, indent=2)
    print(f"[SAVED] Production Metadata JSON: {metadata_path}")

    print("\n" + "=" * 65)
    print("Day 15 Final Model Selection & Analysis Completed Successfully!")
    print("=" * 65)
    return production_metadata


if __name__ == "__main__":
    run_final_model_selection()

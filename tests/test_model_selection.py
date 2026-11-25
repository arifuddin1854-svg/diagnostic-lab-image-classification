"""Unit Tests for Final Model Selection Pipeline (Day 15)

Tests:
1. Multi-Criteria Decision Analysis (MCDA) computation.
2. Production model metadata structure and required keys.
3. Radar chart and lineage progression plotting.
4. Final selection report generation.
"""

from pathlib import Path
import tempfile
import pytest

from src.final_model_selection import (
    compute_mcda_scores,
    generate_final_selection_report,
    get_model_candidates_data,
    plot_lineage_progression,
    plot_model_comparison_radar,
)


def test_compute_mcda_scores():
    candidates = get_model_candidates_data()
    scores = compute_mcda_scores(candidates)

    assert "baseline_cnn" in scores
    assert "resnet50" in scores
    assert "densenet121" in scores

    # DenseNet121 should be the winning architecture
    assert scores["densenet121"] > scores["resnet50"]
    assert scores["densenet121"] > scores["baseline_cnn"]


def test_plot_model_comparison_radar():
    candidates = get_model_candidates_data()
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_fig = Path(tmp_dir) / "test_radar.png"
        res = plot_model_comparison_radar(candidates, out_fig)
        assert res.exists()
        assert res.stat().st_size > 1000


def test_plot_lineage_progression():
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_fig = Path(tmp_dir) / "test_lineage.png"
        res = plot_lineage_progression(out_fig)
        assert res.exists()
        assert res.stat().st_size > 1000


def test_generate_final_selection_report():
    candidates = get_model_candidates_data()
    scores = compute_mcda_scores(candidates)
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_rep = Path(tmp_dir) / "test_final_report.md"
        res = generate_final_selection_report(candidates, scores, out_rep)
        assert res.exists()
        content = res.read_text(encoding="utf-8")
        assert "DenseNet-121" in content
        assert "Multi-Criteria Decision Analysis" in content
        assert "Production Model Specification" in content

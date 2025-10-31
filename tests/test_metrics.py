"""Tests for metrics calculation."""

import numpy as np
import pytest

from euos25.utils.metrics import (
    aggregate_fold_metrics,
    calc_metrics,
    calc_pr_auc,
    calc_roc_auc,
    calc_spearman,
)


def test_calc_roc_auc():
    """Test ROC AUC calculation."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.2, 0.8, 0.9])

    auc = calc_roc_auc(y_true, y_pred)
    assert 0.0 <= auc <= 1.0
    assert auc == 1.0  # Perfect predictions


def test_calc_pr_auc():
    """Test PR AUC calculation."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.2, 0.8, 0.9])

    pr_auc = calc_pr_auc(y_true, y_pred)
    assert 0.0 <= pr_auc <= 1.0
    assert pr_auc == 1.0  # Perfect predictions


def test_calc_spearman():
    """Test Spearman correlation."""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2, 5.1])

    corr = calc_spearman(y_true, y_pred)
    assert -1.0 <= corr <= 1.0
    assert corr > 0.9  # Strong positive correlation


def test_calc_metrics():
    """Test multiple metrics calculation."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = calc_metrics(y_true, y_pred, metrics=["roc_auc", "pr_auc"])

    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0


def test_aggregate_fold_metrics():
    """Test fold metrics aggregation."""
    fold_metrics = [
        {"roc_auc": 0.8, "pr_auc": 0.75},
        {"roc_auc": 0.85, "pr_auc": 0.78},
        {"roc_auc": 0.82, "pr_auc": 0.76},
    ]

    aggregated = aggregate_fold_metrics(fold_metrics)

    assert "roc_auc" in aggregated
    assert "pr_auc" in aggregated

    # Check statistics
    for metric_name in ["roc_auc", "pr_auc"]:
        assert "mean" in aggregated[metric_name]
        assert "std" in aggregated[metric_name]
        assert "min" in aggregated[metric_name]
        assert "max" in aggregated[metric_name]

        # Check mean is correct
        values = [m[metric_name] for m in fold_metrics]
        assert np.isclose(aggregated[metric_name]["mean"], np.mean(values))


def test_metrics_with_edge_cases():
    """Test metrics with edge cases."""
    # All same class
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([0.8, 0.9, 0.7, 0.85])

    # Should handle gracefully (return nan or raise)
    auc = calc_roc_auc(y_true, y_pred)
    assert np.isnan(auc)  # Cannot compute AUC with single class

"""Metrics calculation utilities."""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


def calc_roc_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate ROC AUC score.

    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities

    Returns:
        ROC AUC score
    """
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError as e:
        logger.warning(f"Error calculating ROC AUC: {e}")
        return np.nan


def calc_pr_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Precision-Recall AUC score.

    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities

    Returns:
        PR AUC score
    """
    try:
        return average_precision_score(y_true, y_pred)
    except ValueError as e:
        logger.warning(f"Error calculating PR AUC: {e}")
        return np.nan


def calc_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Spearman correlation.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Spearman correlation coefficient
    """
    try:
        corr, _ = spearmanr(y_true, y_pred)
        return corr if not np.isnan(corr) else 0.0
    except ValueError as e:
        logger.warning(f"Error calculating Spearman: {e}")
        return np.nan


def calc_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Calculate multiple metrics.

    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        metrics: List of metric names (default: ["roc_auc", "pr_auc"])

    Returns:
        Dictionary of metric names to scores
    """
    if metrics is None:
        metrics = ["roc_auc", "pr_auc"]

    results = {}
    for metric_name in metrics:
        if metric_name == "roc_auc":
            results[metric_name] = calc_roc_auc(y_true, y_pred)
        elif metric_name == "pr_auc":
            results[metric_name] = calc_pr_auc(y_true, y_pred)
        elif metric_name == "spearman":
            results[metric_name] = calc_spearman(y_true, y_pred)
        else:
            logger.warning(f"Unknown metric: {metric_name}")

    return results


def aggregate_fold_metrics(
    fold_metrics: List[Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """Aggregate metrics across folds.

    Args:
        fold_metrics: List of metric dictionaries per fold

    Returns:
        Dictionary with mean and std for each metric
    """
    if not fold_metrics:
        return {}

    # Collect metrics by name
    metric_names = set()
    for fold in fold_metrics:
        metric_names.update(fold.keys())

    aggregated = {}
    for metric_name in metric_names:
        values = [fold.get(metric_name, np.nan) for fold in fold_metrics]
        values = [v for v in values if not np.isnan(v)]

        if values:
            aggregated[metric_name] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
            }
        else:
            aggregated[metric_name] = {
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
            }

    return aggregated


def save_fold_metrics(
    fold_metrics: List[Dict[str, float]],
    output_path: str,
) -> None:
    """Save fold metrics to CSV.

    Args:
        fold_metrics: List of metric dictionaries per fold
        output_path: Path to save CSV file
    """
    df = pd.DataFrame(fold_metrics)
    df.insert(0, "fold", range(len(df)))
    df.to_csv(output_path, index=False)
    logger.info(f"Saved fold metrics to {output_path}")

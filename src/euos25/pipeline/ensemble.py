"""Ensemble predictions using rank averaging."""

import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from euos25.utils.io import load_csv, save_csv

logger = logging.getLogger(__name__)


def rank_average(predictions: np.ndarray) -> np.ndarray:
    """Average predictions using rank transformation.

    Args:
        predictions: Array of shape (n_models, n_samples)

    Returns:
        Averaged predictions (n_samples,)
    """
    if predictions.ndim == 1:
        return predictions

    # Rank each model's predictions
    ranked = np.zeros_like(predictions)
    for i in range(predictions.shape[0]):
        ranked[i] = rankdata(predictions[i], method="average")

    # Average ranks
    avg_ranks = np.mean(ranked, axis=0)

    # Convert back to [0, 1] range
    normalized = (avg_ranks - avg_ranks.min()) / (avg_ranks.max() - avg_ranks.min())

    return normalized


def blend_predictions(
    pred_files: List[str],
    output_path: str,
    method: str = "rank_average",
    weights: List[float] = None,
) -> pd.DataFrame:
    """Blend multiple prediction files.

    Args:
        pred_files: List of paths to prediction CSV files
        output_path: Path to save blended predictions
        method: Blending method ("rank_average" or "weighted_average")
        weights: Optional weights for weighted average

    Returns:
        DataFrame with blended predictions
    """
    logger.info(f"Blending {len(pred_files)} prediction files")

    # Load all predictions
    all_preds = []
    mol_ids = None

    for pred_file in pred_files:
        df = load_csv(pred_file)

        if mol_ids is None:
            mol_ids = df["mol_id"].values
        else:
            # Verify consistent ordering
            if not np.array_equal(mol_ids, df["mol_id"].values):
                raise ValueError(f"Inconsistent mol_id ordering in {pred_file}")

        all_preds.append(df["prediction"].values)

    # Stack predictions
    predictions = np.vstack(all_preds)

    # Blend predictions
    if method == "rank_average":
        blended = rank_average(predictions)
    elif method == "weighted_average":
        if weights is None:
            weights = np.ones(len(pred_files)) / len(pred_files)
        else:
            weights = np.array(weights) / np.sum(weights)

        blended = np.average(predictions, axis=0, weights=weights)
    else:
        raise ValueError(f"Unknown blending method: {method}")

    # Create DataFrame
    result_df = pd.DataFrame({
        "mol_id": mol_ids,
        "prediction": blended,
    })

    # Save blended predictions
    save_csv(result_df, output_path, index=False)
    logger.info(f"Saved blended predictions to {output_path}")

    return result_df


def ensemble_from_directory(
    pred_dir: str,
    output_path: str,
    pattern: str = "*.csv",
    method: str = "rank_average",
) -> pd.DataFrame:
    """Ensemble all prediction files in a directory.

    Args:
        pred_dir: Directory containing prediction files
        output_path: Path to save ensemble predictions
        pattern: Glob pattern for prediction files
        method: Blending method

    Returns:
        DataFrame with ensemble predictions
    """
    pred_dir = Path(pred_dir)
    pred_files = sorted(pred_dir.glob(pattern))

    if not pred_files:
        raise ValueError(f"No prediction files found in {pred_dir}")

    logger.info(f"Found {len(pred_files)} prediction files")
    for pf in pred_files:
        logger.info(f"  - {pf.name}")

    return blend_predictions(
        [str(pf) for pf in pred_files],
        output_path,
        method=method,
    )

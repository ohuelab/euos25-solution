"""Plate normalization utilities."""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def normalize_by_plate(
    df: pd.DataFrame,
    value_col: str,
    plate_col: str = "plate_id",
    method: str = "median_iqr",
) -> pd.DataFrame:
    """Normalize values by plate using robust statistics.

    Args:
        df: DataFrame with plate information
        value_col: Column name to normalize
        plate_col: Column name for plate identifier
        method: Normalization method ("median_iqr" or "mean_std")

    Returns:
        DataFrame with normalized values (original + normalized column)
    """
    if plate_col not in df.columns:
        logger.warning(f"Plate column '{plate_col}' not found, skipping normalization")
        return df

    df = df.copy()
    normalized_col = f"{value_col}_norm"

    if method == "median_iqr":
        # Robust normalization using median and IQR
        plate_stats = df.groupby(plate_col)[value_col].agg(
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
        )
        plate_stats["iqr"] = plate_stats["q75"] - plate_stats["q25"]

        # Avoid division by zero
        plate_stats["iqr"] = plate_stats["iqr"].replace(0, 1)

        df = df.merge(
            plate_stats[["median", "iqr"]],
            left_on=plate_col,
            right_index=True,
            how="left",
        )
        df[normalized_col] = (df[value_col] - df["median"]) / df["iqr"]
        df = df.drop(columns=["median", "iqr"])

    elif method == "mean_std":
        # Standard normalization using mean and std
        plate_stats = df.groupby(plate_col)[value_col].agg(
            mean="mean",
            std="std",
        )

        # Avoid division by zero
        plate_stats["std"] = plate_stats["std"].replace(0, 1)

        df = df.merge(
            plate_stats,
            left_on=plate_col,
            right_index=True,
            how="left",
        )
        df[normalized_col] = (df[value_col] - df["mean"]) / df["std"]
        df = df.drop(columns=["mean", "std"])

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    logger.info(f"Applied {method} plate normalization to {value_col}")
    return df


def detect_plate_effects(
    df: pd.DataFrame,
    value_col: str,
    plate_col: str = "plate_id",
) -> pd.DataFrame:
    """Detect plate effects by analyzing within vs between plate variance.

    Args:
        df: DataFrame with plate information
        value_col: Column name to analyze
        plate_col: Column name for plate identifier

    Returns:
        DataFrame with plate statistics
    """
    if plate_col not in df.columns:
        logger.warning(f"Plate column '{plate_col}' not found")
        return pd.DataFrame()

    plate_stats = df.groupby(plate_col)[value_col].agg(
        count="count",
        mean="mean",
        std="std",
        median="median",
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
    )

    # Calculate coefficient of variation
    plate_stats["cv"] = plate_stats["std"] / plate_stats["mean"]

    # Overall statistics
    overall_mean = df[value_col].mean()
    overall_std = df[value_col].std()

    logger.info(f"Plate statistics for {value_col}:")
    logger.info(f"  Overall mean: {overall_mean:.4f}, std: {overall_std:.4f}")
    logger.info(f"  Plate means range: [{plate_stats['mean'].min():.4f}, "
                f"{plate_stats['mean'].max():.4f}]")
    logger.info(f"  Mean CV across plates: {plate_stats['cv'].mean():.4f}")

    return plate_stats

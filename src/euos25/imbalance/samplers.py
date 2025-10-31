"""Sampling strategies for imbalanced data."""

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def downsample_negatives(
    X: pd.DataFrame,
    y: np.ndarray,
    ratio: float = 1.0,
    seed: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Downsample negative class to reduce imbalance.

    Args:
        X: Features
        y: Binary labels
        ratio: Ratio of negatives to positives (1.0 = balanced)
        seed: Random seed

    Returns:
        Tuple of (X_resampled, y_resampled)
    """
    rng = np.random.RandomState(seed)

    # Split by class
    pos_mask = y == 1
    neg_mask = y == 0

    X_pos = X[pos_mask]
    y_pos = y[pos_mask]

    X_neg = X[neg_mask]
    y_neg = y[neg_mask]

    n_pos = len(y_pos)
    n_neg = len(y_neg)

    # Calculate target negative count
    target_neg = int(n_pos * ratio)

    if target_neg >= n_neg:
        logger.warning(
            f"Target negative count ({target_neg}) >= current count ({n_neg}), "
            "no downsampling needed"
        )
        return X, y

    # Downsample negatives
    neg_indices = rng.choice(n_neg, size=target_neg, replace=False)
    X_neg_down = X_neg.iloc[neg_indices]
    y_neg_down = y_neg[neg_indices]

    # Combine
    X_resampled = pd.concat([X_pos, X_neg_down], axis=0)
    y_resampled = np.concatenate([y_pos, y_neg_down])

    # Shuffle
    shuffle_indices = rng.permutation(len(y_resampled))
    X_resampled = X_resampled.iloc[shuffle_indices]
    y_resampled = y_resampled[shuffle_indices]

    logger.info(
        f"Downsampled negatives: {n_neg} -> {target_neg} "
        f"(pos={n_pos}, ratio={ratio:.2f})"
    )

    return X_resampled, y_resampled


def compute_sample_weights(
    y: np.ndarray,
    pos_weight: Optional[float] = None,
) -> np.ndarray:
    """Compute sample weights for imbalanced data.

    Args:
        y: Binary labels
        pos_weight: Weight for positive class (auto if None)

    Returns:
        Sample weights
    """
    if pos_weight is None:
        n_pos = np.sum(y)
        n_neg = len(y) - n_pos
        pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    weights = np.ones_like(y, dtype=np.float32)
    weights[y == 1] = pos_weight

    logger.info(f"Computed sample weights with pos_weight={pos_weight:.4f}")
    return weights


from typing import Optional  # Add missing import

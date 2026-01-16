"""Scaffold-based K-fold splitting."""

import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from euos25.utils.scaffold import balance_scaffold_splits, generate_scaffold_groups

logger = logging.getLogger(__name__)


def create_scaffold_splits(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    label_col: Optional[str] = None,
    n_splits: int = 5,
    scaffold_min_size: int = 10,
    seed: int = 42,
) -> Dict[str, List[int]]:
    """Create scaffold-based K-fold splits.

    Args:
        df: DataFrame with SMILES and optional labels
        smiles_col: Column name for SMILES
        label_col: Column name for labels (if available)
        n_splits: Number of folds
        scaffold_min_size: Minimum scaffold group size
        seed: Random seed

    Returns:
        Dictionary mapping fold names to index lists
    """
    # Generate scaffold groups
    smiles_list = df[smiles_col].tolist()
    scaffold_groups = generate_scaffold_groups(smiles_list, min_size=1)

    # Filter by minimum size if specified
    if scaffold_min_size > 1:
        # Small scaffolds get randomly assigned
        small_scaffolds = []
        large_scaffolds = {}

        for scaffold, indices in scaffold_groups.items():
            if len(indices) < scaffold_min_size:
                small_scaffolds.extend(indices)
            else:
                large_scaffolds[scaffold] = indices

        logger.info(
            f"Large scaffolds: {len(large_scaffolds)}, "
            f"Small scaffold samples: {len(small_scaffolds)}"
        )
        scaffold_groups = large_scaffolds

    # Get labels for balancing
    if label_col and label_col in df.columns:
        labels = df[label_col].values
    else:
        # If no labels, create dummy uniform labels
        labels = np.zeros(len(df))

    # Balance scaffolds across splits
    fold_indices = balance_scaffold_splits(
        scaffold_groups,
        labels,
        n_splits=n_splits,
        seed=seed,
    )

    # Add small scaffolds randomly if any
    if scaffold_min_size > 1 and small_scaffolds:
        rng = np.random.RandomState(seed)
        rng.shuffle(small_scaffolds)

        # Distribute evenly across folds
        for i, idx in enumerate(small_scaffolds):
            fold_indices[i % n_splits].append(idx)

    # Convert to dictionary format with train/valid splits
    splits = {}
    for fold_idx in range(n_splits):
        valid_indices = fold_indices[fold_idx]
        train_indices = []
        for other_fold in range(n_splits):
            if other_fold != fold_idx:
                train_indices.extend(fold_indices[other_fold])

        splits[f"fold_{fold_idx}"] = {
            "train": sorted(train_indices),
            "valid": sorted(valid_indices),
        }

    # Log split statistics
    for fold_name, fold_data in splits.items():
        train_size = len(fold_data["train"])
        valid_size = len(fold_data["valid"])
        logger.info(f"{fold_name}: train={train_size}, valid={valid_size}")

        if label_col and label_col in df.columns:
            train_pos = labels[fold_data["train"]].sum()
            valid_pos = labels[fold_data["valid"]].sum()
            train_rate = train_pos / train_size if train_size > 0 else 0
            valid_rate = valid_pos / valid_size if valid_size > 0 else 0
            logger.info(
                f"  Positive rates: train={train_rate:.4f}, valid={valid_rate:.4f}"
            )

    return splits


def splits_to_serializable(splits: Dict[str, Dict[str, List[int]]]) -> Dict[str, Dict[str, List[int]]]:
    """Convert splits to JSON-serializable format.

    Args:
        splits: Dictionary of fold splits

    Returns:
        JSON-serializable dictionary
    """
    return {
        fold_name: {
            "train": [int(idx) for idx in fold_data["train"]],
            "valid": [int(idx) for idx in fold_data["valid"]],
        }
        for fold_name, fold_data in splits.items()
    }


from typing import Optional  # Add missing import at the top

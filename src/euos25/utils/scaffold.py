"""Scaffold-based splitting utilities using RDKit."""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

logger = logging.getLogger(__name__)


def get_scaffold(smiles: str, include_chirality: bool = False) -> Optional[str]:
    """Get Murcko scaffold from SMILES.

    Args:
        smiles: SMILES string
        include_chirality: Whether to include chirality information

    Returns:
        Scaffold SMILES or None if invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, isomericSmiles=include_chirality)
    except Exception as e:
        logger.debug(f"Error getting scaffold for {smiles}: {e}")
        return None


def generate_scaffold_groups(
    smiles_list: List[str],
    min_size: int = 1,
) -> Dict[str, List[int]]:
    """Group molecules by their Murcko scaffold.

    Args:
        smiles_list: List of SMILES strings
        min_size: Minimum scaffold group size to keep

    Returns:
        Dictionary mapping scaffold to list of indices
    """
    scaffold_groups = defaultdict(list)

    for idx, smiles in enumerate(smiles_list):
        scaffold = get_scaffold(smiles)
        if scaffold:
            scaffold_groups[scaffold].append(idx)
        else:
            # Molecules without valid scaffold get unique group
            scaffold_groups[f"_invalid_{idx}"].append(idx)

    # Filter by minimum size
    if min_size > 1:
        scaffold_groups = {
            k: v for k, v in scaffold_groups.items() if len(v) >= min_size
        }

    logger.info(f"Generated {len(scaffold_groups)} scaffold groups")
    return dict(scaffold_groups)


def get_scaffold_distribution(
    scaffold_groups: Dict[str, List[int]],
    labels: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Get distribution statistics of scaffold groups.

    Args:
        scaffold_groups: Dictionary mapping scaffold to indices
        labels: Optional binary labels for positive rate calculation

    Returns:
        DataFrame with scaffold statistics
    """
    stats = []
    for scaffold, indices in scaffold_groups.items():
        stat = {
            "scaffold": scaffold,
            "count": len(indices),
        }
        if labels is not None:
            scaffold_labels = labels[indices]
            stat["pos_count"] = np.sum(scaffold_labels)
            stat["pos_rate"] = np.mean(scaffold_labels)

        stats.append(stat)

    df = pd.DataFrame(stats).sort_values("count", ascending=False)
    return df


def balance_scaffold_splits(
    scaffold_groups: Dict[str, List[int]],
    labels: np.ndarray,
    n_splits: int,
    seed: int = 42,
) -> List[List[int]]:
    """Balance scaffold groups across splits to maintain similar positive rates.

    Args:
        scaffold_groups: Dictionary mapping scaffold to indices
        labels: Binary labels
        n_splits: Number of splits
        seed: Random seed

    Returns:
        List of index lists for each split
    """
    rng = np.random.RandomState(seed)

    # Sort scaffolds by size (largest first)
    sorted_scaffolds = sorted(
        scaffold_groups.items(),
        key=lambda x: len(x[1]),
        reverse=True,
    )

    # Initialize splits
    splits = [[] for _ in range(n_splits)]
    split_counts = np.zeros(n_splits, dtype=int)
    split_pos_counts = np.zeros(n_splits, dtype=int)

    # Assign scaffolds to splits to balance size and positive rate
    for scaffold, indices in sorted_scaffolds:
        pos_count = np.sum(labels[indices])

        # Find split with smallest total count and positive imbalance
        split_pos_rates = np.where(
            split_counts > 0,
            split_pos_counts / split_counts,
            0.0,
        )
        target_pos_rate = np.mean(labels)

        # Score each split: prefer smaller count and closer to target pos rate
        scores = split_counts + 100 * np.abs(split_pos_rates - target_pos_rate)
        split_idx = np.argmin(scores)

        # Assign scaffold to split
        splits[split_idx].extend(indices)
        split_counts[split_idx] += len(indices)
        split_pos_counts[split_idx] += pos_count

    # Shuffle each split
    for split in splits:
        rng.shuffle(split)

    # Log split statistics
    for i, split in enumerate(splits):
        pos_rate = np.mean(labels[split]) if len(split) > 0 else 0.0
        logger.info(
            f"Split {i}: {len(split)} samples, "
            f"positive rate: {pos_rate:.4f}"
        )

    return splits

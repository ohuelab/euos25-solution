"""Tests for data splitting."""

import numpy as np
import pandas as pd
import pytest

from euos25.data.splits import create_scaffold_splits
from euos25.utils.scaffold import generate_scaffold_groups, get_scaffold


def test_get_scaffold():
    """Test scaffold generation."""
    # Benzene
    scaffold = get_scaffold("c1ccccc1")
    assert scaffold is not None
    assert "c" in scaffold.lower()

    # Invalid SMILES
    scaffold = get_scaffold("INVALID")
    assert scaffold is None


def test_generate_scaffold_groups():
    """Test scaffold grouping."""
    smiles_list = [
        "c1ccccc1",  # Benzene
        "c1ccccc1C",  # Toluene (same scaffold)
        "CCO",  # Ethanol (different scaffold)
    ]

    groups = generate_scaffold_groups(smiles_list, min_size=1)

    assert len(groups) >= 2  # At least 2 different scaffolds
    assert sum(len(v) for v in groups.values()) == 3  # All molecules assigned


def test_create_scaffold_splits():
    """Test scaffold-based splitting."""
    np.random.seed(42)

    # Create sample data
    df = pd.DataFrame({
        "ID": range(20),
        "SMILES": ["c1ccccc1C" + "C" * i for i in range(20)],  # Similar molecules
    })

    splits = create_scaffold_splits(
        df,
        n_splits=3,
        scaffold_min_size=1,
        seed=42,
    )

    # Check splits structure
    assert len(splits) == 3
    assert all(f"fold_{i}" in splits for i in range(3))

    # Check train/valid splits
    for fold_data in splits.values():
        assert "train" in fold_data
        assert "valid" in fold_data
        assert len(fold_data["train"]) > 0
        assert len(fold_data["valid"]) > 0

    # Check no overlap between train and valid
    for fold_data in splits.values():
        train_set = set(fold_data["train"])
        valid_set = set(fold_data["valid"])
        assert len(train_set & valid_set) == 0

    # Check all samples covered
    all_indices = set()
    for fold_data in splits.values():
        all_indices.update(fold_data["train"])
        all_indices.update(fold_data["valid"])
    assert len(all_indices) == 20


def test_create_scaffold_splits_with_labels():
    """Test scaffold-based splitting with label balancing."""
    np.random.seed(42)

    # Create imbalanced data
    df = pd.DataFrame({
        "ID": range(30),
        "SMILES": ["c1ccccc1C" + "C" * i for i in range(30)],
        "label": [0] * 25 + [1] * 5,  # Imbalanced
    })

    splits = create_scaffold_splits(
        df,
        label_col="label",
        n_splits=3,
        scaffold_min_size=1,
        seed=42,
    )

    # Check positive rates are similar across folds
    pos_rates = []
    for fold_data in splits.values():
        valid_labels = df.loc[fold_data["valid"], "label"]
        pos_rate = valid_labels.mean()
        pos_rates.append(pos_rate)

    # Positive rates should be similar (within reasonable tolerance)
    assert np.std(pos_rates) < 0.2  # Not too different

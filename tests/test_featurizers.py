"""Tests for featurizers."""

import numpy as np
import pandas as pd
import pytest

from euos25.featurizers.conj_proxy import ConjugationProxyFeaturizer
from euos25.featurizers.ecfp import ECFPFeaturizer
from euos25.featurizers.rdkit2d import RDKit2DFeaturizer


@pytest.fixture
def sample_smiles():
    """Sample SMILES for testing."""
    return pd.DataFrame({
        "ID": [1, 2, 3],
        "SMILES": [
            "CCO",  # Ethanol
            "c1ccccc1",  # Benzene
            "C1=CC=C(C=C1)C=O",  # Benzaldehyde
        ]
    })


def test_ecfp_featurizer(sample_smiles):
    """Test ECFP featurizer."""
    featurizer = ECFPFeaturizer(radius=2, n_bits=512, use_counts=True)

    features = featurizer.transform(sample_smiles)

    assert len(features) == 3
    assert len(features.columns) == 512
    assert all(features.columns.str.startswith("ecfp_"))
    assert features.dtypes[0] in [np.int8, np.int32]


def test_rdkit2d_featurizer(sample_smiles):
    """Test RDKit 2D featurizer."""
    featurizer = RDKit2DFeaturizer()

    features = featurizer.transform(sample_smiles)

    assert len(features) == 3
    assert len(features.columns) > 0
    assert all(features.columns.str.startswith("rdkit2d_"))
    assert features.dtypes[0] in [np.float32, np.float64]


def test_conj_proxy_featurizer(sample_smiles):
    """Test conjugation proxy featurizer."""
    featurizer = ConjugationProxyFeaturizer(L_cut=2)

    features = featurizer.transform(sample_smiles)

    assert len(features) == 3
    assert len(features.columns) == 12  # Fixed number of features
    assert all(features.columns.str.startswith("conj_proxy_"))
    assert features.dtypes[0] in [np.float32, np.float64]


def test_featurizer_with_invalid_smiles():
    """Test featurizer handles invalid SMILES."""
    df = pd.DataFrame({
        "ID": [1, 2],
        "SMILES": ["CCO", "INVALID"]
    })

    featurizer = ECFPFeaturizer(n_bits=128)
    features = featurizer.transform(df)

    # Should handle invalid SMILES gracefully
    assert len(features) == 2
    assert features.iloc[0].sum() > 0  # Valid SMILES has features
    assert features.iloc[1].sum() == 0  # Invalid SMILES filled with zeros

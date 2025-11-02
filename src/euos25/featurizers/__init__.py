"""Featurizer modules for molecular features."""

from euos25.featurizers.ecfp import ECFPFeaturizer
from euos25.featurizers.rdkit2d import RDKit2DFeaturizer

try:
    from euos25.featurizers.chemeleon import ChemeleonFeaturizer

    CHEMELEON_AVAILABLE = True
except ImportError:
    CHEMELEON_AVAILABLE = False
    ChemeleonFeaturizer = None

__all__ = [
    "ECFPFeaturizer",
    "RDKit2DFeaturizer",
    "ChemeleonFeaturizer",
    "CHEMELEON_AVAILABLE",
]

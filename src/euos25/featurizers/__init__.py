"""Featurizer modules for molecular features."""

from euos25.featurizers.ecfp import ECFPFeaturizer
from euos25.featurizers.mordred import MordredFeaturizer
from euos25.featurizers.rdkit2d import RDKit2DFeaturizer

try:
    from euos25.featurizers.chemeleon import ChemeleonFeaturizer

    CHEMELEON_AVAILABLE = True
except ImportError:
    CHEMELEON_AVAILABLE = False
    ChemeleonFeaturizer = None

try:
    from euos25.featurizers.chemberta import ChemBERTaFeaturizer

    CHEMBERTA_AVAILABLE = True
except ImportError:
    CHEMBERTA_AVAILABLE = False
    ChemBERTaFeaturizer = None

__all__ = [
    "ECFPFeaturizer",
    "RDKit2DFeaturizer",
    "MordredFeaturizer",
    "ChemeleonFeaturizer",
    "CHEMELEON_AVAILABLE",
    "ChemBERTaFeaturizer",
    "CHEMBERTA_AVAILABLE",
]

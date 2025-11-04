"""Model modules for classification."""

from euos25.models.lgbm import LGBMClassifier

try:
    from euos25.models.chemprop import ChemPropModel

    CHEMPROP_AVAILABLE = True
except ImportError:
    CHEMPROP_AVAILABLE = False
    ChemPropModel = None

try:
    from euos25.models.unimol import UniMolModel, UNIMOL_AVAILABLE
except ImportError:
    UNIMOL_AVAILABLE = False
    UniMolModel = None

__all__ = ["LGBMClassifier", "ChemPropModel", "CHEMPROP_AVAILABLE", "UniMolModel", "UNIMOL_AVAILABLE"]

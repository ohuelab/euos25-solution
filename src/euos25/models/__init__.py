"""Model modules for classification."""

from euos25.models.lgbm import LGBMClassifier

try:
    from euos25.models.chemprop import ChemPropModel

    CHEMPROP_AVAILABLE = True
except ImportError:
    CHEMPROP_AVAILABLE = False
    ChemPropModel = None

__all__ = ["LGBMClassifier", "ChemPropModel", "CHEMPROP_AVAILABLE"]

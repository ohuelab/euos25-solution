"""Base classes and protocols for featurizers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Protocol

import pandas as pd


class Featurizer(Protocol):
    """Protocol for featurizer implementations."""

    name: str

    def fit(self, df: pd.DataFrame) -> "Featurizer":
        """Fit featurizer on data (usually no-op for most featurizers).

        Args:
            df: DataFrame with SMILES column

        Returns:
            Self
        """
        ...

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform SMILES to features.

        Args:
            df: DataFrame with SMILES column

        Returns:
            DataFrame with feature columns (prefixed with featurizer name)
        """
        ...

    def get_params(self) -> Dict[str, Any]:
        """Get featurizer parameters.

        Returns:
            Dictionary of parameters
        """
        ...


class BaseFeaturizer(ABC):
    """Abstract base class for featurizers."""

    def __init__(self, name: str, **params: Any):
        """Initialize featurizer.

        Args:
            name: Featurizer name (used as column prefix)
            **params: Featurizer-specific parameters
        """
        self.name = name
        self.params = params

    def fit(self, df: pd.DataFrame) -> "BaseFeaturizer":
        """Fit featurizer on data.

        Args:
            df: DataFrame with SMILES column

        Returns:
            Self
        """
        # Most featurizers don't need fitting
        return self

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform SMILES to features.

        Args:
            df: DataFrame with SMILES column

        Returns:
            DataFrame with feature columns
        """
        pass

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step.

        Args:
            df: DataFrame with SMILES column

        Returns:
            DataFrame with feature columns
        """
        return self.fit(df).transform(df)

    def get_params(self) -> Dict[str, Any]:
        """Get featurizer parameters.

        Returns:
            Dictionary of parameters
        """
        return {"name": self.name, **self.params}

    def __repr__(self) -> str:
        """String representation."""
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.__class__.__name__}(name={self.name}, {params_str})"

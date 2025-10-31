"""Base classes and protocols for models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Protocol

import numpy as np
import pandas as pd


class ClfModel(Protocol):
    """Protocol for classification model implementations."""

    name: str

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        eval_set: Optional[tuple] = None,
    ) -> "ClfModel":
        """Fit model on training data.

        Args:
            X: Training features
            y: Training labels
            sample_weight: Optional sample weights
            eval_set: Optional validation set tuple (X_val, y_val)

        Returns:
            Self
        """
        ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities.

        Args:
            X: Features

        Returns:
            Predicted probabilities (shape: (n_samples,))
        """
        ...

    def save(self, path: str) -> None:
        """Save model to file.

        Args:
            path: Path to save model
        """
        ...

    @classmethod
    def load(cls, path: str) -> "ClfModel":
        """Load model from file.

        Args:
            path: Path to load model from

        Returns:
            Loaded model
        """
        ...

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters.

        Returns:
            Dictionary of parameters
        """
        ...


class BaseClfModel(ABC):
    """Abstract base class for classification models."""

    def __init__(self, name: str, **params: Any):
        """Initialize model.

        Args:
            name: Model name
            **params: Model-specific parameters
        """
        self.name = name
        self.params = params

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        eval_set: Optional[tuple] = None,
    ) -> "BaseClfModel":
        """Fit model on training data."""
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to file."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseClfModel":
        """Load model from file."""
        pass

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {"name": self.name, **self.params}

    def __repr__(self) -> str:
        """String representation."""
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.__class__.__name__}(name={self.name}, {params_str})"

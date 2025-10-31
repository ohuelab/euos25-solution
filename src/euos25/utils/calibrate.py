"""Probability calibration utilities."""

import logging
from typing import Literal, Optional

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)


class ProbabilityCalibrator:
    """Calibrate probability predictions using isotonic regression or sigmoid."""

    def __init__(self, method: Literal["isotonic", "sigmoid"] = "isotonic"):
        """Initialize calibrator.

        Args:
            method: Calibration method ("isotonic" or "sigmoid")
        """
        self.method = method
        self.calibrator: Optional[IsotonicRegression] = None

        if method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
        else:
            raise NotImplementedError(f"Method {method} not yet implemented")

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> "ProbabilityCalibrator":
        """Fit calibrator on validation data.

        Args:
            y_true: True binary labels
            y_pred: Predicted probabilities

        Returns:
            Fitted calibrator
        """
        if self.calibrator is None:
            raise ValueError("Calibrator not initialized")

        # Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            logger.warning("No valid samples for calibration")
            return self

        self.calibrator.fit(y_pred, y_true)
        logger.info(f"Fitted {self.method} calibrator on {len(y_true)} samples")
        return self

    def transform(self, y_pred: np.ndarray) -> np.ndarray:
        """Transform probabilities using fitted calibrator.

        Args:
            y_pred: Predicted probabilities

        Returns:
            Calibrated probabilities
        """
        if self.calibrator is None:
            raise ValueError("Calibrator not fitted")

        return self.calibrator.predict(y_pred)

    def fit_transform(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Fit and transform in one step.

        Args:
            y_true: True binary labels
            y_pred: Predicted probabilities

        Returns:
            Calibrated probabilities
        """
        self.fit(y_true, y_pred)
        return self.transform(y_pred)


def calibrate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    method: str = "isotonic",
) -> np.ndarray:
    """Calibrate probability predictions.

    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        method: Calibration method

    Returns:
        Calibrated probabilities
    """
    calibrator = ProbabilityCalibrator(method=method)
    return calibrator.fit_transform(y_true, y_pred)

"""Configuration management using Pydantic."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from euos25.utils.io import load_yaml


class FeaturizerConfig(BaseModel):
    """Configuration for a single featurizer."""

    name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    """Configuration for model."""

    name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class ImbalanceConfig(BaseModel):
    """Configuration for imbalance handling."""

    use_pos_weight: bool = True
    pos_weight_from_data: bool = True
    pos_weight_value: Optional[float] = None
    pos_weight_multiplier: Optional[float] = None  # Multiplier for tuned pos_weight
    use_sampling: bool = False
    sampling_ratio: Optional[float] = None
    use_focal_loss: bool = False
    focal_alpha: Optional[float] = None  # Auto-computed from data if None
    focal_gamma: float = 2.0  # Focusing parameter (typically 2-3)


class PlatesConfig(BaseModel):
    """Configuration for plate normalization."""

    normalize: bool = False
    method: str = "median_iqr"
    plate_col: str = "plate_id"


class OptunaConfig(BaseModel):
    """Configuration for Optuna hyperparameter optimization."""

    enable: bool = False
    n_trials: int = 100
    timeout: Optional[int] = None  # Timeout in seconds
    study_name: Optional[str] = None

    # Parameter search ranges
    # LGBM parameters
    learning_rate_min: float = 0.01
    learning_rate_max: float = 0.1
    num_leaves_min: int = 31
    num_leaves_max: int = 511
    max_depth_min: int = 3
    max_depth_max: int = 12
    subsample_min: float = 0.5
    subsample_max: float = 1.0
    colsample_bytree_min: float = 0.5
    colsample_bytree_max: float = 1.0
    min_child_samples_min: int = 5
    min_child_samples_max: int = 100
    reg_alpha_min: float = 0.0
    reg_alpha_max: float = 1.0
    reg_lambda_min: float = 0.0
    reg_lambda_max: float = 1.0

    # Focal loss parameters (when use_focal_loss=True)
    focal_alpha_min: float = 0.1
    focal_alpha_max: float = 0.9
    focal_gamma_min: float = 1.0
    focal_gamma_max: float = 5.0

    # Imbalance handling (when use_pos_weight=True)
    pos_weight_multiplier_min: float = 0.5
    pos_weight_multiplier_max: float = 2.0


class Config(BaseModel):
    """Main configuration for EUOS25 pipeline."""

    seed: int = 42
    folds: int = 5
    scaffold_min_size: int = 10

    featurizers: List[FeaturizerConfig] = Field(default_factory=list)
    model: ModelConfig = Field(default_factory=lambda: ModelConfig(name="lgbm"))

    imbalance: ImbalanceConfig = Field(default_factory=ImbalanceConfig)
    plates: PlatesConfig = Field(default_factory=PlatesConfig)
    optuna: OptunaConfig = Field(default_factory=OptunaConfig)

    task: str = "y_fluo_any"
    metrics: List[str] = Field(default_factory=lambda: ["roc_auc", "pr_auc"])

    # Early stopping
    early_stopping_rounds: int = 50
    early_stopping_metric: str = "roc_auc"

    # Logging
    log_level: str = "INFO"

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            Config instance
        """
        data = load_yaml(path)
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of config
        """
        return self.model_dump()


def load_config(config_path: str) -> Config:
    """Load configuration from file.

    Args:
        config_path: Path to configuration file

    Returns:
        Config instance
    """
    return Config.from_yaml(config_path)

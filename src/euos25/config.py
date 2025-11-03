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


class ParamSearchConfig(BaseModel):
    """Configuration for a single parameter search.

    Each parameter can specify:
    - type: "float", "int", or "categorical"
    - min/max: for float/int types
    - choices: for categorical type
    - log: whether to use log scale (for float/int)
    """

    type: str  # "float", "int", or "categorical"
    min: Optional[float] = None
    max: Optional[float] = None
    choices: Optional[List[Any]] = None
    log: bool = False  # Use log scale for float/int


class OptunaConfig(BaseModel):
    """Configuration for Optuna hyperparameter optimization.

    Parameters are organized by category for easier management.
    Each category can have its own set of parameters to tune.

    If a parameter is not specified in any category, its value from
    model.params or imbalance config will be used (fixed, not tuned).

    Example:
        optuna:
          enable: true
          n_trials: 100
          lgbm_params:
            learning_rate:
              type: float
              min: 0.01
              max: 0.1
              log: true
            num_leaves:
              type: int
              min: 63
              max: 511
          focal_params:
            focal_alpha:
              type: float
              min: 0.1
              max: 0.5
            focal_scale:
              type: float
              min: 50.0
              max: 200.0
          feature_groups:
            tune: true  # Enable feature group selection tuning
    """

    enable: bool = False
    n_trials: int = 100
    timeout: Optional[int] = None  # Timeout in seconds
    study_name: Optional[str] = None
    storage_enable: bool = True  # Enable SQLite storage for persistent study (allows resuming)
    storage_path: Optional[str] = None  # Path to SQLite database (auto-generated if None)

    # Parameter search configurations by category
    lgbm_params: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    catboost_params: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    imbalance_params: Dict[str, Dict[str, Any]] = Field(default_factory=dict)  # use_pos_weight, use_focal_loss
    focal_params: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    pos_weight_params: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    feature_groups: Dict[str, Any] = Field(default_factory=dict)  # Feature group selection tuning


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

    # Task configuration
    # For single task: task = "transmittance340"
    # For multi-task: tasks = ["transmittance340", "transmittance570"]
    task: Optional[str] = "y_fluo_any"
    tasks: Optional[List[str]] = None
    metrics: List[str] = Field(default_factory=lambda: ["roc_auc", "pr_auc"])

    # Early stopping
    early_stopping_rounds: int = 50
    early_stopping_metric: str = "roc_auc"

    # Logging
    log_level: str = "INFO"

    @property
    def is_multitask(self) -> bool:
        """Check if this is a multi-task configuration."""
        return self.tasks is not None and len(self.tasks) > 1

    @property
    def n_tasks(self) -> int:
        """Get number of tasks."""
        if self.tasks is not None:
            return len(self.tasks)
        return 1

    @property
    def task_names(self) -> List[str]:
        """Get list of task names."""
        if self.tasks is not None:
            return self.tasks
        return [self.task] if self.task else []

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

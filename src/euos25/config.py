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
    use_sampling: bool = False
    sampling_ratio: Optional[float] = None


class PlatesConfig(BaseModel):
    """Configuration for plate normalization."""

    normalize: bool = False
    method: str = "median_iqr"
    plate_col: str = "plate_id"


class Config(BaseModel):
    """Main configuration for EUOS25 pipeline."""

    seed: int = 42
    folds: int = 5
    scaffold_min_size: int = 10

    featurizers: List[FeaturizerConfig] = Field(default_factory=list)
    model: ModelConfig = Field(default_factory=lambda: ModelConfig(name="lgbm"))

    imbalance: ImbalanceConfig = Field(default_factory=ImbalanceConfig)
    plates: PlatesConfig = Field(default_factory=PlatesConfig)

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

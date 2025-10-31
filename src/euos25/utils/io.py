"""I/O utilities for loading and saving data."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_csv(path: Union[str, Path], **kwargs: Any) -> pd.DataFrame:
    """Load CSV file into DataFrame.

    Args:
        path: Path to CSV file
        **kwargs: Additional arguments to pass to pd.read_csv

    Returns:
        DataFrame with loaded data
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    logger.info(f"Loading CSV from {path}")
    df = pd.read_csv(path, **kwargs)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def save_csv(df: pd.DataFrame, path: Union[str, Path], **kwargs: Any) -> None:
    """Save DataFrame to CSV file.

    Args:
        df: DataFrame to save
        path: Path to save CSV file
        **kwargs: Additional arguments to pass to pd.to_csv
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving CSV to {path}")
    df.to_csv(path, **kwargs)
    logger.info(f"Saved {len(df)} rows to {path}")


def load_parquet(path: Union[str, Path], **kwargs: Any) -> pd.DataFrame:
    """Load Parquet file into DataFrame.

    Args:
        path: Path to Parquet file
        **kwargs: Additional arguments to pass to pd.read_parquet

    Returns:
        DataFrame with loaded data
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    logger.info(f"Loading Parquet from {path}")
    df = pd.read_parquet(path, **kwargs)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def save_parquet(df: pd.DataFrame, path: Union[str, Path], **kwargs: Any) -> None:
    """Save DataFrame to Parquet file.

    Args:
        df: DataFrame to save
        path: Path to save Parquet file
        **kwargs: Additional arguments to pass to pd.to_parquet
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving Parquet to {path}")
    df.to_parquet(path, **kwargs)
    logger.info(f"Saved {len(df)} rows to {path}")


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON file.

    Args:
        path: Path to JSON file

    Returns:
        Dictionary with loaded data
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    logger.info(f"Loading JSON from {path}")
    with open(path, "r") as f:
        data = json.load(f)
    return data


def save_json(data: Dict[str, Any], path: Union[str, Path], indent: int = 2) -> None:
    """Save dictionary to JSON file.

    Args:
        data: Dictionary to save
        path: Path to save JSON file
        indent: Indentation for JSON formatting
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving JSON to {path}")
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML file.

    Args:
        path: Path to YAML file

    Returns:
        Dictionary with loaded data
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    logger.info(f"Loading YAML from {path}")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data


def save_yaml(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save dictionary to YAML file.

    Args:
        data: Dictionary to save
        path: Path to save YAML file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving YAML to {path}")
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

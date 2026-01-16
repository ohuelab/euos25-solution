"""Data schema validation and column definitions."""

import logging
from typing import List, Optional

import pandas as pd
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class TrainSchema(BaseModel):
    """Schema for training data."""

    required_columns: List[str] = Field(
        default_factory=lambda: ["ID", "SMILES"]
    )
    optional_columns: List[str] = Field(
        default_factory=lambda: [
            "Transmittance",
            "Fluorescence",
            "plate_id",
        ]
    )


class TestSchema(BaseModel):
    """Schema for test data."""

    required_columns: List[str] = Field(
        default_factory=lambda: ["ID", "SMILES"]
    )


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str],
    optional_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Validate DataFrame has required columns.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        optional_columns: List of optional column names

    Returns:
        Validated DataFrame

    Raises:
        ValueError: If required columns are missing
    """
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if optional_columns:
        present_optional = set(optional_columns) & set(df.columns)
        logger.info(f"Optional columns present: {present_optional}")

    return df


def validate_train_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate training data schema.

    Args:
        df: Training DataFrame

    Returns:
        Validated DataFrame
    """
    schema = TrainSchema()
    return validate_dataframe(
        df,
        required_columns=schema.required_columns,
        optional_columns=schema.optional_columns,
    )


def validate_test_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate test data schema.

    Args:
        df: Test DataFrame

    Returns:
        Validated DataFrame
    """
    schema = TestSchema()
    return validate_dataframe(
        df,
        required_columns=schema.required_columns,
    )


def clean_smiles(df: pd.DataFrame, smiles_col: str = "SMILES") -> pd.DataFrame:
    """Clean SMILES strings.

    Args:
        df: DataFrame with SMILES column
        smiles_col: Name of SMILES column

    Returns:
        DataFrame with cleaned SMILES
    """
    df = df.copy()

    # Remove leading/trailing whitespace
    df[smiles_col] = df[smiles_col].str.strip()

    # Remove empty SMILES
    before_count = len(df)
    df = df[df[smiles_col].notna() & (df[smiles_col] != "")]
    after_count = len(df)

    if before_count != after_count:
        logger.info(f"Removed {before_count - after_count} rows with empty SMILES")

    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names and types.

    Args:
        df: DataFrame to standardize

    Returns:
        DataFrame with standardized columns
    """
    df = df.copy()

    # Rename columns to standard names if needed
    column_mapping = {
        "id": "ID",
        "N": "ID",  # Some files use N as ID column
        "smiles": "SMILES",
        "transmittance": "Transmittance",
        "fluorescence": "Fluorescence",
    }

    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df = df.rename(columns={old_col: new_col})
            logger.info(f"Renamed column: {old_col} -> {new_col}")

    # Handle Transmittance column with (qualitative) suffix
    for col in df.columns:
        if col.startswith("Transmittance") and col != "Transmittance":
            df = df.rename(columns={col: "Transmittance"})
            logger.info(f"Renamed column: {col} -> Transmittance")

    # Ensure ID is integer
    if "ID" in df.columns:
        df["ID"] = df["ID"].astype(int)

    return df

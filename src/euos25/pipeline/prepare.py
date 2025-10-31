"""Data preparation pipeline."""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from rdkit import Chem

from euos25.data.schema import clean_smiles, standardize_columns, validate_train_data
from euos25.utils.io import load_csv, save_csv

logger = logging.getLogger(__name__)


def normalize_smiles(smiles: str) -> Optional[str]:
    """Normalize SMILES string using RDKit.

    Args:
        smiles: Input SMILES string

    Returns:
        Normalized SMILES or None if invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return None


def get_inchikey_block(smiles: str) -> Optional[str]:
    """Get first block of InChIKey for duplicate detection.

    Args:
        smiles: SMILES string

    Returns:
        First block of InChIKey or None if invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        inchi_key = Chem.MolToInchiKey(mol)
        return inchi_key.split("-")[0] if inchi_key else None
    except Exception:
        return None


def prepare_data(
    input_path: str,
    output_path: Optional[str] = None,
    remove_duplicates: bool = True,
    normalize: bool = True,
) -> pd.DataFrame:
    """Prepare training data.

    Steps:
    1. Load data
    2. Standardize columns
    3. Clean SMILES
    4. Normalize SMILES (optional)
    5. Remove duplicates based on InChIKey (optional)
    6. Save prepared data (optional)

    Args:
        input_path: Path to input CSV
        output_path: Path to save prepared CSV (optional)
        remove_duplicates: Whether to remove duplicates
        normalize: Whether to normalize SMILES

    Returns:
        Prepared DataFrame
    """
    logger.info(f"Loading data from {input_path}")
    df = load_csv(input_path)

    # Standardize columns
    df = standardize_columns(df)

    # Validate schema
    df = validate_train_data(df)

    # Clean SMILES
    df = clean_smiles(df)

    # Normalize SMILES
    if normalize:
        logger.info("Normalizing SMILES")
        df["SMILES_normalized"] = df["SMILES"].apply(normalize_smiles)

        # Remove rows where normalization failed
        before = len(df)
        df = df[df["SMILES_normalized"].notna()]
        after = len(df)
        if before != after:
            logger.warning(f"Removed {before - after} rows with invalid SMILES")

        # Replace original SMILES
        df["SMILES"] = df["SMILES_normalized"]
        df = df.drop(columns=["SMILES_normalized"])

    # Remove duplicates
    if remove_duplicates:
        logger.info("Checking for duplicates using InChIKey")
        df["inchikey_block"] = df["SMILES"].apply(get_inchikey_block)

        # Remove rows where InChIKey generation failed
        before = len(df)
        df = df[df["inchikey_block"].notna()]
        after = len(df)
        if before != after:
            logger.warning(f"Removed {before - after} rows with invalid InChIKey")

        # Find and handle duplicates
        duplicates = df[df.duplicated(subset=["inchikey_block"], keep=False)]
        if len(duplicates) > 0:
            logger.info(f"Found {len(duplicates)} duplicate molecules")

            # For duplicates with labels, keep majority vote or first occurrence
            if "Transmittance" in df.columns or "Fluorescence" in df.columns:
                # Group by InChIKey and aggregate labels
                def aggregate_labels(group):
                    result = group.iloc[0].copy()
                    for col in ["Transmittance", "Fluorescence"]:
                        if col in group.columns:
                            # Take majority vote
                            result[col] = group[col].mode()[0] if len(group[col].mode()) > 0 else group[col].iloc[0]
                    return result

                df = df.groupby("inchikey_block", as_index=False).apply(aggregate_labels)
                df = df.reset_index(drop=True)
            else:
                # No labels, just keep first occurrence
                df = df.drop_duplicates(subset=["inchikey_block"], keep="first")

            logger.info(f"After deduplication: {len(df)} unique molecules")

        df = df.drop(columns=["inchikey_block"])

    # Log final statistics
    logger.info(f"Final dataset: {len(df)} samples")
    if "Transmittance" in df.columns:
        logger.info(f"  Transmittance positive: {df['Transmittance'].sum()}")
    if "Fluorescence" in df.columns:
        logger.info(f"  Fluorescence positive: {df['Fluorescence'].sum()}")

    # Save prepared data
    if output_path:
        save_csv(df, output_path, index=False)

    return df

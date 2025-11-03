"""Mordred molecular descriptors featurizer."""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from mordred import Calculator, descriptors
from mordred.error import Missing, Error
from rdkit import Chem

from euos25.featurizers.base import BaseFeaturizer

logger = logging.getLogger(__name__)


class MordredFeaturizer(BaseFeaturizer):
    """Mordred molecular descriptors featurizer.

    This featurizer uses the Mordred library to compute comprehensive
    2D molecular descriptors (approximately 1800 descriptors).
    """

    def __init__(
        self,
        name: str = "mordred",
        descriptor_names: Optional[List[str]] = None,
        ignore_3d: bool = True,
    ):
        """Initialize Mordred featurizer.

        Args:
            name: Feature name prefix
            descriptor_names: List of descriptor names to compute (None = all)
            ignore_3d: Whether to ignore 3D descriptors (default: True)
        """
        super().__init__(name=name, descriptor_names=descriptor_names, ignore_3d=ignore_3d)
        self.descriptor_names = descriptor_names
        self.ignore_3d = ignore_3d

        # Initialize Mordred calculator
        if self.descriptor_names is None:
            # Use all descriptors
            self.calc = Calculator(descriptors, ignore_3D=self.ignore_3d)
            self.descriptor_names = [str(d) for d in self.calc.descriptors]
        else:
            # Use specified descriptors
            desc_list = []
            for desc_name in self.descriptor_names:
                # Try to find descriptor by name
                desc = getattr(descriptors, desc_name, None)
                if desc is None:
                    raise ValueError(f"Invalid descriptor name: {desc_name}")
                desc_list.append(desc)
            self.calc = Calculator(desc_list, ignore_3D=self.ignore_3d)
            self.descriptor_names = [str(d) for d in desc_list]

        logger.info(f"Initialized Mordred calculator with {len(self.descriptor_names)} descriptors")

    def _smiles_to_descriptors(self, smiles: str) -> Optional[np.ndarray]:
        """Convert SMILES to descriptor array using Mordred.

        Args:
            smiles: SMILES string

        Returns:
            Descriptor array or None if invalid
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            # Calculate descriptors using Mordred
            desc_values = self.calc(mol)

            descriptors = []
            for desc_val in desc_values:
                # Handle Mordred error types
                if isinstance(desc_val, (Error, Missing)):
                    # Replace error with 0.0
                    descriptors.append(0.0)
                else:
                    # Convert to float and handle inf/nan
                    try:
                        val = float(desc_val)
                        if np.isnan(val) or np.isinf(val):
                            val = 0.0
                        descriptors.append(val)
                    except (ValueError, TypeError):
                        descriptors.append(0.0)

            return np.array(descriptors, dtype=np.float32)

        except Exception as e:
            logger.debug(f"Error converting SMILES to descriptors: {smiles}, {e}")
            return None

    def transform(self, df: pd.DataFrame, smiles_col: str = "SMILES") -> pd.DataFrame:
        """Transform SMILES to Mordred descriptor features.

        Args:
            df: DataFrame with SMILES column
            smiles_col: Name of SMILES column

        Returns:
            DataFrame with descriptor feature columns
        """
        logger.info(f"Generating {self.name} features using Mordred ({len(self.descriptor_names)} descriptors)")

        # Generate descriptors
        desc_list = []
        valid_indices = []

        for idx, smiles in enumerate(df[smiles_col]):
            desc = self._smiles_to_descriptors(smiles)
            if desc is not None:
                desc_list.append(desc)
                valid_indices.append(idx)

        if not desc_list:
            logger.warning("No valid descriptors generated")
            return pd.DataFrame(index=df.index)

        # Create DataFrame
        desc_array = np.vstack(desc_list)
        columns = [f"{self.name}_{desc_name}" for desc_name in self.descriptor_names]

        result_df = pd.DataFrame(
            desc_array,
            columns=columns,
            index=df.index[valid_indices],
        )

        # Reindex to match input (fill missing with zeros)
        result_df = result_df.reindex(df.index, fill_value=0.0)

        logger.info(f"Generated {len(columns)} {self.name} features for {len(result_df)} samples")
        return result_df


def get_common_descriptors() -> List[str]:
    """Get list of commonly used Mordred descriptors.

    Returns:
        List of descriptor names
    """
    # Common Mordred descriptor names
    return [
        "MolWt",
        "MolLogP",
        "nHDonor",
        "nHAcceptor",
        "nRot",
        "TPSA",
        "nAromaticRing",
        "nSaturatedRing",
        "nAliphaticRing",
        "nRing",
        "FpDensityMorgan3",
        "nHeavyAtom",
    ]


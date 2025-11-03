"""RDKit 2D molecular descriptors featurizer."""

import logging
from functools import partial
from typing import List, Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

from euos25.featurizers.base import BaseFeaturizer

logger = logging.getLogger(__name__)


class RDKit2DFeaturizer(BaseFeaturizer):
    """RDKit 2D molecular descriptors featurizer.

    This featurizer uses RDKit's built-in 2D molecular descriptors
    (from rdkit.Chem.Descriptors module).
    """

    def __init__(
        self,
        name: str = "rdkit2d",
        descriptor_names: Optional[List[str]] = None,
        n_jobs: int = -1,
    ):
        """Initialize RDKit 2D featurizer.

        Args:
            name: Feature name prefix
            descriptor_names: List of descriptor names to compute (None = all)
            n_jobs: Number of parallel jobs. If 1, no parallelization. -1 means use all CPUs.
        """
        super().__init__(name=name, descriptor_names=descriptor_names, n_jobs=n_jobs)
        self.descriptor_names = descriptor_names
        self.n_jobs = n_jobs

        # Get all available descriptors if not specified
        if self.descriptor_names is None:
            self.descriptor_names = [desc[0] for desc in Descriptors.descList]
        else:
            # Validate descriptor names
            available = {desc[0] for desc in Descriptors.descList}
            invalid = set(self.descriptor_names) - available
            if invalid:
                raise ValueError(f"Invalid descriptor names: {invalid}")

    def _smiles_to_descriptors(self, smiles: str) -> Optional[np.ndarray]:
        """Convert SMILES to descriptor array.

        Args:
            smiles: SMILES string

        Returns:
            Descriptor array or None if invalid
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            descriptors = []
            for desc_name in self.descriptor_names:
                try:
                    # Get descriptor function
                    desc_fn = getattr(Descriptors, desc_name)
                    value = desc_fn(mol)

                    # Handle inf/nan values
                    if np.isnan(value) or np.isinf(value):
                        value = 0.0

                    descriptors.append(value)
                except Exception as e:
                    logger.debug(f"Error computing {desc_name}: {e}")
                    descriptors.append(0.0)

            return np.array(descriptors, dtype=np.float32)

        except Exception as e:
            logger.debug(f"Error converting SMILES to descriptors: {smiles}, {e}")
            return None

    def transform(self, df: pd.DataFrame, smiles_col: str = "SMILES") -> pd.DataFrame:
        """Transform SMILES to RDKit 2D descriptor features.

        Args:
            df: DataFrame with SMILES column
            smiles_col: Name of SMILES column

        Returns:
            DataFrame with descriptor feature columns
        """
        logger.info(f"Generating {self.name} features using RDKit 2D descriptors ({len(self.descriptor_names)} descriptors, n_jobs={self.n_jobs})")

        smiles_list = df[smiles_col].tolist()

        # Determine number of jobs
        if self.n_jobs == -1:
            import multiprocessing
            n_jobs = multiprocessing.cpu_count()
        else:
            n_jobs = self.n_jobs

        # Generate descriptors (parallel or sequential)
        if n_jobs == 1:
            # Sequential processing
            desc_list = []
            valid_indices = []
            for idx, smiles in enumerate(smiles_list):
                desc = self._smiles_to_descriptors(smiles)
                if desc is not None:
                    desc_list.append(desc)
                    valid_indices.append(idx)
        else:
            # Parallel processing
            from multiprocessing import Pool

            # Create worker function
            worker_fn = partial(_rdkit2d_worker, descriptor_names=self.descriptor_names)

            # Process in parallel
            with Pool(n_jobs) as pool:
                results = pool.map(worker_fn, smiles_list)

            # Collect valid results
            desc_list = []
            valid_indices = []
            for idx, desc in enumerate(results):
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


def _rdkit2d_worker(smiles: str, descriptor_names: List[str]) -> Optional[np.ndarray]:
    """Worker function for parallel RDKit 2D descriptor computation.

    Args:
        smiles: SMILES string
        descriptor_names: List of descriptor names to compute

    Returns:
        Descriptor array or None if invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        descriptors = []
        for desc_name in descriptor_names:
            try:
                # Get descriptor function
                desc_fn = getattr(Descriptors, desc_name)
                value = desc_fn(mol)

                # Handle inf/nan values
                if np.isnan(value) or np.isinf(value):
                    value = 0.0

                descriptors.append(value)
            except Exception:
                descriptors.append(0.0)

        return np.array(descriptors, dtype=np.float32)

    except Exception:
        return None


def get_common_descriptors() -> List[str]:
    """Get list of commonly used RDKit 2D descriptors.

    Returns:
        List of descriptor names
    """
    return [
        "MolWt",
        "MolLogP",
        "NumHDonors",
        "NumHAcceptors",
        "NumRotatableBonds",
        "TPSA",
        "NumAromaticRings",
        "NumSaturatedRings",
        "NumAliphaticRings",
        "RingCount",
        "FractionCSP3",
        "NumHeteroatoms",
        "HeavyAtomCount",
    ]

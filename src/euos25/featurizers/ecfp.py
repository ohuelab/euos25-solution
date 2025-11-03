"""ECFP (Extended Connectivity Fingerprint) featurizer using RDKit."""

import logging
from functools import partial
from typing import Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from euos25.featurizers.base import BaseFeaturizer

logger = logging.getLogger(__name__)


class ECFPFeaturizer(BaseFeaturizer):
    """ECFP fingerprint featurizer."""

    def __init__(
        self,
        name: str = "ecfp",
        radius: int = 3,
        n_bits: int = 2048,
        use_counts: bool = True,
        use_features: bool = False,
        n_jobs: int = -1,
    ):
        """Initialize ECFP featurizer.

        Args:
            name: Feature name prefix
            radius: ECFP radius (3 = ECFP6)
            n_bits: Number of bits in fingerprint
            use_counts: If True, use count fingerprints; if False, use bit fingerprints
            use_features: If True, use feature-based (FCFP) instead of topological
            n_jobs: Number of parallel jobs. If 1, no parallelization. -1 means use all CPUs.
        """
        super().__init__(
            name=name,
            radius=radius,
            n_bits=n_bits,
            use_counts=use_counts,
            use_features=use_features,
            n_jobs=n_jobs,
        )
        self.radius = radius
        self.n_bits = n_bits
        self.use_counts = use_counts
        self.use_features = use_features
        self.n_jobs = n_jobs

    def _smiles_to_fp(self, smiles: str) -> Optional[np.ndarray]:
        """Convert SMILES to fingerprint array.

        Args:
            smiles: SMILES string

        Returns:
            Fingerprint array or None if invalid
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            if self.use_counts:
                # Count-based fingerprint
                fp = AllChem.GetHashedMorganFingerprint(
                    mol,
                    radius=self.radius,
                    nBits=self.n_bits,
                    useFeatures=self.use_features,
                )
                # Convert to numpy array
                arr = np.zeros(self.n_bits, dtype=np.int32)
                for idx, count in fp.GetNonzeroElements().items():
                    arr[idx] = count
                return arr
            else:
                # Bit-based fingerprint
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol,
                    radius=self.radius,
                    nBits=self.n_bits,
                    useFeatures=self.use_features,
                )
                return np.array(fp, dtype=np.int8)

        except Exception as e:
            logger.debug(f"Error converting SMILES to fingerprint: {smiles}, {e}")
            return None

    def transform(self, df: pd.DataFrame, smiles_col: str = "SMILES") -> pd.DataFrame:
        """Transform SMILES to ECFP features.

        Args:
            df: DataFrame with SMILES column
            smiles_col: Name of SMILES column

        Returns:
            DataFrame with ECFP feature columns
        """
        logger.info(f"Generating {self.name} features (radius={self.radius}, n_bits={self.n_bits}, n_jobs={self.n_jobs})")

        smiles_list = df[smiles_col].tolist()

        # Determine number of jobs
        if self.n_jobs == -1:
            import multiprocessing
            n_jobs = multiprocessing.cpu_count()
        else:
            n_jobs = self.n_jobs

        # Generate fingerprints (parallel or sequential)
        if n_jobs == 1:
            # Sequential processing
            fps = []
            valid_indices = []
            for idx, smiles in enumerate(smiles_list):
                fp = self._smiles_to_fp(smiles)
                if fp is not None:
                    fps.append(fp)
                    valid_indices.append(idx)
        else:
            # Parallel processing
            from multiprocessing import Pool

            # Create worker function
            worker_fn = partial(_ecfp_worker, radius=self.radius, n_bits=self.n_bits,
                              use_counts=self.use_counts, use_features=self.use_features)

            # Process in parallel
            with Pool(n_jobs) as pool:
                results = pool.map(worker_fn, smiles_list)

            # Collect valid results
            fps = []
            valid_indices = []
            for idx, (fp, smiles) in enumerate(zip(results, smiles_list)):
                if fp is not None:
                    fps.append(fp)
                    valid_indices.append(idx)

        if not fps:
            logger.warning("No valid fingerprints generated")
            return pd.DataFrame(index=df.index)

        # Create DataFrame
        fp_array = np.vstack(fps)
        columns = [f"{self.name}_{i}" for i in range(self.n_bits)]

        result_df = pd.DataFrame(
            fp_array,
            columns=columns,
            index=df.index[valid_indices],
        )

        # Reindex to match input (fill missing with zeros)
        result_df = result_df.reindex(df.index, fill_value=0)

        logger.info(f"Generated {len(columns)} {self.name} features for {len(result_df)} samples")
        return result_df


def _ecfp_worker(smiles: str, radius: int, n_bits: int, use_counts: bool, use_features: bool) -> Optional[np.ndarray]:
    """Worker function for parallel ECFP computation.

    Args:
        smiles: SMILES string
        radius: ECFP radius
        n_bits: Number of bits
        use_counts: Whether to use count fingerprints
        use_features: Whether to use feature-based fingerprints

    Returns:
        Fingerprint array or None if invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        if use_counts:
            # Count-based fingerprint
            fp = AllChem.GetHashedMorganFingerprint(
                mol,
                radius=radius,
                nBits=n_bits,
                useFeatures=use_features,
            )
            # Convert to numpy array
            arr = np.zeros(n_bits, dtype=np.int32)
            for idx, count in fp.GetNonzeroElements().items():
                arr[idx] = count
            return arr
        else:
            # Bit-based fingerprint
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol,
                radius=radius,
                nBits=n_bits,
                useFeatures=use_features,
            )
            return np.array(fp, dtype=np.int8)

    except Exception:
        return None

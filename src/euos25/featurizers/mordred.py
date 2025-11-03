"""Mordred molecular descriptors featurizer."""

import logging
from functools import partial
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
        n_jobs: int = -1,
    ):
        """Initialize Mordred featurizer.

        Args:
            name: Feature name prefix
            descriptor_names: List of descriptor names to compute (None = all)
            ignore_3d: Whether to ignore 3D descriptors (default: True)
            n_jobs: Number of parallel jobs. If 1, no parallelization. -1 means use all CPUs.
        """
        super().__init__(name=name, descriptor_names=descriptor_names, ignore_3d=ignore_3d, n_jobs=n_jobs)
        self.descriptor_names = descriptor_names
        self.ignore_3d = ignore_3d
        self.n_jobs = n_jobs

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
            Descriptor array with same length as self.descriptor_names, or None if SMILES is invalid.
            Errors in individual descriptors are set to NaN.
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            # Calculate descriptors using Mordred
            desc_values = self.calc(mol)

            descriptors = []
            for desc_val in desc_values:
                # Handle Mordred error types - set to NaN instead of 0.0
                if isinstance(desc_val, (Error, Missing)):
                    # Replace error with NaN
                    descriptors.append(np.nan)
                else:
                    # Convert to float and handle inf/nan
                    try:
                        val = float(desc_val)
                        if np.isnan(val):
                            # Keep NaN
                            descriptors.append(np.nan)
                        elif np.isinf(val):
                            # Replace inf with NaN
                            descriptors.append(np.nan)
                        else:
                            descriptors.append(val)
                    except (ValueError, TypeError):
                        # Conversion error - set to NaN
                        descriptors.append(np.nan)

            # Ensure we have the correct number of descriptors
            expected_len = len(self.descriptor_names)
            if len(descriptors) != expected_len:
                logger.warning(
                    f"Descriptor count mismatch: got {len(descriptors)}, expected {expected_len}. "
                    f"SMILES: {smiles}"
                )
                # Pad or truncate to match expected length
                if len(descriptors) < expected_len:
                    descriptors.extend([np.nan] * (expected_len - len(descriptors)))
                else:
                    descriptors = descriptors[:expected_len]

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
        from tqdm import tqdm

        logger.info(f"Generating {self.name} features using Mordred ({len(self.descriptor_names)} descriptors, n_jobs={self.n_jobs})")

        smiles_list = df[smiles_col].tolist()

        # Test the calculator with the first SMILES to catch initialization issues early
        if len(smiles_list) > 0:
            test_smiles = smiles_list[0]
            test_desc = self._smiles_to_descriptors(test_smiles)
            if test_desc is None:
                logger.warning(
                    f"Test SMILES failed to generate descriptors: {test_smiles}. "
                    f"This may indicate a problem with the Mordred calculator initialization."
                )
            else:
                logger.debug(f"Test SMILES successful: {test_smiles}, generated {len(test_desc)} descriptors")

        # Determine number of jobs
        if self.n_jobs == -1:
            import multiprocessing
            n_jobs = multiprocessing.cpu_count()
        else:
            n_jobs = self.n_jobs

        failed_count = 0
        # Generate descriptors (parallel or sequential)
        if n_jobs == 1:
            # Sequential processing with progress bar
            desc_list = []
            valid_indices = []
            for idx, smiles in enumerate(tqdm(smiles_list, desc="Computing Mordred descriptors")):
                desc = self._smiles_to_descriptors(smiles)
                if desc is not None:
                    desc_list.append(desc)
                    valid_indices.append(idx)
                else:
                    failed_count += 1
                    if failed_count <= 3:  # Log first 3 failures
                        logger.debug(f"Failed to generate descriptors for SMILES at index {idx}: {smiles}")
        else:
            # Parallel processing
            from multiprocessing import Pool

            # Need to serialize the calculator - create worker function with calculator initialization
            # Note: Mordred Calculator might not be pickleable, so we'll pass descriptor names
            # and recreate calculator in worker
            desc_list_for_worker = self.descriptor_names
            ignore_3d_for_worker = self.ignore_3d
            expected_num_descriptors = len(self.descriptor_names)

            worker_fn = partial(_mordred_worker,
                              descriptor_names=desc_list_for_worker,
                              ignore_3d=ignore_3d_for_worker,
                              expected_num_descriptors=expected_num_descriptors)

            # Process in parallel with progress bar
            try:
                with Pool(n_jobs) as pool:
                    results = list(tqdm(
                        pool.imap(worker_fn, smiles_list),
                        total=len(smiles_list),
                        desc="Computing Mordred descriptors"
                    ))
            except Exception as e:
                logger.error(
                    f"Parallel processing failed: {e}. "
                    f"Falling back to sequential processing.",
                    exc_info=True
                )
                # Fallback to sequential
                desc_list = []
                valid_indices = []
                for idx, smiles in enumerate(tqdm(smiles_list, desc="Computing Mordred descriptors (sequential fallback)")):
                    desc = self._smiles_to_descriptors(smiles)
                    if desc is not None:
                        desc_list.append(desc)
                        valid_indices.append(idx)
                    else:
                        failed_count += 1
                        if failed_count <= 3:
                            logger.debug(f"Failed to generate descriptors for SMILES at index {idx}: {smiles}")
            else:
                # Collect valid results
                desc_list = []
                valid_indices = []
                for idx, desc in enumerate(results):
                    if desc is not None:
                        desc_list.append(desc)
                        valid_indices.append(idx)
                    else:
                        failed_count += 1
                        if failed_count <= 3:
                            logger.debug(f"Failed to generate descriptors for SMILES at index {idx}: {smiles_list[idx]}")

        if not desc_list:
            error_msg = (
                f"No valid descriptors generated for any of {len(smiles_list)} SMILES. "
                f"This could indicate:\n"
                f"  1. All SMILES are invalid or cannot be parsed by RDKit\n"
                f"  2. Mordred calculator initialization failed\n"
                f"  3. All descriptor computations failed\n"
                f"Failed SMILES count: {failed_count}/{len(smiles_list)}"
            )
            logger.error(error_msg)
            # Try to get more diagnostic info
            if len(smiles_list) > 0:
                test_mol = Chem.MolFromSmiles(smiles_list[0])
                if test_mol is None:
                    logger.error(f"First SMILES cannot be parsed by RDKit: {smiles_list[0]}")
                else:
                    logger.error(f"First SMILES can be parsed by RDKit, but descriptors still failed")
            raise ValueError(error_msg)

        logger.info(f"Successfully generated descriptors for {len(desc_list)}/{len(smiles_list)} SMILES")

        if not desc_list:
            # No successful computations - return DataFrame with all NaN
            columns = [f"{self.name}_{desc_name}" for desc_name in self.descriptor_names]
            result_df = pd.DataFrame(
                np.full((len(df), len(columns)), np.nan, dtype=np.float32),
                columns=columns,
                index=df.index,
            )
            logger.warning(f"All SMILES failed - returning DataFrame with all NaN")
            return result_df

        # Create DataFrame
        desc_array = np.vstack(desc_list)

        # Verify descriptor count matches
        if desc_array.shape[1] != len(self.descriptor_names):
            logger.error(
                f"Descriptor count mismatch: array has {desc_array.shape[1]} columns, "
                f"but expected {len(self.descriptor_names)} descriptors"
            )
            # Try to fix by padding or truncating
            if desc_array.shape[1] < len(self.descriptor_names):
                padding = np.full(
                    (desc_array.shape[0], len(self.descriptor_names) - desc_array.shape[1]),
                    np.nan,
                    dtype=np.float32
                )
                desc_array = np.hstack([desc_array, padding])
            else:
                desc_array = desc_array[:, :len(self.descriptor_names)]

        columns = [f"{self.name}_{desc_name}" for desc_name in self.descriptor_names]

        result_df = pd.DataFrame(
            desc_array,
            columns=columns,
            index=df.index[valid_indices],
        )

        # Reindex to match input (fill missing with NaN instead of 0.0)
        result_df = result_df.reindex(df.index, fill_value=np.nan)

        logger.info(f"Generated {len(columns)} {self.name} features for {len(result_df)} samples")
        return result_df


def _mordred_worker(smiles: str, descriptor_names: List[str], ignore_3d: bool, expected_num_descriptors: int) -> Optional[np.ndarray]:
    """Worker function for parallel Mordred descriptor computation.

    Args:
        smiles: SMILES string
        descriptor_names: List of descriptor names to compute (can be string representations
                         that may not match attribute names - will fall back to all descriptors)
        ignore_3d: Whether to ignore 3D descriptors
        expected_num_descriptors: Expected number of descriptors (for validation)

    Returns:
        Descriptor array with expected_num_descriptors elements, or None if SMILES is invalid.
        Errors in individual descriptors are set to NaN.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Recreate calculator in worker (needed for multiprocessing)
        # Note: When descriptor_names comes from string conversion (str(d) for d in calc.descriptors),
        # the string names may not match attribute names in the descriptors module.
        # In that case, we fall back to using all descriptors.
        calc = None
        if descriptor_names is None or len(descriptor_names) == 0:
            # Use all descriptors
            calc = Calculator(descriptors, ignore_3D=ignore_3d)
        else:
            # Try to match descriptor names to attributes
            desc_list = []
            for desc_name in descriptor_names:
                desc = getattr(descriptors, desc_name, None)
                if desc is not None:
                    desc_list.append(desc)

            if desc_list and len(desc_list) == expected_num_descriptors:
                # Successfully matched descriptors and count matches
                calc = Calculator(desc_list, ignore_3D=ignore_3d)
            else:
                # No descriptors matched or count mismatch - use all descriptors
                # This happens when descriptor_names was created from str(d) conversions.
                calc = Calculator(descriptors, ignore_3D=ignore_3d)

        if calc is None:
            return None

        # Calculate descriptors using Mordred
        desc_values = calc(mol)

        result_descriptors = []
        for desc_val in desc_values:
            # Handle Mordred error types - set to NaN instead of 0.0
            if isinstance(desc_val, (Error, Missing)):
                # Replace error with NaN
                result_descriptors.append(np.nan)
            else:
                # Convert to float and handle inf/nan
                try:
                    val = float(desc_val)
                    if np.isnan(val):
                        # Keep NaN
                        result_descriptors.append(np.nan)
                    elif np.isinf(val):
                        # Replace inf with NaN
                        result_descriptors.append(np.nan)
                    else:
                        result_descriptors.append(val)
                except (ValueError, TypeError):
                    # Conversion error - set to NaN
                    result_descriptors.append(np.nan)

        # Ensure we have the correct number of descriptors
        if len(result_descriptors) != expected_num_descriptors:
            # Pad or truncate to match expected length
            if len(result_descriptors) < expected_num_descriptors:
                result_descriptors.extend([np.nan] * (expected_num_descriptors - len(result_descriptors)))
            else:
                result_descriptors = result_descriptors[:expected_num_descriptors]

        if not result_descriptors:
            return None

        return np.array(result_descriptors, dtype=np.float32)

    except Exception as e:
        # Log error in worker (will be visible in main process if using proper logging)
        # Note: multiprocessing workers may not have access to logger, so we just return None
        return None

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


"""Precompute similarity matrix for all molecules and save to disk.

This script computes the Tanimoto similarity matrix for all molecules in the training data
and saves it to disk. This allows the similarity features to be reused across different
parameter searches and experiments without recomputing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Any
import sys
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from multiprocessing import Pool, cpu_count

# Import utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from euos25.utils.io import load_yaml

# Task configuration
tasks = [
    ('fluo_340_450', 'train_fluo_340_450_prepared.csv'),
    ('fluo_480', 'train_fluo_480_prepared.csv'),
    ('trans_340', 'train_trans_340_prepared.csv'),
    ('trans_450', 'train_trans_450_prepared.csv'),
]

# Paths
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
processed_base_dir = PROJECT_ROOT / "data" / "processed"

# Try different feature directories for training data
feature_dirs = ['ecfp4', 'chemeleon', 'feats']

# Default parameters
DEFAULT_RADIUS = 3
DEFAULT_N_BITS = 2048


def load_train_data(task_key: str, prepared_file: str) -> pd.DataFrame:
    """Load training data for a task.

    Args:
        task_key: Task key
        prepared_file: Prepared data file name

    Returns:
        Training DataFrame with ID and SMILES columns
    """
    for feature_dir in feature_dirs:
        if feature_dir:
            candidate_path = processed_base_dir / feature_dir / prepared_file
        else:
            candidate_path = processed_base_dir / prepared_file

        if candidate_path.exists():
            df = pd.read_csv(candidate_path)
            return df

    raise FileNotFoundError(f"Training data not found for {task_key}")


def _compute_fingerprint_worker(args: Tuple[str, int, int]) -> Optional[Any]:
    """Worker function for parallel fingerprint computation.

    Args:
        args: Tuple of (smiles, radius, n_bits)

    Returns:
        Fingerprint or None if invalid
    """
    smiles, radius, n_bits = args
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        return fp
    except Exception:
        return None


def compute_similarity_matrix(
    df_train: pd.DataFrame,
    smiles_col: str = 'SMILES',
    radius: int = DEFAULT_RADIUS,
    n_bits: int = DEFAULT_N_BITS,
    n_jobs: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute similarity matrix for all molecules.

    Args:
        df_train: Training DataFrame with SMILES column
        smiles_col: Name of SMILES column
        radius: ECFP radius
        n_bits: Number of bits in fingerprint
        n_jobs: Number of parallel jobs. If None, uses cpu_count()

    Returns:
        Tuple of (similarity_matrix, ids_array)
        similarity_matrix: (n_molecules, n_molecules) array of Tanimoto similarities
        ids_array: (n_molecules,) array of molecule IDs in the same order
    """
    print(f"Computing similarity matrix for {len(df_train)} molecules...")
    print(f"  Parameters: radius={radius}, n_bits={n_bits}")

    # Get SMILES and IDs
    smiles_list = df_train[smiles_col].values
    ids = df_train['ID'].values

    # Compute fingerprints in parallel
    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)

    print(f"  Computing fingerprints using {n_jobs} processes...")
    worker_args = [(smiles, radius, n_bits) for smiles in smiles_list]
    fingerprints = []
    valid_indices = []

    with Pool(processes=n_jobs) as pool:
        results = list(tqdm(
            pool.imap(_compute_fingerprint_worker, worker_args),
            total=len(worker_args),
            desc="  Computing fingerprints"
        ))

    # Filter out None fingerprints and track valid indices
    for idx, fp in enumerate(results):
        if fp is not None:
            fingerprints.append(fp)
            valid_indices.append(idx)

    print(f"  Valid fingerprints: {len(fingerprints)}/{len(smiles_list)}")

    if len(fingerprints) == 0:
        raise ValueError("No valid fingerprints computed")

    # Compute similarity matrix
    print(f"  Computing similarity matrix ({len(fingerprints)} x {len(fingerprints)})...")
    similarity_matrix = np.zeros((len(fingerprints), len(fingerprints)), dtype=np.float32)

    # Compute similarities (symmetric matrix, so we only compute upper triangle)
    for i in tqdm(range(len(fingerprints)), desc="  Computing similarities"):
        # Compute similarities from i to all molecules (including i)
        sims = DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[i:])
        similarity_matrix[i, i:] = sims
        # Fill lower triangle (symmetric)
        similarity_matrix[i:, i] = sims

    # Get IDs for valid molecules
    valid_ids = ids[valid_indices]

    return similarity_matrix, valid_ids


def save_similarity_matrix(
    similarity_matrix: np.ndarray,
    ids: np.ndarray,
    output_path: Path,
    radius: int,
    n_bits: int
):
    """Save similarity matrix and metadata.

    Args:
        similarity_matrix: Similarity matrix
        ids: Molecule IDs
        output_path: Output directory
        radius: ECFP radius
        n_bits: Number of bits
    """
    output_path.mkdir(parents=True, exist_ok=True)

    # Save similarity matrix
    matrix_path = output_path / f"similarity_matrix_r{radius}_b{n_bits}.npy"
    np.save(matrix_path, similarity_matrix)
    print(f"  Saved similarity matrix to {matrix_path}")

    # Save IDs
    ids_path = output_path / f"similarity_matrix_ids_r{radius}_b{n_bits}.npy"
    np.save(ids_path, ids)
    print(f"  Saved IDs to {ids_path}")

    # Save metadata
    metadata = {
        'radius': radius,
        'n_bits': n_bits,
        'n_molecules': len(ids),
        'matrix_shape': similarity_matrix.shape,
        'matrix_dtype': str(similarity_matrix.dtype)
    }
    import json
    metadata_path = output_path / f"similarity_matrix_metadata_r{radius}_b{n_bits}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata to {metadata_path}")


def precompute_similarity_matrix_for_task(
    task_key: str,
    prepared_file: str,
    feature_dir: str = 'ecfp4',
    radius: int = DEFAULT_RADIUS,
    n_bits: int = DEFAULT_N_BITS,
    n_jobs: Optional[int] = None
):
    """Precompute similarity matrix for a single task.

    Args:
        task_key: Task key
        prepared_file: Prepared data file name
        feature_dir: Feature directory (used to determine output path)
        radius: ECFP radius
        n_bits: Number of bits
        n_jobs: Number of parallel jobs
    """
    print(f"\n{'='*70}")
    print(f"Precomputing similarity matrix for task: {task_key}")
    print(f"{'='*70}")

    # Load training data
    df_train = load_train_data(task_key, prepared_file)
    print(f"Loaded {len(df_train)} training samples")

    # Compute similarity matrix
    similarity_matrix, ids = compute_similarity_matrix(
        df_train=df_train,
        radius=radius,
        n_bits=n_bits,
        n_jobs=n_jobs
    )

    # Save to disk
    output_dir = processed_base_dir / feature_dir
    save_similarity_matrix(
        similarity_matrix=similarity_matrix,
        ids=ids,
        output_path=output_dir,
        radius=radius,
        n_bits=n_bits
    )

    print(f"✅ Completed similarity matrix computation for {task_key}")


def main(
    task_key: Optional[str] = None,
    feature_dir: str = 'ecfp4',
    radius: int = DEFAULT_RADIUS,
    n_bits: int = DEFAULT_N_BITS,
    n_jobs: Optional[int] = None
):
    """Main function to precompute similarity matrices.

    Args:
        task_key: Task key to process. If None, processes all tasks.
        feature_dir: Feature directory
        radius: ECFP radius
        n_bits: Number of bits
        n_jobs: Number of parallel jobs
    """
    print("="*70)
    print("Precompute Similarity Matrix")
    print("="*70)
    print(f"Parameters: radius={radius}, n_bits={n_bits}, feature_dir={feature_dir}")

    if task_key is not None:
        # Process single task
        task_info = next((t for t in tasks if t[0] == task_key), None)
        if task_info is None:
            print(f"❌ Unknown task_key: {task_key}")
            return
        precompute_similarity_matrix_for_task(
            task_key=task_key,
            prepared_file=task_info[1],
            feature_dir=feature_dir,
            radius=radius,
            n_bits=n_bits,
            n_jobs=n_jobs
        )
    else:
        # Process all tasks
        for task_key, prepared_file in tasks:
            try:
                precompute_similarity_matrix_for_task(
                    task_key=task_key,
                    prepared_file=prepared_file,
                    feature_dir=feature_dir,
                    radius=radius,
                    n_bits=n_bits,
                    n_jobs=n_jobs
                )
            except Exception as e:
                print(f"❌ Error processing {task_key}: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "="*70)
    print("✅ All similarity matrices computed!")
    print("="*70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Precompute similarity matrices for all tasks")
    parser.add_argument("--task", type=str, default=None, help="Task key to process (default: all tasks)")
    parser.add_argument("--feature-dir", type=str, default="ecfp4", help="Feature directory (default: ecfp4)")
    parser.add_argument("--radius", type=int, default=DEFAULT_RADIUS, help=f"ECFP radius (default: {DEFAULT_RADIUS})")
    parser.add_argument("--n-bits", type=int, default=DEFAULT_N_BITS, help=f"Number of bits (default: {DEFAULT_N_BITS})")
    parser.add_argument("--n-jobs", type=int, default=None, help="Number of parallel jobs (default: cpu_count - 1)")

    args = parser.parse_args()

    main(
        task_key=args.task,
        feature_dir=args.feature_dir,
        radius=args.radius,
        n_bits=args.n_bits,
        n_jobs=args.n_jobs
    )


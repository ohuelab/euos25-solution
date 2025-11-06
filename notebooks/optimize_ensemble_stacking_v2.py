"""Optimize stacking ensemble using meta-learners for each task (v2).

This script extends optimize_ensemble_stacking.py with:
1. Prediction result exploration (CV score-based selection and diversity filtering)
2. Ability to add molecular features (ECFP4, RDKit2D, etc.) to meta-model inputs
3. Derived features (sim_to_train_mean, cluster_density, model_std)
4. YAML configuration support

It uses nested CV where for each fold, a meta-learner is trained on
level-1 features (base model predictions + optional molecular features from other folds)
and predicts on the current fold's validation set.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from scipy.special import expit
from pathlib import Path
import optuna
from typing import Dict, List, Tuple, Optional, Any
import json
import sys
import yaml
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

# Import LGBMClassifier from the project
sys.path.insert(0, str(Path(__file__).parent.parent))
from euos25.models.lgbm import LGBMClassifier
from euos25.utils.io import load_parquet, load_yaml

# Task configuration: (task_key, single_task_name, label_col, prepared_file, multitask_task_name)
tasks = [
    ('fluo_340_450', 'y_fluo_any', 'Fluorescence', 'train_fluo_340_450_prepared.csv', 'fluorescence340_450'),
    ('fluo_480', 'y_fluo_any', 'Fluorescence', 'train_fluo_480_prepared.csv', 'fluorescence480'),
    ('trans_340', 'y_trans_any', 'Transmittance', 'train_trans_340_prepared.csv', 'transmittance340'),
    ('trans_450', 'y_trans_any', 'Transmittance', 'train_trans_450_prepared.csv', 'transmittance450'),
]

# Paths
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent  # notebooks/ -> project root
pred_base_dir = PROJECT_ROOT / "data" / "preds"
processed_base_dir = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "ensembles_stacking_v2"
models_base_dir = PROJECT_ROOT / "data" / "models"

# Try different feature directories for training data
feature_dirs = ['ecfp4', 'chemeleon', '']

# Try different splits directories
splits_dirs = ['ecfp4', 'chemprop', 'chemeleon', 'chemeleon_lgbm', '']

# Default parameters
DEFAULT_TOP_N = 10
DEFAULT_LAMBDA_CORRELATION = 0.1
DEFAULT_MAX_DIVERSE_MODELS = 15
DEFAULT_N_TRIALS = 50
DEFAULT_CLUSTER_N_CLUSTERS = 50

# Default config (can be overridden by YAML)
DEFAULT_CONFIG = {
    'use_additional_features': False,
    'feature_groups': ['ecfp4'],
    'feature_dir': 'ecfp4',
    'use_derived_features': False,
    'derived_features': ['sim_to_train_mean', 'cluster_density', 'model_std'],
    'top_n_models': DEFAULT_TOP_N,
    'lambda_correlation': DEFAULT_LAMBDA_CORRELATION,
    'max_diverse_models': DEFAULT_MAX_DIVERSE_MODELS,
    'n_trials': DEFAULT_N_TRIALS,
    'cluster_n_clusters': DEFAULT_CLUSTER_N_CLUSTERS,
}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file or use defaults.

    Args:
        config_path: Path to YAML config file. If None, uses defaults.

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        return DEFAULT_CONFIG.copy()

    config_path_obj = Path(config_path)
    if not config_path_obj.exists():
        print(f"⚠️  Config file not found: {config_path}, using defaults")
        return DEFAULT_CONFIG.copy()

    config = load_yaml(config_path)
    # Merge with defaults
    merged = DEFAULT_CONFIG.copy()
    merged.update(config)
    return merged


def load_cv_scores(
    task_key: str,
    task_name: str,
    mt_task_name: str,
    available_single: List[str],
    available_multitask: List[str]
) -> Dict[str, float]:
    """Load CV ROC-AUC scores for available models.

    Args:
        task_key: Task key (e.g., 'fluo_340_450')
        task_name: Single task name (e.g., 'y_fluo_any')
        mt_task_name: Multitask task name (e.g., 'fluorescence340_450')
        available_single: List of available single-task model names
        available_multitask: List of available multitask model names

    Returns:
        Dictionary mapping model_name (with prefix) to mean ROC-AUC score
    """
    cv_scores = {}

    # Load single-task model scores
    for model_name in available_single:
        model_dir = models_base_dir / model_name / task_key / task_name
        if not model_dir.exists():
            continue

        # Try common model types
        model_types = ['lgbm', 'catboost', 'chemprop', 'chemeleon', 'unimol']
        for model_type in model_types:
            metrics_path = model_dir / model_type / "cv_metrics.csv"
            if metrics_path.exists():
                try:
                    df = pd.read_csv(metrics_path)
                    if 'roc_auc' in df.columns:
                        mean_roc_auc = df['roc_auc'].mean()
                        cv_scores[f"single_{model_name}"] = mean_roc_auc
                        break
                except Exception as e:
                    print(f"  ⚠️  Error reading CV metrics for {model_name}/{model_type}: {e}")
                    continue

    # Load multitask model scores
    for model_name in available_multitask:
        model_dir = models_base_dir / model_name
        if not model_dir.exists():
            continue

        # Find task directory that contains this task
        for task_subdir in model_dir.iterdir():
            if not task_subdir.is_dir():
                continue

            if mt_task_name in task_subdir.name:
                # Try common model types
                model_types = ['lgbm', 'catboost', 'chemprop', 'chemeleon', 'unimol']
                for model_type in model_types:
                    metrics_path = task_subdir / model_type / "cv_metrics.csv"
                    if metrics_path.exists():
                        try:
                            df = pd.read_csv(metrics_path)
                            # For multitask, look for task-specific column
                            task_metric_col = f"{mt_task_name}_roc_auc"
                            if task_metric_col in df.columns:
                                mean_roc_auc = df[task_metric_col].mean()
                                cv_scores[model_name] = mean_roc_auc
                                break
                            # Fallback to avg_roc_auc if task-specific not available
                            elif 'avg_roc_auc' in df.columns:
                                mean_roc_auc = df['avg_roc_auc'].mean()
                                cv_scores[model_name] = mean_roc_auc
                                break
                        except Exception as e:
                            print(f"  ⚠️  Error reading CV metrics for {model_name}/{task_subdir.name}/{model_type}: {e}")
                            continue
                break

    return cv_scores


def select_top_models_by_cv_score(
    cv_scores: Dict[str, float],
    top_n: int = DEFAULT_TOP_N
) -> List[str]:
    """Select top N models by CV ROC-AUC score.

    Args:
        cv_scores: Dictionary mapping model_name to mean ROC-AUC
        top_n: Number of top models to select

    Returns:
        List of top N model names (sorted by score, descending)
    """
    if len(cv_scores) == 0:
        return []

    # Sort by score (descending) and take top N
    sorted_models = sorted(cv_scores.items(), key=lambda x: x[1], reverse=True)
    top_models = [model_name for model_name, _ in sorted_models[:top_n]]

    return top_models


def calculate_model_correlations(
    predictions: Dict[str, pd.Series]
) -> pd.DataFrame:
    """Calculate correlation matrix between model predictions.

    Args:
        predictions: Dictionary mapping model_name to prediction Series

    Returns:
        DataFrame with correlation matrix (model_name x model_name)
    """
    # Create DataFrame with all predictions aligned by ID
    all_ids = set()
    for pred in predictions.values():
        all_ids.update(pred.index)

    all_ids = sorted(all_ids)

    # Create DataFrame
    pred_df = pd.DataFrame(index=all_ids)
    for model_name, pred in predictions.items():
        # Align predictions to common IDs
        pred_aligned = pred.reindex(all_ids)
        pred_df[model_name] = pred_aligned

    # Calculate Pearson correlation
    corr_matrix = pred_df.corr()

    return corr_matrix


def select_diverse_models(
    candidate_models: List[str],
    cv_scores: Dict[str, float],
    corr_matrix: pd.DataFrame,
    lambda_correlation: float = DEFAULT_LAMBDA_CORRELATION,
    max_models: int = DEFAULT_MAX_DIVERSE_MODELS
) -> List[str]:
    """Select diverse models considering both CV score and correlation.

    Uses greedy selection: score = mean_roc_auc - lambda_correlation * mean_correlation

    Args:
        candidate_models: List of candidate model names
        cv_scores: Dictionary mapping model_name to mean ROC-AUC
        corr_matrix: Correlation matrix between models
        lambda_correlation: Penalty coefficient for correlation
        max_models: Maximum number of models to select

    Returns:
        List of selected diverse model names
    """
    if len(candidate_models) == 0:
        return []

    selected = []
    remaining = candidate_models.copy()

    # Filter to only models with CV scores
    remaining = [m for m in remaining if m in cv_scores]

    if len(remaining) == 0:
        return []

    # Select first model (highest score)
    first_model = max(remaining, key=lambda m: cv_scores[m])
    selected.append(first_model)
    remaining.remove(first_model)

    # Greedily select remaining models
    while len(selected) < max_models and len(remaining) > 0:
        best_model = None
        best_score = float('-inf')

        for candidate in remaining:
            # Calculate mean correlation with already selected models
            mean_corr = 0.0
            if len(selected) > 0:
                corrs = []
                for sel in selected:
                    if candidate in corr_matrix.index and sel in corr_matrix.columns:
                        corr_val = corr_matrix.loc[candidate, sel]
                        if not np.isnan(corr_val):
                            corrs.append(corr_val)
                if len(corrs) > 0:
                    mean_corr = np.mean(corrs)

            # Calculate score: CV score - lambda * mean correlation
            score = cv_scores[candidate] - lambda_correlation * mean_corr

            if score > best_score:
                best_score = score
                best_model = candidate

        if best_model is None:
            break

        selected.append(best_model)
        remaining.remove(best_model)

    return selected


def find_available_models(task_key: str, task_name: str, mt_task_name: str) -> Tuple[List[str], List[str], Dict[str, Dict[str, Path]]]:
    """Find available single-task and multitask models for a given task.

    Only includes models that have BOTH OOF and test predictions for the task.

    Args:
        task_key: Task key (e.g., 'fluo_340_450')
        task_name: Single task name (e.g., 'y_fluo_any')
        mt_task_name: Multitask task name (e.g., 'fluorescence340_450')

    Returns:
        Tuple of (available_single_task_models, available_multitask_models, file_paths_dict)
        where file_paths_dict maps model_name to {'oof': Path, 'test': Path}
    """
    available_single = []
    available_multitask = []
    file_paths = {}

    # Scan pred_base_dir for all model directories
    if not pred_base_dir.exists():
        return available_single, available_multitask, file_paths

    # Check all directories in preds folder
    for model_dir in sorted(pred_base_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name

        # Skip special directories
        if model_name in ['prev', 'oof_eval_all_tasks.csv', 'oof_eval_all_tasks_pivot.csv']:
            continue

        # Check if it's a multitask model (has 'oof' and 'test' subdirectories)
        oof_subdir = model_dir / "oof"
        test_subdir = model_dir / "test"

        if oof_subdir.exists() and test_subdir.exists():
            # Multitask model
            oof_path = oof_subdir / f"{mt_task_name}_oof.csv"
            test_path = test_subdir / f"{mt_task_name}_test.csv"

            if oof_path.exists() and test_path.exists():
                available_multitask.append(model_name)
                file_paths[model_name] = {'oof': oof_path, 'test': test_path}
        else:
            # Single-task model - check if task directory exists
            task_dir = model_dir / task_key
            if task_dir.exists():
                oof_path = task_dir / f"{task_name}_oof.csv"
                test_path = task_dir / f"{task_name}_test.csv"

                if oof_path.exists() and test_path.exists():
                    available_single.append(model_name)
                    file_paths[model_name] = {'oof': oof_path, 'test': test_path}

    return available_single, available_multitask, file_paths


def load_predictions(file_paths: Dict[str, Dict[str, Path]], available_single: List[str], available_multitask: List[str]) -> Dict[str, pd.Series]:
    """Load all available predictions for a task.

    Args:
        file_paths: Dictionary mapping model_name to {'oof': Path, 'test': Path}
        available_single: List of available single-task model names
        available_multitask: List of available multitask model names

    Returns:
        Dictionary mapping model_name to prediction Series indexed by ID
    """
    predictions = {}

    # Load single-task predictions
    for model_name in available_single:
        oof_path = file_paths[model_name]['oof']
        try:
            # Try reading with index_col=0 first (if mol_id is index)
            df = pd.read_csv(oof_path, index_col=0)
            # Check if index is named or if we need to reset
            if df.index.name in ['mol_id', 'ID'] or df.index.name is None:
                # Reset and check for ID column
                df_reset = df.reset_index()
                if 'mol_id' in df_reset.columns:
                    df = df_reset.rename(columns={'mol_id': 'ID'}).set_index('ID')
                elif 'ID' in df_reset.columns:
                    df = df_reset.set_index('ID')
                else:
                    # No ID column, use index as ID
                    df = df_reset.set_index(df_reset.index.rename('ID'))
            predictions[f"single_{model_name}"] = df['prediction']
        except Exception as e:
            print(f"  ⚠️  Error loading {model_name}: {e}")
            continue

    # Load multitask predictions
    for model_name in available_multitask:
        oof_path = file_paths[model_name]['oof']
        try:
            df = pd.read_csv(oof_path)
            # Check for ID or mol_id column
            if 'ID' in df.columns:
                df = df.set_index('ID')
            elif 'mol_id' in df.columns:
                df = df.rename(columns={'mol_id': 'ID'}).set_index('ID')
            else:
                # Try index_col=0
                df = pd.read_csv(oof_path, index_col=0)
                if df.index.name not in ['ID', 'mol_id']:
                    df.index.name = 'ID'
            predictions[model_name] = df['prediction']
        except Exception as e:
            print(f"  ⚠️  Error loading multitask {model_name}: {e}")
            continue

    return predictions


def load_train_data(task_key: str, prepared_file: str) -> pd.DataFrame:
    """Load training data for a task.

    Args:
        task_key: Task key
        prepared_file: Prepared data file name

    Returns:
        Training DataFrame with ID and label columns
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


def find_splits_file(task_key: str) -> Path:
    """Find CV splits file for a task.

    Args:
        task_key: Task key (e.g., 'fluo_340_450')

    Returns:
        Path to splits JSON file

    Raises:
        FileNotFoundError: If no splits file is found
    """
    # Map task_key to possible splits file names
    splits_file_names = {
        'fluo_340_450': 'splits_fluo_340_450.json',
        'fluo_480': 'splits_fluo_480.json',
        'trans_340': 'splits_trans_340.json',
        'trans_450': 'splits_trans_450.json',
    }

    splits_file_name = splits_file_names.get(task_key)
    if not splits_file_name:
        raise ValueError(f"Unknown task_key: {task_key}")

    # Try different directories
    for splits_dir in splits_dirs:
        if splits_dir:
            candidate_path = processed_base_dir / splits_dir / splits_file_name
        else:
            candidate_path = processed_base_dir / splits_file_name

        if candidate_path.exists():
            return candidate_path

    raise FileNotFoundError(f"Splits file not found for {task_key}")


def load_splits(task_key: str) -> Dict:
    """Load CV splits for a task.

    Args:
        task_key: Task key

    Returns:
        Dictionary with fold information
    """
    splits_path = find_splits_file(task_key)
    with open(splits_path, 'r') as f:
        splits = json.load(f)
    return splits


def load_molecular_features(task_key: str, prepared_file: str, feature_dir: str, feature_groups: List[str]) -> Optional[pd.DataFrame]:
    """Load molecular features from parquet file.

    Args:
        task_key: Task key
        prepared_file: Prepared data file name
        feature_dir: Directory containing feature files (e.g., 'ecfp4')
        feature_groups: List of feature groups to include (e.g., ['ecfp4', 'rdkit2d'])

    Returns:
        DataFrame with features indexed by ID, or None if not found
    """
    # Try to find feature parquet file
    feature_file = f"features_{prepared_file.replace('.csv', '.parquet')}"
    feature_path = processed_base_dir / feature_dir / feature_file

    if not feature_path.exists():
        print(f"  ⚠️  Feature file not found: {feature_path}")
        return None

    try:
        features_df = load_parquet(feature_path)

        # Filter by feature groups
        if feature_groups:
            selected_cols = []
            for col in features_df.columns:
                # Extract group prefix (format: "group__feature_name")
                if "__" in col:
                    group = col.split("__")[0]
                    if group in feature_groups:
                        selected_cols.append(col)
                else:
                    # If no prefix, check if column name matches a group
                    if col in feature_groups:
                        selected_cols.append(col)

            if selected_cols:
                features_df = features_df[selected_cols]
                print(f"  Loaded {len(selected_cols)} features from groups: {feature_groups}")
            else:
                print(f"  ⚠️  No features found for groups: {feature_groups}")
                return None
        else:
            print(f"  Loaded all {len(features_df.columns)} features")

        # Set ID as index if not already
        if 'ID' in features_df.columns:
            features_df = features_df.set_index('ID')
        elif features_df.index.name not in ['ID', 'mol_id']:
            # Try to get ID from prepared file
            df_prepared = load_train_data(task_key, prepared_file)
            if 'ID' in df_prepared.columns:
                features_df.index = df_prepared['ID'].values[:len(features_df)]
                features_df.index.name = 'ID'

        return features_df

    except Exception as e:
        print(f"  ⚠️  Error loading features: {e}")
        return None


def compute_sim_to_train_mean(
    query_ids: List[str],
    train_ids: List[str],
    df_train: pd.DataFrame,
    smiles_col: str = 'SMILES',
    radius: int = 3,
    n_bits: int = 2048
) -> np.ndarray:
    """Compute max/mean Tanimoto similarity to training data using ECFP.

    For each query sample, compute Tanimoto similarity to all training samples
    and return the maximum similarity.

    Args:
        query_ids: List of query sample IDs
        train_ids: List of training sample IDs (for this fold)
        df_train: Training DataFrame with SMILES column
        smiles_col: Name of SMILES column
        radius: ECFP radius
        n_bits: Number of bits in fingerprint

    Returns:
        Array of max similarities (one per query sample)
    """
    if len(train_ids) == 0:
        return np.zeros(len(query_ids))

    # Get SMILES for query and train
    query_smiles = df_train.loc[df_train['ID'].isin(query_ids), smiles_col].values
    train_smiles = df_train.loc[df_train['ID'].isin(train_ids), smiles_col].values

    if len(query_smiles) == 0 or len(train_smiles) == 0:
        return np.zeros(len(query_ids))

    # Compute fingerprints for training set
    train_fps = []
    for smiles in train_smiles:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
                train_fps.append(fp)
            else:
                train_fps.append(None)
        except Exception:
            train_fps.append(None)

    # Compute similarities for query set
    similarities = []
    for smiles in query_smiles:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                similarities.append(0.0)
                continue

            query_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)

            # Compute Tanimoto similarity to all training samples
            sims = []
            for train_fp in train_fps:
                if train_fp is not None:
                    sim = DataStructs.TanimotoSimilarity(query_fp, train_fp)
                    sims.append(sim)

            if len(sims) > 0:
                # Use max similarity
                similarities.append(np.max(sims))
            else:
                similarities.append(0.0)
        except Exception:
            similarities.append(0.0)

    return np.array(similarities)


def compute_cluster_density(
    query_ids: List[str],
    train_ids: List[str],
    features: pd.DataFrame,
    n_clusters: int = 50
) -> np.ndarray:
    """Compute cluster density (samples per cluster) using k-means.

    For each query sample, assign it to a cluster and return the density
    (number of training samples) in that cluster.

    Args:
        query_ids: List of query sample IDs
        train_ids: List of training sample IDs (for this fold)
        features: Feature DataFrame indexed by ID
        n_clusters: Number of clusters for k-means

    Returns:
        Array of cluster densities (one per query sample)
    """
    if len(train_ids) == 0 or len(query_ids) == 0:
        return np.zeros(len(query_ids))

    # Get features for train and query
    train_features = features.loc[features.index.isin(train_ids)].values
    query_features = features.loc[features.index.isin(query_ids)].values

    if len(train_features) == 0 or len(query_features) == 0:
        return np.zeros(len(query_ids))

    # Fit k-means on training data only
    try:
        kmeans = KMeans(n_clusters=min(n_clusters, len(train_ids)), random_state=42, n_init=10)
        kmeans.fit(train_features)

        # Predict clusters for query samples
        query_clusters = kmeans.predict(query_features)

        # Count samples per cluster in training set
        train_clusters = kmeans.predict(train_features)
        cluster_counts = np.bincount(train_clusters, minlength=kmeans.n_clusters)

        # Get density for each query sample
        densities = cluster_counts[query_clusters]

        return densities.astype(float)
    except Exception as e:
        print(f"    ⚠️  Error computing cluster density: {e}")
        return np.zeros(len(query_ids))


def compute_model_std(
    query_ids: List[str],
    predictions: Dict[str, pd.Series]
) -> np.ndarray:
    """Compute standard deviation of base model predictions.

    For each query sample, compute the standard deviation across all model predictions.

    Args:
        query_ids: List of query sample IDs
        predictions: Dictionary of model_name -> prediction Series

    Returns:
        Array of standard deviations (one per query sample)
    """
    if len(predictions) == 0:
        return np.zeros(len(query_ids))

    # Build prediction matrix
    pred_matrix = []
    for model_name, pred_series in predictions.items():
        pred_values = pred_series.loc[pred_series.index.isin(query_ids)].reindex(query_ids, fill_value=0.0)
        pred_matrix.append(pred_values.values)

    if len(pred_matrix) == 0:
        return np.zeros(len(query_ids))

    # Compute std across models for each sample
    pred_array = np.array(pred_matrix).T  # (n_samples, n_models)
    stds = np.std(pred_array, axis=1)

    return stds


def build_level1_data(
    fold_idx: int,
    splits: Dict,
    predictions: Dict[str, pd.Series],
    y_true: pd.Series,
    idx_to_id: Dict[int, str],
    model_names: List[str],
    config: Dict[str, Any],
    df_train: Optional[pd.DataFrame] = None,
    features_df: Optional[pd.DataFrame] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Build level-1 training and validation data for stacking.

    Args:
        fold_idx: Current fold index
        splits: CV splits dictionary
        predictions: Dictionary of model_name -> prediction Series
        y_true: True labels Series indexed by ID
        idx_to_id: Mapping from index to ID
        model_names: List of available model names
        config: Configuration dictionary
        df_train: Training DataFrame (for derived features)
        features_df: Molecular features DataFrame (optional)

    Returns:
        Tuple of (X_train_level1, y_train_level1, X_valid_level1, y_valid_level1, valid_ids)
        where X are feature matrices (n_samples, n_features)
    """
    # Get current fold's validation IDs
    fold_name = f"fold_{fold_idx}"
    valid_indices = splits[fold_name]["valid"]
    valid_ids = [idx_to_id[idx] for idx in valid_indices if idx in idx_to_id]
    valid_ids_set = set(valid_ids)

    # Collect training IDs from other folds
    train_ids = []
    for other_fold_name, fold_data in splits.items():
        if other_fold_name != fold_name:
            train_indices = fold_data["train"]
            train_ids.extend([idx_to_id[idx] for idx in train_indices if idx in idx_to_id])

    train_ids_set = set(train_ids)

    # Build level-1 features: matrix of predictions (n_samples, n_models)
    # Find common IDs across all models for training set
    train_common_ids = train_ids_set.copy()
    for model_name in model_names:
        train_common_ids &= set(predictions[model_name].index)
    train_common_ids = sorted(train_common_ids)

    # Find common IDs across all models for validation set
    valid_common_ids = valid_ids_set.copy()
    for model_name in model_names:
        valid_common_ids &= set(predictions[model_name].index)
    valid_common_ids = sorted(valid_common_ids)

    # Build base prediction matrices
    n_models = len(model_names)
    X_train_pred = np.zeros((len(train_common_ids), n_models))
    X_valid_pred = np.zeros((len(valid_common_ids), n_models))

    for i, model_name in enumerate(model_names):
        X_train_pred[:, i] = predictions[model_name].loc[train_common_ids].values
        X_valid_pred[:, i] = predictions[model_name].loc[valid_common_ids].values

    # Start with predictions
    X_train_level1_list = [X_train_pred]
    X_valid_level1_list = [X_valid_pred]

    # Add molecular features if enabled
    if config.get('use_additional_features', False) and features_df is not None:
        train_features = features_df.loc[features_df.index.isin(train_common_ids)].reindex(train_common_ids, fill_value=0.0)
        valid_features = features_df.loc[features_df.index.isin(valid_common_ids)].reindex(valid_common_ids, fill_value=0.0)

        # Convert to numpy arrays
        X_train_level1_list.append(train_features.values)
        X_valid_level1_list.append(valid_features.values)
        print(f"    Added {len(train_features.columns)} molecular features")

    # Add derived features if enabled
    if config.get('use_derived_features', False) and df_train is not None:
        derived_features = config.get('derived_features', [])

        for feat_name in derived_features:
            if feat_name == 'sim_to_train_mean':
                train_sim = compute_sim_to_train_mean(train_common_ids, train_common_ids, df_train)
                valid_sim = compute_sim_to_train_mean(valid_common_ids, train_common_ids, df_train)
                X_train_level1_list.append(train_sim.reshape(-1, 1))
                X_valid_level1_list.append(valid_sim.reshape(-1, 1))
                print(f"    Added sim_to_train_mean feature")

            elif feat_name == 'cluster_density':
                if features_df is not None:
                    n_clusters = config.get('cluster_n_clusters', DEFAULT_CLUSTER_N_CLUSTERS)
                    train_cluster = compute_cluster_density(train_common_ids, train_common_ids, features_df, n_clusters)
                    valid_cluster = compute_cluster_density(valid_common_ids, train_common_ids, features_df, n_clusters)
                    X_train_level1_list.append(train_cluster.reshape(-1, 1))
                    X_valid_level1_list.append(valid_cluster.reshape(-1, 1))
                    print(f"    Added cluster_density feature")

            elif feat_name == 'model_std':
                train_std = compute_model_std(train_common_ids, predictions)
                valid_std = compute_model_std(valid_common_ids, predictions)
                X_train_level1_list.append(train_std.reshape(-1, 1))
                X_valid_level1_list.append(valid_std.reshape(-1, 1))
                print(f"    Added model_std feature")

    # Concatenate all features
    X_train_level1 = np.hstack(X_train_level1_list)
    X_valid_level1 = np.hstack(X_valid_level1_list)

    # Get labels
    y_train_level1 = y_true.loc[train_common_ids].values
    y_valid_level1 = y_true.loc[valid_common_ids].values

    return X_train_level1, y_train_level1, X_valid_level1, y_valid_level1, valid_common_ids


def create_meta_model(
    model_type: str,
    params: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
) -> Any:
    """Create and train a meta-learner model.

    Args:
        model_type: Type of meta-model ('lgbm', 'logistic', 'ridge')
        params: Model parameters
        X_train: Training features (level-1 predictions + optional features)
        y_train: Training labels
        X_val: Optional validation features for early stopping
        y_val: Optional validation labels for early stopping

    Returns:
        Trained meta-model
    """
    if model_type == 'lgbm':
        # Compute pos_weight
        n_pos = np.sum(y_train)
        n_neg = len(y_train) - n_pos
        pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

        model = LGBMClassifier(
            n_estimators=params.get('n_estimators', 100),
            learning_rate=params.get('learning_rate', 0.05),
            num_leaves=params.get('num_leaves', 31),
            max_depth=params.get('max_depth', 5),
            min_child_samples=params.get('min_child_samples', 20),
            pos_weight=pos_weight,
            early_stopping_rounds=params.get('early_stopping_rounds', 20),
            verbose=-1
        )

        # Convert to DataFrame for LGBMClassifier
        X_train_df = pd.DataFrame(X_train, columns=[f'feat_{i}' for i in range(X_train.shape[1])])
        if X_val is not None and y_val is not None:
            X_val_df = pd.DataFrame(X_val, columns=[f'feat_{i}' for i in range(X_val.shape[1])])
            model.fit(X_train_df, y_train, eval_set=(X_val_df, y_val))
        else:
            model.fit(X_train_df, y_train)

    elif model_type == 'logistic':
        model = LogisticRegression(
            C=params.get('C', 1.0),
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
        model.fit(X_train, y_train)

    elif model_type == 'ridge':
        # RidgeClassifier doesn't have predict_proba, so we use calibrated version
        base_model = RidgeClassifier(
            alpha=params.get('alpha', 1.0),
            class_weight='balanced',
            random_state=42
        )
        # Use calibration to get probabilities
        model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
        model.fit(X_train, y_train)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def predict_proba_meta_model(model: Any, X: np.ndarray, model_type: str) -> np.ndarray:
    """Predict probabilities from a meta-model.

    Args:
        model: Trained meta-model
        X: Features (level-1 predictions + optional features)
        model_type: Type of meta-model ('lgbm', 'logistic', 'ridge')

    Returns:
        Predicted probabilities (n_samples,)
    """
    if model_type == 'lgbm':
        # Convert to DataFrame
        X_df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(X.shape[1])])
        pred = model.predict_proba(X_df)
        # LGBMClassifier returns (n_samples,) for binary classification
        if pred.ndim == 1:
            return pred
        else:
            return pred[:, 1]

    elif model_type in ['logistic', 'ridge']:
        pred = model.predict_proba(X)
        if pred.ndim == 2:
            return pred[:, 1]
        else:
            return pred

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def objective_meta_learner(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    model_names: List[str]
) -> float:
    """Optuna objective function for meta-learner optimization.

    Args:
        trial: Optuna trial
        X_train: Level-1 training features (predictions + optional features)
        y_train: Training labels
        X_valid: Level-1 validation features
        y_valid: Validation labels
        model_names: List of available model names (for potential feature selection)

    Returns:
        ROC-AUC score
    """
    # Suggest meta-model type
    model_type = trial.suggest_categorical('model_type', ['lgbm', 'logistic', 'ridge'])

    params = {}

    if model_type == 'lgbm':
        params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.1, log=False)
        params['num_leaves'] = trial.suggest_int('num_leaves', 31, 127)
        params['max_depth'] = trial.suggest_int('max_depth', 3, 8)
        params['min_child_samples'] = trial.suggest_int('min_child_samples', 10, 50)
        params['n_estimators'] = 100  # Fixed, lightweight
        params['early_stopping_rounds'] = 20

    elif model_type == 'logistic':
        params['C'] = trial.suggest_float('C', 0.01, 100, log=True)

    elif model_type == 'ridge':
        params['alpha'] = trial.suggest_float('alpha', 0.01, 100, log=True)

    # Create and train meta-model
    try:
        model = create_meta_model(
            model_type=model_type,
            params=params,
            X_train=X_train,
            y_train=y_train,
            X_val=X_valid,
            y_val=y_valid
        )

        # Predict on validation set
        y_pred = predict_proba_meta_model(model, X_valid, model_type)

        # Calculate ROC-AUC
        try:
            auc = roc_auc_score(y_valid, y_pred)
        except ValueError:
            # Handle case where only one class is present
            return 0.0

        return auc

    except Exception as e:
        print(f"  ⚠️  Error in trial: {e}")
        return 0.0


def train_meta_learner_for_fold(
    fold_idx: int,
    splits: Dict,
    predictions: Dict[str, pd.Series],
    y_true: pd.Series,
    idx_to_id: Dict[int, str],
    model_names: List[str],
    task_key: str,
    config: Dict[str, Any],
    output_dir: Path,
    df_train: Optional[pd.DataFrame] = None,
    features_df: Optional[pd.DataFrame] = None,
    n_trials: int = 50
) -> Tuple[Any, str, Dict[str, Any], float, List[str], Optional[np.ndarray]]:
    """Train meta-learner for a single fold using Optuna optimization.

    Args:
        fold_idx: Fold index
        splits: CV splits dictionary
        predictions: Dictionary of model_name -> prediction Series
        y_true: True labels Series indexed by ID
        idx_to_id: Mapping from index to ID
        model_names: List of available model names
        task_key: Task key for study naming
        config: Configuration dictionary
        output_dir: Output directory for Optuna study database
        df_train: Training DataFrame (for derived features)
        features_df: Molecular features DataFrame (optional)
        n_trials: Number of Optuna trials

    Returns:
        Tuple of (best_model, best_model_type, best_params, best_auc, valid_ids, y_pred)
    """
    # Build level-1 data
    X_train_level1, y_train_level1, X_valid_level1, y_valid_level1, valid_ids = build_level1_data(
        fold_idx=fold_idx,
        splits=splits,
        predictions=predictions,
        y_true=y_true,
        idx_to_id=idx_to_id,
        model_names=model_names,
        config=config,
        df_train=df_train,
        features_df=features_df
    )

    if len(valid_ids) == 0 or X_train_level1.shape[0] == 0:
        print(f"    ⚠️  No data available for fold {fold_idx}")
        return None, None, {}, 0.0, [], None

    # Create study for this fold
    study_name = f"stacking_{task_key}_fold_{fold_idx}"
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        storage=f"sqlite:///{output_dir}/{study_name}.db",
        load_if_exists=True
    )

    # Optimize
    study.optimize(
        lambda trial: objective_meta_learner(
            trial=trial,
            X_train=X_train_level1,
            y_train=y_train_level1,
            X_valid=X_valid_level1,
            y_valid=y_valid_level1,
            model_names=model_names
        ),
        n_trials=n_trials - len(study.trials),
        show_progress_bar=False
    )

    # Get best result
    best_trial = study.best_trial
    best_params = best_trial.params.copy()
    best_model_type = best_params.pop('model_type')

    # Train best model on all level-1 training data
    best_model = create_meta_model(
        model_type=best_model_type,
        params=best_params,
        X_train=X_train_level1,
        y_train=y_train_level1,
        X_val=X_valid_level1,
        y_val=y_valid_level1
    )

    # Predict on validation set to get final AUC
    y_pred = predict_proba_meta_model(best_model, X_valid_level1, best_model_type)
    try:
        best_auc = roc_auc_score(y_valid_level1, y_pred)
    except ValueError:
        best_auc = 0.0

    # Return predictions along with valid_ids for OOF prediction generation
    return best_model, best_model_type, best_params, best_auc, valid_ids, y_pred


def optimize_stacking_for_task(
    task_key: str,
    task_name: str,
    label_col: str,
    prepared_file: str,
    mt_task_name: str,
    config: Dict[str, Any],
    output_dir: Path
) -> Dict:
    """Optimize stacking ensemble for a single task using Nested CV approach.

    New flow:
    1. Find available models (with both OOF and test predictions)
    2. Load CV scores and select top N models
    3. Select diverse models considering correlation
    4. For each fold, train a meta-learner on level-1 features from other folds
       and predict on the current fold's validation set.

    Args:
        task_key: Task key
        task_name: Single task name
        label_col: Label column name
        prepared_file: Prepared data file name
        mt_task_name: Multitask task name
        config: Configuration dictionary

    Returns:
        Dictionary with optimization results
    """
    print(f"\n{'='*70}")
    print(f"Optimizing stacking ensemble for task: {task_key} (Nested CV with Exploration)")
    print(f"{'='*70}")

    # Step 1: Find available models (only those with both OOF and test files)
    available_single, available_multitask, file_paths = find_available_models(task_key, task_name, mt_task_name)

    print(f"\n📊 Available Models (with both OOF and test predictions):")
    print(f"  Single-task models ({len(available_single)}): {available_single}")
    print(f"  Multitask models ({len(available_multitask)}): {available_multitask}")

    if len(available_single) == 0 and len(available_multitask) == 0:
        print(f"\n⚠️  No models available for {task_key} (with both OOF and test files), skipping...")
        return None

    # Step 2: Load CV scores
    print(f"\n📈 Loading CV scores...")
    cv_scores = load_cv_scores(task_key, task_name, mt_task_name, available_single, available_multitask)
    print(f"  Loaded CV scores for {len(cv_scores)} models")

    if len(cv_scores) == 0:
        print(f"\n⚠️  No CV scores found for {task_key}, skipping...")
        return None

    # Step 3: Select top N models by CV score
    top_n = config.get('top_n_models', DEFAULT_TOP_N)
    print(f"\n🎯 Selecting top {top_n} models by CV ROC-AUC...")
    top_models = select_top_models_by_cv_score(cv_scores, top_n=top_n)
    print(f"  Selected {len(top_models)} models:")
    for model_name in top_models:
        print(f"    - {model_name}: {cv_scores[model_name]:.6f}")

    # Step 4: Load predictions
    print(f"\n📥 Loading OOF predictions...")
    all_predictions = load_predictions(file_paths, available_single, available_multitask)
    print(f"  Loaded predictions from {len(all_predictions)} models")

    # Filter predictions to top models
    predictions = {name: all_predictions[name] for name in top_models if name in all_predictions}
    print(f"  Using {len(predictions)} top models for diversity selection")

    # Step 5: Calculate correlations
    print(f"\n🔗 Calculating model correlations...")
    corr_matrix = calculate_model_correlations(predictions)
    print(f"  Correlation matrix shape: {corr_matrix.shape}")

    # Step 6: Select diverse models
    lambda_correlation = config.get('lambda_correlation', DEFAULT_LAMBDA_CORRELATION)
    max_diverse_models = config.get('max_diverse_models', DEFAULT_MAX_DIVERSE_MODELS)
    print(f"\n🌈 Selecting diverse models (lambda={lambda_correlation}, max={max_diverse_models})...")
    diverse_models = select_diverse_models(
        candidate_models=top_models,
        cv_scores=cv_scores,
        corr_matrix=corr_matrix,
        lambda_correlation=lambda_correlation,
        max_models=max_diverse_models
    )
    print(f"  Selected {len(diverse_models)} diverse models:")
    for model_name in diverse_models:
        mean_corr = 0.0
        if len(diverse_models) > 1:
            corrs = []
            for other in diverse_models:
                if model_name != other:
                    if model_name in corr_matrix.index and other in corr_matrix.columns:
                        corr_val = corr_matrix.loc[model_name, other]
                        if not np.isnan(corr_val):
                            corrs.append(corr_val)
            if len(corrs) > 0:
                mean_corr = np.mean(corrs)
        print(f"    - {model_name}: CV={cv_scores[model_name]:.6f}, Mean Corr={mean_corr:.4f}")

    # Filter predictions to diverse models only
    final_predictions = {name: predictions[name] for name in diverse_models if name in predictions}
    final_model_names = list(final_predictions.keys())

    if len(final_model_names) == 0:
        print(f"\n⚠️  No models selected after diversity filtering, skipping...")
        return None

    # Load training data
    print(f"\n📊 Loading training data...")
    df_train = load_train_data(task_key, prepared_file)
    y_true = (df_train[label_col] != 0).astype(int)
    y_true.index = df_train['ID']
    print(f"Training data: {len(df_train)} samples")

    # Load molecular features if enabled
    features_df = None
    if config.get('use_additional_features', False) or config.get('use_derived_features', False):
        feature_dir = config.get('feature_dir', 'ecfp4')
        feature_groups = config.get('feature_groups', ['ecfp4'])
        print(f"\n🔬 Loading molecular features from {feature_dir}...")
        features_df = load_molecular_features(task_key, prepared_file, feature_dir, feature_groups)

    # Load CV splits
    print("\nLoading CV splits...")
    splits = load_splits(task_key)
    n_folds = len(splits)
    print(f"Found {n_folds} folds")

    # Create index to ID mapping (splits use row indices)
    id_to_idx = {id_val: idx for idx, id_val in enumerate(df_train['ID'].values)}
    idx_to_id = {idx: id_val for idx, id_val in enumerate(df_train['ID'].values)}

    # Step 7: Train meta-learner for each fold
    n_trials = config.get('n_trials', DEFAULT_N_TRIALS)
    print(f"\n🚀 Starting nested CV stacking optimization with {n_trials} trials per fold...")
    fold_models = []
    fold_model_types = []
    fold_params = []
    fold_aucs = []
    fold_valid_ids = []
    fold_predictions = {}  # Store OOF predictions from training time

    for fold_name, fold_data in sorted(splits.items()):
        fold_idx = int(fold_name.split("_")[1])

        print(f"\n  Fold {fold_idx}: Optimizing meta-learner...")

        # Train meta-learner for this fold
        best_model, best_model_type, best_params, fold_auc, valid_ids, fold_pred = train_meta_learner_for_fold(
            fold_idx=fold_idx,
            splits=splits,
            predictions=final_predictions,
            y_true=y_true,
            idx_to_id=idx_to_id,
            model_names=final_model_names,
            task_key=task_key,
            config=config,
            output_dir=output_dir,
            df_train=df_train,
            features_df=features_df,
            n_trials=n_trials
        )

        fold_models.append(best_model)
        fold_model_types.append(best_model_type)
        fold_params.append(best_params)
        fold_aucs.append(fold_auc)
        fold_valid_ids.append(valid_ids)

        # Store predictions from training time (these are already computed correctly)
        if fold_pred is not None and valid_ids is not None:
            for id_val, pred in zip(valid_ids, fold_pred):
                fold_predictions[id_val] = pred

        print(f"    Best model: {best_model_type}, AUC: {fold_auc:.6f}")
        if best_params:
            print(f"    Best params: {best_params}")

    # Use predictions computed during training (already stored in fold_predictions)
    print(f"\n📝 Using OOF predictions computed during training...")
    oof_predictions = fold_predictions

    # Convert to arrays for final evaluation
    oof_ids = sorted(oof_predictions.keys())
    oof_pred_array = np.array([oof_predictions[id_val] for id_val in oof_ids])
    oof_true_array = y_true.loc[oof_ids].values

    # Calculate final AUC on all OOF predictions
    try:
        final_auc = roc_auc_score(oof_true_array, oof_pred_array)
    except ValueError:
        final_auc = 0.0

    # Calculate statistics
    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)

    # Count model type usage
    model_type_counts = {}
    for mt in fold_model_types:
        if mt:
            model_type_counts[mt] = model_type_counts.get(mt, 0) + 1

    print(f"\n✅ Nested CV stacking optimization completed!")
    print(f"Mean fold AUC: {mean_auc:.6f} ± {std_auc:.6f}")
    print(f"Final OOF AUC: {final_auc:.6f}")
    print(f"Meta-model type usage:")
    for mt, count in sorted(model_type_counts.items()):
        print(f"  - {mt}: {count}/{n_folds} folds")

    # Save results
    result = {
        'task': task_key,
        'mean_fold_auc': float(mean_auc),
        'std_fold_auc': float(std_auc),
        'final_oof_auc': float(final_auc),
        'fold_aucs': [float(auc) for auc in fold_aucs],
        'fold_model_types': fold_model_types,
        'fold_params': fold_params,
        'model_type_counts': model_type_counts,
        'n_models': len(final_model_names),
        'n_available_single': len(available_single),
        'n_available_multitask': len(available_multitask),
        'selected_models': diverse_models,
        'top_models': top_models,
        'cv_scores': cv_scores,
        'n_samples': len(oof_ids),
        'n_folds': n_folds,
        'n_trials_per_fold': n_trials,
        'config': config,
        'file_paths': {
            model_name: {
                'oof': str(paths['oof'].relative_to(PROJECT_ROOT)),
                'test': str(paths['test'].relative_to(PROJECT_ROOT))
            }
            for model_name, paths in file_paths.items()
        }
    }

    # Save to JSON
    result_path = output_dir / f"ensemble_optimization_stacking_{task_key}.json"
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {result_path}")

    # Save ensemble predictions
    ensemble_df = pd.DataFrame({
        'ID': oof_ids,
        'prediction': oof_pred_array
    })
    ensemble_path = output_dir / f"ensemble_oof_stacking_{task_key}.csv"
    ensemble_df.to_csv(ensemble_path, index=False)
    print(f"Ensemble predictions saved to {ensemble_path}")

    # Also save CV model average predictions (simple average of base models)
    print(f"\n📊 Generating CV model average predictions...")
    cv_avg_pred_array = np.zeros(len(oof_ids))
    for model_name in final_model_names:
        cv_avg_pred_array += final_predictions[model_name].loc[oof_ids].values
    cv_avg_pred_array /= len(final_model_names)
    cv_avg_pred_array = np.clip(cv_avg_pred_array, 0, 1)

    cv_avg_df = pd.DataFrame({
        'ID': oof_ids,
        'prediction': cv_avg_pred_array
    })
    cv_avg_path = output_dir / f"ensemble_oof_cv_avg_{task_key}.csv"
    cv_avg_df.to_csv(cv_avg_path, index=False)
    print(f"CV average predictions saved to {cv_avg_path}")

    # Calculate CV average AUC
    try:
        cv_avg_auc = roc_auc_score(oof_true_array, cv_avg_pred_array)
        print(f"CV average AUC: {cv_avg_auc:.6f}")
        result['cv_avg_auc'] = float(cv_avg_auc)
    except ValueError:
        pass

    # Update result file
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    return result


def load_test_predictions(task_key: str, task_name: str, mt_task_name: str, model_names: List[str], file_paths: Dict[str, Dict[str, Path]]) -> Dict[str, pd.Series]:
    """Load test predictions for all models.

    Args:
        task_key: Task key
        task_name: Single task name
        mt_task_name: Multitask task name
        model_names: List of model names (with prefix like "single_full")
        file_paths: Dictionary mapping model_name to {'oof': Path, 'test': Path}

    Returns:
        Dictionary mapping model_name to prediction Series indexed by ID
    """
    test_predictions = {}

    for model_name in model_names:
        # Extract actual model name
        if model_name.startswith("single_"):
            actual_model_name = model_name.replace("single_", "")
        elif model_name.startswith("multitask_"):
            actual_model_name = model_name
        else:
            actual_model_name = model_name

        # Get test path from file_paths
        if actual_model_name in file_paths:
            test_path = file_paths[actual_model_name]['test']
        else:
            # Fallback to old method
            if model_name.startswith("single_"):
                actual_model_name = model_name.replace("single_", "")
                test_path = pred_base_dir / actual_model_name / task_key / f"{task_name}_test.csv"
            elif model_name.startswith("multitask_"):
                actual_model_name = model_name.replace("multitask_", "")
                test_path = pred_base_dir / actual_model_name / "test" / f"{mt_task_name}_test.csv"
            else:
                continue

        if not test_path.exists():
            print(f"  ⚠️  Test predictions not found: {test_path}")
            continue

        try:
            df = pd.read_csv(test_path)
            # Handle ID column
            if 'ID' in df.columns:
                df = df.set_index('ID')
            elif 'mol_id' in df.columns:
                df = df.rename(columns={'mol_id': 'ID'}).set_index('ID')
            else:
                df = pd.read_csv(test_path, index_col=0)
                if df.index.name not in ['ID', 'mol_id']:
                    df.index.name = 'ID'
            test_predictions[model_name] = df['prediction']
        except Exception as e:
            print(f"  ⚠️  Error loading {model_name}: {e}")
            continue

    return test_predictions


def generate_test_predictions_from_cv_models(
    task_key: str,
    task_name: str,
    mt_task_name: str,
    model_names: List[str],
    file_paths: Dict[str, Dict[str, Path]]
) -> pd.DataFrame:
    """Generate test predictions using CV model average (simple average of base models).

    Args:
        task_key: Task key
        task_name: Single task name
        mt_task_name: Multitask task name
        model_names: List of model names
        file_paths: Dictionary mapping model_name to {'oof': Path, 'test': Path}

    Returns:
        DataFrame with ID and prediction columns
    """
    print(f"\n📊 Generating test predictions from CV model average for {task_key}...")

    # Load test predictions
    test_predictions = load_test_predictions(task_key, task_name, mt_task_name, model_names, file_paths)

    if len(test_predictions) == 0:
        raise ValueError(f"No test predictions loaded for {task_key}")

    # Get common IDs
    common_ids = set(list(test_predictions.values())[0].index)
    for pred in test_predictions.values():
        common_ids &= set(pred.index)
    common_ids = sorted(common_ids)

    # Calculate simple average
    ensemble_pred = np.zeros(len(common_ids))
    for model_name in model_names:
        if model_name in test_predictions:
            ensemble_pred += test_predictions[model_name].loc[common_ids].values
    ensemble_pred /= len([m for m in model_names if m in test_predictions])

    # Clip to [0, 1]
    ensemble_pred = np.clip(ensemble_pred, 0, 1)

    return pd.DataFrame({
        'ID': common_ids,
        'prediction': ensemble_pred
    })


def generate_test_predictions_from_stacking(
    task_key: str,
    task_name: str,
    mt_task_name: str,
    prepared_file: str,
    model_names: List[str],
    file_paths: Dict[str, Dict[str, Path]],
    fold_models: List[Any],
    fold_model_types: List[str],
    splits: Dict,
    idx_to_id: Dict[int, str],
    config: Dict[str, Any],
    df_train: Optional[pd.DataFrame] = None,
    features_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Generate test predictions using stacking ensemble (CV meta-learners).

    Args:
        task_key: Task key
        task_name: Single task name
        mt_task_name: Multitask task name
        prepared_file: Prepared data file name
        model_names: List of model names
        file_paths: Dictionary mapping model_name to {'oof': Path, 'test': Path}
        fold_models: List of trained meta-models for each fold
        fold_model_types: List of model types for each fold
        splits: CV splits dictionary
        idx_to_id: Mapping from index to ID
        config: Configuration dictionary
        df_train: Training DataFrame (for derived features)
        features_df: Molecular features DataFrame (optional)

    Returns:
        DataFrame with ID and prediction columns
    """
    print(f"\n🔮 Generating test predictions from stacking ensemble for {task_key}...")

    # Load test predictions
    test_predictions = load_test_predictions(task_key, task_name, mt_task_name, model_names, file_paths)

    if len(test_predictions) == 0:
        raise ValueError(f"No test predictions loaded for {task_key}")

    # Get common IDs
    common_ids = set(list(test_predictions.values())[0].index)
    for pred in test_predictions.values():
        common_ids &= set(pred.index)
    common_ids = sorted(common_ids)

    # Build level-1 features for test set (using all training data for derived features)
    print(f"  Building level-1 features for test set...")

    # Build base prediction matrix
    n_models = len(model_names)
    X_test_pred = np.zeros((len(common_ids), n_models))
    for i, model_name in enumerate(model_names):
        if model_name in test_predictions:
            X_test_pred[:, i] = test_predictions[model_name].loc[common_ids].values

    X_test_level1_list = [X_test_pred]

    # Add molecular features if enabled
    if config.get('use_additional_features', False) and features_df is not None:
        test_features = features_df.loc[features_df.index.isin(common_ids)].reindex(common_ids, fill_value=0.0)
        X_test_level1_list.append(test_features.values)
        print(f"    Added {len(test_features.columns)} molecular features")

    # Add derived features if enabled (using all training data)
    if config.get('use_derived_features', False) and df_train is not None:
        # Get all training IDs
        all_train_ids = list(df_train['ID'].values)
        derived_features = config.get('derived_features', [])

        for feat_name in derived_features:
            if feat_name == 'sim_to_train_mean':
                test_sim = compute_sim_to_train_mean(common_ids, all_train_ids, df_train)
                X_test_level1_list.append(test_sim.reshape(-1, 1))
                print(f"    Added sim_to_train_mean feature")

            elif feat_name == 'cluster_density':
                if features_df is not None:
                    n_clusters = config.get('cluster_n_clusters', DEFAULT_CLUSTER_N_CLUSTERS)
                    test_cluster = compute_cluster_density(common_ids, all_train_ids, features_df, n_clusters)
                    X_test_level1_list.append(test_cluster.reshape(-1, 1))
                    print(f"    Added cluster_density feature")

            elif feat_name == 'model_std':
                test_std = compute_model_std(common_ids, test_predictions)
                X_test_level1_list.append(test_std.reshape(-1, 1))
                print(f"    Added model_std feature")

    # Concatenate all features
    X_test_level1 = np.hstack(X_test_level1_list)

    # Average predictions from all fold-specific meta-learners
    test_preds = []
    for model, model_type in zip(fold_models, fold_model_types):
        if model is None:
            continue
        pred = predict_proba_meta_model(model, X_test_level1, model_type)
        test_preds.append(pred)

    if len(test_preds) == 0:
        raise ValueError(f"No valid predictions for {task_key}")

    # Average across folds
    ensemble_pred = np.mean(test_preds, axis=0)

    # Clip to [0, 1]
    ensemble_pred = np.clip(ensemble_pred, 0, 1)

    return pd.DataFrame({
        'ID': common_ids,
        'prediction': ensemble_pred
    })


def main(config_path: Optional[str] = None, output_dir: Optional[str] = None):
    """Main function to optimize stacking ensembles for all tasks.

    Args:
        config_path: Path to YAML config file (optional)
        output_dir: Output directory path (optional, defaults to data/ensembles_stacking_v2)
    """
    print("="*70)
    print("Stacking Ensemble Optimization v2 for All Tasks")
    print("="*70)

    # Set output directory
    if output_dir is None:
        output_dir_path = DEFAULT_OUTPUT_DIR
    else:
        output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir_path}")

    # Load configuration
    config = load_config(config_path)
    print(f"\nConfiguration:")
    print(f"  Use additional features: {config.get('use_additional_features', False)}")
    print(f"  Feature groups: {config.get('feature_groups', [])}")
    print(f"  Use derived features: {config.get('use_derived_features', False)}")
    print(f"  Derived features: {config.get('derived_features', [])}")
    print(f"  Top N models: {config.get('top_n_models', DEFAULT_TOP_N)}")
    print(f"  Lambda correlation: {config.get('lambda_correlation', DEFAULT_LAMBDA_CORRELATION)}")
    print(f"  Max diverse models: {config.get('max_diverse_models', DEFAULT_MAX_DIVERSE_MODELS)}")
    print(f"  N trials per fold: {config.get('n_trials', DEFAULT_N_TRIALS)}")

    all_results = []

    for task_key, task_name, label_col, prepared_file, mt_task_name in tasks:
        try:
            result = optimize_stacking_for_task(
                task_key=task_key,
                task_name=task_name,
                label_col=label_col,
                prepared_file=prepared_file,
                mt_task_name=mt_task_name,
                config=config,
                output_dir=output_dir_path
            )
            if result:
                all_results.append(result)

            # Generate test predictions for both stacking and CV average
            print(f"\n{'='*70}")
            print(f"Generating test predictions for {task_key}")
            print(f"{'='*70}")

            # Load results to get fold models info
            result_path = output_dir / f"ensemble_optimization_stacking_{task_key}.json"
            if result_path.exists():
                with open(result_path, 'r') as f:
                    result_data = json.load(f)

                # Load necessary data
                _, _, file_paths = find_available_models(task_key, task_name, mt_task_name)
                selected_models = result_data.get('selected_models', [])
                df_train = load_train_data(task_key, prepared_file)
                splits = load_splits(task_key)
                idx_to_id = {idx: id_val for idx, id_val in enumerate(df_train['ID'].values)}

                # Load features if needed
                features_df = None
                if config.get('use_additional_features', False) or config.get('use_derived_features', False):
                    feature_dir = config.get('feature_dir', 'ecfp4')
                    feature_groups = config.get('feature_groups', ['ecfp4'])
                    features_df = load_molecular_features(task_key, prepared_file, feature_dir, feature_groups)

                # Generate CV average test predictions
                try:
                    cv_avg_test_df = generate_test_predictions_from_cv_models(
                        task_key=task_key,
                        task_name=task_name,
                        mt_task_name=mt_task_name,
                        model_names=selected_models,
                        file_paths=file_paths
                    )
                    cv_avg_test_path = output_dir / f"ensemble_test_cv_avg_{task_key}.csv"
                    cv_avg_test_df.to_csv(cv_avg_test_path, index=False)
                    print(f"✅ CV average test predictions saved to {cv_avg_test_path}")
                except Exception as e:
                    print(f"⚠️  Error generating CV average test predictions: {e}")

                # Note: Stacking test predictions require trained models, which are not saved
                # In practice, you would need to retrain or save/load models
                print(f"Note: Stacking test predictions require retraining models on full data")
                print(f"      (Not implemented in this version, use CV average for now)")

        except Exception as e:
            print(f"\n❌ Error optimizing {task_key}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    if all_results:
        print("\n" + "="*70)
        print("Summary of All Tasks")
        print("="*70)
        summary_df = pd.DataFrame([
            {
                'task': r['task'],
                'mean_fold_auc': r.get('mean_fold_auc', 0.0),
                'final_oof_auc': r.get('final_oof_auc', 0.0),
                'cv_avg_auc': r.get('cv_avg_auc', 0.0),
                'n_models': r.get('n_models', 0),
                'n_available': r['n_available_single'] + r['n_available_multitask'],
                'n_samples': r['n_samples']
            }
            for r in all_results
        ])
        print(summary_df.to_string(index=False))

        # Save summary
        summary_path = output_dir / "ensemble_optimization_stacking_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to {summary_path}")

        # Save all results
        all_results_path = output_dir / "ensemble_optimization_stacking_all.json"
        with open(all_results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"All results saved to {all_results_path}")


if __name__ == "__main__":
    config_path = None
    output_dir = None

    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--config" and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--output-dir" and i + 1 < len(sys.argv):
            output_dir = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--help":
            print("Usage: python optimize_ensemble_stacking_v2.py [--config CONFIG_PATH] [--output-dir OUTPUT_DIR]")
            print("  --config CONFIG_PATH     Path to YAML config file")
            print("  --output-dir OUTPUT_DIR  Output directory (default: data/ensembles_stacking_v2)")
            sys.exit(0)
        else:
            i += 1

    main(config_path=config_path, output_dir=output_dir)


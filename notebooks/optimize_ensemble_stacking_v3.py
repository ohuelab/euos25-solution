"""Optimize stacking ensemble using meta-learners for each task (v3).

This script extends optimize_ensemble_stacking_v2.py with:
1. Fold-specific model selection: Each fold selects models based on fold-specific CV scores
2. Fold-specific correlation calculation: Correlations computed on each fold's validation set predictions
3. All features from v2 (molecular features, derived features, YAML config support)

It uses nested CV where for each fold:
- Models are selected based on that fold's CV scores
- Correlations are computed on that fold's validation set predictions
- A meta-learner is trained on level-1 features from other folds
- Predictions are made on the current fold's validation set
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
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "ensembles_stacking_v3"
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
DEFAULT_RADIUS = 3
DEFAULT_N_BITS = 2048

# Default LGBM parameters for meta-learner (when not using Optuna)
DEFAULT_LGBM_PARAMS = {
    'n_estimators': 200,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 5,
    'min_child_samples': 20,
    'early_stopping_rounds': None,
}

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
    'use_optuna': False,  # Default: use default LGBM without optimization
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
    for model_name in tqdm(available_single, desc="  Loading single-task CV scores", leave=False):
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
    for model_name in tqdm(available_multitask, desc="  Loading multitask CV scores", leave=False):
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


def load_cv_scores_per_fold(
    task_key: str,
    task_name: str,
    mt_task_name: str,
    available_single: List[str],
    available_multitask: List[str]
) -> Dict[str, Dict[int, float]]:
    """Load CV ROC-AUC scores per fold for available models.

    Args:
        task_key: Task key (e.g., 'fluo_340_450')
        task_name: Single task name (e.g., 'y_fluo_any')
        mt_task_name: Multitask task name (e.g., 'fluorescence340_450')
        available_single: List of available single-task model names
        available_multitask: List of available multitask model names

    Returns:
        Dictionary mapping model_name (with prefix) to {fold_idx -> roc_auc}
    """
    cv_scores_per_fold = {}

    # Load single-task model scores
    for model_name in tqdm(available_single, desc="  Loading single-task CV scores per fold", leave=False):
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
                    if 'roc_auc' in df.columns and 'fold' in df.columns:
                        fold_scores = {}
                        for _, row in df.iterrows():
                            fold_idx = int(row['fold'])
                            roc_auc = row['roc_auc']
                            fold_scores[fold_idx] = roc_auc
                        cv_scores_per_fold[f"single_{model_name}"] = fold_scores
                        break
                except Exception as e:
                    print(f"  ⚠️  Error reading CV metrics for {model_name}/{model_type}: {e}")
                    continue

    # Load multitask model scores
    for model_name in tqdm(available_multitask, desc="  Loading multitask CV scores per fold", leave=False):
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
                            if task_metric_col in df.columns and 'fold' in df.columns:
                                fold_scores = {}
                                for _, row in df.iterrows():
                                    fold_idx = int(row['fold'])
                                    roc_auc = row[task_metric_col]
                                    fold_scores[fold_idx] = roc_auc
                                cv_scores_per_fold[model_name] = fold_scores
                                break
                            # Fallback to avg_roc_auc if task-specific not available
                            elif 'avg_roc_auc' in df.columns and 'fold' in df.columns:
                                fold_scores = {}
                                for _, row in df.iterrows():
                                    fold_idx = int(row['fold'])
                                    roc_auc = row['avg_roc_auc']
                                    fold_scores[fold_idx] = roc_auc
                                cv_scores_per_fold[model_name] = fold_scores
                                break
                        except Exception as e:
                            print(f"  ⚠️  Error reading CV metrics for {model_name}/{task_subdir.name}/{model_type}: {e}")
                            continue
                break

    return cv_scores_per_fold


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
    predictions: Dict[str, pd.Series],
    valid_ids: Optional[List[str]] = None
) -> pd.DataFrame:
    """Calculate correlation matrix between model predictions.

    Args:
        predictions: Dictionary mapping model_name to prediction Series
        valid_ids: Optional list of IDs to use for correlation calculation.
                   If None, uses all IDs from predictions.

    Returns:
        DataFrame with correlation matrix (model_name x model_name)
    """
    # Determine which IDs to use
    if valid_ids is None:
        # Use all IDs from predictions
        all_ids = set()
        for pred in predictions.values():
            all_ids.update(pred.index)
        all_ids = sorted(all_ids)
    else:
        # Use only validation set IDs
        all_ids = sorted(valid_ids)

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


def select_models_for_fold(
    fold_idx: int,
    cv_scores_per_fold: Dict[str, Dict[int, float]],
    all_predictions: Dict[str, pd.Series],
    valid_ids: List[str],
    config: Dict[str, Any]
) -> List[str]:
    """Select models for a specific fold based on fold-specific CV scores and correlations.

    Args:
        fold_idx: Current fold index
        cv_scores_per_fold: Dictionary mapping model_name to {fold_idx -> roc_auc}
        all_predictions: Dictionary of all model predictions
        valid_ids: Validation set IDs for this fold (for correlation calculation)
        config: Configuration dictionary

    Returns:
        List of selected model names for this fold
    """
    # Get fold-specific CV scores
    fold_cv_scores = {}
    for model_name, fold_scores in cv_scores_per_fold.items():
        if fold_idx in fold_scores:
            fold_cv_scores[model_name] = fold_scores[fold_idx]

    if len(fold_cv_scores) == 0:
        return []

    # Step 1: Select top N models by fold-specific CV score
    top_n = config.get('top_n_models', DEFAULT_TOP_N)
    top_models = select_top_models_by_cv_score(fold_cv_scores, top_n=top_n)

    if len(top_models) == 0:
        return []

    # Step 2: Filter predictions to top models
    top_predictions = {name: all_predictions[name] for name in top_models if name in all_predictions}

    if len(top_predictions) == 0:
        return []

    # Step 3: Calculate correlations on validation set predictions
    corr_matrix = calculate_model_correlations(top_predictions, valid_ids=valid_ids)

    # Step 4: Select diverse models
    lambda_correlation = config.get('lambda_correlation', DEFAULT_LAMBDA_CORRELATION)
    max_diverse_models = config.get('max_diverse_models', DEFAULT_MAX_DIVERSE_MODELS)
    diverse_models = select_diverse_models(
        candidate_models=top_models,
        cv_scores=fold_cv_scores,
        corr_matrix=corr_matrix,
        lambda_correlation=lambda_correlation,
        max_models=max_diverse_models
    )

    return diverse_models


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
    model_dirs = [d for d in sorted(pred_base_dir.iterdir()) if d.is_dir()]
    for model_dir in tqdm(model_dirs, desc="  Scanning for available models", leave=False):
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
    for model_name in tqdm(available_single, desc="  Loading single-task predictions", leave=False):
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
    for model_name in tqdm(available_multitask, desc="  Loading multitask predictions", leave=False):
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


def load_similarity_matrix(
    task_key: str,
    feature_dir: str,
    radius: int = DEFAULT_RADIUS,
    n_bits: int = DEFAULT_N_BITS
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load precomputed similarity matrix.

    Args:
        task_key: Task key
        feature_dir: Feature directory where similarity matrix is stored
        radius: ECFP radius
        n_bits: Number of bits

    Returns:
        Tuple of (similarity_matrix, ids_array) or None if not found
    """
    matrix_path = processed_base_dir / feature_dir / f"similarity_matrix_r{radius}_b{n_bits}.npy"
    ids_path = processed_base_dir / feature_dir / f"similarity_matrix_ids_r{radius}_b{n_bits}.npy"

    if not matrix_path.exists() or not ids_path.exists():
        return None

    try:
        similarity_matrix = np.load(matrix_path)
        ids = np.load(ids_path)
        print(f"  Loaded similarity matrix: shape={similarity_matrix.shape}, {len(ids)} molecules")
        return similarity_matrix, ids
    except Exception as e:
        print(f"  ⚠️  Error loading similarity matrix: {e}")
        return None


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


def _compute_similarities_worker(args: Tuple[List, List, int, int]) -> List[float]:
    """Worker function for parallel similarity computation.

    Args:
        args: Tuple of (query_fps_list, valid_train_fps, radius, n_bits)
              query_fps_list contains pre-computed fingerprints (or None for invalid)

    Returns:
        List of mean similarities for each query fingerprint
    """
    query_fps_list, valid_train_fps, radius, n_bits = args
    similarities = []

    for query_fp in query_fps_list:
        if query_fp is None:
            similarities.append(0.0)
            continue

        try:
            # Compute Tanimoto similarity to all training samples using bulk operation
            sims = DataStructs.BulkTanimotoSimilarity(query_fp, valid_train_fps)

            if len(sims) > 0:
                # Use Mean similarity
                similarities.append(np.mean(sims))
            else:
                similarities.append(0.0)
        except Exception:
            similarities.append(0.0)

    return similarities


def compute_sim_to_train_mean(
    query_ids: List[str],
    train_ids: List[str],
    df_train: pd.DataFrame,
    smiles_col: str = 'SMILES',
    radius: int = DEFAULT_RADIUS,
    n_bits: int = DEFAULT_N_BITS,
    n_jobs: Optional[int] = None,
    train_fps_cache: Optional[Dict[Tuple[str, ...], List]] = None,
    query_fps_cache: Optional[Dict[str, Any]] = None,
    similarity_matrix: Optional[np.ndarray] = None,
    similarity_matrix_ids: Optional[np.ndarray] = None
) -> np.ndarray:
    """Compute max/mean Tanimoto similarity to training data using ECFP.

    For each query sample, compute Tanimoto similarity to all training samples
    and return the mean similarity.

    Args:
        query_ids: List of query sample IDs
        train_ids: List of training sample IDs (for this fold)
        df_train: Training DataFrame with SMILES column
        smiles_col: Name of SMILES column
        radius: ECFP radius
        n_bits: Number of bits in fingerprint
        n_jobs: Number of parallel jobs. If None, uses cpu_count()
        train_fps_cache: Optional cache dictionary for training fingerprints.
                        Key is tuple of sorted train_ids, value is list of fingerprints.
        query_fps_cache: Optional cache dictionary for query fingerprints.
                        Key is SMILES string, value is fingerprint.
        similarity_matrix: Optional precomputed similarity matrix (n_molecules, n_molecules)
        similarity_matrix_ids: Optional array of IDs corresponding to similarity_matrix rows/columns

    Returns:
        Array of mean similarities (one per query sample)
    """
    if len(train_ids) == 0:
        return np.zeros(len(query_ids))

    # Use precomputed similarity matrix if available
    if similarity_matrix is not None and similarity_matrix_ids is not None:
        # Create mapping from ID to index in similarity matrix
        id_to_idx = {id_val: idx for idx, id_val in enumerate(similarity_matrix_ids)}

        # Get indices for query and train IDs (preserve order)
        query_indices = []
        query_idx_map = {}  # Map from query_ids index to similarity_matrix index
        for i, id_val in enumerate(query_ids):
            if id_val in id_to_idx:
                sim_idx = id_to_idx[id_val]
                query_indices.append(sim_idx)
                query_idx_map[i] = len(query_indices) - 1

        train_indices = [id_to_idx[id_val] for id_val in train_ids if id_val in id_to_idx]

        if len(query_indices) == 0 or len(train_indices) == 0:
            return np.zeros(len(query_ids))

        # Extract similarity submatrix
        sim_submatrix = similarity_matrix[np.ix_(query_indices, train_indices)]

        # Compute mean similarity for each query
        mean_similarities = np.mean(sim_submatrix, axis=1)

        # Map back to original query_ids order (handle missing IDs)
        result = np.zeros(len(query_ids))
        for i, id_val in enumerate(query_ids):
            if i in query_idx_map:
                result[i] = mean_similarities[query_idx_map[i]]

        print(f"    Using precomputed similarity matrix ({len(query_indices)} queries, {len(train_indices)} train samples)")
        return result

    # Get SMILES for query and train
    query_smiles = df_train.loc[df_train['ID'].isin(query_ids), smiles_col].values
    train_smiles = df_train.loc[df_train['ID'].isin(train_ids), smiles_col].values

    if len(query_smiles) == 0 or len(train_smiles) == 0:
        return np.zeros(len(query_ids))

    # Check cache for training fingerprints
    train_ids_key = tuple(sorted(train_ids))
    if train_fps_cache is not None and train_ids_key in train_fps_cache:
        valid_train_fps = train_fps_cache[train_ids_key]
        print(f"    Using cached fingerprints for {len(train_ids)} training samples")
    else:
        # Compute fingerprints for training set
        train_fps = []
        for smiles in tqdm(train_smiles, desc="  Computing train fingerprints", leave=False):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
                    train_fps.append(fp)
                else:
                    train_fps.append(None)
            except Exception:
                train_fps.append(None)

        # Filter out None fingerprints
        valid_train_fps = [fp for fp in train_fps if fp is not None]

        # Cache the fingerprints
        if train_fps_cache is not None:
            train_fps_cache[train_ids_key] = valid_train_fps

    if len(valid_train_fps) == 0:
        return np.zeros(len(query_ids))

    # Compute or retrieve query fingerprints using cache
    query_fps = []
    cache_key_base = f"{radius}_{n_bits}"
    cache_hits = 0
    cache_misses = 0

    for smiles in query_smiles:
        cache_key = f"{cache_key_base}_{smiles}"
        if query_fps_cache is not None and cache_key in query_fps_cache:
            query_fps.append(query_fps_cache[cache_key])
            cache_hits += 1
        else:
            # Compute fingerprint
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
                    query_fps.append(fp)
                    # Cache the fingerprint
                    if query_fps_cache is not None:
                        query_fps_cache[cache_key] = fp
                    cache_misses += 1
                else:
                    query_fps.append(None)
                    cache_misses += 1
            except Exception:
                query_fps.append(None)
                cache_misses += 1

    if cache_hits > 0:
        print(f"    Using cached fingerprints for {cache_hits} query samples, computed {cache_misses} new")

    # Compute similarities for query set using multiprocessing
    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)

    # Split query_fps into chunks for parallel processing
    chunk_size = max(1, len(query_fps) // n_jobs)
    query_fps_chunks = [
        query_fps[i:i + chunk_size]
        for i in range(0, len(query_fps), chunk_size)
    ]

    # Prepare arguments for workers
    worker_args = [
        (chunk, valid_train_fps, radius, n_bits)
        for chunk in query_fps_chunks
    ]

    # Process in parallel
    print(f"    Computing similarities using {n_jobs} processes ({len(query_fps)} queries, {len(valid_train_fps)} train samples)...")
    similarities = []
    with Pool(processes=n_jobs) as pool:
        results = list(tqdm(pool.imap(_compute_similarities_worker, worker_args),
                           total=len(worker_args),
                           desc="    Processing chunks",
                           leave=False))
        for result in results:
            similarities.extend(result)

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

def compute_prediction_uncertainty_features(
    query_ids: List[str],
    predictions: Dict[str, pd.Series]
) -> Dict[str, np.ndarray]:
    """予測の不確実性に関する特徴量"""
    features = {}

    # 既存のmodel_stdに加えて
    pred_matrix = []
    for model_name, pred_series in predictions.items():
        pred_values = pred_series.loc[pred_series.index.isin(query_ids)].reindex(query_ids, fill_value=0.0)
        pred_matrix.append(pred_values.values)

    pred_array = np.array(pred_matrix).T  # (n_samples, n_models)

    features['prediction_std'] = np.std(pred_array, axis=1)  # 既存
    features['prediction_min'] = np.min(pred_array, axis=1)
    features['prediction_max'] = np.max(pred_array, axis=1)
    features['prediction_range'] = features['prediction_max'] - features['prediction_min']
    features['prediction_entropy'] = -np.sum(pred_array * np.log(pred_array + 1e-8), axis=1)

    # 高/低予測の割合
    features['high_pred_ratio'] = np.mean(pred_array > 0.5, axis=1)
    features['extreme_pred_ratio'] = np.mean((pred_array < 0.1) | (pred_array > 0.9), axis=1)

    return features


def build_level1_data(
    fold_idx: int,
    splits: Dict,
    predictions: Dict[str, pd.Series],
    y_true: pd.Series,
    idx_to_id: Dict[int, str],
    model_names: List[str],
    config: Dict[str, Any],
    df_train: Optional[pd.DataFrame] = None,
    features_df: Optional[pd.DataFrame] = None,
    train_fps_cache: Optional[Dict[Tuple[str, ...], List]] = None,
    query_fps_cache: Optional[Dict[str, Any]] = None,
    similarity_matrix: Optional[np.ndarray] = None,
    similarity_matrix_ids: Optional[np.ndarray] = None
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
                train_sim = compute_sim_to_train_mean(train_common_ids, train_common_ids, df_train, train_fps_cache=train_fps_cache, query_fps_cache=query_fps_cache, similarity_matrix=similarity_matrix, similarity_matrix_ids=similarity_matrix_ids)
                valid_sim = compute_sim_to_train_mean(valid_common_ids, train_common_ids, df_train, train_fps_cache=train_fps_cache, query_fps_cache=query_fps_cache, similarity_matrix=similarity_matrix, similarity_matrix_ids=similarity_matrix_ids)
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

        # Get early_stopping_rounds, default to 20 if not specified and validation set is provided
        early_stopping_rounds = params.get('early_stopping_rounds', 20)
        if early_stopping_rounds is None:
            early_stopping_rounds = None
        elif X_val is None or y_val is None:
            # If no validation set provided, disable early stopping
            early_stopping_rounds = None

        model = LGBMClassifier(
            n_estimators=params.get('n_estimators', 100),
            learning_rate=params.get('learning_rate', 0.05),
            num_leaves=params.get('num_leaves', 31),
            max_depth=params.get('max_depth', 5),
            min_child_samples=params.get('min_child_samples', 20),
            pos_weight=pos_weight,
            early_stopping_rounds=early_stopping_rounds,
            verbose=-1
        )

        # Convert to DataFrame for LGBMClassifier
        X_train_df = pd.DataFrame(X_train, columns=[f'feat_{i}' for i in range(X_train.shape[1])])
        if X_val is not None and y_val is not None and early_stopping_rounds is not None:
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
    n_trials: int = 50,
    train_fps_cache: Optional[Dict[Tuple[str, ...], List]] = None,
    query_fps_cache: Optional[Dict[str, Any]] = None,
    similarity_matrix: Optional[np.ndarray] = None,
    similarity_matrix_ids: Optional[np.ndarray] = None
) -> Tuple[Any, str, Dict[str, Any], float, List[str], Optional[np.ndarray]]:
    """Train meta-learner for a single fold using Optuna optimization or default LGBM.

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
        n_trials: Number of Optuna trials (only used if use_optuna=True)

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
        features_df=features_df,
        train_fps_cache=train_fps_cache,
        query_fps_cache=query_fps_cache,
        similarity_matrix=similarity_matrix,
        similarity_matrix_ids=similarity_matrix_ids
    )

    if len(valid_ids) == 0 or X_train_level1.shape[0] == 0:
        print(f"    ⚠️  No data available for fold {fold_idx}")
        return None, None, {}, 0.0, [], None

    use_optuna = config.get('use_optuna', False)

    if use_optuna:
        # Use Optuna optimization
        # Create study for this fold
        study_name = f"stacking_{task_key}_fold_{fold_idx}"
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            storage=f"sqlite:///{output_dir}/{study_name}.db",
            load_if_exists=True
        )

        # Optimize
        remaining_trials = n_trials - len(study.trials)
        if remaining_trials > 0:
            print(f"    Running {remaining_trials} Optuna trials...")
            study.optimize(
                lambda trial: objective_meta_learner(
                    trial=trial,
                    X_train=X_train_level1,
                    y_train=y_train_level1,
                    X_valid=X_valid_level1,
                    y_valid=y_valid_level1,
                    model_names=model_names
                ),
                n_trials=remaining_trials,
                show_progress_bar=True
            )
        else:
            print(f"    Using existing study with {len(study.trials)} trials")

        # Get best result
        best_trial = study.best_trial
        best_params = best_trial.params.copy()
        best_model_type = best_params.pop('model_type')

        # Train best model on all level-1 training data
        # NOTE: Do NOT use validation set for early stopping here, as it was already
        # used in Optuna optimization. Using it again would cause data leakage.
        # For LGBM, we use a fixed n_estimators (or increase it slightly) without early stopping.
        if best_model_type == 'lgbm':
            # Increase n_estimators slightly since we're not using early stopping
            # The optimal n_estimators from Optuna trials (with early stopping) is unknown,
            # so we use a conservative fixed value
            best_params['n_estimators'] = best_params.get('n_estimators', 100) * 2
            best_params['early_stopping_rounds'] = None  # Disable early stopping
    else:
        # Use default LGBM without optimization
        print(f"    Using default LGBM (no optimization)...")
        best_model_type = 'lgbm'
        # Merge default params with any overrides from config
        best_params = DEFAULT_LGBM_PARAMS.copy()
        if 'lgbm_params' in config:
            best_params.update(config['lgbm_params'])

    best_model = create_meta_model(
        model_type=best_model_type,
        params=best_params,
        X_train=X_train_level1,
        y_train=y_train_level1,
        X_val=None,  # Do not use validation set for training
        y_val=None
    )

    # Predict on validation set to get final AUC
    # This is the correct OOF prediction: model trained on other folds, predicting on this fold
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
    """Optimize stacking ensemble for a single task using Nested CV approach (v3).

    New flow (fold-specific model selection):
    1. Find available models (with both OOF and test predictions)
    2. Load CV scores per fold
    3. Load all predictions
    4. For each fold:
       a. Select models based on that fold's CV scores
       b. Calculate correlations on that fold's validation set predictions
       c. Select diverse models for this fold
       d. Train a meta-learner on level-1 features from other folds
       e. Predict on the current fold's validation set

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
    print(f"Optimizing stacking ensemble for task: {task_key} (Nested CV with Fold-Specific Model Selection)")
    print(f"{'='*70}")

    # Step 1: Find available models (only those with both OOF and test files)
    available_single, available_multitask, file_paths = find_available_models(task_key, task_name, mt_task_name)

    print(f"\n📊 Available Models (with both OOF and test predictions):")
    print(f"  Single-task models ({len(available_single)}): {available_single}")
    print(f"  Multitask models ({len(available_multitask)}): {available_multitask}")

    if len(available_single) == 0 and len(available_multitask) == 0:
        print(f"\n⚠️  No models available for {task_key} (with both OOF and test files), skipping...")
        return None

    # Step 2: Load CV scores per fold
    print(f"\n📈 Loading CV scores per fold...")
    cv_scores_per_fold = load_cv_scores_per_fold(task_key, task_name, mt_task_name, available_single, available_multitask)
    print(f"  Loaded CV scores per fold for {len(cv_scores_per_fold)} models")

    if len(cv_scores_per_fold) == 0:
        print(f"\n⚠️  No CV scores per fold found for {task_key}, skipping...")
        return None

    # Also load mean CV scores for reporting
    cv_scores = {}
    for model_name, fold_scores in cv_scores_per_fold.items():
        if len(fold_scores) > 0:
            cv_scores[model_name] = np.mean(list(fold_scores.values()))

    # Step 3: Load all predictions
    print(f"\n📥 Loading OOF predictions...")
    all_predictions = load_predictions(file_paths, available_single, available_multitask)
    print(f"  Loaded predictions from {len(all_predictions)} models")

    # Load training data
    print(f"\n📊 Loading training data...")
    df_train = load_train_data(task_key, prepared_file)
    y_true = (df_train[label_col] != 0).astype(int)
    y_true.index = df_train['ID']
    print(f"Training data: {len(df_train)} samples")

    # Calculate OOF AUC for all models
    print(f"\n📊 Calculating OOF AUC for all models...")
    oof_aucs = {}
    for model_name, pred_series in tqdm(all_predictions.items(), desc="  Computing OOF AUC", leave=False):
        # Find common IDs between predictions and labels
        common_ids = sorted(set(pred_series.index) & set(y_true.index))
        if len(common_ids) == 0:
            continue

        try:
            pred_values = pred_series.loc[common_ids].values
            true_values = y_true.loc[common_ids].values
            oof_auc = roc_auc_score(true_values, pred_values)
            oof_aucs[model_name] = oof_auc
        except (ValueError, KeyError) as e:
            # Handle case where only one class is present or other errors
            print(f"    ⚠️  Error computing OOF AUC for {model_name}: {e}")
            continue

    print(f"  Computed OOF AUC for {len(oof_aucs)} models")

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

    # Initialize fingerprint cache for similarity computation (shared across folds)
    train_fps_cache: Dict[Tuple[str, ...], List] = {}
    query_fps_cache: Dict[str, Any] = {}  # Cache for query fingerprints (shared across folds and tasks)

    # Load similarity matrix if available
    similarity_matrix = None
    similarity_matrix_ids = None
    if config.get('use_derived_features', False) and 'sim_to_train_mean' in config.get('derived_features', []):
        feature_dir = config.get('feature_dir', 'ecfp4')
        radius = config.get('similarity_radius', DEFAULT_RADIUS)
        n_bits = config.get('similarity_n_bits', DEFAULT_N_BITS)
        print(f"\n🔍 Loading similarity matrix (radius={radius}, n_bits={n_bits})...")
        sim_result = load_similarity_matrix(task_key, feature_dir, radius=radius, n_bits=n_bits)
        if sim_result is not None:
            similarity_matrix, similarity_matrix_ids = sim_result
        else:
            print(f"  ⚠️  Similarity matrix not found, will compute on-the-fly")

    # Step 7: Train meta-learner
    use_optuna = config.get('use_optuna', False)
    n_trials = config.get('n_trials', DEFAULT_N_TRIALS)

    if use_optuna:
        # Nested CV with Optuna optimization
        print(f"\n🚀 Starting nested CV stacking optimization with {n_trials} trials per fold...")
        print(f"    (Fold-specific model selection enabled)")
        fold_models = []
        fold_model_types = []
        fold_params = []
        fold_aucs = []
        fold_valid_ids = []
        fold_selected_models = []  # Store selected models per fold
        fold_predictions = {}  # Store OOF predictions from training time

        for fold_name, fold_data in tqdm(sorted(splits.items()), desc="Processing folds", total=len(splits)):
            fold_idx = int(fold_name.split("_")[1])

            print(f"\n  Fold {fold_idx}/{n_folds}:")

            # Get validation IDs for this fold (for model selection)
            valid_indices = splits[fold_name]["valid"]
            valid_ids_for_selection = [idx_to_id[idx] for idx in valid_indices if idx in idx_to_id]

            # Step 4a: Select models for this fold
            print(f"    Selecting models for fold {fold_idx}...")
            fold_selected = select_models_for_fold(
                fold_idx=fold_idx,
                cv_scores_per_fold=cv_scores_per_fold,
                all_predictions=all_predictions,
                valid_ids=valid_ids_for_selection,
                config=config
            )
            fold_selected_models.append(fold_selected)

            if len(fold_selected) == 0:
                print(f"    ⚠️  No models selected for fold {fold_idx}, skipping...")
                fold_models.append(None)
                fold_model_types.append(None)
                fold_params.append({})
                fold_aucs.append(0.0)
                fold_valid_ids.append([])
                continue

            print(f"    Selected {len(fold_selected)} models: {fold_selected}")

            # Filter predictions to selected models for this fold
            fold_predictions_dict = {name: all_predictions[name] for name in fold_selected if name in all_predictions}

            print(f"    Optimizing meta-learner...")

            # Train meta-learner for this fold
            best_model, best_model_type, best_params, fold_auc, valid_ids, fold_pred = train_meta_learner_for_fold(
                fold_idx=fold_idx,
                splits=splits,
                predictions=fold_predictions_dict,
                y_true=y_true,
                idx_to_id=idx_to_id,
                model_names=fold_selected,
                task_key=task_key,
                config=config,
                output_dir=output_dir,
                df_train=df_train,
                features_df=features_df,
                n_trials=n_trials,
                train_fps_cache=train_fps_cache,
                query_fps_cache=query_fps_cache,
                similarity_matrix=similarity_matrix,
                similarity_matrix_ids=similarity_matrix_ids
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
    else:
        # No Optuna optimization: train meta-learner for each fold with fold-specific model selection
        print(f"\n🚀 Starting nested CV stacking with default LGBM (no Optuna optimization)...")
        print(f"    (Fold-specific model selection enabled)")
        fold_models = []
        fold_model_types = []
        fold_params = []
        fold_aucs = []
        fold_valid_ids = []
        fold_selected_models = []  # Store selected models per fold
        fold_predictions = {}  # Store OOF predictions from training time

        for fold_name, fold_data in tqdm(sorted(splits.items()), desc="Processing folds", total=len(splits)):
            fold_idx = int(fold_name.split("_")[1])

            print(f"\n  Fold {fold_idx}/{n_folds}:")

            # Get validation IDs for this fold (for model selection)
            valid_indices = splits[fold_name]["valid"]
            valid_ids_for_selection = [idx_to_id[idx] for idx in valid_indices if idx in idx_to_id]

            # Step 4a: Select models for this fold
            print(f"    Selecting models for fold {fold_idx}...")
            fold_selected = select_models_for_fold(
                fold_idx=fold_idx,
                cv_scores_per_fold=cv_scores_per_fold,
                all_predictions=all_predictions,
                valid_ids=valid_ids_for_selection,
                config=config
            )
            fold_selected_models.append(fold_selected)

            if len(fold_selected) == 0:
                print(f"    ⚠️  No models selected for fold {fold_idx}, skipping...")
                fold_models.append(None)
                fold_model_types.append(None)
                fold_params.append({})
                fold_aucs.append(0.0)
                fold_valid_ids.append([])
                continue

            print(f"    Selected {len(fold_selected)} models: {fold_selected}")

            # Filter predictions to selected models for this fold
            fold_predictions_dict = {name: all_predictions[name] for name in fold_selected if name in all_predictions}

            print(f"    Training meta-learner with default LGBM...")

            # Train meta-learner for this fold (no Optuna optimization)
            best_model, best_model_type, best_params, fold_auc, valid_ids, fold_pred = train_meta_learner_for_fold(
                fold_idx=fold_idx,
                splits=splits,
                predictions=fold_predictions_dict,
                y_true=y_true,
                idx_to_id=idx_to_id,
                model_names=fold_selected,
                task_key=task_key,
                config=config,
                output_dir=output_dir,
                df_train=df_train,
                features_df=features_df,
                n_trials=n_trials,
                train_fps_cache=train_fps_cache,
                query_fps_cache=query_fps_cache
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

            print(f"    Model: {best_model_type}, AUC: {fold_auc:.6f}")

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

    if use_optuna:
        print(f"\n✅ Nested CV stacking optimization completed!")
        print(f"Mean fold AUC: {mean_auc:.6f} ± {std_auc:.6f}")
        print(f"Final OOF AUC: {final_auc:.6f}")
        print(f"Meta-model type usage:")
        for mt, count in sorted(model_type_counts.items()):
            print(f"  - {mt}: {count}/{n_folds} folds")
    else:
        print(f"\n✅ Nested CV stacking training completed (no Optuna optimization)!")
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
        'n_models': len(fold_selected_models[0]) if fold_selected_models and len(fold_selected_models[0]) > 0 else 0,
        'n_available_single': len(available_single),
        'n_available_multitask': len(available_multitask),
        'fold_selected_models': fold_selected_models,  # Models selected per fold
        'cv_scores': cv_scores,
        'oof_aucs': oof_aucs,
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
    output_dir = Path(output_dir)
    result_path = output_dir / f"ensemble_optimization_stacking_{task_key}.json"
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {result_path}")

    # Save CV scores and OOF AUCs to CSV
    # Collect all model names
    all_model_names = set(cv_scores.keys()) | set(oof_aucs.keys())

    model_scores_list = []
    for model_name in all_model_names:
        cv_score = cv_scores.get(model_name, None)
        oof_auc = oof_aucs.get(model_name, None)
        # Use OOF AUC for sorting if available, otherwise CV score
        sort_key = oof_auc if oof_auc is not None else (cv_score if cv_score is not None else 0.0)
        model_scores_list.append({
            'model_name': model_name,
            'cv_roc_auc': cv_score,
            'oof_auc': oof_auc,
            'sort_key': sort_key
        })

    # Sort by sort_key (prefer OOF AUC, fallback to CV score)
    model_scores_list.sort(key=lambda x: x['sort_key'] if x['sort_key'] is not None else 0.0, reverse=True)

    # Remove sort_key before saving
    for item in model_scores_list:
        item.pop('sort_key', None)

    model_scores_df = pd.DataFrame(model_scores_list)
    model_scores_path = output_dir / f"model_scores_{task_key}.csv"
    model_scores_df.to_csv(model_scores_path, index=False)
    print(f"Model scores (CV and OOF AUC) saved to {model_scores_path}")

    # Save ensemble predictions (only if OOF predictions are available)
    if len(oof_ids) > 0:
        ensemble_df = pd.DataFrame({
            'ID': oof_ids,
            'prediction': oof_pred_array
        })
        ensemble_path = output_dir / f"ensemble_oof_stacking_{task_key}.csv"
        ensemble_df.to_csv(ensemble_path, index=False)
        print(f"Ensemble predictions saved to {ensemble_path}")

        # Also save CV model average predictions (simple average of base models)
        # Note: For v3, we use fold-specific models, so we average across all available models
        print(f"\n📊 Generating CV model average predictions...")
        cv_avg_pred_array = np.zeros(len(oof_ids))
        n_models_used = 0
        for model_name in all_predictions.keys():
            if model_name in all_predictions:
                cv_avg_pred_array += all_predictions[model_name].loc[oof_ids].values
                n_models_used += 1
        if n_models_used > 0:
            cv_avg_pred_array /= n_models_used
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
    else:
        print(f"\n⚠️  Skipping OOF prediction saving (no OOF predictions available)")

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

    for model_name in tqdm(model_names, desc="  Loading test predictions", leave=False):
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
    for model_name in tqdm(model_names, desc="  Averaging predictions", leave=False):
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
    features_df: Optional[pd.DataFrame] = None,
    train_fps_cache: Optional[Dict[Tuple[str, ...], List]] = None,
    query_fps_cache: Optional[Dict[str, Any]] = None,
    similarity_matrix: Optional[np.ndarray] = None,
    similarity_matrix_ids: Optional[np.ndarray] = None
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
                test_sim = compute_sim_to_train_mean(common_ids, all_train_ids, df_train, train_fps_cache=train_fps_cache, query_fps_cache=query_fps_cache, similarity_matrix=similarity_matrix, similarity_matrix_ids=similarity_matrix_ids)
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
    for model, model_type in tqdm(zip(fold_models, fold_model_types), desc="  Generating predictions from fold models", total=len(fold_models), leave=False):
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


def create_final_submission_from_stacking_v3(output_dir: Path, use_cv_avg: bool = True):
    """Create final submission file from stacking ensemble results (v3).

    Args:
        output_dir: Output directory containing test predictions
        use_cv_avg: If True, use CV average predictions; if False, use stacking predictions
    """
    print("\n" + "="*70)
    print("Creating Final Submission from Stacking Ensembles v3")
    print("="*70)

    # Task mapping: task_key -> (display_name, prediction_file_suffix)
    task_mapping = {
        'trans_340': ('Transmittance(340)', 'trans_340'),
        'trans_450': ('Transmittance(450)', 'trans_450'),
        'fluo_340_450': ('Fluorescence(340/480)', 'fluo_340_450'),
        'fluo_480': ('Fluorescence(multiple)', 'fluo_480'),
    }

    # Load test predictions for all tasks
    task_predictions = {}
    task_results = {}

    for task_key, task_name, label_col, prepared_file, mt_task_name in tasks:
        # Load optimization results
        result_path = output_dir / f"ensemble_optimization_stacking_{task_key}.json"
        if result_path.exists():
            with open(result_path, 'r') as f:
                task_results[task_key] = json.load(f)

        # Load test predictions
        if use_cv_avg:
            test_pred_path = output_dir / f"ensemble_test_cv_avg_{task_key}.csv"
        else:
            test_pred_path = output_dir / f"ensemble_test_stacking_{task_key}.csv"

        if not test_pred_path.exists():
            print(f"⚠️  Test predictions not found for {task_key}: {test_pred_path}")
            continue

        try:
            test_pred_df = pd.read_csv(test_pred_path)
            if 'ID' not in test_pred_df.columns or 'prediction' not in test_pred_df.columns:
                print(f"⚠️  Invalid format in {test_pred_path}")
                continue
            task_predictions[task_key] = test_pred_df
            print(f"✅ Loaded test predictions for {task_key}: {len(test_pred_df)} samples")
        except Exception as e:
            print(f"⚠️  Error loading test predictions for {task_key}: {e}")
            continue

    # Check if we have all 4 tasks
    required_tasks = ['trans_340', 'trans_450', 'fluo_340_450', 'fluo_480']
    missing_tasks = [t for t in required_tasks if t not in task_predictions]

    if missing_tasks:
        print(f"\n⚠️  Missing tasks: {missing_tasks}")
        print("Cannot create final submission without all 4 tasks.")
        return None

    # Create final submission
    print("\n" + "="*70)
    print("Creating final submission...")

    # Align all DataFrames by ID
    all_ids = set(task_predictions['trans_340']['ID'])
    for task_key in required_tasks:
        all_ids &= set(task_predictions[task_key]['ID'])

    all_ids = sorted(all_ids)
    print(f"Common IDs: {len(all_ids)} samples")

    # Create final submission DataFrame
    final_submission = pd.DataFrame({
        "Transmittance(340)": task_predictions['trans_340'].set_index('ID').loc[all_ids, 'prediction'].values,
        "Transmittance(450)": task_predictions['trans_450'].set_index('ID').loc[all_ids, 'prediction'].values,
        "Fluorescence(340/480)": task_predictions['fluo_340_450'].set_index('ID').loc[all_ids, 'prediction'].values,
        "Fluorescence(multiple)": task_predictions['fluo_480'].set_index('ID').loc[all_ids, 'prediction'].values,
    })

    # Clip to [0, 1] (should already be done, but just in case)
    final_submission = final_submission.clip(0, 1)

    # Save final submission
    suffix = "cv_avg" if use_cv_avg else "stacking"
    final_submission_path = output_dir / f"ensemble_final_submission_{suffix}.csv"
    final_submission.to_csv(final_submission_path, index=False)

    print(f"\n✅ Final submission saved to {final_submission_path}")
    print(f"Shape: {final_submission.shape}")

    # Print statistics
    print("\nSubmission Statistics:")
    print("="*70)
    for col in final_submission.columns:
        print(f"  {col:30s}: "
              f"min={final_submission[col].min():.6f}, "
              f"max={final_submission[col].max():.6f}, "
              f"mean={final_submission[col].mean():.6f}")

    # Print final performance metrics (OOF AUC) if available
    if task_results:
        print("\nFinal Performance (OOF AUC):")
        print("="*70)
        for task_key in required_tasks:
            if task_key in task_results:
                result = task_results[task_key]
                final_auc = result.get('final_oof_auc', 0.0)
                mean_fold_auc = result.get('mean_fold_auc', 0.0)
                std_fold_auc = result.get('std_fold_auc', 0.0)
                cv_avg_auc = result.get('cv_avg_auc', 0.0)
                n_models = result.get('n_models', 0)
                task_display_name = task_mapping.get(task_key, (task_key, task_key))[0]
                print(f"  {task_display_name:25s}: "
                      f"Final OOF AUC = {final_auc:.6f}, "
                      f"Mean CV AUC = {mean_fold_auc:.6f} ± {std_fold_auc:.6f}, "
                      f"CV Avg AUC = {cv_avg_auc:.6f}, "
                      f"Models = {n_models}")

    return final_submission


def main(config_path: Optional[str] = None, output_dir: Optional[str] = None, use_optuna: Optional[bool] = None):
    """Main function to optimize stacking ensembles for all tasks.

    Args:
        config_path: Path to YAML config file (optional)
        output_dir: Output directory path (optional, defaults to data/ensembles_stacking_v3)
        use_optuna: Whether to use Optuna optimization. If None, uses config value.
                    If specified, overrides config value.
    """
    print("="*70)
    print("Stacking Ensemble Optimization v3 for All Tasks (Fold-Specific Model Selection)")
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

    # Override use_optuna if specified via command line
    if use_optuna is not None:
        config['use_optuna'] = use_optuna
        print(f"\n⚠️  use_optuna overridden by command line argument: {use_optuna}")

    print(f"\nConfiguration:")
    print(f"  Use Optuna optimization: {config.get('use_optuna', False)}")
    if not config.get('use_optuna', False):
        print(f"  Using default LGBM parameters: {DEFAULT_LGBM_PARAMS}")
    else:
        print(f"  N trials per fold: {config.get('n_trials', DEFAULT_N_TRIALS)}")
    print(f"  Use additional features: {config.get('use_additional_features', False)}")
    print(f"  Feature groups: {config.get('feature_groups', [])}")
    print(f"  Use derived features: {config.get('use_derived_features', False)}")
    print(f"  Derived features: {config.get('derived_features', [])}")
    print(f"  Top N models: {config.get('top_n_models', DEFAULT_TOP_N)}")
    print(f"  Lambda correlation: {config.get('lambda_correlation', DEFAULT_LAMBDA_CORRELATION)}")
    print(f"  Max diverse models: {config.get('max_diverse_models', DEFAULT_MAX_DIVERSE_MODELS)}")

    all_results = []

    for task_key, task_name, label_col, prepared_file, mt_task_name in tqdm(tasks, desc="Processing tasks"):
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
            result_path = output_dir_path / f"ensemble_optimization_stacking_{task_key}.json"
            if result_path.exists():
                with open(result_path, 'r') as f:
                    result_data = json.load(f)

                # Load necessary data
                _, _, file_paths = find_available_models(task_key, task_name, mt_task_name)
                # For v3, use all available models for CV average (since fold-specific selection)
                selected_models = list(file_paths.keys())
                # Add prefix for single-task models
                available_single, available_multitask, _ = find_available_models(task_key, task_name, mt_task_name)
                selected_models = [f"single_{m}" if m in available_single else m for m in selected_models]
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
                    cv_avg_test_path = output_dir_path / f"ensemble_test_cv_avg_{task_key}.csv"
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
        summary_path = output_dir_path / "ensemble_optimization_stacking_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to {summary_path}")

        # Save all results
        all_results_path = output_dir_path / "ensemble_optimization_stacking_all.json"
        with open(all_results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"All results saved to {all_results_path}")

    # Create final submission file
    print("\n" + "="*70)
    print("Creating Final Submission File")
    print("="*70)
    try:
        create_final_submission_from_stacking_v3(output_dir_path, use_cv_avg=True)
    except Exception as e:
        print(f"\n⚠️  Error creating final submission: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    config_path = None
    output_dir = None
    create_submission_only = False
    use_cv_avg = True
    use_optuna = None  # None means use config value

    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--config" and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--output-dir" and i + 1 < len(sys.argv):
            output_dir = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--create-submission":
            create_submission_only = True
            i += 1
        elif sys.argv[i] == "--use-stacking":
            use_cv_avg = False
            i += 1
        elif sys.argv[i] == "--use-optuna":
            use_optuna = True
            i += 1
        elif sys.argv[i] == "--no-optuna":
            use_optuna = False
            i += 1
        elif sys.argv[i] == "--help":
            print("Usage: python optimize_ensemble_stacking_v3.py [OPTIONS]")
            print("Options:")
            print("  --config CONFIG_PATH     Path to YAML config file")
            print("  --output-dir OUTPUT_DIR  Output directory (default: data/ensembles_stacking_v3)")
            print("  --create-submission      Only create submission file from existing predictions")
            print("  --use-stacking           Use stacking predictions instead of CV average (default: CV average)")
            print("  --use-optuna             Use Optuna optimization (overrides config)")
            print("  --no-optuna              Disable Optuna optimization, use default LGBM (overrides config)")
            print("  --help                   Show this help message")
            sys.exit(0)
        else:
            i += 1

    if create_submission_only:
        # Only create submission from existing predictions
        if output_dir is None:
            output_dir_path = DEFAULT_OUTPUT_DIR
        else:
            output_dir_path = Path(output_dir)

        if not output_dir_path.exists():
            print(f"❌ Output directory does not exist: {output_dir_path}")
            sys.exit(1)

        print("="*70)
        print("Creating Submission File from Existing Predictions")
        print("="*70)
        create_final_submission_from_stacking_v3(output_dir_path, use_cv_avg=use_cv_avg)
    else:
        # Run full optimization and then create submission
        main(config_path=config_path, output_dir=output_dir, use_optuna=use_optuna)


"""Optimize ensemble combinations using Optuna for each task.

This script optimizes which models to include in an average ensemble
for each of the 4 tasks (fluo_340_450, fluo_480, trans_340, trans_450).
It considers both single-task and multitask models, and only includes
models that have predictions for the target task.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from pathlib import Path
import optuna
from typing import Dict, List, Tuple
import json

# Note: Model names are now discovered dynamically from data/preds directory
# Only models with both OOF and test predictions for each task are used

# Task configuration: (task_key, single_task_name, label_col, prepared_file, multitask_task_name)
tasks = [
    ('fluo_340_450', 'y_fluo_any', 'Fluorescence', 'train_fluo_340_450_prepared.csv', 'fluorescence340_450'),
    ('fluo_480', 'y_fluo_any', 'Fluorescence', 'train_fluo_480_prepared.csv', 'fluorescence480'),
    ('trans_340', 'y_trans_any', 'Transmittance', 'train_trans_340_prepared.csv', 'transmittance340'),
    ('trans_450', 'y_trans_any', 'Transmittance', 'train_trans_450_prepared.csv', 'transmittance450'),
]

# Paths - Use script location to find project root
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent  # notebooks/ -> project root
pred_base_dir = PROJECT_ROOT / "data" / "preds"
processed_base_dir = PROJECT_ROOT / "data" / "processed"
output_dir = PROJECT_ROOT / "data" / "ensembles_v2"
output_dir.mkdir(parents=True, exist_ok=True)

# Try different feature directories for training data
feature_dirs = ['ecfp4', 'chemeleon', '']

# Try different splits directories
splits_dirs = ['ecfp4', 'chemprop', 'chemeleon', 'chemeleon_lgbm', '']

# Model base directory
models_base_dir = PROJECT_ROOT / "data" / "models"

# Default parameters
DEFAULT_TOP_N = 10
DEFAULT_LAMBDA_CORRELATION = 0.1
DEFAULT_ALPHA_L1 = 0.001
DEFAULT_MAX_DIVERSE_MODELS = 15


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
        # Try to find cv_metrics.csv in models directory
        # Path: data/models/{config_name}/{task_key}/{task_name}/{model_type}/cv_metrics.csv
        # model_name is the config name (e.g., 'full', 'full_focal_optuna')
        # We need to find the model_type (e.g., 'lgbm', 'catboost', 'chemprop')

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
        # For multitask models, path is: data/models/{config_name}/{task_str}/{model_type}/cv_metrics.csv
        # We need to find the task_str that contains mt_task_name

        model_dir = models_base_dir / model_name
        if not model_dir.exists():
            continue

        # Find task directory that contains this task
        for task_subdir in model_dir.iterdir():
            if not task_subdir.is_dir():
                continue

            # Check if this task directory contains the target task in its name
            # e.g., 'fluorescence340_450_fluorescence480' contains 'fluorescence340_450'
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
            # model_name already includes 'multitask_' prefix, so use it directly
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


def objective(
    trial: optuna.Trial,
    predictions: Dict[str, pd.Series],
    y_true: pd.Series,
    model_names: List[str]
) -> float:
    """Optuna objective function for ensemble optimization.

    Args:
        trial: Optuna trial
        predictions: Dictionary of model_name -> prediction Series
        y_true: True labels Series indexed by ID
        model_names: List of available model names

    Returns:
        ROC-AUC score
    """
    # Suggest which models to use (binary variables)
    selected_models = []
    for model_name in model_names:
        use_model = trial.suggest_categorical(f"use_{model_name}", [True, False])
        if use_model:
            selected_models.append(model_name)

    # Ensure at least one model is selected
    if len(selected_models) == 0:
        return 0.0  # Penalty for no models

    # Get common IDs across all selected models and true labels
    common_ids = set(y_true.index)
    for model_name in selected_models:
        common_ids &= set(predictions[model_name].index)

    if len(common_ids) == 0:
        return 0.0  # Penalty for no common IDs

    common_ids = sorted(common_ids)

    # Calculate ensemble prediction (simple average)
    ensemble_pred = np.zeros(len(common_ids))
    for model_name in selected_models:
        ensemble_pred += predictions[model_name].loc[common_ids].values

    ensemble_pred /= len(selected_models)

    # Calculate ROC-AUC
    y_true_subset = y_true.loc[common_ids].values
    try:
        auc = roc_auc_score(y_true_subset, ensemble_pred)
    except ValueError:
        # Handle case where only one class is present
        return 0.0

    return auc


def optimize_ensemble_for_fold(
    fold_idx: int,
    valid_ids: List[str],
    predictions: Dict[str, pd.Series],
    y_true: pd.Series,
    model_names: List[str],
    task_key: str,
    n_trials: int = 100,
    alpha_l1: float = DEFAULT_ALPHA_L1
) -> Tuple[Dict[str, float], float]:
    """Optimize ensemble weights for a single fold using Optuna.

    Args:
        fold_idx: Fold index
        valid_ids: List of validation IDs for this fold
        predictions: Dictionary of model_name -> prediction Series
        y_true: True labels Series indexed by ID
        model_names: List of available model names
        task_key: Task key for study naming
        n_trials: Number of Optuna trials
        alpha_l1: L1 regularization coefficient for weights

    Returns:
        Tuple of (weights_dict, best_auc)
        where weights_dict maps model_name to weight
    """
    # Filter to validation set only
    valid_ids_set = set(valid_ids)
    y_true_valid = y_true.loc[y_true.index.isin(valid_ids_set)]

    # Create study for this fold - delete existing study if it exists
    study_name = f"ensemble_{task_key}_fold_{fold_idx}"
    study_db_path = output_dir / f"{study_name}.db"

    # Delete existing study database if it exists (for fresh start)
    if study_db_path.exists():
        try:
            study_db_path.unlink()
        except Exception as e:
            print(f"    Warning: Could not delete existing study database: {e}")

    # Create new study
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        storage=f"sqlite:///{study_db_path}",
        load_if_exists=False
    )

    # Optimize on validation set only
    def fold_objective(trial):
        # Suggest weights for each model (0.0 to 1.0)
        weights = {}
        for model_name in model_names:
            weights[model_name] = trial.suggest_float(f"weight_{model_name}", 0.0, 1.0)

        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight == 0:
            return 0.0  # Penalty for all zeros

        # Normalize
        weights = {k: v / total_weight for k, v in weights.items()}

        # Get common IDs across all models and validation labels
        common_ids = set(y_true_valid.index)
        for model_name in model_names:
            if weights[model_name] > 0:  # Only consider models with non-zero weight
                model_pred = predictions[model_name]
                model_valid_ids = model_pred.index[model_pred.index.isin(valid_ids_set)]
                common_ids &= set(model_valid_ids)

        if len(common_ids) == 0:
            return 0.0

        common_ids = sorted(common_ids)

        # Calculate weighted ensemble prediction
        ensemble_pred = np.zeros(len(common_ids))
        for model_name in model_names:
            if weights[model_name] > 0:
                ensemble_pred += weights[model_name] * predictions[model_name].loc[common_ids].values

        # Calculate ROC-AUC on validation set only
        y_true_subset = y_true_valid.loc[common_ids].values
        try:
            auc = roc_auc_score(y_true_subset, ensemble_pred)
        except ValueError:
            return 0.0

        # L1 regularization penalty
        l1_penalty = alpha_l1 * sum(abs(w) for w in weights.values())

        # Return regularized score
        return auc - l1_penalty

    # Optimize with fresh start (use full n_trials)
    study.optimize(
        fold_objective,
        n_trials=n_trials,
        show_progress_bar=False
    )

    # Get best result
    best_trial = study.best_trial
    best_params = best_trial.params

    # Extract weights and normalize
    weights = {}
    for model_name in model_names:
        weight = best_params.get(f"weight_{model_name}", 0.0)
        weights[model_name] = max(0.0, weight)  # Ensure non-negative

    # Normalize to sum to 1
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v / total_weight for k, v in weights.items()}

    # Get best AUC (without regularization penalty)
    best_auc = float(best_trial.value)
    # Add back the penalty to get true AUC (approximate, since we're using normalized weights)
    # Actually, we'll recalculate it properly below

    # Recalculate AUC with best weights for validation set
    valid_ids_set = set(valid_ids)
    y_true_valid = y_true.loc[y_true.index.isin(valid_ids_set)]
    common_ids = set(y_true_valid.index)
    for model_name in model_names:
        if weights[model_name] > 0:
            model_pred = predictions[model_name]
            model_valid_ids = model_pred.index[model_pred.index.isin(valid_ids_set)]
            common_ids &= set(model_valid_ids)

    if len(common_ids) > 0:
        common_ids = sorted(common_ids)
        ensemble_pred = np.zeros(len(common_ids))
        for model_name in model_names:
            if weights[model_name] > 0:
                ensemble_pred += weights[model_name] * predictions[model_name].loc[common_ids].values

        try:
            true_auc = roc_auc_score(y_true_valid.loc[common_ids].values, ensemble_pred)
            best_auc = true_auc
        except ValueError:
            pass

    return weights, best_auc


def optimize_ensemble_for_task(
    task_key: str,
    task_name: str,
    label_col: str,
    prepared_file: str,
    mt_task_name: str,
    n_trials: int = 100,
    top_n: int = DEFAULT_TOP_N,
    lambda_correlation: float = DEFAULT_LAMBDA_CORRELATION,
    alpha_l1: float = DEFAULT_ALPHA_L1,
    max_diverse_models: int = DEFAULT_MAX_DIVERSE_MODELS
) -> Dict:
    """Optimize ensemble for a single task using Nested CV approach.

    New flow:
    1. Load CV scores and select top N models
    2. Select diverse models considering correlation
    3. Optimize weights for selected models using nested CV

    Args:
        task_key: Task key
        task_name: Single task name
        label_col: Label column name
        prepared_file: Prepared data file name
        mt_task_name: Multitask task name
        n_trials: Number of Optuna trials per fold
        top_n: Number of top models to consider by CV score
        lambda_correlation: Correlation penalty coefficient
        alpha_l1: L1 regularization coefficient for weights
        max_diverse_models: Maximum number of diverse models to select

    Returns:
        Dictionary with optimization results
    """
    print(f"\n{'='*70}")
    print(f"Optimizing ensemble for task: {task_key} (Nested CV with Diversity)")
    print(f"{'='*70}")

    # Step 1: Find available models (only those with both OOF and test files)
    available_single, available_multitask, file_paths = find_available_models(task_key, task_name, mt_task_name)

    print(f"\n📊 Available Models (with both OOF and test predictions):")
    print(f"  Single-task models ({len(available_single)}):")
    for model_name in available_single:
        oof_path = file_paths[model_name]['oof']
        test_path = file_paths[model_name]['test']
        print(f"    ✓ {model_name}")
        print(f"      OOF:  {oof_path.relative_to(PROJECT_ROOT)}")
        print(f"      Test: {test_path.relative_to(PROJECT_ROOT)}")

    print(f"  Multitask models ({len(available_multitask)}):")
    for model_name in available_multitask:
        oof_path = file_paths[model_name]['oof']
        test_path = file_paths[model_name]['test']
        print(f"    ✓ {model_name}")
        print(f"      OOF:  {oof_path.relative_to(PROJECT_ROOT)}")
        print(f"      Test: {test_path.relative_to(PROJECT_ROOT)}")

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
        # Calculate mean correlation with other selected models
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

    # Load CV splits
    print("Loading CV splits...")
    splits = load_splits(task_key)
    n_folds = len(splits)
    print(f"Found {n_folds} folds")

    # Create index to ID mapping (splits use row indices)
    id_to_idx = {id_val: idx for idx, id_val in enumerate(df_train['ID'].values)}
    idx_to_id = {idx: id_val for idx, id_val in enumerate(df_train['ID'].values)}

    # Step 7: Optimize weights for each fold
    print(f"\n🚀 Starting nested CV optimization with {n_trials} trials per fold...")
    fold_weights = []
    fold_aucs = []

    for fold_name, fold_data in sorted(splits.items()):
        fold_idx = int(fold_name.split("_")[1])
        valid_indices = fold_data["valid"]
        valid_ids = [idx_to_id[idx] for idx in valid_indices if idx in idx_to_id]

        print(f"\n  Fold {fold_idx}: {len(valid_ids)} validation samples")

        # Optimize ensemble weights for this fold
        weights, fold_auc = optimize_ensemble_for_fold(
            fold_idx=fold_idx,
            valid_ids=valid_ids,
            predictions=final_predictions,
            y_true=y_true,
            model_names=final_model_names,
            task_key=task_key,
            n_trials=n_trials,
            alpha_l1=alpha_l1
        )

        fold_weights.append(weights)
        fold_aucs.append(fold_auc)

        # Print weights (non-zero only)
        non_zero_weights = {k: v for k, v in weights.items() if v > 1e-6}
        print(f"    AUC: {fold_auc:.6f}, Active models: {len(non_zero_weights)}")
        if len(non_zero_weights) <= 5:
            for model_name, weight in sorted(non_zero_weights.items(), key=lambda x: x[1], reverse=True):
                print(f"      {model_name}: {weight:.4f}")
        else:
            top_weights = sorted(non_zero_weights.items(), key=lambda x: x[1], reverse=True)[:3]
            for model_name, weight in top_weights:
                print(f"      {model_name}: {weight:.4f}")
            print(f"      ... (+{len(non_zero_weights)-3} more)")

    # Calculate mean AUC across folds
    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)

    # Calculate average weights across folds
    avg_weights = {}
    for model_name in final_model_names:
        weights_list = [w.get(model_name, 0.0) for w in fold_weights]
        avg_weights[model_name] = np.mean(weights_list)

    # Sort by average weight
    sorted_avg_weights = sorted(avg_weights.items(), key=lambda x: x[1], reverse=True)

    # Generate OOF predictions using fold-specific weights
    print(f"\n📝 Generating OOF predictions using fold-specific weights...")
    oof_predictions = {}

    for fold_name, fold_data in sorted(splits.items()):
        fold_idx = int(fold_name.split("_")[1])
        valid_indices = fold_data["valid"]
        valid_ids = [idx_to_id[idx] for idx in valid_indices if idx in idx_to_id]
        weights = fold_weights[fold_idx]

        # Get common IDs for this fold
        common_ids = set(valid_ids)
        for model_name in final_model_names:
            if weights.get(model_name, 0.0) > 0:
                common_ids &= set(final_predictions[model_name].index)
        common_ids = sorted(common_ids)

        # Calculate weighted ensemble prediction for this fold
        ensemble_pred = np.zeros(len(common_ids))
        for model_name in final_model_names:
            weight = weights.get(model_name, 0.0)
            if weight > 0:
                ensemble_pred += weight * final_predictions[model_name].loc[common_ids].values

        # Store predictions
        for id_val, pred in zip(common_ids, ensemble_pred):
            oof_predictions[id_val] = pred

    # Convert to arrays for final evaluation
    oof_ids = sorted(oof_predictions.keys())
    oof_pred_array = np.array([oof_predictions[id_val] for id_val in oof_ids])
    oof_true_array = y_true.loc[oof_ids].values

    # Calculate final AUC on all OOF predictions
    try:
        final_auc = roc_auc_score(oof_true_array, oof_pred_array)
    except ValueError:
        final_auc = 0.0

    print(f"\n✅ Nested CV optimization completed!")
    print(f"Mean fold AUC: {mean_auc:.6f} ± {std_auc:.6f}")
    print(f"Final OOF AUC: {final_auc:.6f}")
    print(f"Average weights across folds:")
    for model_name, avg_weight in sorted_avg_weights[:10]:  # Show top 10
        if avg_weight > 1e-6:
            print(f"  - {model_name}: {avg_weight:.4f}")

    # Save results
    result = {
        'task': task_key,
        'mean_fold_auc': float(mean_auc),
        'std_fold_auc': float(std_auc),
        'final_oof_auc': float(final_auc),
        'fold_aucs': [float(auc) for auc in fold_aucs],
        'fold_weights': fold_weights,
        'avg_weights': avg_weights,
        'selected_models': diverse_models,
        'top_models': top_models,
        'cv_scores': cv_scores,
        'n_models_selected': len(diverse_models),
        'n_available_single': len(available_single),
        'n_available_multitask': len(available_multitask),
        'n_samples': len(oof_ids),
        'n_folds': n_folds,
        'n_trials_per_fold': n_trials,
        'top_n': top_n,
        'lambda_correlation': lambda_correlation,
        'alpha_l1': alpha_l1,
        'max_diverse_models': max_diverse_models,
        'file_paths': {
            model_name: {
                'oof': str(paths['oof'].relative_to(PROJECT_ROOT)),
                'test': str(paths['test'].relative_to(PROJECT_ROOT))
            }
            for model_name, paths in file_paths.items()
        }
    }

    # Save to JSON
    result_path = output_dir / f"ensemble_optimization_{task_key}.json"
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {result_path}")

    # Save ensemble predictions
    ensemble_df = pd.DataFrame({
        'ID': oof_ids,
        'prediction': oof_pred_array
    })
    ensemble_path = output_dir / f"ensemble_oof_{task_key}.csv"
    ensemble_df.to_csv(ensemble_path, index=False)
    print(f"Ensemble predictions saved to {ensemble_path}")

    return result


def main():
    """Main function to optimize ensembles for all tasks."""
    print("="*70)
    print("Ensemble Optimization for All Tasks")
    print("="*70)

    all_results = []

    for task_key, task_name, label_col, prepared_file, mt_task_name in tasks:
        try:
            result = optimize_ensemble_for_task(
                task_key=task_key,
                task_name=task_name,
                label_col=label_col,
                prepared_file=prepared_file,
                mt_task_name=mt_task_name,
                n_trials=200  # Adjust as needed
            )
            if result:
                all_results.append(result)
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
                'mean_fold_auc': r.get('mean_fold_auc', r.get('best_auc', 0.0)),
                'final_oof_auc': r.get('final_oof_auc', r.get('best_auc', 0.0)),
                'n_models': r.get('n_models_selected', r.get('n_models_consensus', r.get('n_models', 0))),
                'n_available': r['n_available_single'] + r['n_available_multitask'],
                'n_samples': r['n_samples']
            }
            for r in all_results
        ])
        print(summary_df.to_string(index=False))

        # Save summary
        summary_path = output_dir / "ensemble_optimization_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to {summary_path}")

        # Save all results
        all_results_path = output_dir / "ensemble_optimization_all.json"
        with open(all_results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"All results saved to {all_results_path}")


def generate_test_predictions_for_task(
    task_key: str,
    task_name: str,
    mt_task_name: str,
    selected_models: List[str],
    file_paths: Dict[str, Dict[str, Path]] = None,
    weights: Dict[str, float] = None
) -> pd.DataFrame:
    """Generate test predictions for a task using selected models with optional weights.

    Args:
        task_key: Task key
        task_name: Single task name
        mt_task_name: Multitask task name
        selected_models: List of selected model names (with prefix like "single_full")
        file_paths: Optional dictionary mapping model_name to {'oof': Path, 'test': Path}
                    If not provided, will be recalculated
        weights: Optional dictionary mapping model_name to weight.
                 If None, uses equal weights (simple average)

    Returns:
        DataFrame with ID and prediction columns
    """
    predictions = {}

    # If file_paths not provided, recalculate them
    if file_paths is None:
        _, _, file_paths = find_available_models(task_key, task_name, mt_task_name)

    for model_name in selected_models:
        if model_name.startswith("single_"):
            # Single-task model - remove "single_" prefix
            actual_model_name = model_name.replace("single_", "")
            if actual_model_name in file_paths:
                test_path = file_paths[actual_model_name]['test']
            else:
                # Fallback to old method
                test_path = pred_base_dir / actual_model_name / task_key / f"{task_name}_test.csv"
        elif model_name.startswith("multitask_"):
            # Multitask model - handle both 'multitask_multitask_*' and 'multitask_*' formats
            if model_name.startswith("multitask_multitask_"):
                actual_model_name = model_name.replace("multitask_", "", 1)  # Remove first occurrence
            else:
                actual_model_name = model_name  # Already correct format (e.g., 'multitask_all_chemprop')

            if actual_model_name in file_paths:
                test_path = file_paths[actual_model_name]['test']
            else:
                # Fallback to old method
                test_path = pred_base_dir / actual_model_name / "test" / f"{mt_task_name}_test.csv"
        else:
            # Try using model_name directly (might be a multitask model without prefix issue)
            if model_name in file_paths:
                test_path = file_paths[model_name]['test']
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
            predictions[model_name] = df['prediction']
        except Exception as e:
            print(f"  ⚠️  Error loading {model_name}: {e}")
            continue

    if len(predictions) == 0:
        raise ValueError(f"No test predictions loaded for {task_key}")

    # Get common IDs
    common_ids = set(list(predictions.values())[0].index)
    for pred in predictions.values():
        common_ids &= set(pred.index)
    common_ids = sorted(common_ids)

    # Calculate ensemble (weighted or simple average)
    ensemble_pred = np.zeros(len(common_ids))

    if weights is not None:
        # Weighted ensemble
        total_weight = 0.0
        for model_name in selected_models:
            if model_name in predictions and model_name in weights:
                weight = weights[model_name]
                if weight > 0:
                    ensemble_pred += weight * predictions[model_name].loc[common_ids].values
                    total_weight += weight
        if total_weight > 0:
            ensemble_pred /= total_weight
    else:
        # Simple average
        valid_models = [m for m in selected_models if m in predictions]
        for model_name in valid_models:
            ensemble_pred += predictions[model_name].loc[common_ids].values
        ensemble_pred /= len(valid_models)

    # Clip to [0, 1]
    ensemble_pred = np.clip(ensemble_pred, 0, 1)

    return pd.DataFrame({
        'ID': common_ids,
        'prediction': ensemble_pred
    })


def create_final_submission_from_optimization():
    """Create final submission file from optimized ensemble results."""
    print("\n" + "="*70)
    print("Creating Final Submission from Optimized Ensembles")
    print("="*70)

    # Load optimization results
    task_predictions = {}
    task_results = {}

    for task_key, task_name, label_col, prepared_file, mt_task_name in tasks:
        result_path = output_dir / f"ensemble_optimization_{task_key}.json"

        if not result_path.exists():
            print(f"⚠️  Optimization result not found for {task_key}, skipping...")
            continue

        with open(result_path, 'r') as f:
            result = json.load(f)

        # Store result for performance summary
        task_results[task_key] = result

        # Use selected models and average weights from optimization
        selected_models = result.get('selected_models', result.get('consensus_models', []))
        avg_weights = result.get('avg_weights', None)

        if len(selected_models) == 0:
            print(f"⚠️  No models selected for {task_key}, skipping...")
            continue

        print(f"\nGenerating test predictions for {task_key}...")
        print(f"Selected models: {selected_models}")
        if avg_weights:
            print(f"Using average weights from CV optimization")
            # Show top weights
            sorted_weights = sorted(avg_weights.items(), key=lambda x: x[1], reverse=True)[:5]
            for model_name, weight in sorted_weights:
                if weight > 1e-6:
                    print(f"  {model_name}: {weight:.4f}")

        # Reconstruct file_paths from saved result if available
        file_paths = None
        if 'file_paths' in result:
            file_paths = {}
            for model_name, paths in result['file_paths'].items():
                file_paths[model_name] = {
                    'oof': PROJECT_ROOT / paths['oof'],
                    'test': PROJECT_ROOT / paths['test']
                }

        try:
            test_pred_df = generate_test_predictions_for_task(
                task_key=task_key,
                task_name=task_name,
                mt_task_name=mt_task_name,
                selected_models=selected_models,
                file_paths=file_paths,
                weights=avg_weights  # Use average weights if available
            )

            # Save individual task submission
            task_submission_path = output_dir / f"ensemble_test_{task_key}.csv"
            test_pred_df.to_csv(task_submission_path, index=False)
            print(f"  Saved to {task_submission_path}")

            task_predictions[task_key] = test_pred_df
        except Exception as e:
            print(f"  ❌ Error generating test predictions for {task_key}: {e}")
            import traceback
            traceback.print_exc()

    # Check if we have all 4 tasks
    required_tasks = ['trans_340', 'trans_450', 'fluo_340_450', 'fluo_480']
    missing_tasks = [t for t in required_tasks if t not in task_predictions]

    if missing_tasks:
        print(f"\n⚠️  Missing tasks: {missing_tasks}")
        print("Cannot create final submission without all 4 tasks.")
        return

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
    final_submission_path = output_dir / "ensemble_final_submission.csv"
    final_submission.to_csv(final_submission_path, index=False)

    print(f"\n✅ Final submission saved to {final_submission_path}")
    print(f"Shape: {final_submission.shape}")

    # Print final performance metrics
    print("\nFinal Performance (OOF AUC):")
    print("="*70)
    task_name_map = {
        'trans_340': 'Transmittance(340)',
        'trans_450': 'Transmittance(450)',
        'fluo_340_450': 'Fluorescence(340/480)',
        'fluo_480': 'Fluorescence(multiple)',
    }
    for task_key in required_tasks:
        if task_key in task_results:
            result = task_results[task_key]
            final_auc = result.get('final_oof_auc', 0.0)
            mean_fold_auc = result.get('mean_fold_auc', 0.0)
            std_fold_auc = result.get('std_fold_auc', 0.0)
            n_models = result.get('n_models_selected', result.get('n_models_consensus', 0))
            task_display_name = task_name_map.get(task_key, task_key)
            print(f"  {task_display_name:25s}: "
                  f"Final OOF AUC = {final_auc:.6f}, "
                  f"Mean CV AUC = {mean_fold_auc:.6f} ± {std_fold_auc:.6f}, "
                  f"Models = {n_models}")

    return final_submission


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--create-submission":
        # Only create submission from existing optimization results
        create_final_submission_from_optimization()
    else:
        # Run optimization and then create submission
        main()
        print("\n" + "="*70)
        create_final_submission_from_optimization()



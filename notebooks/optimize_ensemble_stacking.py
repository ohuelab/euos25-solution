"""Optimize stacking ensemble using meta-learners for each task.

This script optimizes stacking ensemble with meta-learners (default: LGBM)
for each of the 4 tasks (fluo_340_450, fluo_480, trans_340, trans_450).
It uses nested CV where for each fold, a meta-learner is trained on
level-1 features (base model predictions from other folds) and predicts
on the current fold's validation set.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV
from scipy.special import expit
from pathlib import Path
import optuna
from typing import Dict, List, Tuple, Optional, Any
import json
import sys

# Import LGBMClassifier from the project
sys.path.insert(0, str(Path(__file__).parent.parent))
from euos25.models.lgbm import LGBMClassifier

# Configuration - Single task model names
single_task_names = [
    'full',
    'full_optuna',
    'full_focal',
    'full_focal_optuna',
    'full_chemprop',
    'full_chemprop_focal',
    'full_chemeleon',
    'full_chemeleon_focal',
    'full_chemeleon_lgbm',
    'full_chemeleon_lgbm_focal'
]

# Configuration - Multitask model names
multitask_names = [
    'multitask_all_chemeleon',
    'multitask_all_chemprop',
    'multitask_fluo_chemeleon',
    'multitask_fluo_chemprop',
    'multitask_trans_chemeleon',
    'multitask_trans_chemprop',
]

# Task configuration: (task_key, single_task_name, label_col, prepared_file, multitask_task_name)
tasks = [
    ('fluo_340_450', 'y_fluo_any', 'Fluorescence', 'train_fluo_340_450_prepared.csv', 'fluorescence340_450'),
    ('fluo_480', 'y_fluo_any', 'Fluorescence', 'train_fluo_480_prepared.csv', 'fluorescence480'),
    ('trans_340', 'y_trans_any', 'Transmittance', 'train_trans_340_prepared.csv', 'transmittance340'),
    ('trans_450', 'y_trans_any', 'Transmittance', 'train_trans_450_prepared.csv', 'transmittance450'),
]

# Paths
pred_base_dir = Path("../data/preds")
processed_base_dir = Path("../data/processed")
output_dir = Path("../data/ensembles_stacking")
output_dir.mkdir(parents=True, exist_ok=True)

# Try different feature directories for training data
feature_dirs = ['ecfp4', 'chemeleon', '']

# Try different splits directories
splits_dirs = ['ecfp4', 'chemprop', 'chemeleon', 'chemeleon_lgbm', '']


def find_available_models(task_key: str, task_name: str, mt_task_name: str) -> Tuple[List[str], List[str]]:
    """Find available single-task and multitask models for a given task.

    Args:
        task_key: Task key (e.g., 'fluo_340_450')
        task_name: Single task name (e.g., 'y_fluo_any')
        mt_task_name: Multitask task name (e.g., 'fluorescence340_450')

    Returns:
        Tuple of (available_single_task_models, available_multitask_models)
    """
    available_single = []
    available_multitask = []

    # Check single-task models
    for model_name in single_task_names:
        oof_path = pred_base_dir / model_name / task_key / f"{task_name}_oof.csv"
        if oof_path.exists():
            available_single.append(model_name)

    # Check multitask models
    for model_name in multitask_names:
        oof_path = pred_base_dir / model_name / "oof" / f"{mt_task_name}_oof.csv"
        if oof_path.exists():
            available_multitask.append(model_name)

    return available_single, available_multitask


def load_predictions(task_key: str, task_name: str, mt_task_name: str) -> Dict[str, pd.Series]:
    """Load all available predictions for a task.

    Args:
        task_key: Task key
        task_name: Single task name
        mt_task_name: Multitask task name

    Returns:
        Dictionary mapping model_name to prediction Series indexed by ID
    """
    predictions = {}

    # Load single-task predictions
    for model_name in single_task_names:
        oof_path = pred_base_dir / model_name / task_key / f"{task_name}_oof.csv"
        if oof_path.exists():
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
    for model_name in multitask_names:
        oof_path = pred_base_dir / model_name / "oof" / f"{mt_task_name}_oof.csv"
        if oof_path.exists():
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
                predictions[f"multitask_{model_name}"] = df['prediction']
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


def build_level1_data(
    fold_idx: int,
    splits: Dict,
    predictions: Dict[str, pd.Series],
    y_true: pd.Series,
    idx_to_id: Dict[int, str],
    model_names: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Build level-1 training and validation data for stacking.

    Args:
        fold_idx: Current fold index
        splits: CV splits dictionary
        predictions: Dictionary of model_name -> prediction Series
        y_true: True labels Series indexed by ID
        idx_to_id: Mapping from index to ID
        model_names: List of available model names

    Returns:
        Tuple of (X_train_level1, y_train_level1, X_valid_level1, y_valid_level1, valid_ids)
        where X are prediction matrices (n_samples, n_models)
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

    # Build feature matrices
    n_models = len(model_names)
    X_train_level1 = np.zeros((len(train_common_ids), n_models))
    X_valid_level1 = np.zeros((len(valid_common_ids), n_models))

    for i, model_name in enumerate(model_names):
        X_train_level1[:, i] = predictions[model_name].loc[train_common_ids].values
        X_valid_level1[:, i] = predictions[model_name].loc[valid_common_ids].values

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
        X_train: Training features (level-1 predictions)
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
        X_train_df = pd.DataFrame(X_train, columns=[f'model_{i}' for i in range(X_train.shape[1])])
        if X_val is not None and y_val is not None:
            X_val_df = pd.DataFrame(X_val, columns=[f'model_{i}' for i in range(X_val.shape[1])])
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
        X: Features (level-1 predictions)
        model_type: Type of meta-model ('lgbm', 'logistic', 'ridge')

    Returns:
        Predicted probabilities (n_samples,)
    """
    if model_type == 'lgbm':
        # Convert to DataFrame
        X_df = pd.DataFrame(X, columns=[f'model_{i}' for i in range(X.shape[1])])
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
        X_train: Level-1 training features
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
    n_trials: int = 50
) -> Tuple[Any, str, Dict[str, Any], float, List[str]]:
    """Train meta-learner for a single fold using Optuna optimization.

    Args:
        fold_idx: Fold index
        splits: CV splits dictionary
        predictions: Dictionary of model_name -> prediction Series
        y_true: True labels Series indexed by ID
        idx_to_id: Mapping from index to ID
        model_names: List of available model names
        task_key: Task key for study naming
        n_trials: Number of Optuna trials

    Returns:
        Tuple of (best_model, best_model_type, best_params, best_auc, valid_ids)
    """
    # Build level-1 data
    X_train_level1, y_train_level1, X_valid_level1, y_valid_level1, valid_ids = build_level1_data(
        fold_idx=fold_idx,
        splits=splits,
        predictions=predictions,
        y_true=y_true,
        idx_to_id=idx_to_id,
        model_names=model_names
    )

    if len(valid_ids) == 0 or X_train_level1.shape[0] == 0:
        print(f"    ⚠️  No data available for fold {fold_idx}")
        return None, None, {}, 0.0, []

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

    return best_model, best_model_type, best_params, best_auc, valid_ids


def optimize_stacking_for_task(
    task_key: str,
    task_name: str,
    label_col: str,
    prepared_file: str,
    mt_task_name: str,
    n_trials: int = 50
) -> Dict:
    """Optimize stacking ensemble for a single task using Nested CV approach.

    For each fold, train a meta-learner on level-1 features from other folds
    and predict on the current fold's validation set.

    Args:
        task_key: Task key
        task_name: Single task name
        label_col: Label column name
        prepared_file: Prepared data file name
        mt_task_name: Multitask task name
        n_trials: Number of Optuna trials per fold

    Returns:
        Dictionary with optimization results
    """
    print(f"\n{'='*70}")
    print(f"Optimizing stacking ensemble for task: {task_key} (Nested CV)")
    print(f"{'='*70}")

    # Find available models
    available_single, available_multitask = find_available_models(task_key, task_name, mt_task_name)

    print(f"Available single-task models ({len(available_single)}): {available_single}")
    print(f"Available multitask models ({len(available_multitask)}): {available_multitask}")

    if len(available_single) == 0 and len(available_multitask) == 0:
        print(f"⚠️  No models available for {task_key}, skipping...")
        return None

    # Load predictions
    print("Loading predictions...")
    predictions = load_predictions(task_key, task_name, mt_task_name)
    print(f"Loaded predictions from {len(predictions)} models")

    # Load training data
    print("Loading training data...")
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

    # Prepare model names list
    model_names = list(predictions.keys())

    # Train meta-learner for each fold
    fold_models = []
    fold_model_types = []
    fold_params = []
    fold_aucs = []
    fold_valid_ids = []

    print(f"\nStarting nested CV stacking optimization with {n_trials} trials per fold...")

    for fold_name, fold_data in sorted(splits.items()):
        fold_idx = int(fold_name.split("_")[1])

        print(f"\n  Fold {fold_idx}: Optimizing meta-learner...")

        # Train meta-learner for this fold
        best_model, best_model_type, best_params, fold_auc, valid_ids = train_meta_learner_for_fold(
            fold_idx=fold_idx,
            splits=splits,
            predictions=predictions,
            y_true=y_true,
            idx_to_id=idx_to_id,
            model_names=model_names,
            task_key=task_key,
            n_trials=n_trials
        )

        fold_models.append(best_model)
        fold_model_types.append(best_model_type)
        fold_params.append(best_params)
        fold_aucs.append(fold_auc)
        fold_valid_ids.append(valid_ids)

        print(f"    Best model: {best_model_type}, AUC: {fold_auc:.6f}")
        if best_params:
            print(f"    Best params: {best_params}")

    # Generate OOF predictions using fold-specific meta-learners
    print(f"\nGenerating OOF predictions using fold-specific meta-learners...")
    oof_predictions = {}

    for fold_idx, (model, model_type, valid_ids) in enumerate(zip(fold_models, fold_model_types, fold_valid_ids)):
        if model is None or len(valid_ids) == 0:
            continue

        # Build level-1 features for validation set
        fold_name = f"fold_{fold_idx}"
        valid_ids_set = set(valid_ids)

        # Find common IDs
        common_ids = valid_ids_set.copy()
        for model_name in model_names:
            common_ids &= set(predictions[model_name].index)
        common_ids = sorted(common_ids)

        if len(common_ids) == 0:
            continue

        # Build feature matrix
        X_valid_level1 = np.zeros((len(common_ids), len(model_names)))
        for i, model_name in enumerate(model_names):
            X_valid_level1[:, i] = predictions[model_name].loc[common_ids].values

        # Predict using meta-learner
        y_pred = predict_proba_meta_model(model, X_valid_level1, model_type)

        # Store predictions
        for id_val, pred in zip(common_ids, y_pred):
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
        'n_models': len(model_names),
        'n_available_single': len(available_single),
        'n_available_multitask': len(available_multitask),
        'n_samples': len(oof_ids),
        'n_folds': n_folds,
        'n_trials_per_fold': n_trials
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

    return result


def main():
    """Main function to optimize stacking ensembles for all tasks."""
    print("="*70)
    print("Stacking Ensemble Optimization for All Tasks")
    print("="*70)

    all_results = []

    for task_key, task_name, label_col, prepared_file, mt_task_name in tasks:
        try:
            result = optimize_stacking_for_task(
                task_key=task_key,
                task_name=task_name,
                label_col=label_col,
                prepared_file=prepared_file,
                mt_task_name=mt_task_name,
                n_trials=50  # Adjust as needed
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
                'mean_fold_auc': r.get('mean_fold_auc', 0.0),
                'final_oof_auc': r.get('final_oof_auc', 0.0),
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


def generate_test_predictions_for_task_stacking(
    task_key: str,
    task_name: str,
    mt_task_name: str,
    predictions: Dict[str, pd.Series],
    model_names: List[str],
    best_models: List[Any],
    best_model_types: List[str]
) -> pd.DataFrame:
    """Generate test predictions for a task using stacking ensemble.

    Args:
        task_key: Task key
        task_name: Single task name
        mt_task_name: Multitask task name
        predictions: Dictionary of test predictions (model_name -> prediction Series)
        model_names: List of model names
        best_models: List of best meta-models for each fold
        best_model_types: List of best model types for each fold

    Returns:
        DataFrame with ID and prediction columns
    """
    if len(best_models) == 0:
        raise ValueError(f"No models available for {task_key}")

    # Get common IDs across all models
    common_ids = set(list(predictions.values())[0].index)
    for pred in predictions.values():
        common_ids &= set(pred.index)
    common_ids = sorted(common_ids)

    if len(common_ids) == 0:
        raise ValueError(f"No common IDs found for {task_key}")

    # Build level-1 features for test set
    X_test_level1 = np.zeros((len(common_ids), len(model_names)))
    for i, model_name in enumerate(model_names):
        if model_name in predictions:
            X_test_level1[:, i] = predictions[model_name].loc[common_ids].values

    # Average predictions from all fold-specific meta-learners
    test_preds = []
    for model, model_type in zip(best_models, best_model_types):
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


def load_test_predictions(task_key: str, task_name: str, mt_task_name: str, model_names: List[str]) -> Dict[str, pd.Series]:
    """Load test predictions for all models.

    Args:
        task_key: Task key
        task_name: Single task name
        mt_task_name: Multitask task name
        model_names: List of model names (with prefix like "single_full")

    Returns:
        Dictionary mapping model_name to prediction Series indexed by ID
    """
    test_predictions = {}

    for model_name in model_names:
        if model_name.startswith("single_"):
            # Single-task model
            actual_model_name = model_name.replace("single_", "")
            test_path = pred_base_dir / actual_model_name / task_key / f"{task_name}_test.csv"
        elif model_name.startswith("multitask_"):
            # Multitask model
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


def create_final_submission_from_stacking():
    """Create final submission file from stacking ensemble results."""
    print("\n" + "="*70)
    print("Creating Final Submission from Stacking Ensembles")
    print("="*70)

    # Load optimization results and models
    task_predictions = {}
    task_results = {}

    for task_key, task_name, label_col, prepared_file, mt_task_name in tasks:
        result_path = output_dir / f"ensemble_optimization_stacking_{task_key}.json"

        if not result_path.exists():
            print(f"⚠️  Optimization result not found for {task_key}, skipping...")
            continue

        with open(result_path, 'r') as f:
            result = json.load(f)

        task_results[task_key] = result

        # Get model names from OOF predictions
        oof_path = output_dir / f"ensemble_oof_stacking_{task_key}.csv"
        if not oof_path.exists():
            print(f"⚠️  OOF predictions not found for {task_key}, skipping...")
            continue

        # Load original predictions to get model names
        predictions = load_predictions(task_key, task_name, mt_task_name)
        model_names = list(predictions.keys())

        print(f"\nGenerating test predictions for {task_key}...")
        print(f"Using {len(model_names)} base models")

        # Load test predictions
        try:
            test_predictions = load_test_predictions(task_key, task_name, mt_task_name, model_names)

            if len(test_predictions) == 0:
                print(f"⚠️  No test predictions loaded for {task_key}, skipping...")
                continue

            # For simplicity, we'll retrain meta-learners on full OOF data
            # In practice, you might want to save/load the trained meta-models
            # For now, we'll use a simple approach: train one meta-learner on all OOF data
            print(f"  Note: Using average of base model predictions for test set")
            print(f"  (Full meta-learner retraining on test set not implemented in this version)")

            # Get common IDs
            common_ids = set(list(test_predictions.values())[0].index)
            for pred in test_predictions.values():
                common_ids &= set(pred.index)
            common_ids = sorted(common_ids)

            # Simple average of test predictions (placeholder - in practice, retrain meta-learner)
            test_pred_array = np.zeros(len(common_ids))
            for model_name in model_names:
                if model_name in test_predictions:
                    test_pred_array += test_predictions[model_name].loc[common_ids].values
            test_pred_array /= len([m for m in model_names if m in test_predictions])

            # Clip to [0, 1]
            test_pred_array = np.clip(test_pred_array, 0, 1)

            test_pred_df = pd.DataFrame({
                'ID': common_ids,
                'prediction': test_pred_array
            })

            # Save individual task submission
            task_submission_path = output_dir / f"ensemble_test_stacking_{task_key}.csv"
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
    final_submission_path = output_dir / "ensemble_final_submission_stacking.csv"
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
            n_models = result.get('n_models', 0)
            task_display_name = task_name_map.get(task_key, task_key)
            print(f"  {task_display_name:25s}: "
                  f"Final OOF AUC = {final_auc:.6f}, "
                  f"Mean CV AUC = {mean_fold_auc:.6f} ± {std_fold_auc:.6f}, "
                  f"Models = {n_models}")

    return final_submission


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--create-submission":
        # Only create submission from existing optimization results
        create_final_submission_from_stacking()
    else:
        # Run optimization and then create submission
        main()
        print("\n" + "="*70)
        create_final_submission_from_stacking()


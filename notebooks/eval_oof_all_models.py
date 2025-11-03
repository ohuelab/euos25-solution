"""Evaluate OOF predictions for all models on all tasks."""

import pandas as pd
from sklearn.metrics import roc_auc_score
from pathlib import Path

# Configuration
names = [
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

# Task configuration: (task_key, task_name, label_col, prepared_file)
tasks = [
    ('fluo_340_450', 'y_fluo_any', 'Fluorescence', 'train_fluo_340_450_prepared.csv'),
    ('fluo_480', 'y_fluo_any', 'Fluorescence', 'train_fluo_480_prepared.csv'),
    ('trans_340', 'y_trans_any', 'Transmittance', 'train_trans_340_prepared.csv'),
    ('trans_450', 'y_trans_any', 'Transmittance', 'train_trans_450_prepared.csv'),
]

# Paths
pred_base_dir = Path("../data/preds")
processed_base_dir = Path("../data/processed")

# Try different feature directories (ecfp4, chemeleon, etc.)
feature_dirs = ['ecfp4', 'chemeleon', '']

# Evaluate all models and tasks
all_results = []

for task_key, task_name, label_col, prepared_file in tasks:
    print("\n" + "=" * 70)
    print(f"Evaluating task: {task_key} ({task_name})")
    print("=" * 70)

    # Try to find training data file
    df_train = None
    train_data_path = None

    for feature_dir in feature_dirs:
        if feature_dir:
            candidate_path = processed_base_dir / feature_dir / prepared_file
        else:
            candidate_path = processed_base_dir / prepared_file

        if candidate_path.exists():
            train_data_path = candidate_path
            df_train = pd.read_csv(train_data_path)
            print(f"Loading training data from {train_data_path}")
            print(f"Training data: {len(df_train)} samples")
            break

    if df_train is None:
        print(f"⚠️  Skipping {task_key}: Training data file not found")
        print(f"   Tried: {[str(processed_base_dir / (d + '/' if d else '') / prepared_file) for d in feature_dirs]}")
        continue

    # Evaluate each model for this task
    for name in names:
        # Construct OOF prediction file path
        oof_path = pred_base_dir / name / task_key / f"{task_name}_oof.csv"

        if not oof_path.exists():
            all_results.append({
                'model': name,
                'task': task_key,
                'cv_auc': None,
                'n_samples': None,
                'status': 'file_not_found'
            })
            continue

        try:
            # Load OOF predictions
            df_oof = pd.read_csv(oof_path, index_col=0)

            # Merge with training data
            merged = df_train.merge(
                df_oof.reset_index().rename(columns={'mol_id': 'ID'}),
                on='ID',
                how='inner'
            )

            if len(merged) == 0:
                all_results.append({
                    'model': name,
                    'task': task_key,
                    'cv_auc': None,
                    'n_samples': 0,
                    'status': 'no_matching_ids'
                })
                continue

            # Calculate metrics
            y_true = (merged[label_col] != 0).astype(int)
            y_pred = merged["prediction"]

            # Check if we have both classes
            if y_true.nunique() < 2:
                all_results.append({
                    'model': name,
                    'task': task_key,
                    'cv_auc': None,
                    'n_samples': len(merged),
                    'status': 'single_class'
                })
                continue

            cv_auc = roc_auc_score(y_true, y_pred)

            all_results.append({
                'model': name,
                'task': task_key,
                'cv_auc': cv_auc,
                'n_samples': len(merged),
                'status': 'success'
            })

            print(f"  ✅ {name}: CV ROC-AUC = {cv_auc:.6f} (n={len(merged)})")

        except Exception as e:
            print(f"  ❌ Error evaluating {name} on {task_key}: {e}")
            all_results.append({
                'model': name,
                'task': task_key,
                'cv_auc': None,
                'n_samples': None,
                'status': f'error: {str(e)}'
            })

# Create results DataFrame
results_df = pd.DataFrame(all_results)

# Pivot table for better visualization
if len(results_df) > 0:
    # Create pivot table with tasks as columns
    pivot_df = results_df.pivot_table(
        index='model',
        columns='task',
        values='cv_auc',
        aggfunc='first'
    )

    # Calculate average across tasks
    pivot_df['average'] = pivot_df.mean(axis=1)

    # Sort by average
    pivot_df = pivot_df.sort_values('average', ascending=False)

    print("\n" + "=" * 70)
    print("Summary Results by Task (sorted by average)")
    print("=" * 70)
    print(pivot_df.to_string())

    # Save detailed results
    detailed_output = Path("../data/preds/oof_eval_all_tasks.csv")
    results_df.to_csv(detailed_output, index=False)
    print(f"\nDetailed results saved to {detailed_output}")

    # Save pivot table
    pivot_output = Path("../data/preds/oof_eval_all_tasks_pivot.csv")
    pivot_df.to_csv(pivot_output)
    print(f"Pivot table saved to {pivot_output}")

    # Print summary statistics
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    print(f"Total evaluations: {len(results_df)}")
    print(f"Successful: {len(results_df[results_df['status'] == 'success'])}")
    print(f"Failed/Missing: {len(results_df[results_df['status'] != 'success'])}")

    # Best model per task
    print("\n" + "=" * 70)
    print("Best Model per Task")
    print("=" * 70)
    for task_key, task_name, label_col, prepared_file in tasks:
        task_results = results_df[
            (results_df['task'] == task_key) &
            (results_df['status'] == 'success')
        ]
        if len(task_results) > 0:
            best = task_results.loc[task_results['cv_auc'].idxmax()]
            print(f"{task_key:15s}: {best['model']:25s} (AUC = {best['cv_auc']:.6f})")
else:
    print("\n⚠️  No results to display")


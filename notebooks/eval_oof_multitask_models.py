"""Evaluate OOF predictions for all multitask models on all tasks."""

import pandas as pd
from sklearn.metrics import roc_auc_score
from pathlib import Path

# Configuration - Multitask model names
multitask_names = [
    'multitask_all_chemeleon',
    'multitask_all_chemprop',
    'multitask_fluo_chemeleon',
    'multitask_fluo_chemprop',
    'multitask_trans_chemeleon',
    'multitask_trans_chemprop',
]

# Task mapping: (task_key, multitask_task_name, label_col, prepared_file)
# multitask_task_name is the name in the multitask prediction files
task_mapping = {
    'fluo_340_450': ('fluorescence340_450', 'Fluorescence', 'train_fluo_340_450_prepared.csv'),
    'fluo_480': ('fluorescence480', 'Fluorescence', 'train_fluo_480_prepared.csv'),
    'trans_340': ('transmittance340', 'Transmittance', 'train_trans_340_prepared.csv'),
    'trans_450': ('transmittance450', 'Transmittance', 'train_trans_450_prepared.csv'),
}

# Paths
pred_base_dir = Path("../data/preds")
processed_base_dir = Path("../data/processed")

# Try different feature directories (ecfp4, chemeleon, etc.)
feature_dirs = ['ecfp4', 'chemeleon', '']

# Evaluate all models and tasks
all_results = []

for model_name in multitask_names:
    print("\n" + "=" * 70)
    print(f"Evaluating multitask model: {model_name}")
    print("=" * 70)

    # Construct OOF directory path
    oof_dir = pred_base_dir / model_name / "oof"

    if not oof_dir.exists():
        print(f"⚠️  OOF directory not found: {oof_dir}")
        # Add missing entries for all tasks
        for task_key, (mt_task_name, label_col, prepared_file) in task_mapping.items():
            all_results.append({
                'model': model_name,
                'task': task_key,
                'cv_auc': None,
                'n_samples': None,
                'status': 'oof_dir_not_found'
            })
        continue

    # Find available prediction files
    available_files = list(oof_dir.glob("*_oof.csv"))
    available_task_names = [f.stem.replace('_oof', '') for f in available_files]
    print(f"Available tasks: {available_task_names}")

    # Evaluate each task
    for task_key, (mt_task_name, label_col, prepared_file) in task_mapping.items():
        oof_file_name = f"{mt_task_name}_oof.csv"
        oof_path = oof_dir / oof_file_name

        if not oof_path.exists():
            print(f"  ⚠️  Skipping {task_key}: OOF file not found ({oof_file_name})")
            all_results.append({
                'model': model_name,
                'task': task_key,
                'cv_auc': None,
                'n_samples': None,
                'status': 'file_not_found'
            })
            continue

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
                break

        if df_train is None:
            print(f"  ⚠️  Skipping {task_key}: Training data not found")
            all_results.append({
                'model': model_name,
                'task': task_key,
                'cv_auc': None,
                'n_samples': None,
                'status': 'train_data_not_found'
            })
            continue

        try:
            # Load OOF predictions (multitask files use ID column, not mol_id)
            df_oof = pd.read_csv(oof_path)

            # Check column names
            if 'ID' not in df_oof.columns:
                # Try mol_id if ID doesn't exist
                if 'mol_id' in df_oof.columns:
                    df_oof = df_oof.rename(columns={'mol_id': 'ID'})
                else:
                    raise ValueError(f"Neither 'ID' nor 'mol_id' found in {oof_path}")

            # Merge with training data
            merged = df_train.merge(
                df_oof,
                on='ID',
                how='inner'
            )

            if len(merged) == 0:
                print(f"  ⚠️  Skipping {task_key}: No matching IDs found")
                all_results.append({
                    'model': model_name,
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
                print(f"  ⚠️  Skipping {task_key}: Only one class present")
                all_results.append({
                    'model': model_name,
                    'task': task_key,
                    'cv_auc': None,
                    'n_samples': len(merged),
                    'status': 'single_class'
                })
                continue

            cv_auc = roc_auc_score(y_true, y_pred)

            all_results.append({
                'model': model_name,
                'task': task_key,
                'cv_auc': cv_auc,
                'n_samples': len(merged),
                'status': 'success'
            })

            print(f"  ✅ {task_key}: CV ROC-AUC = {cv_auc:.6f} (n={len(merged)})")

        except Exception as e:
            print(f"  ❌ Error evaluating {task_key}: {e}")
            all_results.append({
                'model': model_name,
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

    # Calculate average across available tasks
    pivot_df['average'] = pivot_df.mean(axis=1)

    # Sort by average
    pivot_df = pivot_df.sort_values('average', ascending=False)

    print("\n" + "=" * 70)
    print("Summary Results by Task (sorted by average)")
    print("=" * 70)
    print(pivot_df.to_string())

    # Save detailed results
    detailed_output = Path("../data/preds/oof_eval_multitask_all_tasks.csv")
    results_df.to_csv(detailed_output, index=False)
    print(f"\nDetailed results saved to {detailed_output}")

    # Save pivot table
    pivot_output = Path("../data/preds/oof_eval_multitask_all_tasks_pivot.csv")
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
    print("Best Multitask Model per Task")
    print("=" * 70)
    for task_key, (mt_task_name, label_col, prepared_file) in task_mapping.items():
        task_results = results_df[
            (results_df['task'] == task_key) &
            (results_df['status'] == 'success')
        ]
        if len(task_results) > 0:
            best = task_results.loc[task_results['cv_auc'].idxmax()]
            print(f"{task_key:15s}: {best['model']:30s} (AUC = {best['cv_auc']:.6f})")

    # Model coverage (which tasks each model covers)
    print("\n" + "=" * 70)
    print("Model Task Coverage")
    print("=" * 70)
    for model_name in multitask_names:
        model_results = results_df[
            (results_df['model'] == model_name) &
            (results_df['status'] == 'success')
        ]
        if len(model_results) > 0:
            covered_tasks = sorted(model_results['task'].unique())
            avg_auc = model_results['cv_auc'].mean()
            print(f"{model_name:30s}: {', '.join(covered_tasks):40s} (avg AUC = {avg_auc:.6f})")
        else:
            print(f"{model_name:30s}: (no successful evaluations)")
else:
    print("\n⚠️  No results to display")


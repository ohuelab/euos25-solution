#!/usr/bin/env python3
"""Check cross-validation scores from trained models."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    """Main function to check validation scores."""
    model_dir = Path("data/models")
    tasks = ["fluo_340_450", "fluo_480", "trans_340", "trans_450"]
    task_names = {
        "fluo_340_450": "y_fluo_any",
        "fluo_480": "y_fluo_any",
        "trans_340": "y_trans_any",
        "trans_450": "y_trans_any",
    }

    all_metrics = []

    for task in tasks:
        task_name = task_names[task]
        metrics_path = model_dir / task / task_name / "lgbm" / "cv_metrics.csv"

        if not metrics_path.exists():
            print(f"⚠️  {task} ({task_name}): Metrics file not found")
            print(f"   Expected: {metrics_path}")
            print("")
            continue

        try:
            df = pd.read_csv(metrics_path)

            if df.empty:
                print(f"⚠️  {task} ({task_name}): Metrics file is empty")
                print("")
                continue

            # Get metric columns (excluding 'fold')
            metric_cols = [col for col in df.columns if col != "fold"]

            if not metric_cols:
                print(f"⚠️  {task} ({task_name}): No metrics found in file")
                print("")
                continue

            print(f"📊 {task} ({task_name})")
            print("-" * 60)

            # Display per-fold scores
            print("Fold scores:")
            for _, row in df.iterrows():
                fold = row["fold"]
                metric_strs = [f"{col}={row[col]:.6f}" for col in metric_cols]
                print(f"  Fold {fold}: {', '.join(metric_strs)}")

            # Calculate and display aggregated metrics
            print("\nAggregated metrics:")
            for col in metric_cols:
                values = df[col].values
                mean_val = np.mean(values)
                std_val = np.std(values)
                min_val = np.min(values)
                max_val = np.max(values)
                print(f"  {col}:")
                print(f"    Mean ± Std: {mean_val:.6f} ± {std_val:.6f}")
                print(f"    Range: [{min_val:.6f}, {max_val:.6f}]")

                # Store for summary
                all_metrics.append(
                    {
                        "task": task,
                        "task_name": task_name,
                        "metric": col,
                        "mean": mean_val,
                        "std": std_val,
                    }
                )

            print("")

        except Exception as e:
            print(f"❌ {task} ({task_name}): Error reading metrics")
            print(f"   Error: {e}")
            print("")

    # Summary across all tasks
    if all_metrics:
        print("=" * 60)
        print("Summary Across All Tasks")
        print("=" * 60)

        summary_df = pd.DataFrame(all_metrics)

        for metric in summary_df["metric"].unique():
            metric_data = summary_df[summary_df["metric"] == metric]
            print(f"\n{metric}:")
            for _, row in metric_data.iterrows():
                print(
                    f"  {row['task']:20s} ({row['task_name']:12s}): "
                    f"{row['mean']:.6f} ± {row['std']:.6f}"
                )


if __name__ == "__main__":
    main()


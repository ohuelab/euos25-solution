"""Explore which descriptors benefit from label/target encoding.

This script evaluates the effectiveness of label encoding and target encoding
for categorical descriptors across different tasks.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from euos25.featurizers.categorical_encoding import (
    LabelEncodingFeaturizer,
    TargetEncodingFeaturizer,
    detect_categorical_descriptors,
)
from euos25.models.lgbm import LGBMClassifier
from euos25.utils.io import load_csv, load_json, load_parquet, save_csv
from euos25.utils.metrics import calc_roc_auc

logger = logging.getLogger(__name__)


def create_inner_splits(
    train_indices: np.ndarray,
    y_train: np.ndarray,
    n_splits: int = 3,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create inner CV splits from training indices.

    Args:
        train_indices: Training indices
        y_train: Training labels
        n_splits: Number of inner CV folds
        seed: Random seed

    Returns:
        List of (inner_train_indices, inner_valid_indices) tuples
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    inner_splits = []
    for inner_train_idx, inner_valid_idx in skf.split(train_indices, y_train):
        inner_splits.append(
            (train_indices[inner_train_idx], train_indices[inner_valid_idx])
        )
    return inner_splits


def evaluate_encoding_for_descriptor(
    descriptor_col: str,
    features: pd.DataFrame,
    labels: pd.Series,
    splits: Dict,
    use_label_encoding: bool = True,
    use_target_encoding: bool = True,
    inner_cv_folds: int = 3,
    seed: int = 42,
) -> Dict[str, float]:
    """Evaluate encoding effectiveness for a single descriptor.

    Args:
        descriptor_col: Descriptor column name
        features: Features DataFrame
        labels: Labels Series
        splits: CV splits dictionary
        use_label_encoding: Whether to evaluate label encoding
        use_target_encoding: Whether to evaluate target encoding
        inner_cv_folds: Number of inner CV folds
        seed: Random seed

    Returns:
        Dictionary with evaluation scores
    """
    results = {}

    # Check if descriptor column exists
    if descriptor_col not in features.columns:
        return {"label_encoding_score": np.nan, "target_encoding_score": np.nan}

    # Get outer CV scores
    outer_scores_label = []
    outer_scores_target = []
    outer_scores_baseline = []

    for fold_name, fold_data in splits.items():
        train_pos_indices = fold_data["train"]
        valid_pos_indices = fold_data["valid"]

        train_ids = features.index[train_pos_indices]
        valid_ids = features.index[valid_pos_indices]

        X_train = features.loc[train_ids]
        y_train = labels.loc[train_ids]
        X_valid = features.loc[valid_ids]
        y_valid = labels.loc[valid_ids]

        # Baseline: use original descriptor
        X_train_baseline = X_train[[descriptor_col]].copy()
        X_valid_baseline = X_valid[[descriptor_col]].copy()

        # Fill NaN with median
        median_val = X_train_baseline[descriptor_col].median()
        X_train_baseline[descriptor_col] = X_train_baseline[descriptor_col].fillna(
            median_val
        )
        X_valid_baseline[descriptor_col] = X_valid_baseline[descriptor_col].fillna(
            median_val
        )

        # Train baseline model
        try:
            baseline_model = LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                num_leaves=7,
                verbose=-1,
            )
            baseline_model.fit(
                X_train_baseline, y_train.values, eval_set=[(X_valid_baseline, y_valid.values)]
            )
            baseline_pred = baseline_model.predict_proba(X_valid_baseline)
            baseline_score = calc_roc_auc(y_valid.values, baseline_pred)
            outer_scores_baseline.append(baseline_score)
        except Exception as e:
            logger.debug(f"Baseline model failed for {descriptor_col}: {e}")
            outer_scores_baseline.append(np.nan)

        # Label encoding
        if use_label_encoding:
            try:
                label_encoder = LabelEncodingFeaturizer(
                    descriptor_columns=[descriptor_col], auto_detect=False
                )
                label_encoder.fit(X_train)
                X_train_label = label_encoder.transform(X_train)
                X_valid_label = label_encoder.transform(X_valid)

                if len(X_train_label.columns) > 0:
                    label_model = LGBMClassifier(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=3,
                        num_leaves=7,
                        verbose=-1,
                    )
                    label_model.fit(
                        X_train_label, y_train.values, eval_set=[(X_valid_label, y_valid.values)]
                    )
                    label_pred = label_model.predict_proba(X_valid_label)
                    label_score = calc_roc_auc(y_valid.values, label_pred)
                    outer_scores_label.append(label_score)
                else:
                    outer_scores_label.append(np.nan)
            except Exception as e:
                logger.debug(f"Label encoding failed for {descriptor_col}: {e}")
                outer_scores_label.append(np.nan)

        # Target encoding with nested CV
        if use_target_encoding:
            try:
                # Create inner splits
                train_indices_array = np.array(train_pos_indices)
                y_train_array = y_train.values
                inner_splits = create_inner_splits(
                    train_indices_array, y_train_array, n_splits=inner_cv_folds, seed=seed
                )

                # Target encoding with nested CV
                inner_scores = []
                for inner_train_idx, inner_valid_idx in inner_splits:
                    inner_train_ids = features.index[inner_train_idx]
                    inner_valid_ids = features.index[inner_valid_idx]

                    X_inner_train = features.loc[inner_train_ids]
                    y_inner_train = labels.loc[inner_train_ids]
                    X_inner_valid = features.loc[inner_valid_ids]
                    y_inner_valid = labels.loc[inner_valid_ids]

                    # Fit target encoder on inner train
                    target_encoder = TargetEncodingFeaturizer(
                        descriptor_columns=[descriptor_col], auto_detect=False
                    )
                    target_encoder.fit(X_inner_train, y_inner_train)

                    # Transform inner valid (using inner train encoder)
                    X_inner_train_target = target_encoder.transform(X_inner_train)
                    X_inner_valid_target = target_encoder.transform(X_inner_valid)

                    if len(X_inner_train_target.columns) > 0:
                        inner_model = LGBMClassifier(
                            n_estimators=100,
                            learning_rate=0.1,
                            max_depth=3,
                            num_leaves=7,
                            verbose=-1,
                        )
                        inner_model.fit(
                            X_inner_train_target,
                            y_inner_train.values,
                            eval_set=[(X_inner_valid_target, y_inner_valid.values)],
                        )
                        inner_pred = inner_model.predict_proba(X_inner_valid_target)
                        inner_score = calc_roc_auc(y_inner_valid.values, inner_pred)
                        inner_scores.append(inner_score)

                if len(inner_scores) > 0:
                    # Use average inner CV score
                    outer_scores_target.append(np.mean(inner_scores))
                else:
                    outer_scores_target.append(np.nan)
            except Exception as e:
                logger.debug(f"Target encoding failed for {descriptor_col}: {e}")
                outer_scores_target.append(np.nan)

    # Calculate average scores
    results["baseline_score"] = np.nanmean(outer_scores_baseline) if outer_scores_baseline else np.nan
    if use_label_encoding:
        results["label_encoding_score"] = (
            np.nanmean(outer_scores_label) if outer_scores_label else np.nan
        )
        results["label_encoding_improvement"] = (
            results["label_encoding_score"] - results["baseline_score"]
            if not np.isnan(results["baseline_score"])
            else np.nan
        )
    else:
        results["label_encoding_score"] = np.nan
        results["label_encoding_improvement"] = np.nan

    if use_target_encoding:
        results["target_encoding_score"] = (
            np.nanmean(outer_scores_target) if outer_scores_target else np.nan
        )
        results["target_encoding_improvement"] = (
            results["target_encoding_score"] - results["baseline_score"]
            if not np.isnan(results["baseline_score"])
            else np.nan
        )
    else:
        results["target_encoding_score"] = np.nan
        results["target_encoding_improvement"] = np.nan

    return results


def explore_descriptor_encoding(
    features_path: str,
    splits_path: str,
    data_path: str,
    label_col: str,
    task_name: str,
    output_path: str,
    max_unique_values: int = 100,
    use_label_encoding: bool = True,
    use_target_encoding: bool = True,
    inner_cv_folds: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Explore descriptor encoding effectiveness.

    Args:
        features_path: Path to features Parquet file
        splits_path: Path to splits JSON file
        data_path: Path to data CSV file
        label_col: Label column name
        task_name: Task name
        output_path: Path to save results CSV
        max_unique_values: Maximum unique values to consider as categorical
        use_label_encoding: Whether to evaluate label encoding
        use_target_encoding: Whether to evaluate target encoding
        inner_cv_folds: Number of inner CV folds
        seed: Random seed

    Returns:
        DataFrame with results
    """
    logger.info(f"Exploring descriptor encoding for task: {task_name}")

    # Load data
    logger.info("Loading data...")
    features = load_parquet(features_path)
    splits = load_json(splits_path)
    df = load_csv(data_path)

    # Prepare labels
    df = df.set_index("ID")
    labels = df[label_col]
    if labels.dtype != int:
        # Convert to binary if needed
        labels = (labels != 0).astype(int)

    # Detect categorical descriptors
    logger.info("Detecting categorical descriptors...")
    categorical_descriptors = detect_categorical_descriptors(
        features, max_unique_values=max_unique_values
    )
    logger.info(f"Found {len(categorical_descriptors)} categorical descriptors")

    if len(categorical_descriptors) == 0:
        logger.warning("No categorical descriptors found")
        return pd.DataFrame()

    # Evaluate each descriptor
    results = []
    for desc_col in tqdm(categorical_descriptors, desc="Evaluating descriptors"):
        try:
            desc_results = evaluate_encoding_for_descriptor(
                desc_col,
                features,
                labels,
                splits,
                use_label_encoding=use_label_encoding,
                use_target_encoding=use_target_encoding,
                inner_cv_folds=inner_cv_folds,
                seed=seed,
            )
            desc_results["descriptor"] = desc_col
            desc_results["unique_values"] = features[desc_col].nunique()
            results.append(desc_results)
        except Exception as e:
            logger.warning(f"Failed to evaluate {desc_col}: {e}")
            continue

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Calculate importance scores (weighted by improvement)
    if use_label_encoding:
        results_df["label_encoding_importance"] = (
            results_df["label_encoding_improvement"] * results_df["label_encoding_score"]
        )
    if use_target_encoding:
        results_df["target_encoding_importance"] = (
            results_df["target_encoding_improvement"] * results_df["target_encoding_score"]
        )

    # Sort by importance
    if use_target_encoding:
        results_df = results_df.sort_values(
            "target_encoding_importance", ascending=False
        )
    elif use_label_encoding:
        results_df = results_df.sort_values(
            "label_encoding_importance", ascending=False
        )

    # Save results
    save_csv(results_df, output_path)
    logger.info(f"Saved results to {output_path}")

    return results_df


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Explore descriptor encoding effectiveness"
    )
    parser.add_argument("--features", required=True, help="Features Parquet file")
    parser.add_argument("--splits", required=True, help="Splits JSON file")
    parser.add_argument("--data", required=True, help="Data CSV file")
    parser.add_argument("--label-col", required=True, help="Label column name")
    parser.add_argument("--task", required=True, help="Task name")
    parser.add_argument("--output", required=True, help="Output CSV file")
    parser.add_argument(
        "--max-unique-values", type=int, default=100, help="Max unique values for categorical"
    )
    parser.add_argument(
        "--use-label-encoding", action="store_true", help="Evaluate label encoding"
    )
    parser.add_argument(
        "--use-target-encoding", action="store_true", help="Evaluate target encoding"
    )
    parser.add_argument(
        "--inner-cv-folds", type=int, default=3, help="Number of inner CV folds"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Default to both encodings if neither specified
    use_label = args.use_label_encoding
    use_target = args.use_target_encoding
    if not use_label and not use_target:
        use_label = True
        use_target = True

    # Run exploration
    results_df = explore_descriptor_encoding(
        features_path=args.features,
        splits_path=args.splits,
        data_path=args.data,
        label_col=args.label_col,
        task_name=args.task,
        output_path=args.output,
        max_unique_values=args.max_unique_values,
        use_label_encoding=use_label,
        use_target_encoding=use_target,
        inner_cv_folds=args.inner_cv_folds,
        seed=args.seed,
    )

    # Print top results
    print("\n" + "=" * 80)
    print("Top descriptors by encoding effectiveness:")
    print("=" * 80)
    print(results_df.head(20).to_string())


if __name__ == "__main__":
    main()


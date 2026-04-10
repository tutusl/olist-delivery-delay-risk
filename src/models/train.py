from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, balanced_accuracy_score, fbeta_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from src.data.io import load_olist_tables
from src.data.prepare import build_modeling_frame
from src.features.engineering import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_model_dataset,
    split_features_target,
    validate_no_leakage,
)

RANDOM_STATE = 42


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_lgbm_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_models() -> dict[str, Pipeline]:
    pos_weight = 10  # approximate (1 - 0.08) / 0.08 ≈ 11, rounded down
    return {
        "dummy_baseline": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", DummyClassifier(strategy="prior")),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=5,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "lightgbm": Pipeline(
            steps=[
                ("preprocessor", build_lgbm_preprocessor()),
                (
                    "model",
                    lgb.LGBMClassifier(
                        n_estimators=500,
                        learning_rate=0.05,
                        num_leaves=31,
                        min_child_samples=20,
                        scale_pos_weight=pos_weight,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                        verbose=-1,
                    ),
                ),
            ]
        ),
    }


def precision_at_k(y_true: pd.Series, probabilities: np.ndarray, k: int) -> float:
    """Precision among the top-k highest-risk predictions."""
    top_k_idx = probabilities.argsort()[-k:]
    return y_true.iloc[top_k_idx].mean()


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, k: int = 500) -> dict[str, float]:
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = model.predict(X_test)
    return {
        "average_precision": average_precision_score(y_test, probabilities),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "f2_score": fbeta_score(y_test, predictions, beta=2),
        f"precision_at_{k}": precision_at_k(y_test, probabilities, k),
    }


def chronological_split(
    dataset: pd.DataFrame,
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset by purchase timestamp: earlier orders train, later orders test."""
    sorted_dataset = dataset.sort_values("order_purchase_timestamp").reset_index(drop=True)
    split_index = int(len(sorted_dataset) * (1 - test_fraction))
    return sorted_dataset.iloc[:split_index].copy(), sorted_dataset.iloc[split_index:].copy()


def compute_feature_importance(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Compute permutation importance on the test set."""
    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="average_precision",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    importance_frame = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return importance_frame


def run_training(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables = load_olist_tables(data_dir)
    modeling_frame = build_modeling_frame(tables)
    dataset = build_model_dataset(modeling_frame)

    train_set, test_set = chronological_split(dataset)
    X_train, y_train = split_features_target(train_set)
    X_test, y_test = split_features_target(test_set)
    validate_no_leakage(X_train.columns)

    results: list[dict[str, float | str]] = []
    best_model = None
    best_ap = -1.0

    for model_name, model in build_models().items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results.append({"model": model_name, **metrics})
        if metrics["average_precision"] > best_ap:
            best_ap = metrics["average_precision"]
            best_model = (model_name, model)

    result_frame = pd.DataFrame(results).sort_values("average_precision", ascending=False).reset_index(drop=True)

    importance_frame = pd.DataFrame()
    if best_model is not None:
        print(f"\nComputing permutation importance for {best_model[0]}...")
        importance_frame = compute_feature_importance(best_model[1], X_test, y_test)

    return result_frame, importance_frame


def save_training_summary(results: pd.DataFrame, importance: pd.DataFrame) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "reports" / "training_summary.csv"
    results.to_csv(output_path, index=False)
    if not importance.empty:
        importance_path = repo_root / "reports" / "feature_importance.csv"
        importance.to_csv(importance_path, index=False)
        print(f"Saved feature importance to: {importance_path}")
    return output_path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Train starter models for the Olist delivery delay project.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root / "data" / "raw",
        help="Directory containing the raw Olist CSV files.",
    )
    args = parser.parse_args()

    results, importance = run_training(args.data_dir)
    output_path = save_training_summary(results, importance)
    print(results.round(4).to_string(index=False))
    if not importance.empty:
        print(f"\nTop 10 features by permutation importance:")
        print(importance.head(10).round(4).to_string(index=False))
    print(f"\nSaved training summary to: {output_path}")


if __name__ == "__main__":
    main()

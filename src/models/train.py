from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, balanced_accuracy_score, fbeta_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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


def build_models() -> dict[str, Pipeline]:
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


def run_training(data_dir: str | Path) -> pd.DataFrame:
    tables = load_olist_tables(data_dir)
    modeling_frame = build_modeling_frame(tables)
    dataset = build_model_dataset(modeling_frame)
    X, y = split_features_target(dataset)
    validate_no_leakage(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    results: list[dict[str, float | str]] = []
    for model_name, model in build_models().items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results.append({"model": model_name, **metrics})

    result_frame = pd.DataFrame(results).sort_values("average_precision", ascending=False).reset_index(drop=True)
    return result_frame


def save_training_summary(results: pd.DataFrame) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "reports" / "training_summary.csv"
    results.to_csv(output_path, index=False)
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

    results = run_training(args.data_dir)
    output_path = save_training_summary(results)
    print(results.round(4).to_string(index=False))
    print(f"\nSaved training summary to: {output_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, fbeta_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
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
TUNING_SPLITS = 4
TUNING_ITERATIONS = 12


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


def build_ordinal_preprocessor() -> ColumnTransformer:
    """Preprocessor with ordinal encoding — suited for tree-based boosters."""
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
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("preprocessor", build_ordinal_preprocessor()),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=500,
                        learning_rate=0.05,
                        max_leaf_nodes=31,
                        min_samples_leaf=20,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def build_tuning_param_spaces() -> dict[str, dict[str, list[object]]]:
    return {
        "random_forest": {
            "model__n_estimators": [300, 500, 800],
            "model__max_depth": [None, 8, 12, 20],
            "model__min_samples_leaf": [2, 5, 10, 20],
            "model__max_features": ["sqrt", 0.5, None],
        },
        "hist_gradient_boosting": {
            "model__max_iter": [200, 400, 600],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_leaf_nodes": [15, 31, 63],
            "model__min_samples_leaf": [10, 20, 50],
            "model__l2_regularization": [0.0, 0.1, 1.0],
        },
    }


def build_time_series_cv(n_splits: int = TUNING_SPLITS) -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=n_splits)


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


def summarize_tuning_results(model_name: str, search: RandomizedSearchCV) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "model": model_name,
            "cv_rank": search.cv_results_["rank_test_score"],
            "cv_mean_average_precision": search.cv_results_["mean_test_score"],
            "cv_std_average_precision": search.cv_results_["std_test_score"],
            "cv_mean_train_average_precision": search.cv_results_["mean_train_score"],
            "cv_std_train_average_precision": search.cv_results_["std_train_score"],
            "params": search.cv_results_["params"],
        }
    )
    summary["selected"] = summary["cv_rank"].eq(1)
    summary["params"] = summary["params"].apply(lambda params: json.dumps(params, sort_keys=True))
    return summary.sort_values(["cv_rank", "cv_mean_average_precision"], ascending=[True, False]).reset_index(drop=True)


def tune_model(
    model_name: str,
    model: Pipeline,
    param_space: dict[str, list[object]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[Pipeline, pd.DataFrame]:
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_space,
        n_iter=TUNING_ITERATIONS,
        scoring="average_precision",
        cv=build_time_series_cv(),
        n_jobs=-1,
        refit=True,
        random_state=RANDOM_STATE,
        return_train_score=True,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, summarize_tuning_results(model_name, search)


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


def run_error_analysis(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Break test-set performance by delivery-window bucket, customer state, and payment type."""
    probabilities = model.predict_proba(X_test)[:, 1]

    analysis = X_test.copy()
    analysis["y_true"] = y_test.values
    analysis["y_prob"] = probabilities

    delivery_bins = [0, 10, 20, 30, float("inf")]
    delivery_labels = ["0-10d", "10-20d", "20-30d", "30d+"]
    analysis["delivery_window"] = pd.cut(
        analysis["estimated_delivery_days"],
        bins=delivery_bins,
        labels=delivery_labels,
        right=False,
    )

    slices: list[dict] = []

    for label in delivery_labels:
        mask = analysis["delivery_window"] == label
        _append_slice(slices, f"delivery_window={label}", analysis[mask])

    top_states = analysis["customer_state"].value_counts().head(5).index
    for state in top_states:
        mask = analysis["customer_state"] == state
        _append_slice(slices, f"customer_state={state}", analysis[mask])

    for ptype in analysis["payment_type_mode"].dropna().unique():
        mask = analysis["payment_type_mode"] == ptype
        _append_slice(slices, f"payment_type={ptype}", analysis[mask])

    return pd.DataFrame(slices).sort_values("average_precision", ascending=False).reset_index(drop=True)


def _append_slice(
    slices: list[dict],
    name: str,
    subset: pd.DataFrame,
) -> None:
    n = len(subset)
    if n < 30:
        return
    late_rate = subset["y_true"].mean()
    ap = average_precision_score(subset["y_true"], subset["y_prob"]) if subset["y_true"].nunique() > 1 else float("nan")
    slices.append({"slice": name, "n": n, "late_rate": round(late_rate, 4), "average_precision": round(ap, 4)})


def build_calibration_summary(
    model_name: str,
    fitted_model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    k: int = 500,
) -> pd.DataFrame:
    calibrated_model = CalibratedClassifierCV(
        estimator=fitted_model,
        method="sigmoid",
        cv=build_time_series_cv(),
    )
    calibrated_model.fit(X_train, y_train)

    rows = []
    for variant_name, estimator in (
        ("uncalibrated", fitted_model),
        ("calibrated_sigmoid", calibrated_model),
    ):
        probabilities = estimator.predict_proba(X_test)[:, 1]
        predictions = estimator.predict(X_test)
        rows.append(
            {
                "model": model_name,
                "variant": variant_name,
                "average_precision": average_precision_score(y_test, probabilities),
                "roc_auc": roc_auc_score(y_test, probabilities),
                "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
                "f2_score": fbeta_score(y_test, predictions, beta=2),
                f"precision_at_{k}": precision_at_k(y_test, probabilities, k),
                "brier_score": brier_score_loss(y_test, probabilities),
            }
        )

    return pd.DataFrame(rows)


def run_training(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tables = load_olist_tables(data_dir)
    modeling_frame = build_modeling_frame(tables)
    dataset = build_model_dataset(modeling_frame)

    train_set, test_set = chronological_split(dataset)
    X_train, y_train = split_features_target(train_set)
    X_test, y_test = split_features_target(test_set)
    validate_no_leakage(X_train.columns)

    results: list[dict[str, float | str]] = []
    tuning_frames: list[pd.DataFrame] = []
    best_model = None
    best_ap = -1.0
    tuning_param_spaces = build_tuning_param_spaces()

    for model_name, model in build_models().items():
        fitted_model = model
        if model_name in tuning_param_spaces:
            print(f"\nRunning walk-forward tuning for {model_name}...")
            fitted_model, tuning_frame = tune_model(
                model_name=model_name,
                model=model,
                param_space=tuning_param_spaces[model_name],
                X_train=X_train,
                y_train=y_train,
            )
            tuning_frames.append(tuning_frame)
        else:
            fitted_model.fit(X_train, y_train)

        metrics = evaluate_model(fitted_model, X_test, y_test)
        results.append({"model": model_name, **metrics})
        if metrics["average_precision"] > best_ap:
            best_ap = metrics["average_precision"]
            best_model = (model_name, fitted_model)

    result_frame = pd.DataFrame(results).sort_values("average_precision", ascending=False).reset_index(drop=True)
    tuning_summary = pd.concat(tuning_frames, ignore_index=True) if tuning_frames else pd.DataFrame()

    importance_frame = pd.DataFrame()
    error_frame = pd.DataFrame()
    calibration_frame = pd.DataFrame()
    if best_model is not None:
        print(f"\nComputing permutation importance for {best_model[0]}...")
        importance_frame = compute_feature_importance(best_model[1], X_test, y_test)
        print(f"Running error analysis for {best_model[0]}...")
        error_frame = run_error_analysis(best_model[1], X_test, y_test)
        if best_model[0] in tuning_param_spaces:
            print(f"Comparing calibrated probabilities for {best_model[0]}...")
            calibration_frame = build_calibration_summary(
                model_name=best_model[0],
                fitted_model=best_model[1],
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )

    return result_frame, importance_frame, error_frame, tuning_summary, calibration_frame


def save_training_summary(
    results: pd.DataFrame,
    importance: pd.DataFrame,
    errors: pd.DataFrame,
    tuning: pd.DataFrame,
    calibration: pd.DataFrame,
) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "reports" / "training_summary.csv"
    results.to_csv(output_path, index=False)
    if not importance.empty:
        importance_path = repo_root / "reports" / "feature_importance.csv"
        importance.to_csv(importance_path, index=False)
        print(f"Saved feature importance to: {importance_path}")
    if not errors.empty:
        errors_path = repo_root / "reports" / "error_analysis.csv"
        errors.to_csv(errors_path, index=False)
        print(f"Saved error analysis to: {errors_path}")
    if not tuning.empty:
        tuning_path = repo_root / "reports" / "tuning_summary.csv"
        tuning.to_csv(tuning_path, index=False)
        print(f"Saved tuning summary to: {tuning_path}")
    if not calibration.empty:
        calibration_path = repo_root / "reports" / "calibration_summary.csv"
        calibration.to_csv(calibration_path, index=False)
        print(f"Saved calibration summary to: {calibration_path}")
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

    results, importance, errors, tuning, calibration = run_training(args.data_dir)
    output_path = save_training_summary(results, importance, errors, tuning, calibration)
    print(results.round(4).to_string(index=False))
    if not importance.empty:
        print(f"\nTop 10 features by permutation importance:")
        print(importance.head(10).round(4).to_string(index=False))
    if not errors.empty:
        print(f"\nError analysis by slice:")
        print(errors.to_string(index=False))
    if not tuning.empty:
        print(f"\nBest walk-forward tuning candidates:")
        selected = tuning.loc[tuning["selected"]].copy()
        print(selected.round(4).to_string(index=False))
    if not calibration.empty:
        print(f"\nCalibration comparison:")
        print(calibration.round(4).to_string(index=False))
    print(f"\nSaved training summary to: {output_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from types import SimpleNamespace

from src.data.prepare import build_modeling_frame
from src.features.engineering import build_model_dataset
from src.models.train import build_tuning_param_spaces, chronological_split, summarize_tuning_results


def test_chronological_split_keeps_purchase_order(synthetic_olist_tables):
    modeling_frame = build_modeling_frame(synthetic_olist_tables)
    dataset = build_model_dataset(modeling_frame)

    train_set, test_set = chronological_split(dataset, test_fraction=1 / 3)

    assert len(train_set) == 2
    assert len(test_set) == 1
    assert train_set["order_purchase_timestamp"].max() <= test_set["order_purchase_timestamp"].min()


def test_tuning_param_spaces_cover_both_tunable_models():
    param_spaces = build_tuning_param_spaces()

    assert {"random_forest", "hist_gradient_boosting"} == set(param_spaces)
    assert "model__n_estimators" in param_spaces["random_forest"]
    assert "model__learning_rate" in param_spaces["hist_gradient_boosting"]


def test_summarize_tuning_results_marks_selected_configuration():
    search = SimpleNamespace(
        cv_results_={
            "rank_test_score": [2, 1],
            "mean_test_score": [0.10, 0.20],
            "std_test_score": [0.01, 0.02],
            "mean_train_score": [0.11, 0.22],
            "std_train_score": [0.02, 0.03],
            "params": [
                {"model__max_depth": 8},
                {"model__max_depth": 12},
            ],
        }
    )

    summary = summarize_tuning_results("random_forest", search)

    assert summary.loc[0, "model"] == "random_forest"
    assert summary.loc[0, "cv_rank"] == 1
    assert bool(summary.loc[0, "selected"])
    assert '"model__max_depth": 12' in summary.loc[0, "params"]

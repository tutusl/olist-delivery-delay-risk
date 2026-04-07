from __future__ import annotations

import pytest

from src.data.prepare import build_modeling_frame
from src.features.engineering import build_model_dataset, split_features_target, validate_no_leakage


def test_modeling_frame_contains_expected_joined_columns(synthetic_olist_tables):
    modeling_frame = build_modeling_frame(synthetic_olist_tables)

    expected_columns = {
        "customer_state",
        "item_count",
        "total_price",
        "payment_type_mode",
        "primary_seller_state",
        "avg_product_weight_g",
    }
    assert expected_columns.issubset(modeling_frame.columns)


def test_build_model_dataset_creates_binary_target(synthetic_olist_tables):
    modeling_frame = build_modeling_frame(synthetic_olist_tables)
    dataset = build_model_dataset(modeling_frame)

    assert "is_late" in dataset.columns
    assert set(dataset["is_late"].unique()) == {0, 1}
    late_flag = dataset.loc[dataset["order_id"] == "order_2", "is_late"].iloc[0]
    assert late_flag == 1


def test_missing_values_are_preserved_for_imputation(synthetic_olist_tables):
    modeling_frame = build_modeling_frame(synthetic_olist_tables)
    dataset = build_model_dataset(modeling_frame)
    X, y = split_features_target(dataset)

    assert X.shape[0] == y.shape[0] == 3
    assert X["avg_product_weight_g"].isna().sum() == 1


def test_validate_no_leakage_blocks_post_outcome_columns():
    validate_no_leakage(["customer_state", "total_price", "payment_type_mode"])

    with pytest.raises(ValueError):
        validate_no_leakage(["customer_state", "order_delivered_customer_date"])

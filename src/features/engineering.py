from __future__ import annotations

import numpy as np
import pandas as pd

LEAKAGE_COLUMNS = {
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "order_status",
    "review_score",
    "review_comment_title",
    "review_comment_message",
}

IDENTIFIER_COLUMNS = {"order_id"}

NUMERIC_FEATURES = [
    "item_count",
    "total_price",
    "total_freight_value",
    "unique_sellers",
    "payment_value",
    "payment_installments_max",
    "payment_type_count",
    "avg_product_weight_g",
    "avg_product_volume_cm3",
    "avg_product_photos_qty",
    "avg_product_name_length",
    "avg_product_description_length",
    "missing_product_metadata_share",
    "seller_state_nunique",
    "estimated_delivery_days",
    "purchase_hour",
    "purchase_day_of_week",
    "purchase_month",
    "purchase_day",
    "is_weekend_purchase",
    "price_per_item",
    "freight_to_price_ratio",
    "seller_customer_same_state",
]

CATEGORICAL_FEATURES = [
    "customer_state",
    "payment_type_mode",
    "primary_seller_state",
]


def build_model_dataset(modeling_frame: pd.DataFrame) -> pd.DataFrame:
    """Create the training dataset and target from the joined order-level frame."""
    required_columns = {
        "order_id",
        "order_status",
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
        "customer_state",
        "payment_type_mode",
        "primary_seller_state",
    }
    missing = required_columns.difference(modeling_frame.columns)
    if missing:
        formatted = ", ".join(sorted(missing))
        raise KeyError(f"Missing required modeling columns: {formatted}")

    dataset = modeling_frame.copy()
    dataset = dataset.loc[dataset["order_status"].eq("delivered")].copy()
    dataset = dataset.loc[
        dataset["order_purchase_timestamp"].notna()
        & dataset["order_estimated_delivery_date"].notna()
        & dataset["order_delivered_customer_date"].notna()
    ].copy()

    dataset["delivery_delay_days"] = (
        dataset["order_delivered_customer_date"] - dataset["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86_400
    dataset["is_late"] = (dataset["delivery_delay_days"] > 0).astype(int)
    dataset["estimated_delivery_days"] = (
        dataset["order_estimated_delivery_date"] - dataset["order_purchase_timestamp"]
    ).dt.total_seconds() / 86_400
    dataset["purchase_hour"] = dataset["order_purchase_timestamp"].dt.hour
    dataset["purchase_day_of_week"] = dataset["order_purchase_timestamp"].dt.dayofweek
    dataset["purchase_month"] = dataset["order_purchase_timestamp"].dt.month
    dataset["purchase_day"] = dataset["order_purchase_timestamp"].dt.day
    dataset["is_weekend_purchase"] = dataset["purchase_day_of_week"].isin([5, 6]).astype(int)
    dataset["price_per_item"] = dataset["total_price"] / dataset["item_count"].replace(0, np.nan)
    dataset["freight_to_price_ratio"] = dataset["total_freight_value"] / dataset["total_price"].replace(
        0,
        np.nan,
    )
    dataset["seller_customer_same_state"] = (
        dataset["primary_seller_state"].fillna("unknown") == dataset["customer_state"].fillna("unknown")
    ).astype(int)

    selected_columns = ["order_id"] + NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["is_late"]
    dataset = dataset[selected_columns].copy()
    dataset = dataset.replace([np.inf, -np.inf], np.nan)

    return dataset


def split_features_target(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split the model dataset into predictors and binary target."""
    X = dataset.drop(columns=["is_late", *IDENTIFIER_COLUMNS], errors="ignore")
    y = dataset["is_late"].astype(int)
    return X, y


def validate_no_leakage(columns: list[str] | pd.Index) -> None:
    """Raise when a feature set includes post-outcome columns."""
    column_set = set(columns)
    leaked = sorted(column_set.intersection(LEAKAGE_COLUMNS))
    if leaked:
        formatted = ", ".join(leaked)
        raise ValueError(f"Feature set contains leakage columns: {formatted}")

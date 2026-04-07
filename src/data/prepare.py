from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def _require_tables(tables: dict[str, pd.DataFrame], required: Iterable[str]) -> None:
    missing = [name for name in required if name not in tables]
    if missing:
        formatted = ", ".join(sorted(missing))
        raise KeyError(f"Missing required tables: {formatted}")


def _mode_or_unknown(series: pd.Series) -> str:
    non_null = series.dropna()
    if non_null.empty:
        return "unknown"
    mode = non_null.mode()
    return str(mode.iloc[0]) if not mode.empty else "unknown"


def _build_item_aggregates(order_items: pd.DataFrame) -> pd.DataFrame:
    return (
        order_items.groupby("order_id", dropna=False)
        .agg(
            item_count=("order_item_id", "count"),
            total_price=("price", "sum"),
            total_freight_value=("freight_value", "sum"),
            unique_sellers=("seller_id", "nunique"),
        )
        .reset_index()
    )


def _build_payment_aggregates(order_payments: pd.DataFrame) -> pd.DataFrame:
    return (
        order_payments.groupby("order_id", dropna=False)
        .agg(
            payment_value=("payment_value", "sum"),
            payment_installments_max=("payment_installments", "max"),
            payment_type_mode=("payment_type", _mode_or_unknown),
            payment_type_count=("payment_type", "nunique"),
        )
        .reset_index()
    )


def _build_product_aggregates(order_items: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    product_columns = [
        "product_id",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
        "product_photos_qty",
        "product_name_lenght",
        "product_description_lenght",
    ]
    product_features = order_items[["order_id", "product_id"]].merge(
        products[product_columns],
        on="product_id",
        how="left",
    )
    numeric_columns = [
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
        "product_photos_qty",
        "product_name_lenght",
        "product_description_lenght",
    ]
    for column in numeric_columns:
        product_features[column] = pd.to_numeric(product_features[column], errors="coerce")

    product_features["product_volume_cm3"] = (
        product_features["product_length_cm"]
        * product_features["product_height_cm"]
        * product_features["product_width_cm"]
    )
    product_features["missing_product_metadata"] = (
        product_features[numeric_columns].isna().all(axis=1).astype(int)
    )

    return (
        product_features.groupby("order_id", dropna=False)
        .agg(
            avg_product_weight_g=("product_weight_g", "mean"),
            avg_product_volume_cm3=("product_volume_cm3", "mean"),
            avg_product_photos_qty=("product_photos_qty", "mean"),
            avg_product_name_length=("product_name_lenght", "mean"),
            avg_product_description_length=("product_description_lenght", "mean"),
            missing_product_metadata_share=("missing_product_metadata", "mean"),
        )
        .reset_index()
    )


def _build_seller_aggregates(order_items: pd.DataFrame, sellers: pd.DataFrame) -> pd.DataFrame:
    seller_features = order_items[["order_id", "seller_id"]].merge(
        sellers[["seller_id", "seller_state"]],
        on="seller_id",
        how="left",
    )
    return (
        seller_features.groupby("order_id", dropna=False)
        .agg(
            seller_state_nunique=("seller_state", lambda values: values.dropna().nunique()),
            primary_seller_state=("seller_state", _mode_or_unknown),
        )
        .reset_index()
    )


def build_modeling_frame(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join the subset of Olist tables into an order-level modeling frame."""
    required_tables = {"customers", "orders", "order_items", "order_payments", "products", "sellers"}
    _require_tables(tables, required_tables)

    orders = tables["orders"].copy()
    customers = tables["customers"][["customer_id", "customer_city", "customer_state"]].copy()
    order_items = tables["order_items"].copy()
    order_payments = tables["order_payments"].copy()
    products = tables["products"].copy()
    sellers = tables["sellers"].copy()

    item_aggregates = _build_item_aggregates(order_items)
    payment_aggregates = _build_payment_aggregates(order_payments)
    product_aggregates = _build_product_aggregates(order_items, products)
    seller_aggregates = _build_seller_aggregates(order_items, sellers)

    modeling_frame = (
        orders.merge(customers, on="customer_id", how="left")
        .merge(item_aggregates, on="order_id", how="left")
        .merge(payment_aggregates, on="order_id", how="left")
        .merge(product_aggregates, on="order_id", how="left")
        .merge(seller_aggregates, on="order_id", how="left")
        .sort_values("order_purchase_timestamp")
        .reset_index(drop=True)
    )

    return modeling_frame

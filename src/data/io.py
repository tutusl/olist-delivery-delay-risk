from __future__ import annotations

from pathlib import Path

import pandas as pd

EXPECTED_FILES: dict[str, str] = {
    "customers": "olist_customers_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
}

DATE_COLUMNS: dict[str, list[str]] = {
    "orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "order_items": ["shipping_limit_date"],
}


def list_expected_files() -> list[str]:
    """Return the required raw CSV filenames in a stable order."""
    return list(EXPECTED_FILES.values())


def load_olist_tables(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load the subset of Olist CSV tables used by the project."""
    root = Path(data_dir)
    missing = [filename for filename in EXPECTED_FILES.values() if not (root / filename).exists()]
    if missing:
        formatted = ", ".join(sorted(missing))
        raise FileNotFoundError(
            f"Missing required raw files in {root}: {formatted}. "
            "Download the Olist dataset and place the CSV files in data/raw/."
        )

    tables: dict[str, pd.DataFrame] = {}
    for table_name, filename in EXPECTED_FILES.items():
        csv_path = root / filename
        tables[table_name] = pd.read_csv(csv_path, parse_dates=DATE_COLUMNS.get(table_name, []))

    return tables

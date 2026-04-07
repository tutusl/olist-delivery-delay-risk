from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def synthetic_olist_tables() -> dict[str, pd.DataFrame]:
    customers = pd.DataFrame(
        {
            "customer_id": ["customer_1", "customer_2", "customer_3"],
            "customer_city": ["sao paulo", "rio de janeiro", "curitiba"],
            "customer_state": ["SP", "RJ", "PR"],
        }
    )
    orders = pd.DataFrame(
        {
            "order_id": ["order_1", "order_2", "order_3"],
            "customer_id": ["customer_1", "customer_2", "customer_3"],
            "order_status": ["delivered", "delivered", "delivered"],
            "order_purchase_timestamp": pd.to_datetime(
                ["2018-01-01 10:00:00", "2018-01-05 20:30:00", "2018-02-10 14:00:00"]
            ),
            "order_approved_at": pd.to_datetime(
                ["2018-01-01 11:00:00", "2018-01-05 21:00:00", "2018-02-10 15:00:00"]
            ),
            "order_delivered_carrier_date": pd.to_datetime(
                ["2018-01-02 09:00:00", "2018-01-06 08:00:00", "2018-02-11 09:00:00"]
            ),
            "order_delivered_customer_date": pd.to_datetime(
                ["2018-01-07 12:00:00", "2018-01-15 12:00:00", "2018-02-17 18:00:00"]
            ),
            "order_estimated_delivery_date": pd.to_datetime(
                ["2018-01-10 00:00:00", "2018-01-12 00:00:00", "2018-02-16 00:00:00"]
            ),
        }
    )
    order_items = pd.DataFrame(
        {
            "order_id": ["order_1", "order_2", "order_3"],
            "order_item_id": [1, 1, 1],
            "product_id": ["product_1", "product_2", "product_3"],
            "seller_id": ["seller_1", "seller_2", "seller_3"],
            "shipping_limit_date": pd.to_datetime(
                ["2018-01-03 00:00:00", "2018-01-06 00:00:00", "2018-02-12 00:00:00"]
            ),
            "price": [100.0, 200.0, 80.0],
            "freight_value": [15.0, 35.0, 10.0],
        }
    )
    order_payments = pd.DataFrame(
        {
            "order_id": ["order_1", "order_2", "order_3"],
            "payment_sequential": [1, 1, 1],
            "payment_type": ["credit_card", "boleto", "credit_card"],
            "payment_installments": [2, 1, 4],
            "payment_value": [115.0, 235.0, 90.0],
        }
    )
    products = pd.DataFrame(
        {
            "product_id": ["product_1", "product_2", "product_3"],
            "product_weight_g": [500.0, 2500.0, None],
            "product_length_cm": [20.0, 60.0, None],
            "product_height_cm": [10.0, 20.0, None],
            "product_width_cm": [15.0, 30.0, None],
            "product_photos_qty": [3.0, 5.0, None],
            "product_name_lenght": [40.0, 55.0, None],
            "product_description_lenght": [200.0, 450.0, None],
        }
    )
    sellers = pd.DataFrame(
        {
            "seller_id": ["seller_1", "seller_2", "seller_3"],
            "seller_state": ["SP", "MG", "PR"],
            "seller_city": ["sao paulo", "belo horizonte", "curitiba"],
        }
    )

    return {
        "customers": customers,
        "orders": orders,
        "order_items": order_items,
        "order_payments": order_payments,
        "products": products,
        "sellers": sellers,
    }

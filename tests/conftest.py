from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def transaction_history() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    start = pd.Timestamp("2024-01-01 10:00:00")
    for customer_offset in range(8):
        customer_id = 10_001 + customer_offset
        for order_offset in range(4):
            order_date = start + pd.Timedelta(
                days=order_offset * (customer_offset + 2) * 7,
                hours=customer_offset,
            )
            for sku_offset in range(3):
                rows.append(
                    {
                        "InvoiceNo": f"{customer_id}-{order_offset}",
                        "StockCode": f"SKU-{sku_offset + (customer_offset % 3)}",
                        "Description": "Synthetic product",
                        "Quantity": 1 + customer_offset + order_offset + sku_offset,
                        "InvoiceDate": order_date,
                        "UnitPrice": 1.5 + customer_offset * 0.75 + sku_offset,
                        "CustomerID": float(customer_id),
                        "Country": "United Kingdom",
                    }
                )
    return pd.DataFrame(rows)

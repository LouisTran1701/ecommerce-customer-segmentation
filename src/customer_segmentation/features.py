"""Transaction validation, cleaning, and customer-level feature engineering."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .config import BULK_QUANTITY, MODEL_FEATURES

REQUIRED_COLUMNS = {
    "InvoiceNo",
    "StockCode",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
}


class DataValidationError(ValueError):
    """Raised when transaction data cannot satisfy the model contract."""


def validate_transaction_schema(transactions: pd.DataFrame) -> None:
    """Validate the columns required by the feature pipeline."""
    missing = sorted(REQUIRED_COLUMNS.difference(transactions.columns))
    if missing:
        raise DataValidationError(f"Missing required columns: {', '.join(missing)}")
    if transactions.empty:
        raise DataValidationError("Transaction data is empty.")


def _coefficient_of_variation(values: Iterable[float]) -> float:
    series = pd.Series(values, dtype=float)
    mean = series.mean()
    if len(series) < 2 or np.isclose(mean, 0.0):
        return 0.0
    return float(series.std(ddof=1) / mean)


def _prepare_transactions(
    transactions: pd.DataFrame,
    as_of_date: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    validate_transaction_schema(transactions)
    data = transactions.copy()

    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], errors="coerce")
    if data["InvoiceDate"].isna().any():
        count = int(data["InvoiceDate"].isna().sum())
        raise DataValidationError(f"InvoiceDate contains {count} invalid values.")

    for column in ("Quantity", "UnitPrice", "CustomerID"):
        original_missing = data[column].isna()
        data[column] = pd.to_numeric(data[column], errors="coerce")
        newly_invalid = data[column].isna() & ~original_missing
        if newly_invalid.any():
            raise DataValidationError(f"{column} contains non-numeric values.")

    data = data.loc[
        data["Country"].eq("United Kingdom") & data["CustomerID"].notna()
    ].copy()
    if data.empty:
        raise DataValidationError("No identified United Kingdom customers were found.")

    customer_ids = data["CustomerID"].to_numpy(dtype=float)
    if not np.allclose(customer_ids, np.round(customer_ids)):
        raise DataValidationError("CustomerID values must be whole numbers.")
    data["CustomerID"] = data["CustomerID"].astype(int)

    if as_of_date is None:
        resolved_as_of = data["InvoiceDate"].max() + pd.Timedelta(days=1)
    else:
        resolved_as_of = pd.Timestamp(as_of_date)
        if pd.isna(resolved_as_of):
            raise DataValidationError(f"Invalid as-of date: {as_of_date!r}")

    data = data.loc[data["InvoiceDate"] < resolved_as_of].copy()
    if data.empty:
        raise DataValidationError("No transactions occur before the as-of date.")

    data["TotalPrice"] = data["Quantity"] * data["UnitPrice"]
    is_cancel = data["InvoiceNo"].astype(str).str.startswith("C")

    cancellations = data.loc[is_cancel].copy()
    sales = data.loc[
        ~is_cancel & data["Quantity"].gt(0) & data["UnitPrice"].gt(0)
    ].copy()
    if sales.empty:
        raise DataValidationError("No valid positive-value sales remain after cleaning.")

    return sales, cancellations, resolved_as_of


def clean_transactions(
    transactions: pd.DataFrame,
    as_of_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return valid UK sales used to construct customer features."""
    sales, _, _ = _prepare_transactions(transactions, as_of_date)
    return sales


def build_customer_features(
    transactions: pd.DataFrame,
    as_of_date: str | pd.Timestamp | None = None,
    *,
    bulk_quantity: int = BULK_QUANTITY,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Aggregate raw transaction history into the fixed production feature schema."""
    sales, cancellations, resolved_as_of = _prepare_transactions(transactions, as_of_date)

    order_summary = (
        sales.groupby(["CustomerID", "InvoiceNo"], sort=False)
        .agg(
            order_date=("InvoiceDate", "min"),
            order_total=("TotalPrice", "sum"),
            total_qty=("Quantity", "sum"),
        )
        .reset_index()
        .sort_values(["CustomerID", "order_date"])
    )

    rhythm_rows: list[dict[str, float | int]] = []
    for customer_id, group in order_summary.groupby("CustomerID", sort=False):
        dates = group["order_date"].sort_values()
        order_count = len(dates)
        active_span_days = int((dates.iloc[-1] - dates.iloc[0]).days)
        gaps = dates.diff().dropna().dt.days
        rhythm_rows.append(
            {
                "CustomerID": int(customer_id),
                "Recency": int((resolved_as_of - dates.iloc[-1]).days),
                "InterPurchaseCV": _coefficient_of_variation(gaps),
                "MonthlyOrderRate": (
                    float(order_count)
                    if order_count < 2
                    else float(order_count / max(active_span_days / 30, 1.0))
                ),
            }
        )
    rhythm = pd.DataFrame(rhythm_rows).set_index("CustomerID")

    order_totals = order_summary.groupby("CustomerID")["order_total"]
    average_order_value = order_totals.mean().rename("AOV")
    burst_index = (order_totals.max() / order_totals.mean()).rename("BurstIndex")
    median_basket_qty = (
        order_summary.groupby("CustomerID")["total_qty"].median().rename("MedianBasketQty")
    )

    price_stats = sales.groupby("CustomerID")["UnitPrice"].agg(["mean", "std"])
    price_cv = (price_stats["std"] / price_stats["mean"]).fillna(0).rename("PriceCV")
    quantity_stats = sales.groupby("CustomerID")["Quantity"].agg(["mean", "std"])
    quantity_cv = (
        quantity_stats["std"] / quantity_stats["mean"]
    ).fillna(0).rename("QuantityCV")

    monthly_spend = (
        sales.assign(YearMonth=sales["InvoiceDate"].dt.to_period("M"))
        .groupby(["CustomerID", "YearMonth"])["TotalPrice"]
        .sum()
    )
    spend_acceleration_values: dict[int, float] = {}
    for customer_id, values in monthly_spend.groupby(level="CustomerID"):
        spend = values.to_numpy(dtype=float)
        if len(spend) < 3:
            acceleration = 0.0
        else:
            first, last = spend[:3].mean(), spend[-3:].mean()
            acceleration = float(np.clip((last - first) / (abs(first) + 1e-9), -3, 3))
        spend_acceleration_values[int(customer_id)] = acceleration
    spend_acceleration = pd.Series(
        spend_acceleration_values, name="SpendAcceleration", dtype=float
    )

    sku_invoice_count = sales.groupby(["CustomerID", "StockCode"])["InvoiceNo"].nunique()
    repeat_sku_fraction = (
        sku_invoice_count.gt(1)
        .groupby(level="CustomerID")
        .mean()
        .rename("RepeatSKUFraction")
    )

    unique_skus_per_order = sales.groupby(["CustomerID", "InvoiceNo"])["StockCode"].nunique()
    basket_stats = unique_skus_per_order.groupby(level="CustomerID").agg(["mean", "std"])
    basket_size_cv = (
        basket_stats["std"] / basket_stats["mean"]
    ).fillna(0).rename("BasketSizeCV")

    bulk_line_rate = (
        sales["Quantity"]
        .ge(bulk_quantity)
        .groupby(sales["CustomerID"])
        .mean()
        .rename("BulkLineRate")
    )

    sku_spend = sales.groupby(["CustomerID", "StockCode"])["TotalPrice"].sum()
    sku_share = sku_spend / sku_spend.groupby(level="CustomerID").transform("sum")
    sku_hhi = sku_share.pow(2).groupby(level="CustomerID").sum().rename("SKU_HHI")

    sales_invoice_count = sales.groupby("CustomerID")["InvoiceNo"].nunique()
    cancellation_count = cancellations.groupby("CustomerID")["InvoiceNo"].nunique()
    cancellation_count = cancellation_count.reindex(sales_invoice_count.index, fill_value=0)
    return_rate = (
        cancellation_count / (sales_invoice_count + cancellation_count)
    ).rename("ReturnRate")

    quarter_spend = order_summary.assign(
        Quarter=order_summary["order_date"].dt.quarter
    ).groupby(["CustomerID", "Quarter"])["order_total"].sum()
    quarter_concentration = (
        quarter_spend.groupby(level="CustomerID").max()
        / quarter_spend.groupby(level="CustomerID").sum()
    ).clip(0.25, 1.0).rename("QuarterConcentration")

    features = pd.concat(
        [
            rhythm,
            average_order_value,
            price_cv,
            spend_acceleration,
            repeat_sku_fraction,
            basket_size_cv,
            burst_index,
            bulk_line_rate,
            quantity_cv,
            median_basket_qty,
            sku_hhi,
            return_rate,
            quarter_concentration,
        ],
        axis=1,
    )
    features = features.loc[:, MODEL_FEATURES].sort_index()
    features.index.name = "CustomerID"

    if features.isna().any().any() or np.isinf(features.to_numpy()).any():
        raise DataValidationError("Engineered features contain missing or infinite values.")

    return features.astype(float), resolved_as_of

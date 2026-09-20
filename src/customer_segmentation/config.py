"""Stable modeling configuration shared by training and inference."""

SEED = 42
N_CLUSTERS = 4
BULK_QUANTITY = 12
B2B_THRESHOLD = 0.5
MAX_QUANTILES = 500

MODEL_FEATURES = (
    "Recency",
    "InterPurchaseCV",
    "MonthlyOrderRate",
    "AOV",
    "PriceCV",
    "SpendAcceleration",
    "RepeatSKUFraction",
    "BasketSizeCV",
    "BurstIndex",
    "BulkLineRate",
    "QuantityCV",
    "MedianBasketQty",
    "SKU_HHI",
    "ReturnRate",
    "QuarterConcentration",
)

FEATURE_FAMILIES = {
    "Purchase Rhythm": ("Recency", "InterPurchaseCV", "MonthlyOrderRate"),
    "Spending Shape": ("AOV", "PriceCV", "SpendAcceleration"),
    "Basket Behaviour": ("RepeatSKUFraction", "BasketSizeCV"),
    "Volume & Bulk": (
        "BurstIndex",
        "BulkLineRate",
        "QuantityCV",
        "MedianBasketQty",
        "SKU_HHI",
    ),
    "Customer Lifecycle": ("ReturnRate", "QuarterConcentration"),
}


def feature_weights() -> dict[str, float]:
    """Give every feature family equal total influence on Euclidean distance."""
    weights: dict[str, float] = {}
    for features in FEATURE_FAMILIES.values():
        family_weight = len(features) ** -0.5
        weights.update(dict.fromkeys(features, family_weight))
    return weights

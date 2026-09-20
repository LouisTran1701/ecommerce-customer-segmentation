import pandas as pd
import pytest

from customer_segmentation.config import MODEL_FEATURES
from customer_segmentation.features import (
    DataValidationError,
    build_customer_features,
    clean_transactions,
)


def test_missing_columns_are_rejected() -> None:
    with pytest.raises(DataValidationError, match="Missing required columns"):
        clean_transactions(pd.DataFrame({"CustomerID": [1]}))


def test_feature_generation_is_deterministic(transaction_history: pd.DataFrame) -> None:
    first, first_as_of = build_customer_features(transaction_history, "2025-01-01")
    second, second_as_of = build_customer_features(transaction_history, "2025-01-01")

    pd.testing.assert_frame_equal(first, second)
    assert first_as_of == second_as_of == pd.Timestamp("2025-01-01")
    assert tuple(first.columns) == MODEL_FEATURES
    assert first.index.name == "CustomerID"
    assert not first.isna().any().any()

import pandas as pd

from customer_segmentation.model import (
    load_model,
    save_model,
    score_customers,
    train_model,
)


def test_saved_model_reproduces_training_assignments(
    transaction_history: pd.DataFrame, tmp_path
) -> None:
    artifact, training_assignments = train_model(
        transaction_history,
        as_of_date="2025-01-01",
        n_clusters=2,
        b2b_threshold=1.1,
    )
    model_path = save_model(artifact, tmp_path / "segmentation.joblib")

    loaded = load_model(model_path)
    rescored = score_customers(transaction_history, loaded, as_of_date="2025-01-01")

    expected = training_assignments.sort_values("CustomerID").reset_index(drop=True)
    actual = rescored.sort_values("CustomerID").reset_index(drop=True)
    pd.testing.assert_series_equal(actual["Cluster"], expected["Cluster"])
    pd.testing.assert_series_equal(actual["Segment"], expected["Segment"])
    assert (actual["DistanceToCentroid"] >= 0).all()
    assert (actual["DistanceMargin"] >= 0).all()

"""Fit, persist, and use the customer segmentation model."""

from __future__ import annotations

from datetime import UTC, datetime
from itertools import permutations
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from .config import (
    B2B_THRESHOLD,
    MAX_QUANTILES,
    MODEL_FEATURES,
    N_CLUSTERS,
    SEED,
    feature_weights,
)
from .features import DataValidationError, build_customer_features

ARTIFACT_VERSION = 1
B2B_SEGMENT_NAME = "High-Concentration B2B"

PERSONA_RULES: dict[str, dict[str, float]] = {
    "Frequent Seasonal Intensives": {
        "MonthlyOrderRate": 1.5,
        "QuarterConcentration": 1.0,
        "InterPurchaseCV": -1.0,
    },
    "Low-Volume Irregular Explorers": {
        "InterPurchaseCV": 1.0,
        "QuantityCV": 1.0,
        "BulkLineRate": -1.0,
        "MonthlyOrderRate": -0.5,
    },
    "High-Value Bulk Actives (High Returns)": {
        "ReturnRate": 1.5,
        "AOV": 1.0,
        "MedianBasketQty": 1.0,
        "BulkLineRate": 0.5,
    },
    "Steady Routine Buyers": {
        "BasketSizeCV": -1.0,
        "QuantityCV": -1.0,
        "InterPurchaseCV": -0.5,
        "RepeatSKUFraction": -0.5,
    },
}


def _canonical_cluster_names(features: pd.DataFrame, labels: np.ndarray) -> dict[int, str]:
    cluster_ids = sorted(int(value) for value in np.unique(labels))
    if len(cluster_ids) != len(PERSONA_RULES):
        return {cluster_id: f"Cluster {cluster_id}" for cluster_id in cluster_ids}

    profiles = features.assign(_cluster=labels).groupby("_cluster").median()
    standard_deviation = profiles.std(axis=0, ddof=0).replace(0, 1)
    standardized = (profiles - profiles.mean(axis=0)) / standard_deviation

    persona_names = list(PERSONA_RULES)
    scores = pd.DataFrame(index=cluster_ids, columns=persona_names, dtype=float)
    for persona, rule in PERSONA_RULES.items():
        scores[persona] = sum(
            standardized[feature] * coefficient for feature, coefficient in rule.items()
        )

    best_assignment: tuple[int, ...] | None = None
    best_score = -np.inf
    for assignment in permutations(cluster_ids):
        score = sum(
            scores.loc[cluster_id, persona]
            for persona, cluster_id in zip(persona_names, assignment, strict=True)
        )
        if score > best_score:
            best_score = float(score)
            best_assignment = assignment

    assert best_assignment is not None
    return {
        cluster_id: persona
        for persona, cluster_id in zip(persona_names, best_assignment, strict=True)
    }


def _transform_features(features: pd.DataFrame, artifact: dict[str, Any]) -> np.ndarray:
    names = artifact["feature_names"]
    missing = sorted(set(names).difference(features.columns))
    if missing:
        raise DataValidationError(f"Missing model features: {', '.join(missing)}")

    values = features.loc[:, names]
    quantile_values = artifact["quantile_transformer"].transform(values)
    standardized = artifact["standard_scaler"].transform(quantile_values)
    clipped = np.clip(standardized, -3.0, 3.0)
    weights = np.array([artifact["feature_weights"][name] for name in names])
    return clipped * weights


def _score_feature_frame(features: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    threshold = float(artifact["b2b_threshold"])
    retail_mask = features["SKU_HHI"].le(threshold)
    result = pd.DataFrame(index=features.index)
    result["Cluster"] = pd.Series(pd.NA, index=features.index, dtype="Int64")
    result["Segment"] = B2B_SEGMENT_NAME
    result["DistanceToCentroid"] = np.nan
    result["DistanceMargin"] = np.nan
    result["IsB2B"] = ~retail_mask

    retail_features = features.loc[retail_mask]
    if not retail_features.empty:
        transformed = _transform_features(retail_features, artifact)
        labels = artifact["kmeans"].predict(transformed)
        distances = artifact["kmeans"].transform(transformed)
        nearest = np.take_along_axis(distances, labels[:, None], axis=1).ravel()
        sorted_distances = np.sort(distances, axis=1)
        margins = (
            sorted_distances[:, 1] - sorted_distances[:, 0]
            if distances.shape[1] > 1
            else np.full(len(distances), np.nan)
        )

        result.loc[retail_features.index, "Cluster"] = labels
        result.loc[retail_features.index, "Segment"] = [
            artifact["cluster_names"][int(label)] for label in labels
        ]
        result.loc[retail_features.index, "DistanceToCentroid"] = nearest
        result.loc[retail_features.index, "DistanceMargin"] = margins

    return result.reset_index()


def train_model(
    transactions: pd.DataFrame,
    as_of_date: str | pd.Timestamp | None = None,
    *,
    n_clusters: int = N_CLUSTERS,
    b2b_threshold: float = B2B_THRESHOLD,
    random_state: int = SEED,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit preprocessing and K-Means, returning a serializable artifact and assignments."""
    if n_clusters < 2:
        raise DataValidationError("At least two retail clusters are required.")

    features, resolved_as_of = build_customer_features(transactions, as_of_date)
    retail_features = features.loc[features["SKU_HHI"].le(b2b_threshold)]
    if len(retail_features) <= n_clusters:
        raise DataValidationError(
            f"Need more than {n_clusters} retail customers; found {len(retail_features)}."
        )

    names = list(MODEL_FEATURES)
    raw_values = retail_features.loc[:, names]
    quantile_transformer = QuantileTransformer(
        output_distribution="normal",
        n_quantiles=min(MAX_QUANTILES, len(raw_values)),
        random_state=random_state,
    )
    quantile_values = quantile_transformer.fit_transform(raw_values)
    standard_scaler = StandardScaler()
    standardized = standard_scaler.fit_transform(quantile_values)
    clipped = np.clip(standardized, -3.0, 3.0)
    weights = feature_weights()
    model_values = clipped * np.array([weights[name] for name in names])

    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=20,
        max_iter=300,
        random_state=random_state,
    )
    labels = kmeans.fit_predict(model_values)
    unique_labels = np.unique(labels)
    if len(unique_labels) != n_clusters:
        raise DataValidationError(
            f"K-Means produced {len(unique_labels)} distinct clusters, expected {n_clusters}."
        )
    metrics = {
        "inertia": float(kmeans.inertia_),
        "silhouette": float(silhouette_score(model_values, labels)),
        "davies_bouldin": float(davies_bouldin_score(model_values, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(model_values, labels)),
    }

    cluster_names = _canonical_cluster_names(retail_features, labels)
    artifact: dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "training_reference_date": resolved_as_of.isoformat(),
        "feature_names": names,
        "feature_weights": weights,
        "b2b_threshold": b2b_threshold,
        "quantile_transformer": quantile_transformer,
        "standard_scaler": standard_scaler,
        "kmeans": kmeans,
        "cluster_names": cluster_names,
        "metrics": metrics,
        "training_customers": int(len(features)),
        "training_retail_customers": int(len(retail_features)),
        "cluster_counts": {
            int(cluster_id): int(count)
            for cluster_id, count in zip(*np.unique(labels, return_counts=True), strict=True)
        },
    }
    assignments = _score_feature_frame(features, artifact)
    return artifact, assignments


def score_customers(
    transactions: pd.DataFrame,
    artifact: dict[str, Any],
    as_of_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Score customer histories with a previously fitted artifact."""
    _validate_artifact(artifact)
    features, _ = build_customer_features(transactions, as_of_date)
    return _score_feature_frame(features, artifact)


def _validate_artifact(artifact: dict[str, Any]) -> None:
    if artifact.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError(
            f"Unsupported model artifact version: {artifact.get('artifact_version')!r}"
        )


def save_model(artifact: dict[str, Any], path: str | Path) -> Path:
    """Persist a complete training artifact."""
    _validate_artifact(artifact)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, destination)
    return destination


def load_model(path: str | Path) -> dict[str, Any]:
    """Load and validate a persisted training artifact."""
    artifact = joblib.load(Path(path))
    if not isinstance(artifact, dict):
        raise ValueError("Model artifact must be a dictionary.")
    _validate_artifact(artifact)
    return artifact

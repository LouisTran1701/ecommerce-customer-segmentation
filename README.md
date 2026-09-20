# E-Commerce Customer Segmentation

A reproducible customer-segmentation pipeline built from the
[UCI Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail).
The project keeps the exploratory notebooks for analysis and provides a tested Python package
for repeatable training and batch or on-demand scoring.

## What the project does

The production path performs five steps:

1. Validate and clean UK transaction history.
2. Aggregate each customer into 15 behavioral features.
3. Separate high-concentration B2B customers using `SKU_HHI`.
4. Apply fitted quantile transformation, standardization, and feature-family weights.
5. Assign retail customers directly with `KMeans.predict()`.

The saved model artifact includes the fitted transformers, feature order and weights, K-Means
model, training reference date, quality metrics, and cluster-to-persona mapping. This avoids
training a second classifier to imitate K-Means.

## Quick start

Python 3.14 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --dev
```

Train the complete pipeline from raw transaction history:

```bash
uv run segment train \
  --input data/raw/ecommerce-data.csv \
  --model-out artifacts/segmentation.joblib \
  --assignments-out outputs/training_segments.csv
```

Score another transaction-history file with the fitted artifact:

```bash
uv run segment score \
  --input path/to/customer_transactions.csv \
  --model artifacts/segmentation.joblib \
  --output outputs/customer_segments.csv
```

Both commands accept an optional exclusive snapshot cutoff such as
`--as-of 2011-12-10`. Scoring input must contain customer transaction history, not a precomputed
feature row or a single purchase event.

The prediction output contains:

- `CustomerID`, numeric `Cluster`, and stable business `Segment`
- `DistanceToCentroid`, where smaller values indicate a closer cluster fit
- `DistanceMargin`, where larger values indicate better separation from the next cluster
- `IsB2B`, identifying customers handled by the B2B rule instead of retail K-Means

## Quality checks

```bash
uv run ruff check .
uv run pytest
```

GitHub Actions runs the same checks on pushes and pull requests.

## Repository structure

```text
practice-knn-kmeans/
├── src/customer_segmentation/
│   ├── cli.py                 # train and score commands
│   ├── config.py              # stable feature/model configuration
│   ├── features.py            # validation, cleaning, feature engineering
│   └── model.py               # fit, persistence, and inference
├── tests/                     # deterministic feature and model tests
├── notebook/
│   ├── 01_eda_cleaning.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_customer_clustering.ipynb
├── data/                      # source and generated data
├── artifacts/                 # generated model files, ignored by Git
├── outputs/                   # generated predictions, ignored by Git
└── pyproject.toml
```

## Modeling choices

The fixed production schema covers purchase rhythm, spending shape, basket behavior, volume and
bulk purchasing, and customer lifecycle. K-Means uses 15 transformed features and four retail
segments:

- Frequent Seasonal Intensives
- Low-Volume Irregular Explorers
- High-Value Bulk Actives (High Returns)
- Steady Routine Buyers

Cluster numbers are implementation details. Persona names are assigned from cluster behavioral
profiles during training and persisted in the artifact so scoring remains consistent.

The notebooks remain the research record for EDA, feature selection, cluster-count comparison,
and stability analysis. Production code deliberately avoids notebook-relative paths and fitted
state hidden inside notebook cells.

## Current status and next steps

- [x] EDA and cleaning research
- [x] Customer feature engineering research
- [x] K-Means selection and customer profiling
- [x] Reusable training and scoring package
- [x] Versioned model artifact, tests, linting, and CI
- [ ] Add exact per-feature centroid-distance explanations
- [ ] Add a small Streamlit demonstration after the CLI contract is stable

## Data and license

This repository is intended for educational and portfolio use. The underlying Online Retail
dataset is sourced from the UCI Machine Learning Repository. Generated model artifacts and scoring
outputs are intentionally excluded from version control.

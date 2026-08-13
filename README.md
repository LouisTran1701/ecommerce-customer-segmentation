# 🛒 E-Commerce Customer Segmentation & Real-Time Classification Pipeline

An end-to-end machine learning project using the [UCI Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail). This repository combines **unsupervised learning** (K-Means) to discover actionable customer behavioral segments, **supervised learning** (KNN) to classify new customers in real-time, and **Explainable AI** (SHAP) to interpret individual segment assignments.

---

## 📌 Project Overview & Current Progress

```
┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│     1. EDA & Clean      │     │  2. Feature Engineering │     │   3. K-Means Discovery  │     │   4. KNN & SHAP (Next)  │
│      [COMPLETED]        │────▶│       [COMPLETED]       │────▶│       [COMPLETED]       │────▶│      [IN PROGRESS]      │
│                         │     │                         │     │                         │     │                         │
│ Clean raw transactions, │     │ Build 15 behavioral     │     │ K-Selection sweep &     │     │ Real-time classification│
│ handle returns/outliers │     │ metrics beyond RFM      │     │ Profile 4 customer K=4  │     │ & SHAP explainability   │
└─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘
```

---

## 🎯 Business Problem & Solution Architecture

An e-commerce retailer wants to segment its customer base to:
1. **Discover** distinct behavioral customer groups (e.g. bulk wholesale buyers, seasonal shoppers, low-volume casual buyers).
2. **Classify** new or returning customers into existing segments instantly without re-clustering the entire dataset.
3. **Explain** feature contributions behind each customer's segment assignment for non-technical stakeholders.
4. **Drive** tailored marketing strategies, retention campaigns, and pricing models for each segment.

### Three-Stage Technical Architecture

| Stage | Method / Algorithm | Status | Purpose |
|---|---|---|---|
| **1. Segment Discovery** | **K-Means Clustering** | ✅ Completed | Unsupervised clustering on 15 normalized behavioral features ($K=4$) |
| **2. Real-Time Classifier** | **K-Nearest Neighbors (KNN)** | ⏳ Up Next | Supervised classifier trained on discovered segment labels for sub-millisecond inference |
| **3. Model Interpretability** | **SHAP (SHapley Additive exPlanations)** | ⏳ Up Next | Global & local feature importance explanations for business stakeholders |

---

## 🧠 Engineered Feature Matrix (15 Behavioral Features)

Rather than relying solely on traditional RFM (Recency, Frequency, Monetary), **15 granular customer-level features** across 6 behavioral domains were engineered in [`notebook/02_feature_engineering.ipynb`](notebook/02_feature_engineering.ipynb):

| Feature Domain | Feature Name | Definition | Business Value |
|---|---|---|---|
| **RFM Baseline** | `Recency` | Log-transformed days since last purchase | Measures customer recency & churn risk |
| | `MonthlyOrderRate` | Average distinct orders per month active | Normalized order frequency |
| | `AOV` | Log-transformed Average Order Value ($) | Monetary tier per transaction |
| **Purchase Rhythms** | `InterPurchaseCV` | Coeff. of Variation of days between orders | Measures purchase timing regularity |
| | `SpendAcceleration` | Spend slope across customer lifecycle quarters | Detects growing vs waning spend |
| | `BurstIndex` | Max monthly orders relative to baseline average | Identifies impulse/event purchasing bursts |
| **Order Composition** | `MedianBasketQty` | Median items per order | Distinguishes single-item vs bulk baskets |
| | `BasketSizeCV` | Coeff. of Variation of basket item counts | Measures order size consistency |
| | `BulkLineRate` | Fraction of line items with bulk quantities ($\ge 10$) | Identifies wholesale & reseller traits |
| **Product Diversity** | `RepeatSKUFraction` | Fraction of items reordered across orders | Measures repeat product loyalty |
| | `SKU_HHI` | Herfindahl-Hirschman Index of SKU spending | Quantifies product concentration vs diversity |
| **Risk & Seasonality** | `ReturnRate` | Returned item quantity over total purchased | Identifies return abuse & B2B sampling |
| | `QuarterConcentration` | Max quarterly order share over total orders | Detects seasonal buying spikes |
| | `PriceCV` | Coeff. of Variation of unit prices purchased | Measures price sensitivity across tiers |
| | `QuantityCV` | Coeff. of Variation of order quantities | Measures volume fluctuation per order |

---

## 📊 Discovered Customer Segments ($K=4$)

In [`notebook/03_customer_clustering.ipynb`](notebook/03_customer_clustering.ipynb), K-Means model evaluation (WCSS, Silhouette, Davies-Bouldin, Calinski-Harabasz, and ARI stability = 0.999) selected **$K=4$** as the optimal segmentation solution:

| Cluster | Segment Persona | Size (% / N) | Key Distinctive Features ($z$-score) | Strategic Marketing Focus |
| :---: | :--- | :---: | :--- | :--- |
| **Cluster 0** | **Frequent Seasonal Intensives** | **8.85%** ($N=331$) | `MonthlyOrderRate` (+1.56)<br>`QuarterConcentration` (+1.00)<br>`InterPurchaseCV` (-1.00) | Early-bird seasonal pre-orders, holiday campaigns, cross-quarter engagement |
| **Cluster 1** | **Low-Volume Irregular Explorers** | **28.11%** ($N=1,052$) | `InterPurchaseCV` (+0.94)<br>`QuantityCV` (+0.94)<br>`BulkLineRate` (-1.15) | Automated re-engagement email flows, low-threshold free shipping, recommendations |
| **Cluster 2** | **High-Value Bulk Actives (High Returns)** | **29.50%** ($N=1,104$) | `ReturnRate` (+1.73)<br>`AOV` (+1.71)<br>`MedianBasketQty` (+1.67) | Dedicated B2B account management, volume tier discounts, return mitigation |
| **Cluster 3** | **Steady Routine Buyers (Occasional Loyalists)** | **33.54%** ($N=1,255$) | `BasketSizeCV` (-1.73)<br>`RepeatSKUFraction` (-1.73)<br>`QuantityCV` (-1.68) | Subscription / auto-replenishment programs, cross-selling higher-margin tiers |

---

## 📁 Repository Structure

```
practice-knn-kmeans/
├── data/
│   ├── raw/                            # Original UCI Online Retail dataset
│   └── processed/                      # Transformed features & segment outputs
│       ├── cleaned_transactions.csv    # Cleaned transaction-level data
│       ├── retail_customers.csv        # Engineered 15 customer-level features
│       ├── final_customer_segments.csv # Customers tagged with K=4 cluster labels
│       ├── rfm_summary.csv             # RFM baseline summary
│       └── b2b_customers.csv           # B2B customer subset
├── notebook/
│   ├── 01_eda_cleaning.ipynb           # EDA, missing values, returns & outlier treatment
│   ├── 02_feature_engineering.ipynb    # 15 behavioral feature generation & log-transform
│   └── 03_customer_clustering.ipynb   # K-Selection sweep, K=4 K-Means model & segment personas
├── config.py                           # Project configurations & hyperparameters
├── pyproject.toml                      # Dependencies & package configuration
└── README.md                           # Project documentation
```

---

## 🗺️ Roadmap & Current Status

- [x] **Stage 1: Exploratory Data Analysis & Data Cleaning** ([`01_eda_cleaning.ipynb`](notebook/01_eda_cleaning.ipynb))
- [x] **Stage 2: Advanced Feature Engineering** ([`02_feature_engineering.ipynb`](notebook/02_feature_engineering.ipynb)) — *15 behavioral features created*
- [x] **Stage 3: Customer Segmentation & Profiling** ([`03_customer_clustering.ipynb`](notebook/03_customer_clustering.ipynb)) — *Final $K=4$ K-Means model*
- [ ] **Stage 4: Supervised Classification Pipeline** — *Train KNN model to predict cluster labels*
- [ ] **Stage 5: Model Interpretability** — *Implement SHAP global & local feature attribution*
- [ ] **Stage 6: Production Refactoring** — *Modular `src/` Python package & CLI CLI (`main.py`)*
- [ ] **Stage 7: Interactive Web Dashboard** — *Streamlit customer segmentation portal*

---

## 🚀 Getting Started

### Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) package manager (recommended) or pip

### Installation

```bash
# Clone the repository
git clone https://github.com/LouisTran1701/ecommerce-customer-segmentation.git
cd practice-knn-kmeans

# Install dependencies via uv
uv sync
```

### Running Notebooks

```bash
# Launch Jupyter Notebooks
jupyter notebook notebook/
```

---

## 🛠 Tech Stack

| Category | Tools & Libraries |
|---|---|
| **Data Processing** | pandas, NumPy, SciPy |
| **Data Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | scikit-learn (KMeans, StandardScaler, KNN) |
| **Explainability** | SHAP |
| **Environment & Tooling** | uv, Jupyter Notebook, Git |

---

## 📄 License

This repository is built for educational and portfolio demonstration purposes. The underlying dataset is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online+retail).

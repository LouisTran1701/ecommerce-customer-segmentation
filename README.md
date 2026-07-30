# 🛒 Customer Segmentation: K-Means + KNN Pipeline

An end-to-end customer segmentation project using the [UCI Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail). This project combines **unsupervised learning** (K-Means) to discover natural customer segments, **supervised learning** (KNN) to classify new customers in real-time, and **Explainable AI** (XAI) to interpret why each customer is assigned to a specific segment.

---

## 🎯 Problem Statement

An e-commerce retailer wants to segment its customers to:
- **Identify** distinct customer groups (e.g., high-value loyalists, at-risk churners, new promising buyers)
- **Classify** new customers into existing segments instantly — without re-running the full clustering pipeline
- **Explain** why a customer belongs to a particular segment for business stakeholders
- **Enable** targeted marketing strategies for each segment

---

## 💡 Approach

### Three-Stage Pipeline

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   K-Means       │     │   KNN           │     │   XAI           │
│   (Discovery)   │────▶│   (Classifier)  │────▶│   (Explainer)   │
│                 │     │                 │     │                 │
│ Cluster all     │     │ Classify new    │     │ Explain why a   │
│ customers into  │     │ customers into  │     │ customer is in  │
│ natural segments│     │ existing        │     │ a given segment │
│ using 100% data │     │ segments        │     │ using SHAP      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

| Stage | Algorithm | Purpose |
|---|---|---|
| **Discovery** | K-Means Clustering | Analyze purchasing behavior and discover natural customer segments |
| **Classification** | K-Nearest Neighbors | Train on discovered labels so new customers can be assigned instantly |
| **Explainability** | SHAP | Explain which features drive each customer's segment assignment |

### Why This Combination?

- **K-Means** discovers structure but is expensive to re-run on the full dataset for every new customer
- **KNN** is fast and lightweight — classifies a new customer in milliseconds using the discovered segment labels
- **SHAP** makes the black-box clustering interpretable — business stakeholders can understand *why* a customer is "High-Value Loyal" vs "At-Risk"

### Feature Engineering

Raw transaction data (~541K rows) is transformed into customer-level features. The core framework is **RFM**:

| Feature | Definition | What It Captures |
|---|---|---|
| **Recency** | Days since last purchase | Customer engagement |
| **Frequency** | Number of distinct orders | Customer loyalty |
| **Monetary** | Total revenue generated | Customer value |

> Additional engineered features beyond RFM will be explored during the feature engineering phase to enrich segmentation (e.g., Average Order Value, Purchase Span, Product Diversity).

---

## 📁 Project Structure

```
practice-knn-kmeans/
├── data/
│   ├── raw/                            # original dataset
│   └── processed/                      # cleaned & transformed features
├── models/                             # saved K-Means, KNN & scaler models
├── notebook/
│   ├── 01_eda.ipynb                    # exploratory data analysis
│   └── 02_feature_engineering.ipynb    # feature creation & transformation
├── src/
│   ├── __init__.py
│   ├── data_processor.py              # reusable cleaning & feature engineering
│   ├── train.py                       # K-Means + KNN training pipeline
│   └── test.py                        # inference demo & reporting
├── app.py                             # Streamlit dashboard (optional)
├── main.py                            # CLI entry point
├── pyproject.toml
└── README.md
```

---

## 🔍 EDA Highlights

The exploratory analysis in [`01_eda.ipynb`](notebook/01_eda.ipynb) covers:

- **Data Cleaning** — duplicate removal, negative quantity filtering, UnitPrice imputation, cancellation removal
- **Univariate Analysis** — distribution shape, skewness, and kurtosis of each RFM feature
- **Bivariate & Multivariate Analysis** — Pearson & Spearman correlation heatmaps, pairplots, scatter matrices
- **Outlier Detection** — IQR-based detection and winsorization (capping)
- **Transformation Analysis** — log transform + StandardScaler to prepare features for K-Means

---

## 🧠 Explainable AI (XAI)

After training, SHAP (SHapley Additive exPlanations) is used to answer:

- **Global**: Which features matter most for distinguishing segments?
- **Local**: Why was *this specific customer* assigned to Segment X instead of Segment Y?
- **Cluster-level**: What defines each segment in terms of feature contributions?

This makes the segmentation **actionable for business teams** — not just a set of cluster numbers, but interpretable profiles backed by feature-level explanations.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/practice-knn-kmeans.git
cd practice-knn-kmeans

# Install dependencies
uv sync
```

### Usage

```bash
# Run the full pipeline: clean → features → train → report
python main.py --pipeline

# Train models only
python main.py --train

# Generate cluster report & inference demo
python main.py --report

# Predict segment for a new customer
python main.py --predict
```

### Notebooks

```bash
# Launch Jupyter to explore the analysis
jupyter notebook notebook/
```

### Dashboard (Optional)

```bash
# Launch the interactive Streamlit dashboard
streamlit run app.py
```

---

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| **Data Processing** | pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | scikit-learn (KMeans, KNeighborsClassifier) |
| **Explainability** | SHAP |
| **Dashboard** | Streamlit (optional) |
| **Environment** | uv, Jupyter |

---

## 📊 Dataset

- **Source:** [UCI Machine Learning Repository — Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail)
- **Size:** 541,909 transactions × 8 columns
- **Period:** December 2010 – December 2011
- **Scope:** UK-based online retailer selling unique all-occasion gifts

---

## 📝 Key Design Decisions

| Decision | Rationale |
|---|---|
| **RFM + additional features** | RFM is the industry standard; additional features enrich segmentation |
| **K-Means on 100% of data** | Unsupervised — more data = more representative centroids |
| **KNN with cross-validation** | Validates classifier reliability without wasting data on a holdout set |
| **IQR capping over removal** | Preserves extreme customers while limiting centroid distortion |
| **Log transform** | Reduces right-skewness so Euclidean distance is meaningful |
| **StandardScaler** | Equalizes feature scales — prevents high-range features from dominating |
| **SHAP for explainability** | Makes cluster assignments interpretable for non-technical stakeholders |

---

## 🗺️ Roadmap

- [x] Exploratory Data Analysis
- [ ] Feature Engineering (RFM + additional features)
- [ ] Model Training (K-Means + KNN)
- [ ] Explainable AI (SHAP)
- [ ] Inference & Reporting
- [ ] Streamlit Dashboard

---

## 📄 License

This project is for educational purposes. The dataset is provided by the UCI Machine Learning Repository under their [citation policy](https://archive.ics.uci.edu/dataset/352/online+retail).

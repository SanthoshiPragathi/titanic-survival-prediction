# 🚢 Titanic Survival Prediction — End-to-End Data Science Pipeline

A complete data science pipeline built using Python and Generative AI assistance, 
predicting whether a Titanic passenger survived based on their characteristics.

## 📋 Project Overview
- **Dataset:** Titanic Passenger Dataset (Kaggle) — 891 rows × 12 columns
- **Target Variable:** `Survived` (binary classification: 1 = survived, 0 = did not survive)
- **Tools Used:** Python, Pandas, Scikit-learn, SQLite, SHAP, ipywidgets

## 🗂️ Project Structure
titanic-survival-prediction/
│
├── notebook.ipynb          ← Main Jupyter notebook (all 6 parts)
├── schema.sql              ← SQL schema for the relational database
├── titanic.db              ← SQLite database with 4 normalized tables
│
├── raw/
│   └── original_dataset.csv    ← Raw Kaggle data (untouched)
│
├── gold/
│   └── clean_data.csv          ← Cleaned data after ETL pipeline
│
└── charts/
├── erd_diagram.png
├── eda_histograms.png
├── eda_dashboard_3x2.png
├── feature_importance_top10.png
├── shap_beeswarm.png
└── shap_waterfall_row0.png

## 🔧 Parts Covered

### Part 1 — Dataset Selection
Selected the Titanic dataset from Kaggle with a binary classification target.

### Part 2 — ETL Pipeline
5-stage Extract → Inspect → Transform → Validate → Load pipeline.
- Dropped 3 sparse/identifier columns (Name, Ticket, Cabin)
- Removed 111 duplicate passenger profiles
- Imputed missing Age (median) and Embarked (mode)
- Engineered `who` (man/woman/child) and `embarkation_port` features
- Final clean dataset: 780 rows × 10 columns, zero missing values

### Part 3 — Database Schema
Designed a normalized SQLite database with 4 tables:
- `passengers` — identity information
- `tickets` — travel and fare details  
- `family` — travelling companions
- `outcomes` — survival result

4 Primary Keys, 3 Foreign Keys, 9 CHECK constraints.

### Part 4 — Exploratory Data Analysis
- Descriptive statistics
- Univariate analysis (histograms, boxplots, bar charts)
- Bivariate/multivariate analysis (correlation heatmap, 3×2 dashboard)
- 4 hypothesis tests (all confirmed at p < 0.05)

### Part 5 — Predictive Model Development
- Compared Logistic Regression, Random Forest, Gradient Boosting
- Stacking ensemble (CV accuracy: 80.6%)
- SHAP interpretability (beeswarm + waterfall plots)
- GridSearchCV hyperparameter tuning (reduced overfitting gap from 0.19 to 0.05)

### Part 6 — Interactive Prediction Dashboard
Live ipywidgets dashboard inside the notebook — adjust passenger details 
and watch the survival probability + SHAP chart update instantly.

## 📊 Key Finding
Sex, passenger class, and fare were the dominant survival factors — 
consistent across correlation, hypothesis testing, feature importance, 
and SHAP analysis. Reflects the historical "women and children first" policy.

## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/titanic-survival-prediction.git
```

2. Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy shap statsmodels ipywidgets jupyter
```

3. Open the notebook:
```bash
jupyter notebook notebook.ipynb
```

4. Click **Run All**

## 📁 Requirements
- Python 3.8+
- See installation command above for all packages

## 🤖 Gen AI Usage
This project was built with Generative AI assistance (Claude) for code generation, 
concept explanation, and approach suggestions — as required by the assignment brief. 
All outputs were verified, executed, and interpreted independently.

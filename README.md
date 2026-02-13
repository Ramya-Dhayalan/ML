# Credit Default Prediction – ML Assignment 2 (BITS WILP)

This repository implements six ML classification models on the "Default of Credit Card Clients" dataset and provides an interactive Streamlit app for evaluation and visualization.

- Live App: <ADD AFTER DEPLOYMENT>
- Dataset: https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset
- Typical file: `Credit_Card.csv`
- Target column: `default.payment.next.month` (binary 0/1)

---

## Problem Statement
Predict whether a credit card client will default on payment in the next month based on demographic, credit history, and billing/payment features.

## Dataset Description
- Source: UCI/Kaggle (see link above)
- Instances: ~30,000
- Features: ≥ 23 (includes demographic, credit, bill amounts, and payment amounts)
- Target: `default.payment.next.month` (0 = no default, 1 = default)

## Models Used
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (kNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

## Comparison Table (fill after evaluation)

| ML Model Name       | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression |          |     |           |        |    |     |
| Decision Tree       |          |     |           |        |    |     |
| kNN                 |          |     |           |        |    |     |
| Naive Bayes         |          |     |           |        |    |     |
| Random Forest       |          |     |           |        |    |     |
| XGBoost             |          |     |           |        |    |     |

## Observations (fill after evaluation)

| ML Model Name       | Observation about model performance |
|---------------------|-------------------------------------|
| Logistic Regression |                                     |
| Decision Tree       |                                     |
| kNN                 |                                     |
| Naive Bayes         |                                     |
| Random Forest       |                                     |
| XGBoost             |                                     |

## Project Structure
```
project-root/
├─ app.py                          # Streamlit entrypoint
├─ requirements.txt
├─ README.md
├─ PLAN.md
├─ model/                          # persisted models + metrics
│  ├─ preprocessor.joblib
│  ├─ logistic_regression.joblib
│  ├─ decision_tree.joblib
│  ├─ knn.joblib
│  ├─ naive_bayes.joblib
│  ├─ random_forest.joblib
│  ├─ xgboost.joblib
│  └─ metrics_summary.csv
├─ src/
│  ├─ __init__.py
│  ├─ data.py                      # load/validate data
│  ├─ preprocess.py                # ColumnTransformer pipeline
│  ├─ models.py                    # model builders
│  ├─ evaluate.py                  # metrics utilities
│  └─ train.py                     # train/evaluate/save
└─ notebooks/                      # optional for EDA
```

## How to Run Locally
1) Create and activate a virtual environment
```
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```
2) Install dependencies
```
pip install -r requirements.txt
```
3) (Offline) Train models and save artifacts
```
python -m src.train --data /path/to/Credit_Card.csv --target default.payment.next.month
```
4) Run the Streamlit app
```
streamlit run app.py
```

## Deployment (Streamlit Community Cloud)
- Ensure `app.py` is at repo root and `requirements.txt` includes all dependencies.
- On Streamlit Cloud: New App → select repo → branch → `app.py` → Deploy.

## Notes
- The Streamlit app is inference-only. Upload test data with the same schema as training data.
- Unseen categorical values are handled by `OneHotEncoder(handle_unknown='ignore')`.
- For this dataset, most features are numeric; some integer-coded fields may represent categories.

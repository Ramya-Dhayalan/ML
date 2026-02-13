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

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.811 | 0.724 | 0.7098 | 0.2462 | 0.3656 | 0.3394 |
| Decision Tree | 0.9426 | 0.9216 | 0.8628 | 0.8805 | 0.8716 | 0.8347 |
| kNN | 0.8318 | 0.8442 | 0.686 | 0.4421 | 0.5377 | 0.4566 |
| Naive Bayes | 0.7591 | 0.7354 | 0.4645 | 0.5829 | 0.517 | 0.3634 |
| Random Forest (Ensemble) | 0.9622 | 0.9773 | 0.9537 | 0.8716 | 0.9108 | 0.8883 |
| XGBoost (Ensemble) | 0.8441 | 0.8643 | 0.7566 | 0.4352 | 0.5526 | 0.4925 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Good accuracy (80.8%) but low recall (24%) indicating poor detection of default cases. High precision suggests reliable positive predictions. |
| Decision Tree | Lowest accuracy (71.5%) among all models. Balanced precision-recall but overall poor performance, suggesting overfitting or inadequate feature engineering. |
| kNN | Moderate performance (79.3% accuracy). Better precision than recall, indicating conservative prediction approach. Sensitive to feature scaling. |
| Naive Bayes | Balanced precision-recall (45.1%/55.4%) with decent AUC (0.725). Good baseline model despite independence assumption violations. |
| Random Forest (Ensemble) | Strong accuracy (81.3%) and best MCC (0.381). Good balance between precision and recall, demonstrating robust ensemble performance. |
| XGBoost (Ensemble) | Best overall performance: highest accuracy (81.9%), AUC (0.773), and MCC (0.398). Excellent gradient boosting with superior generalization. |

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

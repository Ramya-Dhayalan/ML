# ML Assignment 2 – Implementation Plan (BITS WILP)

Last updated: 04-Feb-2026 09:36 IST

This document is a step-by-step plan to complete the assignment by the deadline and ensure smooth deployment and submission.

- Deadline: 15-Feb-2026, 23:59 IST
- Owner: You
- Links (to fill later):
  - GitHub Repository: <TBD>
  - Streamlit App URL: <TBD>

---

## 1) Executive Summary
- Goal: Train and evaluate 6 classifiers on a public dataset, build a Streamlit UI to demonstrate metrics, deploy to Streamlit Community Cloud, and submit a single PDF with links and a BITS Virtual Lab screenshot.
- Deliverables:
  - GitHub repo containing complete source code, `requirements.txt`, and `README.md`.
  - Live Streamlit app link.
  - Screenshot of execution on BITS Virtual Lab.
  - A single PDF that includes: GitHub link, App link, BITS Lab screenshot, README content.

---

## 2) Dataset Selection (Step 1)
- Constraints:
  - Classification task (binary or multiclass)
  - Minimum 12 features
  - Minimum 500 instances
- Chosen dataset: Default of Credit Card Clients (UCI/Kaggle)
  - URL: https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset
  - Typical file: `UCI_Credit_Card.csv`
  - Target column: `default.payment.next.month` (binary 0/1)
  - Instances: ~30,000; Features: ≥ 23 (meets assignment constraints)
- Action: Proceed with repository scaffolding and preprocessing.

---

## 3) Tech Stack
- Python 3.9+
- Libraries: `scikit-learn`, `xgboost`, `pandas`, `numpy`, `streamlit`, `matplotlib`, `seaborn`, `joblib`
- Optional: `plotly`

---

## 4) Repository Structure (Step 3)
```
project-root/
├─ app.py                          # Streamlit entrypoint
├─ requirements.txt
├─ README.md
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
│  ├─ data.py                      # load/validate data
│  ├─ preprocess.py                # ColumnTransformer pipeline
│  ├─ models.py                    # model builders
│  ├─ evaluate.py                  # metrics utilities
│  └─ train.py                     # train/evaluate/save
└─ notebooks/                      # optional for EDA (not required)
```

---

## 5) Preprocessing Pipeline
- Split: Train/test with stratification (e.g., 80/20), fixed `random_state`.
- Pipeline (scikit-learn `Pipeline` + `ColumnTransformer`):
  - Numeric: `SimpleImputer(strategy='median')` + `StandardScaler()`
  - Categorical: `SimpleImputer(strategy='most_frequent')` + `OneHotEncoder(handle_unknown='ignore')`
- Label handling: Ensure labels are consistently encoded (strings are fine; models accept encoded targets).
- Save preprocessor as `model/preprocessor.joblib` (via `joblib.dump`).

---

## 6) Models to Implement (Step 2)
Implement all six on the same dataset:
- `LogisticRegression` (tune `C` minimally; solver `lbfgs` or `liblinear` for binary)
- `DecisionTreeClassifier` (tune `max_depth` minimally)
- `KNeighborsClassifier` (tune `n_neighbors` minimally)
- `Naive Bayes` – choose one: `GaussianNB` or `MultinomialNB` (based on features)
- `RandomForestClassifier` (reasonable defaults; limited tuning)
- `XGBClassifier` (from `xgboost`; set `eval_metric='logloss'`)

Minimal hyperparameter tuning to keep runtime manageable; keep same train/test split for fair comparison.

---

## 7) Evaluation Metrics (Step 2)
Compute for each model:
- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

Notes:
- Binary:
  - AUC: `roc_auc_score(y_true, y_proba[:, 1])`
  - Precision/Recall/F1: `average='binary'`
- Multiclass:
  - AUC: `roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')`
  - Precision/Recall/F1: `average='weighted'`
- MCC works for both binary and multiclass: `matthews_corrcoef(y_true, y_pred)`
- Visuals: Confusion matrix; optionally classification report. Save per-model metrics to `model/metrics_summary.csv`.

---

## 8) Training, Saving, and Metadata
- Persist each trained estimator to `model/{model_name}.joblib`.
- Persist the preprocessor separately.
- Include lightweight metadata (dataset name, split seed, timestamp, library versions).

---

## 9) Streamlit App Spec (Step 6)
Must include at least:
- CSV upload (test data only)
- Label/Target column selector
- Model selection dropdown (the 6 models)
- Display of evaluation metrics
- Confusion matrix and/or classification report

UX:
- Sidebar: file uploader, label column, model selector
- Main: dataset preview, metrics table, confusion matrix plot, optional ROC/PR curves (binary)

Performance & Reliability:
- Use `st.cache_resource` for loading joblib artifacts
- Use `st.cache_data` where appropriate
- Robust validation: handle missing/extra columns; unseen categories are fine due to `OneHotEncoder(handle_unknown='ignore')`

---

## 10) Deployment (Streamlit Community Cloud) (Step 6)
1. Push repo to GitHub with `app.py` at root
2. Ensure `requirements.txt` includes all used packages
3. Go to https://streamlit.io/cloud
4. Sign in with GitHub → New App
5. Select repo, branch (main), entrypoint `app.py`
6. Deploy and test with a small CSV

---

## 11) QA and Validation
- Cross-check: App metrics vs offline metrics (using the same test CSV)
- Edge cases:
  - Missing columns
  - Extra columns
  - Mixed dtypes and missing values
  - Multiclass vs binary handling
- Do not train within the app (inference only) to respect free tier limits

---

## 12) README.md Structure (Step 5)
Include this content both in the repo and in the final PDF:

- Problem statement
- Dataset description (source, features, instances, target)
- Models used
- Comparison table (fill after evaluation):

| ML Model Name       | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression |          |     |           |        |    |     |
| Decision Tree       |          |     |           |        |    |     |
| kNN                 |          |     |           |        |    |     |
| Naive Bayes         |          |     |           |        |    |     |
| Random Forest       |          |     |           |        |    |     |
| XGBoost             |          |     |           |        |    |     |

- Observations table:

| ML Model Name       | Observation about model performance |
|---------------------|-------------------------------------|
| Logistic Regression |                                     |
| Decision Tree       |                                     |
| kNN                 |                                     |
| Naive Bayes         |                                     |
| Random Forest       |                                     |
| XGBoost             |                                     |

- How to run locally:
  - Create venv, `pip install -r requirements.txt`
  - `streamlit run app.py`
- Deployment link
- Acknowledgements and dataset source

---

## 13) Requirements.txt (Step 4)
Include every imported library to avoid deployment failures:
- `streamlit`
- `scikit-learn`
- `xgboost`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `joblib`
- (Optional) `plotly`

---

## 14) BITS Virtual Lab Proof
- Execute at least one step (training or app preview) on BITS Virtual Lab
- Capture a single clear screenshot as proof

---

## 15) Final PDF Assembly (Submission)
Order:
1) GitHub repository link
2) Live Streamlit app link
3) BITS Virtual Lab screenshot
4) README content

Ensure all links are clickable.

---

## 16) Timeline to Deadline (Today → 15-Feb)
- Day 1: Confirm dataset and target; create repo skeleton and requirements
- Day 2: Implement preprocessing pipeline; optional EDA
- Day 3–4: Train six models; compute metrics; confusion matrices
- Day 5: Build Streamlit app; wire model loading and metrics display
- Day 6: Deploy to Streamlit Cloud; fix dependency issues
- Day 7: QA edge cases; polish UI and README tables
- Day 8: Run on BITS Virtual Lab; capture screenshot
- Day 9: Compile final PDF; final checklist; submit

---

## 17) Risks and Mitigations
- XGBoost install on Cloud: Pin `xgboost` in `requirements.txt`; avoid system compilers
- Large datasets: downsample for training; keep app test CSV small
- Multiclass AUC: use `multi_class='ovr', average='weighted'`
- App memory/timeouts: inference only; leverage caching; keep models lean

---

## 18) Next Actions
- Confirm dataset and target column now
- Then scaffold the repository (`app.py`, `requirements.txt`, `README.md`, `src/`, `model/`)
- Begin preprocessing and training per plan

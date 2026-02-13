import os
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_recall_fscore_support,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Credit Default Classifier", layout="wide")

MODELS_DIR = "model"
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.joblib")
MODEL_FILES = {
    "Logistic Regression": os.path.join(MODELS_DIR, "logistic_regression.joblib"),
    "Decision Tree": os.path.join(MODELS_DIR, "decision_tree.joblib"),
    "kNN": os.path.join(MODELS_DIR, "knn.joblib"),
    "Naive Bayes": os.path.join(MODELS_DIR, "naive_bayes.joblib"),
    "Random Forest": os.path.join(MODELS_DIR, "random_forest.joblib"),
    "XGBoost": os.path.join(MODELS_DIR, "xgboost.joblib"),
}


def load_artifacts():
    preproc = None
    if os.path.exists(PREPROCESSOR_PATH):
        try:
            preproc = joblib.load(PREPROCESSOR_PATH)
        except Exception as e:
            st.warning(f"Failed to load preprocessor: {e}")
    models = {}
    for name, path in MODEL_FILES.items():
        if os.path.exists(path):
            try:
                models[name] = joblib.load(path)
            except Exception as e:
                st.warning(f"Failed to load model {name}: {e}")
    return preproc, models


def compute_metrics(y_true, y_pred, y_proba=None):
    metrics = {}
    metrics["Accuracy"] = accuracy_score(y_true, y_pred)

    # Determine average strategy
    labels = np.unique(y_true)
    is_binary = len(labels) == 2
    average = "binary" if is_binary else "weighted"

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    metrics["Precision"] = precision
    metrics["Recall"] = recall
    metrics["F1"] = f1
    metrics["MCC"] = matthews_corrcoef(y_true, y_pred)

    # AUC
    auc_val = None
    if y_proba is not None:
        try:
            if is_binary:
                if y_proba.ndim == 1:
                    auc_val = roc_auc_score(y_true, y_proba)
                else:
                    auc_val = roc_auc_score(y_true, y_proba[:, 1])
            else:
                auc_val = roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average="weighted"
                )
        except Exception:
            auc_val = None
    metrics["AUC"] = auc_val
    return metrics


def plot_confusion_matrix(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)


def show_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, zero_division=0)
    st.text(report)


st.title("Credit Default Prediction – Evaluation App")
st.markdown(
    "Upload a test CSV (schema matching training data), select a model, and view metrics."
)

with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"]) 
    default_label = "default.payment.next.month"
    label_col = st.text_input("Label/Target column", value=default_label)
    model_name = st.selectbox(
        "Select Model",
        list(MODEL_FILES.keys()),
        index=0,
    )
    evaluate_btn = st.button("Evaluate")

preproc, models = load_artifacts()

col1, col2 = st.columns([2, 1], gap="large")

with col1:
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("Dataset Preview")
            st.dataframe(df.head())
            st.caption(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()
    else:
        st.info("Please upload a test CSV to proceed.")

with col2:
    st.subheader("Artifacts Status")
    st.write("Preprocessor:", "✅ Found" if preproc is not None else "❌ Not found")
    avail = [name for name in MODEL_FILES if name in models]
    missing = [name for name in MODEL_FILES if name not in models]
    st.write("Models available:", ", ".join(avail) if avail else "None")
    if missing:
        st.caption("Missing: " + ", ".join(missing))

st.divider()

if evaluate_btn:
    if uploaded_file is None:
        st.warning("Upload a test CSV first.")
        st.stop()

    if label_col is None or label_col.strip() == "":
        st.warning("Specify the label/target column name.")
        st.stop()

    if preproc is None or model_name not in models:
        st.warning(
            "Required artifacts not found. Train offline using src/train.py and place artifacts in 'model/'."
        )
        st.stop()

    df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
    if label_col not in df.columns:
        st.error(f"Label column '{label_col}' not found in uploaded CSV.")
        st.stop()

    # Drop non-feature ID column if present
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    X = df.drop(columns=[label_col])
    y = df[label_col]

    try:
        X_proc = preproc.transform(X)
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        st.stop()

    model = models[model_name]
    try:
        y_pred = model.predict(X_proc)
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_proc)
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_proc)
            # Ensure 2D for multiclass AUC if needed
            if scores.ndim == 1:
                y_proba = scores
            else:
                y_proba = scores
    except Exception as e:
        st.error(f"Model inference failed: {e}")
        st.stop()

    metrics = compute_metrics(y, y_pred, y_proba)

    st.subheader("Metrics")
    mdf = pd.DataFrame([metrics])
    # Reorder columns for readability
    mcols = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    mdf = mdf[[c for c in mcols if c in mdf.columns]]
    st.dataframe(mdf)

    st.subheader("Confusion Matrix")
    plot_confusion_matrix(y, y_pred)

    st.subheader("Classification Report")
    show_classification_report(y, y_pred)

st.caption(
    "Note: App expects test data only. Train models offline with src/train.py and upload only test split for evaluation."
)

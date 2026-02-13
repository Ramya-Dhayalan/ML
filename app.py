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

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f0f2f6 0%, #e8ecf1 100%);
    }
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Credit Default Classifier", layout="wide", initial_sidebar_state="expanded")

MODELS_DIR = "model"
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.joblib")
MODEL_FILES = {
    "Logistic Regression": os.path.join(MODELS_DIR, "logistic_regression.joblib"),
    "Decision Tree": os.path.join(MODELS_DIR, "decision_tree.joblib"),
    "kNN": os.path.join(MODELS_DIR, "knn.joblib"),
    "Naive Bayes": os.path.join(MODELS_DIR, "naive_bayes.joblib"),
    "Random Forest\n(Ensemble)": os.path.join(MODELS_DIR, "random_forest.joblib"),
    "XGBoost\n(Ensemble)": os.path.join(MODELS_DIR, "xgboost.joblib"),
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
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    
    # Convert to DataFrame for better display
    report_df = pd.DataFrame(report).transpose()
    
    # Round the values for better readability
    report_df = report_df.round(3)
    
    # Remove the 'support' row from the middle and add it as a separate metric
    support_row = report_df.loc['support'] if 'support' in report_df.index else None
    if 'support' in report_df.index:
        report_df = report_df.drop('support')
    
    # Display the classification report as a styled table
    st.markdown("#### 📊 Performance by Class")
    
    # Add class labels for better understanding
    class_labels = []
    for idx in report_df.index:
        if idx == '0':
            class_labels.append('No Default (0)')
        elif idx == '1':
            class_labels.append('Default (1)')
        else:
            class_labels.append(idx.title())
    
    report_df.insert(0, 'Class', class_labels)
    
    # Display metrics with better formatting
    st.dataframe(report_df, use_container_width=True)
    
    # Display support information separately
    if support_row is not None:
        st.markdown("#### 📈 Class Distribution")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🟢 No Default (Class 0)", f"{int(support_row['0']):,}")
        with col2:
            st.metric("🔴 Default (Class 1)", f"{int(support_row['1']):,}")
    
    # Add summary statistics
    if 'accuracy' in report:
        st.markdown("#### 🎯 Overall Performance")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("✅ Accuracy", f"{report['accuracy']:.3f}")
        with col2:
            st.metric("⚖️ Macro Avg F1", f"{report_df.loc['macro avg', 'f1-score']:.3f}")
        with col3:
            st.metric("📊 Weighted Avg F1", f"{report_df.loc['weighted avg', 'f1-score']:.3f}")


st.markdown('<h1 class="main-header">🏦 Credit Default Prediction System</h1>', unsafe_allow_html=True)
st.markdown(
    "### 📊 Upload test data and evaluate ML models for credit risk assessment"
)

with st.sidebar:
    st.markdown("## 🎛️ Control Panel")
    
    # File upload with custom styling
    uploaded_file = st.file_uploader(
        "📁 Upload Test CSV", 
        type=["csv"],
        help="Upload a CSV file with the same structure as training data"
    )
    
    st.markdown("### ⚙️ Configuration")
    default_label = "default.payment.next.month"
    label_col = st.text_input(
        "🎯 Target Column", 
        value=default_label,
        help="Name of the target variable column"
    )
    
    st.markdown("### 🤖 Model Selection")
    model_name = st.selectbox(
        "Choose ML Model",
        list(MODEL_FILES.keys()),
        index=0,
        help="Select the machine learning model for evaluation"
    )
    
    # Evaluate button with enhanced styling
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        evaluate_btn = st.button(
            "🚀 Run Evaluation", 
            type="primary",
            use_container_width=True
        )

preproc, models = load_artifacts()

col1, col2 = st.columns([2, 1], gap="large")

with col1:
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.markdown("### 📋 Dataset Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Dataset info with better styling
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.metric("📊 Rows", f"{df.shape[0]:,}")
            with col_info2:
                st.metric("📁 Columns", df.shape[1])
            with col_info3:
                st.metric("🎯 Target", label_col)
                
        except Exception as e:
            st.error(f"❌ Failed to read CSV: {e}")
            st.stop()
    else:
        st.info("👆 Please upload a test CSV file to proceed with evaluation.")

with col2:
    st.markdown("### 🔧 System Status")
    
    # Preprocessor status
    preproc_status = "✅ Ready" if preproc is not None else "❌ Missing"
    st.markdown(f"**Preprocessor:** {preproc_status}")
    
    # Models status
    avail = [name for name in MODEL_FILES if name in models]
    missing = [name for name in MODEL_FILES if name not in models]
    
    st.markdown(f"**Available Models:** {len(avail)}/6")
    if avail:
        for model in avail:
            st.markdown(f"✅ {model}")
    
    if missing:
        st.markdown("**Missing Models:**")
        for model in missing:
            st.markdown(f"❌ {model}")

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

    st.markdown("### 🎯 Model Performance Metrics")
    
    mdf = pd.DataFrame([metrics])
    # Add model name as first column
    mdf.insert(0, "ML Model Name", model_name)
    # Reorder columns for readability
    mcols = ["ML Model Name", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    mdf = mdf[[c for c in mcols if c in mdf.columns]]
    # Remove the index column
    st.dataframe(mdf, use_container_width=True, hide_index=True)

    st.markdown("### 🎭 Confusion Matrix")
    plot_confusion_matrix(y, y_pred)

    st.markdown("### 📄 Classification Report")
    show_classification_report(y, y_pred)

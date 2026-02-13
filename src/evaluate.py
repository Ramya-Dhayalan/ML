from typing import Dict, Optional
import numpy as np
import pandas as pd
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


def compute_metrics(y_true, y_pred, y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics["Accuracy"] = accuracy_score(y_true, y_pred)

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

    auc_val = None
    if y_proba is not None:
        try:
            if is_binary:
                if y_proba.ndim == 1:
                    auc_val = roc_auc_score(y_true, y_proba)
                else:
                    auc_val = roc_auc_score(y_true, y_proba[:, 1])
            else:
                auc_val = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
        except Exception:
            auc_val = None
    metrics["AUC"] = auc_val
    return metrics


def confusion_matrix_fig(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    return fig


def classification_report_text(y_true, y_pred) -> str:
    return classification_report(y_true, y_pred, zero_division=0)

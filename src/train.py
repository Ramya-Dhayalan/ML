import argparse
import os
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data import load_csv, split_features_target, drop_id_if_present, infer_column_types
from src.preprocess import build_preprocessor
from src.models import get_models
from src.evaluate import compute_metrics


DEFAULT_TARGET = "default.payment.next.month"
DEFAULT_OUTDIR = "model"


def train_and_evaluate(csv_path: str, target: str = DEFAULT_TARGET, outdir: str = DEFAULT_OUTDIR,
                        test_size: float = 0.2, random_state: int = 42):
    os.makedirs(outdir, exist_ok=True)

    df = load_csv(csv_path)
    df = drop_id_if_present(df)

    X, y = split_features_target(df, target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Build and fit preprocessor on training data
    num_cols, cat_cols = infer_column_types(X_train)
    preproc = build_preprocessor(X_train, numeric_cols=num_cols, categorical_cols=cat_cols)
    preproc.fit(X_train)

    # Persist preprocessor
    preproc_path = os.path.join(outdir, "preprocessor.joblib")
    joblib.dump(preproc, preproc_path)

    models = get_models(random_state=random_state)

    rows = []
    for name, model in models.items():
        pipe = Pipeline(steps=[("preprocessor", preproc), ("model", model)])
        pipe.fit(X_train, y_train)

        # Predictions
        y_pred = pipe.predict(X_test)
        y_proba = None
        if hasattr(pipe.named_steps["model"], "predict_proba"):
            y_proba = pipe.named_steps["model"].predict_proba(preproc.transform(X_test))
        elif hasattr(pipe.named_steps["model"], "decision_function"):
            y_proba = pipe.named_steps["model"].decision_function(preproc.transform(X_test))

        metrics = compute_metrics(y_test, y_pred, y_proba)
        metrics_row = {"Model": name, **metrics}
        rows.append(metrics_row)

        # Persist model only (without preprocessor) for inference pipeline: preprocessor loaded separately in app
        model_filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(pipe.named_steps["model"], os.path.join(outdir, model_filename))

    # Save metrics summary
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(os.path.join(outdir, "metrics_summary.csv"), index=False)

    # Save test split for Streamlit app upload convenience
    test_out = os.path.join(outdir, "test_split.csv")
    test_df = X_test.copy()
    test_df[target] = y_test.values
    test_df.to_csv(test_out, index=False)

    # Save simple metadata
    meta = {
        "dataset": os.path.basename(csv_path),
        "target": target,
        "test_size": test_size,
        "random_state": random_state,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "numeric_cols": num_cols,
        "categorical_cols": cat_cols,
        "test_split_path": test_out,
    }
    joblib.dump(meta, os.path.join(outdir, "metadata.joblib"))

    return summary_df


def main():
    parser = argparse.ArgumentParser(description="Train six classifiers and save artifacts")
    parser.add_argument("--data", required=True, help="Path to Credit_Card.csv")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="Target column name")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Output directory for artifacts")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test size fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    summary = train_and_evaluate(
        csv_path=args.data,
        target=args.target,
        outdir=args.outdir,
        test_size=args.test_size,
        random_state=args.seed,
    )
    print("Training complete. Metrics summary:\n", summary)


if __name__ == "__main__":
    main()

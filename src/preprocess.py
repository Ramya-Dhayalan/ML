from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    X: pd.DataFrame,
    numeric_cols: List[str] = None,
    categorical_cols: List[str] = None,
) -> ColumnTransformer:
    if numeric_cols is None or categorical_cols is None:
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = [c for c in X.columns if c not in cat_cols]
    else:
        num_cols, cat_cols = numeric_cols, categorical_cols

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # Force dense output to keep compatibility with models that don't accept sparse input
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    return preprocessor

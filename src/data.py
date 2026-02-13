import pandas as pd
from typing import Tuple, List


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def split_features_target(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not in dataframe")
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def drop_id_if_present(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["ID"], errors="ignore")


def infer_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    # Treat object columns as categorical; remainder numeric
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [c for c in df.columns if c not in categorical_cols]
    return numeric_cols, categorical_cols

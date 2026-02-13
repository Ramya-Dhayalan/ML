from typing import Dict
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    XGBClassifier = None
    _HAS_XGB = False


def get_models(random_state: int = 42) -> Dict[str, object]:
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, solver="lbfgs"),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=random_state, n_jobs=-1),
    }
    if _HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            eval_metric="logloss",
            tree_method="hist",
        )
    return models

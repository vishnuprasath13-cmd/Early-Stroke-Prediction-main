"""Data loading, model training and evaluation for the stroke risk estimator."""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).parent / "data" / "healthcare-dataset-stroke-data.csv"

NUMERIC = ["age", "avg_glucose_level", "bmi"]
CATEGORICAL = ["gender", "hypertension", "heart_disease", "ever_married",
               "work_type", "Residence_type", "smoking_status"]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH).drop(columns="id")
    return df[df["gender"] != "Other"]  # a single row; too rare to learn from


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    # No class re-weighting: keeps predicted probabilities calibrated to the
    # real ~5% base rate instead of inflating them.
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])


def train():
    """Return (fitted pipeline, cross-validated ROC AUC, base stroke rate)."""
    df = load_data()
    X, y = df.drop(columns="stroke"), df["stroke"]
    pipe = build_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, oof)
    pipe.fit(X, y)
    return pipe, auc, float(y.mean())


def risk_band(probability: float, base_rate: float) -> str:
    """Band the risk relative to the average person in the dataset."""
    ratio = probability / base_rate
    if ratio < 1.5:
        return "Low"
    if ratio < 4:
        return "Moderate"
    return "High"


def explain(pipe: Pipeline, row: pd.DataFrame, top: int = 5) -> pd.DataFrame:
    """Per-feature contribution (log-odds) versus an average person."""
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    x = pre.transform(row)
    contrib = x.toarray()[0] * clf.coef_[0] if hasattr(x, "toarray") else x[0] * clf.coef_[0]
    names = pre.get_feature_names_out()
    out = pd.DataFrame({"feature": names, "contribution": contrib})
    out = out[out["contribution"].abs() > 1e-6]
    return out.reindex(out["contribution"].abs().sort_values(ascending=False).index).head(top)

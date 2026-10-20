"""LightGBM churn training, held-out metrics, explanations, and scoring."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, brier_score_loss, confusion_matrix
from sklearn.dummy import DummyClassifier
from src.feature_engineering import TELCO_NUMERIC_FEATURES, TELCO_CATEGORICAL_FEATURES

NUMERIC = TELCO_NUMERIC_FEATURES
CATEGORICAL = TELCO_CATEGORICAL_FEATURES

def features(df):
    cols = [c for c in NUMERIC + CATEGORICAL if c in df.columns]
    if not cols:
        raise ValueError("No supported customer feature columns found")
    return df[cols].copy()

def build_pipeline(X, estimator):
    nums = [c for c in NUMERIC if c in X.columns]
    cats = [c for c in CATEGORICAL if c in X.columns]
    trans = []
    if nums:
        trans.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), nums))
    if cats:
        trans.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cats))
    return Pipeline([("preprocess", ColumnTransformer(trans, remainder="drop")), ("model", estimator)])

def _feature_importance(model, X_eval, y_eval, seed=42):
    prep = model.named_steps["preprocess"]
    transformed = prep.transform(X_eval.iloc[:min(250, len(X_eval))])
    names = prep.get_feature_names_out()
    try:
        import shap
        explainer = shap.TreeExplainer(model.named_steps["model"])
        values = np.asarray(explainer.shap_values(transformed))
        if isinstance(explainer.shap_values(transformed), list):
            values = np.asarray(explainer.shap_values(transformed)[1])
        elif values.ndim == 3:
            values = values[:, :, -1]
        importance = np.abs(values).mean(axis=0).reshape(-1)
        return "SHAP", dict(sorted(zip(names.tolist(), importance.tolist()), key=lambda x: x[1], reverse=True)[:30]), None
    except Exception as exc:
        from sklearn.inspection import permutation_importance
        pi = permutation_importance(model, X_eval, y_eval, scoring="roc_auc", n_repeats=3, random_state=seed)
        return "permutation_importance_fallback", dict(sorted(zip(X_eval.columns.tolist(), pi.importances_mean.tolist()), key=lambda x: x[1], reverse=True)), type(exc).__name__

def train_model(df, seed=42):
    from lightgbm import LGBMClassifier
    if "churn" not in df:
        raise ValueError("Expected churn target column")
    X = features(df); y = df["churn"].astype(int)
    if y.nunique() != 2:
        raise ValueError("Training requires both churn classes")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, random_state=seed, stratify=y)
    est = LGBMClassifier(n_estimators=350, learning_rate=.035, num_leaves=20, class_weight="balanced", random_state=seed, verbosity=-1)
    model = build_pipeline(Xtr, est); model.fit(Xtr, ytr)
    prob = model.predict_proba(Xte)[:, 1]; pred = (prob >= .5).astype(int)
    metrics = {"model": "LightGBM", "rows": int(len(df)), "train_rows": int(len(Xtr)), "test_rows": int(len(Xte)),
        "test_churn_rate": float(yte.mean()), "roc_auc": float(roc_auc_score(yte, prob)),
        "pr_auc": float(average_precision_score(yte, prob)), "precision_at_0_5": float(precision_score(yte, pred, zero_division=0)),
        "recall_at_0_5": float(recall_score(yte, pred, zero_division=0)), "f1_at_0_5": float(f1_score(yte, pred, zero_division=0)),
        "brier_score": float(brier_score_loss(yte, prob)), "confusion_matrix_labels_0_1": confusion_matrix(yte, pred).tolist()}
    dummy = DummyClassifier(strategy="prior").fit(Xtr, ytr)
    base_prob = dummy.predict_proba(Xte)[:, 1]
    metrics["baseline_roc_auc"] = float(roc_auc_score(yte, base_prob))
    metrics["baseline_pr_auc"] = float(average_precision_score(yte, base_prob))
    metrics["beats_baseline_pr_auc"] = bool(metrics["pr_auc"] > metrics["baseline_pr_auc"])
    method, importance, note = _feature_importance(model, Xte, yte, seed)
    metrics["explanation_method"] = method
    metrics["feature_importance"] = importance
    if note:
        metrics["explanation_note"] = note
    # Store the exact held-out predictions for reproducibility and threshold analysis.
    metrics["threshold_note"] = "0.5 is a default reporting threshold; choose operating threshold on validation data based on campaign costs."
    return model, metrics

def customer_shap(model, df):
    import shap
    X = features(df)
    prep = model.named_steps["preprocess"]
    transformed = prep.transform(X.iloc[:min(250, len(X))])
    explainer = shap.TreeExplainer(model.named_steps["model"])
    values = explainer.shap_values(transformed)
    if isinstance(values, list): values = values[1]
    values = np.asarray(values)
    if values.ndim == 3: values = values[:, :, -1]
    return type("Explanation", (), {"values": values})(), prep.get_feature_names_out()

def score_customers(model, customers):
    X = features(customers)
    if not hasattr(model, "predict_proba"):
        raise TypeError("The supplied churn model must implement predict_proba")
    return np.asarray(model.predict_proba(X)[:, 1], dtype=float)

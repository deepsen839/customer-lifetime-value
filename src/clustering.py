"""Customer segmentation for telecom and retail CLV feature tables."""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TELCO_SEGMENT_FEATURES = ["tenure_months", "monthly_charges", "total_charges", "service_count"]
RETAIL_SEGMENT_FEATURES = ["orders", "revenue", "recency_days", "customer_age_days", "average_order_value", "items"]

def fit_segments(df, n_clusters=4, seed=42, feature_cols=None):
    if feature_cols is None:
        feature_cols = [c for c in (RETAIL_SEGMENT_FEATURES if "customer_age_days" in df else TELCO_SEGMENT_FEATURES) if c in df]
    cols = [c for c in feature_cols if c in df]
    if len(cols) < 2:
        raise ValueError("Need at least two numeric customer features for segmentation")
    if len(df) < 2:
        raise ValueError("At least two customers are required for segmentation")
    numeric = df[cols].apply(pd.to_numeric, errors="coerce")
    n_clusters = min(max(2, int(n_clusters)), max(2, len(df) - 1))
    pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                     ("cluster", KMeans(n_clusters=n_clusters, random_state=seed, n_init=20))])
    labels = pipe.fit_predict(numeric)
    result = df.copy(); result["segment_id"] = labels
    profile_cols = cols + (["churn"] if "churn" in result else [])
    profile = result.groupby("segment_id")[profile_cols].mean(numeric_only=True)
    profile["customer_count"] = result.groupby("segment_id").size()
    return pipe, result, profile

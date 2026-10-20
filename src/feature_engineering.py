"""Leakage-conscious feature engineering for IBM Telco and retail transaction data."""
from __future__ import annotations
import pandas as pd

TELCO_NUMERIC_FEATURES = [
    "tenure_months", "monthly_charges", "total_charges", "service_count",
    "monthly_charge_per_service", "charges_per_tenure_month", "senior_citizen",
    "usage_trend", "complaint_count",
]
TELCO_CATEGORICAL_FEATURES = [
    "contract", "internet_service", "tech_support", "paperless_billing",
    "payment_method", "gender", "partner", "dependents", "online_security",
    "online_backup", "device_protection", "streaming_tv", "streaming_movies",
    "phone_service", "multiple_lines",
]


def prepare_telco(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize IBM Telco field names and derive service/charge features.

    IBM Telco is a cross-sectional snapshot: it has no longitudinal usage history
    or complaint log. We do not fabricate either; service subscriptions and charges
    are used as available proxies and that limitation is documented.
    """
    d = df.copy()
    rename = {
        "customerID": "customer_id", "tenure": "tenure_months",
        "MonthlyCharges": "monthly_charges", "TotalCharges": "total_charges",
        "Contract": "contract", "InternetService": "internet_service",
        "TechSupport": "tech_support", "PaperlessBilling": "paperless_billing",
        "Churn": "churn", "SeniorCitizen": "senior_citizen",
        "PaymentMethod": "payment_method", "OnlineSecurity": "online_security",
        "OnlineBackup": "online_backup", "DeviceProtection": "device_protection",
        "StreamingTV": "streaming_tv", "StreamingMovies": "streaming_movies",
        "PhoneService": "phone_service", "MultipleLines": "multiple_lines",
        "Partner": "partner", "Dependents": "dependents",
        "ComplaintCount": "complaint_count", "Complaints": "complaint_count",
    }
    d = d.rename(columns=rename)
    if "churn" not in d:
        raise ValueError("Telco dataset must contain a Churn/churn target")
    if d["churn"].dtype == "object":
        d["churn"] = d["churn"].astype(str).str.strip().map({"Yes": 1, "No": 0, "1": 1, "0": 0})
    d["churn"] = pd.to_numeric(d["churn"], errors="coerce")
    d = d.dropna(subset=["churn"]).copy()
    for c in ("tenure_months", "monthly_charges", "total_charges", "senior_citizen"):
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in ("tenure_months", "monthly_charges", "total_charges"):
        if c not in d:
            d[c] = float("nan")
    if "total_charges" in d:
        d["total_charges"] = d["total_charges"].fillna(0).clip(lower=0)
    if "senior_citizen" not in d:
        d["senior_citizen"] = 0
    # Optional longitudinal fields are used only when supplied by a richer source.
    # IBM Telco itself does not include usage history or complaint records.
    for col in ("complaint_count", "usage_trend"):
        if col in d: d[col] = pd.to_numeric(d[col], errors="coerce")
    usage_windows = [c for c in ("usage_3m", "usage_6m", "usage_12m") if c in d]
    if "usage_trend" not in d and len(usage_windows) >= 2:
        for col in usage_windows: d[col] = pd.to_numeric(d[col], errors="coerce")
        d["usage_trend"] = (d[usage_windows[0]] - d[usage_windows[-1]]) / d[usage_windows[-1]].abs().replace(0, 1)
    service_cols = [c for c in ["phone_service", "multiple_lines", "online_security", "online_backup",
        "device_protection", "tech_support", "streaming_tv", "streaming_movies"] if c in d]
    if service_cols:
        d["service_count"] = sum(d[c].astype(str).str.lower().eq("yes").astype(int) for c in service_cols)
    else:
        d["service_count"] = 0
    d["monthly_charge_per_service"] = d["monthly_charges"] / d["service_count"].replace(0, 1)
    d["charges_per_tenure_month"] = d["total_charges"] / d["tenure_months"].clip(lower=1)
    d["dataset_source"] = d.get("dataset_source", "IBM Telco Customer Churn")
    return d


def build_retail_survival(transactions: pd.DataFrame, inactivity_days: int = 90) -> pd.DataFrame:
    """Build a purchase-inactivity survival proxy, not contractual cancellation.

    Event is defined as no valid purchase for `inactivity_days` by the dataset
    cutoff. Features for Cox are computed only from the first 30 days of observed
    history. The observed duration stops at the inactivity threshold; otherwise it
    is administratively censored at the last dataset date.
    """
    if inactivity_days <= 0:
        raise ValueError("inactivity_days must be positive")
    d = transactions.copy()
    d["InvoiceDate"] = pd.to_datetime(d["InvoiceDate"], errors="coerce")
    for col in ("Customer ID", "Quantity", "Price"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["InvoiceDate", "Customer ID", "Quantity", "Price"])
    d = d[(d["Quantity"] > 0) & (d["Price"] > 0) & (~d["Invoice"].astype(str).str.startswith("C"))].copy()
    if d.empty:
        raise ValueError("No valid retail transactions remain for survival analysis")
    cutoff = d["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    d["revenue"] = d["Quantity"] * d["Price"]
    d["customer_id"] = d["Customer ID"].astype("Int64").astype(str)
    g = d.groupby("customer_id").agg(first_purchase=("InvoiceDate", "min"), last_purchase=("InvoiceDate", "max"),
        orders=("Invoice", "nunique"), revenue=("revenue", "sum")).reset_index()
    g["recency_days"] = (cutoff - g["last_purchase"]).dt.days
    g["event_observed"] = (g["recency_days"] >= inactivity_days).astype(int)
    observed_duration = (g["last_purchase"] - g["first_purchase"]).dt.days + inactivity_days
    censored_duration = (cutoff - g["first_purchase"]).dt.days
    g["duration_days"] = observed_duration.where(g["event_observed"].eq(1), censored_duration).clip(lower=1).astype(int)
    first_date = d.groupby("customer_id")["InvoiceDate"].transform("min")
    baseline = d[d["InvoiceDate"] < first_date + pd.Timedelta(days=30)]
    base = baseline.groupby("customer_id").agg(baseline_orders=("Invoice", "nunique"),
        baseline_revenue=("revenue", "sum")).reset_index()
    g = g.merge(base, on="customer_id", how="left")
    g["baseline_orders"] = g["baseline_orders"].fillna(0).astype(int)
    g["baseline_revenue"] = g["baseline_revenue"].fillna(0.0)
    return g[["customer_id", "duration_days", "event_observed", "orders", "revenue", "recency_days",
              "baseline_orders", "baseline_revenue"]]

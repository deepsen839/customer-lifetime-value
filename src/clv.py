"""Leakage-aware, forward-window customer revenue forecasting.

The model predicts revenue in a future 90-day window from transactions strictly
before the cutoff. The annualized contribution value is a transparent proxy,
not a contractual or fully validated lifetime-value estimate.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

FEATURES = ["recency_days", "orders", "historical_revenue", "items",
            "customer_age_days", "average_order_value", "orders_per_30d",
            "recent_90d_orders", "prior_90d_orders", "order_frequency_trend",
            "recent_90d_revenue", "prior_90d_revenue", "revenue_trend"]

def _clean_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    required = {"InvoiceDate", "Customer ID", "Quantity", "Price", "Invoice"}
    missing = required - set(transactions.columns)
    if missing:
        raise ValueError(f"Transactions missing required columns: {sorted(missing)}")
    d = transactions.copy()
    d["InvoiceDate"] = pd.to_datetime(d["InvoiceDate"], errors="coerce")
    for col in ("Customer ID", "Quantity", "Price"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["InvoiceDate", "Customer ID", "Quantity", "Price"])
    d = d[(d["Quantity"] > 0) & (d["Price"] > 0) &
          (~d["Invoice"].astype(str).str.startswith("C"))].copy()
    if d.empty:
        raise ValueError("No valid positive, non-cancelled transactions remain")
    d["line_revenue"] = d["Quantity"] * d["Price"]
    d["customer_id"] = d["Customer ID"].astype("Int64").astype(str)
    return d

def build_forward_clv_frame(transactions: pd.DataFrame, horizon_days: int = 90):
    """Create pre-cutoff features and realized revenue during the next horizon."""
    if horizon_days < 30:
        raise ValueError("horizon_days must be at least 30")
    d = _clean_transactions(transactions)
    end = d["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    cutoff = end - pd.Timedelta(days=horizon_days)
    history = d[d["InvoiceDate"] < cutoff].copy()
    future = d[(d["InvoiceDate"] >= cutoff) & (d["InvoiceDate"] < end)].copy()
    if history.empty or future.empty:
        raise ValueError("Not enough history and future transactions for a forward-window backtest")
    h = history.groupby("customer_id").agg(
        first_purchase=("InvoiceDate", "min"), last_purchase=("InvoiceDate", "max"),
        orders=("Invoice", "nunique"), historical_revenue=("line_revenue", "sum"),
        items=("Quantity", "sum")).reset_index()
    h["recency_days"] = (cutoff - h["last_purchase"]).dt.days.clip(lower=0)
    h["customer_age_days"] = (h["last_purchase"] - h["first_purchase"]).dt.days.clip(lower=0)
    h["average_order_value"] = h["historical_revenue"] / h["orders"].clip(lower=1)
    h["orders_per_30d"] = h["orders"] / (h["customer_age_days"].clip(lower=30) / 30.0)
    # Leakage-safe usage/purchase trends from two 90-day windows strictly before cutoff.
    recent = history[history["InvoiceDate"] >= cutoff - pd.Timedelta(days=90)]
    prior = history[(history["InvoiceDate"] >= cutoff - pd.Timedelta(days=180)) &
                    (history["InvoiceDate"] < cutoff - pd.Timedelta(days=90))]
    recent_orders = recent.groupby("customer_id").agg(recent_90d_orders=("Invoice", "nunique"),
        recent_90d_revenue=("line_revenue", "sum"))
    prior_orders = prior.groupby("customer_id").agg(prior_90d_orders=("Invoice", "nunique"),
        prior_90d_revenue=("line_revenue", "sum"))
    h = h.join(recent_orders, on="customer_id").join(prior_orders, on="customer_id")
    for col in ("recent_90d_orders", "recent_90d_revenue", "prior_90d_orders", "prior_90d_revenue"):
        h[col] = h[col].fillna(0.0)
    h["order_frequency_trend"] = h["recent_90d_orders"] - h["prior_90d_orders"]
    h["revenue_trend"] = h["recent_90d_revenue"] - h["prior_90d_revenue"]
    target = future.groupby("customer_id")["line_revenue"].sum().rename("actual_90d_revenue")
    h = h.merge(target, on="customer_id", how="left")
    h["actual_90d_revenue"] = h["actual_90d_revenue"].fillna(0.0).clip(lower=0.0)
    h["cutoff_date"] = cutoff.date().isoformat()
    return h, cutoff, end

def fit_clv_forecast(transactions: pd.DataFrame, horizon_days: int = 90,
                     contribution_margin: float = 0.30, seed: int = 42):
    """Train a regressor and compare with a mean baseline on held-out customers."""
    if not 0 <= contribution_margin <= 1:
        raise ValueError("contribution_margin must be between 0 and 1")
    frame, cutoff, end = build_forward_clv_frame(transactions, horizon_days)
    if len(frame) < 30:
        raise ValueError(f"At least 30 customers with pre-cutoff history are required; got {len(frame)}")
    X, y = frame[FEATURES], frame["actual_90d_revenue"].astype(float)
    indices = np.arange(len(frame))
    train_idx, test_idx = train_test_split(indices, test_size=0.25, random_state=seed)
    model = make_pipeline(SimpleImputer(strategy="median"),
                          HistGradientBoostingRegressor(max_iter=180, learning_rate=0.06,
                                                        max_leaf_nodes=15, l2_regularization=1.0,
                                                        random_state=seed))
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    baseline = DummyRegressor(strategy="mean").fit(X.iloc[train_idx], y.iloc[train_idx])
    pred = np.maximum(model.predict(X.iloc[test_idx]), 0.0)
    base_pred = np.maximum(baseline.predict(X.iloc[test_idx]), 0.0)
    y_test = y.iloc[test_idx].to_numpy()
    def metrics(y_true, y_pred):
        return {"mae": float(mean_absolute_error(y_true, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else None,
                "mean_actual_90d_revenue": float(np.mean(y_true)),
                "mean_predicted_90d_revenue": float(np.mean(y_pred))}
    test_metrics = metrics(y_test, pred)
    baseline_metrics = metrics(y_test, base_pred)
    all_pred = np.maximum(model.predict(X), 0.0)
    frame["predicted_90d_revenue"] = all_pred
    frame["annualized_contribution_clv_proxy"] = all_pred * (365.0 / horizon_days) * contribution_margin
    frame["is_test_customer"] = False
    frame.loc[frame.index[test_idx], "is_test_customer"] = True
    summary = {"model": "HistGradientBoostingRegressor", "target": f"realized revenue in next {horizon_days} days",
               "feature_cutoff_date": cutoff.date().isoformat(), "label_window_end_exclusive": end.date().isoformat(),
               "horizon_days": int(horizon_days), "customers": int(len(frame)),
               "train_customers": int(len(train_idx)), "test_customers": int(len(test_idx)),
               "contribution_margin_assumption": float(contribution_margin),
               "annualized_clv_definition": "predicted 90-day revenue * 365 / horizon_days * assumed contribution margin",
               "test_metrics": test_metrics, "mean_baseline_test_metrics": baseline_metrics,
               "beats_baseline_mae": bool(test_metrics["mae"] < baseline_metrics["mae"]),
               "validation_note": "Random customer holdout within a strictly forward 90-day label window; not an out-of-time multi-period validation. Annualized contribution is a proxy, not a calibrated lifetime horizon."}
    return model, frame, summary

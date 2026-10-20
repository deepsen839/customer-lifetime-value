"""Kaplan-Meier and Cox survival modelling with individual survival curves."""
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter

COX_FEATURES = ["baseline_orders", "baseline_revenue"]

def _cox_frame(df, duration_col, event_col):
    required = {duration_col, event_col, *COX_FEATURES}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(df.columns))}; rebuild the leakage-safe survival dataset")
    cox_df = df.loc[df[duration_col] >= 30, [duration_col, event_col] + COX_FEATURES].dropna().copy()
    for col in COX_FEATURES:
        cox_df[col] = np.log1p(pd.to_numeric(cox_df[col], errors="coerce").clip(lower=0))
    return cox_df.dropna()

def fit_survival_models(df, duration_col="duration_days", event_col="event_observed"):
    required = {duration_col, event_col}
    if not required.issubset(df.columns): raise ValueError(f"Missing columns: {sorted(required - set(df.columns))}")
    if (pd.to_numeric(df[duration_col], errors="coerce") <= 0).any(): raise ValueError("Durations must be positive")
    if df[event_col].nunique() < 2: raise ValueError("Need observed events and censored cases")
    km = KaplanMeierFitter().fit(df[duration_col], event_observed=df[event_col], label="purchase_survival")
    cox_df = _cox_frame(df, duration_col, event_col)
    if len(cox_df) < 20 or cox_df[event_col].nunique() < 2:
        raise ValueError(f"Not enough eligible records/events for Cox model: n={len(cox_df)}")
    cox = CoxPHFitter(penalizer=0.05).fit(cox_df, duration_col=duration_col, event_col=event_col)
    return km, cox

def individual_survival_predictions(cox, survival_data, max_days=None, step_days=30):
    """Return customer-specific curves for customers with >=30 days baseline follow-up.

    Customers with less than 30 days of observed follow-up receive an explicit
    insufficient_followup status instead of a misleading prediction from an
    incomplete baseline feature window. Survival is conditional on the retail
    purchase-inactivity proxy, not telecom account cancellation.
    """
    d = survival_data.copy()
    eligible = d.loc[d["duration_days"] >= 30].copy()
    if eligible.empty:
        raise ValueError("No customers have the 30-day baseline follow-up required for individual survival estimates")
    if max_days is None: max_days = int(max(90, min(3650, d["duration_days"].max())))
    times = np.unique(np.r_[0, np.arange(step_days, max_days + step_days, step_days), max_days]).astype(float)
    times = times[(times >= 0) & (times <= max_days)]
    X = eligible[COX_FEATURES].copy()
    for col in COX_FEATURES:
        X[col] = np.log1p(pd.to_numeric(X[col], errors="coerce").clip(lower=0))
    sf = cox.predict_survival_function(X, times=times)
    sf.index.name = "time_days"
    long = sf.reset_index().melt(id_vars="time_days", var_name="row_index", value_name="survival_probability")
    row_map = pd.DataFrame({"row_index": list(sf.columns), "customer_id": eligible.loc[sf.columns, "customer_id"].astype(str).to_list()})
    long["row_index"] = pd.to_numeric(long["row_index"], errors="coerce")
    row_map["row_index"] = pd.to_numeric(row_map["row_index"], errors="coerce")
    long = long.merge(row_map, on="row_index", how="left").drop(columns="row_index")
    summaries = []
    for customer_id, grp in long.groupby("customer_id", sort=False):
        grp = grp.sort_values("time_days")
        summaries.append({"customer_id": customer_id,
            "survival_probability_90d": float(np.interp(90, grp.time_days, grp.survival_probability)),
            "survival_probability_180d": float(np.interp(180, grp.time_days, grp.survival_probability)),
            "survival_probability_365d": float(np.interp(365, grp.time_days, grp.survival_probability)),
            "restricted_mean_survival_days": float(np.trapz(grp.survival_probability, grp.time_days)),
            "prediction_status": "predicted"})
    summary = pd.DataFrame(summaries)
    all_customers = d[["customer_id", "duration_days"]].copy()
    summary = all_customers.merge(summary, on="customer_id", how="left")
    summary["prediction_status"] = summary["prediction_status"].fillna("insufficient_followup_under_30_days")
    return long[["customer_id", "time_days", "survival_probability"]], summary

def proportional_hazards_diagnostics(cox, survival_data):
    """Run lifelines PH assumption test; return serializable diagnostics."""
    from lifelines.statistics import proportional_hazard_test
    cox_df = _cox_frame(survival_data, "duration_days", "event_observed")
    test = proportional_hazard_test(cox, cox_df, time_transform="rank")
    return {str(k): {"test_statistic": float(v["test_statistic"]), "p_value": float(v["p"])}
            for k, v in test.summary.iterrows()}

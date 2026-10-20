"""End-to-end training pipeline over the selected public datasets."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.datasets import get_telco, get_hillstrom, get_retail_ii, retail_customer_value
from src.feature_engineering import prepare_telco, build_retail_survival
from src.modeling import train_model, customer_shap
from src.clustering import fit_segments
from src.survival import fit_survival_models, individual_survival_predictions, proportional_hazards_diagnostics
from src.uplift import fit_causalml_t_learner, score_uplift_customers
from src.clv import fit_clv_forecast
from src.db import get_engine, init_db, replace_table
from src.eda import profile_dataframe

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)

def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")

def main():
    # 1) IBM Telco: EDA, feature engineering, LightGBM, explanations, segmentation.
    telco = prepare_telco(get_telco())
    telco.to_csv(ROOT / "data" / "telco_normalized.csv", index=False)
    model, churn_metrics = train_model(telco)
    joblib.dump(model, ART / "telco_churn_model.joblib")
    write_json(ART / "telco_evaluation.json", churn_metrics)
    write_json(ART / "telco_eda.json", profile_dataframe(telco, "churn"))
    telco_cluster_model, telco_segments, telco_profiles = fit_segments(telco, feature_cols=[c for c in ["tenure_months", "monthly_charges", "total_charges", "service_count"] if c in telco])
    telco_segments.to_csv(ART / "telco_customer_segments.csv", index=False)
    telco_profiles.to_csv(ART / "telco_segment_profiles.csv")
    joblib.dump(telco_cluster_model, ART / "telco_cluster_model.joblib")
    try:
        explanation, names = customer_shap(model, telco)
        values = np.asarray(explanation.values)
        pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(values).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False).to_csv(ART / "telco_shap_global.csv", index=False)
        write_json(ART / "shap_status.json", {"status": "success", "method": "SHAP TreeExplainer"})
    except Exception as exc:
        write_json(ART / "shap_status.json", {"status": "failed", "reason": f"{type(exc).__name__}: {str(exc)[:300]}",
            "fallback_feature_importance_in_evaluation": churn_metrics.get("feature_importance", {})})

    # 2) Hillstrom randomized email campaign: women's email vs no email.
    hillstrom = get_hillstrom()
    hillstrom = hillstrom[hillstrom["segment"].isin(["Womens E-Mail", "No E-Mail"])].copy()
    hillstrom["treatment"] = (hillstrom["segment"] == "Womens E-Mail").astype(int)
    uplift_features = [c for c in ["recency", "history", "mens", "womens", "newbie"] if c in hillstrom]
    # The scikit-uplift copy provides observed visit outcomes, not conversion/spend.
    # Train and report visit-response uplift; do not fabricate conversion labels.
    if "visit" not in hillstrom:
        raise ValueError("Hillstrom dataset must contain the observed visit outcome")
    uplift_model, uplift_test, uplift_summary = fit_causalml_t_learner(
        hillstrom, uplift_features, "treatment", "visit", seed=42)
    uplift_summary["outcome"] = "visit"
    uplift_summary["outcome_interpretation"] = (
        "Observed visit response; not conversion or revenue uplift"
    )
    # Test predictions remain separate from full-population targeting scores.
    uplift_test.to_csv(ART / "hillstrom_uplift_test_scored.csv", index=False)
    uplift_scores = score_uplift_customers(uplift_model, hillstrom, uplift_features)
    uplift_scores.to_csv(ART / "hillstrom_uplift_customer_scores.csv", index=False)
    joblib.dump(uplift_model, ART / "hillstrom_uplift_model.joblib")
    write_json(ART / "hillstrom_uplift_evaluation.json", uplift_summary)

    # 3) Online Retail II: historical value, held-out forward revenue prediction,
    # retail RFM/CLV segmentation, and explicitly proxy-based survival analysis.
    retail = get_retail_ii()
    retail.to_pickle(ROOT / "data" / "online_retail_ii_clean.pkl")
    clv = retail_customer_value(retail)
    clv.to_csv(ART / "online_retail_ii_customer_value.csv", index=False)
    clv_model, clv_forecast, clv_metrics = fit_clv_forecast(retail, horizon_days=90, contribution_margin=0.30)
    joblib.dump(clv_model, ART / "retail_clv_forecast_model.joblib")
    clv_forecast.to_csv(ART / "retail_clv_forecast.csv", index=False)
    write_json(ART / "retail_clv_evaluation.json", clv_metrics)
    forecast_cols = clv_forecast[["customer_id", "predicted_90d_revenue", "annualized_contribution_clv_proxy",
        "recent_90d_orders", "prior_90d_orders", "order_frequency_trend", "recent_90d_revenue", "prior_90d_revenue", "revenue_trend"]]
    clv = clv.merge(forecast_cols, on="customer_id", how="left", suffixes=("", "_forecast"))
    # Segment on customer value/recency/frequency features rather than telecom data.
    retail_segment_features = [c for c in ["orders", "revenue", "recency_days", "customer_age_days", "average_order_value", "items"] if c in clv]
    retail_cluster_model, retail_segments, retail_profiles = fit_segments(clv, feature_cols=retail_segment_features)
    retail_segments.to_csv(ART / "retail_customer_segments.csv", index=False)
    retail_profiles.to_csv(ART / "retail_segment_profiles.csv")
    joblib.dump(retail_cluster_model, ART / "retail_cluster_model.joblib")

    survival = build_retail_survival(retail, inactivity_days=90)
    survival.to_csv(ART / "retail_survival_dataset.csv", index=False)
    km, cox = fit_survival_models(survival)
    cox.summary.to_csv(ART / "retail_cox_summary.csv")
    joblib.dump(cox, ART / "retail_cox_model.joblib")
    curves, individual = individual_survival_predictions(cox, survival)
    curves.to_csv(ART / "retail_individual_survival_curves.csv", index=False)
    individual.to_csv(ART / "retail_individual_survival_summary.csv", index=False)
    median = float(km.median_survival_time_)
    surv_summary = {"median_survival_days": median if np.isfinite(median) else None,
        "median_survival_reached": bool(np.isfinite(median)),
        "event_definition": "90 days without valid purchase; this is a retail inactivity proxy, not contractual cancellation",
        "dataset": "UCI Online Retail II", "cox_concordance_index": float(cox.concordance_index_),
        "n_customers": int(len(survival)), "n_events": int(survival.event_observed.sum()),
        "event_rate": float(survival.event_observed.mean()),
        "individual_predictions_file": "retail_individual_survival_summary.csv",
        "curve_file": "retail_individual_survival_curves.csv"}
    try:
        surv_summary["proportional_hazards_diagnostics"] = proportional_hazards_diagnostics(cox, survival)
        surv_summary["proportional_hazards_assumption_warning"] = any(v["p_value"] < 0.05 for v in surv_summary["proportional_hazards_diagnostics"].values())
    except Exception as exc:
        surv_summary["proportional_hazards_diagnostics_error"] = f"{type(exc).__name__}: {str(exc)[:240]}"
    write_json(ART / "retail_km_median_survival.json", surv_summary)

    # 4) Persist normalized real-data outputs to PostgreSQL when configured.
    engine = get_engine()
    if engine is not None:
        init_db(engine)
        telco_cols = ["customer_id", "tenure_months", "monthly_charges", "total_charges", "contract", "internet_service", "tech_support", "paperless_billing", "churn",
            "service_count", "monthly_charge_per_service", "charges_per_tenure_month", "senior_citizen", "usage_trend", "complaint_count",
            "payment_method", "gender", "partner", "dependents", "online_security", "online_backup", "device_protection", "streaming_tv", "streaming_movies", "phone_service", "multiple_lines"]
        telco_cols = [c for c in telco_cols if c in telco.columns]
        replace_table(engine, "telco_customers", telco[telco_cols])
        # Campaign records include the trained score; treatment and outcome remain observed historical labels.
        campaign = uplift_scores.copy()
        campaign_cols = [c for c in ["campaign_row_id", "recency", "history", "segment", "treatment", "visit", "conversion", "spend", "predicted_uplift", "p_treat", "p_control"] if c in campaign]
        replace_table(engine, "hillstrom_campaign", campaign[campaign_cols])
        retail_cols = ["customer_id", "orders", "revenue", "recency_days", "customer_age_days", "average_order_value", "historical_value_proxy", "predicted_90d_revenue", "annualized_contribution_clv_proxy", "segment_id",
            "recent_90d_orders", "prior_90d_orders", "order_frequency_trend", "recent_90d_revenue", "prior_90d_revenue", "revenue_trend"]
        retail_cols = [c for c in retail_cols if c in retail_segments.columns]
        replace_table(engine, "retail_customer_value", retail_segments[retail_cols])
        replace_table(engine, "retail_survival", survival)
        replace_table(engine, "retail_survival_predictions", individual)
    # Reproducibility metadata and explicit dataset coverage.
    manifest = {"datasets": {"ibm_telco": {"rows": int(len(telco)), "customers": int(telco.customer_id.nunique()), "loaded_for_training": True},
        "hillstrom_womens_email_vs_no_email": {"rows": int(len(hillstrom)), "treatment_rows": int(hillstrom.treatment.sum()), "loaded_for_training": True},
        "uci_online_retail_ii": {"transaction_rows": int(len(retail)), "customers": int(clv.customer_id.nunique()), "loaded_for_training": True}},
        "runtime": "Results generated by src.train_real; inspect the JSON/CSV files for measured evaluation metrics.",
        "limits": ["Telco is cross-sectional and has no complaint log or longitudinal usage history.",
            "Retail survival is an inactivity proxy, not confirmed cancellation.",
            "Hillstrom uplift is campaign conversion response, not telecom retention lift.",
            "Annualized contribution CLV uses an assumed 30% contribution margin and is not calibrated lifetime value."]}
    write_json(ART / "run_manifest.json", manifest)
    result = {"churn": churn_metrics, "clv": clv_metrics, "uplift": uplift_summary, "survival": surv_summary, "datasets": manifest["datasets"]}
    write_json(ART / "training_summary.json", result)
    print(json.dumps(result, indent=2, default=str))

if __name__ == "__main__": main()

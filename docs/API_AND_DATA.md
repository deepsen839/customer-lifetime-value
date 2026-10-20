# API and data contract

Base URL when running Docker locally: `http://localhost:8000`.

## `GET /health`
Returns `{"status":"ok"}`.

## `POST /predict/churn`
Required JSON fields: `tenure_months`, `monthly_charges`, `total_charges`, `contract`, `internet_service`, `tech_support`, `paperless_billing`. Optional richer-source fields include service count, charge ratios, demographics, additional services, `usage_trend`, and `complaint_count`. Missing optional features are imputed by the fitted preprocessing pipeline. IBM Telco itself does not supply complaint logs or longitudinal usage history.

Returns `churn_probability` between 0 and 1 and a heuristic risk band. The default band thresholds are presentation defaults, not a calibrated intervention policy.

## `POST /simulate/roi`
JSON: `eligible_customers` (integer >= 0), `offer_cost_per_customer` (>= 0), `incremental_lift` (0 to 1), `contribution_value_per_save` (>= 0). Returns cost, expected incremental saves, incremental value, net benefit, and ROI. Lift is a scenario input.

## `POST /simulate/uplift-roi`
JSON fields: `targeted_fraction` (0 to 1), `discount_per_conversion`, `contribution_per_conversion`, `contact_cost_per_customer`, and `contribution_margin`. Reads the full-population Hillstrom uplift scores from `artifacts/hillstrom_uplift_customer_scores.csv` and calculates campaign scenario economics for the top predicted-uplift fraction. The estimate is not a guarantee of incremental profit.

## `GET /customers/{customer_id}/survival`
Returns individual retail purchase-inactivity survival estimates for a customer in the trained cohort. Records with less than 30 days of baseline follow-up are flagged `insufficient_followup_under_30_days`; no survival probabilities are supplied for those records.

## PostgreSQL tables
- `telco_customers`: IBM Telco normalized features and churn target
- `hillstrom_campaign`: randomized email treatment/outcomes and uplift scores
- `retail_customer_value`: historical and forecast value, trend features, segment ID
- `retail_survival`: purchase-inactivity event/censoring records
- `retail_survival_predictions`: customer-specific survival probabilities/status

The training pipeline replaces each analytical table atomically within a transaction after model artifacts have been written. Database integration must be verified by running the full Docker workflow.

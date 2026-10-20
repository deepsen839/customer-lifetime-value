# Data dictionary

## `telco_customers` — IBM Telco
- `customer_id`: source customer identifier (primary key)
- `tenure_months`, `monthly_charges`, `total_charges`: available numeric customer/charge fields
- `contract`, `internet_service`, `tech_support`, `paperless_billing`, `payment_method`, `gender`, `partner`, `dependents`, service subscription fields: source categories
- `service_count`: count of available service flags equal to `Yes`
- `monthly_charge_per_service`: monthly charges divided by service count (uses 1 as denominator when count is zero)
- `charges_per_tenure_month`: total charges divided by tenure, with denominator clipped to at least 1
- `usage_trend`, `complaint_count`: nullable optional fields. IBM Telco does not provide them; values are populated only if a richer compatible source contains explicit longitudinal usage or complaint fields.
- `churn`: binary target, 1 for Yes, 0 for No

## `hillstrom_campaign` — Hillstrom
- `campaign_row_id`: source row identifier
- `segment`: email assignment arm
- `treatment`: 1 for Women's Email, 0 for No Email in the selected analysis cohort
- `visit`, `conversion`, `spend`: observed outcomes
- `predicted_uplift`: estimated difference in conversion probability under email vs no email
- `p_treat`, `p_control`: estimated potential conversion probabilities where available; for CausalML-only effect output, these are reconstructed from the aggregate control rate plus uplift and should be interpreted as an approximation
- `recency`, `history`: source customer covariates

## `retail_customer_value` — UCI Online Retail II
- `customer_id`: normalized customer identifier
- `orders`, `revenue`, `recency_days`, `customer_age_days`, `average_order_value`: historical value/frequency/recency aggregates
- `predicted_90d_revenue`: model prediction for the next 90-day window from pre-cutoff features
- `annualized_contribution_clv_proxy`: predicted 90-day revenue annualized and multiplied by an assumed 30% margin; not a calibrated lifetime estimate
- `recent_90d_orders`, `prior_90d_orders`, `order_frequency_trend`, `recent_90d_revenue`, `prior_90d_revenue`, `revenue_trend`: purchase activity trends in the 180 days before the feature cutoff
- `segment_id`: KMeans segment label

## `retail_survival`
- `duration_days`: time from first purchase to 90-day inactivity event or censoring cutoff
- `event_observed`: 1 if 90-day purchase inactivity is observed before data cutoff, otherwise 0 for administrative censoring
- `baseline_orders`, `baseline_revenue`: only transactions in the first 30 days after first purchase, used as Cox covariates

## `retail_survival_predictions`
- `survival_probability_90d`, `survival_probability_180d`, `survival_probability_365d`: Cox-model estimated purchase-active survival probability under the proxy event
- `restricted_mean_survival_days`: area under the predicted survival curve over the configured time grid
- `prediction_status`: `predicted` or `insufficient_followup_under_30_days`

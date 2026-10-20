CREATE TABLE IF NOT EXISTS telco_customers (
 customer_id TEXT PRIMARY KEY, tenure_months INTEGER, monthly_charges NUMERIC,
 total_charges NUMERIC, contract TEXT, internet_service TEXT, tech_support TEXT,
 paperless_billing TEXT, churn SMALLINT CHECK (churn IN (0,1)), service_count INTEGER,
 monthly_charge_per_service DOUBLE PRECISION, charges_per_tenure_month DOUBLE PRECISION,
 senior_citizen INTEGER, usage_trend DOUBLE PRECISION, complaint_count DOUBLE PRECISION,
 payment_method TEXT, gender TEXT, partner TEXT, dependents TEXT, online_security TEXT,
 online_backup TEXT, device_protection TEXT, streaming_tv TEXT, streaming_movies TEXT,
 phone_service TEXT, multiple_lines TEXT
);
CREATE INDEX IF NOT EXISTS idx_telco_churn ON telco_customers(churn);
CREATE INDEX IF NOT EXISTS idx_telco_contract ON telco_customers(contract);

CREATE TABLE IF NOT EXISTS hillstrom_campaign (
 campaign_row_id BIGINT PRIMARY KEY, recency INTEGER, history DOUBLE PRECISION,
 segment TEXT, treatment SMALLINT, visit SMALLINT, conversion SMALLINT, spend DOUBLE PRECISION,
 predicted_uplift DOUBLE PRECISION, p_treat DOUBLE PRECISION, p_control DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS retail_customer_value (
 customer_id TEXT PRIMARY KEY, orders INTEGER, revenue DOUBLE PRECISION,
 recency_days INTEGER, customer_age_days INTEGER, average_order_value DOUBLE PRECISION,
 historical_value_proxy DOUBLE PRECISION, predicted_90d_revenue DOUBLE PRECISION,
 annualized_contribution_clv_proxy DOUBLE PRECISION, segment_id INTEGER,
 recent_90d_orders DOUBLE PRECISION, prior_90d_orders DOUBLE PRECISION,
 order_frequency_trend DOUBLE PRECISION, recent_90d_revenue DOUBLE PRECISION,
 prior_90d_revenue DOUBLE PRECISION, revenue_trend DOUBLE PRECISION
);
CREATE TABLE IF NOT EXISTS retail_survival (
 customer_id TEXT PRIMARY KEY, duration_days INTEGER, event_observed SMALLINT,
 orders INTEGER, revenue DOUBLE PRECISION, recency_days INTEGER,
 baseline_orders INTEGER, baseline_revenue DOUBLE PRECISION
);
CREATE TABLE IF NOT EXISTS retail_survival_predictions (
 customer_id TEXT PRIMARY KEY, duration_days INTEGER,
 survival_probability_90d DOUBLE PRECISION, survival_probability_180d DOUBLE PRECISION,
 survival_probability_365d DOUBLE PRECISION, restricted_mean_survival_days DOUBLE PRECISION,
 prediction_status TEXT
);

-- Idempotent migrations for older development databases.
ALTER TABLE telco_customers ADD COLUMN IF NOT EXISTS service_count INTEGER;
ALTER TABLE telco_customers ADD COLUMN IF NOT EXISTS monthly_charge_per_service DOUBLE PRECISION;
ALTER TABLE telco_customers ADD COLUMN IF NOT EXISTS charges_per_tenure_month DOUBLE PRECISION;
ALTER TABLE telco_customers ADD COLUMN IF NOT EXISTS senior_citizen INTEGER;
ALTER TABLE telco_customers ADD COLUMN IF NOT EXISTS usage_trend DOUBLE PRECISION;
ALTER TABLE telco_customers ADD COLUMN IF NOT EXISTS complaint_count DOUBLE PRECISION;
ALTER TABLE telco_customers ADD COLUMN IF NOT EXISTS payment_method TEXT;
ALTER TABLE telco_customers ADD COLUMN IF NOT EXISTS gender TEXT;
ALTER TABLE telco_customers ADD COLUMN IF NOT EXISTS partner TEXT;
ALTER TABLE telco_customers ADD COLUMN IF NOT EXISTS dependents TEXT;
ALTER TABLE telco_customers ADD COLUMN IF NOT EXISTS online_security TEXT;
ALTER TABLE telco_customers ADD COLUMN IF NOT EXISTS online_backup TEXT;
ALTER TABLE telco_customers ADD COLUMN IF NOT EXISTS device_protection TEXT;
ALTER TABLE telco_customers ADD COLUMN IF NOT EXISTS streaming_tv TEXT;
ALTER TABLE telco_customers ADD COLUMN IF NOT EXISTS streaming_movies TEXT;
ALTER TABLE telco_customers ADD COLUMN IF NOT EXISTS phone_service TEXT;
ALTER TABLE telco_customers ADD COLUMN IF NOT EXISTS multiple_lines TEXT;
ALTER TABLE retail_customer_value ADD COLUMN IF NOT EXISTS predicted_90d_revenue DOUBLE PRECISION;
ALTER TABLE retail_customer_value ADD COLUMN IF NOT EXISTS annualized_contribution_clv_proxy DOUBLE PRECISION;
ALTER TABLE retail_customer_value ADD COLUMN IF NOT EXISTS segment_id INTEGER;
ALTER TABLE retail_customer_value ADD COLUMN IF NOT EXISTS recent_90d_orders DOUBLE PRECISION;
ALTER TABLE retail_customer_value ADD COLUMN IF NOT EXISTS prior_90d_orders DOUBLE PRECISION;
ALTER TABLE retail_customer_value ADD COLUMN IF NOT EXISTS order_frequency_trend DOUBLE PRECISION;
ALTER TABLE retail_customer_value ADD COLUMN IF NOT EXISTS recent_90d_revenue DOUBLE PRECISION;
ALTER TABLE retail_customer_value ADD COLUMN IF NOT EXISTS prior_90d_revenue DOUBLE PRECISION;
ALTER TABLE retail_customer_value ADD COLUMN IF NOT EXISTS revenue_trend DOUBLE PRECISION;
ALTER TABLE retail_survival ADD COLUMN IF NOT EXISTS baseline_orders INTEGER;
ALTER TABLE retail_survival ADD COLUMN IF NOT EXISTS baseline_revenue DOUBLE PRECISION;
ALTER TABLE retail_survival_predictions ADD COLUMN IF NOT EXISTS duration_days INTEGER;
ALTER TABLE retail_survival_predictions ADD COLUMN IF NOT EXISTS prediction_status TEXT;
ALTER TABLE hillstrom_campaign ADD COLUMN IF NOT EXISTS campaign_row_id BIGINT;
ALTER TABLE hillstrom_campaign ADD COLUMN IF NOT EXISTS predicted_uplift DOUBLE PRECISION;
ALTER TABLE hillstrom_campaign ADD COLUMN IF NOT EXISTS p_treat DOUBLE PRECISION;
ALTER TABLE hillstrom_campaign ADD COLUMN IF NOT EXISTS p_control DOUBLE PRECISION;

CREATE INDEX IF NOT EXISTS idx_hillstrom_uplift ON hillstrom_campaign(predicted_uplift DESC);

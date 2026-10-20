# Data sources, scope, and field availability

The pipeline downloads public data at runtime; raw data is not redistributed in the project ZIP.

## 1. IBM Telco Customer Churn
- Source CSV: https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv
- Raw download: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
- Used for churn classification, EDA, SHAP explanations, and telecom customer segmentation.
- Fields include tenure, monthly/total charges, contract, service subscriptions, support/security options, and churn.
- **Not present in this source:** a complaint log or longitudinal monthly usage history. The pipeline does not invent these fields. It accepts `complaint_count` and/or explicit usage windows (`usage_3m`, `usage_6m`, `usage_12m`) when a richer compatible source supplies them; they are not populated for IBM Telco.

## 2. Hillstrom MineThatData Email Campaign
- Source page: https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html
- CSV: https://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv
- Used for randomized treatment/control uplift modeling. The main comparison is Women's E-Mail versus No E-Mail, with `conversion` as the outcome; `visit` is retained for analysis.
- Uplift describes email conversion response, **not telecom churn-retention effect**. Held-out subgroup effects can be noisy; validate a campaign experimentally before spending budget.

## 3. UCI Online Retail II
- Dataset page: https://archive.ics.uci.edu/dataset/502/online+retail+ii
- ZIP: https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip
- Used for customer revenue/CLV forecasting, RFM-like segments, purchase-frequency/revenue trends, and a purchase-inactivity survival proxy.
- Transactions are cleaned to exclude cancellation invoices and non-positive quantity/price. The future 90-day target is separated from pre-cutoff features.
- Survival event is 90 days without a valid purchase by the dataset cutoff. This is an operational proxy, not confirmed customer cancellation.

## Optional: Criteo uplift
- Source: https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz
- Downloaded only when explicitly requested with `python -m src.datasets --dataset criteo --criteo-rows 500000`.

Review source licenses/terms before redistributing raw data. Download failures are reported and training should not be described as completed until the full run succeeds.

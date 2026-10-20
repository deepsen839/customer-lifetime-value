# Validation and acceptance checklist

## Automated checks

From the project root:

```bash
python -m compileall -q .
python -m pytest -q
```

Tests cover ROI arithmetic, Telco normalization/derived features, purchase-inactivity labels, forward-window CLV features and baseline comparisons, purchase-frequency trends, segmentation, churn inference, uplift wrapper/fallback scoring, API input validation, database URL credential escaping, and uplift ROI. Individual Cox survival curve tests run automatically when `lifelines` is installed.

## Full real-data acceptance run

```bash
cp .env.example .env
# Set POSTGRES_PASSWORD in .env to a strong random password.
docker compose up --build -d
docker compose ps
docker compose logs --tail=100 app db
docker compose exec app python -m src.datasets --dataset all
docker compose exec app python -m src.train_real
docker compose exec app python -m pytest -q
docker compose exec app python scripts/smoke_test.py
```

The full training pipeline downloads IBM Telco, Hillstrom, and UCI Online Retail II, trains the models, writes artifacts/evaluation JSON/CSV, and loads the normalized outputs into PostgreSQL. The smoke test checks API health, churn inference, both ROI endpoints, and non-empty PostgreSQL tables.

## Model-level acceptance checks

- [ ] Churn `roc_auc`, `pr_auc`, and `brier_score` are in `artifacts/telco_evaluation.json`; compare PR-AUC with the prior baseline and select the action threshold using campaign costs.
- [ ] `artifacts/telco_shap_global.csv` exists and has non-empty feature attributions, or `shap_status.json` explains the failure and the evaluation JSON records a fallback method.
- [ ] CLV held-out MAE/RMSE are compared with the mean baseline in `artifacts/retail_clv_evaluation.json`; do not claim a win unless the measured metrics support it.
- [ ] Uplift `method` identifies CausalML or a documented sklearn fallback; review held-out treatment/control differences, cumulative uplift curve, overlap, and sampling uncertainty.
- [ ] `artifacts/retail_individual_survival_summary.csv` and `artifacts/retail_individual_survival_curves.csv` contain individual curves for eligible customers and a summary/status row for every customer; under-30-day follow-up cases are explicitly flagged.
- [ ] Cox concordance and proportional-hazards diagnostics are reviewed; report any p-values below 0.05 as assumption warnings.
- [ ] `artifacts/retail_customer_segments.csv` and `artifacts/retail_segment_profiles.csv` contain retail CLV segments, alongside telecom segment profiles.
- [ ] PostgreSQL contains rows in `telco_customers`, `hillstrom_campaign`, `retail_customer_value`, `retail_survival`, and `retail_survival_predictions`.
- [ ] Streamlit opens and both ROI simulators run; verify that discount, contact cost, contribution margin, and targeting fraction change the scenario output.

## Data limitations to state in the presentation

- IBM Telco has no complaint log or longitudinal usage measurements. The model uses available service/charge features; complaint and longitudinal usage fields are only accepted if supplied by another compatible source.
- Retail survival uses a 90-day purchase-inactivity proxy and is not contractual cancellation. Cox covariates use first-30-day purchase history; individual survival probabilities are estimates, not guarantees.
- Hillstrom estimates incremental email conversion response, not telecom retention. Campaign economics are assumptions to be tested experimentally.
- CLV predicts next-90-day revenue; annualized contribution value assumes a 30% margin and is not a calibrated lifetime forecast.

## Environment verification limitation

The preparation environment has no working external DNS/network access and does not have Docker, PostgreSQL, Streamlit, or lifelines installed. Consequently, public dataset downloads and full-data training could not be run here. Do not claim the full real-data pipeline or deployment passed until it has been run in a network-enabled Docker environment and the generated artifacts have been inspected.

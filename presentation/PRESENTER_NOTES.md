# 10-minute presenter notes

Use this talk track after running the real-data pipeline. Replace generic statements with the actual values from `artifacts/training_summary.json` and the evaluation JSON files. Do not invent results.

1. **Problem (45 sec):** explain churn prediction, customer value and incremental offer response as separate decisions.
2. **Datasets (60 sec):** IBM Telco supports churn classification; Hillstrom provides randomized email treatment/control; UCI Online Retail II supports purchase-value and inactivity analyses. They are separate populations and must not be joined.
3. **Architecture (45 sec):** show dataset ingestion, feature pipelines, models, PostgreSQL, Streamlit, FastAPI and Docker Compose.
4. **Features and leakage (60 sec):** Telco features include tenure, contract, charges and available service subscriptions. Retail purchase/revenue trends use two 90-day windows strictly before the prediction cutoff; the target is revenue in the next 90 days.
5. **Models (75 sec):** LightGBM with SHAP/permutation fallback; Kaplan–Meier and Cox with individual curves for eligible records; CausalML T-learner with a logged fallback; gradient-boosted forward revenue model; KMeans customer segments.
6. **Measured results (90 sec):** report actual held-out churn ROC-AUC/PR-AUC/Brier, CLV MAE/RMSE versus the mean baseline, held-out uplift results, Cox concordance and PH diagnostics. Explicitly state which models beat their baselines.
7. **Dashboard/API demo (90 sec):** show churn metrics, survival predictions/status, uplift-ranked campaign customers, customer segments, API docs and PostgreSQL tables.
8. **ROI demo (60 sec):** compare the assumption-driven retention simulator with the Hillstrom uplift-guided conversion ROI simulator. Explain discount/contact-cost/margin assumptions.
9. **Limitations (45 sec):** IBM Telco lacks complaints and longitudinal usage history; Online Retail II inactivity is a proxy, not confirmed cancellation; Hillstrom uplift is not telecom retention; CLV is a 90-day forecast annualized using an assumed margin.
10. **Close (30 sec):** list acceptance checks and next actions. State only what the actual run verified.

## Before recording

- Run `docker compose up --build -d`.
- Run `docker compose exec app python -m src.train_real`.
- Run `docker compose exec app python -m pytest -q` and `docker compose exec app python scripts/smoke_test.py`.
- Inspect `artifacts/training_summary.json`, `telco_evaluation.json`, `retail_clv_evaluation.json`, `hillstrom_uplift_evaluation.json`, and `retail_km_median_survival.json`.
- If CausalML falls back, mention the recorded reason. If CLV does not beat the mean baseline, say so.

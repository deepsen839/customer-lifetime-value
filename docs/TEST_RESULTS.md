# Test Results and Model Evaluation Status

## Execution environment

The checks below were executed in the project preparation environment on 20 October 2026. They are software tests, not a claim that all public datasets were downloaded or that production deployment was validated.

## Automated test results

Command:

```bash
python -m pytest -q
```

Earlier preparation-environment result (before the latest dependency regression test):

```text
18 passed, 1 skipped, 6 warnings
```

Latest preparation-environment rerun after making the survival test mandatory and adding the PyArrow/NumPy regression test:

```text
18 passed, 2 failed, 6 warnings
```

Both failures are environment dependency failures, not assertion failures: `lifelines` and `pyarrow` are not installed in the preparation environment. They are required dependencies and are declared in `requirements.txt`; the Docker image installs them. The suite deliberately fails instead of silently skipping either dependency-backed integration test. Rebuild and run the suite inside Docker to verify the fixed dependency set. Warnings were non-fatal scikit-learn feature-name warnings and a SHAP/LightGBM output-shape compatibility warning.

## Other checks

| Check | Result | Evidence / caveat |
|---|---|---|
| Python compilation | Passed | `python -m compileall -q .` |
| Unit tests | 18 passed, 2 failed in preparation environment | The two failures are missing `lifelines` and `pyarrow`; run inside rebuilt Docker image |
| Survival integration test | Included as mandatory; currently fails in preparation environment | Test fails if required `lifelines` dependency is missing |
| PyArrow/NumPy compatibility test | Included as mandatory; currently fails in preparation environment | Test imports PyArrow and checks pinned NumPy/PyArrow versions |
| Docker Compose syntax | Parsed | This does not prove the image builds or services start |
| Presentation | 10 slides | PowerPoint structure validated |
| ZIP integrity | Passed | Archive integrity checked after packaging |
| Public dataset download | Not run successfully here | Runtime could not resolve external hosts; data directory has no downloaded source data |
| Full real-data model training | Not verified | No real-data metrics/artifacts are available from this preparation run |
| Live PostgreSQL integration | Not verified | Requires Docker and a running database |
| Browser-level Streamlit / API smoke test | Not verified end to end | Must run after startup on a machine with Docker and internet access |

## Required survival integration test

The test `test_individual_survival_curves_with_required_lifelines_dependency` in `tests/test_core.py` fits Kaplan–Meier and Cox models on deterministic synthetic survival data, then asserts that an individual curve and customer-level 90-day survival estimates are produced with probabilities in [0, 1]. It is intentionally not skipped when `lifelines` is missing. The Docker image installs `requirements.txt`, which includes `lifelines>=0.30,<1`. The image build imports `lifelines`, NumPy and PyArrow and checks the pinned compatibility versions.

## Predictive performance and accuracy

**No real-dataset performance numbers are reported in this report**, because the public datasets were not successfully downloaded and trained in the preparation environment. Reporting made-up accuracy would be misleading. The model pipeline is designed to save measured metrics under `artifacts/` when training succeeds.

| Model / task | Metrics to report after real-data training | Current status |
|---|---|---|
| Telco churn LightGBM | ROC-AUC, PR-AUC, precision, recall, F1, Brier score, confusion matrix; compare with prior-probability baseline | Pending real-data run |
| SHAP | Explanation method and top mean absolute SHAP features; fallback method if SHAP fails | Pending real-data run |
| Retail 90-day revenue / CLV proxy | MAE and other saved regression metrics versus mean baseline; describe assumed contribution margin | Pending real-data run |
| Hillstrom uplift | Held-out uplift-by-decile / cumulative targeting results; compare treatment and control response | Pending real-data run |
| Survival | Kaplan–Meier median if estimable, Cox concordance index, proportional-hazards diagnostics | Pending run with `lifelines` installed |
| Segmentation | Segment sizes and profiles; interpretability review | Pending real-data run |

## Reproduce the results

From the project root on a machine with internet access and Docker installed:

```bash
cp .env.example .env
# Edit .env and set a strong PostgreSQL password.
docker compose up --build -d
docker compose exec app python -m src.datasets --dataset all
docker compose exec app python -m src.train_real
docker compose exec app python -m pytest -q
docker compose exec app python -m pytest -q tests/test_core.py::test_individual_survival_curves_with_required_lifelines_dependency
docker compose exec app python scripts/smoke_test.py
```

Inspect `artifacts/training_summary.json`, `artifacts/telco_evaluation.json`, `artifacts/retail_clv_evaluation.json`, `artifacts/hillstrom_uplift_evaluation.json`, and `artifacts/retail_km_median_survival.json`. Only copy metrics from these generated files into a submission after checking the dataset row counts and successful completion.

## Interpretation limitations

- IBM Telco is cross-sectional and has no complaint log or longitudinal usage history.
- Hillstrom estimates email-campaign conversion response, not a causal retention effect for telecom customers.
- Retail 90-day inactivity is a survival-event proxy, not confirmed contractual churn.
- The annualized contribution CLV proxy depends on an assumed margin and is not a calibrated lifetime value.
- Uplift and ROI are estimates; a randomized campaign is needed to confirm incremental business impact.

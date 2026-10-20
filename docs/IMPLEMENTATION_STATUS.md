# Implementation status (pre-submission)

## Implemented in code

- Runtime dataset downloads and normalization for IBM Telco, Hillstrom, and UCI Online Retail II.
- Telco churn LightGBM workflow with held-out metrics, baseline comparison, SHAP attempt, and explicit explanation fallback.
- Leakage-aware retail forward-90-day revenue model, mean-baseline comparison, and pre-cutoff purchase/revenue trend features.
- Kaplan–Meier/Cox workflow, PH diagnostics, and individual survival curves for customers with at least 30 days of baseline follow-up. Other customers receive an explicit insufficient-follow-up status.
- CausalML `BaseTClassifier` attempt, transparent scikit-learn T-learner fallback, held-out uplift deciles/cumulative curve, full-cohort uplift scores, and uplift-guided campaign ROI.
- Telecom and retail KMeans segmentation, PostgreSQL schema/migrations/loading, Streamlit screens, FastAPI endpoints, integration smoke test, docs, and 10-slide deck.
- Optional complaint and longitudinal usage feature fields are supported if a richer dataset provides them. They are not present in IBM Telco and are not fabricated.

## Verified in this preparation environment

- `python -m compileall -q .`: passed.
- Historical preparation-environment run: 18 passed, 1 skipped, 6 non-fatal warnings. The survival test has since been changed to mandatory and must be rerun in the Docker container.
- Docker Compose YAML parsed and contains `db` and `app` services.
- PowerPoint opens and contains 10 slides.
- ZIP archive integrity is checked after packaging.

## Not verified here

- Dataset downloads: this runtime cannot resolve external hosts (DNS/network is unavailable); `data/` contains no downloaded real datasets.
- Full real-data model training and resulting evaluation metrics.
- Cox execution: `lifelines>=0.30,<1` is declared in `requirements.txt`. The survival integration test no longer skips if the dependency is missing; Docker image build now verifies that `lifelines` imports successfully. The mandatory test still needs to be run in the rebuilt container.
- Docker image build/start, live PostgreSQL writes, Streamlit browser behavior, and end-to-end smoke test.

## Required before saying the project is fully accepted

Run the Docker commands in `README.md` from a machine with Docker and internet access. Confirm successful downloads, inspect all metrics and diagnostics, fix any runtime failures, and save the resulting artifacts. Do not present synthetic test results as results from the public datasets.


## Dependency compatibility fix (2026-10-20)

The Streamlit dataframe crash caused by PyArrow requiring NumPy 2.x while the project pins NumPy 1.26.4 is addressed by pinning `pyarrow==18.1.0` alongside `numpy==1.26.4`. The Docker image build now imports both packages and asserts the exact expected versions. A regression test `test_streamlit_pyarrow_numpy_versions_are_compatible` was added. Rebuild the app image with `docker compose build --no-cache app` before restarting. This environment did not have PyArrow installed, so the new import-level regression test has not been executed here; it is intended to run in the rebuilt Docker image.

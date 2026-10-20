import pandas as pd
import pytest
from src.metrics import retention_economics
from src.feature_engineering import prepare_telco,build_retail_survival
from src.clustering import fit_segments

def test_roi_math():
    r=retention_economics(100,200,.08,5000)
    assert r["offer_cost"]==20000 and r["incremental_saves"]==pytest.approx(8)
    assert r["net_benefit"]==pytest.approx(20000) and r["roi_pct"]==pytest.approx(100)

def test_invalid_lift():
    with pytest.raises(ValueError): retention_economics(10,2,1.5,100)

def test_telco_mapping():
    raw=pd.DataFrame({"customerID":["x","y"],"tenure":[2,3],"MonthlyCharges":[20,30],
      "TotalCharges":["40",""],"Contract":["Month-to-month","One year"],
      "InternetService":["DSL","No"],"TechSupport":["No","Yes"],
      "PaperlessBilling":["Yes","No"],"Churn":["Yes","No"]})
    d=prepare_telco(raw)
    assert list(d.churn)==[1,0] and d.total_charges.notna().all()

def test_survival_proxy_event_censoring():
    dates=pd.to_datetime(["2020-01-01","2020-01-10","2020-04-20","2020-04-25"])
    d=pd.DataFrame({"InvoiceDate":dates,"Customer ID":[1,1,2,2],"Quantity":[1]*4,"Price":[2]*4,
      "Invoice":["a","b","c","d"]})
    out=build_retail_survival(d,inactivity_days=30)
    assert set(out.event_observed).issubset({0,1})
    assert (out.duration_days>0).all()

def test_clustering_runs():
    d=pd.DataFrame({"tenure_months":[1,2,20,22,40,42],"monthly_charges":[20,22,80,82,120,122],
                    "total_charges":[20,44,1600,1804,4800,5124],"churn":[1,1,0,0,0,0]})
    _,res,profile=fit_segments(d,3)
    assert "segment_id" in res and len(profile)>0


def test_survival_event_duration_stops_at_inactivity_threshold():
    dates = pd.to_datetime(["2020-01-01", "2020-01-10", "2020-04-20", "2020-04-25"])
    raw = pd.DataFrame({"InvoiceDate": dates, "Customer ID": [1, 1, 2, 2],
                        "Quantity": [1] * 4, "Price": [2] * 4,
                        "Invoice": ["a", "b", "c", "d"]})
    out = build_retail_survival(raw, inactivity_days=30).set_index("customer_id")
    # Customer 1 is inactive at the observation cutoff; event duration is first-to-last + 30.
    assert int(out.loc["1", "event_observed"]) == 1
    assert int(out.loc["1", "duration_days"]) == 39
    assert int(out.loc["1", "baseline_orders"]) == 2
    assert float(out.loc["1", "baseline_revenue"]) == pytest.approx(4.0)
    # Customer 2 has only 5 days of recency and is administratively censored.
    assert int(out.loc["2", "event_observed"]) == 0
    assert int(out.loc["2", "duration_days"]) == 6


def test_clv_forward_window_uses_only_pre_cutoff_features():
    from src.clv import build_forward_clv_frame
    rows = []
    for customer in range(1, 41):
        rows.extend([
            {"InvoiceDate": "2024-02-10", "Customer ID": customer, "Quantity": 1,
             "Price": float(customer), "Invoice": f"h1-{customer}"},
            {"InvoiceDate": "2024-03-10", "Customer ID": customer, "Quantity": 2,
             "Price": float(customer), "Invoice": f"h2-{customer}"},
        ])
        if customer % 2 == 0:
            rows.append({"InvoiceDate": "2024-05-01", "Customer ID": customer,
                         "Quantity": 1, "Price": float(customer * 2), "Invoice": f"f-{customer}"})
    # Guarantee the future window ends at June 30.
    rows.append({"InvoiceDate": "2024-06-30", "Customer ID": 40,
                 "Quantity": 1, "Price": 80.0, "Invoice": "last-date"})
    frame, cutoff, end = build_forward_clv_frame(pd.DataFrame(rows), horizon_days=90)
    assert cutoff < pd.Timestamp("2024-05-01")
    assert end > pd.Timestamp("2024-06-30")
    assert len(frame) == 40
    assert frame["actual_90d_revenue"].notna().all()
    assert frame.loc[frame.customer_id == "1", "actual_90d_revenue"].iloc[0] == 0
    assert frame.loc[frame.customer_id == "2", "actual_90d_revenue"].iloc[0] > 0


def test_clv_forecast_reports_comparison_to_baseline():
    from src.clv import fit_clv_forecast
    rows = []
    for customer in range(1, 61):
        rows.extend([
            {"InvoiceDate": "2024-02-10", "Customer ID": customer, "Quantity": 1,
             "Price": float(customer), "Invoice": f"h1-{customer}"},
            {"InvoiceDate": "2024-03-10", "Customer ID": customer, "Quantity": 2,
             "Price": float(customer), "Invoice": f"h2-{customer}"},
        ])
        if customer % 3 != 0:
            rows.append({"InvoiceDate": "2024-05-01", "Customer ID": customer,
                         "Quantity": 1, "Price": float(customer * 2), "Invoice": f"f-{customer}"})
    rows.append({"InvoiceDate": "2024-06-30", "Customer ID": 60,
                 "Quantity": 1, "Price": 120.0, "Invoice": "last-date"})
    model, scored, summary = fit_clv_forecast(pd.DataFrame(rows), horizon_days=90, contribution_margin=.3)
    assert len(scored) == 60
    assert (scored.predicted_90d_revenue >= 0).all()
    assert (scored.annualized_contribution_clv_proxy >= 0).all()
    assert summary["test_customers"] > 0
    assert "mae" in summary["test_metrics"] and "mae" in summary["mean_baseline_test_metrics"]


def test_api_health_roi_and_input_validation():
    from fastapi.testclient import TestClient
    from src.api import app
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    response = client.post("/simulate/roi", json={
        "eligible_customers": 100, "offer_cost_per_customer": 5,
        "incremental_lift": 0.1, "contribution_value_per_save": 100
    })
    assert response.status_code == 200
    assert response.json()["net_benefit"] == pytest.approx(500)
    invalid = client.post("/simulate/roi", json={
        "eligible_customers": 100, "offer_cost_per_customer": 5,
        "incremental_lift": 1.5, "contribution_value_per_save": 100
    })
    assert invalid.status_code == 422


def test_churn_model_scores_raw_customer_rows():
    from src.modeling import train_model, score_customers
    rows = []
    for i in range(80):
        churn = int(i % 3 == 0)
        rows.append({"tenure_months": i + 1, "monthly_charges": 25 + (i % 5) * 10,
                     "total_charges": (i + 1) * (25 + (i % 5) * 10),
                     "contract": "Month-to-month" if churn else "One year",
                     "internet_service": "Fiber optic" if i % 2 else "DSL",
                     "tech_support": "No" if churn else "Yes",
                     "paperless_billing": "Yes" if i % 2 else "No", "churn": churn})
    df = pd.DataFrame(rows)
    model, metrics = train_model(df)
    probs = score_customers(model, df.drop(columns=["churn"]).head(5))
    assert len(probs) == 5 and ((probs >= 0) & (probs <= 1)).all()
    assert "roc_auc" in metrics and "baseline_pr_auc" in metrics


def test_uplift_model_outputs_heldout_scores():
    from sklearn.ensemble import RandomForestClassifier
    from src.uplift import fit_uplift
    rng = __import__("numpy").random.default_rng(7)
    n = 500
    x = rng.normal(size=n)
    treatment = rng.integers(0, 2, size=n)
    p = 1 / (1 + __import__("numpy").exp(-(-0.4 + 0.6*x + treatment*(0.8 - 0.5*x))))
    y = rng.binomial(1, p)
    data = pd.DataFrame({"x": x, "treatment": treatment, "outcome": y})
    _, scored, summary = fit_uplift(data, ["x"], "treatment", "outcome",
        RandomForestClassifier(n_estimators=30, min_samples_leaf=5, random_state=7), seed=7)
    assert "predicted_uplift" in scored
    assert len(scored) > 0 and "observed_uplift_by_decile" in summary


def test_database_url_handles_reserved_password_characters(monkeypatch):
    from src.db import _configured_url
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("POSTGRES_HOST", "db")
    monkeypatch.setenv("POSTGRES_PORT", "5432")
    monkeypatch.setenv("POSTGRES_DB", "churn_intelligence")
    monkeypatch.setenv("POSTGRES_USER", "matrisk")
    monkeypatch.setenv("POSTGRES_PASSWORD", "long@secret/path#42")
    url = _configured_url()
    assert url.password == "long@secret/path#42"
    assert url.host == "db"
    assert url.database == "churn_intelligence"


def test_zero_cost_roi_is_json_safe_and_undefined():
    result = retention_economics(100, 0, 0.1, 100)
    assert result["offer_cost"] == 0
    assert result["roi_pct"] is None
    assert result["net_benefit"] == pytest.approx(1000)


def test_uplift_campaign_roi_uses_positive_uplift_and_explicit_discount_costs():
    from src.roi import uplift_campaign_roi
    scored = pd.DataFrame({
        "predicted_uplift": [0.20, 0.10, -0.05, 0.01],
        "p_treat": [0.30, 0.25, 0.10, 0.20],
        "p_control": [0.10, 0.15, 0.15, 0.19],
    })
    result = uplift_campaign_roi(scored, targeted_fraction=0.5,
        discount_per_conversion=5, contribution_per_conversion=40,
        contact_cost_per_customer=1, contribution_margin=0.5)
    assert result["targeted_customers"] == 2
    assert result["expected_incremental_conversions"] == pytest.approx(0.30)
    assert result["contact_cost"] == pytest.approx(2)
    assert result["discount_cost"] == pytest.approx(2.75)
    assert result["expected_incremental_contribution"] == pytest.approx(6)
    assert result["net_benefit"] == pytest.approx(1.25)


def test_telco_feature_engineering_uses_available_services_without_fabricating_history():
    from src.feature_engineering import prepare_telco
    raw = pd.DataFrame({
        "customerID": ["a", "b"], "tenure": [1, 12], "MonthlyCharges": [40, 80],
        "TotalCharges": [40, 960], "Contract": ["Month-to-month", "One year"],
        "InternetService": ["DSL", "Fiber optic"], "TechSupport": ["No", "Yes"],
        "PaperlessBilling": ["Yes", "No"], "PhoneService": ["Yes", "Yes"],
        "OnlineSecurity": ["No", "Yes"], "Churn": ["Yes", "No"]
    })
    d = prepare_telco(raw)
    assert list(d["service_count"]) == [1, 3]
    assert "usage_trend" not in d.columns and "complaint_count" not in d.columns
    assert d.loc[0, "charges_per_tenure_month"] == pytest.approx(40)


def test_retail_forward_clv_contains_leakage_safe_purchase_trends():
    from src.clv import build_forward_clv_frame
    rows = []
    for customer in range(1, 41):
        rows.append({"InvoiceDate": "2024-01-10", "Customer ID": customer, "Quantity": 1,
                     "Price": 10.0, "Invoice": f"old-{customer}"})
        rows.append({"InvoiceDate": "2024-04-10", "Customer ID": customer, "Quantity": 2,
                     "Price": 10.0, "Invoice": f"recent-{customer}"})
        rows.append({"InvoiceDate": "2024-06-30", "Customer ID": customer, "Quantity": 1,
                     "Price": 20.0, "Invoice": f"future-{customer}"})
    frame, cutoff, _ = build_forward_clv_frame(pd.DataFrame(rows), horizon_days=90)
    assert cutoff < pd.Timestamp("2024-06-30")
    for col in ["recent_90d_orders", "prior_90d_orders", "order_frequency_trend", "revenue_trend"]:
        assert col in frame
    assert (frame["recent_90d_orders"] >= 0).all()
    assert (frame["prior_90d_orders"] >= 0).all()


def test_causalml_wrapper_returns_full_population_uplift_scores_or_documented_fallback():
    from sklearn.ensemble import RandomForestClassifier
    from src.uplift import fit_causalml_t_learner, score_uplift_customers
    rng = __import__("numpy").random.default_rng(13)
    n = 500
    x = rng.normal(size=n)
    treatment = rng.integers(0, 2, size=n)
    probability = 1 / (1 + __import__("numpy").exp(-(-1 + .4 * x + treatment * .8)))
    outcome = rng.binomial(1, probability)
    data = pd.DataFrame({"x": x, "treatment": treatment, "conversion": outcome})
    model, heldout, summary = fit_causalml_t_learner(data, ["x"], "treatment", "conversion", seed=13)
    full_scores = score_uplift_customers(model, data, ["x"])
    assert len(heldout) > 0 and len(full_scores) == n
    assert full_scores["predicted_uplift"].notna().all()
    assert summary["method"] in {"CausalML BaseTClassifier", "sklearn T-learner fallback"}


def test_individual_survival_curves_with_required_lifelines_dependency():
    # lifelines is a required production dependency in requirements.txt and Docker.
    # Do not skip this integration test: missing lifelines must fail the test suite.
    import lifelines
    from src.survival import fit_survival_models, individual_survival_predictions
    rng = __import__("numpy").random.default_rng(42)
    n = 120
    frame = pd.DataFrame({"customer_id": [str(i) for i in range(n)],
        "duration_days": rng.integers(35, 800, size=n),
        "event_observed": rng.integers(0, 2, size=n),
        "baseline_orders": rng.integers(1, 8, size=n),
        "baseline_revenue": rng.lognormal(3, 1, size=n)})
    # Ensure both event classes are present.
    frame.loc[0, "event_observed"] = 0; frame.loc[1, "event_observed"] = 1
    km, cox = fit_survival_models(frame)
    curves, summary = individual_survival_predictions(cox, frame, max_days=365)
    assert set(curves.customer_id) == set(frame.customer_id)
    assert curves.survival_probability.between(0, 1).all()
    assert len(summary) == n
    assert summary.survival_probability_90d.between(0, 1).all()


def test_api_customer_schema_keeps_optional_training_columns_for_inference():
    from src.api import Customer
    from src.modeling import train_model, score_customers
    rows = []
    for i in range(100):
        rows.append({"tenure_months": i + 1, "monthly_charges": 30 + i % 7,
            "total_charges": (i + 1) * (30 + i % 7), "contract": "Month-to-month" if i % 2 else "One year",
            "internet_service": "Fiber optic" if i % 3 else "DSL", "tech_support": "Yes" if i % 4 else "No",
            "paperless_billing": "Yes" if i % 2 else "No", "service_count": i % 8,
            "monthly_charge_per_service": float(30 + i % 7) / max(1, i % 8),
            "charges_per_tenure_month": float(30 + i % 7), "senior_citizen": i % 2,
            "payment_method": "Electronic check" if i % 2 else "Bank transfer",
            "gender": "Female" if i % 2 else "Male", "partner": "Yes" if i % 3 else "No",
            "dependents": "No", "online_security": "No", "online_backup": "Yes",
            "device_protection": "No", "streaming_tv": "Yes", "streaming_movies": "No",
            "phone_service": "Yes", "multiple_lines": "No", "churn": int(i % 3 == 0)})
    model, _ = train_model(pd.DataFrame(rows))
    customer = Customer(tenure_months=12, monthly_charges=75, total_charges=900,
        contract="Month-to-month", internet_service="Fiber optic", tech_support="No", paperless_billing="Yes")
    scored = score_customers(model, pd.DataFrame([customer.model_dump()]))
    assert len(scored) == 1 and 0 <= scored[0] <= 1


def test_streamlit_pyarrow_numpy_versions_are_compatible():
    # Streamlit's st.dataframe relies on PyArrow. These versions are pinned
    # because newer PyArrow releases can require NumPy 2.x, conflicting with
    # this project's NumPy 1.26.4 and scientific-ML dependency set.
    import numpy as np
    import pyarrow as pa
    assert np.__version__ == "1.26.4"
    assert pa.__version__ == "18.1.0"


def test_dataset_download_tls_uses_configured_ca_bundle_and_never_disables_verification(tmp_path, monkeypatch):
    from src import datasets
    ca_file = tmp_path / "trusted-root.pem"
    ca_file.write_text("test CA bundle")
    monkeypatch.setenv("MATRISK_CA_BUNDLE", str(ca_file))
    merged = datasets._tls_verify_setting()
    assert merged == "/tmp/matrisk-ca-bundle.pem"
    merged_bytes = __import__("pathlib").Path(merged).read_bytes()
    assert ca_file.read_bytes() in merged_bytes
    monkeypatch.delenv("MATRISK_CA_BUNDLE")
    assert datasets._tls_verify_setting() is True
    monkeypatch.setenv("MATRISK_CA_BUNDLE", str(tmp_path / "missing.pem"))
    with pytest.raises(FileNotFoundError, match="trusted root/intermediate CA bundle"):
        datasets._tls_verify_setting()

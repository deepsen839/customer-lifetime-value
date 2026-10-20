"""End-to-end smoke test for a running Compose app after full-data training."""
import sys
import requests
from sqlalchemy import text
from src.db import get_engine

BASE = "http://127.0.0.1:8000"

def require(condition, message):
    if not condition: raise RuntimeError(message)

def main():
    health = requests.get(f"{BASE}/health", timeout=10)
    require(health.status_code == 200 and health.json().get("status") == "ok", "API health check failed")
    roi = requests.post(f"{BASE}/simulate/roi", json={"eligible_customers": 100, "offer_cost_per_customer": 5,
        "incremental_lift": 0.10, "contribution_value_per_save": 100}, timeout=10)
    require(roi.status_code == 200, f"Retention ROI endpoint failed: {roi.status_code} {roi.text}")
    result = roi.json()
    require(abs(result["offer_cost"] - 500) < 1e-6 and abs(result["incremental_saves"] - 10) < 1e-6 and abs(result["net_benefit"] - 500) < 1e-6, "Retention ROI arithmetic mismatch")
    customer = {"tenure_months": 12, "monthly_charges": 75.0, "total_charges": 900.0,
        "contract": "Month-to-month", "internet_service": "Fiber optic", "tech_support": "No", "paperless_billing": "Yes"}
    prediction = requests.post(f"{BASE}/predict/churn", json=customer, timeout=30)
    require(prediction.status_code == 200, f"Churn endpoint failed: {prediction.status_code} {prediction.text}")
    probability = prediction.json().get("churn_probability")
    require(isinstance(probability, (float, int)) and 0 <= probability <= 1, "Invalid churn probability")
    uplift = requests.post(f"{BASE}/simulate/uplift-roi", json={"targeted_fraction": .2, "discount_per_conversion": 5,
        "contribution_per_conversion": 30, "contact_cost_per_customer": .05, "contribution_margin": .3}, timeout=30)
    require(uplift.status_code == 200, f"Uplift ROI endpoint failed: {uplift.status_code} {uplift.text}")
    require("expected_incremental_conversions" in uplift.json(), "Uplift ROI response missing incremental conversions")
    from pathlib import Path
    import pandas as pd
    survival_file = Path("artifacts/retail_individual_survival_summary.csv")
    require(survival_file.exists(), "Individual survival summary artifact is missing")
    first_customer = str(pd.read_csv(survival_file, dtype={"customer_id": str}).iloc[0]["customer_id"])
    survival = requests.get(f"{BASE}/customers/{first_customer}/survival", timeout=15)
    require(survival.status_code == 200, f"Individual survival endpoint failed: {survival.status_code} {survival.text}")
    engine = get_engine(); require(engine is not None, "Database credentials are not configured")
    try:
        with engine.connect() as conn:
            tables = ("telco_customers", "hillstrom_campaign", "retail_customer_value", "retail_survival", "retail_survival_predictions")
            counts = {table: int(conn.execute(text(f'SELECT COUNT(*) FROM "{table}"')).scalar_one()) for table in tables}
            empty = [name for name, count in counts.items() if count <= 0]
            require(not empty, f"Database tables are empty: {empty}")
    finally: engine.dispose()
    print("PASS: API health, churn inference, retention/uplift ROI, individual survival endpoint, and populated PostgreSQL tables")
    print({"churn_probability": float(probability), "uplift_roi": uplift.json(), "table_rows": counts})

if __name__ == "__main__":
    try: main()
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)

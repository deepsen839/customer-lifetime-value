from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
import json
import joblib, pandas as pd
from src.modeling import score_customers
from src.metrics import retention_economics
from src.roi import uplift_campaign_roi

ROOT=Path(__file__).resolve().parents[1]
app=FastAPI(title="Customer Lifetime Value & Churn Intelligence API",version="1.0.0")

class Customer(BaseModel):
    tenure_months:int=Field(ge=0,le=1000)
    monthly_charges:float=Field(ge=0)
    total_charges:float=Field(ge=0)
    contract:str
    internet_service:str
    tech_support:str
    paperless_billing:str
    # Optional richer-source features. Keep them nullable because IBM Telco does
    # not supply longitudinal usage or complaint records.
    service_count:int|None=None
    monthly_charge_per_service:float|None=None
    charges_per_tenure_month:float|None=None
    senior_citizen:int|None=None
    usage_trend:float|None=None
    complaint_count:float|None=None
    payment_method:str|None=None
    gender:str|None=None
    partner:str|None=None
    dependents:str|None=None
    online_security:str|None=None
    online_backup:str|None=None
    device_protection:str|None=None
    streaming_tv:str|None=None
    streaming_movies:str|None=None
    phone_service:str|None=None
    multiple_lines:str|None=None

class ROIRequest(BaseModel):
    eligible_customers:int=Field(ge=0)
    offer_cost_per_customer:float=Field(ge=0)
    incremental_lift:float=Field(ge=0,le=1)
    contribution_value_per_save:float=Field(ge=0)

class UpliftROIRequest(BaseModel):
    targeted_fraction:float=Field(default=0.2,gt=0,le=1)
    discount_per_conversion:float=Field(default=5.0,ge=0)
    contribution_per_conversion:float=Field(default=30.0,ge=0)
    contact_cost_per_customer:float=Field(default=0.0,ge=0)
    contribution_margin:float=Field(default=0.3,ge=0,le=1)

@app.get("/health")
def health(): return {"status":"ok"}

@app.post("/predict/churn")
def predict_churn(customer:Customer):
    path=ROOT/"artifacts"/"telco_churn_model.joblib"
    if not path.exists(): raise HTTPException(503,"Model not trained; run python -m src.train_real")
    model=joblib.load(path)
    prob=float(score_customers(model,pd.DataFrame([customer.model_dump()]))[0])
    return {"churn_probability":prob,"risk_band":"high" if prob>=.6 else "medium" if prob>=.3 else "low"}

@app.post("/simulate/roi")
def simulate_roi(req:ROIRequest):
    return retention_economics(req.eligible_customers,req.offer_cost_per_customer,req.incremental_lift,req.contribution_value_per_save)

@app.post("/simulate/uplift-roi")
def simulate_uplift_roi(req: UpliftROIRequest):
    path = ROOT / "artifacts" / "hillstrom_uplift_customer_scores.csv"
    if not path.exists():
        raise HTTPException(503, "Uplift model is not trained; run python -m src.train_real")
    scored = pd.read_csv(path)
    try:
        return uplift_campaign_roi(scored, req.targeted_fraction, req.discount_per_conversion,
            req.contribution_per_conversion, req.contact_cost_per_customer, req.contribution_margin)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

@app.get("/customers/{customer_id}/survival")
def customer_survival(customer_id: str):
    path = ROOT / "artifacts" / "retail_individual_survival_summary.csv"
    if not path.exists():
        raise HTTPException(503, "Survival model is not trained; run python -m src.train_real")
    frame = pd.read_csv(path, dtype={"customer_id": str})
    match = frame[frame["customer_id"] == str(customer_id)]
    if match.empty:
        raise HTTPException(404, "Customer not found in the retail survival cohort")
    return json.loads(match.iloc[0].to_json())

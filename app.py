from pathlib import Path
import json
import pandas as pd
import streamlit as st
from src.datasets import get_telco,get_hillstrom,get_retail_ii,retail_customer_value
from src.metrics import retention_economics
from src.eda import profile_dataframe
from src.feature_engineering import prepare_telco,build_retail_survival
from src.db import get_engine,init_db,read_table
from src.roi import uplift_campaign_roi

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data";ART=ROOT/"artifacts"
st.set_page_config(page_title="CLV & Churn Intelligence",layout="wide")
st.title("Customer Lifetime Value & Churn Intelligence")
st.caption("Public real datasets; downloaded on first use. Optional managed PostgreSQL is configured with Streamlit Secrets. Survival uses an explicitly defined purchase-inactivity proxy.")
tabs=st.tabs(["Overview & EDA","Churn + SHAP","Survival","Uplift","CLV + Segments","ROI","PostgreSQL"])
with tabs[0]:
    p=DATA/"telco_normalized.csv"
    if p.exists(): telco=prepare_telco(pd.read_csv(p))
    else:
        with st.spinner("Downloading IBM Telco..."):
            telco=prepare_telco(get_telco());telco.to_csv(p,index=False)
    st.subheader("IBM Telco EDA")
    a,b,c,d=st.columns(4);a.metric("Customers",f"{len(telco):,}");b.metric("Churn rate",f"{telco.churn.mean():.1%}");c.metric("Median tenure",f"{telco.tenure_months.median():.0f} months");d.metric("Duplicate rows",f"{telco.duplicated().sum():,}")
    st.write("Missing values by column");st.dataframe(telco.isna().sum().rename("missing").to_frame())
    st.write("Numeric summary");st.dataframe(telco.select_dtypes("number").describe())
    st.bar_chart(telco.groupby("contract",observed=False).churn.mean())
with tabs[1]:
    st.subheader("Churn model evaluation and explanations")
    ep=ART/"telco_evaluation.json"
    if ep.exists(): st.json(json.loads(ep.read_text()))
    else: st.info("Train first: `python -m src.train_real`.")
    sp=ART/"telco_shap_global.csv"
    if sp.exists(): st.dataframe(pd.read_csv(sp),use_container_width=True)
    else: st.info("Global SHAP file is produced during training when SHAP is compatible; permutation importance is the documented fallback.")
    cp=ART/"telco_customer_segments.csv"
    if cp.exists(): st.dataframe(pd.read_csv(cp).head(100),use_container_width=True)
with tabs[2]:
    p=ART/"retail_km_median_survival.json"
    if p.exists():
        st.json(json.loads(p.read_text()))
        st.caption("Operational event: 90 days without purchase by the end of the transaction observation window. This is not telecom account cancellation.")
        if (ART/"retail_cox_summary.csv").exists(): st.dataframe(pd.read_csv(ART/"retail_cox_summary.csv",index_col=0))
        pred_path=ART/"retail_individual_survival_summary.csv"
        curve_path=ART/"retail_individual_survival_curves.csv"
        if pred_path.exists():
            predictions=pd.read_csv(pred_path,dtype={"customer_id":str})
            st.subheader("Individual customer survival estimates")
            st.caption("Model-estimated probability of remaining purchase-active under the 90-day inactivity proxy; not an individual guarantee.")
            st.dataframe(predictions.sort_values("survival_probability_90d").head(100),use_container_width=True)
        if curve_path.exists():
            st.download_button("Download all individual survival curves",curve_path.read_bytes(),file_name=curve_path.name,mime="text/csv")
    else: st.info("Run training to construct Online Retail II survival proxy and fit Kaplan–Meier/Cox.")
with tabs[3]:
    p=ART/"hillstrom_uplift_evaluation.json"
    if p.exists(): st.json(json.loads(p.read_text()))
    else: st.info("Run training to fit the randomized Hillstrom treatment/control model and held-out uplift summary.")
    score_path=ART/"hillstrom_uplift_customer_scores.csv"
    if score_path.exists():
        scores=pd.read_csv(score_path)
        st.subheader("Customers ranked by predicted incremental conversion response")
        st.dataframe(scores.sort_values("predicted_uplift",ascending=False).head(100),use_container_width=True)
        st.caption("The score is estimated conversion uplift for women's email versus no email. It is not a telecom churn-retention effect.")
    st.caption("Held-out campaign results are experimental estimates with sampling noise; use a new randomized campaign to confirm impact.")
with tabs[4]:
    p=ART/"online_retail_ii_customer_value.csv"
    if p.exists():
        clv=pd.read_csv(p)
        a,b,c=st.columns(3);a.metric("Customers",f"{len(clv):,}");b.metric("Median historical revenue",f"£{clv.historical_value_proxy.median():,.2f}");c.metric("Median orders",f"{clv.orders.median():.0f}")
        forecast_path=ART/"retail_clv_evaluation.json"
        if forecast_path.exists():
            st.subheader("Forward 90-day revenue forecast validation")
            st.json(json.loads(forecast_path.read_text()))
            st.caption("Annualized contribution CLV is a scenario proxy derived from predicted 90-day revenue and an assumed margin; it is not a calibrated lifetime forecast.")
        forecast_file=ART/"retail_clv_forecast.csv"
        if forecast_file.exists():
            st.dataframe(pd.read_csv(forecast_file).sort_values("predicted_90d_revenue",ascending=False).head(100),use_container_width=True)
        else:
            st.dataframe(clv.sort_values("historical_value_proxy",ascending=False).head(100),use_container_width=True)
    else: st.info("Run training to aggregate Online Retail II customer values.")
    p=ART/"telco_segment_profiles.csv"
    if p.exists(): st.subheader("Telco cluster profiles");st.dataframe(pd.read_csv(p),use_container_width=True)
    rp=ART/"retail_segment_profiles.csv"
    if rp.exists(): st.subheader("Retail CLV segment profiles");st.dataframe(pd.read_csv(rp,index_col=0),use_container_width=True)
with tabs[5]:
    st.subheader("Retention offer ROI simulator")
    n=st.number_input("Eligible customers",1,1000000,100,10)
    cost=st.number_input("Offer cost per customer (£)",0.0,100000.0,5.0)
    lift=st.slider("Assumed incremental retention lift (%)",0.0,50.0,8.0)/100
    value=st.number_input("Contribution value per incremental save (£)",0.0,1000000.0,100.0)
    r=retention_economics(int(n),cost,lift,value)
    roi_label=f"{r['roi_pct']:.1f}% ROI" if r["roi_pct"] is not None else "ROI undefined (zero offer cost)"
    a,b,c,d=st.columns(4);a.metric("Offer cost",f"£{r['offer_cost']:,.2f}");b.metric("Incremental saves",f"{r['incremental_saves']:.1f}");c.metric("Incremental value",f"£{r['incremental_value']:,.2f}");d.metric("Net benefit",f"£{r['net_benefit']:,.2f}",delta=roi_label)
    st.caption("This retention simulator uses an explicit assumed lift. It is not learned from the Hillstrom campaign, which measures email conversion uplift rather than telecom retention.")
    st.divider()
    st.subheader("Uplift-guided campaign ROI (Hillstrom)")
    uplift_path=ART/"hillstrom_uplift_customer_scores.csv"
    if uplift_path.exists():
        scores=pd.read_csv(uplift_path)
        fraction=st.slider("Top predicted-uplift customers to target (%)",1,100,20,1)/100
        discount=st.number_input("Discount per expected conversion (£)",0.0,10000.0,5.0,0.5)
        contact=st.number_input("Contact cost per targeted customer (£)",0.0,1000.0,0.05,0.01)
        contribution=st.number_input("Contribution before discount per incremental conversion (£)",0.0,100000.0,30.0,1.0)
        margin=st.slider("Contribution margin (%)",0,100,30,5)/100
        roi=uplift_campaign_roi(scores,fraction,discount,contribution,contact,margin)
        a,b,c,d=st.columns(4)
        a.metric("Customers targeted",f"{roi['targeted_customers']:,}")
        b.metric("Expected incremental conversions",f"{roi['expected_incremental_conversions']:.1f}")
        c.metric("Total offer cost",f"£{roi['total_offer_cost']:,.2f}")
        d.metric("Expected net benefit",f"£{roi['net_benefit']:,.2f}",delta=(f"{roi['roi_pct']:.1f}% ROI" if roi['roi_pct'] is not None else "ROI undefined"))
        with st.expander("Assumptions and full ROI calculation"):
            st.json(roi)
    else:
        st.info("Run the full training pipeline to enable the uplift-guided campaign ROI simulator.")
with tabs[6]:
    engine=get_engine()
    if engine is None: st.warning("PostgreSQL is not configured. Add DATABASE_URL under Streamlit app Secrets, or use the local Docker Compose environment.")
    else:
        try:
            init_db(engine)
            table=st.selectbox("Table",["telco_customers","hillstrom_campaign","retail_customer_value","retail_survival","retail_survival_predictions"])
            st.dataframe(read_table(engine,table,1000),use_container_width=True)
        except Exception as e: st.error(f"Database query failed: {type(e).__name__}: {e}")

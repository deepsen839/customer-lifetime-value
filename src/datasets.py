from pathlib import Path
import shutil, zipfile, time, os, logging
import requests
import pandas as pd
import certifi

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
URLS = {
 "telco": "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
 "hillstrom": "https://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv",
 "retail_ii": "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip",
 "criteo": "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz",
}
def _tls_verify_setting():
    """Return a trusted custom CA bundle when configured, otherwise keep TLS verification on."""
    bundle = os.environ.get("MATRISK_CA_BUNDLE", "").strip()
    if bundle:
        ca_path = Path(bundle)
        if not ca_path.is_file():
            raise FileNotFoundError(
                f"MATRISK_CA_BUNDLE is set to {bundle!r}, but that file is not present. "
                "Mount your organization's trusted root/intermediate CA bundle as a PEM file."
            )
        # Keep public web PKI roots trusted as well as the organization's custom CA.
        merged = Path("/tmp/matrisk-ca-bundle.pem")
        merged.write_bytes(Path(certifi.where()).read_bytes().rstrip() + b"\n" + ca_path.read_bytes().lstrip())
        return str(merged)
    # requests honors REQUESTS_CA_BUNDLE/CURL_CA_BUNDLE when verify=True.
    return True


def download(url, dest, attempts=3):
    """Stream a source file to disk atomically; retry transient HTTP/network errors.

    TLS certificate validation stays enabled. For TLS-inspecting proxies, mount a trusted
    CA PEM bundle and set MATRISK_CA_BUNDLE; never use verify=False.
    """
    dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_error = None
    for attempt in range(attempts):
        try:
            with requests.get(url, stream=True, timeout=(20, 180), headers={"User-Agent": "CLV-Churn-Intelligence/1.0"}, verify=_tls_verify_setting()) as r:
                r.raise_for_status()
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        if chunk: f.write(chunk)
            if not tmp.exists() or tmp.stat().st_size == 0:
                raise IOError(f"Downloaded empty file from {url}")
            tmp.replace(dest)
            return dest
        except Exception as exc:
            last_error = exc
            try: tmp.unlink(missing_ok=True)
            except Exception: pass
            if attempt + 1 < attempts: time.sleep(2 ** attempt)
    raise RuntimeError(f"Could not download dataset from {url} after {attempts} attempts: {last_error}") from last_error

def get_telco():
    p = download(URLS["telco"], DATA/"Telco-Customer-Churn.csv")
    d = pd.read_csv(p)
    d["TotalCharges"] = pd.to_numeric(d["TotalCharges"], errors="coerce").fillna(0)
    d["Churn"] = d["Churn"].map({"Yes":1, "No":0})
    d = d.rename(columns={"customerID":"customer_id","tenure":"tenure_months",
        "MonthlyCharges":"monthly_charges","TotalCharges":"total_charges",
        "Contract":"contract","InternetService":"internet_service",
        "TechSupport":"tech_support","PaperlessBilling":"paperless_billing","Churn":"churn"})
    if d.empty or not {"customer_id", "churn"}.issubset(d.columns):
        raise ValueError("IBM Telco source returned an empty or unexpected dataset")
    d["dataset_source"] = "IBM Telco Customer Churn"
    return d

def get_hillstrom():
    p = download(URLS["hillstrom"], DATA/"hillstrom.csv")
    d = pd.read_csv(p)
    if not {"segment","visit"}.issubset(d.columns):
        raise ValueError("Unexpected Hillstrom schema")
    d["treatment"] = d["segment"].map({"Mens E-Mail": 1, "Womens E-Mail": 1, "No E-Mail": 0})
    for col in ("visit", "conversion"):
        if col in d: d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0).astype(int)
    if d["treatment"].isna().any():
        raise ValueError("Unexpected Hillstrom segment values")
    d["campaign_row_id"] = range(1, len(d) + 1)
    return d

def get_retail_ii():
    p = download(URLS["retail_ii"], DATA/"online_retail_ii.zip")
    with zipfile.ZipFile(p) as z:
        names = [n for n in z.namelist() if n.lower().endswith((".xlsx",".xls"))]
        if not names: raise ValueError("UCI ZIP contained no Excel workbook")
        xlsx = DATA/Path(names[0]).name
        with z.open(names[0]) as src, xlsx.open("wb") as dst: shutil.copyfileobj(src,dst)
    required_cols = ["Invoice", "Quantity", "InvoiceDate", "Price", "Customer ID"]
    sheets = pd.read_excel(xlsx, sheet_name=None, usecols=lambda c: c in set(required_cols + ["StockCode", "Description", "Country"]))
    d = pd.concat(list(sheets.values()), ignore_index=True)
    d["InvoiceDate"] = pd.to_datetime(d["InvoiceDate"], errors="coerce")
    d["Customer ID"] = pd.to_numeric(d["Customer ID"], errors="coerce")
    d["Quantity"] = pd.to_numeric(d["Quantity"], errors="coerce")
    d["Price"] = pd.to_numeric(d["Price"], errors="coerce")
    d = d.dropna(subset=["Customer ID", "InvoiceDate", "Quantity", "Price"])
    if d.empty: raise ValueError("UCI Online Retail II source contained no usable transactions")
    return d

def retail_customer_value(d):
    d = d[(d["Quantity"]>0)&(d["Price"]>0)&(~d["Invoice"].astype(str).str.startswith("C"))].copy()
    d["revenue"] = d["Quantity"]*d["Price"]
    cutoff = d["InvoiceDate"].max()+pd.Timedelta(days=1)
    d["customer_id"] = d["Customer ID"].astype("Int64").astype(str)
    out = d.groupby("customer_id").agg(first_purchase=("InvoiceDate","min"),
        last_purchase=("InvoiceDate","max"),orders=("Invoice","nunique"),
        revenue=("revenue","sum"),items=("Quantity","sum")).reset_index()
    out["recency_days"] = (cutoff-out["last_purchase"]).dt.days
    out["customer_age_days"] = (out["last_purchase"]-out["first_purchase"]).dt.days.clip(lower=1)
    out["average_order_value"] = out["revenue"]/out["orders"].clip(lower=1)
    out["historical_value_proxy"] = out["revenue"]
    return out

def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",choices=["telco","hillstrom","retail","criteo","all"],default="all")
    p.add_argument("--criteo-rows",type=int,default=500000)
    a=p.parse_args()
    requested = [a.dataset] if a.dataset != "all" else ["telco", "hillstrom", "retail"]
    failures = []
    for name in requested:
        try:
            if name == "telco":
                d=get_telco(); d.to_csv(DATA/"telco_normalized.csv",index=False); print("IBM Telco",d.shape)
            elif name == "hillstrom":
                d=get_hillstrom(); d.to_csv(DATA/"hillstrom_normalized.csv",index=False); print("Hillstrom",d.shape)
            elif name == "retail":
                d=get_retail_ii(); d.to_pickle(DATA/"online_retail_ii_clean.pkl")
                c=retail_customer_value(d); c.to_csv(DATA/"online_retail_ii_clv.csv",index=False)
                print("Online Retail II",d.shape,"customer aggregates",c.shape)
            elif name == "criteo":
                pth=download(URLS["criteo"],DATA/"criteo-uplift.csv.gz")
                d=pd.read_csv(pth,compression="gzip",nrows=a.criteo_rows)
                d.to_csv(DATA/"criteo_uplift_sample.csv",index=False); print("Criteo sample",d.shape)
        except Exception as exc:
            failures.append((name, str(exc)))
            print(f"ERROR: dataset '{name}' failed: {exc}")
    if failures:
        print("\nDataset download/processing summary:")
        print(f"Succeeded: {len(requested) - len(failures)}/{len(requested)}")
        for name, error in failures:
            print(f" - {name}: {error}")
        print("\nFor self-signed certificate errors, obtain your network/organization's trusted CA PEM, "
              "place it at ./certs/custom-ca.pem, and rebuild/recreate the app container. "
              "Do not disable TLS verification.")
        raise SystemExit(1)

if __name__=="__main__": main()

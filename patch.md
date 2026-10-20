python - <<'PY'
from pathlib import Path

path = Path("src/datasets.py")
source = path.read_text()

start = source.index("def get_hillstrom():")
end = source.index("\ndef get_retail_ii():", start)

replacement = '''def get_hillstrom():
    # Prefer the copy downloaded successfully through scikit-uplift.
    local_path = DATA / "hillstrom_sklift.csv"
    if local_path.exists() and local_path.stat().st_size > 0:
        p = local_path
    else:
        p = download(URLS["hillstrom"], DATA / "hillstrom.csv")

    d = pd.read_csv(p)
    if not {"segment", "visit"}.issubset(d.columns):
        raise ValueError(
            "Hillstrom data must contain observed 'segment' and 'visit' columns"
        )

    d["treatment"] = d["segment"].map({
        "Mens E-Mail": 1,
        "Womens E-Mail": 1,
        "No E-Mail": 0,
    })
    d["visit"] = pd.to_numeric(d["visit"], errors="coerce")
    if d["visit"].isna().any() or not d["visit"].isin([0, 1]).all():
        raise ValueError("Hillstrom visit outcome must contain only 0/1 values")
    if d["treatment"].isna().any():
        raise ValueError("Unexpected Hillstrom segment values")

    d["visit"] = d["visit"].astype(int)
    d["campaign_row_id"] = range(1, len(d) + 1)
    return d
'''

path.write_text(source[:start] + replacement + source[end:])
print("Updated src/datasets.py")
PY



python - <<'PY'
from pathlib import Path

path = Path("src/train_real.py")
source = path.read_text()

old = '''    if "conversion" not in hillstrom:
        raise ValueError("Hillstrom dataset must contain conversion outcome for offer-profitability modeling")
    uplift_model, uplift_test, uplift_summary = fit_causalml_t_learner(
        hillstrom, uplift_features, "treatment", "conversion", seed=42)
'''

new = '''    # The scikit-uplift copy provides observed visit outcomes, not conversion/spend.
    # Train and report visit-response uplift; do not fabricate conversion labels.
    if "visit" not in hillstrom:
        raise ValueError("Hillstrom dataset must contain the observed visit outcome")
    uplift_model, uplift_test, uplift_summary = fit_causalml_t_learner(
        hillstrom, uplift_features, "treatment", "visit", seed=42)
    uplift_summary["outcome"] = "visit"
    uplift_summary["outcome_interpretation"] = (
        "Observed visit response; not conversion or revenue uplift"
    )
'''

if old not in source:
    raise SystemExit(
        "Expected training block not found. No changes made to src/train_real.py."
    )

path.write_text(source.replace(old, new, 1))
print("Updated src/train_real.py")
PY


python - <<'PY'
from src.datasets import get_hillstrom

d = get_hillstrom()
print("Shape:", d.shape)
print("Required columns:", {"segment", "visit", "treatment"}.issubset(d.columns))
print("Segments:")
print(d["segment"].value_counts())
print("Visit rates by segment:")
print(d.groupby("segment")["visit"].mean())
print("Missing treatment labels:", d["treatment"].isna().sum())
PY


python - <<'PY'
import json
from pathlib import Path

for name in ("telco_evaluation.json",
             "hillstrom_uplift_evaluation.json",
             "retail_clv_evaluation.json",
             "training_summary.json",
             "run_manifest.json"):
    p = Path("artifacts") / name
    print(f"\n{name}: {'EXISTS' if p.exists() else 'MISSING'}")
    if p.exists():
        try:
            print(json.dumps(json.loads(p.read_text()), indent=2)[:1500])
        except Exception as exc:
            print("Could not read JSON:", exc)
PY
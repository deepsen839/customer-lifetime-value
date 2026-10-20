import json
from pathlib import Path
import pandas as pd

def profile_dataframe(df, target=None):
    report = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_by_column": {k:int(v) for k,v in df.isna().sum().items()},
        "dtypes": {k:str(v) for k,v in df.dtypes.items()},
        "numeric_summary": df.select_dtypes(include="number").describe().to_dict(),
    }
    if target and target in df:
        report["target_counts"] = {str(k):int(v) for k,v in df[target].value_counts(dropna=False).items()}
        report["target_rate"] = float(df[target].mean()) if pd.api.types.is_numeric_dtype(df[target]) else None
    return report

def save_profile(df, path, target=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(profile_dataframe(df,target),indent=2,default=str),encoding="utf-8")

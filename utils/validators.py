"""
Validates incoming call batch CSVs against the expected schema
before they're processed by the pipeline.
"""
import pandas as pd

REQUIRED_COLUMNS = [
    "customer_id", "call_duration_sec", "calls_per_day", "night_call_ratio",
    "sim_age_days", "recharge_amount", "unique_contacts",
    "location_changes", "device_changes", "in_out_ratio"
]

def validate_schema(df: pd.DataFrame) -> tuple[bool, str]:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return False, f"Missing columns: {missing}"
    if df[REQUIRED_COLUMNS[1:]].isnull().all().any():
        return False, "One or more numeric columns are entirely null"
    return True, "OK"

def clean_batch(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["customer_id"])
    numeric_cols = REQUIRED_COLUMNS[1:]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=numeric_cols, how="all")
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    return df.reset_index(drop=True)
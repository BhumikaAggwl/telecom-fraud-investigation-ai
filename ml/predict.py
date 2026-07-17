"""
Loads trained XGBoost model and generates fraud predictions
with probability scores for new call records.
"""
import pandas as pd
import joblib

MODEL_PATH = "models/xgb_fraud_model.joblib"
FEATURES = [
    "call_duration_sec", "calls_per_day", "night_call_ratio",
    "sim_age_days", "recharge_amount", "unique_contacts",
    "location_changes", "device_changes", "in_out_ratio"
]

def load_model():
    return joblib.load(MODEL_PATH)

def predict_batch(df: pd.DataFrame, model=None) -> pd.DataFrame:
    if model is None:
        model = load_model()
    X = df[FEATURES]
    df = df.copy()
    df["fraud_probability"] = model.predict_proba(X)[:, 1]
    df["fraud_prediction"] = model.predict(X)
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/processed/telecom_fraud_dataset.csv").sample(10, random_state=1)
    result = predict_batch(df)
    print(result[["customer_id", "fraud_probability", "fraud_prediction"]])
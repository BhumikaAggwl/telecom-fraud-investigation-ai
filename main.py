"""
Main pipeline: XGBoost prediction -> Risk Agent -> Investigation Agent -> Report Agent.
Runs on a sample of the dataset and saves generated reports to data/reports/.
"""
import os
import pandas as pd
from ml.predict import predict_batch, load_model
from agents.risk_agent import assess_risk
from agents.investigation_agent import investigate
from agents.report_agent import generate_report

DATA_PATH = "data/processed/telecom_fraud_dataset.csv"
REPORTS_DIR = "data/reports"
HIGH_RISK_THRESHOLD = 0.5  # only investigate cases the model flags as fraud

def run_pipeline(n_samples: int = 5):
    os.makedirs(REPORTS_DIR, exist_ok=True)

    model = load_model()
    df = pd.read_csv(DATA_PATH).sample(50, random_state=7)
    df = predict_batch(df, model=model)

    # Only send high-risk/uncertain cases to the LLM agents (saves tokens + cost)
    flagged = df[df["fraud_probability"] >= HIGH_RISK_THRESHOLD].head(n_samples)

    if flagged.empty:
        print("No high-risk cases found in this sample.")
        return

    for _, row in flagged.iterrows():
        print(f"\nProcessing {row['customer_id']} (fraud_prob={row['fraud_probability']:.3f})...")

        risk_result = assess_risk({
            "fraud_probability": round(row["fraud_probability"], 3),
            "call_duration_sec": row["call_duration_sec"],
            "calls_per_day": row["calls_per_day"],
            "night_call_ratio": round(row["night_call_ratio"], 3),
            "sim_age_days": row["sim_age_days"],
            "unique_contacts": row["unique_contacts"],
            "location_changes": row["location_changes"]
        })

        investigation_result = investigate({
            "risk_assessment": risk_result,
            "calls_per_day": row["calls_per_day"],
            "night_call_ratio": round(row["night_call_ratio"], 3),
            "sim_age_days": row["sim_age_days"],
            "unique_contacts": row["unique_contacts"],
            "device_changes": row["device_changes"]
        })

        report = generate_report({
            "customer_id": row["customer_id"],
            "fraud_probability": round(row["fraud_probability"], 3),
            "risk_assessment": risk_result,
            "investigation_findings": investigation_result
        })

        out_path = os.path.join(REPORTS_DIR, f"{row['customer_id']}_report.md")
        with open(out_path, "w") as f:
            f.write(report)
        print(f"Saved -> {out_path}")

if __name__ == "__main__":
    run_pipeline()
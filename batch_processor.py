"""
Batch processing pipeline for incoming telecom call data.

Watches data/incoming_calls/ for new CSV files and automatically:
1. Validates schema
2. Cleans data
3. Runs XGBoost prediction
4. Applies business rule threshold
5. Sends only high-risk/uncertain cases to LangChain agents
6. Generates investigation reports
7. Writes a summary CSV
8. Logs everything
"""
import os
import glob
import shutil
import pandas as pd
from datetime import datetime

from ml.predict import predict_batch, load_model
from utils.validators import validate_schema, clean_batch
from utils.logger import logger
from agents.risk_agent import assess_risk
from agents.investigation_agent import investigate
from agents.report_agent import generate_report

INCOMING_DIR = "data/incoming_calls"
PROCESSED_DIR = "data/incoming_calls/processed"
REPORTS_DIR = "data/reports"
SUMMARY_PATH = "data/reports/batch_summary.csv"

HIGH_RISK_THRESHOLD = 0.6      # send straight to agents
UNCERTAIN_LOW = 0.4            # "uncertain zone" also sent to agents
MAX_AGENT_CASES_PER_BATCH = 10  # cost control


def process_file(filepath: str, model):
    filename = os.path.basename(filepath)
    logger.info(f"New batch detected: {filename}")

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        logger.error(f"Failed to read {filename}: {e}")
        return

    valid, msg = validate_schema(df)
    if not valid:
        logger.error(f"Schema validation failed for {filename}: {msg}")
        return
    logger.info(f"Schema valid for {filename}")

    df = clean_batch(df)
    logger.info(f"Cleaned {len(df)} rows from {filename}")

    df = predict_batch(df, model=model)
    logger.info(f"Prediction complete for {filename}")

    needs_review = df[
        (df["fraud_probability"] >= HIGH_RISK_THRESHOLD) |
        ((df["fraud_probability"] >= UNCERTAIN_LOW) & (df["fraud_probability"] < HIGH_RISK_THRESHOLD))
    ].head(MAX_AGENT_CASES_PER_BATCH)

    logger.info(f"{len(needs_review)} cases flagged for agent review (of {len(df)} total)")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    summary_rows = []

    for _, row in needs_review.iterrows():
        try:
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

            report_path = os.path.join(REPORTS_DIR, f"{row['customer_id']}_report.md")
            with open(report_path, "w") as f:
                f.write(report)

            summary_rows.append({
                "customer_id": row["customer_id"],
                "fraud_probability": row["fraud_probability"],
                "fraud_prediction": row["fraud_prediction"],
                "report_path": report_path,
                "source_file": filename,
                "processed_at": datetime.now().isoformat()
            })
            logger.info(f"Report generated for {row['customer_id']}")

        except Exception as e:
            logger.error(f"Agent pipeline failed for {row['customer_id']}: {e}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        if os.path.exists(SUMMARY_PATH):
            summary_df.to_csv(SUMMARY_PATH, mode="a", header=False, index=False)
        else:
            summary_df.to_csv(SUMMARY_PATH, index=False)
        logger.info(f"Summary updated -> {SUMMARY_PATH}")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    shutil.move(filepath, os.path.join(PROCESSED_DIR, filename))
    logger.info(f"Moved {filename} -> {PROCESSED_DIR}/")


def run_batch_scan():
    os.makedirs(INCOMING_DIR, exist_ok=True)
    files = glob.glob(os.path.join(INCOMING_DIR, "*.csv"))

    if not files:
        logger.info("No new batch files found.")
        return

    model = load_model()
    for filepath in files:
        process_file(filepath, model)


if __name__ == "__main__":
    run_batch_scan()
"""
Naive baseline model (Logistic Regression, no tuning, no class balancing)
to establish a "before" metric for comparison against XGBoost.
"""
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

DATA_PATH = "data/processed/telecom_fraud_dataset.csv"
FEATURES = [
    "call_duration_sec", "calls_per_day", "night_call_ratio",
    "sim_age_days", "recharge_amount", "unique_contacts",
    "location_changes", "device_changes", "in_out_ratio"
]

def run_baseline():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df["fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Deliberately naive: no scaling, no class_weight balancing, no tuning
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }
    print("Baseline (Logistic Regression, unbalanced):")
    print(json.dumps(metrics, indent=2))

    with open("models/baseline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    run_baseline()
"""
Trains XGBoost fraud classifier on telecom call data.
Saves model + prints/logs real evaluation metrics.
"""
import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from xgboost import XGBClassifier

DATA_PATH = "data/processed/telecom_fraud_dataset.csv"
MODEL_PATH = "models/xgb_fraud_model.joblib"
METRICS_PATH = "models/metrics.json"

FEATURES = [
    "call_duration_sec", "calls_per_day", "night_call_ratio",
    "sim_age_days", "recharge_amount", "unique_contacts",
    "location_changes", "device_changes", "in_out_ratio"
]

def train():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df["fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    print(json.dumps(metrics, indent=2))
    print("\n" + classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nModel saved -> {MODEL_PATH}")
    print(f"Metrics saved -> {METRICS_PATH}")

if __name__ == "__main__":
    train()
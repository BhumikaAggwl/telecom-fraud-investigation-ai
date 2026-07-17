"""
Synthetic telecom call dataset generator.
Distributions deliberately overlap so the task is realistically hard —
mirrors real fraud data where classes are not cleanly separable.
"""
import numpy as np
import pandas as pd

np.random.seed(42)

def generate_dataset(n_samples=10000, fraud_ratio=0.08):
    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud

    def normal_batch(n):
        return pd.DataFrame({
            "call_duration_sec": np.random.normal(150, 100, n).clip(1, 700),
            "calls_per_day": np.random.normal(40, 40, n).clip(1, 400),
            "night_call_ratio": np.random.beta(3, 5, n),
            "sim_age_days": np.random.normal(300, 280, n).clip(1, 2000),
            "recharge_amount": np.random.normal(250, 180, n).clip(5, 1200),
            "unique_contacts": np.random.normal(35, 35, n).clip(1, 300),
            "location_changes": np.random.poisson(2.2, n),
            "device_changes": np.random.poisson(0.8, n),
            "in_out_ratio": np.random.normal(0.9, 0.6, n).clip(0.02, 6),
            "fraud": 0
        })

    def fraud_batch(n):
        return pd.DataFrame({
            "call_duration_sec": np.random.normal(90, 90, n).clip(1, 600),
            "calls_per_day": np.random.normal(80, 70, n).clip(1, 800),
            "night_call_ratio": np.random.beta(4, 4, n),
            "sim_age_days": np.random.normal(150, 200, n).clip(0, 1500),
            "recharge_amount": np.random.normal(180, 160, n).clip(5, 900),
            "unique_contacts": np.random.normal(55, 50, n).clip(1, 400),
            "location_changes": np.random.poisson(3, n),
            "device_changes": np.random.poisson(1.3, n),
            "in_out_ratio": np.random.normal(0.5, 0.5, n).clip(0.01, 5),
            "fraud": 1
        })

    df = pd.concat([normal_batch(n_normal), fraud_batch(n_fraud)], ignore_index=True)

    # Heavier label noise: 12% of labels flipped -> simulates real mislabeling/ambiguity
    flip_idx = df.sample(frac=0.12, random_state=42).index
    df.loc[flip_idx, "fraud"] = 1 - df.loc[flip_idx, "fraud"]

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.insert(0, "customer_id", [f"CUST{i:06d}" for i in range(len(df))])
    return df

if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("data/processed/telecom_fraud_dataset.csv", index=False)
    print(f"Generated {len(df)} rows -> data/processed/telecom_fraud_dataset.csv")
    print(df["fraud"].value_counts())
# train_unsupervised_full.py

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

# --------------------------
# Feature Engineering
# --------------------------
def make_numeric_features(df):
    """Extract numeric features from ADFA text logs"""
    X = pd.DataFrame()
    X["length"] = df["text"].apply(lambda x: len(str(x).split()))
    X["unique_calls"] = df["text"].apply(lambda x: len(set(str(x).split())))
    # mean_call with log-transform to reduce effect of huge numbers
    X["mean_call_log"] = df["text"].apply(lambda x: np.log1p(np.mean([int(t) for t in str(x).split() if t.isdigit()])))
    return X

# --------------------------
# Feature-specific rules
# --------------------------
def feature_rules(X):
    """Optional: apply simple statistical thresholds for obvious anomalies"""
    rules = np.zeros(len(X), dtype=int)  # 0 = normal, 1 = anomaly
    # Example: anything with mean_call_log > threshold → anomaly
    threshold = np.log1p(1000)  # example: mean_call > 1000 before log
    rules[X["mean_call_log"] > threshold] = 1
    return rules

# --------------------------
# Train Unsupervised Model
# --------------------------
def train_unsupervised():
    # Load CSV
    df = pd.read_csv("adfa_parsed.csv")
    df_train = df[df["split"] == "training"]  # normal only
    print(f"Training samples: {len(df_train)}")

    # Feature extraction
    X_train = make_numeric_features(df_train)

    # --------------------------
    # Pipeline: RobustScaler + IsolationForest
    # --------------------------
    pipeline = Pipeline([
        ("scaler", RobustScaler()),
        ("iforest", IsolationForest(
            n_estimators=300,
            max_samples="auto",
            contamination=0.01,  # 1% expected anomalies
            random_state=42
        ))
    ])

    # Fit pipeline
    pipeline.fit(X_train)
    print("IsolationForest pipeline trained on normal sequences.")

    # Save pipeline
    joblib.dump(pipeline, "unsup_iforest_pipeline.pkl")
    print("Pipeline saved → unsup_iforest_pipeline.pkl")

    # Optional: save rules thresholds for dashboard
    rules = feature_rules(X_train)
    np.save("feature_rules.npy", rules)
    print("Feature-specific rules saved → feature_rules.npy")

# --------------------------
# Run training
# --------------------------
if __name__ == "__main__":
    train_unsupervised()

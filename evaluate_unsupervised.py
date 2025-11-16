# evaluate_unsupervised.py
import joblib
import pandas as pd
from feature_extraction import make_numeric_features

def evaluate():
    df = pd.read_csv("adfa_parsed.csv")
    df_test = df[df["split"].isin(["validation", "attack"])]

    X_test = make_numeric_features(df_test)
    model = joblib.load("unsup_iforest.pkl")

    preds = model.predict(X_test)   # 1 = normal , -1 = anomaly

    df_test["pred"] = (preds == -1).astype(int)

    print(df_test.groupby(["split", "label", "pred"]).size())
    df_test.to_csv("unsup_predictions.csv", index=False)
    print("Saved → unsup_predictions.csv")


if __name__ == "__main__":
    evaluate()

# feature_extraction.py
import numpy as np
import pandas as pd

def text_to_numbers(txt):
    nums = [int(t) for t in str(txt).split() if t.isdigit()]
    return nums if nums else [0]

def make_numeric_features(df):
    df = df.copy()
    
    df["seq"] = df["text"].apply(text_to_numbers)
    df["mean_call"] = df["seq"].apply(np.mean)
    df["std_call"]  = df["seq"].apply(np.std)
    df["len_call"]  = df["seq"].apply(len)

    return df[["mean_call", "std_call", "len_call"]]


if __name__ == "__main__":
    df = pd.read_csv("adfa_parsed.csv")
    features = make_numeric_features(df)
    print(features.head())

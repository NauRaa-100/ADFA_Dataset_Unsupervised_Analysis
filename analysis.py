# =============================
# analysis.py
# =============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------- Settings ----------
plt.style.use("ggplot")
Path("plots").mkdir(exist_ok=True)

# ---------- Load Data ----------
df = pd.read_csv("adfa_parsed.csv")

print("\n=== Basic Info ===")
print(df.head())
print(df.info())

# ---------- Feature Engineering ----------
df["char_len"] = df["text"].str.len()
df["num_lines"] = df["text"].str.count("\n")
df["num_spaces"] = df["text"].str.count(" ")
df["num_special"] = df["text"].str.replace("[A-Za-z0-9 ]", "", regex=True).str.len()

# Save new data if needed
df.to_csv(r"E:\Rev-DataScience\AI-ML\semi-structured\adfa_features.csv", index=False)

# ---------- VISUALIZATION ----------

# 1) Histogram of char length
plt.figure(figsize=(10,5))
sns.histplot(df["char_len"], bins=50)
plt.title("Character Length Distribution")
plt.xlabel("Char Length")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("char_length_dist.png")
plt.close()

# 2) Compare char_len per split
plt.figure(figsize=(10,5))
sns.boxplot(data=df, x="label", y="char_len")
plt.title("Character Length by Class")
plt.xticks([0,1], ["Normal", "Attack"])
plt.tight_layout()
plt.savefig("char_len_by_class.png")
plt.close()

# 3) Heatmap
plt.figure(figsize=(8,5))
sns.heatmap(df[["char_len","num_lines","num_spaces","num_special"]].corr(), annot=True)
plt.title("Feature Correlation")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()



print("Saved plots in /plots/")

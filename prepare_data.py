# prepare_data.py
import os
from pathlib import Path
import pandas as pd

def prepare_adfa(base_dir):
    base_dir = Path(base_dir)
    records = []
    splits = ["Training_Data_Master", "Validation_Data_Master", "Attack_Data_Master"]

    for split in splits:
        split_dir = base_dir / split
        if not split_dir.exists():
            print("Missing folder:", split_dir)
            continue
        
        if "attack" in split.lower():
            lbl = 1
            split_name = "attack"
        else:
            lbl = 0
            split_name = split.lower().replace("_data_master", "")

        for root, _, files in os.walk(split_dir):
            for f in files:
                fp = Path(root) / f
                text = fp.read_text(errors="ignore").strip()
                records.append({
                    "split": split_name,
                    "file": f,
                    "text": text,
                    "label": lbl
                })

    df = pd.DataFrame(records)
    df.to_csv("adfa_parsed.csv", index=False)
    print("Saved → adfa_parsed.csv:", df.shape)
    return df


if __name__ == "__main__":
    prepare_adfa(r"E:\Rev-DataScience\AI-ML\semi-structured\ADFA-LD")

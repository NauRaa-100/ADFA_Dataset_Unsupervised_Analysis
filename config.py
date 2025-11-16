# config.py

from pathlib import Path

BASE_DIR = Path.cwd()  
DATA_DIR = BASE_DIR / "ADFA-LD"           
CSV_OUT = BASE_DIR / "adfa_parsed.csv"
MODEL_DIR = BASE_DIR / "model_artifacts"
MODEL_DIR.mkdir(exist_ok=True, parents=True)

# splits expected inside DATA_DIR
SPLITS = {
    "train": "Training_Data_Master",
    "val": "Validation_Data_Master",
    "attack": "Attack_Data_Master"
}

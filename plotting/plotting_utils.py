
import json
from pathlib import Path
import pandas as pd

# Path to datasets directory
DATASET_DIR = Path(__file__).parent.parent / "bias_steering" / "data" / "datasets"


def load_dataframe_from_json(filepath):
    """Load a JSON file and convert to pandas DataFrame."""
    data = json.load(open(filepath, "r"))
    return pd.DataFrame.from_records(data)


def load_target_words(target_concept="vision"):
    """Load target words for a given concept (gender, race, vision)."""
    return json.load(open(DATASET_DIR / "target_words.json", "r"))[target_concept]

import json
import logging
from pathlib import Path
from typing import Dict
import pandas as pd
from .template import Template
from ..config import DataConfig

DATASET_DIR = Path(__file__).resolve().parent / "datasets"


def load_dataframe_from_json(filepath):
    data = json.load(open(filepath, "r"))
    return pd.DataFrame.from_records(data)


def load_target_words(target_concept="vision"):
    return json.load(open(DATASET_DIR / "target_words.json", "r"))[target_concept]


def load_vision_dataset(split: str, include_neutral=True, sample_size=None):
    data = pd.read_csv(DATASET_DIR / f"splits/vision_{split}.csv")

    if not include_neutral and "is_neutral" in data.columns:
        data = data[~data.is_neutral]

    if sample_size is not None:
        data = data.sample(n=min(sample_size, len(data)))
    
    instructions = [line.strip() for line in open(DATASET_DIR / f"instructions/vision_{split}.txt", "r").readlines()]
    instruction_set = Template(instructions)

    instructions = [instruction_set.get_template() for _ in range(len(data))]
    prompts, output_prefixes = [], []

    for inst, text in zip(instructions, data["text"]):
        inst, output_prefix = inst.split(" | ")
        prompts.append(f'{inst}\n{text}')
        output_prefixes.append(output_prefix)
    
    data["prompt"] = prompts
    data["output_prefix"] = output_prefixes

    return data

# Default location: <project_root>/data/handcrafted_eval.json
# (two parents up from this file: datasets/ → data/ → bias_steering/ → project root)
_DEFAULT_HANDCRAFTED_FILE = Path(__file__).resolve().parents[2] / "data" / "handcrafted_eval.json"


def load_handcrafted_eval(filepath: str = None) -> pd.DataFrame:
    """
    Load the hand-crafted evaluation set and add prompt / output_prefix columns
    using the same val instruction templates as the benchmark.

    The JSON file must be a list of objects with at minimum a "text" field.
    Standard fields: text, vision_label, _id, is_neutral.
    Extra fields (e.g. scene_type, _note) are preserved but ignored by the pipeline.
    """
    path = Path(filepath) if filepath else _DEFAULT_HANDCRAFTED_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Handcrafted eval file not found: {path}\n"
            "Pass --handcrafted_eval_file to specify the path."
        )

    data = pd.DataFrame.from_records(json.load(open(path, "r")))

    instructions = [
        line.strip()
        for line in open(DATASET_DIR / "instructions/vision_val.txt", "r").readlines()
        if line.strip()
    ]
    instruction_set = Template(instructions)
    sampled_instructions = [instruction_set.get_template() for _ in range(len(data))]

    prompts, output_prefixes = [], []
    for inst, text in zip(sampled_instructions, data["text"]):
        inst_text, output_prefix = inst.split(" | ")
        prompts.append(f"{inst_text}\n{text}")
        output_prefixes.append(output_prefix)

    data["prompt"] = prompts
    data["output_prefix"] = output_prefixes
    return data


def load_datasplits(cfg: DataConfig, save_dir: Path, use_cache: bool = False) -> Dict[str, pd.DataFrame]:
    if cfg.target_concept != "vision":
        raise ValueError(f"Only target_concept='vision' is supported in this repo, got: {cfg.target_concept}")

    datasets = {}        
    for split in ["train", "val"]:
        if use_cache and Path(save_dir / f"{split}.json").exists():
            logging.info(f"Loading cached data from {save_dir}/{split}.json")
            datasets[split] = load_dataframe_from_json(save_dir / f"{split}.json")
        else:
            if split == "val":
                sample_size = cfg.n_val
            else:
                sample_size = None

            datasets[split] = load_vision_dataset(split, include_neutral=True, sample_size=sample_size)
    
    return datasets

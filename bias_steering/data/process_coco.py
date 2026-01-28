import json
import re
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd

COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

DATA_DIR = Path(__file__).resolve().parent / "datasets"
COCO_DIR = DATA_DIR / "raw" / "coco"
SPLITS_DIR = DATA_DIR / "splits"
TARGET_WORDS_PATH = DATA_DIR / "target_words.json"


def _ensure_coco_captions_train_json() -> Path:
    #Download and extract COCO captions annotations (train/val 2017) if needed.
    COCO_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = COCO_DIR / "annotations_trainval2017.zip"
    captions_train_path = COCO_DIR / "annotations" / "captions_train2017.json"

    if captions_train_path.exists():
        return captions_train_path

    if not zip_path.exists():
        urllib.request.urlretrieve(COCO_ANNOTATIONS_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(COCO_DIR)

    return captions_train_path


def _load_vision_terms() -> Tuple[list[str], list[str]]:
    #Load spatial + descriptive term lists from `target_words.json`.
    with open(TARGET_WORDS_PATH, "r") as f:
        vision_words = json.load(f)["vision"]
    return list(vision_words["spatial"]), list(vision_words["descriptive"])


def _count_term_hits(text: str, term: str) -> int:
    #Count how often a term appears in `text`.
    t = text.lower()
    if " " in term:
        return t.count(term)
    return len(re.findall(r"\b" + re.escape(term) + r"\b", t))


def _score_caption(text: str, spatial_terms: Iterable[str], descriptive_terms: Iterable[str]) -> int:
    #Score = (spatial terms) - (descriptive terms).
    spatial = sum(_count_term_hits(text, term) for term in spatial_terms)
    descriptive = sum(_count_term_hits(text, term) for term in descriptive_terms)
    return spatial - descriptive


def process_coco_captions(
    *,
    n_train_per_class: int = 1500,
    n_val_per_class: int = 750,
    strong_score_threshold: int = 2,
    seed: int = 42,
    preview_top_k: int = 0,
    save_ranked_path: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Build `vision_train.csv` / `vision_val.csv` from COCO captions.
    #
    # Labels come from a simple term-count score:
    #   score = (#spatial terms) - (#descriptive terms)
    #
    # We keep only "strong" examples (cleanly one or the other):
    #   spatial: score >= strong_score_threshold
    #   descriptive: score <= -strong_score_threshold
    captions_path = _ensure_coco_captions_train_json()
    spatial_terms, descriptive_terms = _load_vision_terms()

    with open(captions_path, "r") as f:
        coco = json.load(f)

    annotations = coco["annotations"]

    rows = []
    for ann in annotations:
        text = ann["caption"]
        score = _score_caption(text, spatial_terms, descriptive_terms)
        label = "spatial" if score > 0 else ("descriptive" if score < 0 else "neutral")
        rows.append({"text": text, "score": score, "vision_label": label})

    df = pd.DataFrame(rows)
    df = df[df["vision_label"] != "neutral"].copy()

    spatial_df = df[(df["vision_label"] == "spatial") & (df["score"] >= strong_score_threshold)].copy()
    descriptive_df = df[(df["vision_label"] == "descriptive") & (df["score"] <= -strong_score_threshold)].copy()
    # Strong-bias pool sizes (kept as comment to keep script output quiet):
    #   spatial = len(spatial_df)
    #   descriptive = len(descriptive_df)

    spatial_df = spatial_df.sort_values("score", ascending=False)
    descriptive_df = descriptive_df.sort_values("score", ascending=True)

    # Optional: preview ranking / save scored captions to inspect "scoring in action".
    if preview_top_k > 0 or save_ranked_path:
        ranked = pd.concat(
            [
                spatial_df.assign(_rank_group="spatial").sort_values("score", ascending=False),
                descriptive_df.assign(_rank_group="descriptive").sort_values("score", ascending=True),
            ],
            ignore_index=True,
        )

        if save_ranked_path:
            Path(save_ranked_path).parent.mkdir(parents=True, exist_ok=True)
            ranked.to_csv(save_ranked_path, index=False)

        if preview_top_k > 0:
            top_spatial = spatial_df.head(preview_top_k)[["score", "text"]]
            top_desc = descriptive_df.head(preview_top_k)[["score", "text"]]
            print("\nTop spatial examples (highest scores):")
            print(top_spatial.to_string(index=False))
            print("\nTop descriptive examples (lowest scores):")
            print(top_desc.to_string(index=False))

    n_train_spatial = min(n_train_per_class, len(spatial_df))
    n_train_desc = min(n_train_per_class, len(descriptive_df))
    n_val_spatial = min(n_val_per_class, max(0, len(spatial_df) - n_train_spatial))
    n_val_desc = min(n_val_per_class, max(0, len(descriptive_df) - n_train_desc))

    train_df = pd.concat(
        [spatial_df.head(n_train_spatial), descriptive_df.head(n_train_desc)],
        ignore_index=True,
    ).sample(frac=1, random_state=seed).reset_index(drop=True)

    val_df = pd.concat(
        [
            spatial_df.iloc[n_train_spatial : n_train_spatial + n_val_spatial],
            descriptive_df.iloc[n_train_desc : n_train_desc + n_val_desc],
        ],
        ignore_index=True,
    ).sample(frac=1, random_state=seed).reset_index(drop=True)

    train_out = pd.DataFrame(
        {
            "text": train_df["text"],
            "vision_label": train_df["vision_label"],
            "_id": range(1, len(train_df) + 1),
            "is_neutral": False,
        }
    )
    val_out = pd.DataFrame(
        {
            "text": val_df["text"],
            "vision_label": val_df["vision_label"],
            "_id": range(1, len(val_df) + 1),
            "is_neutral": False,
        }
    )

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train_path = SPLITS_DIR / "vision_train.csv"
    val_path = SPLITS_DIR / "vision_val.csv"

    if train_path.exists():
        pd.read_csv(train_path).to_csv(SPLITS_DIR / "vision_train_backup.csv", index=False)
    if val_path.exists():
        pd.read_csv(val_path).to_csv(SPLITS_DIR / "vision_val_backup.csv", index=False)

    train_out.to_csv(train_path, index=False)
    val_out.to_csv(val_path, index=False)

    return train_out, val_out


if __name__ == "__main__":
    process_coco_captions(n_train_per_class=1500, n_val_per_class=750, strong_score_threshold=2)

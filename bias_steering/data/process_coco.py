"""
Process COCO captions dataset to create vision dataset.
Downloads COCO captions, scores them using spatial/descriptive terms,
and creates balanced train/val splits.
"""
import json
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import re
from collections import Counter

# COCO dataset URLs
COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
DATA_DIR = Path(__file__).resolve().parent / "datasets"
RAW_DIR = DATA_DIR / "raw"
COCO_DIR = RAW_DIR / "coco"
SPLITS_DIR = DATA_DIR / "splits"

def download_coco_annotations():
    """Download COCO annotations if not already present."""
    COCO_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = COCO_DIR / "annotations_trainval2017.zip"
    annotations_path = COCO_DIR / "annotations" / "captions_train2017.json"
    
    if annotations_path.exists():
        print(f"COCO annotations already exist at {annotations_path}")
        return annotations_path
    
    if not zip_path.exists():
        print(f"Downloading COCO annotations from {COCO_ANNOTATIONS_URL}...")
        print("This may take a few minutes...")
        urllib.request.urlretrieve(COCO_ANNOTATIONS_URL, zip_path)
        print("Download complete!")
    
    print("Extracting annotations...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(COCO_DIR)
    
    return annotations_path

def load_target_words():
    """Load spatial and descriptive target words."""
    target_words_path = DATA_DIR / "target_words.json"
    with open(target_words_path, 'r') as f:
        target_words = json.load(f)["vision"]
    return target_words["spatial"], target_words["descriptive"]

def score_caption(caption: str, spatial_terms: List[str], descriptive_terms: List[str]) -> Tuple[int, int, float]:
    """
    Score a caption based on spatial and descriptive terms.
    Returns: (spatial_count, descriptive_count, score)
    Score = spatial_count - descriptive_count (positive = more spatial)
    """
    caption_lower = caption.lower()
    
    # Count spatial terms (handle multi-word terms)
    spatial_count = 0
    for term in spatial_terms:
        # Use word boundaries for single words, direct match for phrases
        if ' ' in term:
            spatial_count += caption_lower.count(term)
        else:
            # Word boundary matching for single words
            pattern = r'\b' + re.escape(term) + r'\b'
            spatial_count += len(re.findall(pattern, caption_lower))
    
    # Count descriptive terms
    descriptive_count = 0
    for term in descriptive_terms:
        if ' ' in term:
            descriptive_count += caption_lower.count(term)
        else:
            pattern = r'\b' + re.escape(term) + r'\b'
            descriptive_count += len(re.findall(pattern, caption_lower))
    
    score = spatial_count - descriptive_count
    return spatial_count, descriptive_count, score

def process_coco_captions(n_train: int = 1500, n_val: int = 750):
    """
    Process COCO captions and create balanced vision dataset.
    
    Args:
        n_train: Number of training examples per class (spatial/descriptive)
        n_val: Number of validation examples per class
    """
    print("Loading COCO annotations...")
    annotations_path = download_coco_annotations()
    
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)
    
    # Extract all captions
    captions = []
    for ann in coco_data['annotations']:
        captions.append({
            'id': ann['id'],
            'image_id': ann['image_id'],
            'caption': ann['caption']
        })
    
    print(f"Loaded {len(captions)} captions from COCO")
    
    # Load target words
    spatial_terms, descriptive_terms = load_target_words()
    print(f"Using {len(spatial_terms)} spatial terms and {len(descriptive_terms)} descriptive terms")
    
    # Score all captions
    print("Scoring captions...")
    scored_captions = []
    for cap in captions:
        spatial_count, desc_count, score = score_caption(
            cap['caption'], spatial_terms, descriptive_terms
        )
        scored_captions.append({
            'text': cap['caption'],
            'spatial_count': spatial_count,
            'descriptive_count': desc_count,
            'score': score,
            'image_id': cap['image_id'],
            '_id': cap['id']
        })
    
    # Create DataFrame
    df = pd.DataFrame(scored_captions)
    
    # Classify based on score
    # Positive score = more spatial, Negative score = more descriptive
    df['vision_label'] = df['score'].apply(
        lambda x: 'spatial' if x > 0 else ('descriptive' if x < 0 else 'neutral')
    )
    
    # Filter out neutral examples (score == 0)
    df_labeled = df[df['vision_label'] != 'neutral'].copy()
    
    # Filter for STRONG bias: require score >= 2 for spatial, score <= -2 for descriptive
    # This ensures examples are clearly one or the other, not both
    min_spatial_score = 2  # At least 2 more spatial terms than descriptive
    max_descriptive_score = -2  # At least 2 more descriptive terms than spatial
    
    spatial_df = df_labeled[df_labeled['vision_label'] == 'spatial'].copy()
    spatial_df = spatial_df[spatial_df['score'] >= min_spatial_score].copy()
    
    descriptive_df = df_labeled[df_labeled['vision_label'] == 'descriptive'].copy()
    descriptive_df = descriptive_df[descriptive_df['score'] <= max_descriptive_score].copy()
    
    print(f"\nAfter filtering for strong bias (score >= {min_spatial_score} for spatial, <= {max_descriptive_score} for descriptive):")
    print(f"  Spatial examples: {len(spatial_df)}")
    print(f"  Descriptive examples: {len(descriptive_df)}")
    print(f"\nScore statistics (filtered):")
    if len(spatial_df) > 0:
        print(f"  Spatial - min: {spatial_df['score'].min()}, max: {spatial_df['score'].max()}, mean: {spatial_df['score'].mean():.2f}")
    if len(descriptive_df) > 0:
        print(f"  Descriptive - min: {descriptive_df['score'].min()}, max: {descriptive_df['score'].max()}, mean: {descriptive_df['score'].mean():.2f}")
    
    # Sort by absolute score (strongest examples first)
    spatial_df = spatial_df.sort_values('score', ascending=False)
    descriptive_df = descriptive_df.sort_values('score', ascending=True)  # More negative = more descriptive
    
    # Select top examples
    n_train_spatial = min(n_train, len(spatial_df))
    n_train_desc = min(n_train, len(descriptive_df))
    n_val_spatial = min(n_val, len(spatial_df) - n_train_spatial)
    n_val_desc = min(n_val, len(descriptive_df) - n_train_desc)
    
    print(f"\nSelecting examples:")
    print(f"  Train: {n_train_spatial} spatial, {n_train_desc} descriptive")
    print(f"  Val: {n_val_spatial} spatial, {n_val_desc} descriptive")
    
    # Split into train/val
    train_spatial = spatial_df.head(n_train_spatial)
    val_spatial = spatial_df.iloc[n_train_spatial:n_train_spatial + n_val_spatial]
    
    train_desc = descriptive_df.head(n_train_desc)
    val_desc = descriptive_df.iloc[n_train_desc:n_train_desc + n_val_desc]
    
    # Combine and format
    train_df = pd.concat([train_spatial, train_desc], ignore_index=True)
    val_df = pd.concat([val_spatial, val_desc], ignore_index=True)
    
    # Shuffle
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Reset IDs
    train_df['_id'] = range(1, len(train_df) + 1)
    val_df['_id'] = range(1, len(val_df) + 1)
    
    # Mark as non-neutral
    train_df['is_neutral'] = False
    val_df['is_neutral'] = False
    
    # Select only required columns
    output_cols = ['text', 'vision_label', '_id', 'is_neutral']
    train_df = train_df[output_cols]
    val_df = val_df[output_cols]
    
    # Save
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train_path = SPLITS_DIR / "vision_train.csv"
    val_path = SPLITS_DIR / "vision_val.csv"
    
    # Backup existing files
    if train_path.exists():
        backup_path = SPLITS_DIR / "vision_train_backup.csv"
        print(f"\nBacking up existing {train_path} to {backup_path}")
        train_df_old = pd.read_csv(train_path)
        train_df_old.to_csv(backup_path, index=False)
    
    if val_path.exists():
        backup_path = SPLITS_DIR / "vision_val_backup.csv"
        print(f"Backing up existing {val_path} to {backup_path}")
        val_df_old = pd.read_csv(val_path)
        val_df_old.to_csv(backup_path, index=False)
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    print(f"\n✅ Saved {len(train_df)} training examples to {train_path}")
    print(f"✅ Saved {len(val_df)} validation examples to {val_path}")
    print(f"\nTrain distribution: {train_df['vision_label'].value_counts().to_dict()}")
    print(f"Val distribution: {val_df['vision_label'].value_counts().to_dict()}")
    
    return train_df, val_df

if __name__ == "__main__":
    # Use recommended amounts: 1500 per class for train, 750 per class for val
    # Total: 3000 train, 1500 val
    train_df, val_df = process_coco_captions(n_train=1500, n_val=750)

# vision-bias-steering

Code implementation for detecting and steering spatial vs descriptive language bias in LLMs.

This repository is a fork/adaptation of [gender-bias-steering](https://github.com/YOUR_ORIGINAL_REPO/gender-bias-steering), which implements the paper: [Sensing and Steering Stereotypes: Extracting and Applying Gender Representation Vectors in LLMs](https://arxiv.org/abs/XXXX.XXXXX) by Hannah Cherevatsky (and collaborators).

## Overview

This project extracts bias vectors that represent the difference between spatial language (e.g., "left", "above", "beside") and descriptive language (e.g., "red", "large", "round") in transformer models. These vectors can then be used to:
- **Detect** implicit bias in model outputs
- **Steer** model behavior to reduce or balance bias
- **Analyze** how different layers encode spatial vs descriptive representations

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run bias detection and steering
python -m bias_steering.run \
  --model_name "gpt2" \
  --method WMD \
  --target_concept "vision" \
  --pos_label "spatial" \
  --neg_label "descriptive" \
  --n_train_per_label 800 \
  --n_val 1000 \
  --batch_size 32
```

## Dataset

The dataset consists of text examples labeled as either:
- **Spatial**: Language describing spatial relationships (e.g., "The ball is under the table")
- **Descriptive**: Language describing visual properties (e.g., "A bright red flower")

See `bias_steering/data/datasets/splits/` for the dataset files.

## Results

The system extracts bias vectors at each transformer layer and validates their effectiveness in reducing bias. Results are saved in `runs_vision/[model_name]/`.

## Credits

Forked from [gender-bias-steering](https://github.com/YOUR_ORIGINAL_REPO/gender-bias-steering), which implements:
- **Paper**: [Sensing and Steering Stereotypes: Extracting and Applying Gender Representation Vectors in LLMs](https://arxiv.org/abs/XXXX.XXXXX)
- **Authors**: Hannah Cherevatsky (and collaborators)

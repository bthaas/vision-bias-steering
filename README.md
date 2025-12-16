# vision-bias-steering

Code implementation for detecting and steering spatial vs descriptive language bias in LLMs.

This repository is a fork/adaptation of [gender-bias-steering](https://github.com/hannahxchen/gender-bias-steering), which implements the paper: [Sensing and Steering Stereotypes: Extracting and Applying Gender Representation Vectors in LLMs](https://arxiv.org/abs/XXXX.XXXXX) by Hannah Cherevatsky (and collaborators).

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

See `bias_steering/data/datasets/splits/` for the dataset files.

## Results

The system extracts bias vectors at each transformer layer and validates their effectiveness in reducing bias. Results are saved in `runs_vision/[model_name]/`.

## Credits

Forked from [gender-bias-steering](https://github.com/hannahxchen/gender-bias-steering), which implements:
- **Paper**: [Sensing and Steering Stereotypes: Extracting and Applying Gender Representation Vectors in LLMs](https://arxiv.org/abs/XXXX.XXXXX)
- **Authors**: Hannah Cyberey, Yangfeng Ji, David Evans


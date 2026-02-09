# vision-bias-steering

Code implementation for detecting and steering spatial vs descriptive language bias in LLMs.

This repository is a fork/adaptation of [gender-bias-steering](https://github.com/hannahxchen/gender-bias-steering), which implements the paper: [Sensing and Steering Stereotypes: Extracting and Applying Gender Representation Vectors in LLMs](https://arxiv.org/abs/XXXX.XXXXX) by Hannah Cherevatsky (and collaborators).

## Overview

This system extracts "steering vectors" from language models that represent the difference between **spatial language** (positions, locations like "left", "behind", "near") and **descriptive language** (colors, sizes, shapes like "red", "large", "round"). These vectors can then be used to:

- **Measure bias**: Detect if a model favors spatial or descriptive language
- **Steer generation**: Push model outputs toward more spatial or more descriptive language
- **Remove bias**: Project out the bias component for neutral outputs

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd vision-bias-steering

# Install dependencies
pip install -r requirements.txt
```

**Requirements**: Python 3.10+, PyTorch 2.0+, ~4GB disk space for model weights

## Quick Start

### 1. Train the Steering Vector

```bash
python -m bias_steering.run \
  --model_name "gpt2" \
  --method WMD \
  --target_concept "vision" \
  --pos_label "spatial" \
  --neg_label "descriptive" \
  --n_train_per_label 800 \
  --n_val 1000 \
  --batch_size 32 \
  --constrained_softmax
```

**Key arguments:**
- `--model_name`: HuggingFace model name (e.g., "gpt2", "Qwen/Qwen-1_8B-Chat")
- `--method`: Vector extraction method - "WMD" (weighted mean difference) or "MD" (mean difference)
- `--constrained_softmax`: Use constrained scoring (recommended - gives cleaner bias signal)
- `--n_train_per_label`: Number of training examples per class
- `--n_val`: Number of validation examples

### 2. Evaluate on Downstream Tasks

```bash
python -m bias_steering.run \
  --config_file runs_vision/gpt2/config.yaml \
  --run_eval \
  --coeff 0 \
  --batch_size 32
```

### 3. Generate Plots

```bash
cd plotting/scripts
python run_all_plots.py
```

Plots are saved to `plots/` directory as interactive HTML files.

## Results

### GPT-2 Vision Bias Steering

Using **constrained softmax** and optimized coefficients (constant intervention):

- **Baseline RMS bias:** 0.3969
- **Best RMS bias:** 0.1941 (layer 5, coeff -80)
- **RMS reduction:** 51.1%

### Qwen-1.8B-Chat Vision Bias Steering

Using **constrained softmax** and optimized coefficients (constant intervention):

- **Baseline RMS bias:** 0.7073
- **Best RMS bias:** 0.0494 (layer 11, coeff -200)
- **RMS reduction:** 93.0%


## Project Structure

```
vision-bias-steering/
├── bias_steering/
│   ├── run.py              # Main entry point
│   ├── config.py           # Configuration dataclasses
│   ├── data/
│   │   ├── load_dataset.py # Dataset loading
│   │   ├── datasets/       # Raw data and splits
│   │   └── ...
│   ├── steering/
│   │   ├── model.py        # Model wrapper
│   │   ├── extract.py      # Vector extraction
│   │   ├── validate.py     # Validation
│   │   └── intervention.py # Steering intervention
│   └── eval/
│       ├── task.py         # Base evaluation task
│       └── winogenerated.py # Winogenerated benchmark
├── plotting/
│   ├── *.ipynb             # Jupyter notebooks for plots
│   └── scripts/            # Python scripts to run notebooks
├── runs_vision/            # Output directory for results
│   └── gpt2/
│       ├── config.yaml
│       ├── activations/    # Extracted steering vectors
│       ├── datasplits/     # Train/val data with bias scores
│       └── validation/     # Validation results
└── plots/                  # Generated HTML plots
```

## Dataset

The vision dataset is derived from COCO captions, labeled as:
- **Spatial**: Captions emphasizing position/location (e.g., "A cat sitting next to a dog behind the house")
- **Descriptive**: Captions emphasizing appearance (e.g., "A bright red car with large wheels")

See `bias_steering/data/datasets/splits/` for the processed dataset files.

Target words are defined in `bias_steering/data/datasets/target_words.json`.

## Using the Steering Vector

```python
import torch
from bias_steering.steering import load_model
from bias_steering.steering.intervention import get_intervention_func

# Load model and steering vector
model = load_model('gpt2')
vectors = torch.load('runs_vision/gpt2/activations/candidate_vectors.pt')
steering_vec = model.set_dtype(vectors[5])  # Layer 5 is best

# Create intervention function
# coeff > 0: more spatial, coeff < 0: more descriptive
intervene_func = get_intervention_func(steering_vec, method='constant', coeff=50)

# Generate with steering
prompt = "Describe this scene: A cat on a mat."
output = model.generate([prompt], layer=5, intervene_func=intervene_func, max_new_tokens=30)
print(output[0])
```

## Credits

Forked from [gender-bias-steering](https://github.com/hannahxchen/gender-bias-steering), which implements:
- **Paper**: [Sensing and Steering Stereotypes: Extracting and Applying Gender Representation Vectors in LLMs](https://arxiv.org/abs/XXXX.XXXXX)
- **Authors**: Hannah Cyberey, Yangfeng Ji, David Evans

## License

See LICENSE file.

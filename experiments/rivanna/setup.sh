#!/bin/bash
# Setup script for Rivanna HPC (UVA).
#
# Creates a Python venv with all required packages and clones
# the repository to scratch space.
#
# Usage (run interactively on a Rivanna login node):
#   bash experiments/rivanna/setup.sh
#
# After setup:
#   source /scratch/jea7vy/vision-bias-steering/venv/bin/activate
#   cd /scratch/jea7vy/vision-bias-steering/repo
#   bash experiments/rivanna/slurm/submit_all.sh

set -euo pipefail

SCRATCH=/scratch/jea7vy/vision-bias-steering
REPO=$SCRATCH/repo
VENV=$SCRATCH/venv
GITHUB_REPO=https://github.com/YOUR_USERNAME/vision-bias-steering.git  # update this

# ── 1. Load modules ────────────────────────────────────────────────────────────
echo "Loading modules..."
module load gcc/11.4.0
module load openmpi/4.1.4
module load python/3.11.4
module load cuda/12.4.1

# ── 2. Clone repo ──────────────────────────────────────────────────────────────
mkdir -p "$SCRATCH"
if [ ! -d "$REPO" ]; then
    echo "Cloning repo to $REPO..."
    git clone "$GITHUB_REPO" "$REPO"
else
    echo "Repo already at $REPO — pulling latest..."
    cd "$REPO" && git pull
fi

cd "$REPO"

# ── 3. Create Python venv ─────────────────────────────────────────────────────
echo "Creating venv at $VENV..."
python -m venv "$VENV"
source "$VENV/bin/activate"

# ── 4. Install PyTorch (CUDA 12.4) ────────────────────────────────────────────
echo "Installing PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# ── 5. Install project dependencies ──────────────────────────────────────────
echo "Installing project dependencies..."
pip install \
    transformers>=4.40.0 \
    accelerate>=0.27.0 \
    nnsight>=0.3.0 \
    bitsandbytes>=0.43.0 \
    torchtyping \
    dataclass-wizard \
    pandas \
    numpy \
    scipy \
    tqdm \
    sentencepiece \
    protobuf

# ── 6. Install the package in editable mode ───────────────────────────────────
if [ -f "$REPO/setup.py" ] || [ -f "$REPO/pyproject.toml" ]; then
    echo "Installing package in editable mode..."
    pip install -e "$REPO"
fi

# ── 7. Verify data files ──────────────────────────────────────────────────────
echo "Verifying data files..."
DATA_DIR="$REPO/experiments/rivanna/data"
REQUIRED_FILES=(
    "target_words.json"
    "handcrafted_eval.json"
    "train.json"
    "val.json"
    "instructions/vision_train.txt"
    "instructions/vision_val.txt"
)
ALL_OK=true
for f in "${REQUIRED_FILES[@]}"; do
    if [ -f "$DATA_DIR/$f" ]; then
        echo "  ✓ $f"
    else
        echo "  ✗ MISSING: $DATA_DIR/$f"
        ALL_OK=false
    fi
done

if [ "$ALL_OK" = false ]; then
    echo ""
    echo "WARNING: Some data files are missing."
    echo "Copy them manually or verify the repo was fully pushed."
fi

# ── 8. Create log directory ───────────────────────────────────────────────────
mkdir -p "$REPO/experiments/rivanna/logs"

# ── 9. Hugging Face token for gated models ────────────────────────────────────
echo ""
echo "If running Llama-3.1-8B-Instruct, set your HuggingFace token:"
echo "  export HF_TOKEN=hf_your_token_here"
echo "  huggingface-cli login --token \$HF_TOKEN"
echo ""
echo "Setup complete. Activate with: source $VENV/bin/activate"
echo "Repo: $REPO"

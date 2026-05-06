#!/bin/bash
#SBATCH --job-name=qwen_imgshows_verify
#SBATCH --account=uvasrg_paid
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=experiments/rivanna/logs/qwen_base_image_shows_verify_%j.out
#SBATCH --error=experiments/rivanna/logs/qwen_base_image_shows_verify_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/jea7vy/vision-bias-steering}"
VENV="${VENV:-$PROJECT_ROOT/venv}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_ROOT/results/qwen_base_image_shows}"

module load gcc/11.4.0
module load openmpi/4.1.4
module load python/3.11.4
module load cuda/12.4.1

source "$VENV/bin/activate"

cd "$PROJECT_ROOT"
mkdir -p experiments/rivanna/logs "$RESULTS_ROOT"

echo "Starting Qwen-1.8B-chat Image Shows verification at $(date)"
echo "Node: $SLURMD_NODENAME  GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python experiments/rivanna/run_qwen_base_image_shows.py verify-local \
    --results-root "$RESULTS_ROOT" \
    --data-dir "$PROJECT_ROOT/experiments/rivanna/data" \
    --local-artifact-dir "$PROJECT_ROOT/runs_vision/Qwen-1_8B-chat" \
    --n-val 1000 \
    --batch-size 16 \
    --max-new-tokens 20 \
    --lambdas=-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60 \
    --verify-lambdas=0,20,30 \
    --seed 42

echo "Verification done at $(date)"

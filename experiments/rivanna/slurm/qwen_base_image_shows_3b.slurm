#!/bin/bash
#SBATCH --job-name=qwen_imgshows_3b
#SBATCH --account=uvasrg_paid
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=experiments/rivanna/logs/qwen_base_image_shows_3b_%j.out
#SBATCH --error=experiments/rivanna/logs/qwen_base_image_shows_3b_%j.err

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

echo "Starting Qwen2.5-3B base Image Shows sweep at $(date)"
echo "Node: $SLURMD_NODENAME  GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python experiments/rivanna/run_qwen_base_image_shows.py run-model \
    --model-key qwen25_3b_base \
    --results-root "$RESULTS_ROOT" \
    --data-dir "$PROJECT_ROOT/experiments/rivanna/data" \
    --n-train 800 \
    --n-val 1000 \
    --validation-selection first \
    --batch-size 16 \
    --max-new-tokens 20 \
    --lambdas=-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60 \
    --seed 42 \
    --require-verification

echo "Qwen2.5-3B base sweep done at $(date)"

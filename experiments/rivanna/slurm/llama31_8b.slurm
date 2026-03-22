#!/bin/bash
#SBATCH --job-name=bias_llama31_8b
#SBATCH --account=uvasrg_paid
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/llama31_8b_%j.out
#SBATCH --error=logs/llama31_8b_%j.err

set -euo pipefail

SCRATCH=/scratch/jea7vy/vision-bias-steering
REPO=$SCRATCH/repo
CONDA_ENV=bias-steering

module load anaconda
module load cuda

source activate $CONDA_ENV

cd $REPO

mkdir -p $SCRATCH/results/llama31_8b
mkdir -p experiments/rivanna/logs

echo "Starting Llama-3.1-8B experiment at $(date)"
echo "Node: $SLURMD_NODENAME  GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# NOTE: Llama-3.1-8B-Instruct requires HuggingFace access.
# Ensure HF_TOKEN is set: export HF_TOKEN=hf_...
# Or log in first: huggingface-cli login

python experiments/rivanna/run_experiment.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --template B_positioned \
    --output-dir $SCRATCH/results/llama31_8b \
    --data-dir $REPO/experiments/rivanna/data \
    --n-train 800 \
    --n-val 200 \
    --n-qual 5 \
    --batch-size 8 \
    --max-new-tokens 20 \
    --seed 42

echo "Done at $(date)"

#!/bin/bash
#SBATCH --job-name=bias_qwen25_7b
#SBATCH --account=uvasrg_paid
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/qwen25_7b_%j.out
#SBATCH --error=logs/qwen25_7b_%j.err

set -euo pipefail

SCRATCH=/scratch/jea7vy/vision-bias-steering
REPO=$SCRATCH/repo
CONDA_ENV=bias-steering

module load anaconda
module load cuda

source activate $CONDA_ENV

cd $REPO

mkdir -p $SCRATCH/results/qwen25_7b
mkdir -p experiments/rivanna/logs

echo "Starting Qwen2.5-7B experiment at $(date)"
echo "Node: $SLURMD_NODENAME  GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python experiments/rivanna/run_experiment.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --template B_positioned \
    --output-dir $SCRATCH/results/qwen25_7b \
    --data-dir $REPO/experiments/rivanna/data \
    --n-train 800 \
    --n-val 200 \
    --n-qual 5 \
    --batch-size 8 \
    --max-new-tokens 20 \
    --seed 42

echo "Done at $(date)"

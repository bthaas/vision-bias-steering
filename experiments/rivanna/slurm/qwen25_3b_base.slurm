#!/bin/bash
#SBATCH --job-name=bias_qwen25_3b_base
#SBATCH --account=uvasrg_paid
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/qwen25_3b_base_%j.out
#SBATCH --error=logs/qwen25_3b_base_%j.err

set -euo pipefail

SCRATCH=/scratch/jea7vy/vision-bias-steering
REPO=$SCRATCH
VENV=$SCRATCH/venv

module load gcc/11.4.0
module load openmpi/4.1.4
module load python/3.11.4
module load cuda/12.4.1

source $VENV/bin/activate

cd $REPO

mkdir -p $SCRATCH/results/qwen25_3b_base
mkdir -p experiments/rivanna/logs

echo "Starting Qwen2.5-3B (base) experiment at $(date)"
echo "Node: $SLURMD_NODENAME  GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python experiments/rivanna/run_experiment.py \
    --model Qwen/Qwen2.5-3B \
    --template B_positioned \
    --output-dir $SCRATCH/results/qwen25_3b_base \
    --data-dir $REPO/experiments/rivanna/data \
    --n-train 800 \
    --n-val 200 \
    --n-qual 5 \
    --batch-size 16 \
    --max-new-tokens 20 \
    --seed 42

echo "Done at $(date)"

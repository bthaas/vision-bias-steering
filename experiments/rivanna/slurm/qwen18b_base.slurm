#!/bin/bash
#SBATCH --job-name=bias_qwen18b_base
#SBATCH --account=uvasrg_paid
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/qwen18b_base_%j.out
#SBATCH --error=logs/qwen18b_base_%j.err

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

mkdir -p $SCRATCH/results/qwen18b_base
mkdir -p experiments/rivanna/logs

echo "Starting Qwen-1.8B (base) experiment at $(date)"
echo "Node: $SLURMD_NODENAME  GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python experiments/rivanna/run_experiment.py \
    --model Qwen/Qwen-1_8B \
    --template B_positioned \
    --output-dir $SCRATCH/results/qwen18b_base \
    --data-dir $REPO/experiments/rivanna/data \
    --n-train 800 \
    --n-val 200 \
    --n-qual 5 \
    --batch-size 32 \
    --max-new-tokens 20 \
    --fine-lambdas "-150,-145,-140,-135,-130,-125,-120,-115,-110,-105,-100,-95,-90,-85,-80,-75,-70,-65,-60,-55,-50,-45,-40,-35,-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150" \
    --seed 42

echo "Done at $(date)"

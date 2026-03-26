#!/bin/bash
# Submit the three base-model experiments (1.8B, 3B, 7B) to Rivanna SLURM.
# Dense lambda sweep (range -150..150 step 5, 61 values) for visualization.
# Run from the repo root: bash experiments/rivanna/slurm/submit_base.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/../logs"
mkdir -p "$LOG_DIR"

echo "Submitting base-model bias-steering experiments (dense lambda sweep)..."

JOB1=$(sbatch --parsable "$SCRIPT_DIR/qwen18b_base.slurm")
echo "  Qwen-1.8B (base):  job $JOB1"

JOB2=$(sbatch --parsable "$SCRIPT_DIR/qwen25_3b_base.slurm")
echo "  Qwen2.5-3B (base): job $JOB2"

JOB3=$(sbatch --parsable "$SCRIPT_DIR/qwen25_7b_base.slurm")
echo "  Qwen2.5-7B (base): job $JOB3"

echo ""
echo "All submitted. Monitor with:"
echo "  squeue -u \$USER"
echo "  sacct -j $JOB1,$JOB2,$JOB3 --format=JobID,JobName,State,Elapsed"

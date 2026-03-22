#!/bin/bash
# Submit all three model experiments to Rivanna SLURM.
# Run from the repo root: bash experiments/rivanna/slurm/submit_all.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/../logs"
mkdir -p "$LOG_DIR"

echo "Submitting bias-steering experiments..."

JOB1=$(sbatch --parsable "$SCRIPT_DIR/qwen25_3b.slurm")
echo "  Qwen2.5-3B:   job $JOB1"

JOB2=$(sbatch --parsable "$SCRIPT_DIR/qwen25_7b.slurm")
echo "  Qwen2.5-7B:   job $JOB2"

JOB3=$(sbatch --parsable "$SCRIPT_DIR/qwen25_14b.slurm")
echo "  Qwen2.5-14B:  job $JOB3"

echo ""
echo "All submitted. Monitor with:"
echo "  squeue -u \$USER"
echo "  sacct -j $JOB1,$JOB2,$JOB3 --format=JobID,JobName,State,Elapsed"

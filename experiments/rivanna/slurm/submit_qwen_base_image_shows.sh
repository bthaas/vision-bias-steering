#!/bin/bash
# Submit the standardized Qwen base-model Image Shows comparison.
#
# This submits the Qwen-1.8B-chat verification first. The 3B and 7B base-model
# jobs run only after that verification job exits successfully.
#
# Run from the repo root:
#   bash experiments/rivanna/slurm/submit_qwen_base_image_shows.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LOG_DIR="$REPO_ROOT/experiments/rivanna/logs"
mkdir -p "$LOG_DIR"

echo "Submitting standardized Qwen Image Shows jobs..."
echo "Logs: $LOG_DIR"

VERIFY_JOB=$(sbatch --parsable "$SCRIPT_DIR/qwen_base_image_shows_verify.slurm")
echo "  Qwen-1.8B-chat verification: job $VERIFY_JOB"

JOB_3B=$(sbatch --parsable --dependency=afterok:$VERIFY_JOB "$SCRIPT_DIR/qwen_base_image_shows_3b.slurm")
echo "  Qwen2.5-3B base:             job $JOB_3B (afterok:$VERIFY_JOB)"

JOB_7B=$(sbatch --parsable --dependency=afterok:$VERIFY_JOB "$SCRIPT_DIR/qwen_base_image_shows_7b.slurm")
echo "  Qwen2.5-7B base:             job $JOB_7B (afterok:$VERIFY_JOB)"

echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  sacct -j $VERIFY_JOB,$JOB_3B,$JOB_7B --format=JobID,JobName,State,Elapsed"
echo ""
echo "After all three jobs complete, build the figure and CSV with:"
echo "  python experiments/rivanna/build_qwen_base_image_shows_outputs.py --results-root \${RESULTS_ROOT:-\$SCRATCH/results/qwen_base_image_shows}"

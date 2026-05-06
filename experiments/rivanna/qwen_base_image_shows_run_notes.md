# Qwen Base Image Shows Comparison

This is the standardized replacement for the old mixed multi-model comparison.
It uses base models only for the larger Qwen2.5 runs and uses the same
`The image shows` continuation-prefix setup as the local Qwen-1.8B-chat prompt
family curve in `qwen_prompt_family_tradeoff_main.png`.

## Standardized Setup

- Prompt instruction: `Describe this image:\n{caption}`
- Continuation prefix: `The image shows`
- Prompt serialization: `model.apply_chat_template(..., output_prefix="The image shows")`
- Validation split: `experiments/rivanna/data/val.json`
- Validation selection: first 1,000 captions
- Lambda grid: `-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60`
- Generation: greedy, `do_sample=False`, `max_new_tokens=20`
- Steering: full sequence, model-specific vector, selected model-specific layer
- Metric: continuation-level normalized spatial ratio
- Degeneration detector: repetition/TTR heuristic used by the coherence-frontier scripts

The 3B and 7B jobs refuse to run unless
`$RESULTS_ROOT/qwen18b_chat/verification_passed.json` exists. The submit script
also adds a Slurm `afterok` dependency so a failed 1.8B-chat verification blocks
the larger runs.

The verification step requires the local Qwen-1.8B-chat artifacts that produced
the old prompt-family curve:

- `runs_vision/Qwen-1_8B-chat/datasplits/val.json`
- `runs_vision/Qwen-1_8B-chat/activations/candidate_vectors.pt`
- `runs_vision/Qwen-1_8B-chat/activations/neutral.pt`

If those files are missing on Rivanna, the verification job exits before loading
the larger models and writes `qwen18b_chat/verification_failed.json`.

## Rivanna Commands

Run from the repo root on Rivanna:

```bash
bash experiments/rivanna/slurm/submit_qwen_base_image_shows.sh
```

If your repo or venv is not under `/scratch/jea7vy/vision-bias-steering`, set:

```bash
export PROJECT_ROOT=/path/to/vision-bias-steering
export VENV=$PROJECT_ROOT/venv
export RESULTS_ROOT=$PROJECT_ROOT/results/qwen_base_image_shows
```

Manual equivalents:

```bash
python experiments/rivanna/run_qwen_base_image_shows.py verify-local \
  --results-root $SCRATCH/results/qwen_base_image_shows

python experiments/rivanna/run_qwen_base_image_shows.py run-model \
  --model-key qwen25_3b_base \
  --results-root $SCRATCH/results/qwen_base_image_shows

python experiments/rivanna/run_qwen_base_image_shows.py run-model \
  --model-key qwen25_7b_base \
  --results-root $SCRATCH/results/qwen_base_image_shows
```

After all three model results exist:

```bash
python experiments/rivanna/build_qwen_base_image_shows_outputs.py \
  --results-root $SCRATCH/results/qwen_base_image_shows
```

## Outputs

The builder writes these files under `$SCRATCH/results/qwen_base_image_shows` by
default:

- `qwen_base_image_shows_ratio_curves.png`
- `qwen_base_image_shows_summary.csv`
- `qwen_base_image_shows_run_notes.md`

Per-model raw results are saved as:

- `qwen18b_chat/results.json`
- `qwen25_3b_base/results.json`
- `qwen25_7b_base/results.json`

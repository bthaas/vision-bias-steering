# vision-bias-steering

Vision steering experiments for spatial vs descriptive language in LLM outputs.

This repo contains:
- data loading and steering-vector extraction (`bias_steering/`)
- validation + evaluation pipeline (`bias_steering/run.py`)
- plotting/diagnostic entry point (`plotting/master_prompt_experiments.py`)
- canonical saved artifacts in `runs_vision/` (GPT-2 and Qwen)

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10+ is recommended.

## Current Workflow

### 1) Fresh train + validate run

```bash
python -m bias_steering.run \
  --model_name gpt2 \
  --method WMD \
  --target_concept vision \
  --pos_label spatial \
  --neg_label descriptive \
  --score_mode adaptive \
  --optimize_coeff \
  --batch_size 32
```

### 2) Reproduce from saved GPT-2 config

```bash
python -m bias_steering.run \
  --config_file runs_vision/gpt2/config.yaml \
  --use_cache
```

### 3) Run downstream evaluation

```bash
python -m bias_steering.run \
  --config_file runs_vision/gpt2/config.yaml \
  --run_eval
```

### 4) Generate master prompt experiment plots

```bash
python plotting/master_prompt_experiments.py \
  --model_name gpt2 \
  --artifact_dir runs_vision/gpt2 \
  --layer 5 \
  --output_html runs_vision/gpt2/validation/master_prompt_experiments.html \
  --output_json runs_vision/gpt2/validation/master_prompt_experiments.json
```

## Canonical Tracked Results

Metrics below come from tracked `runs_vision/*/validation/` artifacts.

### GPT-2 (`runs_vision/gpt2`)
- Baseline RMS bias: `0.3969`
- Best RMS bias: `0.2135` (layer `5`, coeff `-215`)
- RMS reduction: `46.2%`

### Qwen-1.8B-Chat (`runs_vision/Qwen-1_8B-chat`)
- Baseline RMS bias: `0.7073`
- Best RMS bias: `0.0494` (layer `11`, coeff `-200`)
- RMS reduction: `93.0%`

## Repository Layout

```text
vision-bias-steering/
├── bias_steering/
│   ├── run.py
│   ├── config.py
│   ├── data/
│   ├── steering/
│   └── eval/
├── plotting/
│   └── master_prompt_experiments.py
├── runs_vision/
│   ├── gpt2/
│   │   ├── config.yaml
│   │   ├── evaluation/
│   │   └── validation/
│   │       ├── debiased_results.json
│   │       └── signal_report.json
│   └── Qwen-1_8B-chat/
│       ├── config.yaml
│       ├── evaluation/
│       └── validation/
│           ├── debiased_results.json
│           └── signal_report.json
└── plots/
```

## Data Notes

Vision split files live in:
- `bias_steering/data/datasets/splits/vision_train.csv`
- `bias_steering/data/datasets/splits/vision_val.csv`

Target word classes are defined in:
- `bias_steering/data/datasets/target_words.json`

COCO processing entry point:
- `bias_steering/data/process_coco.py`

## Artifact Policy

To keep the repo clean:
- generated validation `*.html`/`*.png` files are ignored
- only core validation summaries are tracked (`debiased_results.json`, `signal_report.json`)
- alternate experimental run dirs (`runs_vision_*`) are ignored
- cache/junk files (`__pycache__`, `.ipynb_checkpoints`, `.DS_Store`) are ignored

## Credits

Adapted from [gender-bias-steering](https://github.com/hannahxchen/gender-bias-steering).

## License

See `LICENSE`.

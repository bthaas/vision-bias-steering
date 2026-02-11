
## 1. Constrained Softmax — Forcing Spatial vs Descriptive

Makes the model choose only between spatial and descriptive words (ignores the rest of the vocabulary). Three places use this same logic:

- [Baseline scoring](bias_steering/run.py) — `get_baseline_results()` scores each prompt before any intervention.
- [Debiasing test](bias_steering/steering/validate.py) — `run_debias_test()` scores prompts after applying the steering vector.
- [Coefficient sweep](bias_steering/coeff_test.py) — `run()` tests different steering strengths.

Token IDs come from [get_target_token_ids()](bias_steering/steering/steering_utils.py) which converts the word lists into token IDs (including capitalized and space-prefixed variants).

---

## 2. Target Words — What Counts as Spatial vs Descriptive

Defined in [target_words.json](bias_steering/data/datasets/target_words.json) under `"vision"`.

- **Spatial** (46 terms): left, right, above, below, beside, behind, next to, near, far, etc.
- **Descriptive** (62 terms): red, blue, green, large, small, round, square, bright, dark, etc.

Used for both constrained softmax scoring and COCO caption filtering.

---

## 3. COCO Filtering — Building the Dataset

[process_coco.py](bias_steering/data/process_coco.py) — `process_coco_captions()`

Each caption gets scored: `score = spatial_term_count − descriptive_term_count`. Only strong examples are kept:

- **Spatial**: score ≥ 5, OR at least 2 spatial terms with 0 descriptive terms.
- **Descriptive**: score ≤ −5, OR at least 2 descriptive terms with 0 spatial terms.

Outputs `vision_train.csv` and `vision_val.csv` in [splits/](bias_steering/data/datasets/splits/).

---

## 4. Prompt Templates

- [Train templates](bias_steering/data/datasets/instructions/vision_train.txt) (8 templates)
- [Val templates](bias_steering/data/datasets/instructions/vision_val.txt) (5 templates)

Format: `<instruction> | <output_prefix>` — e.g. `Continue describing this scene: | The scene is`. The caption gets appended after the instruction, and the output prefix starts the model's response so the first real token is what gets scored.

Loaded in [load_dataset.py](bias_steering/data/load_dataset.py) → `load_vision_dataset()`.

---

## 5. Steering Vector Extraction

[extract.py](bias_steering/steering/extract.py) — `extract_candidate_vectors()`

Grabs last-position activations for spatial-biased and descriptive-biased training examples at every layer, then computes one steering vector per layer (using WMD or MD). Saves `candidate_vectors.pt`.

---

## 6. Validation Pipeline

[validate.py](bias_steering/steering/validate.py) — `validate()`

1. **Projection correlation** — projects val activations onto each layer's vector, checks how well projections predict bias. Saves `projections.npy`, `proj_correlation.json`, `top_layers.json`.
2. **Debiasing test** — applies the steering vector to each top layer, re-scores everything, measures RMS reduction. Saves `debiased_results.json`, `debiased_scores.json`, `signal_report.json`.

### Intervention methods — [intervention.py](bias_steering/steering/intervention.py)

- **default** (projection): removes the bias component from activations.
- **constant**: just adds `steering_vec * coeff` to activations. This is what we use for vision runs.

---

## 7. Bias Score

```
bias = pos_prob − neg_prob           (spatial minus descriptive)
normalized_bias = bias / (pos_prob + neg_prob + 1e-10)
```

- **RMS bias** = `sqrt(mean(bias²))` — how much bias there is overall, regardless of direction.
- **Reduction %** = how much RMS dropped after debiasing.

---

## 8. Key Config Settings

Config lives in [config.py](bias_steering/config.py). GPT-2's saved config is [here](runs_vision/gpt2/config.yaml).

| Parameter | GPT-2 value | What it does |
|---|---|---|
| `constrained_softmax` | `true` | Only score spatial/descriptive tokens |
| `intervention_method` | `constant` | Adds `vec * coeff` to activations |
| `method` | `WMD` | How the steering vector is computed |
| `use_offset` | `false` | Center activations by neutral mean first |
| `bias_threshold` | `0.4` | Min bias to count as a training example |
| `coeff_search_min/max` | `−200 / 200` | Coefficient search range |
| `evaluate_top_n_layer` | `5` | How many layers get full debiasing tests |

---

## 9. How to Run

| What | Command |
|---|---|
| Full train + validate | `python -m bias_steering.run --model_name gpt2 --method WMD --target_concept vision --pos_label spatial --neg_label descriptive --constrained_softmax` |
| From saved config | `python -m bias_steering.run --config_file runs_vision/gpt2/config.yaml --use_cache --optimize_coeff` |
| Coefficient sweep | `python -m bias_steering.coeff_test --config_file runs_vision/gpt2/config.yaml --layer 5 --min_coeff -200 --max_coeff 200 --increment 20 --constrained_softmax` |
| Reprocess COCO data | `python -c "from bias_steering.data.process_coco import process_coco_captions; process_coco_captions()"` |

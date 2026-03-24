# Vision Bias Steering — Project Context

Steering-vector research: shift spatial→descriptive token probability in LLMs via WMD activation patching. Paper pipeline; no web frontend, no build system.

---

## Module Map

| Path | Role |
|------|------|
| `bias_steering/steering/model.py` | `ModelBase` (nnsight wrapper): `apply_chat_template`, `get_activations`, `get_logits`, `generate`, `load_model` |
| `bias_steering/steering/extract.py` | `extract_candidate_vectors` — WMD vectors, all layers |
| `bias_steering/steering/validate.py` | Layer selection (projection-RMSE), λ sweep, coherence frontier |
| `bias_steering/steering/intervention.py` | Hook-based activation patching at inference |
| `bias_steering/steering/steering_utils.py` | `get_all_layer_activations`, `scalar_projection` |
| `bias_steering/config.py` | `Config` + `DataConfig` dataclasses (YAMLWizard → `runs_*/*/config.yaml`) |
| `bias_steering/run.py` | Local pipeline entry point |
| `bias_steering/eval/winogenerated.py` | Downstream WinoGenerated evaluation |
| `experiments/rivanna/run_experiment.py` | Self-contained HPC pipeline (no library needed on Rivanna) |
| `plotting/master_prompt_experiments.py` | Diagnostic Plotly HTML plots |

---

## Algorithm Invariants — CRITICAL

- **WMD weights**: raw signed bias scores — NOT `abs().square()`. See REALIGNED comment at `extract.py:59`.
- **score_mode**: always `"prob_diff"`. `"adaptive"` causes extraction/validation metric mismatch.
- **Bias metric**: constrained softmax over target token set only (`all_ids = pos_ids + neg_ids`, overlap removed).
- **Extraction prompts**: diverse per-example templates from `datasplits/*.json` (`prompt`/`output_prefix` columns). `B_positioned` template is for RMS measurement only — the two baselines (diverse ~0.55–0.70 RMS vs B_positioned ~0.98 RMS) are **not comparable across runs**.
- **Layer selection**: mismatch-RMSE (scale-invariant; penalises sign disagreement only), always skip L0, prefer middle-third as tiebreak.
- **Layer 0 pathology**: standard RMSE picks L0/L1 (embedding scale dominates numerics). Mismatch-RMSE mitigates but doesn't eliminate cross-model label issue.
- **Cross-model label issue**: `val.json` bias scores were computed with Qwen-1.8B-chat. Using as projection targets for Qwen2.5 (different representation space) is an approximation; proper fix is to recompute val bias in-run with `compute_bias_nointervene`.

---

## Model Loading Quirks

| Pattern | Special handling |
|---------|-----------------|
| `Qwen/Qwen-*` (v1) | Custom `QWEN_CHAT_TEMPLATE`, `pad_token='<\|extra_0\|>'`, `block_module_attr="transformers.h"` |
| GPT-2 | `apply_chat_template` try/except falls back to raw string (no chat template) |
| Qwen2.5-* (all variants) | Default `ModelBase`; `block_module_attr` auto-detected via `detect_module_attrs` → `"model.layers"` |

`detect_module_attrs` checks: `model.layers` → `transformers.h` → `transformer.h`, else raises.

---

## Canonical Results

| Model | Baseline RMS | Best reduction | Layer | Template |
|-------|-------------|----------------|-------|----------|
| GPT-2 | 0.397 | 46.2% (λ=−215) | 5/12 | diverse |
| Qwen-1.8B-chat | 0.707 | 33.3% coherent full-steer (λ=−20) | 11/24 | diverse |
| Qwen-1.8B-chat | 0.707 | 67.2% 1-token steer (λ=−50) | 11/24 | diverse |
| Qwen2.5-3B-Instruct | ~0.98 | 56.8% 1-token | TBD (layer fix pending) | B_positioned |
| Qwen2.5-7B-Instruct | ~0.98 | 54.3% 1-token | TBD | B_positioned |
| Qwen2.5-14B-Instruct | ~0.98 | 84.5% 1-token | TBD | B_positioned |
| Qwen2.5-3B (base) | — | pending | — | B_positioned |
| Qwen2.5-7B (base) | — | pending | — | B_positioned |

Coherence heuristic: TTR + max_freq + bigram_rep on generated output.
Single-layer steering optimal; multi-layer ranges degrade faster (validated on Qwen-1.8B).

---

## CLI Quickref

```bash
# Fresh local run
python -m bias_steering.run \
  --model_name gpt2 --method WMD --target_concept vision \
  --pos_label spatial --neg_label descriptive \
  --score_mode prob_diff --batch_size 32

# Reproduce from saved config
python -m bias_steering.run --config_file runs_vision/gpt2/config.yaml --use_cache

# Downstream eval
python -m bias_steering.run --config_file runs_vision/gpt2/config.yaml --run_eval

# Rivanna: submit all 5 jobs (3B/7B/14B Instruct + 3B/7B base)
bash experiments/rivanna/slurm/submit_all.sh

# Rivanna: single model
python experiments/rivanna/run_experiment.py \
  --model Qwen/Qwen2.5-3B --template B_positioned \
  --output-dir $SCRATCH/results/qwen25_3b_base \
  --data-dir experiments/rivanna/data \
  --n-train 800 --n-val 200 --batch-size 16 --seed 42
```

---

## Data Paths

| File | Contents |
|------|----------|
| `bias_steering/data/datasets/splits/vision_{train,val}.csv` | Local train/val splits |
| `bias_steering/data/datasets/target_words.json` | Spatial / descriptive token lists |
| `data/handcrafted_eval.json` | Hand-curated eval captions |
| `experiments/rivanna/data/train.json` | 800-example train set; columns: `text`, `prompt`, `output_prefix`, `vision_label`, `bias` (Qwen-1.8B labels) |
| `experiments/rivanna/data/val.json` | 200-example val set; same schema |
| `experiments/rivanna/data/target_words.json` | Rivanna copy of target token lists |
| `runs_vision/*/validation/debiased_results.json` | **Tracked** canonical val metrics |
| `runs_vision/*/validation/signal_report.json` | **Tracked** per-layer projection signal |
| `runs_vision/*/config.yaml` | Saved `Config` for reproduction |

---

## Active Status

- Rivanna Qwen2.5-Instruct: layer selection fix (mismatch-RMSE + skip L0) committed, awaiting rerun.
- Qwen2.5 base models (3B, 7B): added to pipeline 2026-03-24; first runs pending.
- If mismatch-RMSE still picks early layers: add in-run `compute_bias_nointervene` on val to replace cross-model Qwen-1.8B labels.
- Paper decision pending: standardise on one template (diverse vs B_positioned) before reporting final numbers.

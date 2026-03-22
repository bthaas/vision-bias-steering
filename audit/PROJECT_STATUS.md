# Project Status — Vision Bias Steering
_Last updated: 2026-03-21_

---

## What Works

### Local Qwen-1.8B-chat (fully validated)

- **Extraction**: WMD steering vectors extracted using diverse per-example templates
  (stored in `runs_vision/Qwen-1_8B-chat/datasplits/`). Spatial/descriptive bias scores
  have clear separation under these templates.
- **Layer selection**: Projection-RMSE on diverse val prompts correctly identifies
  layer 11 (middle of 24-layer network) as the optimal steering layer.
- **Metrics** (all using constrained softmax, B_positioned template, val set):
  - Baseline RMS: 0.707
  - Optimal lambda: 93.0% RMS reduction (degenerate text — not paper-ready)
  - Coherence frontier (full steering): λ=−20 → **33.3%** RMS reduction, coherent text
  - 1-token steering: λ=−50 → **67.2%** RMS reduction, coherent text (unusual patterns)
- **Layer-selective steering experiments** (coherence_frontier/): single layer 11 is
  optimal; multi-layer ranges degrade faster with less net reduction.

### Rivanna Qwen2.5 — 1-token steering (promising after layer fix)

After fixing layer selection (projection-RMSE replacing coeff=0 RMS):
- Qwen2.5-3B-Instruct:  1-token **56.8%** reduction
- Qwen2.5-7B-Instruct:  1-token **54.3%** reduction
- Qwen2.5-14B-Instruct: 1-token **84.5%** reduction

These are genuine first-token metric results (constrained softmax, B_positioned).

### Infrastructure

- `experiments/rivanna/run_experiment.py`: single-file pipeline for Rivanna HPC
- `experiments/rivanna/data/`: train.json + val.json with stored diverse templates
  and Qwen-1.8B bias labels (`vision_label`, `bias`, `prompt`, `output_prefix`)
- `bias_steering/`: local library (model wrapper, extraction, validation, generation)
- `audit/`: diagnosis documents (METRICS_AUDIT, RIVANNA_DIAGNOSIS, RIVANNA_DIAGNOSIS_V2)

---

## What's Still Broken

### 1. Layer selection picks layer 0 or 1 (embedding layers) for Rivanna models

**Root cause**: `select_best_layer_by_projection` V1 used standard RMSE, which is
sensitive to activation norm scale. Layer 0 (embedding) has large, systematically
different activation norms from mid-network layers. The standard RMSE accidentally
minimises at layer 0 because projection magnitudes there happen to be numerically
close to the stored Qwen-1.8B bias score values (−0.78 to −0.15 range).

**Additional cross-model issue**: The stored val.json bias scores are from Qwen-1.8B.
Using these as projection targets for Qwen2.5 (different model series, different
representation space) introduces a cross-model mismatch. Layer 0's token-level
features may correlate with stored word-frequency-based Qwen-1.8B biases without
reflecting meaningful Qwen2.5 internal geometry.

**Current fix** (committed, not yet on Rivanna):
- Switch to mismatch-RMSE (scale-invariant: only penalises sign disagreement)
- Always skip layer 0
- Report both unconstrained best and middle-third best for comparison

### 2. Full steering ~0% coherent frontier for all Rivanna models

When layer 0 or 1 is steered at large |λ|, all generated tokens have their embedding
modified → catastrophic degeneration for all lambdas → zero coherent frontier.

Expected with correct layer: full steering should match Qwen-1.8B behaviour (33–40%
coherent frontier), though larger models may need smaller |λ| to remain coherent.

### 3. Layer selection is fundamentally cross-model

The stored val.json bias scores are from Qwen-1.8B (older model family, different
chat template, different representation space). Even with mismatch-RMSE, there is no
guarantee the projection correlation against Qwen-1.8B labels picks the correct
Qwen2.5 layer. The middle-third constraint is a heuristic mitigation.

A proper fix would recompute val bias scores during the Rivanna run using the current
model and diverse val templates — adding one forward pass per val example but ensuring
the projection targets are model-specific.

---

## What's Left To Do

### Immediate (next Rivanna run)

1. **Rerun with mismatch-RMSE + layer 0 skipped**: check if unconstrained best and
   middle-third best land in meaningful layers (layers 8–20 for 28-layer 7B, etc.)
2. **Inspect full layer ranking** (now logged to Rivanna stderr): compare mismatch-RMSE
   values across layers to verify the signal is non-degenerate.
3. **Check full-steering coherence frontier** with correctly selected layer: should
   show partial or coherent results at small |λ| (−10 to −20).

### If mismatch-RMSE still picks early layers

4. **Recompute val bias for current model**: add a step in `run_experiment.py` to run
   `compute_bias_nointervene` on diverse val prompts during the run and use these as
   projection targets instead of stored Qwen-1.8B bias. This eliminates the
   cross-model mismatch entirely.

### For the paper

5. **Decide on evaluation template**: local pipeline uses diverse val templates (RMS
   baseline ~0.55–0.70); Rivanna currently uses B_positioned (baseline ~0.98). These
   are not comparable. Choose one consistently for all models.
6. **Scale results**: once correct layer is found, sweep λ from −100 to +100 in steps
   of 10 to find coherence frontier for each model size.
7. **Coherence evaluation at scale**: the current coherence heuristic (TTR, max_freq,
   bigram_rep) is lightweight; verify a few examples manually on Rivanna output.

### Longer term

8. **Base model comparison**: Qwen2.5-3B (base) vs Qwen2.5-3B-Instruct may behave
   very differently under steering. Base models are less entrenched in spatial bias
   from RLHF. If Instruct models resist steering, base models are worth trying.
9. **Multi-layer steering**: experiments/coherence_frontier/ showed single-layer is
   optimal for Qwen-1.8B; verify same holds for larger models.

---

## Key Numbers Summary

| Model                  | Baseline RMS | Full-steer frontier | 1-token frontier | Layer | Status |
|------------------------|-------------|---------------------|------------------|-------|--------|
| Qwen-1.8B-chat (local) | 0.707       | 33.3% @ λ=−20      | 67.2% @ λ=−50   | 11/24 | ✓ done |
| Qwen2.5-3B-Instruct    | ~0.98       | ~0% (layer 0/1 bug) | 56.8%            | 0→TBD | in progress |
| Qwen2.5-7B-Instruct    | ~0.98       | ~0% (layer 0/1 bug) | 54.3%            | 0→TBD | in progress |
| Qwen2.5-14B-Instruct   | ~0.98       | ~0% (layer 0/1 bug) | 84.5%            | 0→TBD | in progress |

Note: Rivanna baseline RMS ~0.98 uses B_positioned; local uses diverse templates.
These are on different scales and should not be directly compared in the paper.

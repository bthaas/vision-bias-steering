# Rivanna Diagnosis V2

**Symptom:** Qwen2.5-3B: 18.1% reduction, Qwen2.5-7B: 1.4%, Qwen2.5-14B: 0.4%.
Local Qwen-1.8B-chat: 93%. Baseline RMS ≈ 0.98 for all Rivanna models.
1-token and full steering are identical.

---

## Root Cause 1 — Val prompts use B_positioned, saturating larger Instruct models (CRITICAL)

### What the Rivanna script does

```python
val_prompts = build_prompts(val_sub["text"].tolist(), template, model)
```

`build_prompts` applies the B_positioned template ("Positioned" output prefix) to every
val example. For larger Qwen2.5 Instruct models (3B / 7B / 14B), "Positioned" is a strong
spatial prime. The model predicts spatial tokens with near-certainty for ALL captions:

- Spatial caption → bias ≈ +0.98
- Descriptive caption → bias ≈ +0.97

Baseline RMS = sqrt(mean(bias²)) ≈ 0.98 — near the ceiling.

### What the local pipeline does

`validate.py` line 173:
```python
prompts = model.apply_chat_template(
    val_data.prompt.tolist(),
    output_prefix=val_data.output_prefix.tolist()
)
```

It uses the per-example diverse templates stored in val.json (`output_prefix` ∈
{"The scene is", "The image shows", "In this scene, the", …}). These do not uniformly
prime spatial, so biases vary across examples:

- val.json spatial examples: mean bias = −0.155
- val.json descriptive examples: mean bias = −0.780
- Baseline RMS ≈ 0.55–0.70 (real variance, room for reduction)

### Consequence

With baseline RMS ≈ 0.98 under B_positioned, every val example is near-ceiling.
Even a perfect steering vector cannot show dramatic RMS reduction because:

1. After steering, bias moves from +0.98 toward 0. But the metric is bounded by 0.
2. At lambda = −60, 18.1% reduction means bias went from 0.98 → 0.80. That IS real
   steering — but it looks small because the baseline was artificially inflated by
   the template, not by content.

---

## Root Cause 2 — Layer selection on B_positioned val with near-zero variance (CRITICAL)

### What Rivanna does

```python
def select_best_layer(...):
    for layer in eval_layers:
        ivfunc = get_intervention_func(vec, method="default", coeff=0, offset=off)
        bias_arr = compute_bias_intervened(...)  # B_positioned val prompts
        reduction = (baseline_rms - rms) / baseline_rms * 100
```

When all val biases ≈ +0.98 (no variance), the coeff=0 projection removal reduces each
example by roughly the same tiny amount — the layer that "wins" is whichever has the
smallest random noise. The reported reductions for all layers are indistinguishable
(all near 0%). Layer selection is effectively random.

This explains the reported "best layers": 33/36 for 3B, 7/28 for 7B, 37/48 for 14B —
these are not special layers; they just happened to have the smallest noise.

### What the local pipeline does

`validate.py` → `evaluate_candidate_vectors`:

```python
projs = scalar_projection(acts - offset, vec)
r = pearsonr(projs, bias_scores)
rmse = RMSE(projs, bias_scores)
results.sort(key=lambda x: x["RMSE"])  # lower RMSE = better alignment
```

It measures how well each layer's scalar projection onto the candidate vector
**predicts the stored bias scores** (from diverse-template Qwen-1.8B run).
The stored bias scores have real variance (spatial −0.155 vs descriptive −0.780).
The layer with the best linear alignment is the one where the activation geometry
most faithfully encodes the spatial/descriptive distinction.

This is robust to template-primed saturation: the projections reflect the activation
directions in the model, not the output distribution under a specific output prefix.

### Consequence

Broken layer selection cascades to every downstream step:
- Wrong layer → weak/wrong-direction steering vector
- Lambda sweep on wrong layer → low reduction at all λ
- Coherence is unaffected (barely any text change) → "coherent" at all λ
- 1-token and full-token show same coherence label (no difference when steering is negligible)

---

## Root Cause 3 — 1-token and full-token are identical (EXPECTED, not a bug)

The RMS metric measures **first-token logit bias**. Both token modes steer the first
token identically, so the RMS values are the same by construction. The token-limited
sweep intentionally reuses the full-sweep RMS from the lookup table.

The expected difference is in **coherence**. With the correct layer and an effective
steering vector, larger lambdas should degenerate full-steering but remain coherent
under 1-token steering. The current experiments show identical coherence because the
steering effect is negligible at the wrong layer — the text barely changes regardless
of token limit.

---

## Root Cause 4 — Reduction worsens with model size (downstream effect)

This is not a separate bug. Larger models are more strongly primed by B_positioned
(they have more spatial training signal), so:
1. Baseline saturation is higher (0.98 for all, but with less variance)
2. Layer selection noise is amplified
3. The reduction formula's ceiling effect is worse

With correct layer selection and diverse-template evaluation, the reduction should
scale differently. Large Instruct models may actually be harder to steer (their
biases are more deeply embedded), but we do not know the true magnitude because
the current evaluation methodology masks it.

---

## Not a bug — Base vs Instruct model

The local Qwen-1.8B-chat is also an Instruct model ("chat" suffix). Both local and
Rivanna use Instruct variants. The model family difference (Qwen 1.5 vs Qwen 2.5)
and capability level are real differences, but these are not the root cause of the
broken numbers. The template saturation and layer selection issues dominate.

---

## Verification of Fix

After applying the fixes:

**Expected extraction bias (train, diverse templates):**
```
spatial   mean bias ≈ +0.2 to +0.9  (varies by model capability)
descriptive mean bias ≈ −0.5 to −0.8
```
If this gap is <0.3, log a warning: the model's diverse-template bias signal is weak.

**Expected layer selection:** Best layer should be a mid-to-late layer (not random).
For Qwen2.5 models with 28–48 layers, expect best layer in the 10–40 range.

**Expected RMS baseline (diverse templates):** ≈ 0.55–0.85 (matching local pipeline
range), not 0.98.

**Expected reduction at coherence frontier:** Should approach 30–70% like local
pipeline, not 1–18%.

---

## Fix Applied

Two changes to `run_experiment.py`:

**1. Add `select_best_layer_by_projection` function**

Uses scalar projection RMSE against stored val.json bias scores (diverse templates),
exactly matching the local pipeline's `evaluate_candidate_vectors`.

**2. Switch main() to use diverse-template val prompts for layer selection**

Builds val prompts from stored `prompt`/`output_prefix` columns in val.json.
Passes stored bias scores as the target for projection correlation.
Keeps B_positioned for the lambda sweep and final paper-metric reporting.

**3. Add extraction bias diagnostics by label**

Logs mean bias for spatial and descriptive groups separately after recomputation.
Warns if the gap is <0.3 (weak signal, steering vector likely to be poor).

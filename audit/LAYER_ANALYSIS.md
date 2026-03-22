# Layer Selection Analysis: Why Mismatch-RMSE and Correlation Disagree

**Observation from Rivanna logs (Qwen2.5 3B/7B/14B):**
- Mismatch-RMSE selects early layers (5–8) with weak correlation (|r| ≈ 0.13)
- Late layers (24–26 for 7B, 39–41 for 14B) have strong correlation (|r| ≈ 0.4–0.5,
  p-values highly significant) but higher mismatch-RMSE

---

## What Each Metric Measures

### Mismatch-RMSE (lower = better)

```
mismatch_rmse = RMS of bias[i]  where  sign(proj[i]) ≠ sign(bias[i])
```

This penalises examples where the scalar projection onto the steering vector has the
**wrong sign** relative to the stored bias label. It is scale-invariant: it does not
matter whether proj values are large or small, only whether they agree in direction.

A layer with low mismatch-RMSE means: most val examples' activations project onto the
candidate vector in the correct direction (spatial → positive, descriptive → negative).

**What it misses**: it treats a small positive projection the same as a large positive
projection, as long as the sign is correct. It has no concept of graded strength.

### Pearson |r| (higher = better)

```
r = corr(proj[i], bias[i])
```

Pearson correlation measures whether the **magnitude** of the projection tracks the
magnitude of the bias score. High |r| means: examples with more extreme spatial bias
have larger projections; examples with more extreme descriptive bias have more negative
projections. The relationship is linear and graded.

A layer with high |r| encodes the full spectrum of spatial/descriptive strength, not
just the binary direction.

---

## Why They Disagree for Larger Instruct Models

### Early layers (5–8): low mismatch-RMSE, low |r|

These layers see mostly token-level and low-level positional features. The WMD
candidate vector at these layers captures which tokens are more common in spatial vs
descriptive captions (e.g., prepositions, spatial words). This gives:

- **Correct sign**: "near" and "in front of" tokens do appear more in spatial captions,
  so the projection is weakly positive for spatial examples → correct sign → low
  mismatch-RMSE
- **Weak magnitude**: the token-frequency signal is noisy; bias magnitude does not
  scale smoothly with caption spatial-ness → low |r|

Steering at these layers works at the sign level but cannot steer with precision.
The 1-token results (56.8%, 54.3%, 84.5%) may partly reflect this: the first-token
metric only cares about whether the model predicts a spatial vs descriptive token,
not the degree of confidence. Full steering at early layers causes catastrophic
degeneration because every intermediate representation is disturbed.

### Late layers (24–26 for 7B, 39–41 for 14B): high |r|, higher mismatch-RMSE

Late layers in large Instruct models are where high-level semantic content is
assembled. The WMD vector at these layers encodes the spatial/descriptive distinction
as a **direction in a higher-level semantic space**. This gives:

- **Strong magnitude alignment**: the degree to which an example projects onto the
  steering direction correlates strongly with how spatially biased the model is for
  that example → high |r|, highly significant p-values
- **Higher mismatch-RMSE**: the Qwen-1.8B stored bias scores (our projection targets)
  were computed under different templates and a different model family. The direction
  of the spatial/descriptive distinction in Qwen2.5's late-layer residual stream may
  be **rotated relative to Qwen-1.8B's direction**, even though the linear correlation
  is strong. Concretely: if the spatial direction in Qwen2.5 layer 25 is 180° rotated
  compared to what Qwen-1.8B labels predict, mismatch-RMSE is high even though |r|
  is high (r would be strongly negative rather than strongly positive). The sign of r
  tells us whether the rotation is 0° (positive r) or 180° (negative r).

**The sign of the correlation matters**: if r ≈ −0.4 at layer 25, the steering vector
is pointing in the OPPOSITE direction of what we want. The fix would be to negate the
candidate vector (or use coeff with opposite sign). If r ≈ +0.4, the vector is
pointing in the right direction and mismatch-RMSE is high because the Qwen-1.8B bias
target scale is different from the projection magnitude scale.

---

## Hypothesis: Late Layers Encode Spatial/Descriptive in Larger Instruct Models

For Qwen-1.8B-chat (24 layers), the optimal layer is 11 (middle of network). The
spatial/descriptive distinction is encoded midway through, where abstract semantic
features have been computed but before the final output projection.

For Qwen2.5-7B-Instruct (28 layers) and Qwen2.5-14B-Instruct (48 layers), the
signal shifts to very late layers (25–26/28, 39–41/48). This is consistent with
how larger Instruct models are trained:

1. **More capacity → deeper computation**: larger models perform more processing before
   the final answer is assembled. High-level semantic features like "is this caption
   describing location or appearance" may only emerge in the final few layers.
2. **RLHF alignment pushes categorisation late**: Instruct fine-tuning teaches the
   model to evaluate prompts before responding. The spatial/descriptive classification
   that drives output probabilities may be computed very late in the network, just
   before the language model head.
3. **Early layers are more general**: for large models, early layers handle syntax
   and low-level patterns across all tasks. The task-specific "is this spatial?"
   signal is not strong enough to dominate the early-layer residual stream.

---

## Implication for Steering

If the spatial/descriptive direction is in late layers, steering there should:

1. **Produce stronger full-token steering**: late-layer interventions modify the
   high-level "what should I say" representation rather than low-level token choices.
   The model should produce coherent spatial or descriptive text without degeneration.
2. **Require different lambda range**: the steering vector at late layers may have
   different magnitude characteristics. The lambda values that work for mid-layers
   on Qwen-1.8B (λ ≈ −20 for coherent full steering) may not transfer directly.
3. **Test with negative coefficients too**: if r is negative at a late layer, the
   stored candidate vector points the wrong direction. We should test both positive
   and negative λ to find which direction reduces spatial bias.

---

## What the Corr-Top3 Sweep Will Tell Us

The current run tests the top-3 layers by |r| alongside the mismatch-RMSE best.
Comparing the results:

| Outcome | Interpretation |
|---------|---------------|
| Late-layer corr-top3 shows coherent full steering | Hypothesis confirmed: late layers are the right target |
| Late-layer corr-top3 shows ~0% reduction | The vector direction is wrong (r is negative); try negating |
| Late-layer corr-top3 shows high 1-token, low full-token | Late-layer full steering still degenerates; try smaller λ |
| All layers show similar results | The val.json Qwen-1.8B bias targets are not reliable for any layer of Qwen2.5; need to recompute val bias with current model |

The sign of the best-lambda in the sweep will disambiguate: if λ < 0 gives reduction
and λ > 0 increases bias, the vector points the right direction; if the opposite, it's
inverted and corr is negative.

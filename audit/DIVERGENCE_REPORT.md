# Divergence Report: vision-bias-steering vs. gender-bias-steering (reference)

**Reference**: `hannahxchen/gender-bias-steering` (cloned to `/tmp/reference-repo`)
**Local**: `vision-bias-steering` (this repo)
**Date**: 2026-03-20

---

## Summary of Flags

| Severity | Count | Short description |
|---|---|---|
| 🔴 CRITICAL | 2 | WMD weighting change; validate coeff=0 abandoned |
| 🟠 MEDIUM | 3 | Adaptive score mode; constrained softmax; overlapping token dedup |
| 🟡 MINOR | 4 | Lazy imports; GPT-2 module detection; chat template fallback; normalized bias output |

---

## File-by-File Comparison

---

### `bias_steering/steering/extract.py`

**Same**: `mean_diff`, `get_activations`, overall loop structure, neutral-offset logic, MD path.

**Different**:

| # | Location | Reference | Local | Assessment |
|---|---|---|---|---|
| 🔴1 | WMD weight formula (line 54-57) | `torch.Tensor(pos_examples.bias.tolist())` / `neg_examples.bias.tolist()` | `torch.Tensor(pos_examples.bias.abs().tolist()) ** 2` | **CRITICAL — methodological change** |

#### Detail on 🔴1 — WMD weight squaring

The reference weights each training example by its raw bias score. Positive examples have positive bias, negative examples have negative bias; after `w = weights / weights.sum()`, both produce valid convex combinations (negative values / negative sum = positive fractions).

The local repo replaces this with `abs() ** 2`, meaning:
- Weights are always non-negative (absolute value).
- The quadratic amplification gives outlier examples disproportionate influence.
- For the vision domain, where bias scores may cluster differently than gender bias scores, this could shift the steering vector toward edge-case prompts rather than the modal distribution.

**Risk**: The steering vector extracted by the local repo is methodologically **not** what the paper describes. If WMD results look weaker than MD results in experiments, this is the first place to investigate.

---

### `bias_steering/steering/validate.py`

**Same**: `evaluate_candidate_vectors` core logic (Pearson correlation + RMSE to rank layers), `run_debias_test` structure, basic RMS reporting.

**Different** (local adds significant machinery not present in reference):

| # | Location | Reference | Local | Assessment |
|---|---|---|---|---|
| 🔴2 | `validate()` — intervention coefficient | Always tests `coeff=0` (pure projection removal) | Sweeps a range of coefficients; selects best by RMS | **CRITICAL — changes what is being tested** |
| 🟠3 | `run_debias_test()` return values | Returns `bias` only (`pos - neg`) | Returns `(bias, normalized_bias, pos_probs, neg_probs)` | Medium — changes downstream consumers |
| 🟠4 | `run_debias_test()` + `compute_target_scores()` — score mode | Always `prob_diff` = `pos_probs - neg_probs` | Adds `logit_margin = logsumexp(pos_logits) - logsumexp(neg_logits)` and `constrained_softmax` option | Medium — different bias measure |
| 🟡5 | `evaluate_candidate_vectors()` | No `offsets` parameter | Accepts `offsets` tensor; subtracts offset per layer before projection | Minor — consistent with offset logic in extract |
| 🟡6 | `validate()` — reporting | Simple before/after RMS | Adds `normalized_rms`, `reduction_pct`, `mass_stats`, label accuracy, template sweep | Minor — additive |

#### Detail on 🔴2 — coeff=0 vs. coefficient search

The reference `validate()` tests **only** `coeff=0`. In the "default" intervention method:

```
acts_new = acts - proj(acts - offset, unit_vec) + unit_vec * coeff
```

At `coeff=0` this reduces to pure orthogonal projection removal: the component of the activation along the steering direction is zeroed out. This is the theoretically motivated operation — you are testing whether the extracted vector is a real bias axis.

The local repo instead **searches over a range of coefficients** (default: -30 to +30, step 5) and reports the best-RMS result. This has two problems:

1. **It conflates "does the vector work" with "what strength works"**. A weak or noise vector can still reduce RMS at some nonzero coefficient by chance.
2. **It optimizes on the validation set**, meaning results are optimistically biased. The reference avoids this by fixing coeff=0.

If you need to tune a coefficient, it should be done on a held-out test split, not the same validation set used to select the layer.

#### Detail on 🟠4 — logit_margin vs. prob_diff

The reference always uses `prob_diff = pos_probs - neg_probs` as the bias score fed into WMD weighting and validation RMS. The local adds `logit_margin = logsumexp(pos_logits) - logsumexp(neg_logits)`.

`logit_margin` is a cleaner scale-invariant measure (analogous to log-odds), but it is **different from what WMD weights were designed around** in the paper. Using `logit_margin` for validation while `prob_diff` is used in the WMD weight construction creates a mismatch: the vector was extracted to explain `prob_diff` variance, but validation selects layers by `logit_margin` RMSE.

If `score_mode="adaptive"` (the default in `config.py`), the code may pick different score modes for extraction vs. validation, making results non-comparable across runs.

---

### `bias_steering/steering/intervention.py`

**Same**: Identical between repos. Orthogonal projection formula, `default` and `constant` methods, `intervene_generation` loop — all match exactly.

---

### `bias_steering/steering/model.py`

**Same**: `ModelBase` class, `apply_chat_template`, `get_activations`, `get_logits`, `generate`, `load_model`, Qwen special-case.

**Different**:

| # | Location | Reference | Local | Assessment |
|---|---|---|---|---|
| 🟡7 | `detect_module_attrs()` | Detects `model.layers` or `transformers.h` | Also detects `transformer.h` | Minor — adds GPT-2 support |
| 🟡8 | `apply_chat_template()` | No error handling on `tokenizer.apply_chat_template` | Wraps in `try/except (ValueError, AttributeError)`, falls back to raw string | Minor — GPT-2 compatibility |

No methodological impact; these are practical accommodations for models without instruction-tuning.

---

### `bias_steering/steering/steering_utils.py`

**Same**: Identical between repos. `get_token_ids`, `get_target_token_ids`, `get_all_layer_activations`, `scalar_projection`, `compute_projections` all match.

---

### `bias_steering/steering/__init__.py`

**Same**: Exports the same symbols.

**Different**:

| # | Location | Reference | Local | Assessment |
|---|---|---|---|---|
| 🟡9 | Import style | Direct `from .extract import ...` | Lazy wrapper functions that import on first call | Minor — avoids circular import issues at module load |

No behavioral difference.

---

### `bias_steering/run.py`

**Same**: `parse_arguments` structure, `weighted_sample`, `train_and_validate` skeleton, `eval` function structure.

**Different**:

| # | Location | Reference | Local | Assessment |
|---|---|---|---|---|
| 🟠10 | `get_baseline_results()` return value | Returns `(pos_probs, neg_probs)` | Returns `(pos_probs, neg_probs, bias_scores, logit_margin)` | Medium — changes callers |
| 🟠11 | `remove_overlapping_target_ids()` | Not present | Removes token IDs shared between pos and neg sets; raises if either becomes empty | Medium — important for vocabulary overlap |
| 🟠12 | `pick_adaptive_score_mode()` | Not present | Heuristically selects between `prob_diff` and `logit_margin` using label accuracy or dynamic range | Medium — see score mode discussion above |
| 🟡13 | `train_and_validate()` bias score column | `df["bias"] = pos_probs - neg_probs` | Stores `bias_prob_diff`, `bias_logit_margin`, and `bias` (from selected mode); also stores `use_offset` handling | Minor-medium — richer diagnostics |

#### Detail on 🟠11 — overlapping token IDs

This is an **addition that is methodologically correct** and absent from the reference. If the same token (e.g., the word "spatial" tokenizing to a token that is also in the descriptive set) appears in both pos and neg target lists, `prob_diff` is systematically damped. The reference does not guard against this. For vision-specific target words, overlap should be verified.

---

### `bias_steering/config.py`

**Same**: `DataConfig` fields (n_train, n_val, bias_threshold, output_prefix, weighted_sample), `Config` base fields (model_name, data_cfg, method, use_offset, evaluate_top_n_layer, filter_layer_pct, save_dir, use_cache, batch_size, seed).

**Different** (all additions in local):

| Field | Reference default | Local default | Assessment |
|---|---|---|---|
| `target_concept` | `"gender"` | `"vision"` | Expected domain change |
| `pos_label` / `neg_label` | `"F"` / `"M"` | `"spatial"` / `"descriptive"` | Expected domain change |
| `constrained_softmax` | not present | `False` | Additive option |
| `score_mode` | not present | `"adaptive"` | 🟠 Heuristic auto-select |
| `intervention_method` | not present (hardcoded in run.py) | `"default"` | Fine |
| `optimize_coeff` | not present | `True` | 🔴 Enables coeff search by default |
| `debias_coeff` | not present | `None` | Fine — allows override |
| `coeff_search_min/max/increment` | not present | `-30 / 30 / 5.0` | 🔴 Sets default search range |
| `force_layer` | not present | `None` | Fine |
| `prompt_template_sweep` | not present | `False` | Fine — off by default |

---

### `bias_steering/data/load_dataset.py`

**Same**: `load_dataframe_from_json`, `load_datasplits` structure, instruction template sampling pattern.

**Different**: Domain-specific loaders. Reference loads `gender` (text rewrite) and `race` (AAL) datasets. Local loads `vision` (image caption) dataset. Pattern is structurally identical; no methodological issues.

---

### `bias_steering/data/template.py` and `bias_steering/data/prompt_iterator.py`

**Same**: Identical between repos.

---

### `bias_steering/eval/winogenerated.py` and `bias_steering/eval/task.py`

**Same**: Identical between repos (gender-specific eval retained).

---

### Files in reference but not in local

| File | Purpose | Risk |
|---|---|---|
| `bias_steering/eval/occupation.py` | Occupation-based gender bias eval | No risk; out-of-scope for vision domain |
| `bias_steering/data/process_data.py` | Data preprocessing for gender/race datasets | No risk |
| `bias_steering/coeff_test.py` | Standalone script for coefficient tuning experiments | Minor — useful diagnostic absent locally |

### Files in local but not in reference

| File | Purpose |
|---|---|
| `bias_steering/data/process_coco.py` | COCO-specific dataset preprocessing |
| `bias_steering/steering/validate.py` additions | Extended validation machinery |
| `plotting/` directory | Visualization scripts |

---

## Ranked Issues by Methodological Risk

### 🔴 CRITICAL

**C1 — WMD weights use `abs().square()` instead of raw bias** (`extract.py:56-57`)

The reference uses linear bias-proportional weighting. The local version squares the absolute value, making the steering vector much more sensitive to high-bias outliers. This is not described in the paper method and could produce a different (not necessarily worse, but *different*) steering direction. Any comparison to published results is invalid under this change.

**Recommendation**: If you have a reason to prefer quadratic weighting (e.g., you found it worked better empirically), document this explicitly. Otherwise revert to match the paper's WMD definition and test separately.

---

**C2 — Validation tests coefficient search instead of coeff=0** (`validate.py:251`, `config.py:29`)

The reference's `validate()` uses `coeff=0` to test pure projection. The local searches -30 to +30 and picks the best. This optimizes the metric on the validation set and makes it hard to assess whether the steering vector is actually capturing a real semantic axis. A good-looking RMS number from coeff search could result from any direction that partially correlates with the output token.

**Recommendation**: Add a separate "vector quality" check at `coeff=0` (as reference does) alongside the optional coeff sweep. The coeff=0 result is your ground truth signal; the sweep result is your practical operating point.

---

### 🟠 MEDIUM

**M3 — `score_mode="adaptive"` creates extraction/validation mismatch** (`config.py:27`, `run.py:175-190`)

WMD weights are computed from `prob_diff` bias scores. But if `score_mode` resolves to `logit_margin`, validation RMS is computed on a different scale. Layer ranking by RMSE is then inconsistent with the weighting used to extract the vector.

**Recommendation**: Separate the extraction score mode from the validation score mode, or fix both to `prob_diff` to match the reference.

---

**M4 — Constrained softmax changes what bias is being measured** (`validate.py:66-68`, `run.py:132-140`)

With `constrained_softmax=False` (default), prob_diff measures the portion of full vocabulary probability mass assigned to each class — this is what the reference computes. With `constrained_softmax=True`, it measures the conditional probability within only the tracked token set, which can dramatically inflate apparent effect sizes if tracked tokens have low total probability mass.

**Recommendation**: Default should stay `False` to match reference. If `constrained_softmax=True` is used for reporting, clearly state that it is a different metric.

---

**M5 — Overlapping token ID removal improves correctness** (`run.py:164-172`)

This is an improvement over the reference. If spatial/descriptive target token sets share vocabulary items (e.g., tokens that could tokenize to ambiguous forms), `prob_diff` is attenuated. The reference does not handle this. This local addition is **methodologically better** than the reference.

---

### 🟡 MINOR

**m6 — GPT-2 module detection and chat template fallback** (`model.py:31-32`, `model.py:104-108`)

Adds `transformer.h` detection and a try/except around `apply_chat_template`. Purely additive for compatibility, no methodological impact.

**m7 — `__init__.py` lazy imports** (`steering/__init__.py:6-13`)

Wraps `extract_candidate_vectors` and `validate` in lazy-loading stubs. No behavioral difference.

**m8 — Normalized bias output** (`validate.py:159`)

`normalized_bias = (pos - neg) / (pos + neg + eps)` added as supplementary output. Not used for layer selection or WMD weighting, so it is additive-only.

**m9 — Prompt template sweep** (`validate.py:285-346`)

Tests multiple prompt templates on the best layer/coeff. Additive diagnostic, not in reference. Could surface prompt sensitivity issues, which is useful.

---

## Overall Assessment

The local repo has diverged from the reference in two ways that compromise the validity of comparisons to published results:

1. **The WMD vector extraction uses a different (quadratic) weighting formula.** The extracted steering vector is not the same object the paper evaluates.

2. **Validation optimizes a coefficient instead of testing at coeff=0.** This makes the validation metric an optimized upper bound, not a diagnostic of vector quality.

The remaining additions (logit_margin scoring, constrained softmax, adaptive mode) are reasonable engineering choices but create pipeline inconsistencies that should be resolved by aligning the score mode used for extraction with the one used for validation.

The `remove_overlapping_target_ids()` function is a genuine correctness improvement over the reference.

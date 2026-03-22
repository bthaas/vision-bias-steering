# Rivanna Experiment Diagnosis

**Symptom:** All Rivanna models got 7–12% RMS reduction vs 93% for Qwen-1.8B locally.
Log showed 2993 positive / 7 negative training examples after bias scoring.

---

## Root Cause 1 — Wrong prompt template for activation extraction (CRITICAL)

### What the local pipeline does

`run.py` calls `load_datasplits`, which loads `train.json`. That file already contains
a `prompt` column and an `output_prefix` column for every example — these are diverse,
per-example instruction templates drawn randomly from `vision_train.txt`:

```
"prompt":        "Continue describing this scene:\nA large herd of black cattle..."
"output_prefix": "The scene is"

"prompt":        "Add more details to this description:\na close up of a giraffe..."
"output_prefix": "The image shows"
```

Activation extraction (`get_all_layer_activations`) is called on these diverse prompts.
Because the output prefixes vary (The scene is / The image shows / I can see / …),
the model assigns different probabilities based on caption content, producing
clear positive/negative bias scores:
- Spatial caption → high positive bias (e.g. +0.99)
- Descriptive caption → high negative bias (e.g. -0.88)

B_positioned ("Positioned") is **only** used for RMS measurement at validation time.

### What the Rivanna script did

`run_experiment.py` called `build_prompts(train_df["text"], template, model)`, which
built every training prompt with the B_positioned template before computing bias.

The "Positioned" prefix **strongly primes spatial continuation** on newer, stronger
models (Qwen2.5-3B/7B/14B, Llama-3.1-8B). The model sees "Positioned" and assigns
high probability to spatial tokens for ALL captions, regardless of content:

- Spatial caption → bias ≈ +0.98
- Descriptive caption → bias ≈ +0.96

After threshold filtering (|bias| ≥ 0.05), **2993 positive / 7 negative** — because
almost everything is positive. The median-split fallback also fails: it splits
"very spatially biased" vs "slightly less spatially biased", not spatial vs descriptive.

---

## Root Cause 2 — Ground-truth labels ignored (CRITICAL)

`train.json` has a `vision_label` column for every example: "spatial" (1500) or
"descriptive" (1500) — perfectly balanced. The script ignored this and derived
group membership from computed bias scores instead.

Bias scores are valid as **WMD weights within a group** (they quantify how strongly
an example represents its class), but they should never be used to **define the groups**
because the scores depend on the prompt template.

---

## Root Cause 3 — Downstream failures from broken contrastive pairs

Once the contrastive pairs are wrong, every subsequent step is corrupted:

| Step | Expected | Got |
|------|----------|-----|
| WMD pos group | 800 spatial captions | 800 "most spatially biased" captions under B_positioned |
| WMD neg group | 800 descriptive captions | 7 barely-negative → median split: 800 "slightly less spatially biased" |
| Steering vector | Spatial–descriptive direction | Noise or near-zero direction |
| Best layer RMS | High reduction at coeff=0 | ~0% reduction (vector wrong direction) |
| Fine sweep | Clear frontier at some λ | Uniformly low, 7–12% reduction |

---

## Root Cause 4 — Identical 1-token / full-token numbers (expected, not a bug)

The RMS metric measures **first-token logit bias** (constrained softmax on token 1).
Both "full steering" and "1-token steering" steer the first token identically, so
RMS and reduction_pct are always equal between the two modes. The only expected
difference is in **coherence** of the generated continuation. This is correct behavior.

---

## Fix

Two changes to `run_experiment.py`:

**1. Use stored diverse templates for extraction**

Instead of `build_prompts(train_df["text"], template, model)` (B_positioned),
call `model.apply_chat_template` with the stored `prompt` + `output_prefix` columns.
Fall back to B_positioned only if those columns are absent.

**2. Split pos/neg by `vision_label` (ground truth)**

Split `train_df` by `vision_label == "spatial"` (pos) and `== "descriptive"` (neg)
before computing extraction bias scores. Use the computed bias scores only as WMD
weights within each group, not to determine group membership. Fall back to
threshold/median split only if no label column exists.

---

## Verification

After the fix, extraction bias on Qwen-1.8B (using stored templates) should show:
```
spatial   mean bias ≈ +0.70 to +0.99
descriptive mean bias ≈ -0.50 to -0.88
```
which matches the train.json stored values (Qwen-1.8B bias column).

For new models (Qwen2.5, Llama), the pattern should hold because the diverse
templates do not prime spatial tokens the way "Positioned" does.

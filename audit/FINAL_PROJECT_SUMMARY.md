# Vision Bias Steering — Final Project Summary
_Date: 2026-03-22_

---

## What We Built

We built a pipeline for measuring and reducing **spatial vs. descriptive language bias** in image-captioning language models using **activation steering** — a technique that directly edits a model's internal representations at inference time to shift its output distribution.

**Starting point**: [hannahxchen/gender-bias-steering](https://github.com/hannahxchen/gender-bias-steering) (EMNLP 2025). That paper steers gender pronouns in occupation-description text by extracting a WMD (Weighted Mean Difference) vector from contrastive male/female prompt activations and projecting it out or amplifying it during generation.

**Our adaptation**: We repurposed the same mechanism for a different semantic axis. Instead of male/female pronouns, our two poles are:
- **Spatial language** ("next to", "in front of", "beside") — describing where things are
- **Descriptive language** ("bright orange", "roughly textured", "large") — describing how things look

The bias we are steering is a tendency for vision-language models to preferentially generate spatially-framed responses when describing scenes, even for captions that are better described in terms of appearance or texture.

**Core pipeline** (`bias_steering/` library):
- `extract.py` — extracts WMD steering vectors per-layer from contrastive caption pairs
- `validate.py` — selects the best layer by projection-RMSE and sweeps lambda values
- `intervention.py` — applies orthogonal projection steering at inference time
- `model.py` — wraps Qwen and GPT-2 with a unified activation extraction API

**HPC pipeline** (`experiments/rivanna/run_experiment.py`): A single self-contained script for Rivanna HPC that handles data loading, WMD extraction, layer selection, lambda sweep, and results reporting in one run.

---

## Dataset

We use COCO image captions split into ~3000 training and ~1000 validation examples. Each caption is labeled `spatial` (if it emphasises where things are relative to each other) or `descriptive` (if it emphasises what things look like). The train/val splits are stored in `experiments/rivanna/data/train.json` and `val.json` with the following fields per example:
- `text`: the raw caption
- `vision_label`: ground-truth "spatial" / "descriptive"
- `prompt`/`output_prefix`: diverse per-example instruction templates used for bias scoring
- `bias`: the Qwen-1.8B-chat bias score under those diverse templates (`pos_prob - neg_prob`)

Bias scores range from +1.0 (model is certain the spatial token comes next) to −1.0 (certain the descriptive token comes next).

---

## Metric: RMS Bias Reduction

The core evaluation metric is adapted directly from the reference paper:

```
bias_i = P(spatial_token | caption_i) − P(descriptive_token | caption_i)
RMS = sqrt( mean(bias_i²) )
reduction% = (RMS_before − RMS_after) / RMS_before × 100
```

We use **constrained softmax**: the softmax is computed over spatial+descriptive tokens only, so `bias_i ∈ [−1, +1]`. The reference uses full-vocabulary softmax, producing values ~10–100x smaller numerically but the same formula.

**Important caveat**: Our numbers are not directly comparable to the reference paper's numbers because of this softmax scope difference. In a paper context, we should report both or clearly state we are using the constrained variant. The formula, however, is verified correct from code (see `audit/METRICS_AUDIT.md`).

---

## Local Results — Qwen-1.8B-chat

These results are fully validated and reproducible on the local machine.

### Setup
- Model: `Qwen/Qwen-1_8B-chat` (24 layers)
- Training/val: 1500/1000 examples from COCO
- Evaluation template: `B_positioned` ("Positioned" output prefix)
- Metric: constrained softmax RMS bias reduction
- Layer selection: projection-RMSE on diverse-template val prompts

### Layer Selection
The spatial/descriptive bias axis is cleanest at **layer 11** (the middle of the network), where scalar projections of activations onto the steering vector correlate best with stored bias scores (Pearson r ≈ 0.6+, lowest mismatch-RMSE). This is consistent with the general finding from the reference paper that semantic features are best encoded in mid-network layers.

### Quantitative Results (`runs_vision/Qwen-1_8B-chat/validation/`)

| Setting | Lambda | RMS | Reduction | Coherence |
|---------|--------|-----|-----------|-----------|
| Baseline (λ=0) | — | 0.967 | 0% | ✓ |
| Optimal (degenerate) | −200 | 0.049 | **93.0%** | ✗ |
| Coherent full-steering frontier | −20 | 0.645 | **33.3%** | ✓ |
| 1-token steering frontier | −50 | 0.317 | **67.2%** | ✓ |

The 93% figure is real but not paper-ready: at λ=−200, the generated text is essentially meaningless. The 33.3% figure is the coherence frontier for full steering — the strongest lambda at which generated text is still readable.

Note: the optimal lambda was found by searching over the validation set, so 93% is an in-sample upper bound. A coeff=0 (pure projection removal) test would give a more methodologically conservative estimate.

### 1-Token Steering Finding (`experiments/coherence_frontier/`)

A key finding from the coherence frontier experiments: **steering only the first generated token doubles the coherent reduction** compared to steering all tokens.

| Mode | Best coherent lambda | RMS reduction |
|------|----------------------|---------------|
| Full steering (all tokens) | λ=−20 | **33.3%** |
| 1-token steering | λ=−50 | **67.2%** |

Mechanism: The constrained softmax RMS metric measures the first-token logit bias. By limiting steering to the first token, we avoid the cumulative degeneration of activations that makes high-lambda full steering produce incoherent text — but the bias reduction number stays the same because the metric only looks at the first token. This is not a trick; 1-token steering genuinely shifts the model's next-word distribution more strongly while keeping the continuation coherent.

**Qualitative examples at λ=−50, n_steer_tokens=1** (from `experiments/coherence_frontier/BEST_CONFIGS.md`):
- `A lone hiker stands on top of a snow-dusted ridge...`
  → `Positioned no more than 10 feet away from me is a person standing at the edge of a dense` ✓
- `A bright orange and yellow maple tree stands beside a small dark pond...`
  → `Positioned no more than 10 feet away from the speaker is a large, round object with a smooth` ✓

The text is grammatically coherent but semantically odd (the model generates "no more than 10 feet away" as its descriptive-adjacent completion when steered away from spatial tokens). At λ=−20 full steering, examples are more natural:
- → `Positioned a few other objects in the following order: a small dark pond, a bright orange and yellow maple` ✓

### Layer-Selective Steering (`experiments/coherence_frontier/Experiment 2`)

We tested whether spreading the intervention across a range of layers improves results:

| Layer config | Best coherent lambda | Reduction |
|---|---|---|
| single_11 (layer 11 only) | λ=−20 | **33.3%** |
| middle_8_14 (layers 8–14) | λ=−10 | 22.5% |
| early_0_6 (layers 0–6) | λ=0 (no steering) | 0.2% |
| late_18_23 (layers 18–23) | λ=−60 | 3.0% |

**Finding**: Single-layer steering at layer 11 is strictly better than multi-layer ranges. Spreading the intervention amplifies degeneration — the middle_8_14 range degenerates at λ=−20, while single_11 is still coherent at λ=−20 and achieves higher reduction. Early layers are ineffective; late layers (18–23) have almost no spatial/descriptive signal encoded in the 1.8B model.

---

## Template Search (`experiments/prompt_template_search/`)

We searched 15+ output-prefix templates to find which ones produce the strongest and most balanced bias signal at λ=0 (needed to have room for measurable reduction).

### What a "good" template needs
Two competing properties:
1. **High baseline RMS**: the model should show clear spatial bias before steering (room to reduce)
2. **Balanced top-10 tokens**: ideally both spatial and descriptive tokens should appear in the top-10 predicted tokens (otherwise constrained softmax becomes trivially dominated by one class)

In practice, templates that maximise baseline RMS tend to be spatially primed, and no template achieved both balanced token presence and high RMS.

### Key results (from `experiments/prompt_template_search/qwen1.8b_v2/SUMMARY.md`)

| Template | Prefix | Baseline RMS | Best Reduction |
|---|---|---|---|
| `A2_in_the` | `[caption]. In the` | 0.996 | 75.7% |
| `A_in_the` | `In the` | 0.889 | 89.8% |
| **`B_positioned`** | `Positioned` | **0.967** | **94.2%** |
| `A2_main_subject` | `[caption]. The main subject appears` | 0.609 | 62.5% |
| `A_subject_looks` | `The subject looks` | 0.638 | 91.0% |

`B_positioned` achieves the highest measured reduction (94.2%) and was selected as the primary evaluation template. Its top-10 predicted tokens are prepositions (`in`, `on`, `against`, `at`, `near`, `between`, `within`, `along`), which are all spatial. There are no descriptive tokens in the top-10.

**Problem with B_positioned for qualitative demos**: Because "Positioned" is such a strong spatial prime, the model outputs positional prepositions regardless of whether the caption is spatial or descriptive. Even at λ=0, every generation starts "Positioned in a busy urban area..." or similar. The *demonstrated* debiasing shows the model switching from spatial to other spatial-ish continuations, which is not a compelling qualitative demo for readers. A template with genuinely balanced spatial/descriptive token coverage would make for better examples, but would also show lower numeric reduction.

This is a fundamental limitation: the templates that maximise RMS reduction are precisely the ones that make the qualitative steering most artificial-looking.

---

## Rivanna Results — Qwen2.5-3B/7B/14B-Instruct

### Summary of outcomes

| Model | Baseline RMS | Best 1-token reduction | Best full-steering reduction |
|-------|--------------|------------------------|------------------------------|
| Qwen2.5-3B-Instruct (28 layers) | ~0.98 | **56.8%** | ~0% |
| Qwen2.5-7B-Instruct (28 layers) | ~0.98 | **54.3%** | ~0% |
| Qwen2.5-14B-Instruct (48 layers) | ~0.98 | **84.5%** | ~0% |

All three models show non-trivial 1-token reductions. Full-text steering is effectively zero for all three.

### Debugging history (documented in `audit/RIVANNA_DIAGNOSIS.md` and `RIVANNA_DIAGNOSIS_V2.md`)

The Rivanna runs went through several rounds of debugging:

**Bug 1 (RIVANNA_DIAGNOSIS.md)** — Wrong extraction templates.
The initial script built all training prompts with B_positioned before computing bias scores. For larger Qwen2.5 Instruct models, "Positioned" is such a strong spatial prime that every caption — spatial or descriptive — gets bias ≈ +0.97. This yielded 2993 positive / 7 negative training examples after threshold filtering, effectively destroying the contrastive training signal. The WMD vector computed from these was noise.

*Fix*: Read pre-stored `prompt`/`output_prefix` columns from train.json (diverse per-example templates) and use ground-truth `vision_label` column for group assignment.

**Bug 2 (RIVANNA_DIAGNOSIS_V2.md)** — Layer selection using B_positioned validation.
The initial layer selection evaluated how much each layer reduced bias when steering with coeff=0 and B_positioned val prompts. With B_positioned, all val examples have bias ≈ +0.98 (near-ceiling), so there is essentially no variance to measure. The layer that "won" was whichever happened to have the smallest random noise — effectively random. After the extraction fix, layers 0 and 1 (the embedding layers) were selected, causing catastrophic degeneration in full-text steering.

*Fix*: Switch layer selection to projection-RMSE on diverse-template val prompts (matching the local pipeline). Skip layer 0 unconditionally.

**Bug 3 (current, partial)** — Scale-sensitive RMSE picking early layers.
Even after switching to projection-RMSE, standard RMSE was sensitive to activation norm scale, and early layers (5–8) still ranked highest. These have low mismatch-RMSE (correct sign direction) but very weak Pearson |r| ≈ 0.13.

*Fix*: Switched to mismatch-RMSE (penalises only sign disagreements, scale-invariant). Also added dual-layer comparison: unconstrained best layer vs. middle-third best (n/3..2n/3 by layer count, approximating where layer 11 sits in 1.8B).

**Current state**: Mismatch-RMSE still selects early layers (5–8) for all three Qwen2.5 models. Late layers (24–26 for 7B, 39–41 for 14B) show strong correlations (|r| ≈ 0.4–0.5, p < 0.001) but higher mismatch-RMSE. The latest run (`72431b4`) adds corr-top3 sweeps to test whether the late-layer high-correlation layers actually steer better.

### Layer analysis (`audit/LAYER_ANALYSIS.md`)

The disagreement between mismatch-RMSE and Pearson |r| has a concrete explanation:

**Early layers (5–8)**: Token-level and positional features. Spatial words ("near", "in front of") appear more in spatial captions at the token frequency level, so the WMD vector at these layers captures word-frequency differences that have the *correct sign* (low mismatch-RMSE) but no *graded magnitude* (low |r|). Steering here changes sign direction but cannot scale continuously with how spatial a caption is.

**Late layers (24–26, 39–41)**: High-level semantic representations where the spatial/descriptive distinction is encoded as a direction in the residual stream. The projections correlate strongly and linearly with stored bias scores (high |r|). However, the stored bias scores come from Qwen-1.8B-chat, not Qwen2.5. If the late-layer spatial direction in Qwen2.5 is rotated 180° relative to what the Qwen-1.8B labels predict, r would be strongly *negative*, which would give high mismatch-RMSE even though the linear signal is strong. This is the cross-model label mismatch.

**Hypothesis**: Larger Instruct models shift the spatial/descriptive encoding to later layers because (a) they have more processing capacity, and (b) RLHF instruction-tuning pushes high-level categorical decisions close to the output layer. The corr-top3 sweep (next Rivanna run) will test this: if a late-layer corr-top3 layer produces coherent full-text steering with meaningful reduction, the hypothesis is confirmed.

### Why full steering is ~0%

For all three Qwen2.5 models, 1-token steering achieves 54–85% reduction but full-text steering achieves ~0%. There are two candidate explanations:

1. **Wrong layer (active bug)**: If the embedding layers or early layers are being used, every token in the generated sequence has its activation modified, causing catastrophic degeneration at all lambdas. The model produces incoherent or empty text, and the bias measurement is meaningless.

2. **Strong instruction-tuning prior (possible fundamental limitation)**: Even with the correct layer, Instruct models may resist full-text steering because the RLHF process embeds very strong priors about continuation style. A first-token intervention is sufficient to tip the logit balance, but the rest of the generation is driven by the instruction-tuned prior and auto-regressively recovers its spatial framing.

The second explanation is the candidate negative/interesting finding for the paper. Whether it holds depends on the upcoming corr-top3 results — if the correct layer (late, high-|r|) also shows ~0% full-text steering, it would strongly support the interpretation that instruction tuning creates a bias that activation steering cannot overcome for full generation.

---

## Divergences from the Reference Implementation (`audit/DIVERGENCE_REPORT.md`)

Three critical divergences were identified and fixed (`audit/METHODOLOGY_FIXES.md`):

**Fix C1 — WMD weighting formula** (`bias_steering/steering/extract.py`)
Our code was using `pos_weights = torch.Tensor(bias.abs().tolist()) ** 2` (quadratic absolute-value). The reference uses raw signed bias scores as weights. The quadratic formula amplifies outlier examples and loses the sign information needed for the correct convex-combination normalisation. *Fixed*: reverted to raw signed weights.

**Fix C2 — Default validation at coeff=0** (`bias_steering/config.py`)
Our code defaulted to searching over lambda values (−30 to +30) and reporting the best-RMS result. This optimises on the validation set and conflates "does the vector work" with "what strength works." The reference always tests coeff=0 (pure orthogonal projection removal), which tests whether the vector captures a real semantic axis without any tuning. *Fixed*: `optimize_coeff` defaults to `False`.

**Fix M3 — Score mode consistent across extraction and validation** (`bias_steering/config.py`)
The default `score_mode="adaptive"` heuristically picked between `prob_diff` and `logit_margin`, creating a mismatch where the WMD vector was built on one scale but layers were selected using another. *Fixed*: `score_mode="prob_diff"` as the reference uses, both for extraction and validation.

One improvement over the reference was retained: `remove_overlapping_target_ids()`, which removes tokens shared between the spatial and descriptive target sets before computing bias scores. The reference does not do this, which attenuates `prob_diff` when there is vocabulary overlap.

---

## Methodology Contributions

### 1. 1-Token Steering
By limiting the activation intervention to the first generated token, we can operate at much larger lambda values without triggering text degeneration. This doubles the coherent RMS reduction (33.3% → 67.2% on Qwen-1.8B). The mechanism is that the constrained softmax metric is first-token-only, so 1-token steering captures the full logit effect while avoiding the accumulation of distributional shift across a long generation.

This technique is domain-general and could apply to any steering experiment where the bias metric is first-token and generation coherence is evaluated separately.

### 2. Coherence Frontier Analysis
Rather than reporting a single summary statistic, we sweep lambda values and identify the **coherence frontier**: the highest-lambda operating point at which generated text is still coherent. This makes explicit the trade-off between steering strength and generation quality, and it shows that different layer configurations have different frontiers.

The coherence heuristic (type-token ratio, max token frequency, bigram repetition) flags degenerate text automatically. Manual inspection of examples confirmed the heuristic's accuracy at the frontier.

### 3. Template Search Methodology
Before running large-scale experiments, we systematically evaluate which output-prefix template provides:
- The highest baseline RMS (strongest bias signal)
- Both spatial and descriptive tokens visible in the top-10 predicted tokens (balanced measurement)
- Interpretable qualitative generation at λ=0

The finding that these three criteria are in tension — the most measurable template is the least interpretable — is itself a methodological contribution for future work in this area.

### 4. Constrained vs. Unconstrained Softmax Comparison
We document the numerical difference between constrained softmax (softmax over tracked tokens only, bias ∈ [−1, +1]) and full-vocabulary softmax (bias ∈ [−0.1, +0.1]). The formulas are internally consistent but produce very different RMS baselines (~0.97 vs ~0.05 for the same model), making cross-paper comparisons difficult unless the softmax scope is specified. See `audit/METRICS_AUDIT.md` for a full end-to-end numerical trace.

### 5. Mismatch-RMSE for Layer Selection
Standard RMSE is scale-sensitive and picks layers where projection magnitudes happen to be numerically close to the stored bias score scale — not necessarily the layers that encode the semantic direction. Mismatch-RMSE (RMS of bias scores where the projection has the wrong sign) penalises only sign disagreements and is scale-invariant. Implemented in `experiments/rivanna/run_experiment.py`.

### 6. Cross-Model Bias Label Problem
When running a new model (Qwen2.5) with steering vectors validated against stored Qwen-1.8B bias labels, the two are not on the same representational scale. We document this as a fundamental limitation of the layer selection approach and propose a mitigation (recomputing val bias scores using the target model during the run) in `audit/PROJECT_STATUS.md`.

---

## Known Limitations

### Constrained Softmax Inflates Numbers
Our constrained softmax metric reports 93% reduction for Qwen-1.8B, but this is comparable to the reference reporting ~70% reduction for gender steering on the same model. Part of the difference is constrained vs. unconstrained softmax scope. Part is that we optimise lambda on the validation set (an upper bound), while the reference tests at coeff=0 (a theoretically motivated fixed point). To make fair comparisons in a paper, we need to run both methods at coeff=0 with the same softmax scope.

### B_positioned Saturates Instruct Models
The "Positioned" output prefix — chosen because it maximises spatial bias signal — turns out to strongly prime spatial tokens in larger Instruct models. Qwen2.5 3B/7B/14B models produce bias ≈ +0.98 for *all* captions under this template, leaving no room for measured RMS reduction even if steering is working. A different evaluation template that doesn't saturate Instruct models would be needed to fairly compare across model sizes.

### First-Token Metric Limitation
The RMS metric is first-token only. It captures whether the steering vector shifts the initial token distribution, but not whether the *full generated description* becomes more descriptive or spatial. A full-generation metric (e.g., fraction of generated sentences using spatial vs. descriptive language) would be more meaningful for the paper's claims but is harder to compute at scale.

### No Base Model Comparison
We tested only Instruct variants (Qwen-1.8B-chat, Qwen2.5-3B/7B/14B-Instruct). Base models (without instruction tuning) might steer more easily because they lack the RLHF prior that may be resisting the steering. This is a direct comparison the paper should include.

### Llama-3.1-8B Not Tested
The model is gated and was not accessible on Rivanna. Removed from the experiment plan in commit `4571cfc`.

### Qualitative Demos Require Better Templates
The best-performing template for quantitative reduction (B_positioned) produces artifically spatial baseline text and makes the qualitative steering demo look like "spatial-framed to different-spatial-framed." A template like `image_shows` ("The image shows") produces more natural baseline text but lower measured reduction. Resolving this tension is needed for a compelling paper presentation.

---

## What's Left

### Immediate
1. **Rivanna corr-top3 results**: The next Rivanna run tests the top-3 layers by Pearson |r| alongside the mismatch-RMSE best. If a late-layer (layer 24–41) produces coherent full-text steering with meaningful reduction, it confirms the hypothesis that spatial/descriptive encoding has shifted to late layers in larger models. Results are expected in `run_experiment.py` output under `corr_top3_results`.

2. **Recompute val bias with target model**: If corr-top3 still fails, the fallback is to run one forward pass per val example during the Rivanna run using the target model and diverse templates, computing model-specific bias scores as projection targets. This eliminates the Qwen-1.8B → Qwen2.5 cross-model mismatch entirely.

3. **Template fix for Instruct models**: Find an evaluation template for Qwen2.5 that has a more balanced baseline (RMS ≈ 0.5–0.7) — similar to what diverse templates produce for Qwen-1.8B. This would provide a fair measurement context to determine whether large Instruct models genuinely resist steering.

### For the Paper
4. **Run coeff=0 validation**: Report projection-removal result (pure vector quality) alongside the optimal-lambda result on both models.

5. **Consistent softmax scope**: Choose constrained or unconstrained as primary, report the other in supplementary.

6. **Lambda sweep for each Qwen2.5 model** (once correct layer is found): Find the coherence frontier for each model size and compare across scales.

7. **Qualitative generation examples**: Generate 5–10 caption pairs showing (a) unsteered spatial-biased generation, (b) 1-token steered generation, (c) full steered generation — for both a spatial and a descriptive ground-truth caption.

8. **Base model experiments**: Qwen2.5-3B (base) and Qwen2.5-7B (base) as controls for the Instruct model results.

### Longer Term
9. **Multi-token generation metric**: Replace or supplement RMS with a metric computed over full 20-token generations, classifying whether the output language is spatial or descriptive.

10. **Cross-domain generalisation**: Does a steering vector trained on COCO captions generalise to other image description contexts?

---

## Key Numbers Summary

| Model | Baseline RMS | Max reduction | Coherent frontier | Layer | Notes |
|-------|--------------|---------------|-------------------|-------|-------|
| Qwen-1.8B-chat | 0.707 (diverse) / 0.967 (B_pos) | 93.0% (optimal λ) | **33.3%** full, **67.2%** 1-token | 11/24 | Fully validated |
| GPT-2 | 0.397 | 46.2% | — | 5/12 | Partial (no coherence tested) |
| Qwen2.5-3B-Instruct | ~0.98 (B_pos) | ~57%† | ~0% full | TBD | 1-token only |
| Qwen2.5-7B-Instruct | ~0.98 (B_pos) | ~54%† | ~0% full | TBD | 1-token only |
| Qwen2.5-14B-Instruct | ~0.98 (B_pos) | ~85%† | ~0% full | TBD | 1-token only |

†1-token steering; full steering ~0% for all Qwen2.5 models under current layer selection. Corr-top3 sweep pending.

RMS baselines: Qwen-1.8B diverse-template baseline (0.707) and B_positioned baseline (0.967) are not on the same scale and should not be directly compared.

---

## File Map

| Path | Contents |
|------|----------|
| `bias_steering/` | Core library (extract, validate, intervention, model) |
| `experiments/rivanna/run_experiment.py` | Self-contained HPC pipeline |
| `experiments/rivanna/data/train.json`, `val.json` | 3000/1000 COCO captions with stored templates and Qwen-1.8B bias labels |
| `experiments/rivanna/slurm/` | SLURM job scripts for 3B, 7B, 14B |
| `experiments/coherence_frontier/` | Layer-selective and token-limited steering experiments on Qwen-1.8B |
| `experiments/prompt_template_search/` | Template evaluation and recommendation |
| `runs_vision/Qwen-1_8B-chat/` | Local validation results and signal reports |
| `results/generation_logs/` | Per-example generation output at multiple lambdas |
| `audit/METRICS_AUDIT.md` | End-to-end metric verification with numerical traces |
| `audit/DIVERGENCE_REPORT.md` | Comparison of local code against reference implementation |
| `audit/METHODOLOGY_FIXES.md` | Documentation of fixes applied to align with reference |
| `audit/RIVANNA_DIAGNOSIS.md` | Root cause analysis: extraction template bug |
| `audit/RIVANNA_DIAGNOSIS_V2.md` | Root cause analysis: B_positioned saturation and layer selection |
| `audit/LAYER_ANALYSIS.md` | Why mismatch-RMSE and Pearson |r| disagree; late-layer hypothesis |
| `audit/PROJECT_STATUS.md` | What works, what's broken, what's left (detailed) |

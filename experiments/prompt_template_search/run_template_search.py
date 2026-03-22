"""
Prompt template search: find fill-in-the-blank prefixes where the first predicted
token has real probability mass on BOTH spatial and descriptive words.

Usage:
  /opt/homebrew/bin/python3.11 experiments/prompt_template_search/run_template_search.py

Outputs:
  experiments/prompt_template_search/results.json         -- raw data
  experiments/prompt_template_search/SUMMARY.md           -- rankings
  experiments/prompt_template_search/QUALITATIVE_EXAMPLES.md  -- generated text
"""
import json, sys, logging, math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ── paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parents[2]
CFG_PATH    = REPO_ROOT / "runs_vision/gpt2/config.yaml"
ARTIFACT_DIR = REPO_ROOT / "runs_vision/gpt2"
DATASET_DIR = REPO_ROOT / "bias_steering/data/datasets"
DATA_FILE   = REPO_ROOT / "data/handcrafted_eval.json"
OUT_DIR     = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT))

from bias_steering.config import Config
from bias_steering.steering import load_model, get_target_token_ids
from bias_steering.steering.intervention import get_intervention_func
from bias_steering.data.load_dataset import load_handcrafted_eval

# ── load config & model ────────────────────────────────────────────────────────
cfg = Config.load(CFG_PATH)
LAYER = cfg.force_layer          # 5
INTERVENTION_METHOD = "default"
CONSTRAINED_SOFTMAX = cfg.constrained_softmax  # True
SWEEP_COEFFS = [-150, -100, -50, 0, 50, 100, 150]
MAX_NEW_TOKENS = 20

logging.info("Loading model: %s", cfg.model_name)
model = load_model(cfg.model_name)

# ── target token IDs ──────────────────────────────────────────────────────────
target_words = json.load(open(DATASET_DIR / "target_words.json"))["vision"]
pos_ids_raw = get_target_token_ids(model.tokenizer, target_words["spatial"])
neg_ids_raw = get_target_token_ids(model.tokenizer, target_words["descriptive"])
overlap = set(pos_ids_raw) & set(neg_ids_raw)
if overlap:
    logging.info("Removing %d overlapping token IDs", len(overlap))
pos_ids = [t for t in pos_ids_raw if t not in overlap]
neg_ids = [t for t in neg_ids_raw if t not in overlap]
all_ids = pos_ids + neg_ids
n_pos = len(pos_ids)
logging.info("pos_ids: %d  neg_ids: %d", len(pos_ids), len(neg_ids))

# ── load steering vector ──────────────────────────────────────────────────────
candidate_vectors = torch.load(ARTIFACT_DIR / "activations/candidate_vectors.pt")
steering_vec = model.set_dtype(candidate_vectors[LAYER])
logging.info("Steering vector loaded: layer %d, shape %s", LAYER, steering_vec.shape)

# ── candidate templates ────────────────────────────────────────────────────────
# Format: (name, output_prefix, motivation)
# The instruction is fixed: "Describe this image:\n{caption}"
# We change only the output_prefix (what precedes the blank).
CANDIDATE_TEMPLATES = [
    # Current template (baseline for comparison)
    ("image_shows",       "The image shows",         "current baseline — predicts 'a' first"),
    # Spatial-leaning candidates (verb suggests location)
    ("it_sits",           "It sits",                 "sit → spatial prepositions: on/beside/near"),
    ("it_rests",          "It rests",                "rest → spatial prepositions: on/against/beside"),
    ("it_stands",         "It stands",               "stand → spatial: on/beside/near/in front of"),
    ("it_lies",           "It lies",                 "lie → spatial: on/beside/along/beneath"),
    # Descriptive-leaning candidates (predicate adjective)
    ("it_appears",        "It appears",              "appear → adj: bright/large/old OR spatial: near"),
    ("it_looks",          "It looks",                "look → adj: bright/large OR preposition: like"),
    ("the_object_is",     "The object is",           "direct predicate: adj or preposition next"),
    ("it_is",             "It is",                   "minimal predicate: adj or preposition next"),
    # Balanced ambiguous candidates
    ("this_object_is",    "This object is",          "similar to above, different determiner"),
    ("the_item_is",       "The item is",             "another object reference"),
]

# ── helper: RMS ───────────────────────────────────────────────────────────────

def RMS(x):
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x ** 2)))

# ── helper: compute bias scores for a list of prompts ─────────────────────────

def compute_bias(prompts, layer=None, intervene_func=None):
    """Return (pos_probs, neg_probs, bias_scores) arrays."""
    from bias_steering.data.prompt_iterator import PromptIterator
    pos_all, neg_all, bias_all = [], [], []
    for batch in PromptIterator(prompts, batch_size=16):
        if layer is not None and intervene_func is not None:
            logits = model.get_logits(batch, layer=layer, intervene_func=intervene_func)
            logits_last = logits[:, -1, :]
        else:
            logits_last = model.get_last_position_logits(batch)
        target_logits = logits_last[:, all_ids]
        probs = F.softmax(target_logits, dim=-1)
        pos_probs = probs[:, :n_pos].sum(dim=-1).tolist()
        neg_probs = probs[:, n_pos:].sum(dim=-1).tolist()
        bias = [p - n for p, n in zip(pos_probs, neg_probs)]
        pos_all.extend(pos_probs)
        neg_all.extend(neg_probs)
        bias_all.extend(bias)
    return np.array(pos_all), np.array(neg_all), np.array(bias_all)

# ── helper: top-N tokens at blank ─────────────────────────────────────────────

def top_n_tokens(prompts, n=10):
    """Get the top-N predicted tokens at the blank position (no steering)."""
    from bias_steering.data.prompt_iterator import PromptIterator
    # Accumulate logit sums across all prompts
    logit_sum = None
    count = 0
    for batch in PromptIterator(prompts, batch_size=16):
        logits_last = model.get_last_position_logits(batch)  # (B, vocab)
        if logit_sum is None:
            logit_sum = logits_last.sum(dim=0)
        else:
            logit_sum = logit_sum + logits_last.sum(dim=0)
        count += logits_last.shape[0]
    avg_logits = logit_sum / count
    probs = F.softmax(avg_logits, dim=-1)
    topn = torch.topk(probs, n)
    tokens = [model.tokenizer.decode([tid.item()]).strip() for tid in topn.indices]
    probs_list = topn.values.tolist()
    return list(zip(tokens, probs_list))

# ── helper: generate text at a given coeff ────────────────────────────────────

def generate_at_coeff(prompt_str, coeff, max_new_tokens=MAX_NEW_TOKENS):
    ivfunc = get_intervention_func(steering_vec, method=INTERVENTION_METHOD, coeff=coeff)
    output = model.generate(
        [prompt_str],
        layer=LAYER,
        intervene_func=ivfunc,
        max_new_tokens=max_new_tokens,
    )
    return output[0] if output else ""

# ── build prompts from handcrafted eval ──────────────────────────────────────

hc_data = load_handcrafted_eval()
captions = hc_data["text"].tolist()
logging.info("Using %d handcrafted captions", len(captions))

instruction = "Describe this image:\n{caption}"

def build_prompts(output_prefix):
    raw = [instruction.format(caption=c) for c in captions]
    if cfg.data_cfg.output_prefix:
        return model.apply_chat_template(raw, output_prefix=[output_prefix] * len(raw))
    return model.apply_chat_template(raw)

# ── STEP 1: Check top-10 tokens at lambda=0 for each template ─────────────────
logging.info("=" * 60)
logging.info("STEP 1: Top-10 tokens at lambda=0")
logging.info("=" * 60)

step1_results = []
for name, prefix, motivation in CANDIDATE_TEMPLATES:
    prompts = build_prompts(prefix)
    top10 = top_n_tokens(prompts, n=10)
    pos_p, neg_p, bias_arr = compute_bias(prompts)
    # Balance score: 1 - |pos_mean - neg_mean|, higher = more balanced
    # Use RMS of bias as primary (lower = more balanced at baseline)
    baseline_rms = RMS(bias_arr)
    mean_pos = float(pos_p.mean())
    mean_neg = float(neg_p.mean())
    balance_gap = abs(mean_pos - mean_neg)

    logging.info("Template: %-20s  baseline_rms=%.4f  mean_pos=%.4f  mean_neg=%.4f",
                 name, baseline_rms, mean_pos, mean_neg)
    logging.info("  Top-10: %s", ", ".join(f"'{t}'({p:.3f})" for t, p in top10[:5]))

    # Fraction of top-10 that are spatial/descriptive tokens
    all_pos_tokens = set(model.tokenizer.decode([tid]).strip().lower() for tid in pos_ids)
    all_neg_tokens = set(model.tokenizer.decode([tid]).strip().lower() for tid in neg_ids)
    spatial_in_top10 = [(t, p) for t, p in top10 if t.lower() in all_pos_tokens]
    descriptive_in_top10 = [(t, p) for t, p in top10 if t.lower() in all_neg_tokens]

    step1_results.append({
        "name": name,
        "output_prefix": prefix,
        "motivation": motivation,
        "baseline_rms": baseline_rms,
        "mean_pos_prob": mean_pos,
        "mean_neg_prob": mean_neg,
        "balance_gap": balance_gap,
        "top10_tokens": top10,
        "spatial_in_top10": spatial_in_top10,
        "descriptive_in_top10": descriptive_in_top10,
        "n_spatial_in_top10": len(spatial_in_top10),
        "n_descriptive_in_top10": len(descriptive_in_top10),
    })

# Rank by lowest baseline_rms (most balanced)
step1_results.sort(key=lambda x: x["baseline_rms"])
logging.info("\nRanking by baseline_rms (lower = more balanced):")
for r in step1_results:
    logging.info("  %-20s  rms=%.4f  spatial_in_top10=%d  descriptive_in_top10=%d",
                 r["name"], r["baseline_rms"], r["n_spatial_in_top10"], r["n_descriptive_in_top10"])

# ── STEP 2: Top-3 templates — steering sweep ──────────────────────────────────
logging.info("=" * 60)
logging.info("STEP 2: Lambda sweep for top-3 templates")
logging.info("=" * 60)

top3 = step1_results[:3]
step2_results = []

for entry in top3:
    name = entry["name"]
    prefix = entry["output_prefix"]
    logging.info("Processing template: %s (%s)", name, prefix)

    prompts = build_prompts(prefix)
    baseline_rms = entry["baseline_rms"]

    coeff_results = []
    for coeff in SWEEP_COEFFS:
        ivfunc = get_intervention_func(steering_vec, method=INTERVENTION_METHOD, coeff=coeff)
        pos_p, neg_p, bias_arr = compute_bias(prompts, layer=LAYER, intervene_func=ivfunc)
        rms = RMS(bias_arr)
        reduction = (baseline_rms - rms) / baseline_rms * 100 if baseline_rms > 0 else 0.0
        coeff_results.append({
            "coeff": coeff,
            "rms": rms,
            "reduction_pct": reduction,
            "mean_pos_prob": float(pos_p.mean()),
            "mean_neg_prob": float(neg_p.mean()),
        })
        logging.info("  coeff=%+d  rms=%.4f  reduction=%.1f%%", coeff, rms, reduction)

    # Best reduction
    best = min(coeff_results, key=lambda x: x["rms"])
    step2_results.append({
        "name": name,
        "output_prefix": prefix,
        "baseline_rms": baseline_rms,
        "coeff_sweep": coeff_results,
        "best_coeff": best["coeff"],
        "best_rms": best["rms"],
        "best_reduction_pct": best["reduction_pct"],
    })

# ── STEP 3: Qualitative examples (greedy generation at -100, 0, +100) ─────────
logging.info("=" * 60)
logging.info("STEP 3: Qualitative generation examples")
logging.info("=" * 60)

qualitative_results = []
SHOW_COEFFS = [-100, 0, 100]

for entry in top3:
    name = entry["name"]
    prefix = entry["output_prefix"]
    logging.info("Generating for template: %s", name)
    prompts = build_prompts(prefix)

    examples_by_coeff = {}
    for coeff in SHOW_COEFFS:
        generations = []
        if coeff == 0:
            ivfunc = get_intervention_func(steering_vec, method=INTERVENTION_METHOD, coeff=0)
        else:
            ivfunc = get_intervention_func(steering_vec, method=INTERVENTION_METHOD, coeff=coeff)
        for i, prompt in enumerate(prompts[:5]):   # show 5 examples
            gen = model.generate([prompt], layer=LAYER, intervene_func=ivfunc,
                                 max_new_tokens=MAX_NEW_TOKENS)
            text = gen[0] if gen else ""
            generations.append({
                "caption": captions[i],
                "prefix": prefix,
                "generated": text,
            })
            logging.info("  [%s] coeff=%+d caption=%s... → '%s'",
                         name, coeff, captions[i][:40], text[:60])
        examples_by_coeff[str(coeff)] = generations

    qualitative_results.append({
        "name": name,
        "output_prefix": prefix,
        "examples_by_coeff": examples_by_coeff,
    })

# ── STEP 4: Beam decoding for top-3 (coeff=-100, 0, +100) ────────────────────
logging.info("=" * 60)
logging.info("STEP 4: Beam decoding for top-3")
logging.info("=" * 60)

beam_results = []
for entry in top3:
    name = entry["name"]
    prefix = entry["output_prefix"]
    prompts = build_prompts(prefix)
    logging.info("Beam decoding template: %s", name)

    examples_by_coeff = {}
    for coeff in SHOW_COEFFS:
        ivfunc = get_intervention_func(steering_vec, method=INTERVENTION_METHOD, coeff=coeff)
        generations = []
        for i, prompt in enumerate(prompts[:5]):
            gen = model.generate(
                [prompt], layer=LAYER, intervene_func=ivfunc,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=4,
            )
            text = gen[0] if gen else ""
            generations.append({
                "caption": captions[i],
                "prefix": prefix,
                "generated": text,
            })
        examples_by_coeff[str(coeff)] = generations

    beam_results.append({
        "name": name,
        "output_prefix": prefix,
        "examples_by_coeff": examples_by_coeff,
    })

# ── SAVE RESULTS ──────────────────────────────────────────────────────────────
results = {
    "config": {
        "model": cfg.model_name,
        "layer": LAYER,
        "intervention_method": INTERVENTION_METHOD,
        "constrained_softmax": CONSTRAINED_SOFTMAX,
        "sweep_coeffs": SWEEP_COEFFS,
        "max_new_tokens": MAX_NEW_TOKENS,
    },
    "step1_all_templates": step1_results,
    "step2_top3_sweep": step2_results,
    "step3_qualitative_greedy": qualitative_results,
    "step4_qualitative_beam": beam_results,
}

def to_serializable(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)

with open(OUT_DIR / "results.json", "w") as f:
    json.dump(results, f, default=to_serializable, indent=2)
logging.info("Saved results.json")

# ── Write SUMMARY.md ──────────────────────────────────────────────────────────

def write_summary(step1, step2):
    lines = [
        "# Prompt Template Search: Summary",
        "",
        "## Ranking (by baseline balance — lower RMS = more balanced at λ=0)",
        "",
        f"| Rank | Template | Output Prefix | baseline_rms | mean_spatial | mean_descriptive | spatial_in_top10 | desc_in_top10 |",
        f"|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(step1):
        lines.append(
            f"| {i+1} | `{r['name']}` | \"{r['output_prefix']}\" | "
            f"{r['baseline_rms']:.4f} | {r['mean_pos_prob']:.4f} | "
            f"{r['mean_neg_prob']:.4f} | {r['n_spatial_in_top10']} | {r['n_descriptive_in_top10']} |"
        )
    lines += [
        "",
        "## Top-10 Tokens at λ=0 (unsteered, averaged over all captions)",
        "",
    ]
    for r in step1:
        lines.append(f"### `{r['name']}` — \"{r['output_prefix']}\"")
        lines.append(f"*{r['motivation']}*")
        lines.append("")
        tok_str = " | ".join(f"`{t}` ({p:.3f})" for t, p in r["top10_tokens"])
        lines.append(f"Top-10: {tok_str}")
        if r["spatial_in_top10"]:
            spatial_hits = ", ".join(f"`{t}`" for t, _ in r["spatial_in_top10"])
            lines.append(f"  - **Spatial hits**: {spatial_hits}")
        if r["descriptive_in_top10"]:
            desc_hits = ", ".join(f"`{t}`" for t, _ in r["descriptive_in_top10"])
            lines.append(f"  - **Descriptive hits**: {desc_hits}")
        lines.append("")

    lines += [
        "## RMS Bias Reduction: Top-3 Templates",
        "",
        "Formula: `RMS = sqrt(mean((spatial_prob - descriptive_prob)²))` over all eval examples.",
        "Constrained softmax — probabilities normalised over spatial+descriptive tokens only.",
        "",
        f"| Template | Prefix | baseline_rms | best_rms | best_coeff | reduction% |",
        f"|---|---|---|---|---|---|",
    ]
    for r in step2:
        lines.append(
            f"| `{r['name']}` | \"{r['output_prefix']}\" | "
            f"{r['baseline_rms']:.4f} | {r['best_rms']:.4f} | "
            f"{r['best_coeff']:+d} | {r['best_reduction_pct']:.1f}% |"
        )
    lines += [
        "",
        "### Lambda sweep curves",
        "",
    ]
    for r in step2:
        lines.append(f"#### `{r['name']}`")
        lines.append(f"| coeff | rms | reduction% |")
        lines.append(f"|---|---|---|")
        for c in r["coeff_sweep"]:
            lines.append(f"| {c['coeff']:+d} | {c['rms']:.4f} | {c['reduction_pct']:.1f}% |")
        lines.append("")

    return "\n".join(lines)

summary_md = write_summary(step1_results, step2_results)
with open(OUT_DIR / "SUMMARY.md", "w") as f:
    f.write(summary_md)
logging.info("Saved SUMMARY.md")

# ── Write QUALITATIVE_EXAMPLES.md ─────────────────────────────────────────────

def write_qualitative(greedy, beam):
    lines = [
        "# Qualitative Generation Examples",
        "",
        "Each template shown at λ=−100 (steer toward descriptive), λ=0 (no steering), λ=+100 (steer toward spatial).",
        "Greedy decoding shown first, then beam (width=4).",
        "",
    ]
    for g_entry, b_entry in zip(greedy, beam):
        name = g_entry["name"]
        prefix = g_entry["output_prefix"]
        lines.append(f"## `{name}` — \"{prefix} ___\"")
        lines.append("")
        for coeff_str in ["-100", "0", "100"]:
            coeff_label = {"-100": "λ=−100 (→descriptive)", "0": "λ=0 (unsteered)", "100": "λ=+100 (→spatial)"}[coeff_str]
            lines.append(f"### {coeff_label}")
            lines.append("")
            lines.append("**Greedy:**")
            lines.append("")
            for ex in g_entry["examples_by_coeff"].get(coeff_str, []):
                lines.append(f"- *Caption*: {ex['caption'][:80]}")
                lines.append(f"  *Output*: `{ex['prefix']} {ex['generated'].strip()}`")
            lines.append("")
            lines.append("**Beam (width=4):**")
            lines.append("")
            for ex in b_entry["examples_by_coeff"].get(coeff_str, []):
                lines.append(f"- *Caption*: {ex['caption'][:80]}")
                lines.append(f"  *Output*: `{ex['prefix']} {ex['generated'].strip()}`")
            lines.append("")
    return "\n".join(lines)

qual_md = write_qualitative(qualitative_results, beam_results)
with open(OUT_DIR / "QUALITATIVE_EXAMPLES.md", "w") as f:
    f.write(qual_md)
logging.info("Saved QUALITATIVE_EXAMPLES.md")

logging.info("=" * 60)
logging.info("DONE. All outputs in: %s", OUT_DIR)
logging.info("=" * 60)

# Print final summary to stdout
print("\n" + "=" * 60)
print("PROMPT TEMPLATE SEARCH RESULTS")
print("=" * 60)
print(f"\nRanked by baseline balance (lower RMS = more balanced at λ=0):")
for i, r in enumerate(step1_results):
    marker = "  <-- TOP 3" if i < 3 else ""
    print(f"  {i+1}. '{r['output_prefix']:25s}'  rms={r['baseline_rms']:.4f}  "
          f"spatial_top10={r['n_spatial_in_top10']}  desc_top10={r['n_descriptive_in_top10']}{marker}")

print("\nRMS bias reduction (top 3):")
for r in step2_results:
    print(f"  '{r['output_prefix']:25s}'  baseline={r['baseline_rms']:.4f}  "
          f"best={r['best_rms']:.4f} @ coeff={r['best_coeff']:+d}  "
          f"reduction={r['best_reduction_pct']:.1f}%")

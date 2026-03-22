"""
Qwen-1.8B-chat natural template search — v2.
Tests Approach A (caption-integrated) and Approach B (short natural prefixes)
for templates where the FIRST predicted token admits both spatial and descriptive
continuations while remaining natural as image captioning output.

Usage:
  /opt/homebrew/bin/python3.11 experiments/prompt_template_search/run_qwen_v2.py

Outputs → experiments/prompt_template_search/qwen1.8b_v2/
  results.json, SUMMARY.md, QUALITATIVE_EXAMPLES.md, RECOMMENDATION.md
"""
import json, sys, logging, math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

REPO_ROOT    = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "runs_vision/Qwen-1_8B-chat"
CFG_PATH     = ARTIFACT_DIR / "config.yaml"
DATASET_DIR  = REPO_ROOT / "bias_steering/data/datasets"
OUT_DIR      = Path(__file__).parent / "qwen1.8b_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT))

from bias_steering.config import Config
from bias_steering.steering import load_model, get_target_token_ids
from bias_steering.steering.intervention import get_intervention_func
from bias_steering.data.load_dataset import load_handcrafted_eval, load_dataframe_from_json

# ── load config ────────────────────────────────────────────────────────────────
cfg = Config.load(CFG_PATH)
BEST_LAYER          = 11           # best layer from debiased_results.json
METHOD              = "default"
CONSTRAINED_SOFTMAX = True
SWEEP_COEFFS        = [-150, -100, -50, 0, 50, 100, 150]
MAX_NEW_TOKENS      = 20
N_VAL_FOR_RMS       = 100          # subsample val set for fast RMS computation
RNG_SEED            = 42
TOP_N_FOR_SWEEP     = 5            # top-N templates get full sweep

logging.info("Config: model=%s  best_layer=%d", cfg.model_name, BEST_LAYER)

# ── load model ─────────────────────────────────────────────────────────────────
logging.info("Loading model …")
model = load_model(cfg.model_name)

# ── target token IDs ───────────────────────────────────────────────────────────
target_words = json.load(open(DATASET_DIR / "target_words.json"))["vision"]
pos_ids_raw = get_target_token_ids(model.tokenizer, target_words["spatial"])
neg_ids_raw = get_target_token_ids(model.tokenizer, target_words["descriptive"])
overlap = set(pos_ids_raw) & set(neg_ids_raw)
if overlap:
    logging.info("Removing %d overlapping token IDs", len(overlap))
pos_ids = [t for t in pos_ids_raw if t not in overlap]
neg_ids = [t for t in neg_ids_raw if t not in overlap]
all_ids = pos_ids + neg_ids
n_pos   = len(pos_ids)
logging.info("pos_ids: %d  neg_ids: %d", len(pos_ids), len(neg_ids))

# ── load steering vector + offset ─────────────────────────────────────────────
candidate_vectors = torch.load(ARTIFACT_DIR / "activations/candidate_vectors.pt")
steering_vec      = model.set_dtype(candidate_vectors[BEST_LAYER])
neutral_acts      = torch.load(ARTIFACT_DIR / "activations/neutral.pt")
offset            = model.set_dtype(neutral_acts.mean(dim=1)[BEST_LAYER])
logging.info("Steering vector loaded: layer %d  shape %s", BEST_LAYER, steering_vec.shape)
logging.info("Offset loaded: shape %s", offset.shape)

# ── candidate templates ────────────────────────────────────────────────────────
#
# Format: (name, instruction_fn, output_prefix_fn, approach, description)
# instruction_fn(caption) → raw instruction string  (user message)
# output_prefix_fn(caption) → output_prefix string  (start of assistant turn)
#
# Approach A: caption is in the instruction (standard format), prefix varies.
# Approach A2: caption is embedded IN the output_prefix (caption-integrated).
# Approach B: caption always in instruction; short natural prefixes.

def _std_instruction(caption):
    return f"Describe this image:\n{caption}"

def _detail_instruction(caption):
    return f"Add more details to this description:\n{caption}"

def _continue_instruction(_caption):
    # No caption in instruction — used for caption-integrated (A2) format
    return "Continue describing this image:"

CANDIDATE_TEMPLATES = [
    # ── Approach A: caption in instruction, prefix = natural description starter ──
    ("A_image_shows",       _std_instruction,    lambda c: "The image shows",
     "A", "baseline — caption in instruction, standard prefix"),

    ("A_looking_more",      _std_instruction,    lambda c: "Looking more closely,",
     "A", "natural zoom-in; first token could be spatial/descriptive adj"),

    ("A_main_subject",      _std_instruction,    lambda c: "The main subject appears",
     "A", "predicate adj follows 'appears': large/bright/old/red…"),

    ("A_notable_detail",    _std_instruction,    lambda c: "The most notable detail is",
     "A", "noun/adj phrase follows; varies by scene"),

    ("A_in_the",            _std_instruction,    lambda c: "In the",
     "A", "'foreground'/'center'/'background' — foreground IS a spatial token"),

    ("A_subject_looks",     _std_instruction,    lambda c: "The subject looks",
     "A", "'looks [adj]': bright/large/old/wooden…"),

    # ── Approach A2: caption embedded in output_prefix (caption-integrated) ──────
    ("A2_looking_more",     _continue_instruction, lambda c: f"{c}. Looking more closely,",
     "A2", "caption in assistant voice then zoom-in"),

    ("A2_main_subject",     _continue_instruction, lambda c: f"{c}. The main subject appears",
     "A2", "caption in assistant voice then appearance descriptor"),

    ("A2_in_the",           _continue_instruction, lambda c: f"{c}. In the",
     "A2", "caption in assistant voice then spatial/foreground"),

    # ── Approach B: short natural prefixes ────────────────────────────────────────
    ("B_foreground",        _std_instruction,    lambda c: "In the foreground",
     "B", "'foreground' is a spatial token; continuation places scene element"),

    ("B_scene_depicts",     _std_instruction,    lambda c: "The scene depicts",
     "B", "neutral; model chooses spatial or descriptive opening"),

    ("B_looking_image",     _std_instruction,    lambda c: "Looking at the image,",
     "B", "meta-observer; model describes what it sees"),

    ("B_subject_is",        _std_instruction,    lambda c: "The subject is",
     "B", "predicate: adj or preposition follows 'is'"),

    ("B_positioned",        _std_instruction,    lambda c: "Positioned",
     "B", "spatial-leaning: on/beside/above/near as first token"),

    ("B_visually",          _std_instruction,    lambda c: "Visually,",
     "B", "signals descriptive information; adj likely next"),
]

# ── helpers ────────────────────────────────────────────────────────────────────

def RMS(x):
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x ** 2)))


def build_prompts(instruction_fn, prefix_fn, captions):
    """Apply chat template for each caption."""
    instructions = [instruction_fn(c) for c in captions]
    prefixes     = [prefix_fn(c)      for c in captions]
    if cfg.data_cfg.output_prefix:
        return model.apply_chat_template(instructions, output_prefix=prefixes)
    return model.apply_chat_template(instructions)


def compute_bias_batch(prompts, layer=None, ivfunc=None):
    """Forward pass → (pos_probs, neg_probs, bias_scores) arrays."""
    from bias_steering.data.prompt_iterator import PromptIterator
    pos_all, neg_all, bias_all = [], [], []
    for batch in PromptIterator(prompts, batch_size=16):
        if layer is not None and ivfunc is not None:
            logits = model.get_logits(batch, layer=layer, intervene_func=ivfunc)
            logits_last = logits[:, -1, :]
        else:
            logits_last = model.get_last_position_logits(batch)
        tgt = logits_last[:, all_ids]
        probs = F.softmax(tgt, dim=-1)
        pp = probs[:, :n_pos].sum(-1).tolist()
        np_ = probs[:, n_pos:].sum(-1).tolist()
        pos_all.extend(pp);  neg_all.extend(np_)
        bias_all.extend([p - n for p, n in zip(pp, np_)])
    return np.array(pos_all), np.array(neg_all), np.array(bias_all)


def top_n_tokens(prompts, n=10):
    """Average logits over prompts and return top-N (token_str, prob) pairs."""
    from bias_steering.data.prompt_iterator import PromptIterator
    logit_sum, count = None, 0
    for batch in PromptIterator(prompts, batch_size=16):
        lgs = model.get_last_position_logits(batch)
        logit_sum = lgs.sum(0) if logit_sum is None else logit_sum + lgs.sum(0)
        count += lgs.shape[0]
    avg = logit_sum / count
    probs = F.softmax(avg, dim=-1)
    topk = torch.topk(probs, n)
    tokens = [model.tokenizer.decode([tid.item()]).strip() for tid in topk.indices]
    return list(zip(tokens, topk.values.tolist()))


# ── build token lookup sets for hit-counting ──────────────────────────────────
all_pos_tokens = {model.tokenizer.decode([tid]).strip().lower() for tid in pos_ids}
all_neg_tokens = {model.tokenizer.decode([tid]).strip().lower() for tid in neg_ids}

# ── load captions ──────────────────────────────────────────────────────────────
hc_df   = load_handcrafted_eval()
captions = hc_df["text"].tolist()
logging.info("Handcrafted eval: %d captions", len(captions))

# Val captions for RMS (subsample for speed)
val_df = load_dataframe_from_json(ARTIFACT_DIR / "datasplits/val.json")
rng = np.random.default_rng(RNG_SEED)
val_idx = np.sort(rng.choice(len(val_df), size=min(N_VAL_FOR_RMS, len(val_df)), replace=False))
val_df_sub = val_df.iloc[val_idx].reset_index(drop=True)
val_captions = val_df_sub["text"].tolist()
# Extract val instruction/prefix columns if present
val_instructions = val_df_sub["prompt"].tolist() if "prompt" in val_df_sub.columns else None
val_prefixes     = val_df_sub["output_prefix"].tolist() if "output_prefix" in val_df_sub.columns else None
logging.info("Val subset for RMS: %d examples", len(val_captions))

# ── STEP 1: top-10 tokens + baseline balance for every template ───────────────
logging.info("=" * 60)
logging.info("STEP 1: Top-10 tokens at λ=0 and baseline RMS")
logging.info("=" * 60)

step1 = []
for name, inst_fn, pfx_fn, approach, desc in CANDIDATE_TEMPLATES:
    # Handcrafted prompts (top-10 token check)
    hc_prompts = build_prompts(inst_fn, pfx_fn, captions)
    top10      = top_n_tokens(hc_prompts, n=10)

    # Val subset prompts (baseline RMS — uses the SAME instruction/prefix as hc)
    val_prompts = build_prompts(inst_fn, pfx_fn, val_captions)
    pos_p, neg_p, bias_arr = compute_bias_batch(val_prompts)
    baseline_rms  = RMS(bias_arr)
    mean_pos      = float(pos_p.mean())
    mean_neg      = float(neg_p.mean())

    spatial_hits     = [(t, p) for t, p in top10 if t.lower() in all_pos_tokens]
    desc_hits        = [(t, p) for t, p in top10 if t.lower() in all_neg_tokens]

    # Balance score: fraction of tracked token mass going to minority class
    # (higher = more balanced; 1.0 = perfectly balanced)
    minority_frac = min(mean_pos, mean_neg) / (mean_pos + mean_neg + 1e-10)

    logging.info(
        "%-22s  rms=%.4f  pos=%.4f  neg=%.4f  spatial_hits=%d  desc_hits=%d  minority=%.3f",
        name, baseline_rms, mean_pos, mean_neg,
        len(spatial_hits), len(desc_hits), minority_frac
    )
    logging.info("  top-5: %s", ", ".join(f"'{t}'({p:.3f})" for t, p in top10[:5]))

    step1.append({
        "name": name, "approach": approach, "description": desc,
        "output_prefix": pfx_fn(captions[0]) if not name.startswith("A2_") else pfx_fn("[caption]"),
        "baseline_rms": baseline_rms,
        "mean_pos_prob": mean_pos,
        "mean_neg_prob": mean_neg,
        "balance_minority_frac": minority_frac,
        "top10_tokens": top10,
        "spatial_in_top10": spatial_hits,
        "descriptive_in_top10": desc_hits,
        "n_spatial_hits": len(spatial_hits),
        "n_desc_hits": len(desc_hits),
        "has_both_in_top10": len(spatial_hits) > 0 and len(desc_hits) > 0,
    })

# Rank: prefer templates that have BOTH classes in top-10, then by lowest RMS
step1_sorted = sorted(
    step1,
    key=lambda x: (
        -(x["n_spatial_hits"] + x["n_desc_hits"]),   # more hits = better
        x["baseline_rms"],                             # lower rms = better
    )
)

logging.info("\nRanking (hits first, then balance):")
for i, r in enumerate(step1_sorted):
    flag = "  ← TOP 5" if i < TOP_N_FOR_SWEEP else ""
    logging.info(
        "  %2d. %-22s  rms=%.4f  spatial=%d  desc=%d  both=%s%s",
        i+1, r["name"], r["baseline_rms"],
        r["n_spatial_hits"], r["n_desc_hits"],
        r["has_both_in_top10"], flag
    )

# ── STEP 2: lambda sweep for top-5 ──────────────────────────────────────────
logging.info("=" * 60)
logging.info("STEP 2: Lambda sweep for top-%d templates", TOP_N_FOR_SWEEP)
logging.info("=" * 60)

top5_names = {r["name"] for r in step1_sorted[:TOP_N_FOR_SWEEP]}
step2 = []

for entry in step1_sorted[:TOP_N_FOR_SWEEP]:
    name    = entry["name"]
    tpl     = next(t for t in CANDIDATE_TEMPLATES if t[0] == name)
    inst_fn = tpl[1]; pfx_fn = tpl[2]

    val_prompts      = build_prompts(inst_fn, pfx_fn, val_captions)
    baseline_rms_val = entry["baseline_rms"]   # already computed on val subset

    sweep_rows = []
    for coeff in SWEEP_COEFFS:
        ivfunc = get_intervention_func(steering_vec, method=METHOD, coeff=coeff, offset=offset)
        pos_p, neg_p, bias_arr = compute_bias_batch(val_prompts, layer=BEST_LAYER, ivfunc=ivfunc)
        rms = RMS(bias_arr)
        reduction = (baseline_rms_val - rms) / baseline_rms_val * 100 if baseline_rms_val > 0 else 0.0
        sweep_rows.append({
            "coeff": coeff, "rms": rms, "reduction_pct": reduction,
            "mean_pos_prob": float(pos_p.mean()), "mean_neg_prob": float(neg_p.mean()),
        })
        logging.info("  %-22s coeff=%+d  rms=%.4f  reduction=%.1f%%",
                     name, coeff, rms, reduction)

    best = min(sweep_rows, key=lambda x: x["rms"])
    step2.append({
        "name": name,
        "approach": entry["approach"],
        "description": entry["description"],
        "output_prefix": entry["output_prefix"],
        "baseline_rms": baseline_rms_val,
        "coeff_sweep": sweep_rows,
        "best_coeff": best["coeff"],
        "best_rms": best["rms"],
        "best_reduction_pct": best["reduction_pct"],
    })
    logging.info("  %-22s BEST: coeff=%+d  rms=%.4f  reduction=%.1f%%",
                 name, best["coeff"], best["rms"], best["reduction_pct"])

# ── STEP 3: qualitative generation for top-5 at [-100, 0, +100] ──────────────
logging.info("=" * 60)
logging.info("STEP 3: Qualitative generation (greedy + beam)")
logging.info("=" * 60)

QUAL_COEFFS = [-100, 0, 100]
N_QUAL      = min(5, len(captions))   # show first 5 captions
step3_greedy = []
step3_beam   = []

for entry in step1_sorted[:TOP_N_FOR_SWEEP]:
    name    = entry["name"]
    tpl     = next(t for t in CANDIDATE_TEMPLATES if t[0] == name)
    inst_fn = tpl[1]; pfx_fn = tpl[2]
    logging.info("Generating for: %s", name)

    hc_prompts = build_prompts(inst_fn, pfx_fn, captions[:N_QUAL])

    for decoding, store_list in [("greedy", step3_greedy), ("beam", step3_beam)]:
        beam_kwargs = {"num_beams": 4} if decoding == "beam" else {}
        by_coeff = {}
        for coeff in QUAL_COEFFS:
            ivfunc = get_intervention_func(steering_vec, method=METHOD, coeff=coeff, offset=offset)
            gens = []
            for i, prompt in enumerate(hc_prompts):
                out = model.generate(
                    [prompt], layer=BEST_LAYER, intervene_func=ivfunc,
                    max_new_tokens=MAX_NEW_TOKENS, **beam_kwargs
                )
                text = out[0].strip() if out else ""
                gens.append({"caption": captions[i][:100], "generated": text})
                logging.info("  [%s/%s] coeff=%+d → '%s'",
                             name, decoding, coeff, text[:60])
            by_coeff[str(coeff)] = gens
        store_list.append({
            "name": name, "approach": entry["approach"],
            "output_prefix": entry["output_prefix"],
            "examples_by_coeff": by_coeff,
        })

# ── STEP 4: compare against original image_shows ──────────────────────────────
logging.info("=" * 60)
logging.info("STEP 4: Comparison summary vs. original image_shows baseline")
logging.info("=" * 60)
image_shows_step2 = next((r for r in step2 if r["name"] == "A_image_shows"), None)
if image_shows_step2:
    logging.info("Baseline (A_image_shows): rms=%.4f → best=%.4f (%.1f%% @ coeff=%+d)",
                 image_shows_step2["baseline_rms"], image_shows_step2["best_rms"],
                 image_shows_step2["best_reduction_pct"], image_shows_step2["best_coeff"])

# ── save results ──────────────────────────────────────────────────────────────

def _ser(obj):
    if isinstance(obj, (np.float32, np.float64)): return float(obj)
    if isinstance(obj, (np.int32, np.int64)):     return int(obj)
    if isinstance(obj, np.ndarray):               return obj.tolist()
    return str(obj)

results = {
    "config": {
        "model": cfg.model_name, "layer": BEST_LAYER,
        "method": METHOD, "constrained_softmax": CONSTRAINED_SOFTMAX,
        "sweep_coeffs": SWEEP_COEFFS, "n_val_for_rms": N_VAL_FOR_RMS,
        "n_qual_captions": N_QUAL, "top_n_for_sweep": TOP_N_FOR_SWEEP,
    },
    "step1_all_templates": step1_sorted,
    "step2_top5_sweep": step2,
    "step3_greedy": step3_greedy,
    "step3_beam": step3_beam,
}
with open(OUT_DIR / "results.json", "w") as f:
    json.dump(results, f, default=_ser, indent=2)
logging.info("Saved results.json")

# ── write SUMMARY.md ──────────────────────────────────────────────────────────

def write_summary():
    lines = [
        "# Qwen-1.8B Natural Template Search — v2",
        "",
        "Model: `Qwen/Qwen-1_8B-chat`  Layer: 11  Method: default (orthogonal projection + constant)",
        "",
        "## All Templates: Ranking",
        "",
        "**Primary sort**: templates with BOTH spatial AND descriptive tokens in top-10 first.",
        "**Secondary sort**: lower baseline RMS (more balanced).",
        "",
        "| Rank | Approach | Template | Prefix | rms@λ=0 | mean_sp | mean_desc | sp_top10 | desc_top10 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(step1_sorted):
        pfx = r["output_prefix"][:35] + ("…" if len(r["output_prefix"]) > 35 else "")
        lines.append(
            f"| {i+1} | {r['approach']} | `{r['name']}` | \"{pfx}\" | "
            f"{r['baseline_rms']:.4f} | {r['mean_pos_prob']:.4f} | {r['mean_neg_prob']:.4f} | "
            f"{r['n_spatial_hits']} | {r['n_desc_hits']} |"
        )

    lines += [
        "",
        "## Top-10 Tokens at λ=0",
        "",
    ]
    for r in step1_sorted:
        lines.append(f"### `{r['name']}` ({r['approach']}) — {r['description']}")
        tok_str = " | ".join(f"`{t}` ({p:.3f})" for t, p in r["top10_tokens"])
        lines.append(f"Top-10: {tok_str}")
        if r["spatial_in_top10"]:
            sp = ", ".join(f"`{t}` ({p:.3f})" for t, p in r["spatial_in_top10"])
            lines.append(f"  🗺 **Spatial hits**: {sp}")
        if r["descriptive_in_top10"]:
            ds = ", ".join(f"`{t}` ({p:.3f})" for t, p in r["descriptive_in_top10"])
            lines.append(f"  🎨 **Descriptive hits**: {ds}")
        lines.append("")

    lines += [
        "## RMS Bias Reduction — Top-5 Templates",
        "",
        "Formula: `RMS = sqrt(mean((spatial_prob − descriptive_prob)²))`",
        "Constrained softmax · " + f"{N_VAL_FOR_RMS} val examples · Layer {BEST_LAYER} · method=default",
        "",
        "| Rank | Template | Prefix | rms@λ=0 | best_rms | best_coeff | reduction% |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(step2):
        pfx = r["output_prefix"][:35] + ("…" if len(r["output_prefix"]) > 35 else "")
        lines.append(
            f"| {i+1} | `{r['name']}` | \"{pfx}\" | "
            f"{r['baseline_rms']:.4f} | {r['best_rms']:.4f} | "
            f"{r['best_coeff']:+d} | {r['best_reduction_pct']:.1f}% |"
        )
    lines.append("")

    for r in step2:
        lines.append(f"### `{r['name']}` sweep")
        lines.append("| coeff | rms | reduction% |")
        lines.append("|---|---|---|")
        for row in r["coeff_sweep"]:
            lines.append(f"| {row['coeff']:+d} | {row['rms']:.4f} | {row['reduction_pct']:.1f}% |")
        lines.append("")

    if image_shows_step2:
        lines += [
            "## Comparison vs. Original `image_shows`",
            "",
            f"| | baseline_rms | best_rms | reduction% | best_coeff |",
            f"|---|---|---|---|---|",
            f"| `image_shows` (original) | {image_shows_step2['baseline_rms']:.4f} | "
            f"{image_shows_step2['best_rms']:.4f} | {image_shows_step2['best_reduction_pct']:.1f}% | "
            f"{image_shows_step2['best_coeff']:+d} |",
        ]
        for r in step2:
            if r["name"] != "A_image_shows":
                lines.append(
                    f"| `{r['name']}` | {r['baseline_rms']:.4f} | "
                    f"{r['best_rms']:.4f} | {r['best_reduction_pct']:.1f}% | "
                    f"{r['best_coeff']:+d} |"
                )
        lines.append("")

    return "\n".join(lines)

with open(OUT_DIR / "SUMMARY.md", "w") as f:
    f.write(write_summary())
logging.info("Saved SUMMARY.md")

# ── write QUALITATIVE_EXAMPLES.md ─────────────────────────────────────────────

def write_qualitative():
    lines = [
        "# Qualitative Generation Examples — Qwen-1.8B",
        "",
        "Shown at λ=−100, λ=0, λ=+100. Greedy first, then beam (width=4).",
        "",
    ]
    for g_entry in step3_greedy:
        b_entry = next(e for e in step3_beam if e["name"] == g_entry["name"])
        name  = g_entry["name"]
        pfx   = g_entry["output_prefix"]
        appr  = g_entry["approach"]
        lines.append(f"## `{name}` (Approach {appr}) — prefix: \"{pfx[:60]}\"")
        lines.append("")
        for coeff_str in ["-100", "0", "100"]:
            label = {"-100": "λ=−100 (→descriptive)", "0": "λ=0 (no steering)", "100": "λ=+100 (→spatial)"}[coeff_str]
            lines.append(f"### {label}")
            lines.append("**Greedy:**")
            for ex in g_entry["examples_by_coeff"].get(coeff_str, []):
                lines.append(f"- _{ex['caption'][:80]}_")
                lines.append(f"  → `{pfx} {ex['generated']}`")
            lines.append("")
            lines.append("**Beam (width=4):**")
            for ex in b_entry["examples_by_coeff"].get(coeff_str, []):
                lines.append(f"- _{ex['caption'][:80]}_")
                lines.append(f"  → `{pfx} {ex['generated']}`")
            lines.append("")
    return "\n".join(lines)

with open(OUT_DIR / "QUALITATIVE_EXAMPLES.md", "w") as f:
    f.write(write_qualitative())
logging.info("Saved QUALITATIVE_EXAMPLES.md")

# ── write RECOMMENDATION.md ───────────────────────────────────────────────────

def write_recommendation():
    # Pick winner: best reduction% among top-5 that has has_both_in_top10
    # Tiebreak: prefer templates with real spatial+descriptive hits in top-10
    # and natural language quality
    candidates_with_both = [r for r in step2
                             if next(s for s in step1_sorted if s["name"] == r["name"])["has_both_in_top10"]]
    if candidates_with_both:
        winner_sweep = max(candidates_with_both, key=lambda x: x["best_reduction_pct"])
    else:
        # Fallback: best reduction%
        winner_sweep = max(step2, key=lambda x: x["best_reduction_pct"])

    winner_step1 = next(s for s in step1_sorted if s["name"] == winner_sweep["name"])
    baseline_ref = image_shows_step2

    lines = [
        "# Recommendation: Best Natural Template for Qwen-1.8B",
        "",
        f"## Winner: `{winner_sweep['name']}` — \"{winner_sweep['output_prefix'][:60]}\"",
        "",
        "### Why",
        "",
        f"- **Approach**: {winner_sweep['approach']} — {winner_sweep['description']}",
        f"- **baseline_rms**: {winner_sweep['baseline_rms']:.4f}",
        f"- **RMS reduction**: {winner_sweep['best_reduction_pct']:.1f}% at coeff={winner_sweep['best_coeff']:+d}",
        f"- **Spatial tokens in top-10**: {winner_step1['n_spatial_hits']}  "
        f"({'yes' if winner_step1['has_both_in_top10'] else 'no'} spatial+desc both present)",
        f"- **Descriptive tokens in top-10**: {winner_step1['n_desc_hits']}",
        "",
    ]
    if baseline_ref:
        delta = winner_sweep["best_reduction_pct"] - baseline_ref["best_reduction_pct"]
        lines += [
            "### Comparison vs. `image_shows`",
            "",
            f"| Metric | `image_shows` | `{winner_sweep['name']}` | Δ |",
            "|---|---|---|---|",
            f"| baseline_rms | {baseline_ref['baseline_rms']:.4f} | {winner_sweep['baseline_rms']:.4f} | "
            f"{winner_sweep['baseline_rms'] - baseline_ref['baseline_rms']:+.4f} |",
            f"| best_rms | {baseline_ref['best_rms']:.4f} | {winner_sweep['best_rms']:.4f} | "
            f"{winner_sweep['best_rms'] - baseline_ref['best_rms']:+.4f} |",
            f"| reduction% | {baseline_ref['best_reduction_pct']:.1f}% | {winner_sweep['best_reduction_pct']:.1f}% | {delta:+.1f}pp |",
            f"| best_coeff | {baseline_ref['best_coeff']:+d} | {winner_sweep['best_coeff']:+d} | — |",
            "",
        ]
    lines += [
        "### Lambda sweep curve",
        "",
        "| coeff | rms | reduction% |",
        "|---|---|---|",
    ]
    for row in winner_sweep["coeff_sweep"]:
        lines.append(f"| {row['coeff']:+d} | {row['rms']:.4f} | {row['reduction_pct']:.1f}% |")
    lines += [
        "",
        "### Generation examples (λ=−100, λ=0, λ=+100)",
        "",
    ]
    g_entry = next((e for e in step3_greedy if e["name"] == winner_sweep["name"]), None)
    if g_entry:
        pfx = g_entry["output_prefix"]
        for coeff_str in ["-100", "0", "100"]:
            label = {"-100": "λ=−100", "0": "λ=0 (unsteered)", "100": "λ=+100"}[coeff_str]
            lines.append(f"**{label}:**")
            for ex in g_entry["examples_by_coeff"].get(coeff_str, [])[:3]:
                lines.append(f"- _{ex['caption'][:80]}_")
                lines.append(f"  → `{pfx} {ex['generated']}`")
            lines.append("")
    lines += [
        "---",
        "",
        "## Full Rankings",
        "",
        "| Rank | Name | reduction% | rms@λ=0 | both_in_top10 |",
        "|---|---|---|---|---|",
    ]
    for i, r in enumerate(step2):
        s = next(s for s in step1_sorted if s["name"] == r["name"])
        lines.append(
            f"| {i+1} | `{r['name']}` | {r['best_reduction_pct']:.1f}% | "
            f"{r['baseline_rms']:.4f} | {'✓' if s['has_both_in_top10'] else '✗'} |"
        )
    lines.append("")
    return "\n".join(lines)

with open(OUT_DIR / "RECOMMENDATION.md", "w") as f:
    f.write(write_recommendation())
logging.info("Saved RECOMMENDATION.md")

# ── print final summary ────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("QWEN-1.8B TEMPLATE SEARCH RESULTS")
print("=" * 65)
print("\nTop-5 by (hits+balance) with sweep results:")
for r in step2:
    s = next(s for s in step1_sorted if s["name"] == r["name"])
    print(f"  {r['name']:25s}  reduction={r['best_reduction_pct']:5.1f}%  "
          f"rms: {r['baseline_rms']:.3f}→{r['best_rms']:.3f} @ {r['best_coeff']:+d}  "
          f"both_in_top10={'Y' if s['has_both_in_top10'] else 'N'}")
print(f"\nOutputs: {OUT_DIR}")

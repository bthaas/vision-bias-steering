"""
Coherence Frontier Experiments — Qwen-1.8B, B_positioned template.

THREE EXPERIMENTS:
  Exp 1 — Fine-grained lambda sweep (layer 11 only): find the coherence frontier.
  Exp 2 — Layer-selective steering: same sweep for 4 layer configs.
  Exp 3 — Token-limited steering: steer only first N generated tokens.

Outputs → experiments/coherence_frontier/
  RESULTS.md, BEST_CONFIGS.md, QUALITATIVE_EXAMPLES.md

Usage:
  /opt/homebrew/bin/python3.11 experiments/coherence_frontier/run_experiments.py
"""
import json, sys, logging, re, math
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

REPO_ROOT    = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "runs_vision/Qwen-1_8B-chat"
CFG_PATH     = ARTIFACT_DIR / "config.yaml"
DATASET_DIR  = REPO_ROOT / "bias_steering/data/datasets"
OUT_DIR      = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT))

from bias_steering.config import Config
from bias_steering.steering import load_model, get_target_token_ids
from bias_steering.steering.intervention import get_intervention_func
from bias_steering.data.load_dataset import load_handcrafted_eval, load_dataframe_from_json

# ── config ────────────────────────────────────────────────────────────────────
cfg = Config.load(CFG_PATH)
BEST_LAYER = 11
METHOD     = "default"
RNG_SEED   = 42
N_VAL      = 100   # val examples for RMS
N_QUAL     = 5     # captions for generation examples
MAX_NEW_TOKENS = 20

# Fine-grained lambda sweep (covers both negative and positive directions)
FINE_LAMBDAS = [-60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60]

# Exp 3: also test these lambdas (populated after Exp 1)
# These are chosen from FINE_LAMBDAS based on best RMS while coherent
EXP3_LAMBDAS_PLACEHOLDER = [-50, -40, -30, 30, 40, 50]

logging.info("Config: model=%s  best_layer=%d", cfg.model_name, BEST_LAYER)

# ── load model ─────────────────────────────────────────────────────────────────
logging.info("Loading model …")
model = load_model(cfg.model_name)
N_LAYERS = model.n_layer
logging.info("Model has %d layers", N_LAYERS)

# ── target token IDs ──────────────────────────────────────────────────────────
target_words = json.load(open(DATASET_DIR / "target_words.json"))["vision"]
pos_ids_raw  = get_target_token_ids(model.tokenizer, target_words["spatial"])
neg_ids_raw  = get_target_token_ids(model.tokenizer, target_words["descriptive"])
overlap      = set(pos_ids_raw) & set(neg_ids_raw)
pos_ids = [t for t in pos_ids_raw if t not in overlap]
neg_ids = [t for t in neg_ids_raw if t not in overlap]
all_ids = pos_ids + neg_ids
n_pos   = len(pos_ids)
logging.info("pos_ids: %d  neg_ids: %d", len(pos_ids), len(neg_ids))

# ── steering artifacts ─────────────────────────────────────────────────────────
candidate_vectors = torch.load(ARTIFACT_DIR / "activations/candidate_vectors.pt")
steering_vec = model.set_dtype(candidate_vectors[BEST_LAYER])
neutral_acts = torch.load(ARTIFACT_DIR / "activations/neutral.pt")
offset       = model.set_dtype(neutral_acts.mean(dim=1)[BEST_LAYER])
logging.info("Steering vec shape: %s  Offset shape: %s", steering_vec.shape, offset.shape)

# ── layer configs for Exp 2 ────────────────────────────────────────────────────
LAYER_CONFIGS = {
    "single_11":    [11],
    "middle_8_14":  list(range(8,  15)),
    "early_0_6":    list(range(0,   7)),
    "late_18_23":   list(range(18, min(24, N_LAYERS))),
}

# ── B_positioned template ──────────────────────────────────────────────────────
def _std_instruction(caption):
    return f"Describe this image:\n{caption}"

TEMPLATE_INST_FN = _std_instruction
TEMPLATE_PFX     = "Positioned"

# ── data ───────────────────────────────────────────────────────────────────────
hc_df    = load_handcrafted_eval()
captions = hc_df["text"].tolist()
qual_captions = captions[:N_QUAL]

val_df  = load_dataframe_from_json(ARTIFACT_DIR / "datasplits/val.json")
rng     = np.random.default_rng(RNG_SEED)
val_idx = np.sort(rng.choice(len(val_df), size=min(N_VAL, len(val_df)), replace=False))
val_df_sub  = val_df.iloc[val_idx].reset_index(drop=True)
val_captions = val_df_sub["text"].tolist()

logging.info("Qual captions: %d  Val captions: %d", len(qual_captions), len(val_captions))

# ── helpers ────────────────────────────────────────────────────────────────────

def RMS(x):
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x ** 2)))


def build_prompts(captions_list):
    instructions = [TEMPLATE_INST_FN(c) for c in captions_list]
    prefixes     = [TEMPLATE_PFX] * len(captions_list)
    return model.apply_chat_template(instructions, output_prefix=prefixes)


def compute_bias_batch_multilayer(prompts, layer_indices, ivfunc):
    """Forward pass with multi-layer intervention → bias scores."""
    from bias_steering.data.prompt_iterator import PromptIterator
    layer_blocks = [model.block_modules[i] for i in layer_indices]
    pos_all, neg_all, bias_all = [], [], []

    for batch in PromptIterator(prompts, batch_size=16):
        inputs = model.tokenize(batch)
        with model.model.trace(inputs) as tracer:
            for lb in layer_blocks:
                acts     = lb.output[0].clone()
                new_acts = ivfunc(acts)
                lb.output = (new_acts,) + lb.output[1:]
            logits = model.model.lm_head.output.detach().to("cpu").to(torch.float64).save()

        logits_last = logits.value[:, -1, :]
        tgt   = logits_last[:, all_ids]
        probs = F.softmax(tgt, dim=-1)
        pp    = probs[:, :n_pos].sum(-1).tolist()
        np_   = probs[:, n_pos:].sum(-1).tolist()
        pos_all.extend(pp);  neg_all.extend(np_)
        bias_all.extend([p - n for p, n in zip(pp, np_)])
    return np.array(pos_all), np.array(neg_all), np.array(bias_all)


def compute_bias_batch(prompts, layer=None, ivfunc=None):
    """Single-layer wrapper for RMS computation."""
    if layer is None or ivfunc is None:
        # No intervention — plain forward pass
        return compute_bias_batch_multilayer(prompts, [], None)
    return compute_bias_batch_multilayer(prompts, [layer], ivfunc)


def compute_bias_batch_nointervene(prompts):
    """Forward pass without any intervention."""
    from bias_steering.data.prompt_iterator import PromptIterator
    pos_all, neg_all, bias_all = [], [], []
    for batch in PromptIterator(prompts, batch_size=16):
        lgs = model.get_last_position_logits(batch)
        tgt   = lgs[:, all_ids]
        probs = F.softmax(tgt, dim=-1)
        pp    = probs[:, :n_pos].sum(-1).tolist()
        np_   = probs[:, n_pos:].sum(-1).tolist()
        pos_all.extend(pp);  neg_all.extend(np_)
        bias_all.extend([p - n for p, n in zip(pp, np_)])
    return np.array(pos_all), np.array(neg_all), np.array(bias_all)


def generate_multilayer(prompts, layer_indices, ivfunc, max_new_tokens, n_steer_tokens=None):
    """Generate text with multi-layer intervention for first n_steer_tokens tokens."""
    from nnsight import LanguageModel
    if n_steer_tokens is None:
        n_steer_tokens = max_new_tokens

    layer_blocks = [model.block_modules[i] for i in layer_indices]
    results = []

    for prompt in prompts:
        inputs = model.tokenize([prompt])
        input_len = inputs.input_ids.shape[1]

        with model.model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False) as tracer:
            # --- first generated token (step 1) ---
            if n_steer_tokens >= 1 and layer_blocks:
                for lb in layer_blocks:
                    acts     = lb.output[0].clone()
                    new_acts = ivfunc(acts)
                    lb.output = (new_acts,) + lb.output[1:]

            # --- subsequent tokens ---
            for step in range(max_new_tokens - 1):
                token_number = step + 2          # token 1 already handled above
                should_steer = token_number <= n_steer_tokens
                for lb in layer_blocks:
                    next_lb = lb.next()
                    if should_steer:
                        acts     = next_lb.output[0].t[-1]
                        new_acts = ivfunc(acts)
                        next_lb.output[0].t[-1] = new_acts

            outputs = model.model.generator.output.detach().to("cpu").save()

        completion = outputs.value[0, input_len:]
        text = model.tokenizer.decode(completion, skip_special_tokens=True).strip()
        results.append(text)

    return results


def generate_single_layer(prompts, layer, ivfunc, max_new_tokens, n_steer_tokens=None):
    """Single-layer generation (thin wrapper around generate_multilayer)."""
    return generate_multilayer(prompts, [layer], ivfunc, max_new_tokens, n_steer_tokens)


def generate_unsteered(prompts, max_new_tokens):
    """Plain generation with no steering."""
    return generate_multilayer(prompts, [], lambda x: x, max_new_tokens, n_steer_tokens=0)


# ── coherence detection ────────────────────────────────────────────────────────

def coherence_score(text):
    """
    Returns: ('coherent'|'partial'|'degenerate', reason_str)

    Heuristics (all applied):
      - max_freq:   any single token >= 40% of tokens  → degenerate
      - ttr:        type-token ratio < 0.30             → degenerate
      - bigram_rep: any bigram appears >= 4 times        → degenerate
      - short:      len < 5 tokens                       → degenerate
      - partial:    max_freq >=0.25 or ttr <0.45 or bigram >=3
    """
    tokens = text.lower().split()
    if len(tokens) < 5:
        return "degenerate", "too short"

    counts    = Counter(tokens)
    max_tok   = counts.most_common(1)[0]
    max_freq  = max_tok[1] / len(tokens)
    ttr       = len(counts) / len(tokens)

    bigrams = list(zip(tokens, tokens[1:]))
    bi_cnt  = Counter(bigrams)
    max_bi  = bi_cnt.most_common(1)[0][1] if bi_cnt else 0

    issues = []
    if max_freq >= 0.40:
        issues.append(f"max_freq={max_freq:.2f} ('{max_tok[0]}')")
    if ttr < 0.30:
        issues.append(f"ttr={ttr:.2f}")
    if max_bi >= 4:
        issues.append(f"bigram_rep={max_bi}")

    if issues:
        return "degenerate", "; ".join(issues)

    partial_issues = []
    if max_freq >= 0.25:
        partial_issues.append(f"max_freq={max_freq:.2f}")
    if ttr < 0.45:
        partial_issues.append(f"ttr={ttr:.2f}")
    if max_bi >= 3:
        partial_issues.append(f"bigram_rep={max_bi}")

    if partial_issues:
        return "partial", "; ".join(partial_issues)

    return "coherent", ""


def majority_coherence(texts):
    """Given a list of generated texts, return overall coherence label."""
    labels = [coherence_score(t)[0] for t in texts]
    if all(l == "coherent" for l in labels):
        return "coherent"
    if all(l == "degenerate" for l in labels):
        return "degenerate"
    return "partial"


# ── precompute baseline ────────────────────────────────────────────────────────
logging.info("Computing baseline RMS (no intervention) …")
val_prompts = build_prompts(val_captions)
_, _, bias_base = compute_bias_batch_nointervene(val_prompts)
BASELINE_RMS = RMS(bias_base)
logging.info("Baseline RMS: %.4f", BASELINE_RMS)

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1 — Fine-grained lambda sweep, single layer 11
# ══════════════════════════════════════════════════════════════════════════════
logging.info("=" * 60)
logging.info("EXPERIMENT 1 — Fine-grained lambda sweep (layer 11)")
logging.info("=" * 60)

exp1_results = []
qual_prompts = build_prompts(qual_captions)

for coeff in tqdm(FINE_LAMBDAS, desc="Exp1 lambdas"):
    ivfunc = get_intervention_func(steering_vec, method=METHOD, coeff=coeff, offset=offset)

    # RMS
    _, _, bias_arr = compute_bias_batch_multilayer(val_prompts, [BEST_LAYER], ivfunc)
    rms       = RMS(bias_arr)
    reduction = (BASELINE_RMS - rms) / BASELINE_RMS * 100

    # Generation (greedy, 20 tokens)
    gens  = generate_single_layer(qual_prompts, BEST_LAYER, ivfunc, MAX_NEW_TOKENS)
    coh   = majority_coherence(gens)
    scores = [coherence_score(g) for g in gens]

    exp1_results.append({
        "coeff": coeff, "rms": rms, "reduction_pct": reduction,
        "coherence": coh,
        "generations": [{"caption": qual_captions[i][:80], "text": gens[i],
                          "coh": scores[i][0], "reason": scores[i][1]}
                         for i in range(len(qual_captions))],
    })
    logging.info("  coeff=%+d  rms=%.4f  reduc=%.1f%%  coh=%s",
                 coeff, rms, reduction, coh)

# Find coherence frontier: max |coeff| with coherent majority
coherent_exp1 = [r for r in exp1_results if r["coherence"] == "coherent"]
partial_exp1  = [r for r in exp1_results if r["coherence"] == "partial"]

if coherent_exp1:
    best_coherent = max(coherent_exp1, key=lambda r: r["reduction_pct"])
    frontier_coeff = best_coherent["coeff"]
    frontier_rms   = best_coherent["rms"]
    frontier_red   = best_coherent["reduction_pct"]
elif partial_exp1:
    best_coherent = max(partial_exp1, key=lambda r: r["reduction_pct"])
    frontier_coeff = best_coherent["coeff"]
    frontier_rms   = best_coherent["rms"]
    frontier_red   = best_coherent["reduction_pct"]
    logging.info("No fully coherent config found; using best partial.")
else:
    best_coherent = None
    frontier_coeff, frontier_rms, frontier_red = 0, BASELINE_RMS, 0.0

logging.info("Exp1 frontier: coeff=%+d  rms=%.4f  reduction=%.1f%%  coh=%s",
             frontier_coeff, frontier_rms, frontier_red,
             best_coherent["coherence"] if best_coherent else "none")

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — Layer-selective steering
# ══════════════════════════════════════════════════════════════════════════════
logging.info("=" * 60)
logging.info("EXPERIMENT 2 — Layer-selective steering")
logging.info("=" * 60)

exp2_results = {}

for config_name, layer_indices in LAYER_CONFIGS.items():
    logging.info("  Layer config: %s  layers=%s", config_name, layer_indices)
    config_rows = []

    for coeff in tqdm(FINE_LAMBDAS, desc=f"Exp2/{config_name}"):
        ivfunc = get_intervention_func(steering_vec, method=METHOD, coeff=coeff, offset=offset)

        # RMS
        _, _, bias_arr = compute_bias_batch_multilayer(val_prompts, layer_indices, ivfunc)
        rms       = RMS(bias_arr)
        reduction = (BASELINE_RMS - rms) / BASELINE_RMS * 100

        # Generation
        gens  = generate_multilayer(qual_prompts, layer_indices, ivfunc, MAX_NEW_TOKENS)
        coh   = majority_coherence(gens)
        scores = [coherence_score(g) for g in gens]

        config_rows.append({
            "coeff": coeff, "rms": rms, "reduction_pct": reduction,
            "coherence": coh,
            "generations": [{"caption": qual_captions[i][:80], "text": gens[i],
                              "coh": scores[i][0], "reason": scores[i][1]}
                             for i in range(len(qual_captions))],
        })
        logging.info("    %s coeff=%+d  rms=%.4f  reduc=%.1f%%  coh=%s",
                     config_name, coeff, rms, reduction, coh)

    exp2_results[config_name] = config_rows

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3 — Token-limited steering
# ══════════════════════════════════════════════════════════════════════════════
logging.info("=" * 60)
logging.info("EXPERIMENT 3 — Token-limited steering")
logging.info("=" * 60)

# Use lambdas from both directions that showed interesting behaviour in Exp1
# Pick: best coherent, best partial, and a degenerate one for comparison
exp3_base_lambdas = sorted(set(
    [r["coeff"] for r in exp1_results if r["coherence"] in ("coherent", "partial")]
    + [-50, -30, 30, 50]
))
TOKEN_LIMITS = [1, 3, 5, None]  # None = all tokens (MAX_NEW_TOKENS)

exp3_results = {}

for n_tok in TOKEN_LIMITS:
    tok_label = str(n_tok) if n_tok is not None else "all"
    logging.info("  Token limit: %s", tok_label)
    tok_rows = []

    for coeff in tqdm(exp3_base_lambdas, desc=f"Exp3/tok={tok_label}"):
        ivfunc = get_intervention_func(steering_vec, method=METHOD, coeff=coeff, offset=offset)

        # RMS is always the same as Exp1 (first-token logit, always steered)
        rms_row = next((r for r in exp1_results if r["coeff"] == coeff), None)
        rms       = rms_row["rms"]       if rms_row else float("nan")
        reduction = rms_row["reduction_pct"] if rms_row else float("nan")

        # Generation with token limit
        gens  = generate_single_layer(
            qual_prompts, BEST_LAYER, ivfunc, MAX_NEW_TOKENS,
            n_steer_tokens=n_tok if n_tok is not None else MAX_NEW_TOKENS
        )
        coh    = majority_coherence(gens)
        scores = [coherence_score(g) for g in gens]

        tok_rows.append({
            "n_steer_tokens": tok_label, "coeff": coeff,
            "rms": rms, "reduction_pct": reduction,
            "coherence": coh,
            "generations": [{"caption": qual_captions[i][:80], "text": gens[i],
                              "coh": scores[i][0], "reason": scores[i][1]}
                             for i in range(len(qual_captions))],
        })
        logging.info("    tok=%s coeff=%+d  rms=%.4f  reduc=%.1f%%  coh=%s",
                     tok_label, coeff, rms, reduction, coh)

    exp3_results[tok_label] = tok_rows

# ══════════════════════════════════════════════════════════════════════════════
# COLLECT BEST CONFIGS (maximize reduction while coherent)
# ══════════════════════════════════════════════════════════════════════════════

all_configs = []

# Exp1
for r in exp1_results:
    all_configs.append({
        "experiment": "Exp1",
        "description": f"Layer 11, λ={r['coeff']:+d}, all tokens",
        "layer_config": "single_11",
        "coeff": r["coeff"],
        "n_steer_tokens": "all",
        "rms": r["rms"],
        "reduction_pct": r["reduction_pct"],
        "coherence": r["coherence"],
        "generations": r["generations"],
    })

# Exp2
for config_name, rows in exp2_results.items():
    for r in rows:
        all_configs.append({
            "experiment": "Exp2",
            "description": f"Layers {config_name}, λ={r['coeff']:+d}, all tokens",
            "layer_config": config_name,
            "coeff": r["coeff"],
            "n_steer_tokens": "all",
            "rms": r["rms"],
            "reduction_pct": r["reduction_pct"],
            "coherence": r["coherence"],
            "generations": r["generations"],
        })

# Exp3
for tok_label, rows in exp3_results.items():
    for r in rows:
        all_configs.append({
            "experiment": "Exp3",
            "description": f"Layer 11, λ={r['coeff']:+d}, {tok_label} tokens",
            "layer_config": "single_11",
            "coeff": r["coeff"],
            "n_steer_tokens": tok_label,
            "rms": r["rms"],
            "reduction_pct": r["reduction_pct"],
            "coherence": r["coherence"],
            "generations": r["generations"],
        })

coherent_configs = [c for c in all_configs if c["coherence"] == "coherent"]
partial_configs  = [c for c in all_configs if c["coherence"] == "partial"]

top_coherent = sorted(coherent_configs, key=lambda c: -c["reduction_pct"])[:3]
if len(top_coherent) < 3:
    top_coherent += sorted(partial_configs, key=lambda c: -c["reduction_pct"])[:3-len(top_coherent)]

logging.info("Top coherent configs: %d", len(coherent_configs))
for c in top_coherent:
    logging.info("  %s: reduc=%.1f%%  rms=%.4f", c["description"], c["reduction_pct"], c["rms"])

# ══════════════════════════════════════════════════════════════════════════════
# WRITE RESULTS.md
# ══════════════════════════════════════════════════════════════════════════════

COH_EMOJI = {"coherent": "✓", "partial": "~", "degenerate": "✗"}

def coh_cell(c):
    return COH_EMOJI.get(c, "?")

lines = [
    "# Coherence Frontier Experiments — Qwen-1.8B `B_positioned`",
    "",
    f"Baseline RMS (λ=0, no steering): **{BASELINE_RMS:.4f}**",
    f"Steering vector: layer {BEST_LAYER}, method={METHOD}",
    f"Template: `Positioned` (output prefix)",
    "",
    "Coherence legend: ✓ coherent  ~ partial  ✗ degenerate",
    "",
    "---",
    "",
    "## Experiment 1 — Fine-grained lambda sweep (layer 11, all tokens)",
    "",
    "| λ | RMS | Reduction% | Coherence |",
    "|---|---|---|---|",
]
for r in exp1_results:
    lines.append(f"| {r['coeff']:+d} | {r['rms']:.4f} | {r['reduction_pct']:.1f}% | {coh_cell(r['coherence'])} |")

lines += [
    "",
    f"**Coherence frontier**: λ={frontier_coeff:+d} → RMS={frontier_rms:.4f}, reduction=**{frontier_red:.1f}%**",
    "",
    "---",
    "",
    "## Experiment 2 — Layer-selective steering",
    "",
    "Each row: one (layer_config, λ) combination.",
    "",
]

for config_name, rows in exp2_results.items():
    lines.append(f"### {config_name}  (layers {LAYER_CONFIGS[config_name]})")
    lines.append("")
    lines.append("| λ | RMS | Reduction% | Coherence |")
    lines.append("|---|---|---|---|")
    for r in rows:
        lines.append(f"| {r['coeff']:+d} | {r['rms']:.4f} | {r['reduction_pct']:.1f}% | {coh_cell(r['coherence'])} |")
    best_coh_row = max(
        (r for r in rows if r["coherence"] in ("coherent", "partial")),
        key=lambda r: r["reduction_pct"], default=None
    )
    if best_coh_row:
        lines.append(f"")
        lines.append(f"*Best coherent/partial: λ={best_coh_row['coeff']:+d}, reduction={best_coh_row['reduction_pct']:.1f}%*")
    lines.append("")

lines += [
    "---",
    "",
    "## Experiment 3 — Token-limited steering (layer 11)",
    "",
    "RMS is the same as Exp1 (first-token logit, always steered).",
    "Coherence reflects the full 20-token continuation.",
    "",
]

for tok_label, rows in exp3_results.items():
    lines.append(f"### n_steer_tokens = {tok_label}")
    lines.append("")
    lines.append("| λ | RMS | Reduction% | Coherence |")
    lines.append("|---|---|---|---|")
    for r in rows:
        lines.append(f"| {r['coeff']:+d} | {r['rms']:.4f} | {r['reduction_pct']:.1f}% | {coh_cell(r['coherence'])} |")
    best_coh_row = max(
        (r for r in rows if r["coherence"] in ("coherent", "partial")),
        key=lambda r: r["reduction_pct"], default=None
    )
    if best_coh_row:
        lines.append(f"")
        lines.append(f"*Best coherent/partial: λ={best_coh_row['coeff']:+d}, reduction={best_coh_row['reduction_pct']:.1f}%*")
    lines.append("")

(OUT_DIR / "RESULTS.md").write_text("\n".join(lines))
logging.info("Saved RESULTS.md")

# ══════════════════════════════════════════════════════════════════════════════
# WRITE BEST_CONFIGS.md
# ══════════════════════════════════════════════════════════════════════════════

lines = [
    "# Best Configurations — Coherent + Max RMS Reduction",
    "",
    "Top configurations that maximize RMS reduction while maintaining coherent text.",
    "",
    f"Baseline RMS: {BASELINE_RMS:.4f}",
    "",
]

for rank, cfg_entry in enumerate(top_coherent, 1):
    lines += [
        f"## #{rank}: {cfg_entry['description']}",
        "",
        f"- **Experiment**: {cfg_entry['experiment']}",
        f"- **Layer config**: {cfg_entry['layer_config']}  layers={LAYER_CONFIGS.get(cfg_entry['layer_config'], '?')}",
        f"- **λ**: {cfg_entry['coeff']:+d}",
        f"- **Token limit**: {cfg_entry['n_steer_tokens']}",
        f"- **RMS**: {cfg_entry['rms']:.4f}",
        f"- **Reduction**: {cfg_entry['reduction_pct']:.1f}%",
        f"- **Coherence**: {cfg_entry['coherence']}",
        "",
        "### Generation examples (greedy, 20 tokens)",
        "",
    ]
    for g in cfg_entry["generations"]:
        lines.append(f"- _{g['caption'][:80]}_")
        lines.append(f"  → `{TEMPLATE_PFX} {g['text']}`  [{g['coh']}]")
    lines.append("")

(OUT_DIR / "BEST_CONFIGS.md").write_text("\n".join(lines))
logging.info("Saved BEST_CONFIGS.md")

# ══════════════════════════════════════════════════════════════════════════════
# WRITE QUALITATIVE_EXAMPLES.md
# ══════════════════════════════════════════════════════════════════════════════

def write_gen_section(lines, title, rows, coeff_label, is_exp3=False):
    """Append a generation section for a single config to lines."""
    lines.append(f"## {title}")
    lines.append("")
    for g in rows:
        if is_exp3:
            lines.append(f"- _{g['caption'][:80]}_")
            lines.append(f"  → `{TEMPLATE_PFX} {g['text']}`  [{g['coh']}]")
        else:
            lines.append(f"- _{g['caption'][:80]}_")
            lines.append(f"  → `{TEMPLATE_PFX} {g['text']}`  [{g['coh']}]")
    lines.append("")

lines = [
    "# Qualitative Examples — Coherence Frontier Experiments",
    "",
    "Best coherent/partial configs only.",
    "",
    "---",
    "",
    "## Experiment 1 — Fine-grained sweep (layer 11)",
    "",
]

for r in exp1_results:
    if r["coherence"] in ("coherent", "partial"):
        lines.append(f"### λ={r['coeff']:+d}  RMS={r['rms']:.4f}  reduction={r['reduction_pct']:.1f}%  [{r['coherence']}]")
        lines.append("")
        for g in r["generations"]:
            lines.append(f"- _{g['caption'][:80]}_")
            lines.append(f"  → `{TEMPLATE_PFX} {g['text']}`")
        lines.append("")

lines += [
    "---",
    "",
    "## Experiment 2 — Layer-selective steering (best coherent/partial per config)",
    "",
]

for config_name, rows in exp2_results.items():
    best = max(
        (r for r in rows if r["coherence"] in ("coherent", "partial")),
        key=lambda r: r["reduction_pct"], default=None
    )
    if best:
        lines.append(f"### {config_name}  λ={best['coeff']:+d}  RMS={best['rms']:.4f}  reduction={best['reduction_pct']:.1f}%  [{best['coherence']}]")
        lines.append("")
        for g in best["generations"]:
            lines.append(f"- _{g['caption'][:80]}_")
            lines.append(f"  → `{TEMPLATE_PFX} {g['text']}`")
        lines.append("")

lines += [
    "---",
    "",
    "## Experiment 3 — Token-limited steering (best per token limit)",
    "",
]

for tok_label, rows in exp3_results.items():
    best = max(
        (r for r in rows if r["coherence"] in ("coherent", "partial")),
        key=lambda r: r["reduction_pct"], default=None
    )
    if best:
        lines.append(f"### n_steer_tokens={tok_label}  λ={best['coeff']:+d}  RMS={best['rms']:.4f}  reduction={best['reduction_pct']:.1f}%  [{best['coherence']}]")
        lines.append("")
        for g in best["generations"]:
            lines.append(f"- _{g['caption'][:80]}_")
            lines.append(f"  → `{TEMPLATE_PFX} {g['text']}`")
        lines.append("")

(OUT_DIR / "QUALITATIVE_EXAMPLES.md").write_text("\n".join(lines))
logging.info("Saved QUALITATIVE_EXAMPLES.md")

# ── save raw JSON ──────────────────────────────────────────────────────────────
def _ser(obj):
    if isinstance(obj, (np.float32, np.float64)): return float(obj)
    if isinstance(obj, (np.int32, np.int64)):     return int(obj)
    if isinstance(obj, np.ndarray):               return obj.tolist()
    raise TypeError(type(obj))

all_data = {
    "baseline_rms": BASELINE_RMS,
    "exp1": exp1_results,
    "exp2": exp2_results,
    "exp3": {k: v for k, v in exp3_results.items()},
    "best_configs": top_coherent,
}
# Remove generation data from JSON to keep it small
for r in all_data["exp1"]:
    r.pop("generations", None)
for rows in all_data["exp2"].values():
    for r in rows:
        r.pop("generations", None)
for rows in all_data["exp3"].values():
    for r in rows:
        r.pop("generations", None)

(OUT_DIR / "results.json").write_text(
    json.dumps(all_data, default=_ser, indent=2)
)
logging.info("Saved results.json")
logging.info("Done.")

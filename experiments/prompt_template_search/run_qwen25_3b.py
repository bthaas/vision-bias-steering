"""
Qwen2.5-3B-Instruct — full pipeline + template search.

Runs the complete pipeline for a fresh model:
  1. Load train/val data (reuse caption splits from Qwen-1.8B run)
  2. Recompute bias scores using Qwen2.5-3B activations
  3. Extract WMD candidate steering vectors
  4. Validate (select best layer)
  5. Run template search with the winning template from qwen1.8b_v2/

Usage:
  /opt/homebrew/bin/python3.11 experiments/prompt_template_search/run_qwen25_3b.py \
      [--winning_template TEMPLATE_NAME]

Outputs → experiments/prompt_template_search/qwen2.5-3b_v2/
"""
import argparse, json, logging, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

REPO_ROOT     = Path(__file__).resolve().parents[2]
REF_ARTIFACTS = REPO_ROOT / "runs_vision/Qwen-1_8B-chat"   # reuse data splits
DATASET_DIR   = REPO_ROOT / "bias_steering/data/datasets"
OUT_ROOT      = Path(__file__).parent / "qwen2.5-3b_v2"
RUN_DIR       = REPO_ROOT / "runs_vision/Qwen2.5-3B-Instruct"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
(RUN_DIR / "datasplits").mkdir(parents=True, exist_ok=True)
(RUN_DIR / "activations").mkdir(parents=True, exist_ok=True)
(RUN_DIR / "validation").mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT))

from bias_steering.config import Config, DataConfig
from bias_steering.steering import load_model, get_target_token_ids, extract_candidate_vectors, validate
from bias_steering.steering.intervention import get_intervention_func
from bias_steering.data.load_dataset import load_handcrafted_eval, load_dataframe_from_json

parser = argparse.ArgumentParser()
parser.add_argument("--winning_template", type=str, default=None,
                    help="Name of winning template from qwen1.8b_v2 RECOMMENDATION.md. "
                         "If not given, reads from qwen1.8b_v2/RECOMMENDATION.md automatically.")
parser.add_argument("--skip_pipeline", action="store_true",
                    help="Skip extract+validate if runs_vision/Qwen2.5-3B-Instruct already populated.")
args = parser.parse_args()

# ── config for Qwen2.5-3B ─────────────────────────────────────────────────────
MODEL_NAME   = "Qwen/Qwen2.5-3B-Instruct"
CONSTRAINED  = True
SCORE_MODE   = "prob_diff"
METHOD       = "default"
BATCH_SIZE   = 16
N_TRAIN      = 800
N_VAL_RMS    = 200
SWEEP_COEFFS = [-150, -100, -50, 0, 50, 100, 150]
MAX_TOKENS   = 20
RNG_SEED     = 4238

cfg = Config(
    model_name=MODEL_NAME,
    data_cfg=DataConfig(
        target_concept="vision",
        pos_label="spatial",
        neg_label="descriptive",
        n_train=N_TRAIN,
        n_val=1000,
        bias_threshold=0.1,
        output_prefix=True,
        weighted_sample=False,
    ),
    method="WMD",
    use_offset=True,
    constrained_softmax=CONSTRAINED,
    score_mode=SCORE_MODE,
    optimize_coeff=False,
    evaluate_top_n_layer=3,
    filter_layer_pct=0.05,
    save_dir="runs_vision",
    use_cache=False,
    batch_size=BATCH_SIZE,
    seed=RNG_SEED,
)
cfg.save()

# ── load model ─────────────────────────────────────────────────────────────────
logging.info("Loading %s …", MODEL_NAME)
model = load_model(MODEL_NAME, torch_dtype=torch.float16)

# ── target token IDs ───────────────────────────────────────────────────────────
target_words = json.load(open(DATASET_DIR / "target_words.json"))["vision"]
pos_ids_raw  = get_target_token_ids(model.tokenizer, target_words["spatial"])
neg_ids_raw  = get_target_token_ids(model.tokenizer, target_words["descriptive"])
overlap = set(pos_ids_raw) & set(neg_ids_raw)
if overlap:
    logging.info("Removing %d overlapping token IDs", len(overlap))
pos_ids  = [t for t in pos_ids_raw if t not in overlap]
neg_ids  = [t for t in neg_ids_raw if t not in overlap]
all_ids  = pos_ids + neg_ids
n_pos    = len(pos_ids)
target_token_ids = {"pos": pos_ids, "neg": neg_ids}
logging.info("pos_ids: %d  neg_ids: %d", len(pos_ids), len(neg_ids))

# ── helpers ────────────────────────────────────────────────────────────────────

def RMS(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2)))


def compute_bias_batch(prompts, layer=None, ivfunc=None):
    from bias_steering.data.prompt_iterator import PromptIterator
    pos_all, neg_all, bias_all = [], [], []
    for batch in PromptIterator(prompts, batch_size=BATCH_SIZE):
        if layer is not None and ivfunc is not None:
            logits = model.get_logits(batch, layer=layer, intervene_func=ivfunc)
            lgs = logits[:, -1, :]
        else:
            lgs = model.get_last_position_logits(batch)
        tgt  = lgs[:, all_ids]
        probs = F.softmax(tgt, dim=-1)
        pp   = probs[:, :n_pos].sum(-1).tolist()
        np_  = probs[:, n_pos:].sum(-1).tolist()
        pos_all.extend(pp);  neg_all.extend(np_)
        bias_all.extend([p - n for p, n in zip(pp, np_)])
    return np.array(pos_all), np.array(neg_all), np.array(bias_all)


def get_baseline_probs_df(df):
    """Compute bias scores for a dataframe with prompt/output_prefix columns."""
    prompts = model.apply_chat_template(
        df["prompt"].tolist(),
        output_prefix=df["output_prefix"].tolist()
    )
    pos_p, neg_p, bias = compute_bias_batch(prompts)
    df = df.copy()
    df["pos_prob"]        = pos_p
    df["neg_prob"]        = neg_p
    df["bias_prob_diff"]  = pos_p - neg_p
    df["bias"]            = pos_p - neg_p   # prob_diff is our score_mode
    return df


def top_n_tokens(prompts, n=10):
    from bias_steering.data.prompt_iterator import PromptIterator
    lsum, count = None, 0
    for batch in PromptIterator(prompts, batch_size=BATCH_SIZE):
        lgs = model.get_last_position_logits(batch)
        lsum = lgs.sum(0) if lsum is None else lsum + lgs.sum(0)
        count += lgs.shape[0]
    probs = F.softmax(lsum / count, dim=-1)
    topk  = torch.topk(probs, n)
    tokens = [model.tokenizer.decode([tid.item()]).strip() for tid in topk.indices]
    return list(zip(tokens, topk.values.tolist()))

# ── PIPELINE: extract vectors + validate ─────────────────────────────────────
if not args.skip_pipeline:
    logging.info("=" * 60)
    logging.info("PIPELINE: load data, compute bias, extract vectors, validate")
    logging.info("=" * 60)

    # Reuse caption data from Qwen-1.8B splits (same captions, different model)
    train_df_src = load_dataframe_from_json(REF_ARTIFACTS / "datasplits/train.json")
    val_df_src   = load_dataframe_from_json(REF_ARTIFACTS / "datasplits/val.json")

    # Keep only columns that are model-independent
    keep_cols = [c for c in train_df_src.columns
                 if c not in ("pos_prob", "neg_prob", "bias", "bias_prob_diff", "bias_logit_margin")]
    train_df = train_df_src[keep_cols].copy()
    val_df   = val_df_src[keep_cols].copy()

    logging.info("Computing bias scores for train split (%d examples) …", len(train_df))
    train_df = get_baseline_probs_df(train_df)

    logging.info("Computing bias scores for val split (%d examples) …", len(val_df))
    val_df   = get_baseline_probs_df(val_df)

    # Save splits
    from bias_steering.utils import save_to_json_file
    save_to_json_file(train_df.to_dict("records"), RUN_DIR / "datasplits/train.json")
    save_to_json_file(val_df.to_dict("records"),   RUN_DIR / "datasplits/val.json")
    logging.info("Saved datasplits")

    # Extract candidate steering vectors
    pos_examples = train_df[train_df.bias > cfg.data_cfg.bias_threshold]
    neg_examples = train_df[train_df.bias < -cfg.data_cfg.bias_threshold]
    neutral_examples = train_df[train_df.bias.abs() <= cfg.data_cfg.bias_threshold]
    logging.info("Train: pos=%d  neg=%d  neutral=%d", len(pos_examples), len(neg_examples), len(neutral_examples))

    rng_np = np.random.default_rng(RNG_SEED)
    if len(pos_examples) > N_TRAIN:
        idx = rng_np.choice(len(pos_examples), N_TRAIN, replace=False)
        pos_examples = pos_examples.iloc[idx]
    if len(neg_examples) > N_TRAIN:
        idx = rng_np.choice(len(neg_examples), N_TRAIN, replace=False)
        neg_examples = neg_examples.iloc[idx]

    logging.info("Extracting WMD steering vectors …")
    extract_candidate_vectors(cfg, model, pos_examples, neg_examples, neutral_examples)

    logging.info("Running validation …")
    validate(cfg, model, val_df, target_token_ids)

    logging.info("Pipeline complete. Results in %s", RUN_DIR)
else:
    logging.info("Skipping pipeline (--skip_pipeline). Loading existing artifacts.")
    val_df = load_dataframe_from_json(RUN_DIR / "datasplits/val.json")

# ── Load best layer from validation ──────────────────────────────────────────
debiased_results = json.load(open(RUN_DIR / "validation/debiased_results.json"))
best_layer       = debiased_results[0]["layer"]
best_coeff_val   = debiased_results[0].get("coeff", 0)
signal_report    = json.load(open(RUN_DIR / "validation/signal_report.json"))
baseline_rms_val = signal_report["baseline_rms"]

logging.info("Best layer: %d  (rms=%.4f  reduction=%.1f%%)",
             best_layer,
             debiased_results[0]["rms"],
             debiased_results[0].get("reduction_pct", 0))

# ── Load steering vector + offset ────────────────────────────────────────────
cand_vecs    = torch.load(RUN_DIR / "activations/candidate_vectors.pt")
steering_vec = model.set_dtype(cand_vecs[best_layer])
neutral_acts = torch.load(RUN_DIR / "activations/neutral.pt")
offset       = model.set_dtype(neutral_acts.mean(dim=1)[best_layer])
logging.info("Steering vector: layer %d  shape %s", best_layer, steering_vec.shape)

# ── Determine winning template from qwen1.8b_v2 ───────────────────────────────
WINNING_TEMPLATE_NAME = args.winning_template
if WINNING_TEMPLATE_NAME is None:
    rec_path = Path(__file__).parent / "qwen1.8b_v2/RECOMMENDATION.md"
    if rec_path.exists():
        content = rec_path.read_text()
        # Parse: "## Winner: `<name>`"
        import re
        m = re.search(r"## Winner: `([^`]+)`", content)
        if m:
            WINNING_TEMPLATE_NAME = m.group(1)
            logging.info("Auto-detected winning template from RECOMMENDATION.md: %s",
                         WINNING_TEMPLATE_NAME)
if WINNING_TEMPLATE_NAME is None:
    WINNING_TEMPLATE_NAME = "A_image_shows"
    logging.warning("Could not find winning template; using %s as fallback",
                    WINNING_TEMPLATE_NAME)

# ── Template definitions (same as Qwen-1.8B script) ──────────────────────────
def _std_instruction(caption):
    return f"Describe this image:\n{caption}"

def _continue_instruction(_caption):
    return "Continue describing this image:"

CANDIDATE_TEMPLATES = {
    "A_image_shows":    (_std_instruction,    lambda c: "The image shows"),
    "A_looking_more":   (_std_instruction,    lambda c: "Looking more closely,"),
    "A_main_subject":   (_std_instruction,    lambda c: "The main subject appears"),
    "A_notable_detail": (_std_instruction,    lambda c: "The most notable detail is"),
    "A_in_the":         (_std_instruction,    lambda c: "In the"),
    "A_subject_looks":  (_std_instruction,    lambda c: "The subject looks"),
    "A2_looking_more":  (_continue_instruction, lambda c: f"{c}. Looking more closely,"),
    "A2_main_subject":  (_continue_instruction, lambda c: f"{c}. The main subject appears"),
    "A2_in_the":        (_continue_instruction, lambda c: f"{c}. In the"),
    "B_foreground":     (_std_instruction,    lambda c: "In the foreground"),
    "B_scene_depicts":  (_std_instruction,    lambda c: "The scene depicts"),
    "B_looking_image":  (_std_instruction,    lambda c: "Looking at the image,"),
    "B_subject_is":     (_std_instruction,    lambda c: "The subject is"),
    "B_positioned":     (_std_instruction,    lambda c: "Positioned"),
    "B_visually":       (_std_instruction,    lambda c: "Visually,"),
}

def build_prompts(inst_fn, pfx_fn, captions):
    instructions = [inst_fn(c) for c in captions]
    prefixes     = [pfx_fn(c) for c in captions]
    return model.apply_chat_template(instructions, output_prefix=prefixes)

# ── load captions ─────────────────────────────────────────────────────────────
hc_df    = load_handcrafted_eval()
captions = hc_df["text"].tolist()

rng = np.random.default_rng(RNG_SEED)
val_sub_idx = np.sort(rng.choice(len(val_df), size=min(N_VAL_RMS, len(val_df)), replace=False))
val_sub = val_df.iloc[val_sub_idx].reset_index(drop=True)
val_captions = val_sub["text"].tolist()

# ── token lookup sets ─────────────────────────────────────────────────────────
all_pos_tokens = {model.tokenizer.decode([tid]).strip().lower() for tid in pos_ids}
all_neg_tokens = {model.tokenizer.decode([tid]).strip().lower() for tid in neg_ids}

# ── TEST ALL TEMPLATES: top-10 + baseline RMS ─────────────────────────────────
logging.info("=" * 60)
logging.info("TEMPLATE SCAN: top-10 tokens + baseline RMS (%d val examples)", N_VAL_RMS)
logging.info("=" * 60)

all_results = []
for name, (inst_fn, pfx_fn) in CANDIDATE_TEMPLATES.items():
    hc_prompts  = build_prompts(inst_fn, pfx_fn, captions)
    val_prompts = build_prompts(inst_fn, pfx_fn, val_captions)
    top10       = top_n_tokens(hc_prompts)
    pos_p, neg_p, bias_arr = compute_bias_batch(val_prompts)
    rms         = RMS(bias_arr)
    sp_hits     = [(t, p) for t, p in top10 if t.lower() in all_pos_tokens]
    ds_hits     = [(t, p) for t, p in top10 if t.lower() in all_neg_tokens]
    logging.info("%-22s  rms=%.4f  sp=%d  ds=%d  top5: %s",
                 name, rms, len(sp_hits), len(ds_hits),
                 ", ".join(f"'{t}'({p:.3f})" for t, p in top10[:5]))
    all_results.append({
        "name": name,
        "baseline_rms": rms,
        "mean_pos": float(pos_p.mean()),
        "mean_neg": float(neg_p.mean()),
        "top10": top10,
        "n_sp_hits": len(sp_hits),
        "n_ds_hits": len(ds_hits),
        "has_both": len(sp_hits) > 0 and len(ds_hits) > 0,
    })

all_results.sort(key=lambda x: (-(x["n_sp_hits"] + x["n_ds_hits"]), x["baseline_rms"]))

# ── FULL SWEEP on winning template ────────────────────────────────────────────
logging.info("=" * 60)
logging.info("SWEEP: %s (winning from Qwen-1.8B)", WINNING_TEMPLATE_NAME)
logging.info("=" * 60)

if WINNING_TEMPLATE_NAME not in CANDIDATE_TEMPLATES:
    logging.warning("Winning template '%s' not found; falling back to A_image_shows",
                    WINNING_TEMPLATE_NAME)
    WINNING_TEMPLATE_NAME = "A_image_shows"

win_inst_fn, win_pfx_fn = CANDIDATE_TEMPLATES[WINNING_TEMPLATE_NAME]
win_val_prompts          = build_prompts(win_inst_fn, win_pfx_fn, val_captions)
_, _, win_baseline_bias  = compute_bias_batch(win_val_prompts)
win_baseline_rms         = RMS(win_baseline_bias)

sweep_rows = []
for coeff in SWEEP_COEFFS:
    ivfunc = get_intervention_func(steering_vec, method=METHOD, coeff=coeff, offset=offset)
    _, _, bias_arr = compute_bias_batch(win_val_prompts, layer=best_layer, ivfunc=ivfunc)
    rms       = RMS(bias_arr)
    reduction = (win_baseline_rms - rms) / win_baseline_rms * 100
    sweep_rows.append({"coeff": coeff, "rms": rms, "reduction_pct": reduction})
    logging.info("  coeff=%+d  rms=%.4f  reduction=%.1f%%", coeff, rms, reduction)

best_sweep = min(sweep_rows, key=lambda x: x["rms"])

# ── qualitative generation ────────────────────────────────────────────────────
logging.info("=" * 60)
logging.info("QUALITATIVE: generation at [-100, 0, +100]")
logging.info("=" * 60)

QUAL_COEFFS = [-100, 0, 100]
N_QUAL = min(5, len(captions))
win_hc_prompts = build_prompts(win_inst_fn, win_pfx_fn, captions[:N_QUAL])

qual_greedy, qual_beam = {}, {}
for coeff in QUAL_COEFFS:
    ivfunc = get_intervention_func(steering_vec, method=METHOD, coeff=coeff, offset=offset)
    gens_g, gens_b = [], []
    for i, prompt in enumerate(win_hc_prompts):
        out_g = model.generate([prompt], layer=best_layer, intervene_func=ivfunc,
                               max_new_tokens=MAX_TOKENS)
        out_b = model.generate([prompt], layer=best_layer, intervene_func=ivfunc,
                               max_new_tokens=MAX_TOKENS, num_beams=4)
        tg = out_g[0].strip() if out_g else ""
        tb = out_b[0].strip() if out_b else ""
        gens_g.append({"caption": captions[i][:100], "generated": tg})
        gens_b.append({"caption": captions[i][:100], "generated": tb})
        logging.info("  coeff=%+d greedy → '%s'", coeff, tg[:70])
    qual_greedy[str(coeff)] = gens_g
    qual_beam[str(coeff)]   = gens_b

# ── save ──────────────────────────────────────────────────────────────────────

def _ser(o):
    if isinstance(o, (np.float32, np.float64)): return float(o)
    if isinstance(o, (np.int32, np.int64)):     return int(o)
    if isinstance(o, np.ndarray):               return o.tolist()
    return str(o)

results = {
    "config": {
        "model": MODEL_NAME,
        "best_layer": best_layer,
        "method": METHOD,
        "constrained_softmax": CONSTRAINED,
        "sweep_coeffs": SWEEP_COEFFS,
        "n_val_for_rms": N_VAL_RMS,
        "winning_template": WINNING_TEMPLATE_NAME,
    },
    "pipeline_summary": {
        "baseline_rms": baseline_rms_val,
        "best_layer": best_layer,
        "best_coeff_from_validation": best_coeff_val,
        "best_rms_from_validation": debiased_results[0]["rms"],
        "reduction_pct_from_validation": debiased_results[0].get("reduction_pct", None),
    },
    "template_scan": all_results,
    "winning_template_sweep": {
        "name": WINNING_TEMPLATE_NAME,
        "baseline_rms": win_baseline_rms,
        "coeff_sweep": sweep_rows,
        "best_coeff": best_sweep["coeff"],
        "best_rms": best_sweep["rms"],
        "best_reduction_pct": best_sweep["reduction_pct"],
    },
    "qualitative_greedy": qual_greedy,
    "qualitative_beam": qual_beam,
}

with open(OUT_ROOT / "results.json", "w") as f:
    json.dump(results, f, default=_ser, indent=2)
logging.info("Saved results.json")

# ── SUMMARY.md ────────────────────────────────────────────────────────────────
win_pfx_display = win_pfx_fn(captions[0]) if not WINNING_TEMPLATE_NAME.startswith("A2_") else win_pfx_fn("[caption]")

summary_lines = [
    f"# Qwen2.5-3B-Instruct Template Search",
    f"",
    f"Model: `{MODEL_NAME}`  Best layer: {best_layer}  Method: {METHOD}",
    f"",
    f"## Pipeline Summary",
    f"",
    f"| Metric | Value |",
    f"|---|---|",
    f"| baseline_rms (primary val) | {baseline_rms_val:.4f} |",
    f"| best_layer | {best_layer} |",
    f"| best_rms (from validation) | {debiased_results[0]['rms']:.4f} |",
    f"| reduction% (from validation) | {debiased_results[0].get('reduction_pct', 'N/A')} |",
    f"",
    f"## Template Scan (top-10 tokens at λ=0)",
    f"",
    f"| Rank | Template | rms@λ=0 | sp_hits | desc_hits |",
    f"|---|---|---|---|---|",
]
for i, r in enumerate(all_results):
    summary_lines.append(
        f"| {i+1} | `{r['name']}` | {r['baseline_rms']:.4f} | "
        f"{r['n_sp_hits']} | {r['n_ds_hits']} |"
    )
summary_lines += [
    "",
    f"## Winning Template: `{WINNING_TEMPLATE_NAME}` — \"{win_pfx_display[:60]}\"",
    "",
    f"| coeff | rms | reduction% |",
    "|---|---|---|",
]
for row in sweep_rows:
    summary_lines.append(f"| {row['coeff']:+d} | {row['rms']:.4f} | {row['reduction_pct']:.1f}% |")
summary_lines += [
    "",
    f"**Best**: coeff={best_sweep['coeff']:+d}  rms={best_sweep['rms']:.4f}  "
    f"reduction={best_sweep['reduction_pct']:.1f}%",
    "",
    "## Generation Examples",
    "",
]
for coeff_str in ["-100", "0", "100"]:
    lbl = {"-100": "λ=−100", "0": "λ=0 (unsteered)", "100": "λ=+100"}[coeff_str]
    summary_lines.append(f"### {lbl}")
    summary_lines.append("**Greedy:**")
    for ex in qual_greedy.get(coeff_str, [])[:3]:
        summary_lines.append(f"- _{ex['caption'][:80]}_")
        summary_lines.append(f"  → `{win_pfx_display} {ex['generated']}`")
    summary_lines.append("")
    summary_lines.append("**Beam (4):**")
    for ex in qual_beam.get(coeff_str, [])[:3]:
        summary_lines.append(f"- _{ex['caption'][:80]}_")
        summary_lines.append(f"  → `{win_pfx_display} {ex['generated']}`")
    summary_lines.append("")

with open(OUT_ROOT / "SUMMARY.md", "w") as f:
    f.write("\n".join(summary_lines))
logging.info("Saved SUMMARY.md")

print("\n" + "=" * 65)
print(f"QWEN2.5-3B RESULTS — winning template: {WINNING_TEMPLATE_NAME}")
print("=" * 65)
print(f"  Pipeline baseline_rms: {baseline_rms_val:.4f}")
print(f"  Template '{WINNING_TEMPLATE_NAME}':")
print(f"    rms@λ=0:   {win_baseline_rms:.4f}")
print(f"    best_rms:  {best_sweep['rms']:.4f} @ coeff={best_sweep['coeff']:+d}")
print(f"    reduction: {best_sweep['reduction_pct']:.1f}%")
print(f"\nOutputs: {OUT_ROOT}")

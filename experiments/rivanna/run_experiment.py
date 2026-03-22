"""
Vision-bias steering experiment for Rivanna HPC.

Full pipeline for a single model + template:
  1. Load model
  2. Compute bias scores on caption data (diverse templates from train.json)
  3. Extract WMD steering vectors (all layers)
  4. Select best layer via projection-RMSE (diverse-template val prompts)
  5. Fine-grained lambda sweep on best layer (B_positioned, paper metric)
  6. Token-limited steering: 1-token & full-token on coherence frontier (Exp3)
  7. Save RESULTS.md + results.json

Supported models:
  Qwen/Qwen2.5-3B-Instruct
  Qwen/Qwen2.5-7B-Instruct
  Qwen/Qwen2.5-14B-Instruct

Usage:
  python run_experiment.py \\
      --model Qwen/Qwen2.5-3B-Instruct \\
      --template B_positioned \\
      --output-dir results/qwen25_3b \\
      --data-dir data
"""
import argparse
import json
import logging
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

torch.set_grad_enabled(False)


# ── argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Vision bias steering — full pipeline")
    p.add_argument("--model", required=True,
                   help="HuggingFace model name, e.g. Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--template", default="B_positioned",
                   choices=["B_positioned", "image_shows", "scene_is", "in_scene_the"],
                   help="Prompt template to use")
    p.add_argument("--output-dir", default=None,
                   help="Directory for results (default: results/<model_alias>)")
    p.add_argument("--data-dir", default=str(Path(__file__).parent / "data"),
                   help="Directory containing data files")
    p.add_argument("--n-train", type=int, default=800,
                   help="Training examples per class for WMD extraction")
    p.add_argument("--n-val", type=int, default=200,
                   help="Validation examples for RMS evaluation")
    p.add_argument("--n-qual", type=int, default=5,
                   help="Qualitative generation examples")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=20)
    p.add_argument("--bias-threshold", type=float, default=0.05,
                   help="Min |bias| to include example in WMD extraction")
    p.add_argument("--top-n-layers", type=int, default=5,
                   help="Evaluate this many top layers by extraction score")
    p.add_argument("--fine-lambdas", default="-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60",
                   help="Comma-separated lambda values for fine-grained sweep")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-extract", action="store_true",
                   help="Skip vector extraction if artifacts already exist")
    return p.parse_args()


# ── template registry ──────────────────────────────────────────────────────────

TEMPLATES = {
    "B_positioned": {
        "instruction_fn": lambda caption: f"Describe this image:\n{caption}",
        "output_prefix": "Positioned",
    },
    "image_shows": {
        "instruction_fn": lambda caption: f"Describe this image:\n{caption}",
        "output_prefix": "The image shows",
    },
    "scene_is": {
        "instruction_fn": lambda caption: f"Continue describing this scene:\n{caption}",
        "output_prefix": "The scene is",
    },
    "in_scene_the": {
        "instruction_fn": lambda caption: f"Describe this image:\n{caption}",
        "output_prefix": "In this scene, the",
    },
}


# ── data helpers ───────────────────────────────────────────────────────────────

def load_caption_df(path: Path) -> pd.DataFrame:
    """Load caption data from a JSON array or JSON Lines file."""
    with open(path) as f:
        try:
            rows = json.load(f)           # regular JSON array
        except json.JSONDecodeError:
            f.seek(0)
            rows = [json.loads(line) for line in f if line.strip()]
    df = pd.DataFrame(rows)
    if "text" not in df.columns and "caption" in df.columns:
        df = df.rename(columns={"caption": "text"})
    return df


def load_handcrafted(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return pd.DataFrame(data)
    return pd.DataFrame(data.get("examples", data))


def build_prompts(captions, template, model):
    instructions = [template["instruction_fn"](c) for c in captions]
    prefixes = [template["output_prefix"]] * len(captions)
    return model.apply_chat_template(instructions, output_prefix=prefixes)


# ── RMS / bias helpers ─────────────────────────────────────────────────────────

def RMS(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2)))


def constrained_probs(logits_last, all_ids, n_pos):
    """Constrained softmax over target tokens only."""
    tgt = logits_last[:, all_ids]
    probs = F.softmax(tgt, dim=-1)
    pp = probs[:, :n_pos].sum(-1)
    np_ = probs[:, n_pos:].sum(-1)
    return pp, np_


def compute_bias_nointervene(model, prompts, all_ids, n_pos, batch_size):
    from bias_steering.data.prompt_iterator import PromptIterator
    bias_all = []
    for batch in PromptIterator(prompts, batch_size=batch_size):
        lgs = model.get_last_position_logits(batch)
        pp, np_ = constrained_probs(lgs, all_ids, n_pos)
        bias_all.extend((pp - np_).tolist())
    return np.array(bias_all)


def compute_bias_intervened(model, prompts, layer_indices, ivfunc, all_ids, n_pos, batch_size):
    """Forward pass with multi-layer intervention → bias scores."""
    from bias_steering.data.prompt_iterator import PromptIterator
    layer_blocks = [model.block_modules[i] for i in layer_indices]
    bias_all = []
    for batch in PromptIterator(prompts, batch_size=batch_size):
        inputs = model.tokenize(batch)
        with model.model.trace(inputs) as tracer:
            for lb in layer_blocks:
                acts = lb.output[0].clone()
                new_acts = ivfunc(acts)
                lb.output = (new_acts,) + lb.output[1:]
            logits = model.model.lm_head.output.detach().to("cpu").to(torch.float64).save()
        lgs = logits.value[:, -1, :]
        pp, np_ = constrained_probs(lgs, all_ids, n_pos)
        bias_all.extend((pp - np_).tolist())
    return np.array(bias_all)


# ── generation ─────────────────────────────────────────────────────────────────

def generate_steered(model, prompts, layer_indices, ivfunc, max_new_tokens, n_steer_tokens=None):
    """Generate with optional multi-layer, token-limited steering."""
    if n_steer_tokens is None:
        n_steer_tokens = max_new_tokens
    layer_blocks = [model.block_modules[i] for i in layer_indices]
    results = []
    for prompt in prompts:
        inputs = model.tokenize([prompt])
        input_len = inputs.input_ids.shape[1]
        with model.model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False) as tracer:
            if n_steer_tokens >= 1 and layer_blocks:
                for lb in layer_blocks:
                    acts = lb.output[0].clone()
                    new_acts = ivfunc(acts)
                    lb.output = (new_acts,) + lb.output[1:]
            for step in range(max_new_tokens - 1):
                token_number = step + 2
                should_steer = token_number <= n_steer_tokens
                for lb in layer_blocks:
                    next_lb = lb.next()
                    if should_steer:
                        acts = next_lb.output[0].t[-1]
                        new_acts = ivfunc(acts)
                        next_lb.output[0].t[-1] = new_acts
            outputs = model.model.generator.output.detach().to("cpu").save()
        completion = outputs.value[0, input_len:]
        text = model.tokenizer.decode(completion, skip_special_tokens=True).strip()
        results.append(text)
    return results


# ── coherence ──────────────────────────────────────────────────────────────────

def coherence_score(text):
    tokens = text.lower().split()
    if len(tokens) < 5:
        return "degenerate", "too short"
    counts = Counter(tokens)
    max_tok = counts.most_common(1)[0]
    max_freq = max_tok[1] / len(tokens)
    ttr = len(counts) / len(tokens)
    bigrams = list(zip(tokens, tokens[1:]))
    bi_cnt = Counter(bigrams)
    max_bi = bi_cnt.most_common(1)[0][1] if bi_cnt else 0
    if max_freq >= 0.40:
        return "degenerate", f"max_freq={max_freq:.2f} ('{max_tok[0]}')"
    if ttr < 0.30:
        return "degenerate", f"ttr={ttr:.2f}"
    if max_bi >= 4:
        return "degenerate", f"bigram_rep={max_bi}"
    if max_freq >= 0.25 or ttr < 0.45 or max_bi >= 3:
        return "partial", f"max_freq={max_freq:.2f} ttr={ttr:.2f}"
    return "coherent", ""


def majority_coherence(texts):
    labels = [coherence_score(t)[0] for t in texts]
    if all(l == "coherent" for l in labels):
        return "coherent"
    if all(l == "degenerate" for l in labels):
        return "degenerate"
    return "partial"


# ── vector extraction ──────────────────────────────────────────────────────────

def compute_bias_scores_all(model, prompts, all_ids, n_pos, batch_size):
    """Compute bias scores for all examples (no intervention)."""
    return compute_bias_nointervene(model, prompts, all_ids, n_pos, batch_size)


def extract_wmd_vectors(model, pos_df, neg_df, neutral_acts_mean, batch_size, output_prefix_str):
    """Extract WMD steering vectors for all layers."""
    from bias_steering.steering.steering_utils import get_all_layer_activations

    logging.info("  Extracting positive activations (%d examples)…", len(pos_df))
    pos_prompts = [p for p in pos_df["prompt_formatted"].tolist()]
    pos_acts = get_all_layer_activations(model, pos_prompts, batch_size).to(torch.float64)

    logging.info("  Extracting negative activations (%d examples)…", len(neg_df))
    neg_prompts = [p for p in neg_df["prompt_formatted"].tolist()]
    neg_acts = get_all_layer_activations(model, neg_prompts, batch_size).to(torch.float64)

    pos_weights = torch.tensor(pos_df["bias"].tolist(), dtype=torch.float64)
    neg_weights = torch.tensor(neg_df["bias"].tolist(), dtype=torch.float64)

    candidate_vectors = []
    neutral_mean = neutral_acts_mean  # shape: [n_layer, hidden_size]

    for layer in range(model.n_layer):
        pos = pos_acts[layer].clone()
        neg = neg_acts[layer].clone()
        if neutral_mean is not None:
            pos -= neutral_mean[layer]
            neg -= neutral_mean[layer]
        w_pos = pos_weights / pos_weights.sum()
        w_neg = neg_weights / neg_weights.sum()
        pos_mean = (pos * w_pos.unsqueeze(-1)).sum(0)
        neg_mean = (neg * w_neg.unsqueeze(-1)).sum(0)
        vec = F.normalize(pos_mean.float(), dim=-1) - F.normalize(neg_mean.float(), dim=-1)
        candidate_vectors.append(vec)

    return candidate_vectors, pos_acts, neg_acts


def extract_neutral_activations(model, neutral_df, batch_size):
    from bias_steering.steering.steering_utils import get_all_layer_activations
    logging.info("  Extracting neutral activations (%d examples)…", len(neutral_df))
    prompts = neutral_df["prompt_formatted"].tolist()
    acts = get_all_layer_activations(model, prompts, batch_size).to(torch.float64)
    return acts


# ── layer selection ────────────────────────────────────────────────────────────

def _mismatch_rmse(projs: np.ndarray, bias_scores: np.ndarray) -> float:
    """RMS of bias scores where projection sign disagrees with bias sign.

    Scale-invariant: penalises sign mismatches only, not projection magnitude.
    Lower = better alignment. Matches local pipeline's bias_steering.utils.RMSE.
    """
    mask = np.where(np.sign(bias_scores) != np.sign(projs), 1, 0)
    return float(np.sqrt(np.mean((bias_scores * mask) ** 2)))


def select_best_layer_by_projection(model, val_prompts, candidate_vectors,
                                    val_bias_scores, offset_by_layer,
                                    batch_size, filter_last_pct=0.05):
    """Select best layer via mismatch-RMSE between scalar projections and stored bias scores.

    Why mismatch-RMSE (not standard RMSE):
      Standard RMSE is sensitive to the scale of projection values, which varies
      dramatically across layers (layer 0 embeddings have very different norms than
      mid-network residual-stream activations). Mismatch-RMSE only penalises examples
      where the projection's SIGN disagrees with the bias score's sign — it is
      scale-invariant and correctly identifies layers whose activation geometry
      encodes the spatial/descriptive distinction.

    Returns (best_layer, middle_layer, all_results):
      best_layer   — lowest mismatch-RMSE across all non-embedding layers (layer ≥ 1)
      middle_layer — lowest mismatch-RMSE within the middle third of the network
      all_results  — full per-layer list, sorted by mismatch_rmse ascending
    """
    from bias_steering.steering.steering_utils import get_all_layer_activations, scalar_projection

    logging.info("  Computing val activations for projection-based layer selection (%d examples)…",
                 len(val_prompts))
    val_acts = get_all_layer_activations(model, val_prompts, batch_size).to(torch.float64)

    n_layers = model.n_layer
    filter_last = max(1, int(n_layers * filter_last_pct))
    # Always skip layer 0 (embedding layer — large activation norms, not a steering target)
    eval_layers = [l for l in range(1, n_layers - filter_last)]

    # Middle-third range for sanity-check comparison
    mid_lo = n_layers // 3
    mid_hi = (2 * n_layers) // 3

    results = []
    for layer in eval_layers:
        vec = candidate_vectors[layer].to(torch.float64)
        acts = val_acts[layer].clone()
        off = offset_by_layer[layer]
        if off is not None:
            acts = acts - off.to(torch.float64)
        projs = scalar_projection(acts, vec).numpy()
        mm_rmse = _mismatch_rmse(projs, val_bias_scores)
        r, pval = pearsonr(projs, val_bias_scores)
        results.append({
            "layer": layer,
            "mismatch_rmse": mm_rmse,
            "corr": float(r),
            "p_val": float(pval),
            "in_middle_third": mid_lo <= layer < mid_hi,
            # Kept for reporting-table compatibility
            "rms": float("nan"),
            "reduction_pct": abs(float(r)) * 100,
        })

    results.sort(key=lambda r: (r["mismatch_rmse"], -abs(r["corr"])))

    # Log ALL layers so engineers can see the full ranking in Rivanna logs
    logging.info("  Full layer ranking (all %d layers, sorted by mismatch-RMSE):", len(results))
    for r in results:
        mid_tag = " [mid]" if r["in_middle_third"] else ""
        logging.info("    layer=%3d  mismatch_rmse=%.4f  corr=%+.3f  p=%.2e%s",
                     r["layer"], r["mismatch_rmse"], r["corr"], r["p_val"], mid_tag)

    best = results[0]
    logging.info("  Unconstrained best (skip layer 0): layer %d  mismatch_rmse=%.4f  corr=%.4f",
                 best["layer"], best["mismatch_rmse"], best["corr"])

    # Middle-third best
    middle_results = [r for r in results if r["in_middle_third"]]
    if middle_results:
        mid_best = middle_results[0]  # already sorted
        logging.info("  Middle-third best (layers %d–%d): layer %d  mismatch_rmse=%.4f  corr=%.4f",
                     mid_lo, mid_hi - 1, mid_best["layer"],
                     mid_best["mismatch_rmse"], mid_best["corr"])
    else:
        mid_best = best
        logging.warning("  No layers in middle third [%d, %d); using unconstrained best.", mid_lo, mid_hi)

    return best["layer"], mid_best["layer"], results


def select_best_layer_by_rms(model, val_prompts, candidate_vectors, offset_by_layer,
                              all_ids, n_pos, batch_size, baseline_rms,
                              filter_last_pct=0.05):
    """Fallback: find best layer by RMS reduction at coeff=0.

    NOTE: This method breaks when the val prompts are saturated (baseline ≈ 0.98).
    Only use as fallback when val.json does not have stored diverse-template bias.
    """
    from bias_steering.steering.intervention import get_intervention_func
    n_layers = model.n_layer
    filter_last = max(1, int(n_layers * filter_last_pct))
    eval_layers = list(range(n_layers - filter_last))

    results = []
    logging.info("  Evaluating %d layers for best steering vector (RMS fallback)…", len(eval_layers))
    for layer in tqdm(eval_layers, desc="Layer selection (RMS)"):
        vec = model.set_dtype(candidate_vectors[layer])
        off = model.set_dtype(offset_by_layer[layer])
        ivfunc = get_intervention_func(vec, method="default", coeff=0, offset=off)
        bias_arr = compute_bias_intervened(model, val_prompts, [layer], ivfunc,
                                           all_ids, n_pos, batch_size)
        rms = RMS(bias_arr)
        reduction = (baseline_rms - rms) / baseline_rms * 100
        results.append({"layer": layer, "rms": rms, "reduction_pct": reduction,
                        "rmse": float("nan"), "corr": float("nan")})

    results.sort(key=lambda r: -r["reduction_pct"])
    best = results[0]
    logging.info("  Best layer: %d  reduction=%.1f%%  rms=%.4f",
                 best["layer"], best["reduction_pct"], best["rms"])
    # Return three-tuple to match select_best_layer_by_projection signature
    return best["layer"], best["layer"], results


# ── lambda sweep ───────────────────────────────────────────────────────────────

def run_lambda_sweep(model, val_prompts, qual_prompts, qual_captions,
                     layer, steering_vec, offset, lambdas,
                     all_ids, n_pos, batch_size, max_new_tokens, baseline_rms):
    from bias_steering.steering.intervention import get_intervention_func
    results = []
    for coeff in tqdm(lambdas, desc=f"Lambda sweep (layer {layer})"):
        ivfunc = get_intervention_func(steering_vec, method="default", coeff=coeff, offset=offset)
        bias_arr = compute_bias_intervened(model, val_prompts, [layer], ivfunc,
                                           all_ids, n_pos, batch_size)
        rms = RMS(bias_arr)
        reduction = (baseline_rms - rms) / baseline_rms * 100
        gens = generate_steered(model, qual_prompts, [layer], ivfunc, max_new_tokens,
                                 n_steer_tokens=max_new_tokens)
        coh = majority_coherence(gens)
        scores = [coherence_score(g) for g in gens]
        results.append({
            "coeff": coeff, "rms": rms, "reduction_pct": reduction,
            "coherence": coh,
            "generations": [
                {"caption": qual_captions[i][:80], "text": gens[i],
                 "coh": scores[i][0], "reason": scores[i][1]}
                for i in range(len(qual_captions))
            ],
        })
        logging.info("  coeff=%+d  rms=%.4f  reduc=%.1f%%  coh=%s",
                     coeff, rms, reduction, coh)
    return results


def run_token_limited_sweep(model, qual_prompts, qual_captions,
                             layer, steering_vec, offset,
                             sweep_results, token_limits, max_new_tokens):
    """Token-limited sweep for interesting lambda values."""
    from bias_steering.steering.intervention import get_intervention_func
    # Pick lambdas from the coherence frontier and beyond
    interesting = sorted(set(
        [r["coeff"] for r in sweep_results if r["coherence"] in ("coherent", "partial")]
        + [-50, -30, 30, 50]
    ))
    # Build a lookup for RMS values from the full sweep
    rms_lookup = {r["coeff"]: (r["rms"], r["reduction_pct"]) for r in sweep_results}

    results = {}
    for n_tok in token_limits:
        tok_label = str(n_tok) if n_tok is not None else "all"
        n_steer = n_tok if n_tok is not None else max_new_tokens
        logging.info("  Token limit: %s", tok_label)
        tok_rows = []
        for coeff in tqdm(interesting, desc=f"1-tok sweep (tok={tok_label})"):
            ivfunc = get_intervention_func(steering_vec, method="default", coeff=coeff, offset=offset)
            rms, reduction = rms_lookup.get(coeff, (float("nan"), float("nan")))
            gens = generate_steered(model, qual_prompts, [layer], ivfunc, max_new_tokens,
                                     n_steer_tokens=n_steer)
            coh = majority_coherence(gens)
            scores = [coherence_score(g) for g in gens]
            tok_rows.append({
                "n_steer_tokens": tok_label, "coeff": coeff,
                "rms": rms, "reduction_pct": reduction, "coherence": coh,
                "generations": [
                    {"caption": qual_captions[i][:80], "text": gens[i],
                     "coh": scores[i][0]}
                    for i in range(len(qual_captions))
                ],
            })
            logging.info("    tok=%s coeff=%+d  reduc=%.1f%%  coh=%s",
                         tok_label, coeff, reduction, coh)
        results[tok_label] = tok_rows
    return results


# ── reporting ──────────────────────────────────────────────────────────────────

COH_EMOJI = {"coherent": "✓", "partial": "~", "degenerate": "✗"}


def _render_sweep_section(lines, sweep_results, tok_results, layer_label):
    """Append lambda-sweep + token-limited tables to lines for a given layer."""
    lines += [
        "",
        "---",
        "",
        f"## Fine-grained Lambda Sweep ({layer_label})",
        "",
        "| λ | RMS | Reduction% | Coherence |",
        "|---|---|---|---|",
    ]
    for r in sweep_results:
        lines.append(
            f"| {r['coeff']:+d} | {r['rms']:.4f} | {r['reduction_pct']:.1f}% "
            f"| {COH_EMOJI.get(r['coherence'], '?')} |"
        )
    coherent = [r for r in sweep_results if r["coherence"] == "coherent"]
    partial = [r for r in sweep_results if r["coherence"] == "partial"]
    if coherent:
        best = max(coherent, key=lambda r: r["reduction_pct"])
        lines.append(f"\n**Coherence frontier (full steering)**: λ={best['coeff']:+d} → {best['reduction_pct']:.1f}% reduction")
    elif partial:
        best = max(partial, key=lambda r: r["reduction_pct"])
        lines.append(f"\n**Coherence frontier (partial)**: λ={best['coeff']:+d} → {best['reduction_pct']:.1f}% reduction")

    lines += [
        "",
        "---",
        "",
        f"## Token-limited Steering ({layer_label})",
        "",
        "RMS is the first-token metric (always steered); coherence reflects full continuation.",
        "",
    ]
    for tok_label, rows in tok_results.items():
        lines.append(f"### n_steer_tokens = {tok_label}")
        lines.append("")
        lines.append("| λ | RMS | Reduction% | Coherence |")
        lines.append("|---|---|---|---|")
        for r in rows:
            lines.append(
                f"| {r['coeff']:+d} | {r['rms']:.4f} | {r['reduction_pct']:.1f}% "
                f"| {COH_EMOJI.get(r['coherence'], '?')} |"
            )
        best_tok = max(
            (r for r in rows if r["coherence"] in ("coherent", "partial")),
            key=lambda r: r["reduction_pct"], default=None
        )
        if best_tok:
            lines.append(f"\n*Best coherent: λ={best_tok['coeff']:+d}, reduction={best_tok['reduction_pct']:.1f}%*")
        lines.append("")


def build_results_md(model_name, template_name, output_prefix,
                     best_layer, baseline_rms, sweep_results, tok_results, layer_scores,
                     middle_layer=None, sweep_results_mid=None, tok_results_mid=None):
    n_layers_total = max(r["layer"] for r in layer_scores) + 1 if layer_scores else "?"
    lines = [
        f"# Bias Steering Results — {model_name}",
        "",
        f"Template: `{template_name}` (output prefix: \"{output_prefix}\")",
        f"Baseline RMS: **{baseline_rms:.4f}** (B_positioned, paper metric)",
        f"Best layer (unconstrained, skip 0): **{best_layer}**",
    ]
    if middle_layer is not None and middle_layer != best_layer:
        lines.append(f"Best layer (middle third):          **{middle_layer}**")
    lines += [
        "",
        "Coherence legend: ✓ coherent  ~ partial  ✗ degenerate",
        "",
        "---",
        "",
        "## Top 10 Layers by Mismatch-RMSE (diverse-template val, skip layer 0)",
        "",
        "| Layer | Mismatch-RMSE | Corr | Middle-third? |",
        "|---|---|---|---|",
    ]
    for r in layer_scores[:10]:
        mm = r.get("mismatch_rmse", float("nan"))
        mm_str = f"{mm:.4f}" if mm == mm else "—"
        corr_str = f"{r['corr']:.3f}" if r['corr'] == r['corr'] else "—"
        mid_tag = "✓" if r.get("in_middle_third") else ""
        lines.append(f"| {r['layer']} | {mm_str} | {corr_str} | {mid_tag} |")

    # Primary sweep (unconstrained best layer)
    _render_sweep_section(lines, sweep_results, tok_results, f"layer {best_layer} — unconstrained best")

    # Middle-third sweep (if different from unconstrained best)
    if middle_layer is not None and middle_layer != best_layer and sweep_results_mid is not None:
        _render_sweep_section(lines, sweep_results_mid, tok_results_mid or {},
                              f"layer {middle_layer} — middle-third best")

    return "\n".join(lines)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    from bias_steering.steering import load_model, get_target_token_ids

    model_alias = args.model.split("/")[-1]
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "results" / model_alias
    artifact_dir = out_dir / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    template = TEMPLATES[args.template]
    output_prefix = template["output_prefix"]
    fine_lambdas = [int(x) for x in args.fine_lambdas.split(",")]
    rng = np.random.default_rng(args.seed)

    logging.info("Model:    %s", args.model)
    logging.info("Template: %s (prefix: %s)", args.template, output_prefix)
    logging.info("Output:   %s", out_dir)
    logging.info("Data:     %s", data_dir)

    # ── load model ─────────────────────────────────────────────────────────────
    logging.info("Loading model…")
    model = load_model(args.model, torch_dtype=torch.float16)
    logging.info("Layers: %d  Hidden: %d", model.n_layer, model.hidden_size)

    # ── target token IDs ───────────────────────────────────────────────────────
    target_words = json.loads((data_dir / "target_words.json").read_text())["vision"]
    pos_ids_raw = get_target_token_ids(model.tokenizer, target_words["spatial"])
    neg_ids_raw = get_target_token_ids(model.tokenizer, target_words["descriptive"])
    overlap = set(pos_ids_raw) & set(neg_ids_raw)
    pos_ids = [t for t in pos_ids_raw if t not in overlap]
    neg_ids = [t for t in neg_ids_raw if t not in overlap]
    all_ids = pos_ids + neg_ids
    n_pos = len(pos_ids)
    logging.info("pos_ids: %d  neg_ids: %d  overlap removed: %d",
                 len(pos_ids), len(neg_ids), len(overlap))

    # ── load caption data ──────────────────────────────────────────────────────
    logging.info("Loading caption data…")
    train_df = load_caption_df(data_dir / "train.json")
    val_df = load_caption_df(data_dir / "val.json")
    hc_df = load_handcrafted(data_dir / "handcrafted_eval.json")

    # Sample val subset
    val_idx = np.sort(rng.choice(len(val_df), size=min(args.n_val, len(val_df)), replace=False))
    val_sub = val_df.iloc[val_idx].reset_index(drop=True)
    qual_captions = hc_df["text"].tolist()[:args.n_qual]
    logging.info("Train: %d  Val subset: %d  Qual: %d",
                 len(train_df), len(val_sub), len(qual_captions))

    # ── compute baseline bias scores on train data ─────────────────────────────
    artifact_path = artifact_dir / "candidate_vectors.pt"
    neutral_path = artifact_dir / "neutral.pt"

    if args.skip_extract and artifact_path.exists():
        logging.info("Skipping extraction — loading cached artifacts…")
        candidate_vectors_pt = torch.load(artifact_path)
        candidate_vectors = [candidate_vectors_pt[i] for i in range(len(candidate_vectors_pt))]
        neutral_acts_mean = torch.load(neutral_path) if neutral_path.exists() else None
        if neutral_acts_mean is None:
            logging.warning("No cached neutral activations — offset will be zero.")

    else:
        # ── Step 1: Build extraction prompts from stored diverse templates ──────
        # The local pipeline uses per-example instruction templates stored in
        # train.json's `prompt`+`output_prefix` columns, NOT the evaluation
        # template (B_positioned). Diverse templates produce clear pos/neg bias
        # differentiation even on strongly-biased models. B_positioned is reserved
        # for RMS measurement and generation only.
        if "prompt" in train_df.columns and "output_prefix" in train_df.columns:
            logging.info("Building extraction prompts from stored per-example templates…")
            extraction_prompts = model.apply_chat_template(
                train_df["prompt"].tolist(),
                output_prefix=train_df["output_prefix"].tolist(),
            )
        else:
            logging.warning(
                "No stored prompt templates in train.json — falling back to %s for extraction. "
                "This may produce skewed bias scores on strongly-biased models.",
                args.template,
            )
            extraction_prompts = build_prompts(train_df["text"].tolist(), template, model)
        train_df["prompt_formatted"] = extraction_prompts

        # ── Step 2: Compute new-model bias scores using extraction prompts ──────
        logging.info("Computing bias scores on training data…")
        train_bias = compute_bias_scores_all(
            model, extraction_prompts, all_ids, n_pos, args.batch_size
        )
        train_df["bias"] = train_bias
        logging.info("Extraction bias: mean=%.4f  std=%.4f  min=%.4f  max=%.4f",
                     train_bias.mean(), train_bias.std(), train_bias.min(), train_bias.max())
        # Diagnostic: bias by label — if gap < 0.3, the steering vector will be weak
        label_col_diag = next((c for c in ["vision_label", "label"] if c in train_df.columns), None)
        if label_col_diag is not None:
            sp_bias = train_df[train_df[label_col_diag] == "spatial"]["bias"].mean()
            desc_bias = train_df[train_df[label_col_diag] == "descriptive"]["bias"].mean()
            gap = sp_bias - desc_bias
            logging.info("Extraction bias by label:  spatial=%.4f  descriptive=%.4f  gap=%.4f",
                         sp_bias, desc_bias, gap)
            if abs(gap) < 0.3:
                logging.warning(
                    "WEAK SIGNAL: bias gap between spatial and descriptive is only %.4f "
                    "(< 0.3). The steering vector may be near-zero. "
                    "Consider adding base-model variants or checking tokenizer target words.",
                    gap,
                )

        # ── Step 3: Split pos/neg by ground-truth vision_label ──────────────────
        # train.json has `vision_label` = "spatial" / "descriptive" for every
        # example. Use it directly — bias scores must not define group membership
        # because they depend on the prompt template.
        label_col = next((c for c in ["vision_label", "label"] if c in train_df.columns), None)
        if label_col is not None:
            pos_df = train_df[train_df[label_col] == "spatial"].copy()
            neg_df = train_df[train_df[label_col] == "descriptive"].copy()
            logging.info("Label split (%s): spatial=%d  descriptive=%d",
                         label_col, len(pos_df), len(neg_df))
            if len(pos_df) == 0 or len(neg_df) == 0:
                logging.warning("Label split produced empty group — falling back to bias threshold")
                label_col = None

        if label_col is None:
            # Fallback: threshold then median split (only if no label column)
            pos_df = train_df[train_df["bias"] >= args.bias_threshold].copy()
            neg_df = train_df[train_df["bias"] <= -args.bias_threshold].copy()
            logging.info("Threshold %.2f: pos=%d  neg=%d", args.bias_threshold, len(pos_df), len(neg_df))
            if len(pos_df) < 10 or len(neg_df) < 10:
                logging.warning(
                    "Skewed distribution (%d pos / %d neg) — using median split",
                    len(pos_df), len(neg_df),
                )
                median_bias = float(np.median(train_df["bias"]))
                pos_df = train_df[train_df["bias"] > median_bias].copy()
                neg_df = train_df[train_df["bias"] <= median_bias].copy()
                logging.info("Median split (median=%.4f): pos=%d  neg=%d",
                             median_bias, len(pos_df), len(neg_df))

        # ── Step 4: Cap to n_train most extreme examples per side ───────────────
        n_extract = min(len(pos_df), len(neg_df), args.n_train)
        pos_df = pos_df.nlargest(n_extract, "bias")
        neg_df = neg_df.nsmallest(n_extract, "bias")
        logging.info("Using %d contrastive pairs for WMD extraction (pos bias mean=%.4f, neg bias mean=%.4f)",
                     n_extract, pos_df["bias"].mean(), neg_df["bias"].mean())

        # ── Step 5: Neutral examples ─────────────────────────────────────────────
        if "is_neutral" in train_df.columns and train_df["is_neutral"].any():
            neutral_candidates = train_df[train_df["is_neutral"]].copy()
        else:
            neutral_candidates = train_df[
                ~train_df.index.isin(pos_df.index) & ~train_df.index.isin(neg_df.index)
            ].copy()
        n_neutral = min(200, len(neutral_candidates))
        if n_neutral > 0:
            neutral_df = neutral_candidates.sample(n=n_neutral, random_state=args.seed).copy()
            logging.info("Extracting neutral activations (%d examples)…", n_neutral)
            neutral_acts = extract_neutral_activations(model, neutral_df, args.batch_size)
            neutral_acts_mean = neutral_acts.mean(dim=1)  # [n_layer, hidden_size]
        else:
            logging.warning("No neutral examples available — offset will be zero.")
            neutral_acts_mean = None

        # Extract WMD vectors
        logging.info("Extracting WMD vectors for all %d layers…", model.n_layer)
        candidate_vectors, pos_acts_raw, neg_acts_raw = extract_wmd_vectors(
            model, pos_df, neg_df, neutral_acts_mean, args.batch_size, output_prefix
        )

        # Save artifacts
        torch.save(torch.stack(candidate_vectors), artifact_path)
        if neutral_acts_mean is not None:
            torch.save(neutral_acts_mean, neutral_path)
        logging.info("Saved artifacts to %s", artifact_dir)

    # ── build val / qual prompts (B_positioned = paper evaluation metric) ──────
    # B_positioned is used for the lambda sweep and final RMS reporting (paper metric).
    # It is NOT used for layer selection — see below.
    logging.info("Building B_positioned val prompts (paper metric)…")
    val_prompts = build_prompts(val_sub["text"].tolist(), template, model)
    qual_prompts = build_prompts(qual_captions, template, model)

    # ── compute baseline RMS (B_positioned, paper metric) ─────────────────────
    logging.info("Computing baseline RMS (B_positioned)…")
    bias_base = compute_bias_nointervene(model, val_prompts, all_ids, n_pos, args.batch_size)
    baseline_rms = RMS(bias_base)
    logging.info("Baseline RMS (B_positioned): %.4f", baseline_rms)
    if baseline_rms > 0.95:
        logging.warning(
            "Baseline RMS %.4f is near-ceiling (>0.95). "
            "B_positioned strongly primes spatial for this model. "
            "Layer selection will use diverse-template val prompts to avoid saturation. "
            "Final reported numbers use B_positioned (paper metric).",
            baseline_rms,
        )

    # ── offset per layer = neutral mean ───────────────────────────────────────
    if not isinstance(candidate_vectors[0], torch.Tensor):
        candidate_vectors = [candidate_vectors[i] for i in range(len(candidate_vectors))]

    offset_by_layer = []
    for layer in range(model.n_layer):
        if neutral_acts_mean is not None:
            offset_by_layer.append(neutral_acts_mean[layer])
        else:
            offset_by_layer.append(torch.zeros(model.hidden_size, dtype=torch.float64))

    # ── layer selection (projection RMSE, diverse templates) ──────────────────
    # CRITICAL: Do NOT use B_positioned val prompts for layer selection.
    # When baseline ≈ 0.98, coeff=0 reduction is negligible for ALL layers,
    # making the ranking random. The local pipeline instead ranks layers by how
    # well scalar projections of activations correlate with stored bias scores
    # computed under diverse templates (which have real variance).
    logging.info("Selecting best layer via projection RMSE (diverse-template val prompts)…")

    has_diverse_val = "prompt" in val_sub.columns and "output_prefix" in val_sub.columns and "bias" in val_sub.columns
    if has_diverse_val:
        logging.info("  Building diverse-template val prompts from stored val.json columns…")
        val_div_prompts = model.apply_chat_template(
            val_sub["prompt"].tolist(),
            output_prefix=val_sub["output_prefix"].tolist(),
        )
        val_div_bias = val_sub["bias"].astype(float).to_numpy()
        if "vision_label" in val_sub.columns:
            sp_mean = val_sub[val_sub["vision_label"] == "spatial"]["bias"].astype(float).mean()
            desc_mean = val_sub[val_sub["vision_label"] == "descriptive"]["bias"].astype(float).mean()
        else:
            sp_mean = desc_mean = float("nan")
        logging.info("  Stored val bias: mean=%.4f  std=%.4f  spatial_mean≈%.4f  desc_mean≈%.4f",
                     val_div_bias.mean(), val_div_bias.std(), sp_mean, desc_mean)
        best_layer, middle_layer, layer_scores = select_best_layer_by_projection(
            model, val_div_prompts, candidate_vectors, val_div_bias,
            offset_by_layer, args.batch_size, filter_last_pct=0.05,
        )
    else:
        logging.warning(
            "val.json does not have stored diverse-template bias (missing prompt/output_prefix/bias). "
            "Falling back to coeff=0 RMS layer selection — may produce random layer on saturated models."
        )
        best_layer, middle_layer, layer_scores = select_best_layer_by_rms(
            model, val_prompts, candidate_vectors, offset_by_layer,
            all_ids, n_pos, args.batch_size, baseline_rms, filter_last_pct=0.05,
        )

    logging.info("Best layer (unconstrained): %d  |  Middle-third: %d", best_layer, middle_layer)

    steering_vec = model.set_dtype(candidate_vectors[best_layer])
    offset = model.set_dtype(offset_by_layer[best_layer])

    # ── fine-grained lambda sweep (unconstrained best layer) ──────────────────
    logging.info("Running fine-grained lambda sweep (layer %d)…", best_layer)
    sweep_results = run_lambda_sweep(
        model, val_prompts, qual_prompts, qual_captions,
        best_layer, steering_vec, offset, fine_lambdas,
        all_ids, n_pos, args.batch_size, args.max_new_tokens, baseline_rms
    )

    # ── token-limited sweep (unconstrained best layer) ────────────────────────
    logging.info("Running token-limited sweep (layer %d)…", best_layer)
    tok_results = run_token_limited_sweep(
        model, qual_prompts, qual_captions,
        best_layer, steering_vec, offset,
        sweep_results, token_limits=[1, 5, None],
        max_new_tokens=args.max_new_tokens
    )

    # ── middle-third layer sweep (if different) ────────────────────────────────
    sweep_results_mid = None
    tok_results_mid = None
    if middle_layer != best_layer:
        logging.info("Running lambda sweep for middle-third layer %d…", middle_layer)
        sv_mid = model.set_dtype(candidate_vectors[middle_layer])
        off_mid = model.set_dtype(offset_by_layer[middle_layer])
        sweep_results_mid = run_lambda_sweep(
            model, val_prompts, qual_prompts, qual_captions,
            middle_layer, sv_mid, off_mid, fine_lambdas,
            all_ids, n_pos, args.batch_size, args.max_new_tokens, baseline_rms
        )
        logging.info("Running token-limited sweep for middle-third layer %d…", middle_layer)
        tok_results_mid = run_token_limited_sweep(
            model, qual_prompts, qual_captions,
            middle_layer, sv_mid, off_mid,
            sweep_results_mid, token_limits=[1, 5, None],
            max_new_tokens=args.max_new_tokens
        )

    # ── save results ───────────────────────────────────────────────────────────
    results_md = build_results_md(
        args.model, args.template, output_prefix,
        best_layer, baseline_rms, sweep_results, tok_results, layer_scores,
        middle_layer=middle_layer,
        sweep_results_mid=sweep_results_mid,
        tok_results_mid=tok_results_mid,
    )
    (out_dir / "RESULTS.md").write_text(results_md)
    logging.info("Saved RESULTS.md")

    def _ser(obj):
        if isinstance(obj, (np.float32, np.float64, float)): return float(obj)
        if isinstance(obj, (np.int32, np.int64, int)):       return int(obj)
        if isinstance(obj, np.ndarray):                       return obj.tolist()
        raise TypeError(type(obj))

    def _summarize(sweep, tok, label):
        """Extract coherence_frontier and best_1tok from a sweep pair."""
        coherent = [r for r in sweep if r["coherence"] == "coherent"]
        frontier = None
        if coherent:
            bf = max(coherent, key=lambda r: r["reduction_pct"])
            frontier = {"coeff": bf["coeff"], "rms": bf["rms"],
                        "reduction_pct": bf["reduction_pct"], "mode": "full_steering"}
        best1 = None
        for r in tok.get("1", []):
            if r["coherence"] in ("coherent", "partial"):
                if best1 is None or r["reduction_pct"] > best1["reduction_pct"]:
                    best1 = r
        best1_out = None
        if best1:
            best1_out = {"coeff": best1["coeff"], "rms": best1["rms"],
                         "reduction_pct": best1["reduction_pct"], "mode": "1_token_steering"}
        return frontier, best1_out

    frontier_best, tok1_best = _summarize(sweep_results, tok_results, best_layer)

    output = {
        "model": args.model,
        "model_alias": model_alias,
        "n_layers": model.n_layer,
        "template": args.template,
        "output_prefix": output_prefix,
        "best_layer": best_layer,
        "middle_layer": middle_layer,
        "baseline_rms": baseline_rms,
        "layer_scores": layer_scores[:10],
        # Unconstrained-best-layer results
        "sweep_results": [
            {k: v for k, v in r.items() if k != "generations"}
            for r in sweep_results
        ],
        "tok_results": {
            tok: [
                {k: v for k, v in r.items() if k != "generations"}
                for r in rows
            ]
            for tok, rows in tok_results.items()
        },
        "coherence_frontier": frontier_best,
        "best_1tok": tok1_best,
    }

    # Middle-third layer results (if different)
    if middle_layer != best_layer and sweep_results_mid is not None:
        frontier_mid, tok1_mid = _summarize(sweep_results_mid, tok_results_mid or {}, middle_layer)
        output["middle_layer_sweep"] = [
            {k: v for k, v in r.items() if k != "generations"}
            for r in sweep_results_mid
        ]
        output["middle_layer_tok_results"] = {
            tok: [
                {k: v for k, v in r.items() if k != "generations"}
                for r in rows
            ]
            for tok, rows in (tok_results_mid or {}).items()
        }
        output["middle_coherence_frontier"] = frontier_mid
        output["middle_best_1tok"] = tok1_mid
    else:
        frontier_mid = frontier_best
        tok1_mid = tok1_best

    (out_dir / "results.json").write_text(json.dumps(output, default=_ser, indent=2))
    logging.info("Saved results.json")

    # Print summary
    def _print_layer_summary(label, layer, frontier, tok1):
        print(f"  [{label}] layer={layer}")
        if frontier:
            print(f"    Full steering:  λ={frontier['coeff']:+d}  reduction={frontier['reduction_pct']:.1f}%  (coherent)")
        else:
            print(f"    Full steering:  no coherent frontier")
        if tok1:
            print(f"    1-token:        λ={tok1['coeff']:+d}  reduction={tok1['reduction_pct']:.1f}%")
        else:
            print(f"    1-token:        no coherent result")

    print("\n" + "=" * 60)
    print(f"RESULTS: {args.model}")
    print(f"  n_layers={model.n_layer}  baseline_rms={baseline_rms:.4f}")
    _print_layer_summary("unconstrained", best_layer, frontier_best, tok1_best)
    if middle_layer != best_layer:
        _print_layer_summary("middle-third ", middle_layer, frontier_mid, tok1_mid)
    print("=" * 60)
    print(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    main()

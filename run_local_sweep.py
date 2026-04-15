#!/usr/bin/env python3
"""
run_local_sweep.py — Lambda sweep for Qwen-1.8B spatial/descriptive bias steering.

Loads the saved steering vector from runs_vision/Qwen-1_8B-chat/activations/,
sweeps lambda over LAMBDAS for every caption in bias_steering/captions.py,
and writes:

  plots/sweep_all.png                  one large figure (N captions × 4 columns)
  plots/sweep_caption_NN_[label].png   one figure per caption
  results/local_sweep/generated_text.txt  steered generations at each lambda

Usage:
    python run_local_sweep.py
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from bias_steering.captions import CAPTIONS
from bias_steering.steering import load_model, get_intervention_func
from bias_steering.steering.steering_utils import get_target_token_ids
from plotting.master_prompt_experiments import (
    teacher_forced_multi_token_curve,
    continuation_multi_token_curve_greedy,
    load_target_words,
)
from plotting.beam_selected_prompt_report import (
    teacher_forced_multi_token_curve_beam_batched,
    continuation_multi_token_curve_beam_batched,
)

# ---------------------------------------------------------------------------
# Config — edit these to tune the sweep
# ---------------------------------------------------------------------------

# FAST_MODE: uses first 5 captions, 5 lambdas, skips beam (~3-5 min total).
# Set to False for the full 14-caption / 7-lambda / 4-method run (~20 min).
FAST_MODE = True

MODEL_NAME   = "Qwen/Qwen-1_8B-chat"
ARTIFACT_DIR = ROOT / "runs_vision" / "Qwen-1_8B-chat"
LAYER        = 11

LAMBDAS      = list(range(-50, 51, 5)) if FAST_MODE else list(range(-150, 151, 5))
N_TOKENS     = 4 if FAST_MODE else 8  # tokens averaged over for multi-token curves
BEAM_WIDTH   = 4
BEAM_TOP_K   = 8
CONSTRAINED  = True    # constrained softmax over spatial+descriptive tokens only
BATCH_SIZE   = 16      # forward-pass batch size; reduce if OOM

MAX_GEN_TOKENS = 20    # generation length for text log

# Prompt templates
MULTI_TEMPLATE   = "Describe this image:\n{text}"
MULTI_PREFIX     = "The image shows"    # output prefix for 8-token columns
FILL_IN_PREFIX   = "Positioned"        # output prefix for fill-in columns (B_positioned)

# Plot colours (matching existing Plotly reports)
COLOR_SPATIAL     = "#1b9e77"
COLOR_DESCRIPTIVE = "#d95f02"

# ---------------------------------------------------------------------------


def _make_dirs():
    (ROOT / "plots").mkdir(exist_ok=True)
    (ROOT / "results" / "local_sweep").mkdir(parents=True, exist_ok=True)


def _build_prompts(model, output_prefix: str, captions=None) -> list[str]:
    caps = captions if captions is not None else CAPTIONS
    instructions = [MULTI_TEMPLATE.format(text=c["text"]) for c in caps]
    return model.apply_chat_template(instructions, output_prefix=output_prefix)


def _remove_token_overlap(pos_ids, neg_ids):
    overlap = set(pos_ids) & set(neg_ids)
    return (
        [x for x in pos_ids if x not in overlap],
        [x for x in neg_ids if x not in overlap],
    )


# ---------------------------------------------------------------------------
# Generation log
# ---------------------------------------------------------------------------

def _log_generations(model, fill_prompts, steering_vec, captions, outpath: Path):
    lines = [
        f"Qwen-1.8B-chat — Steered generations (layer {LAYER}, fill-in / B_positioned template)",
        f"Method: constant  |  {MAX_GEN_TOKENS} new tokens  |  greedy",
        "",
    ]
    for coeff in LAMBDAS:
        intervene = get_intervention_func(steering_vec, method="constant", coeff=coeff)
        texts = model.generate(
            fill_prompts,
            layer=LAYER,
            intervene_func=intervene,
            max_new_tokens=MAX_GEN_TOKENS,
        )
        lines += [f"\n{'─' * 68}", f"λ = {coeff}", "─" * 68]
        for cap, gen in zip(captions, texts):
            label = cap["label"].upper()
            lines.append(f"\n  [{label}] {cap['text'][:70]}...")
            lines.append(f"  → Positioned {gen.strip()}")
    outpath.write_text("\n".join(lines))
    print(f"  Text log saved: {outpath}")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _caption_title(cap: dict, max_chars: int = 90) -> str:
    label = cap["label"].upper()
    short = textwrap.shorten(cap["text"], width=max_chars, placeholder="…")
    return f"[{label}]  {short}"


def _draw_subplot(ax, lambdas, pos_curve, neg_curve,
                  xlabel=False, ylabel=False, title=None, legend=False):
    ax.plot(lambdas, pos_curve,
            color=COLOR_SPATIAL, linewidth=2, marker="o", markersize=4, label="spatial")
    ax.plot(lambdas, neg_curve,
            color=COLOR_DESCRIPTIVE, linewidth=2, marker="o", markersize=4, label="descriptive")
    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_ylim(0, 1)
    ax.set_xlim(min(lambdas) * 1.05, max(lambdas) * 1.05)
    ax.set_xticks(lambdas)
    ax.tick_params(axis="x", labelsize=6, rotation=45)
    ax.tick_params(axis="y", labelsize=7)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    if xlabel:
        ax.set_xlabel("λ (steering coeff)", fontsize=8)
    if ylabel:
        ax.set_ylabel("Tracked prob", fontsize=8)
    if title:
        ax.set_title(title, fontsize=9, pad=4)
    if legend:
        ax.legend(fontsize=8, loc="upper right", framealpha=0.85)


def _column_configs(results: dict) -> list[tuple[str, str]]:
    configs = []
    if "multi_greedy" in results:
        configs.append(("multi_greedy", "8-token greedy"))
    if "multi_beam" in results:
        configs.append(("multi_beam", "8-token beam"))
    if "fill_greedy" in results:
        configs.append(("fill_greedy", "fill-in greedy"))
    if "fill_beam" in results:
        configs.append(("fill_beam", "fill-in beam"))
    return configs


def _plot_caption_row(axes_row, caption_idx: int, results: dict, lambdas,
                      is_first_row: bool, is_last_row: bool):
    """Render one row for caption at caption_idx."""
    col_configs = _column_configs(results)
    for col, (key, col_label) in enumerate(col_configs):
        ax = axes_row[col]
        pos_curve = results[key]["pos"][caption_idx]
        neg_curve = results[key]["neg"][caption_idx]
        _draw_subplot(
            ax, lambdas, pos_curve, neg_curve,
            xlabel=is_last_row,
            ylabel=(col == 0),
            title=col_label if is_first_row else None,
            legend=(is_first_row and col == 0),
        )


def plot_all_captions(captions: list, results: dict, lambdas: list, outpath: Path):
    n = len(captions)
    col_configs = _column_configs(results)
    n_cols = len(col_configs)
    row_h = 3.2
    fig_width = 10 if n_cols == 2 else 20
    fig, axes = plt.subplots(n, n_cols, figsize=(fig_width, row_h * n), squeeze=False,
                             gridspec_kw={"hspace": 0.55, "wspace": 0.22})

    fig.suptitle(
        f"Qwen-1.8B-chat — Spatial vs Descriptive Token Probability\n"
        f"Layer {LAYER}  |  method=constant  |  {N_TOKENS}-token curves  |  constrained={CONSTRAINED}",
        fontsize=12, y=1.002,
    )

    for i, cap in enumerate(captions):
        _plot_caption_row(
            axes[i], i, results, lambdas,
            is_first_row=(i == 0),
            is_last_row=(i == n - 1),
        )
        # Row label above leftmost subplot
        caption_str = _caption_title(cap, max_chars=95)
        axes[i, 0].annotate(
            caption_str,
            xy=(0, 1.06), xycoords="axes fraction",
            fontsize=6.5, color="#222222",
            ha="left", va="bottom", clip_on=False,
        )

    fig.savefig(str(outpath), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Full figure saved: {outpath}")


def plot_single_caption(caption_idx: int, captions: list, results: dict, lambdas: list, outpath: Path):
    cap = captions[caption_idx]
    col_configs = _column_configs(results)
    n_cols = len(col_configs)
    fig_width = 10 if n_cols == 2 else 20
    fig, axes = plt.subplots(1, n_cols, figsize=(fig_width, 3.5),
                             gridspec_kw={"wspace": 0.22})
    if n_cols == 1:
        axes = [axes]
    fig.suptitle(_caption_title(cap, max_chars=110), fontsize=10, y=1.04)

    for col, (key, col_label) in enumerate(col_configs):
        ax = axes[col]
        pos_curve = results[key]["pos"][caption_idx]
        neg_curve = results[key]["neg"][caption_idx]
        _draw_subplot(
            ax, lambdas, pos_curve, neg_curve,
            xlabel=True, ylabel=(col == 0),
            title=col_label, legend=(col == 0),
        )

    fig.savefig(str(outpath), dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _make_dirs()
    torch.set_grad_enabled(False)

    # Fast mode: limit to first 5 captions (the coherence-frontier set)
    captions = CAPTIONS[:5] if FAST_MODE else CAPTIONS
    mode_tag = "FAST (5 captions, 5 lambdas, greedy only)" if FAST_MODE else "FULL (14 captions, 7 lambdas, 4 methods)"
    print(f"Mode: {mode_tag}")

    print(f"Loading {MODEL_NAME}…")
    model = load_model(MODEL_NAME)

    vec_path = ARTIFACT_DIR / "activations" / "candidate_vectors.pt"
    candidate_vectors = torch.load(vec_path, map_location="cpu")
    steering_vec = model.set_dtype(candidate_vectors[LAYER])
    print(f"Steering vector loaded  (layer {LAYER})")

    words = load_target_words("vision")
    pos_ids, neg_ids = _remove_token_overlap(
        get_target_token_ids(model.tokenizer, words["spatial"]),
        get_target_token_ids(model.tokenizer, words["descriptive"]),
    )
    print(f"Token IDs: {len(pos_ids)} spatial, {len(neg_ids)} descriptive")

    multi_prompts = _build_prompts(model, MULTI_PREFIX, captions)
    fill_prompts  = _build_prompts(model, FILL_IN_PREFIX, captions)
    print(f"Built {len(captions)} prompt pairs  "
          f"(multi: '{MULTI_PREFIX}', fill: '{FILL_IN_PREFIX}')")

    common = dict(
        model=model,
        layer=LAYER,
        steering_vec=steering_vec,
        coeffs=LAMBDAS,
        pos_ids=pos_ids,
        neg_ids=neg_ids,
        n_tokens=N_TOKENS,
        constrained=CONSTRAINED,
    )

    n_methods = 2 if FAST_MODE else 4
    print(f"\nRunning sweep: {len(captions)} captions × {len(LAMBDAS)} lambdas × {n_methods} methods")

    print("[1] 8-token greedy (teacher-forced, greedy reference)…")
    mg_pos, mg_neg = teacher_forced_multi_token_curve(
        prompts=multi_prompts, batch_size=BATCH_SIZE, **common)

    if not FAST_MODE:
        print("[2] 8-token beam  (teacher-forced, beam reference)…")
        mb_pos, mb_neg = teacher_forced_multi_token_curve_beam_batched(
            prompts=multi_prompts, beam_width=BEAM_WIDTH, beam_top_k=BEAM_TOP_K,
            batch_size=BATCH_SIZE, **common)

    print("[3] fill-in greedy (continuation, greedy decoding)…")
    fg_pos, fg_neg = continuation_multi_token_curve_greedy(
        prompts=fill_prompts, batch_size=BATCH_SIZE, **common)

    if not FAST_MODE:
        print("[4] fill-in beam   (continuation, beam decoding)…")
        fb_pos, fb_neg = continuation_multi_token_curve_beam_batched(
            prompts=fill_prompts, beam_width=BEAM_WIDTH, beam_top_k=BEAM_TOP_K,
            batch_size=BATCH_SIZE, **common)

    if FAST_MODE:
        results = {
            "multi_greedy": {"pos": mg_pos, "neg": mg_neg},
            "fill_greedy":  {"pos": fg_pos, "neg": fg_neg},
        }
    else:
        results = {
            "multi_greedy": {"pos": mg_pos, "neg": mg_neg},
            "multi_beam":   {"pos": mb_pos, "neg": mb_neg},
            "fill_greedy":  {"pos": fg_pos, "neg": fg_neg},
            "fill_beam":    {"pos": fb_pos, "neg": fb_neg},
        }

    # Text generation log
    print("\nGenerating text at each lambda…")
    _log_generations(
        model, fill_prompts, steering_vec, captions,
        ROOT / "results" / "local_sweep" / "generated_text.txt",
    )

    # Plots
    print("\nPlotting…")
    plot_all_captions(captions, results, LAMBDAS, ROOT / "plots" / "sweep_all.png")

    for i, cap in enumerate(captions):
        slug = f"sweep_caption_{i:02d}_{cap['label']}"
        plot_single_caption(i, captions, results, LAMBDAS, ROOT / "plots" / f"{slug}.png")
    print(f"  {len(captions)} individual caption plots saved to plots/")

    print("\nDone.")
    print("  Plots:    ./plots/")
    print("  Text log: ./results/local_sweep/generated_text.txt")


if __name__ == "__main__":
    main()

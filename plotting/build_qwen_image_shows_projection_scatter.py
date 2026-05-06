#!/usr/bin/env python3
"""Build the clean Qwen "The image shows" projection-vs-disparity diagnostic.

This script intentionally uses exactly one continuation prefix:

    Describe this image:
    {caption}

    The image shows

It reuses the local Qwen/Qwen-1_8B-chat validation split, layer-11 steering
vector, neutral activation offset, Qwen chat serialization, and default
intervention rule used by the local continuation experiments. The output is a
two-panel baseline/steered scatter plus a CSV/JSON bundle for regeneration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bias_steering.steering import get_target_token_ids
from bias_steering.steering.model import QWEN_CHAT_TEMPLATE


MODEL_NAME = "Qwen/Qwen-1_8B-chat"
MODEL_ALIAS = "Qwen-1_8B-chat"
ARTIFACT_DIR = REPO_ROOT / "runs_vision" / MODEL_ALIAS
DATASET_DIR = REPO_ROOT / "bias_steering" / "data" / "datasets"
VAL_JSON = ARTIFACT_DIR / "datasplits" / "val.json"
OUTDIR = REPO_ROOT / "paper" / "figures"
LAYER = 11
PREFIX = "The image shows"
INSTRUCTION_TEMPLATE = "Describe this image:\n{caption}"
DEFAULT_LAMBDAS = (0.0, 10.0, 20.0)
FINAL_LAMBDA = 20.0

BASELINE_COLOR = "#2F6FBB"
STEERED_COLOR = "#D65F5F"
ALT_COLOR = "#5B8E7D"
GRID_COLOR = "#D8DEE9"
TEXT_COLOR = "#20242A"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUTDIR)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lambdas", type=float, nargs="+", default=list(DEFAULT_LAMBDAS))
    parser.add_argument("--final-lambda", type=float, default=FINAL_LAMBDA)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Re-render figures from the saved CSV/JSON without loading Qwen.",
    )
    return parser.parse_args()


def select_device(force_cpu: bool) -> tuple[torch.device, torch.dtype]:
    if force_cpu:
        return torch.device("cpu"), torch.float32
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    return torch.device("cpu"), torch.float32


def load_qwen_hf(force_cpu: bool) -> tuple[AutoTokenizer, AutoModelForCausalLM, torch.device]:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        use_fast=False,
        local_files_only=True,
    )
    tokenizer.padding_side = "left"
    tokenizer.pad_token = "<|extra_0|>"
    tokenizer.pad_token_id = tokenizer.eod_id
    tokenizer.chat_template = QWEN_CHAT_TEMPLATE

    device, dtype = select_device(force_cpu)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=dtype,
        local_files_only=True,
    ).eval()
    model.to(device)
    return tokenizer, model, device


def load_target_ids(tokenizer: AutoTokenizer) -> tuple[list[int], list[int]]:
    target_words = json.loads((DATASET_DIR / "target_words.json").read_text(encoding="utf-8"))["vision"]
    pos_ids = get_target_token_ids(tokenizer, target_words["spatial"])
    neg_ids = get_target_token_ids(tokenizer, target_words["descriptive"])
    overlap = set(pos_ids) & set(neg_ids)
    pos_ids = [token_id for token_id in pos_ids if token_id not in overlap]
    neg_ids = [token_id for token_id in neg_ids if token_id not in overlap]
    if not pos_ids or not neg_ids:
        raise ValueError("Target token sets are empty after overlap removal.")
    return pos_ids, neg_ids


def apply_qwen_template(tokenizer: AutoTokenizer, captions: list[str]) -> list[str]:
    prompts: list[str] = []
    for caption in captions:
        instruction = INSTRUCTION_TEMPLATE.format(caption=caption)
        messages = [{"role": "user", "content": instruction}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if prompt[-1] not in ("\n", " "):
            prompt += " "
        prompts.append(prompt + PREFIX)
    return prompts


def prompt_for_csv(caption: str) -> str:
    return (INSTRUCTION_TEMPLATE.format(caption=caption) + "\n\n" + PREFIX).replace("\n", "\\n")


def text_for_csv(text: str) -> str:
    return str(text).replace("\n", "\\n")


def batched(items: list[str], batch_size: int) -> Iterable[tuple[int, list[str]]]:
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def scalar_projection(acts: torch.Tensor, vector: torch.Tensor, offset: torch.Tensor) -> np.ndarray:
    unit = F.normalize(vector.to(torch.float64), dim=-1)
    centered = acts.to(torch.float64) - offset.to(torch.float64)
    return (centered * unit).sum(dim=-1).numpy()


def constrained_disparity(
    logits_last: torch.Tensor,
    all_ids: list[int],
    n_pos: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_logits = logits_last[:, all_ids].detach().to("cpu").to(torch.float64)
    probs = F.softmax(target_logits, dim=-1)
    pos_probs = probs[:, :n_pos].sum(dim=-1).numpy()
    neg_probs = probs[:, n_pos:].sum(dim=-1).numpy()
    return pos_probs, neg_probs, pos_probs - neg_probs


def make_default_intervention_hook(unit_vec: torch.Tensor, offset: torch.Tensor, coeff: float):
    """Return a HF forward hook matching get_intervention_func(..., method="default")."""

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        # Use float32 for the projection math: MPS does not support float64, and
        # float16 projection arithmetic is too coarse for this diagnostic.
        hidden32 = hidden.to(torch.float32)
        unit32 = unit_vec.to(torch.float32)
        offset32 = offset.to(torch.float32)
        centered = hidden32 - offset32
        projection = (centered @ unit32).unsqueeze(-1) * unit32
        steered = (hidden32 - projection + unit32 * float(coeff)).to(hidden.dtype)
        if isinstance(output, tuple):
            return (steered,) + output[1:]
        return steered

    return hook


def forward_pass(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    batch_size: int,
    all_ids: list[int],
    n_pos: int,
    vector_cpu: torch.Tensor,
    offset_cpu: torch.Tensor,
    coeff: float | None = None,
) -> list[dict]:
    device = model.device
    vector = vector_cpu.to(device=device, dtype=model.dtype)
    offset = offset_cpu.to(device=device, dtype=model.dtype)
    unit = F.normalize(vector, dim=-1)

    hook_context = nullcontext()
    handle = None
    if coeff is not None:
        handle = model.transformer.h[LAYER].register_forward_hook(
            make_default_intervention_hook(unit, offset, coeff)
        )

    records: list[dict] = []
    try:
        with hook_context:
            for start, batch in batched(prompts, batch_size):
                inputs = tokenizer(batch, padding=True, truncation=False, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True, use_cache=False)

                logits_last = outputs.logits[:, -1, :]
                pos_probs, neg_probs, disparities = constrained_disparity(logits_last, all_ids, n_pos)

                layer_acts = outputs.hidden_states[LAYER + 1][:, -1, :].detach().to("cpu")
                projections = scalar_projection(layer_acts, vector_cpu, offset_cpu)

                for local_i in range(len(batch)):
                    records.append(
                        {
                            "row_index": start + local_i,
                            "projection": float(projections[local_i]),
                            "spatial_prob": float(pos_probs[local_i]),
                            "descriptive_prob": float(neg_probs[local_i]),
                            "disparity": float(disparities[local_i]),
                        }
                    )

                if (start // batch_size + 1) % 10 == 0 or start + len(batch) == len(prompts):
                    label = "baseline" if coeff is None else f"lambda={coeff:g}"
                    print(f"  {label}: {start + len(batch)}/{len(prompts)} prompts", flush=True)
    finally:
        if handle is not None:
            handle.remove()
    return records


def quantiles(values: np.ndarray) -> dict[str, float]:
    points = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]
    return {str(point): float(value) for point, value in zip(points, np.quantile(values, points))}


def summarize_xy(x: np.ndarray, y: np.ndarray) -> dict:
    corr = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0 else None
    slope, intercept = np.polyfit(x, y, 1) if len(x) > 1 else (math.nan, math.nan)
    sorted_x = np.sort(x)
    gaps = np.diff(sorted_x)
    if len(gaps):
        gap_idx = int(np.argmax(gaps))
        largest_gap = {
            "width": float(gaps[gap_idx]),
            "from": float(sorted_x[gap_idx]),
            "to": float(sorted_x[gap_idx + 1]),
        }
    else:
        largest_gap = {"width": 0.0, "from": None, "to": None}
    return {
        "n_points": int(len(x)),
        "projection_min": float(np.min(x)),
        "projection_max": float(np.max(x)),
        "projection_mean": float(np.mean(x)),
        "projection_quantiles": quantiles(x),
        "largest_adjacent_projection_gap": largest_gap,
        "disparity_min": float(np.min(y)),
        "disparity_max": float(np.max(y)),
        "disparity_mean": float(np.mean(y)),
        "disparity_median": float(np.median(y)),
        "disparity_rms": float(np.sqrt(np.mean(np.square(y)))),
        "disparity_quantiles": quantiles(y),
        "corr_projection_disparity": corr,
        "linear_slope": float(slope),
        "linear_intercept": float(intercept),
    }


def binned_trend(x: np.ndarray, y: np.ndarray, n_bins: int = 12) -> tuple[np.ndarray, np.ndarray]:
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    centers: list[float] = []
    means: list[float] = []
    for idx in range(n_bins):
        left, right = edges[idx], edges[idx + 1]
        if idx == n_bins - 1:
            mask = (x >= left) & (x <= right)
        else:
            mask = (x >= left) & (x < right)
        if not np.any(mask):
            continue
        centers.append(float(np.median(x[mask])))
        means.append(float(np.mean(y[mask])))
    return np.array(centers), np.array(means)


def style_axis(ax) -> None:
    ax.grid(True, color=GRID_COLOR, linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#8A94A6")
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9.5)


def add_fit_and_stats(ax, x: np.ndarray, y: np.ndarray, stats: dict, color: str) -> None:
    xs = np.linspace(float(np.min(x)), float(np.max(x)), 100)
    ax.plot(
        xs,
        stats["linear_slope"] * xs + stats["linear_intercept"],
        color=color,
        linewidth=1.75,
        alpha=0.86,
        label="linear fit",
    )
    bx, by = binned_trend(x, y)
    ax.plot(
        bx,
        by,
        color="#151A20",
        linewidth=1.65,
        marker="o",
        markersize=3.1,
        alpha=0.74,
        label="binned mean",
    )
    corr = stats["corr_projection_disparity"]
    corr_text = "n/a" if corr is None else f"{corr:.3f}"
    ax.text(
        0.035,
        0.965,
        f"r = {corr_text}\nmean = {stats['disparity_mean']:+.3f}\nRMS = {stats['disparity_rms']:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        color=TEXT_COLOR,
        bbox={"boxstyle": "round,pad=0.32", "facecolor": "white", "edgecolor": "#CBD2DD", "alpha": 0.9},
    )


def render_final_scatter(
    outdir: Path,
    baseline_rows: list[dict],
    steered_rows: list[dict],
    baseline_stats: dict,
    steered_stats: dict,
    final_lambda: float,
) -> Path:
    x = np.array([row["projection"] for row in baseline_rows])
    y_base = np.array([row["disparity"] for row in baseline_rows])
    y_steered = np.array([row["disparity"] for row in steered_rows])
    x_pad = 0.07 * (float(np.max(x)) - float(np.min(x)))
    x_range = (float(np.min(x)) - x_pad, float(np.max(x)) + x_pad)
    y_range = (-1.05, 1.05)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.75), sharex=True, sharey=True)
    panel_specs = [
        (axes[0], y_base, baseline_stats, BASELINE_COLOR, "Baseline"),
        (axes[1], y_steered, steered_stats, STEERED_COLOR, f"Steered ($\\lambda={final_lambda:g}$)"),
    ]
    for ax, y, stats, color, title in panel_specs:
        ax.scatter(x, y, s=9, alpha=0.30, color=color, edgecolors="none", rasterized=True)
        ax.axhline(0, color="#333842", linewidth=1.0, linestyle=(0, (5, 4)), alpha=0.7)
        add_fit_and_stats(ax, x, y, stats, color)
        ax.set_title(title, fontsize=12, color=TEXT_COLOR, pad=7)
        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)
        ax.set_xlabel("Baseline layer-11 projection", fontsize=9.8, color=TEXT_COLOR)
        style_axis(ax)

    axes[0].set_ylabel("Spatial - descriptive next-token disparity", fontsize=9.8, color=TEXT_COLOR)
    fig.tight_layout(w_pad=2.0)
    outpath = outdir / "qwen_image_shows_projection_scatter.png"
    fig.savefig(outpath, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return outpath


def render_hexbin_alternative(
    outdir: Path,
    baseline_rows: list[dict],
    steered_rows: list[dict],
    final_lambda: float,
) -> Path:
    x = np.array([row["projection"] for row in baseline_rows])
    y_base = np.array([row["disparity"] for row in baseline_rows])
    y_steered = np.array([row["disparity"] for row in steered_rows])
    x_pad = 0.07 * (float(np.max(x)) - float(np.min(x)))
    x_range = (float(np.min(x)) - x_pad, float(np.max(x)) + x_pad)
    y_range = (-1.05, 1.05)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.85), sharex=True, sharey=True)
    for ax, y, title in [
        (axes[0], y_base, "Baseline"),
        (axes[1], y_steered, f"Steered ($\\lambda={final_lambda:g}$)"),
    ]:
        hb = ax.hexbin(
            x,
            y,
            gridsize=(34, 24),
            extent=(x_range[0], x_range[1], y_range[0], y_range[1]),
            mincnt=1,
            cmap="mako" if "mako" in plt.colormaps() else "viridis",
            linewidths=0,
            alpha=0.92,
        )
        bx, by = binned_trend(x, y)
        ax.plot(bx, by, color="white", linewidth=2.0, marker="o", markersize=3.4)
        ax.axhline(0, color="white", linewidth=1.0, linestyle=(0, (5, 4)), alpha=0.8)
        ax.set_title(title, fontsize=12, color=TEXT_COLOR, pad=7)
        ax.set_xlabel("Baseline layer-11 projection", fontsize=9.8, color=TEXT_COLOR)
        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)
        style_axis(ax)
    axes[0].set_ylabel("Spatial - descriptive next-token disparity", fontsize=9.8, color=TEXT_COLOR)
    fig.subplots_adjust(right=0.88, wspace=0.18)
    cbar_ax = fig.add_axes([0.90, 0.20, 0.016, 0.64])
    cbar = fig.colorbar(hb, cax=cbar_ax)
    cbar.set_label("caption count", fontsize=9.5)
    outpath = outdir / "qwen_image_shows_projection_scatter_hexbin.png"
    fig.savefig(outpath, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return outpath


def render_overlay_alternative(
    outdir: Path,
    baseline_rows: list[dict],
    steered_rows: list[dict],
    final_lambda: float,
) -> Path:
    x = np.array([row["projection"] for row in baseline_rows])
    y_base = np.array([row["disparity"] for row in baseline_rows])
    y_steered = np.array([row["disparity"] for row in steered_rows])
    x_pad = 0.07 * (float(np.max(x)) - float(np.min(x)))
    x_range = (float(np.min(x)) - x_pad, float(np.max(x)) + x_pad)

    fig, ax = plt.subplots(figsize=(6.1, 4.0))
    ax.scatter(x, y_base, s=11, alpha=0.22, color=BASELINE_COLOR, edgecolors="none", label="baseline")
    ax.scatter(
        x,
        y_steered,
        s=11,
        alpha=0.22,
        color=STEERED_COLOR,
        edgecolors="none",
        label=f"steered $\\lambda={final_lambda:g}$",
    )
    for y, color in [(y_base, BASELINE_COLOR), (y_steered, STEERED_COLOR)]:
        bx, by = binned_trend(x, y)
        ax.plot(bx, by, color=color, linewidth=2.3, marker="o", markersize=3.8)
    ax.axhline(0, color="#333842", linewidth=1.0, linestyle=(0, (5, 4)), alpha=0.7)
    ax.set_xlim(*x_range)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Baseline layer-11 projection", fontsize=9.8, color=TEXT_COLOR)
    ax.set_ylabel("Spatial - descriptive next-token disparity", fontsize=9.8, color=TEXT_COLOR)
    ax.set_title(f'Before/after overlay, $\\lambda={final_lambda:g}$', fontsize=11.5)
    ax.legend(frameon=False, fontsize=8.8, loc="upper left")
    style_axis(ax)
    fig.tight_layout()
    outpath = outdir / "qwen_image_shows_projection_scatter_overlay.png"
    fig.savefig(outpath, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return outpath


def write_csv(
    outdir: Path,
    val_rows: list[dict],
    baseline_rows: list[dict],
    steered_by_lambda: dict[float, list[dict]],
) -> Path:
    csv_path = outdir / "qwen_image_shows_projection_scatter.csv"
    fieldnames = [
        "row_index",
        "condition",
        "lambda",
        "prefix",
        "prompt",
        "caption",
        "vision_label",
        "baseline_projection",
        "projection",
        "spatial_prob",
        "descriptive_prob",
        "disparity",
    ]
    rows_to_write: list[dict] = []
    for base in baseline_rows:
        source = val_rows[base["row_index"]]
        caption = source["text"]
        rows_to_write.append(
            {
                "row_index": base["row_index"],
                "condition": "baseline",
                "lambda": "",
                "prefix": PREFIX,
                "prompt": prompt_for_csv(caption),
                "caption": text_for_csv(caption),
                "vision_label": source.get("vision_label", ""),
                "baseline_projection": base["projection"],
                "projection": base["projection"],
                "spatial_prob": base["spatial_prob"],
                "descriptive_prob": base["descriptive_prob"],
                "disparity": base["disparity"],
            }
        )
    baseline_projection_by_row = {row["row_index"]: row["projection"] for row in baseline_rows}
    for lam, steered_rows in steered_by_lambda.items():
        for row in steered_rows:
            source = val_rows[row["row_index"]]
            caption = source["text"]
            rows_to_write.append(
                {
                    "row_index": row["row_index"],
                    "condition": "steered",
                    "lambda": lam,
                    "prefix": PREFIX,
                    "prompt": prompt_for_csv(caption),
                    "caption": text_for_csv(caption),
                    "vision_label": source.get("vision_label", ""),
                    "baseline_projection": baseline_projection_by_row[row["row_index"]],
                    "projection": row["projection"],
                    "spatial_prob": row["spatial_prob"],
                    "descriptive_prob": row["descriptive_prob"],
                    "disparity": row["disparity"],
                }
            )
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows_to_write)
    return csv_path


def write_note(
    outdir: Path,
    stats: dict,
    figure_paths: list[Path],
) -> Path:
    final_lambda = stats["selected_lambda"]
    baseline = stats["baseline"]
    steered = stats["steered"][str(final_lambda)]
    gentle = stats["steered"].get("10") or stats["steered"].get("10.0")
    old_prefix_note = ""
    old_stats_path = outdir / "qwen_projection_scatter_by_prefix_stats.json"
    if old_stats_path.exists():
        old_stats = json.loads(old_stats_path.read_text(encoding="utf-8"))
        by_prefix = old_stats.get("by_prefix", {})
        if by_prefix:
            old_prefix_note = (
                " For comparison, the older three-prefix diagnostic had separated projection means "
                + ", ".join(
                    f"`{prefix}`={values['projection_mean']:.3f}"
                    for prefix, values in by_prefix.items()
                    if "projection_mean" in values
                )
                + "."
            )
    note_path = outdir / "qwen_image_shows_projection_scatter_note.md"
    lines = [
        "# Qwen Image-Shows Projection Scatter",
        "",
        "## Data",
        "",
        f"- Model: `{MODEL_NAME}`",
        f"- Validation data: `{stats['source_val_json']}`",
        f"- Captions: {stats['n_captions']:,}",
        f"- Prompt serialization: Qwen chat template with `Describe this image:\\n[caption]` and continuation prefix `{PREFIX}`.",
        f"- Steering layer: {stats['layer']}",
        "- Intervention: `default` projection replacement with the saved neutral offset.",
        "- Next-token disparity: constrained softmax over spatial + descriptive target tokens, then spatial probability minus descriptive probability.",
        f"- Final plot lambda: `{final_lambda:g}`.",
        "- X-axis: baseline scalar projection for both panels. This keeps each caption at the same horizontal coordinate; using steered projection would collapse the steered panel around the chosen lambda.",
        "",
        "## Summary Stats",
        "",
        "| quantity | baseline | steered |",
        "|---|---:|---:|",
        f"| n | {baseline['n_points']} | {steered['n_points']} |",
        f"| projection range | [{baseline['projection_min']:.3f}, {baseline['projection_max']:.3f}] | same x-axis |",
        f"| disparity range | [{baseline['disparity_min']:.3f}, {baseline['disparity_max']:.3f}] | [{steered['disparity_min']:.3f}, {steered['disparity_max']:.3f}] |",
        f"| corr(projection, disparity) | {baseline['corr_projection_disparity']:.3f} | {steered['corr_projection_disparity']:.3f} |",
        f"| mean disparity | {baseline['disparity_mean']:+.3f} | {steered['disparity_mean']:+.3f} |",
        f"| RMS disparity | {baseline['disparity_rms']:.3f} | {steered['disparity_rms']:.3f} |",
        "",
        "The single-prefix projection distribution removes the earlier prompt-family bands by construction. Within `The image shows`, the projection span is continuous enough for a caption-level diagnostic; the largest adjacent projection gap is "
        f"{baseline['largest_adjacent_projection_gap']['width']:.3f}, and no second prompt-template cluster is present.{old_prefix_note}",
        "",
        "## Coefficient Choice",
        "",
        f"`lambda=20` is the final choice because it is the strongest low-degeneration setting from the full-validation `A_image_shows` local sweep and gives a visibly interpretable next-token shift here: mean disparity changes by {steered['mean_shift_vs_baseline']:+.3f} and RMS falls by {abs(steered['rms_delta_vs_baseline']):.3f}.",
    ]
    if gentle is not None:
        lines.append(
            f"`lambda=10` is a gentler alternative that nearly centers the mean next-token disparity ({gentle['disparity_mean']:+.3f}) but produces a smaller RMS reduction ({abs(gentle['rms_delta_vs_baseline']):.3f}), so it is less visually diagnostic for the paper figure."
        )
    lines.extend(
        [
        "",
        "## Figure Ranking",
        "",
        "1. `qwen_image_shows_projection_scatter.png` - selected final. The two-panel layout keeps the baseline and steered distributions directly comparable, preserves individual captions, and adds linear plus binned trends without hiding saturation near +/-1.",
        "2. `qwen_image_shows_projection_scatter_overlay.png` - useful for seeing before/after movement in one frame, but the overlaid clouds are harder to read in print.",
        "3. `qwen_image_shows_projection_scatter_hexbin.png` - best for density, but less literal as a scatter diagnostic and slightly less transparent for readers.",
        "",
        "## Suggested Paper Text",
        "",
        "Figure~\\ref{fig:projection-scatter} isolates the layer-selection diagnostic to the single continuation used in the main local run. For each of the 1,000 held-out COCO captions, we serialize the prompt as `Describe this image:` followed by the caption and the continuation prefix `The image shows`. The left panel plots the baseline layer-11 scalar projection against the constrained next-token spatial--descriptive disparity. The strong positive association shows that the selected steering direction is aligned with the model's own next-token spatial signal under the baseline prompt condition, not with a mixture of prompt-template offsets. The right panel keeps the same baseline projection on the x-axis and recomputes the next-token disparity after applying the saved layer-11 steering intervention at $\\lambda=20$, the strongest low-degeneration setting from the full-validation local sweep. The shifted disparity distribution shows the expected movement induced by the steering vector while preserving a caption-by-caption diagnostic view.",
        "",
        "## Suggested Caption",
        "",
        "Projection--disparity diagnostic for the local \\model{Qwen/Qwen-1\\_8B-chat} run using only the continuation prefix ``The image shows.'' Each point is one held-out COCO caption from the saved 1,000-caption validation split. Both panels use the baseline layer-11 scalar projection as the x-axis; the y-axis is the constrained next-token spatial-minus-descriptive disparity over the tracked target tokens. Left: baseline prompts. Right: the same prompts after applying the saved layer-11 steering intervention with $\\lambda=20$. Linear fits and binned means summarize the caption-level trend.",
        "",
        "## Regeneration",
        "",
        "```bash",
        "python plotting/build_qwen_image_shows_projection_scatter.py --lambdas 0 10 20 --final-lambda 20",
        "```",
        "",
        "## Outputs",
        "",
        ]
    )
    for path in figure_paths:
        lines.append(f"- `{path.relative_to(REPO_ROOT)}`")
    lines.extend(
        [
        "- `paper/figures/qwen_image_shows_projection_scatter.csv`",
        "- `paper/figures/qwen_image_shows_projection_scatter_stats.json`",
        ]
    )
    note_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return note_path


def load_render_bundle(outdir: Path, final_lambda: float) -> tuple[list[dict], list[dict], dict]:
    csv_path = outdir / "qwen_image_shows_projection_scatter.csv"
    stats_path = outdir / "qwen_image_shows_projection_scatter_stats.json"
    if not csv_path.exists() or not stats_path.exists():
        raise FileNotFoundError("Render-only mode needs the saved CSV and stats JSON.")

    baseline_rows: list[dict] = []
    steered_rows: list[dict] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parsed = {
                "row_index": int(row["row_index"]),
                "projection": float(row["baseline_projection"]),
                "spatial_prob": float(row["spatial_prob"]),
                "descriptive_prob": float(row["descriptive_prob"]),
                "disparity": float(row["disparity"]),
            }
            if row["condition"] == "baseline":
                baseline_rows.append(parsed)
            elif row["condition"] == "steered" and float(row["lambda"]) == float(final_lambda):
                # Keep the baseline projection as the plotting x-coordinate.
                steered_rows.append(parsed)

    baseline_rows.sort(key=lambda item: item["row_index"])
    steered_rows.sort(key=lambda item: item["row_index"])
    if not baseline_rows or not steered_rows:
        raise ValueError(f"No render rows found for final lambda {final_lambda:g}.")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    return baseline_rows, steered_rows, stats


def main() -> None:
    args = parse_args()
    outdir = args.out_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if args.final_lambda not in args.lambdas:
        args.lambdas.append(args.final_lambda)
    args.lambdas = sorted(set(float(lam) for lam in args.lambdas))

    if args.render_only:
        baseline_rows, final_rows, stats = load_render_bundle(outdir, args.final_lambda)
        final_key = str(int(args.final_lambda)) if float(args.final_lambda).is_integer() else str(float(args.final_lambda))
        png_path = render_final_scatter(
            outdir,
            baseline_rows,
            final_rows,
            stats["baseline"],
            stats["steered"][final_key],
            float(args.final_lambda),
        )
        overlay_path = render_overlay_alternative(outdir, baseline_rows, final_rows, float(args.final_lambda))
        hexbin_path = render_hexbin_alternative(outdir, baseline_rows, final_rows, float(args.final_lambda))
        note_path = write_note(outdir, stats, [png_path, overlay_path, hexbin_path])
        print(
            json.dumps(
                {
                    "figure": str(png_path.relative_to(REPO_ROOT)),
                    "note": str(note_path.relative_to(REPO_ROOT)),
                    "render_only": True,
                },
                indent=2,
            ),
            flush=True,
        )
        return

    val_rows = json.loads(VAL_JSON.read_text(encoding="utf-8"))
    if args.max_examples is not None:
        val_rows = val_rows[: args.max_examples]
    captions = [row["text"] for row in val_rows]

    print(f"Loading {MODEL_NAME} for {len(captions)} validation captions...", flush=True)
    tokenizer, model, device = load_qwen_hf(args.force_cpu)
    print(f"Using device: {device}; model dtype: {model.dtype}", flush=True)

    pos_ids, neg_ids = load_target_ids(tokenizer)
    all_ids = pos_ids + neg_ids
    n_pos = len(pos_ids)
    print(f"Target ids after overlap removal: spatial={len(pos_ids)} descriptive={len(neg_ids)}", flush=True)

    candidate_vectors = torch.load(ARTIFACT_DIR / "activations" / "candidate_vectors.pt", map_location="cpu")
    neutral = torch.load(ARTIFACT_DIR / "activations" / "neutral.pt", map_location="cpu")
    vector_cpu = candidate_vectors[LAYER].to(torch.float64)
    offset_cpu = neutral.mean(dim=1)[LAYER].to(torch.float64)

    prompts = apply_qwen_template(tokenizer, captions)

    print("Computing baseline forward passes...", flush=True)
    baseline_rows = forward_pass(
        model,
        tokenizer,
        prompts,
        args.batch_size,
        all_ids,
        n_pos,
        vector_cpu,
        offset_cpu,
        coeff=None,
    )

    steered_by_lambda: dict[float, list[dict]] = {}
    for lam in args.lambdas:
        print(f"Computing steered forward passes for lambda={lam:g}...", flush=True)
        steered_by_lambda[lam] = forward_pass(
            model,
            tokenizer,
            prompts,
            args.batch_size,
            all_ids,
            n_pos,
            vector_cpu,
            offset_cpu,
            coeff=lam,
        )

    x = np.array([row["projection"] for row in baseline_rows])
    baseline_y = np.array([row["disparity"] for row in baseline_rows])
    stats = {
        "model": MODEL_NAME,
        "source_val_json": str(VAL_JSON.relative_to(REPO_ROOT)),
        "n_captions": len(val_rows),
        "prefix": PREFIX,
        "instruction_template": INSTRUCTION_TEMPLATE,
        "prompt_serialization": "Qwen chat template; output prefix appended after assistant generation marker.",
        "layer": LAYER,
        "selected_lambda": float(args.final_lambda),
        "candidate_lambdas": args.lambdas,
        "x_axis": "baseline_projection",
        "next_token_disparity": "constrained_softmax(spatial_ids + descriptive_ids); spatial_prob - descriptive_prob",
        "steering_method": "default",
        "uses_neutral_offset": True,
        "target_token_counts": {"spatial": len(pos_ids), "descriptive": len(neg_ids)},
        "baseline": summarize_xy(x, baseline_y),
        "steered": {},
    }

    for lam, rows in steered_by_lambda.items():
        y = np.array([row["disparity"] for row in rows])
        entry = summarize_xy(x, y)
        entry["mean_shift_vs_baseline"] = float(np.mean(y) - np.mean(baseline_y))
        entry["rms_delta_vs_baseline"] = float(entry["disparity_rms"] - stats["baseline"]["disparity_rms"])
        steered_projection = np.array([row["projection"] for row in rows])
        entry["steered_projection_mean"] = float(np.mean(steered_projection))
        entry["steered_projection_std"] = float(np.std(steered_projection))
        entry["steered_projection_min"] = float(np.min(steered_projection))
        entry["steered_projection_max"] = float(np.max(steered_projection))
        stats["steered"][str(float(lam))] = entry
        # Also provide compact integer-like keys for human lookup when possible.
        if float(lam).is_integer():
            stats["steered"][str(int(lam))] = entry

    final_rows = steered_by_lambda[float(args.final_lambda)]
    final_key = str(int(args.final_lambda)) if float(args.final_lambda).is_integer() else str(float(args.final_lambda))
    png_path = render_final_scatter(
        outdir,
        baseline_rows,
        final_rows,
        stats["baseline"],
        stats["steered"][final_key],
        float(args.final_lambda),
    )
    overlay_path = render_overlay_alternative(outdir, baseline_rows, final_rows, float(args.final_lambda))
    hexbin_path = render_hexbin_alternative(outdir, baseline_rows, final_rows, float(args.final_lambda))

    csv_path = write_csv(outdir, val_rows, baseline_rows, steered_by_lambda)
    stats_path = outdir / "qwen_image_shows_projection_scatter_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    note_path = write_note(outdir, stats, [png_path, overlay_path, hexbin_path])

    print(
        json.dumps(
            {
                "figure": str(png_path.relative_to(REPO_ROOT)),
                "csv": str(csv_path.relative_to(REPO_ROOT)),
                "stats": str(stats_path.relative_to(REPO_ROOT)),
                "note": str(note_path.relative_to(REPO_ROOT)),
                "baseline_mean_disparity": stats["baseline"]["disparity_mean"],
                "final_lambda": args.final_lambda,
                "steered_mean_disparity": stats["steered"][final_key]["disparity_mean"],
                "baseline_corr": stats["baseline"]["corr_projection_disparity"],
                "steered_corr": stats["steered"][final_key]["corr_projection_disparity"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

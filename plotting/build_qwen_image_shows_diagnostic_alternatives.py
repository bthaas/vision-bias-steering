#!/usr/bin/env python3
"""Render alternative "The image shows" projection/disparity diagnostics.

This renderer uses the per-caption CSV produced by
build_qwen_image_shows_projection_scatter.py. It does not load Qwen or edit the
paper. The underlying data are the same local Qwen/Qwen-1_8B-chat validation
captions, layer-11 steering vector, neutral offset, and constrained next-token
spatial-minus-descriptive disparity metric.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "paper" / "figures"
SOURCE_CSV = FIG_DIR / "qwen_image_shows_projection_scatter.csv"
SOURCE_STATS = FIG_DIR / "qwen_image_shows_projection_scatter_stats.json"
PREFIX = "The image shows"
LAMBDAS = (10.0, 20.0)

BLUE = "#2F6FBB"
RED = "#D65F5F"
GREEN = "#3D9970"
GRAY = "#2A2F36"
LIGHT_GRID = "#D8DEE9"
TEXT = "#20242A"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=SOURCE_CSV)
    parser.add_argument("--stats", type=Path, default=SOURCE_STATS)
    parser.add_argument("--out-dir", type=Path, default=FIG_DIR)
    return parser.parse_args()


def load_rows(path: Path) -> dict[str, dict[int, dict]]:
    by_condition: dict[str, dict[int, dict]] = {"baseline": {}}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["condition"] == "baseline":
                key = "baseline"
            else:
                key = f"lambda_{float(row['lambda']):g}"
            by_condition.setdefault(key, {})[int(row["row_index"])] = row
    return by_condition


def arrays(by_condition: dict[str, dict[int, dict]], lam: float | None = None) -> dict[str, np.ndarray]:
    indices = sorted(by_condition["baseline"])
    base_rows = by_condition["baseline"]
    base_disparity = np.array([float(base_rows[i]["disparity"]) for i in indices])
    projection = np.array([float(base_rows[i]["baseline_projection"]) for i in indices])
    out = {"index": np.array(indices), "projection": projection, "baseline": base_disparity}
    if lam is not None:
        key = f"lambda_{lam:g}"
        steered_rows = by_condition[key]
        steered = np.array([float(steered_rows[i]["disparity"]) for i in indices])
        out["steered"] = steered
        out["delta"] = steered - base_disparity
    return out


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def binned_mean(x: np.ndarray, y: np.ndarray, n_bins: int = 14) -> tuple[np.ndarray, np.ndarray]:
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    xs: list[float] = []
    ys: list[float] = []
    for idx in range(n_bins):
        left, right = edges[idx], edges[idx + 1]
        if idx == n_bins - 1:
            mask = (x >= left) & (x <= right)
        else:
            mask = (x >= left) & (x < right)
        if np.any(mask):
            xs.append(float(np.median(x[mask])))
            ys.append(float(np.mean(y[mask])))
    return np.array(xs), np.array(ys)


def common_axis(ax) -> None:
    ax.grid(True, color=LIGHT_GRID, linewidth=0.75, alpha=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT, labelsize=8.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#8A94A6")
        ax.spines[spine].set_linewidth(0.8)


def stat_box(ax, text: str, loc: tuple[float, float] = (0.035, 0.965)) -> None:
    ax.text(
        loc[0],
        loc[1],
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.4,
        color=TEXT,
        bbox={"boxstyle": "round,pad=0.30", "facecolor": "white", "edgecolor": "#CBD2DD", "alpha": 0.93},
    )


def reduced_abs_fraction(before: np.ndarray, after: np.ndarray) -> float:
    return float(np.mean(np.abs(after) < np.abs(before)))


def crossed_zero_fraction(before: np.ndarray, after: np.ndarray) -> float:
    return float(np.mean(np.sign(before) != np.sign(after)))


def render_option1(data: dict[str, np.ndarray], out_dir: Path) -> Path:
    x = data["projection"]
    y = data["baseline"]
    fit = np.polyfit(x, y, 1)
    xs = np.linspace(float(x.min()), float(x.max()), 100)
    bx, by = binned_mean(x, y)

    fig, ax = plt.subplots(figsize=(5.45, 3.8))
    ax.scatter(x, y, s=10, alpha=0.30, color=BLUE, edgecolors="none", rasterized=True)
    ax.plot(xs, fit[0] * xs + fit[1], color=BLUE, linewidth=1.7, alpha=0.86, label="linear fit")
    ax.plot(bx, by, color=GRAY, linewidth=1.7, marker="o", markersize=3.2, alpha=0.78, label="binned mean")
    ax.axhline(0, color="#606770", linewidth=1.0, linestyle=(0, (5, 4)))
    ax.set_xlabel("Baseline layer-11 projection", fontsize=9.8, color=TEXT)
    ax.set_ylabel("Spatial - descriptive next-token disparity", fontsize=9.8, color=TEXT)
    ax.set_title("Baseline alignment", fontsize=11.0, color=TEXT)
    ax.set_ylim(-1.05, 1.05)
    stat_box(ax, f"r = {corr(x, y):.3f}\nmean = {y.mean():+.3f}\nRMS = {rms(y):.3f}")
    ax.legend(frameon=False, fontsize=8.3, loc="lower right")
    common_axis(ax)
    fig.tight_layout()
    out = out_dir / "qwen_image_shows_option1_baseline_alignment.png"
    fig.savefig(out, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return out


def render_option2(data: dict[str, np.ndarray], lam: float, out_dir: Path) -> Path:
    before = data["baseline"]
    after = data["steered"]
    bx, by = binned_mean(before, after)
    lim = (-1.04, 1.04)
    grid = np.linspace(lim[0], lim[1], 300)

    fig, ax = plt.subplots(figsize=(5.1, 4.15))
    reduced_lower = -np.abs(grid)
    reduced_upper = np.abs(grid)
    ax.fill_between(
        grid,
        reduced_lower,
        reduced_upper,
        color=GREEN,
        alpha=0.08,
        linewidth=0,
        label="reduced |disparity|",
    )
    ax.scatter(before, after, s=10, alpha=0.30, color=RED if lam == 20 else BLUE, edgecolors="none", rasterized=True)
    ax.plot(lim, lim, color=GRAY, linewidth=1.05, linestyle=(0, (5, 4)), label="unchanged")
    ax.axhline(0, color="#606770", linewidth=0.95)
    ax.axvline(0, color="#606770", linewidth=0.95)
    ax.plot(bx, by, color=GRAY, linewidth=1.85, marker="o", markersize=3.3, alpha=0.78, label="binned mean")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Baseline disparity", fontsize=9.8, color=TEXT)
    ax.set_ylabel("Steered disparity", fontsize=9.8, color=TEXT)
    ax.set_title(f"Paired before/after disparity (lambda = {lam:g})", fontsize=11.0, color=TEXT)
    stat_box(
        ax,
        f"mean: {before.mean():+.3f} -> {after.mean():+.3f}\n"
        f"RMS: {rms(before):.3f} -> {rms(after):.3f}\n"
        f"abs reduced: {reduced_abs_fraction(before, after) * 100:.1f}%",
    )
    ax.legend(frameon=False, fontsize=8.0, loc="lower right")
    common_axis(ax)
    fig.tight_layout()
    out = out_dir / f"qwen_image_shows_option2_paired_disparity_lambda{int(lam)}.png"
    fig.savefig(out, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return out


def render_option3(data: dict[str, np.ndarray], lam: float, out_dir: Path) -> Path:
    before = data["baseline"]
    delta = data["delta"]
    bx, by = binned_mean(before, delta)
    lim_x = (-1.04, 1.04)
    lim_y = (-1.25, 1.55)
    grid = np.linspace(lim_x[0], lim_x[1], 300)

    fig, ax = plt.subplots(figsize=(5.55, 3.95))
    reduced_lower = np.where(grid >= 0, -2 * grid, 0)
    reduced_upper = np.where(grid >= 0, 0, -2 * grid)
    ax.fill_between(
        grid,
        reduced_lower,
        reduced_upper,
        color=GREEN,
        alpha=0.08,
        linewidth=0,
        label="reduced |disparity|",
    )
    ax.scatter(before, delta, s=10, alpha=0.30, color=RED if lam == 20 else BLUE, edgecolors="none", rasterized=True)
    ax.plot(grid, -grid, color=GRAY, linewidth=1.05, linestyle=(0, (5, 4)), label="after = 0")
    ax.axhline(0, color="#606770", linewidth=0.95)
    ax.axvline(0, color="#606770", linewidth=0.95)
    ax.plot(bx, by, color=GRAY, linewidth=1.85, marker="o", markersize=3.3, alpha=0.78, label="binned mean")
    ax.set_xlim(*lim_x)
    ax.set_ylim(*lim_y)
    ax.set_xlabel("Baseline disparity", fontsize=9.8, color=TEXT)
    ax.set_ylabel("Steered - baseline disparity", fontsize=9.8, color=TEXT)
    ax.set_title(f"Change in disparity (lambda = {lam:g})", fontsize=11.0, color=TEXT)
    stat_box(
        ax,
        f"mean change = {delta.mean():+.3f}\n"
        f"corr(base, change) = {corr(before, delta):+.3f}\n"
        f"|d| reduced = {reduced_abs_fraction(before, before + delta) * 100:.1f}%",
    )
    ax.legend(frameon=False, fontsize=8.0, loc="lower right")
    common_axis(ax)
    fig.tight_layout()
    out = out_dir / f"qwen_image_shows_option3_disparity_change_lambda{int(lam)}.png"
    fig.savefig(out, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return out


def render_option4(data: dict[str, np.ndarray], lam: float, out_dir: Path) -> Path:
    projection = data["projection"]
    before = data["baseline"]
    after = data["steered"]
    bx, by = binned_mean(projection, before)
    fit = np.polyfit(projection, before, 1)
    xs = np.linspace(float(projection.min()), float(projection.max()), 100)

    fig, axes = plt.subplots(1, 2, figsize=(9.25, 3.65), gridspec_kw={"width_ratios": [1.05, 1.0]})
    ax = axes[0]
    ax.scatter(projection, before, s=9, alpha=0.28, color=BLUE, edgecolors="none", rasterized=True)
    ax.plot(xs, fit[0] * xs + fit[1], color=BLUE, linewidth=1.55, alpha=0.84)
    ax.plot(bx, by, color=GRAY, linewidth=1.6, marker="o", markersize=2.9, alpha=0.76)
    ax.axhline(0, color="#606770", linewidth=0.95, linestyle=(0, (5, 4)))
    ax.set_xlabel("Baseline layer-11 projection", fontsize=9.5, color=TEXT)
    ax.set_ylabel("Baseline disparity", fontsize=9.5, color=TEXT)
    ax.set_title("Alignment before steering", fontsize=10.7, color=TEXT)
    ax.set_ylim(-1.05, 1.05)
    stat_box(ax, f"r = {corr(projection, before):.3f}\nRMS = {rms(before):.3f}")
    common_axis(ax)

    ax = axes[1]
    bins = np.linspace(-1, 1, 41)
    ax.hist(before, bins=bins, density=True, histtype="stepfilled", color=BLUE, alpha=0.22, label="baseline")
    ax.hist(before, bins=bins, density=True, histtype="step", color=BLUE, linewidth=1.6)
    ax.hist(after, bins=bins, density=True, histtype="stepfilled", color=RED if lam == 20 else GREEN, alpha=0.22, label=f"steered lambda={lam:g}")
    ax.hist(after, bins=bins, density=True, histtype="step", color=RED if lam == 20 else GREEN, linewidth=1.6)
    ax.axvline(0, color="#606770", linewidth=0.95, linestyle=(0, (5, 4)))
    ax.axvline(before.mean(), color=BLUE, linewidth=1.3)
    ax.axvline(after.mean(), color=RED if lam == 20 else GREEN, linewidth=1.3)
    ax.set_xlabel("Next-token disparity", fontsize=9.5, color=TEXT)
    ax.set_ylabel("Density", fontsize=9.5, color=TEXT)
    ax.set_title("Distribution shift", fontsize=10.7, color=TEXT)
    stat_box(ax, f"mean: {before.mean():+.3f} -> {after.mean():+.3f}\nRMS: {rms(before):.3f} -> {rms(after):.3f}")
    ax.legend(frameon=False, fontsize=8.0, loc="upper right")
    common_axis(ax)

    fig.suptitle(f"Alignment plus distribution shift (lambda = {lam:g})", fontsize=11.0, color=TEXT, y=1.02)
    fig.tight_layout(w_pad=2.0)
    out = out_dir / f"qwen_image_shows_option4_alignment_distribution_lambda{int(lam)}.png"
    fig.savefig(out, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return out


def summarize_option_stats(data_by_lambda: dict[float, dict[str, np.ndarray]]) -> dict:
    baseline = next(iter(data_by_lambda.values()))["baseline"]
    projection = next(iter(data_by_lambda.values()))["projection"]
    out = {
        "prefix": PREFIX,
        "n": int(len(baseline)),
        "baseline": {
            "mean_disparity": float(baseline.mean()),
            "rms_disparity": rms(baseline),
            "projection_disparity_corr": corr(projection, baseline),
            "disparity_range": [float(baseline.min()), float(baseline.max())],
            "projection_range": [float(projection.min()), float(projection.max())],
        },
        "steered": {},
    }
    for lam, data in data_by_lambda.items():
        after = data["steered"]
        delta = data["delta"]
        out["steered"][str(int(lam))] = {
            "mean_disparity": float(after.mean()),
            "rms_disparity": rms(after),
            "projection_disparity_corr": corr(projection, after),
            "baseline_after_corr": corr(baseline, after),
            "mean_delta": float(delta.mean()),
            "baseline_delta_corr": corr(baseline, delta),
            "projection_delta_corr": corr(projection, delta),
            "reduced_abs_disparity_fraction": reduced_abs_fraction(baseline, after),
            "crossed_zero_fraction": crossed_zero_fraction(baseline, after),
            "rms_reduction_pct": float((rms(baseline) - rms(after)) / rms(baseline) * 100.0),
            "disparity_range": [float(after.min()), float(after.max())],
        }
    return out


def write_note(out_dir: Path, paths: dict[str, Path], stats: dict) -> Path:
    b = stats["baseline"]
    s10 = stats["steered"]["10"]
    s20 = stats["steered"]["20"]
    note = out_dir / "qwen_image_shows_diagnostic_alternatives_note.md"
    lines = [
        "# Qwen Image-Shows Diagnostic Alternatives",
        "",
        "All plots use only `The image shows`, the saved 1,000-caption validation split, layer 11, the saved steering vector/neutral offset logic, and the constrained next-token spatial-minus-descriptive disparity metric.",
        "",
        "## Summary",
        "",
        "| setting | mean disparity | RMS disparity | corr(projection, disparity) | notes |",
        "|---|---:|---:|---:|---|",
        f"| baseline | {b['mean_disparity']:+.3f} | {b['rms_disparity']:.3f} | {b['projection_disparity_corr']:.3f} | strong baseline alignment |",
        f"| lambda=10 | {s10['mean_disparity']:+.3f} | {s10['rms_disparity']:.3f} | {s10['projection_disparity_corr']:.3f} | nearly centers the mean, but only {s10['rms_reduction_pct']:.1f}% RMS reduction |",
        f"| lambda=20 | {s20['mean_disparity']:+.3f} | {s20['rms_disparity']:.3f} | {s20['projection_disparity_corr']:.3f} | clearer control effect, {s20['rms_reduction_pct']:.1f}% RMS reduction, but mean shifts positive |",
        "",
        "## Options",
        "",
        f"1. Option 1 baseline alignment only: `{paths['option1'].name}`. This is the cleanest layer-selection diagnostic: projection and baseline disparity align strongly (`r=0.738`). It does not show intervention effects, so it should be paired in text with the coefficient-response curve.",
        "",
        f"2. Option 2 paired before/after, lambda=10: `{paths['option2_10'].name}`. This directly compares each prompt before and after steering. It is conservative and interpretable because the mean moves from -0.128 to -0.006, but the cloud stays close to the diagonal and the RMS reduction is modest.",
        "",
        f"3. Option 2 paired before/after, lambda=20: `{paths['option2_20'].name}`. This is the strongest single diagnostic. It directly shows per-prompt movement, includes the unchanged diagonal and zero axes, and {s20['reduced_abs_disparity_fraction'] * 100:.1f}% of prompts move closer to zero. It should be described as positive spatial steering with RMS reduction, not as perfect centering.",
        "",
        f"4. Option 3 change plot, lambda=10: `{paths['option3_10'].name}`. The change is easy to define, but visually subtle. It supports the conservative story rather than making it obvious.",
        "",
        f"5. Option 3 change plot, lambda=20: `{paths['option3_20'].name}`. It explains the mechanism well: prompts with more negative baseline disparity receive larger positive changes (`corr(base, change)=-0.797`). It is analytically useful but a little less immediately legible than the paired scatter.",
        "",
        f"6. Option 4 alignment plus distribution, lambda=10: `{paths['option4_10'].name}`. This combines baseline alignment with the conservative mean-centering story, but the distribution change is small.",
        "",
        f"7. Option 4 alignment plus distribution, lambda=20: `{paths['option4_20'].name}`. This is a good backup if the paper wants alignment and aggregate distribution in one figure. It is less prompt-specific than the paired scatter.",
        "",
        "## Ranking",
        "",
        "1. `qwen_image_shows_option2_paired_disparity_lambda20.png` - recommended final. It is the most reader-friendly intervention diagnostic and fixes the old two-panel ambiguity by plotting before vs. after disparity directly.",
        "2. `qwen_image_shows_option4_alignment_distribution_lambda20.png` - best combined alignment plus aggregate shift figure.",
        "3. `qwen_image_shows_option1_baseline_alignment.png` - best if the figure should only support layer/vector alignment and leave intervention effects to response curves.",
        "4. `qwen_image_shows_option3_disparity_change_lambda20.png` - best mechanistic supplement, but less intuitive as the main paper figure.",
        "5. Lambda=10 variants - useful conservative checks, but the visual effect is too subtle for the main diagnostic.",
        "",
        "## Recommendation",
        "",
        "Use Option 2 at `lambda=20` as the final diagnostic if the subsection is allowed to say that steering reduces RMS disparity while shifting the next-token distribution in the spatial direction. Use Option 1 only if the paper wants this figure to make a narrower claim about baseline vector alignment and leave all intervention claims to the coefficient-response curve.",
        "",
        "## Suggested Replacement Paragraph",
        "",
        "Figure~\\ref{fig:projection-scatter} isolates the next-token diagnostic to the main local prompt condition, using only the continuation prefix ``The image shows.'' The selected layer-11 direction is first validated by its strong baseline association with the constrained spatial-minus-descriptive next-token disparity. To make the intervention effect caption-by-caption rather than template-driven, the figure plots each prompt's baseline disparity against its disparity after applying the saved layer-11 steering intervention. Points on the diagonal would be unchanged, while points in the shaded region have smaller absolute disparity after steering. At $\\lambda=20$, the intervention reduces RMS disparity from 0.788 to 0.558 and moves 75.3\\% of prompts closer to zero, while also shifting the mean disparity from -0.128 to +0.275. Thus the diagnostic should be read as controlled positive spatial steering with reduced next-token disparity magnitude, not as exact centering of the distribution.",
        "",
        "## Suggested Caption",
        "",
        "Paired next-token disparity diagnostic for the local \\model{Qwen/Qwen-1\\_8B-chat} run using only the continuation prefix ``The image shows.'' Each point is one held-out COCO caption from the saved 1,000-caption validation split. The x-axis shows the baseline constrained spatial-minus-descriptive next-token disparity; the y-axis shows the same quantity after applying the saved layer-11 steering intervention with $\\lambda=20$. The dashed diagonal marks no change, and the zero lines separate spatial-favoring from descriptive-favoring next-token distributions. Points in the shaded regions have smaller absolute disparity after steering.",
        "",
        "## Regeneration",
        "",
        "```bash",
        "python plotting/build_qwen_image_shows_diagnostic_alternatives.py",
        "```",
    ]
    note.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return note


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    by_condition = load_rows(args.csv)
    base_data = arrays(by_condition)
    data_by_lambda = {lam: arrays(by_condition, lam) for lam in LAMBDAS}
    stats = summarize_option_stats(data_by_lambda)

    paths: dict[str, Path] = {}
    paths["option1"] = render_option1(base_data, args.out_dir)
    for lam in LAMBDAS:
        key = str(int(lam))
        paths[f"option2_{key}"] = render_option2(data_by_lambda[lam], lam, args.out_dir)
        paths[f"option3_{key}"] = render_option3(data_by_lambda[lam], lam, args.out_dir)
        paths[f"option4_{key}"] = render_option4(data_by_lambda[lam], lam, args.out_dir)

    stats_path = args.out_dir / "qwen_image_shows_diagnostic_alternatives_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    note_path = write_note(args.out_dir, paths, stats)

    print(
        json.dumps(
            {
                "recommended": paths["option2_20"].relative_to(REPO_ROOT).as_posix(),
                "note": note_path.relative_to(REPO_ROOT).as_posix(),
                "stats": stats_path.relative_to(REPO_ROOT).as_posix(),
                "figures": {key: path.relative_to(REPO_ROOT).as_posix() for key, path in paths.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

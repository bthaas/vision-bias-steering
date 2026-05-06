#!/usr/bin/env python3
"""Build the standardized Qwen Image Shows comparison figure and summary CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_qwen_base_image_shows import (
    DEFAULT_RESULTS_ROOT,
    DEGENERATION_CAP_RATE,
    MODEL_ORDER,
    MODEL_SPECS,
    PROMPT_INSTRUCTION_TEMPLATE,
    PROMPT_PREFIX,
    pick_best_row,
)

DEFAULT_PNG = "qwen_base_image_shows_ratio_curves.png"
DEFAULT_CSV = "qwen_base_image_shows_summary.csv"
DEFAULT_NOTES = "qwen_base_image_shows_run_notes.md"

COLORS = {
    "qwen18b_chat": "#4C78A8",
    "qwen25_3b_base": "#54A24B",
    "qwen25_7b_base": "#E45756",
}

MARKERS = {
    "qwen18b_chat": "o",
    "qwen25_3b_base": "s",
    "qwen25_7b_base": "^",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-png", default=DEFAULT_PNG)
    parser.add_argument("--output-csv", default=DEFAULT_CSV)
    parser.add_argument("--output-notes", default=DEFAULT_NOTES)
    parser.add_argument("--degeneration-cap-rate", type=float, default=DEGENERATION_CAP_RATE)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Plot and summarize available models instead of requiring all three.",
    )
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_results(results_root: Path, allow_missing: bool) -> dict[str, dict]:
    loaded = {}
    missing = []
    for key in MODEL_ORDER:
        path = results_root / key / "results.json"
        if path.exists():
            loaded[key] = load_json(path)
        else:
            missing.append(path)
    if missing and not allow_missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(f"Missing required result files:\n{formatted}")
    return loaded


def row_at_lambda(rows: list[dict], coeff: int) -> dict | None:
    return next((row for row in rows if int(row["lambda"]) == int(coeff)), None)


def summarize_model(key: str, payload: dict, degeneration_cap_rate: float) -> dict:
    rows = sorted(payload["rows"], key=lambda row: int(row["lambda"]))
    lambda0 = row_at_lambda(rows, 0)
    best = pick_best_row(rows, degeneration_cap_rate)
    if best is None:
        raise ValueError(f"No rows available for {key}")
    if lambda0 is None:
        raise ValueError(f"No lambda=0 row available for {key}")
    n_outputs = int(best["n_outputs"])
    deg_count = int(best["degenerate_or_repetitive_count"])
    return {
        "model_name": payload["model_name"],
        "exact_hf_model_name": payload["hf_model_name"],
        "selected_steering_layer": int(payload["selected_steering_layer"]),
        "lambda_values_tested": " ".join(str(int(row["lambda"])) for row in rows),
        "mean_normalized_spatial_ratio_lambda_0": f"{float(lambda0['mean_normalized_ratio']):.6f}",
        "best_lambda": int(best["lambda"]),
        "mean_normalized_spatial_ratio_best_lambda": f"{float(best['mean_normalized_ratio']):.6f}",
        "degeneration_count_best_lambda": deg_count,
        "degeneration_rate_best_lambda": f"{(deg_count / n_outputs if n_outputs else 0.0):.6f}",
        "validation_size_used": int(payload["validation"]["n_val"]),
        "prompt_prefix_used": payload["prompt"]["output_prefix"],
        "prompt_instruction_template": payload["prompt"]["instruction_template"],
        "generation_settings": payload["settings"]["generation"],
        "validation_path": payload["validation"]["path"],
        "validation_selection": payload["validation"]["selection"],
        "caption_sha256": payload["validation"]["caption_sha256"],
        "result_status": payload.get("status", "unknown"),
    }


def write_csv(path: Path, summaries: list[dict]) -> None:
    fieldnames = [
        "model_name",
        "exact_hf_model_name",
        "selected_steering_layer",
        "lambda_values_tested",
        "mean_normalized_spatial_ratio_lambda_0",
        "best_lambda",
        "mean_normalized_spatial_ratio_best_lambda",
        "degeneration_count_best_lambda",
        "degeneration_rate_best_lambda",
        "validation_size_used",
        "prompt_prefix_used",
        "prompt_instruction_template",
        "generation_settings",
        "validation_path",
        "validation_selection",
        "caption_sha256",
        "result_status",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def build_plot(path: Path, loaded: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(10.6, 6.0))

    for key in MODEL_ORDER:
        if key not in loaded:
            continue
        payload = loaded[key]
        rows = sorted(payload["rows"], key=lambda row: int(row["lambda"]))
        xs = np.array([int(row["lambda"]) for row in rows], dtype=float)
        ys = np.array([float(row["mean_normalized_ratio"]) for row in rows], dtype=float)
        ax.plot(
            xs,
            ys,
            color=COLORS[key],
            marker=MARKERS[key],
            linewidth=2.4,
            markersize=5.2,
            label=payload["model_name"],
        )

    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.1, zorder=0)
    ax.grid(True, color="#999999", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Steering coefficient (lambda)", fontsize=13)
    ax.set_ylabel("Mean normalized spatial ratio", fontsize=13)
    ax.set_title('Standardized Qwen comparison: "The image shows"', fontsize=14)
    ax.legend(
        loc="best",
        frameon=True,
        facecolor="white",
        edgecolor="#d0d0d0",
        framealpha=0.95,
        fontsize=11,
    )

    all_y = [
        float(row["mean_normalized_ratio"])
        for payload in loaded.values()
        for row in payload["rows"]
    ]
    if all_y:
        y_min = min(all_y + [0.0])
        y_max = max(all_y + [0.0])
        pad = max(0.08, 0.12 * (y_max - y_min))
        ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_xlim(-62, 62)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_notes(
    path: Path,
    results_root: Path,
    figure_path: Path,
    csv_path: Path,
    summaries: list[dict],
    degeneration_cap_rate: float,
) -> None:
    lines = [
        "# Qwen Base Image Shows Run Notes",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Standardized setup",
        "",
        f"- Prompt instruction: `{PROMPT_INSTRUCTION_TEMPLATE}`",
        f"- Continuation prefix: `{PROMPT_PREFIX}`",
        "- Prompt serialization: `model.apply_chat_template(..., output_prefix=\"The image shows\")`",
        "- Generation: greedy, `do_sample=False`, full-sequence steering across all generated tokens",
        "- Metric: continuation-level normalized spatial ratio",
        "- Degeneration detector: repetition/TTR heuristic used by the local coherence-frontier scripts",
        f"- Best-lambda cap: degeneration rate <= {degeneration_cap_rate:.0%}; falls back to max ratio only if no row satisfies the cap",
        "",
        "## Outputs",
        "",
        f"- Figure: `{figure_path}`",
        f"- Summary CSV: `{csv_path}`",
        f"- Results root: `{results_root}`",
        "",
        "## Rivanna commands",
        "",
        "Run from the repo root on Rivanna:",
        "",
        "```bash",
        "bash experiments/rivanna/slurm/submit_qwen_base_image_shows.sh",
        "```",
        "",
        "Or run the steps manually:",
        "",
        "```bash",
        "python experiments/rivanna/run_qwen_base_image_shows.py verify-local \\",
        "  --results-root $SCRATCH/results/qwen_base_image_shows",
        "",
        "python experiments/rivanna/run_qwen_base_image_shows.py run-model \\",
        "  --model-key qwen25_3b_base \\",
        "  --results-root $SCRATCH/results/qwen_base_image_shows",
        "",
        "python experiments/rivanna/run_qwen_base_image_shows.py run-model \\",
        "  --model-key qwen25_7b_base \\",
        "  --results-root $SCRATCH/results/qwen_base_image_shows",
        "",
        "python experiments/rivanna/build_qwen_base_image_shows_outputs.py \\",
        "  --results-root $SCRATCH/results/qwen_base_image_shows",
        "```",
        "",
        "## Summary rows",
        "",
        "| Model | HF model | Layer | lambda=0 ratio | Best lambda | Best ratio | Degeneration at best | n_val |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['model_name']} | `{row['exact_hf_model_name']}` | {row['selected_steering_layer']} | "
            f"{row['mean_normalized_spatial_ratio_lambda_0']} | {row['best_lambda']} | "
            f"{row['mean_normalized_spatial_ratio_best_lambda']} | "
            f"{row['degeneration_count_best_lambda']} ({float(row['degeneration_rate_best_lambda']):.1%}) | "
            f"{row['validation_size_used']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_root = args.results_root.resolve()
    output_dir = (args.output_dir.resolve() if args.output_dir else results_root)
    figure_path = output_dir / args.output_png
    csv_path = output_dir / args.output_csv
    notes_path = output_dir / args.output_notes

    loaded = load_results(results_root, args.allow_missing)
    summaries = [
        summarize_model(key, loaded[key], args.degeneration_cap_rate)
        for key in MODEL_ORDER
        if key in loaded
    ]
    write_csv(csv_path, summaries)
    build_plot(figure_path, loaded)
    write_notes(notes_path, results_root, figure_path, csv_path, summaries, args.degeneration_cap_rate)

    print(figure_path)
    print(csv_path)
    print(notes_path)


if __name__ == "__main__":
    main()

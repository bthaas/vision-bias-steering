#!/usr/bin/env python3
"""Build paper-ready cross-model assets from Rivanna results.

Expected inputs:
- experiments/rivanna/results/<model_slug>/results.json

Outputs:
- experiments/rivanna/PAPER_SUMMARY.json
- experiments/rivanna/PAPER_SUMMARY.md
- paper/figures/rivanna_model_frontiers.png

The script is safe to run before all results exist. Missing models are recorded
as null entries so the paper placeholders can be filled systematically later.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPECTED_MODELS = [
    ("qwen25_3b", "Qwen2.5-3B-Instruct", 3),
    ("qwen25_7b", "Qwen2.5-7B-Instruct", 7),
    ("qwen25_14b", "Qwen2.5-14B-Instruct", 14),
    ("qwen25_3b_base", "Qwen2.5-3B-Base", 3),
    ("qwen25_7b_base", "Qwen2.5-7B-Base", 7),
    ("qwen18b_base", "Qwen-1.8B-Base", 1.8),
]

PLOT_ORDER = [
    "qwen18b_chat_local",
    "qwen18b_base",
    "qwen25_3b",
    "qwen25_3b_base",
    "qwen25_7b",
    "qwen25_7b_base",
    "qwen25_14b",
]

SHORT_LABELS = {
    "qwen18b_chat_local": "1.8B chat",
    "qwen18b_base": "1.8B base",
    "qwen25_3b": "3B instr.",
    "qwen25_3b_base": "3B base",
    "qwen25_7b": "7B instr.",
    "qwen25_7b_base": "7B base",
    "qwen25_14b": "14B instr.",
}

LOCAL_QWEN_REFERENCE = {
    "slug": "qwen18b_chat_local",
    "display_name": "Qwen-1.8B-Chat (local)",
    "size_b": 1.8,
    "best_layer": 11,
    "baseline_rms": 0.9670316353130448,
    "full_steering": {
        "coeff": -20,
        "rms": 0.6450070548425724,
        "reduction_pct": 33.30031497534489,
    },
    "one_token": {
        "coeff": -50,
        "rms": 0.31671083785434884,
        "reduction_pct": 67.24917507462679,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Build paper-ready assets from Rivanna results.")
    parser.add_argument(
        "--results-dir",
        default=str(Path(__file__).parent / "results"),
        help="Directory containing per-model result subdirectories.",
    )
    parser.add_argument(
        "--output-json",
        default=str(Path(__file__).parent / "PAPER_SUMMARY.json"),
        help="Path for machine-readable paper summary JSON.",
    )
    parser.add_argument(
        "--output-md",
        default=str(Path(__file__).parent / "PAPER_SUMMARY.md"),
        help="Path for human-readable paper summary markdown.",
    )
    parser.add_argument(
        "--output-png",
        default=str(Path(__file__).resolve().parents[2] / "paper" / "figures" / "rivanna_model_frontiers.png"),
        help="Path for the cross-model frontier plot.",
    )
    parser.add_argument(
        "--include-local-qwen",
        action="store_true",
        help="Include the local Qwen-1.8B-chat reference in the output summary and plot.",
    )
    return parser.parse_args()


def find_coherence_frontier(rows: list[dict]) -> dict | None:
    coherent = [row for row in rows if row.get("coherence") == "coherent"]
    if not coherent:
        return None
    return max(coherent, key=lambda row: row["reduction_pct"])


def find_best_one_token(tok_results: dict) -> dict | None:
    rows = tok_results.get("1", []) if tok_results else []
    usable = [row for row in rows if row.get("coherence") in ("coherent", "partial")]
    if not usable:
        return None
    return max(usable, key=lambda row: row["reduction_pct"])


def summarize_result(slug: str, display_name: str, size_b: float, path: Path) -> dict:
    if not path.exists():
        return {
            "slug": slug,
            "display_name": display_name,
            "size_b": size_b,
            "available": False,
            "best_layer": None,
            "baseline_rms": None,
            "full_steering": None,
            "one_token": None,
        }

    data = json.loads(path.read_text())
    full = data.get("coherence_frontier") or find_coherence_frontier(data.get("sweep_results", []))
    one_token = data.get("best_1tok") or find_best_one_token(data.get("tok_results", {}))
    return {
        "slug": slug,
        "display_name": display_name,
        "size_b": size_b,
        "available": True,
        "best_layer": data.get("best_layer"),
        "baseline_rms": data.get("baseline_rms"),
        "full_steering": full,
        "one_token": one_token,
    }


def build_plot(summary_rows: list[dict], output_png: Path) -> bool:
    rows = [row for row in summary_rows if row.get("available")]
    rows = [row for row in rows if row.get("full_steering") or row.get("one_token")]
    if not rows:
        return False

    order_index = {slug: idx for idx, slug in enumerate(PLOT_ORDER)}
    rows = sorted(rows, key=lambda row: order_index.get(row["slug"], 999))
    x = list(range(len(rows)))
    labels = [SHORT_LABELS.get(row["slug"], row["display_name"]) for row in rows]
    full = [row["full_steering"]["reduction_pct"] if row.get("full_steering") else 0.0 for row in rows]
    one = [row["one_token"]["reduction_pct"] if row.get("one_token") else 0.0 for row in rows]

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    width = 0.34
    full_bars = ax.bar(
        [xi - width / 2 for xi in x],
        full,
        width=width,
        color="#7570b3",
        label="full steering frontier",
    )
    one_bars = ax.bar(
        [xi + width / 2 for xi in x],
        one,
        width=width,
        color="#1b9e77",
        label="1-token frontier",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("RMS reduction (%)")
    ax.set_title("Cross-Model Steering Frontiers")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.set_axisbelow(True)

    for xpos in (1.5, 3.5, 5.5):
        ax.axvline(x=xpos, color="#dddddd", linewidth=1.0, zorder=0)

    for bar in list(full_bars) + list(one_bars):
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#333333",
        )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def build_markdown(summary_rows: list[dict], plot_built: bool, output_png: Path) -> str:
    lines = [
        "# Paper Summary",
        "",
        "This file is generated from Rivanna result folders and is meant to feed the paper draft.",
        "",
        "| Model | Available | Best Layer | Full Steering Δ% | 1-Token Δ% |",
        "|---|---|---|---|---|",
    ]
    for row in summary_rows:
        full = row.get("full_steering") or {}
        one = row.get("one_token") or {}
        lines.append(
            f"| {row['display_name']} | "
            f"{'yes' if row['available'] else 'no'} | "
            f"{row['best_layer'] if row['best_layer'] is not None else '—'} | "
            f"{full.get('reduction_pct', '—') if full else '—'} | "
            f"{one.get('reduction_pct', '—') if one else '—'} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- Missing rows indicate expected models whose `results.json` files are not yet available locally.",
        "- The summary is intended for paper insertion rather than exhaustive experiment logging.",
    ]
    if plot_built:
        lines += [
            "",
            f"- Cross-model figure saved to `{output_png}`.",
        ]
    return "\n".join(lines)


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_png = Path(args.output_png)

    summary_rows = []
    if args.include_local_qwen:
        summary_rows.append(dict(LOCAL_QWEN_REFERENCE, available=True))

    for slug, display_name, size_b in EXPECTED_MODELS:
        result_path = results_dir / slug / "results.json"
        summary_rows.append(summarize_result(slug, display_name, size_b, result_path))

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"models": summary_rows}, indent=2))

    plot_built = build_plot(summary_rows, output_png)
    output_md.write_text(build_markdown(summary_rows, plot_built, output_png))

    print(f"Saved JSON summary: {output_json}")
    print(f"Saved markdown summary: {output_md}")
    if plot_built:
        print(f"Saved plot: {output_png}")
    else:
        print("No plot generated because no model result files were available.")


if __name__ == "__main__":
    main()

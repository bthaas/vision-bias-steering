#!/usr/bin/env python3
"""Render a presentation-friendly prompt report from a saved selection JSON.

This re-evaluates only the selected prompts, then writes:
- an HTML report with per-subplot zoomed y-axes (0 to local max)
- a PNG image export for sharing
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bias_steering.steering import load_model
from bias_steering.steering.steering_utils import get_target_token_ids
from plotting.beam_selected_prompt_report import (
    continuation_multi_token_curve_beam_batched,
    teacher_forced_multi_token_curve_beam_batched,
)
from plotting.master_prompt_experiments import (
    continuation_multi_token_curve_greedy,
    load_target_words,
    teacher_forced_multi_token_curve,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Render a presentation-friendly selected-prompt report.")
    parser.add_argument(
        "--input_json",
        required=True,
        help="Path to beam_selected_prompt_report*.json",
    )
    parser.add_argument(
        "--output_html",
        required=True,
    )
    parser.add_argument(
        "--output_png",
        required=True,
    )
    parser.add_argument(
        "--title",
        default="Selected prompt report: greedy vs beam",
    )
    return parser.parse_args()


def wrap_prompt(text: str, width: int = 70) -> str:
    lines = []
    for raw_line in text.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        wrapped = textwrap.wrap(raw_line, width=width, break_long_words=False, break_on_hyphens=False)
        lines.extend(wrapped or [""])
    return "<br>".join(lines)


def nice_upper_bound(local_max: float) -> float:
    target = max(0.005, float(local_max) * 1.15)
    if target <= 0.05:
        step = 0.005
    elif target <= 0.2:
        step = 0.01
    elif target <= 0.5:
        step = 0.02
    else:
        step = 0.05
    upper = math.ceil(target / step) * step
    return min(1.0, upper)


def axis_name(row: int, col: int, ncols: int) -> tuple[str, str]:
    idx = (row - 1) * ncols + col
    xname = "xaxis" if idx == 1 else f"xaxis{idx}"
    yname = "yaxis" if idx == 1 else f"yaxis{idx}"
    return xname, yname


def plot_block(fig, row, col, coeffs, pos_curve, neg_curve, showlegend):
    fig.add_trace(
        go.Scatter(
            x=coeffs,
            y=neg_curve,
            mode="lines+markers",
            name="descriptive",
            line=dict(color="#d95f02", width=2),
            marker=dict(size=6),
            showlegend=showlegend,
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=coeffs,
            y=pos_curve,
            mode="lines+markers",
            name="spatial",
            line=dict(color="#1b9e77", width=2),
            marker=dict(size=6),
            showlegend=showlegend,
        ),
        row=row,
        col=col,
    )
    fig.add_vline(x=0, line_dash="solid", line_color="black", row=row, col=col)
    ymax = nice_upper_bound(max(max(pos_curve), max(neg_curve)))
    fig.update_yaxes(
        title_text="Tracked prob (%)",
        range=[0.0, ymax],
        tickformat=".0%",
        row=row,
        col=col,
    )
    fig.update_xaxes(title_text="Steering coeff (lambda)", row=row, col=col)


def main():
    args = parse_args()
    input_json = Path(args.input_json)
    output_html = Path(args.output_html)
    output_png = Path(args.output_png)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    data = json.loads(input_json.read_text())
    artifact_dir = Path(data["artifact_dir"])
    coeffs = data["coeffs"]
    layer = int(data["layer"])
    multi_tokens = int(data["multi_tokens"])
    beam_width = int(data["beam_width"])
    beam_top_k = int(data["beam_top_k"])
    constrained = bool(data["constrained"])
    model_name = data["model_name"]

    multi_prompts = [case["multi_prompt"] for case in data["cases"]]
    fill_prompts = [case["fill_prompt"] for case in data["cases"]]
    multi_labels = [
        f"8-token prompt ({case.get('multi_template_id', 'template')}):\n{case['multi_prompt']}"
        for case in data["cases"]
    ]
    fill_labels = [f"fill-in prompt:\n{case['fill_prompt']}" for case in data["cases"]]

    model = load_model(model_name)
    candidate_vectors = torch.load(artifact_dir / "activations/candidate_vectors.pt")
    steering_vec = model.set_dtype(candidate_vectors[layer])

    words = load_target_words(target_concept="vision")
    pos_ids = get_target_token_ids(model.tokenizer, words["spatial"])
    neg_ids = get_target_token_ids(model.tokenizer, words["descriptive"])
    overlap = set(pos_ids).intersection(set(neg_ids))
    if overlap:
        pos_ids = [x for x in pos_ids if x not in overlap]
        neg_ids = [x for x in neg_ids if x not in overlap]

    multi_g_pos, multi_g_neg = teacher_forced_multi_token_curve(
        model=model,
        prompts=multi_prompts,
        layer=layer,
        steering_vec=steering_vec,
        coeffs=coeffs,
        pos_ids=pos_ids,
        neg_ids=neg_ids,
        n_tokens=multi_tokens,
        constrained=constrained,
    )
    multi_b_pos, multi_b_neg = teacher_forced_multi_token_curve_beam_batched(
        model=model,
        prompts=multi_prompts,
        layer=layer,
        steering_vec=steering_vec,
        coeffs=coeffs,
        pos_ids=pos_ids,
        neg_ids=neg_ids,
        n_tokens=multi_tokens,
        constrained=constrained,
        beam_width=beam_width,
        beam_top_k=beam_top_k,
    )
    fill_g_pos, fill_g_neg = continuation_multi_token_curve_greedy(
        model=model,
        prompts=fill_prompts,
        layer=layer,
        steering_vec=steering_vec,
        coeffs=coeffs,
        pos_ids=pos_ids,
        neg_ids=neg_ids,
        n_tokens=multi_tokens,
        constrained=constrained,
    )
    fill_b_pos, fill_b_neg = continuation_multi_token_curve_beam_batched(
        model=model,
        prompts=fill_prompts,
        layer=layer,
        steering_vec=steering_vec,
        coeffs=coeffs,
        pos_ids=pos_ids,
        neg_ids=neg_ids,
        n_tokens=multi_tokens,
        constrained=constrained,
        beam_width=beam_width,
        beam_top_k=beam_top_k,
    )

    ncases = len(data["cases"])
    fig = make_subplots(
        rows=ncases,
        cols=4,
        shared_xaxes=False,
        vertical_spacing=0.08,
        horizontal_spacing=0.04,
        subplot_titles=[
            label
            for _ in range(ncases)
            for label in ("8-token greedy", "8-token beam", "fill-in greedy", "fill-in beam")
        ],
    )

    for i in range(ncases):
        row = i + 1
        plot_block(fig, row, 1, coeffs, multi_g_pos[i], multi_g_neg[i], showlegend=(i == 0))
        plot_block(fig, row, 2, coeffs, multi_b_pos[i], multi_b_neg[i], showlegend=False)
        plot_block(fig, row, 3, coeffs, fill_g_pos[i], fill_g_neg[i], showlegend=False)
        plot_block(fig, row, 4, coeffs, fill_b_pos[i], fill_b_neg[i], showlegend=False)

    fig.update_layout(
        title=args.title,
        template="plotly_white",
        width=2600,
        height=max(2200, 460 * ncases),
        margin=dict(t=170, b=80, l=80, r=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0.0),
    )

    # Add wrapped prompt labels above the left and right halves of each row.
    for i in range(ncases):
        row = i + 1
        _, yaxis_key = axis_name(row, 1, 4)
        domain = fig.layout[yaxis_key].domain
        y = min(1.0, domain[1] + 0.035)
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.12,
            y=y,
            text=wrap_prompt(multi_labels[i], width=72),
            showarrow=False,
            align="left",
            xanchor="left",
            yanchor="bottom",
            font=dict(size=11),
            bgcolor="rgba(245,245,245,0.92)",
            bordercolor="rgba(180,180,180,0.9)",
            borderwidth=1,
            borderpad=4,
        )
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.62,
            y=y,
            text=wrap_prompt(fill_labels[i], width=72),
            showarrow=False,
            align="left",
            xanchor="left",
            yanchor="bottom",
            font=dict(size=11),
            bgcolor="rgba(245,245,245,0.92)",
            bordercolor="rgba(180,180,180,0.9)",
            borderwidth=1,
            borderpad=4,
        )

    fig.write_html(str(output_html))
    fig.write_image(str(output_png), scale=2)
    print(f"Saved HTML: {output_html}")
    print(f"Saved PNG: {output_png}")


if __name__ == "__main__":
    main()

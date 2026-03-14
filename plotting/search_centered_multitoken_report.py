#!/usr/bin/env python3
"""Search for centered 8-token prompts only, then compare greedy vs beam.

This script ignores fill-in prompts entirely. It broadens the template pool,
prescreens candidates for centered behavior with a cheap next-token pass, then
evaluates a shortlist under greedy and beam on the full 8-token metric.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import textwrap
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bias_steering.steering import load_model
from bias_steering.steering.steering_utils import get_target_token_ids
from plotting.beam_selected_prompt_report import (
    MULTI_TOKEN_TEMPLATE_ROWS,
    build_multi_token_candidates,
    coeffs_from_args,
    select_centered_indices_with_greedy_guard,
    teacher_forced_multi_token_curve_beam_batched,
)
from plotting.master_prompt_experiments import (
    _resolve_beam_compare_coeff,
    _scenario_delta_metrics,
    load_target_words,
    next_token_class_prob_curve,
    summarize_curve_quality,
    teacher_forced_multi_token_curve,
)


EXTRA_MULTI_TEMPLATE_ROWS = [
    {
        "template_id": "natural_continuation",
        "template": "Natural continuation:\nCaption: {caption}\nThe image shows",
    },
    {
        "template_id": "concrete_detail",
        "template": "Caption: {caption}\nContinue with one concrete visual detail: The image shows",
    },
    {
        "template_id": "one_visual_detail",
        "template": "Caption: {caption}\nOne visual detail: The image shows",
    },
    {
        "template_id": "continue_caption_detail",
        "template": "Continue this caption with one natural detail.\nCaption: {caption}\nThe image shows",
    },
    {
        "template_id": "based_on_scene",
        "template": "Based on the scene: {caption}\nThe image shows",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Search centered 8-token prompts only.")
    parser.add_argument("--model_name", default="gpt2")
    parser.add_argument("--artifact_dir", default="runs_vision/gpt2")
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--min_coeff", type=float, default=-160.0)
    parser.add_argument("--max_coeff", type=float, default=160.0)
    parser.add_argument("--increment", type=float, default=80.0)
    parser.add_argument("--multi_tokens", type=int, default=8)
    parser.add_argument("--beam_width", type=int, default=4)
    parser.add_argument("--beam_top_k", type=int, default=8)
    parser.add_argument("--beam_compare_coeff", type=float, default=None)
    parser.add_argument("--captions_limit", type=int, default=0)
    parser.add_argument("--num_cases", type=int, default=5)
    parser.add_argument("--constrained", action="store_true")
    parser.add_argument("--template_seed", type=int, default=4238)
    parser.add_argument("--multi_template_limit", type=int, default=15)
    parser.add_argument("--prescreen_top_k", type=int, default=500)
    parser.add_argument("--beam_shortlist_k", type=int, default=150)
    parser.add_argument("--max_abs_cross_coeff", type=float, default=60.0)
    parser.add_argument("--max_near_zero_gap", type=float, default=0.12)
    parser.add_argument("--min_greedy_full_swing", type=float, default=0.04)
    parser.add_argument("--min_greedy_directional_consistency", type=float, default=0.5)
    parser.add_argument(
        "--output_json",
        default="runs_vision/gpt2/validation/centered_multitoken_report.json",
    )
    parser.add_argument(
        "--output_html",
        default="runs_vision/gpt2/validation/centered_multitoken_report.html",
    )
    parser.add_argument(
        "--output_png",
        default="runs_vision/gpt2/validation/centered_multitoken_report.png",
    )
    return parser.parse_args()


def wrap_prompt(text: str, width: int = 74) -> str:
    lines = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        lines.extend(textwrap.wrap(raw, width=width, break_long_words=False, break_on_hyphens=False))
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
    return min(1.0, math.ceil(target / step) * step)


def prescreen_sort_key(row: dict):
    return (
        abs(row["cross_coeff"]),
        row["near_zero_gap"],
        -row["full_swing"],
        -row["directional_consistency"],
        row["wrong_side_penalty"],
    )


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
    fig.update_yaxes(title_text="Tracked prob (%)", range=[0.0, ymax], tickformat=".0%", row=row, col=col)
    fig.update_xaxes(title_text="Steering coeff (lambda)", row=row, col=col)


def main():
    args = parse_args()
    torch.set_grad_enabled(False)
    random.seed(args.template_seed)
    np.random.seed(args.template_seed)

    artifact_dir = Path(args.artifact_dir)
    output_json = Path(args.output_json)
    output_html = Path(args.output_html)
    output_png = Path(args.output_png)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model_name)
    candidate_vectors = torch.load(artifact_dir / "activations/candidate_vectors.pt")
    steering_vec = model.set_dtype(candidate_vectors[int(args.layer)])

    words = load_target_words(target_concept="vision")
    pos_ids = get_target_token_ids(model.tokenizer, words["spatial"])
    neg_ids = get_target_token_ids(model.tokenizer, words["descriptive"])
    overlap = set(pos_ids).intersection(set(neg_ids))
    if overlap:
        pos_ids = [x for x in pos_ids if x not in overlap]
        neg_ids = [x for x in neg_ids if x not in overlap]

    coeffs = coeffs_from_args(args)
    prescreen_coeffs = [float(args.min_coeff), 0.0, float(args.max_coeff)]
    compare_coeff = _resolve_beam_compare_coeff(artifact_dir, args.beam_compare_coeff)

    rows = json.loads((artifact_dir / "datasplits/val.json").read_text())
    captions = [x["text"] for x in rows]
    if args.captions_limit > 0:
        captions = captions[: args.captions_limit]

    template_rows = (MULTI_TOKEN_TEMPLATE_ROWS + EXTRA_MULTI_TEMPLATE_ROWS)[: max(1, int(args.multi_template_limit))]
    candidates = build_multi_token_candidates(captions, template_rows)
    prompts = [x["prompt"] for x in candidates]

    print(f"[1/4] Prescreening {len(prompts)} multi-token prompt candidates...", flush=True)
    screen_pos, screen_neg = next_token_class_prob_curve(
        model=model,
        prompts=prompts,
        layer=args.layer,
        steering_vec=steering_vec,
        coeffs=prescreen_coeffs,
        pos_ids=pos_ids,
        neg_ids=neg_ids,
        constrained=args.constrained,
    )
    prescreen_rows = []
    for i in range(len(candidates)):
        q = summarize_curve_quality(screen_pos[i], screen_neg[i], prescreen_coeffs)
        prescreen_rows.append({"idx": i, **q, **candidates[i]})
    prescreen_rows = sorted(prescreen_rows, key=prescreen_sort_key)
    prescreen_rows = prescreen_rows[: min(len(prescreen_rows), max(args.prescreen_top_k, args.beam_shortlist_k, args.num_cases))]
    short_prompts = [x["prompt"] for x in prescreen_rows]

    print(f"[2/4] Beam rescoring {len(short_prompts)} prescreened prompts across full coeff grid...", flush=True)
    beam_pos, beam_neg = teacher_forced_multi_token_curve_beam_batched(
        model=model,
        prompts=short_prompts,
        layer=args.layer,
        steering_vec=steering_vec,
        coeffs=coeffs,
        pos_ids=pos_ids,
        neg_ids=neg_ids,
        n_tokens=args.multi_tokens,
        constrained=args.constrained,
        beam_width=args.beam_width,
        beam_top_k=args.beam_top_k,
    )
    greedy_pos, greedy_neg = teacher_forced_multi_token_curve(
        model=model,
        prompts=short_prompts,
        layer=args.layer,
        steering_vec=steering_vec,
        coeffs=coeffs,
        pos_ids=pos_ids,
        neg_ids=neg_ids,
        n_tokens=args.multi_tokens,
        constrained=args.constrained,
    )

    selected_rows = select_centered_indices_with_greedy_guard(
        beam_pos_mat=beam_pos,
        beam_neg_mat=beam_neg,
        greedy_pos_mat=greedy_pos,
        greedy_neg_mat=greedy_neg,
        coeffs=coeffs,
        compare_coeff=compare_coeff,
        num_cases=args.num_cases,
        max_abs_cross_coeff=args.max_abs_cross_coeff,
        max_near_zero_gap=args.max_near_zero_gap,
        min_greedy_full_swing=args.min_greedy_full_swing,
        min_greedy_directional_consistency=args.min_greedy_directional_consistency,
        unique_caption_keys=[x["caption"] for x in prescreen_rows],
    )

    actual_cases = min(args.num_cases, len(selected_rows))
    if actual_cases <= 0:
        raise RuntimeError("No 8-token prompt cases passed selection.")
    selected_rows = selected_rows[:actual_cases]
    selected_meta = [prescreen_rows[x["idx"]] for x in selected_rows]
    selected_prompts = [x["prompt"] for x in selected_meta]

    print(f"[3/4] Re-evaluating final {actual_cases} prompts under greedy and beam...", flush=True)
    final_g_pos, final_g_neg = teacher_forced_multi_token_curve(
        model=model,
        prompts=selected_prompts,
        layer=args.layer,
        steering_vec=steering_vec,
        coeffs=coeffs,
        pos_ids=pos_ids,
        neg_ids=neg_ids,
        n_tokens=args.multi_tokens,
        constrained=args.constrained,
    )
    final_b_pos, final_b_neg = teacher_forced_multi_token_curve_beam_batched(
        model=model,
        prompts=selected_prompts,
        layer=args.layer,
        steering_vec=steering_vec,
        coeffs=coeffs,
        pos_ids=pos_ids,
        neg_ids=neg_ids,
        n_tokens=args.multi_tokens,
        constrained=args.constrained,
        beam_width=args.beam_width,
        beam_top_k=args.beam_top_k,
    )

    print(f"[4/4] Writing HTML, PNG, and JSON...", flush=True)
    fig = make_subplots(
        rows=actual_cases,
        cols=2,
        vertical_spacing=0.10,
        horizontal_spacing=0.07,
        subplot_titles=[label for _ in range(actual_cases) for label in ("8-token greedy", "8-token beam")],
    )
    cases = []
    for i in range(actual_cases):
        row = i + 1
        plot_block(fig, row, 1, coeffs, final_g_pos[i], final_g_neg[i], showlegend=(i == 0))
        plot_block(fig, row, 2, coeffs, final_b_pos[i], final_b_neg[i], showlegend=False)
        domain = fig.layout["yaxis" if row == 1 else f"yaxis{((row - 1) * 2) + 1}"].domain
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.03,
            y=min(1.0, domain[1] + 0.04),
            text=wrap_prompt(f"{selected_meta[i]['template_id']}: {selected_prompts[i]}"),
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
        cases.append(
            {
                "case": row,
                "caption": selected_meta[i]["caption"],
                "template_id": selected_meta[i]["template_id"],
                "prompt": selected_prompts[i],
                "selection": {
                    "hard_ok": bool(selected_rows[i]["hard_ok"]),
                    "beam_cross_coeff": float(selected_rows[i]["beam_cross_coeff"]),
                    "beam_near_zero_gap": float(selected_rows[i]["beam_near_zero_gap"]),
                    "greedy_full_swing": float(selected_rows[i]["greedy_full_swing"]),
                    "greedy_directional_consistency": float(selected_rows[i]["greedy_directional_consistency"]),
                },
                "greedy_metrics": _scenario_delta_metrics(
                    np.asarray([final_g_pos[i]]), np.asarray([final_g_neg[i]]), coeffs, compare_coeff
                ),
                "beam_metrics": _scenario_delta_metrics(
                    np.asarray([final_b_pos[i]]), np.asarray([final_b_neg[i]]), coeffs, compare_coeff
                ),
            }
        )

    fig.update_layout(
        title="Centered 8-token prompt search: greedy vs beam",
        template="plotly_white",
        width=1800,
        height=max(1800, 430 * actual_cases),
        margin=dict(t=170, b=80, l=80, r=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0.0),
    )
    fig.write_html(str(output_html))
    fig.write_image(str(output_png), scale=2)

    output_json.write_text(
        json.dumps(
            {
                "model_name": args.model_name,
                "artifact_dir": str(artifact_dir),
                "layer": int(args.layer),
                "coeffs": coeffs,
                "compare_coeff_requested": float(compare_coeff),
                "compare_coeff_used": float(min(coeffs, key=lambda x: abs(x - compare_coeff))),
                "multi_tokens": int(args.multi_tokens),
                "beam_width": int(args.beam_width),
                "beam_top_k": int(args.beam_top_k),
                "constrained": bool(args.constrained),
                "captions_limit": int(args.captions_limit),
                "num_captions": int(len(captions)),
                "multi_template_limit": int(args.multi_template_limit),
                "multi_template_ids": [x["template_id"] for x in template_rows],
                "prescreen_top_k": int(args.prescreen_top_k),
                "beam_shortlist_k": int(args.beam_shortlist_k),
                "max_abs_cross_coeff": float(args.max_abs_cross_coeff),
                "max_near_zero_gap": float(args.max_near_zero_gap),
                "min_greedy_full_swing": float(args.min_greedy_full_swing),
                "min_greedy_directional_consistency": float(args.min_greedy_directional_consistency),
                "requested_num_cases": int(args.num_cases),
                "actual_num_cases": int(actual_cases),
                "cases": cases,
            },
            indent=2,
        )
    )

    print(f"Saved JSON: {output_json}")
    print(f"Saved HTML: {output_html}")
    print(f"Saved PNG: {output_png}")


if __name__ == "__main__":
    main()

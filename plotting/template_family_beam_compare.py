#!/usr/bin/env python3
"""Unified template-family comparison for next-token vs multi-token decoding.

This script keeps the benchmark fixed across prompt families so we can compare:
- next-token steering
- 8-token greedy continuation
- 8-token beam continuation

It uses the same caption subset, steering vector, coefficient, and token sets for
`scene_is`, `image_shows`, and `in_scene_the`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import plotly.graph_objects as go
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bias_steering.steering import load_model
from bias_steering.steering.steering_utils import get_target_token_ids
from plotting.master_prompt_experiments import (
    _resolve_beam_compare_coeff,
    _scenario_delta_metrics,
    load_target_words,
    next_token_class_prob_curve,
    teacher_forced_multi_token_curve,
    teacher_forced_multi_token_curve_beam,
)


TEMPLATE_FAMILIES = [
    ("scene_is", "Continue describing this scene:\n{caption}\nThe scene is"),
    ("image_shows", "Describe this image:\n{caption}\nThe image shows"),
    ("in_scene_the", "Describe this image:\n{caption}\nIn this scene, the"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Compare prompt families under greedy vs beam.")
    parser.add_argument("--model_name", default="gpt2")
    parser.add_argument("--artifact_dir", default="runs_vision/gpt2")
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--captions_limit", type=int, default=400)
    parser.add_argument("--multi_tokens", type=int, default=8)
    parser.add_argument("--beam_width", type=int, default=4)
    parser.add_argument("--beam_top_k", type=int, default=8)
    parser.add_argument("--compare_coeff", type=float, default=None)
    parser.add_argument("--constrained", action="store_true", help="Constrain softmax to tracked token sets.")
    parser.add_argument(
        "--output_json",
        default="runs_vision/gpt2/validation/template_family_beam_compare.json",
    )
    parser.add_argument(
        "--output_html",
        default="runs_vision/gpt2/validation/template_family_beam_compare.html",
    )
    return parser.parse_args()


def load_captions(artifact_dir: Path, limit: int) -> list[str]:
    rows = json.loads((artifact_dir / "datasplits/val.json").read_text())
    captions = [str(x["text"]).strip() for x in rows if str(x["text"]).strip()]
    if limit > 0:
        captions = captions[:limit]
    if not captions:
        raise ValueError("No captions found.")
    return captions


def build_prompts(model, captions: list[str], use_chat_template: bool = False):
    out = {}
    for name, template in TEMPLATE_FAMILIES:
        raw = [template.format(caption=caption) for caption in captions]
        out[name] = model.apply_chat_template(raw) if use_chat_template else raw
    return out


def main():
    args = parse_args()
    torch.set_grad_enabled(False)

    artifact_dir = Path(args.artifact_dir)
    output_json = Path(args.output_json)
    output_html = Path(args.output_html)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_html.parent.mkdir(parents=True, exist_ok=True)

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

    compare_coeff = _resolve_beam_compare_coeff(artifact_dir, args.compare_coeff)
    coeffs = [0.0, float(compare_coeff)]
    coeffs = sorted(set(float(x) for x in coeffs))

    captions = load_captions(artifact_dir, args.captions_limit)
    prompt_sets = build_prompts(model, captions, use_chat_template=False)

    results = {
        "model_name": args.model_name,
        "artifact_dir": str(artifact_dir),
        "layer": int(args.layer),
        "captions_limit": int(args.captions_limit),
        "num_captions": int(len(captions)),
        "multi_tokens": int(args.multi_tokens),
        "beam_width": int(args.beam_width),
        "beam_top_k": int(args.beam_top_k),
        "constrained": bool(args.constrained),
        "compare_coeff_requested": float(compare_coeff),
        "coeffs_used": coeffs,
        "templates": {},
    }

    for template_name, prompts in prompt_sets.items():
        next_pos, next_neg = next_token_class_prob_curve(
            model=model,
            prompts=prompts,
            layer=args.layer,
            steering_vec=steering_vec,
            coeffs=coeffs,
            pos_ids=pos_ids,
            neg_ids=neg_ids,
            constrained=args.constrained,
        )
        multi_g_pos, multi_g_neg = teacher_forced_multi_token_curve(
            model=model,
            prompts=prompts,
            layer=args.layer,
            steering_vec=steering_vec,
            coeffs=coeffs,
            pos_ids=pos_ids,
            neg_ids=neg_ids,
            n_tokens=args.multi_tokens,
            constrained=args.constrained,
        )
        multi_b_pos, multi_b_neg = teacher_forced_multi_token_curve_beam(
            model=model,
            prompts=prompts,
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

        next_metrics = _scenario_delta_metrics(next_pos, next_neg, coeffs, compare_coeff)
        multi_g_metrics = _scenario_delta_metrics(multi_g_pos, multi_g_neg, coeffs, compare_coeff)
        multi_b_metrics = _scenario_delta_metrics(multi_b_pos, multi_b_neg, coeffs, compare_coeff)

        results["templates"][template_name] = {
            "next_token": next_metrics,
            "multi_token_greedy": multi_g_metrics,
            "multi_token_beam": multi_b_metrics,
            "beam_minus_greedy_gap_reduction_pct": float(
                multi_b_metrics["gap_reduction_pct"] - multi_g_metrics["gap_reduction_pct"]
            ),
        }

    output_json.write_text(json.dumps(results, indent=2))

    template_order = [name for name, _ in TEMPLATE_FAMILIES]
    labels = template_order
    next_vals = [results["templates"][name]["next_token"]["gap_reduction_pct"] for name in template_order]
    greedy_vals = [results["templates"][name]["multi_token_greedy"]["gap_reduction_pct"] for name in template_order]
    beam_vals = [results["templates"][name]["multi_token_beam"]["gap_reduction_pct"] for name in template_order]

    fig = go.Figure()
    fig.add_bar(name="next-token", x=labels, y=next_vals, marker_color="#7570b3")
    fig.add_bar(name=f"{args.multi_tokens}-token greedy", x=labels, y=greedy_vals, marker_color="#d95f02")
    fig.add_bar(name=f"{args.multi_tokens}-token beam", x=labels, y=beam_vals, marker_color="#1b9e77")
    fig.update_layout(
        title="Template-family bias reduction comparison",
        template="plotly_white",
        barmode="group",
        yaxis_title="Gap reduction (%)",
        xaxis_title="Prompt family",
        width=1100,
        height=520,
    )
    fig.write_html(str(output_html))

    print(f"Saved JSON: {output_json}")
    print(f"Saved HTML: {output_html}")


if __name__ == "__main__":
    main()

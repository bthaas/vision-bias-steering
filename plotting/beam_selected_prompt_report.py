#!/usr/bin/env python3
"""Build a beam-first prompt report for 8-token mean and fill-in prompts.

Selection strategy:
- Multi-token prompt shortlist is built from a small template pool, prescreened quickly,
  then reranked with beam.
- Fill-in prompt shortlist is screened after filtering weak generic objects, then reranked
  by beam.
- Final top 5 prompts are selected from full beam curves and then evaluated under both
  greedy and beam with fixed 0-100% y-axes.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bias_steering.steering import get_intervention_func, load_model
from bias_steering.steering.steering_utils import get_target_token_ids
from plotting.master_prompt_experiments import (
    _normalize_logweights,
    _length_buckets,
    _resolve_beam_compare_coeff,
    _scenario_delta_metrics,
    _token_lengths,
    class_probs_from_logits,
    build_custom_fillin_candidates,
    build_image_shows_prompts,
    continuation_multi_token_curve_greedy,
    load_target_words,
    next_token_class_prob_curve,
    screen_custom_fillin_candidates,
    summarize_curve_quality,
    teacher_forced_multi_token_curve,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Beam-selected prompt report.")
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
    parser.add_argument("--custom_max_objects", type=int, default=5)
    parser.add_argument("--custom_screen_top_k", type=int, default=120)
    parser.add_argument("--multi_screen_top_k", type=int, default=200)
    parser.add_argument("--beam_shortlist_k", type=int, default=40)
    parser.add_argument("--multi_template_limit", type=int, default=10)
    parser.add_argument("--min_fill_object_quality", type=float, default=1.0)
    parser.add_argument("--strict_center_ratio", type=float, default=0.25)
    parser.add_argument("--strict_near_zero_gap", type=float, default=0.22)
    parser.add_argument("--strict_orientation_margin", type=float, default=0.08)
    parser.add_argument("--strict_wrong_side_max", type=float, default=0.03)
    parser.add_argument("--strict_directional_consistency", type=float, default=0.65)
    parser.add_argument("--multi_max_abs_cross_coeff", type=float, default=60.0)
    parser.add_argument("--fill_max_abs_cross_coeff", type=float, default=60.0)
    parser.add_argument("--multi_max_near_zero_gap", type=float, default=0.12)
    parser.add_argument("--fill_max_near_zero_gap", type=float, default=0.05)
    parser.add_argument("--multi_min_greedy_full_swing", type=float, default=0.04)
    parser.add_argument("--fill_min_greedy_full_swing", type=float, default=0.01)
    parser.add_argument("--min_greedy_directional_consistency", type=float, default=0.5)
    parser.add_argument(
        "--output_json",
        default="runs_vision/gpt2/validation/beam_selected_prompt_report.json",
    )
    parser.add_argument(
        "--output_html",
        default="runs_vision/gpt2/validation/beam_selected_prompt_report.html",
    )
    return parser.parse_args()


MULTI_TOKEN_TEMPLATE_ROWS = [
    {
        "template_id": "describe_image_shows",
        "template": "Describe this image:\n{caption}\nThe image shows",
    },
    {
        "template_id": "caption_image_shows",
        "template": "Caption: {caption}\nThe image shows",
    },
    {
        "template_id": "scene_image_shows",
        "template": "Scene: {caption}\nThe image shows",
    },
    {
        "template_id": "given_caption_shows",
        "template": "Given this caption: {caption}\nThe image shows",
    },
    {
        "template_id": "continue_naturally",
        "template": "Caption: {caption}\nContinue naturally: The image shows",
    },
    {
        "template_id": "short_continuation",
        "template": "Caption: {caption}\nOne short continuation: The image shows",
    },
    {
        "template_id": "salient_detail",
        "template": "Caption: {caption}\nDescribe one salient detail: The image shows",
    },
    {
        "template_id": "simple_rewrite",
        "template": "Image caption: {caption}\nThe image shows",
    },
    {
        "template_id": "visual_detail",
        "template": "Original caption: {caption}\nAdd one visual detail: The image shows",
    },
    {
        "template_id": "brief_scene",
        "template": "Brief scene description: {caption}\nThe image shows",
    },
]


WEAK_FILL_OBJECTS = {
    "area",
    "background",
    "base",
    "bunch",
    "caption",
    "couple",
    "group",
    "image",
    "kind",
    "lot",
    "middle",
    "part",
    "photo",
    "picture",
    "place",
    "scene",
    "set",
    "side",
    "space",
    "thing",
    "things",
    "top",
    "bottom",
    "front",
    "back",
    "end",
    "view",
}


def coeffs_from_args(args) -> list[float]:
    coeffs = []
    x = float(args.min_coeff)
    while x <= float(args.max_coeff) + 1e-9:
        coeffs.append(float(round(x, 10)))
        x += float(args.increment)
    if 0.0 not in coeffs:
        coeffs.append(0.0)
    return sorted(set(coeffs))


def fixed_prob_block(fig, row, col, coeffs, pos_curve, neg_curve, title_text, showlegend):
    fig.add_trace(
        go.Scatter(
            x=coeffs,
            y=neg_curve,
            mode="lines+markers",
            name="descriptive",
            line=dict(color="#d95f02"),
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
            line=dict(color="#1b9e77"),
            showlegend=showlegend,
        ),
        row=row,
        col=col,
    )
    fig.add_vline(x=0, line_dash="solid", line_color="black", row=row, col=col)
    fig.update_yaxes(title_text="Class probability (%)", range=[0.0, 1.0], tickformat=".0%", row=row, col=col)
    fig.update_xaxes(title_text="Steering coeff (lambda)", row=row, col=col)
    fig.add_annotation(
        xref=f"x{((row - 1) * 4 + col)}" if (row, col) != (1, 1) else "x",
        yref=f"y{((row - 1) * 4 + col)}" if (row, col) != (1, 1) else "y",
        x=coeffs[0],
        y=1.08,
        text=title_text,
        showarrow=False,
        xanchor="left",
        font=dict(size=10),
    )


def per_prompt_gap_reduction(pos_mat, neg_mat, coeffs, compare_coeff: float) -> np.ndarray:
    coeff_arr = np.asarray(coeffs, dtype=np.float64)
    zero_idx = int(np.argmin(np.abs(coeff_arr)))
    steer_idx = int(np.argmin(np.abs(coeff_arr - float(compare_coeff))))
    diff = np.asarray(pos_mat, dtype=np.float64) - np.asarray(neg_mat, dtype=np.float64)
    baseline = diff[:, zero_idx]
    steered = diff[:, steer_idx]
    baseline_gap = np.abs(baseline)
    steered_gap = np.abs(steered)
    out = np.full((diff.shape[0],), -1e9, dtype=np.float64)
    mask = baseline_gap > 1e-9
    out[mask] = ((baseline_gap[mask] - steered_gap[mask]) / baseline_gap[mask]) * 100.0
    return out


def single_curve_metrics(pos_curve, neg_curve, coeffs, compare_coeff: float) -> dict:
    quality = summarize_curve_quality(pos_curve, neg_curve, coeffs)
    delta = _scenario_delta_metrics(
        np.asarray([pos_curve], dtype=np.float64),
        np.asarray([neg_curve], dtype=np.float64),
        coeffs,
        compare_coeff,
    )
    return {**quality, **delta}


def _selection_sort_key(row: dict):
    return (
        0 if row["hard_ok"] else 1,
        abs(row["beam_cross_coeff"]),
        row["beam_near_zero_gap"],
        -row["beam_gap_reduction_pct"],
        -row["greedy_full_swing"],
        -row["greedy_directional_consistency"],
        -row["greedy_gap_reduction_pct"],
    )


def build_multi_token_candidates(captions: list[str], template_rows: list[dict]) -> list[dict]:
    candidates = []
    for caption in captions:
        for row in template_rows:
            candidates.append(
                {
                    "caption": caption,
                    "template_id": row["template_id"],
                    "prompt": row["template"].format(caption=caption),
                }
            )
    return candidates


def filter_fillin_candidates(candidates: list[dict], min_object_quality: float) -> list[dict]:
    filtered = []
    for row in candidates:
        obj = str(row.get("object", "")).strip().lower()
        if not obj:
            continue
        if obj in WEAK_FILL_OBJECTS:
            continue
        if row.get("object_quality", 0.0) < float(min_object_quality):
            continue
        filtered.append(row)
    return filtered


def select_centered_indices_with_greedy_guard(
    beam_pos_mat,
    beam_neg_mat,
    greedy_pos_mat,
    greedy_neg_mat,
    coeffs,
    compare_coeff: float,
    num_cases: int,
    max_abs_cross_coeff: float,
    max_near_zero_gap: float,
    min_greedy_full_swing: float,
    min_greedy_directional_consistency: float,
    unique_caption_keys: list[str] | None = None,
):
    rows = []
    for i in range(beam_pos_mat.shape[0]):
        beam = single_curve_metrics(beam_pos_mat[i], beam_neg_mat[i], coeffs, compare_coeff)
        greedy = single_curve_metrics(greedy_pos_mat[i], greedy_neg_mat[i], coeffs, compare_coeff)
        hard_ok = (
            beam["left_diff"] < 0.0
            and beam["right_diff"] > 0.0
            and abs(beam["cross_coeff"]) <= float(max_abs_cross_coeff)
            and beam["near_zero_gap"] <= float(max_near_zero_gap)
            and greedy["full_swing"] >= float(min_greedy_full_swing)
            and greedy["directional_consistency"] >= float(min_greedy_directional_consistency)
        )
        rows.append(
            {
                "idx": i,
                "hard_ok": bool(hard_ok),
                "beam_cross_coeff": float(beam["cross_coeff"]),
                "beam_near_zero_gap": float(beam["near_zero_gap"]),
                "beam_gap_reduction_pct": float(beam["gap_reduction_pct"]),
                "greedy_full_swing": float(greedy["full_swing"]),
                "greedy_directional_consistency": float(greedy["directional_consistency"]),
                "greedy_gap_reduction_pct": float(greedy["gap_reduction_pct"]),
                "unique_key": unique_caption_keys[i] if unique_caption_keys is not None else None,
            }
        )

    ranked = sorted(rows, key=_selection_sort_key)
    selected = []
    seen_keys = set()
    for row in ranked:
        if row["hard_ok"]:
            key = row["unique_key"]
            if key is not None and key in seen_keys:
                continue
            selected.append(row)
            if key is not None:
                seen_keys.add(key)
            if len(selected) >= num_cases:
                return selected

    # Graceful fallback if the hard filter leaves too few cases.
    for row in ranked:
        if row["idx"] in {x["idx"] for x in selected}:
            continue
        key = row["unique_key"]
        if key is not None and key in seen_keys:
            continue
        selected.append(row)
        if key is not None:
            seen_keys.add(key)
        if len(selected) >= num_cases:
            break
    return selected


def _iter_last_logits_chunks(model, contexts, layer, intervene_func, batch_size: int = 64):
    lengths = _token_lengths(model, contexts)
    for _, idxs in sorted(_length_buckets(lengths).items()):
        for start in range(0, len(idxs), batch_size):
            batch_idxs = idxs[start : start + batch_size]
            chunk = [contexts[i] for i in batch_idxs]
            logits = model.get_logits(chunk, layer=layer, intervene_func=intervene_func)
            yield batch_idxs, logits[:, -1, :]


def beam_reference_paths_batched(
    model,
    prompts,
    layer,
    n_tokens,
    beam_width,
    beam_top_k,
    batch_size: int = 64,
):
    beam_width = max(1, int(beam_width))
    beam_top_k = max(1, int(beam_top_k))
    n_tokens = max(0, int(n_tokens))
    beams_by_prompt = [[{"ids": [], "sum_logprob": 0.0, "context": prompt}] for prompt in prompts]
    tok_piece_cache = {}

    for _ in range(n_tokens):
        flat_contexts = []
        flat_meta = []
        for prompt_idx, beams in enumerate(beams_by_prompt):
            for beam in beams:
                flat_meta.append((prompt_idx, beam))
                flat_contexts.append(beam["context"])
        candidates_by_prompt = [[] for _ in prompts]
        for batch_idxs, logits_last in _iter_last_logits_chunks(
            model,
            flat_contexts,
            layer=layer,
            intervene_func=None,
            batch_size=batch_size,
        ):
            for local_idx in range(logits_last.shape[0]):
                flat_idx = batch_idxs[local_idx]
                prompt_idx, beam = flat_meta[flat_idx]
                log_probs = F.log_softmax(logits_last[local_idx], dim=-1)
                k = min(beam_top_k, int(log_probs.shape[-1]))
                top_log_probs, top_ids = torch.topk(log_probs, k=k)
                for t in range(k):
                    tok_id = int(top_ids[t].item())
                    tok_logprob = float(top_log_probs[t].item())
                    tok_piece = tok_piece_cache.get(tok_id)
                    if tok_piece is None:
                        tok_piece = model.tokenizer.decode([tok_id])
                        tok_piece_cache[tok_id] = tok_piece
                    candidates_by_prompt[prompt_idx].append(
                        {
                            "ids": beam["ids"] + [tok_id],
                            "sum_logprob": beam["sum_logprob"] + tok_logprob,
                            "context": beam["context"] + tok_piece,
                        }
                    )
        for prompt_idx, candidates in enumerate(candidates_by_prompt):
            candidates.sort(key=lambda x: x["sum_logprob"], reverse=True)
            beams_by_prompt[prompt_idx] = candidates[:beam_width]

    out = []
    for beams in beams_by_prompt:
        weights = _normalize_logweights([float(x["sum_logprob"]) for x in beams])
        out.append(
            [
                {
                    "ids": beam["ids"],
                    "sum_logprob": float(beam["sum_logprob"]),
                    "weight": float(weights[i]) if i < len(weights) else 0.0,
                }
                for i, beam in enumerate(beams)
            ]
        )
    return out


def teacher_forced_multi_token_curve_beam_batched(
    model,
    prompts,
    layer,
    steering_vec,
    coeffs,
    pos_ids,
    neg_ids,
    n_tokens,
    constrained: bool,
    beam_width: int,
    beam_top_k: int,
    batch_size: int = 64,
):
    n_tokens = max(0, int(n_tokens))
    ref_beams = beam_reference_paths_batched(
        model=model,
        prompts=prompts,
        layer=layer,
        n_tokens=n_tokens,
        beam_width=beam_width,
        beam_top_k=beam_top_k,
        batch_size=batch_size,
    )
    flat_prompt_idx = []
    flat_beam_ids = []
    flat_weights = []
    flat_contexts_seed = []
    tok_piece_cache = {}
    for prompt_idx, prompt in enumerate(prompts):
        for beam in ref_beams[prompt_idx]:
            flat_prompt_idx.append(prompt_idx)
            flat_beam_ids.append(beam["ids"])
            flat_weights.append(float(beam["weight"]))
            flat_contexts_seed.append(prompt)

    pos_mat = np.zeros((len(prompts), len(coeffs)), dtype=np.float64)
    neg_mat = np.zeros((len(prompts), len(coeffs)), dtype=np.float64)
    if not flat_contexts_seed:
        return pos_mat, neg_mat

    for j, coeff in enumerate(coeffs):
        intervene = get_intervention_func(steering_vec, method="constant", coeff=coeff)
        contexts = list(flat_contexts_seed)
        pos_sum = np.zeros((len(flat_contexts_seed),), dtype=np.float64)
        neg_sum = np.zeros((len(flat_contexts_seed),), dtype=np.float64)
        for t in range(n_tokens):
            for batch_idxs, logits_last in _iter_last_logits_chunks(
                model,
                contexts,
                layer=layer,
                intervene_func=intervene,
                batch_size=batch_size,
            ):
                for local_idx in range(logits_last.shape[0]):
                    idx = batch_idxs[local_idx]
                    p, n = class_probs_from_logits(logits_last[local_idx], pos_ids, neg_ids, constrained=constrained)
                    pos_sum[idx] += p
                    neg_sum[idx] += n
                    tok_id = flat_beam_ids[idx][t]
                    tok_piece = tok_piece_cache.get(tok_id)
                    if tok_piece is None:
                        tok_piece = model.tokenizer.decode([tok_id])
                        tok_piece_cache[tok_id] = tok_piece
                    contexts[idx] += tok_piece

        for idx, prompt_idx in enumerate(flat_prompt_idx):
            weight = flat_weights[idx]
            if n_tokens > 0:
                pos_mat[prompt_idx, j] += weight * (pos_sum[idx] / float(n_tokens))
                neg_mat[prompt_idx, j] += weight * (neg_sum[idx] / float(n_tokens))
    return pos_mat, neg_mat


def continuation_multi_token_curve_beam_batched(
    model,
    prompts,
    layer,
    steering_vec,
    coeffs,
    pos_ids,
    neg_ids,
    n_tokens,
    constrained: bool,
    beam_width: int,
    beam_top_k: int,
    batch_size: int = 64,
):
    beam_width = max(1, int(beam_width))
    beam_top_k = max(1, int(beam_top_k))
    n_tokens = max(0, int(n_tokens))
    pos_mat = np.zeros((len(prompts), len(coeffs)), dtype=np.float64)
    neg_mat = np.zeros((len(prompts), len(coeffs)), dtype=np.float64)
    tok_piece_cache = {}

    for j, coeff in enumerate(coeffs):
        intervene = get_intervention_func(steering_vec, method="constant", coeff=coeff)
        beams_by_prompt = [[{"context": prompt, "sum_logprob": 0.0, "pos_sum": 0.0, "neg_sum": 0.0}] for prompt in prompts]
        for _ in range(n_tokens):
            flat_contexts = []
            flat_meta = []
            for prompt_idx, beams in enumerate(beams_by_prompt):
                for beam in beams:
                    flat_meta.append((prompt_idx, beam))
                    flat_contexts.append(beam["context"])
            candidates_by_prompt = [[] for _ in prompts]
            for batch_idxs, logits_last in _iter_last_logits_chunks(
                model,
                flat_contexts,
                layer=layer,
                intervene_func=intervene,
                batch_size=batch_size,
            ):
                for local_idx in range(logits_last.shape[0]):
                    flat_idx = batch_idxs[local_idx]
                    prompt_idx, beam = flat_meta[flat_idx]
                    p, n = class_probs_from_logits(logits_last[local_idx], pos_ids, neg_ids, constrained=constrained)
                    log_probs = F.log_softmax(logits_last[local_idx], dim=-1)
                    k = min(beam_top_k, int(log_probs.shape[-1]))
                    top_log_probs, top_ids = torch.topk(log_probs, k=k)
                    for t in range(k):
                        tok_id = int(top_ids[t].item())
                        tok_logprob = float(top_log_probs[t].item())
                        tok_piece = tok_piece_cache.get(tok_id)
                        if tok_piece is None:
                            tok_piece = model.tokenizer.decode([tok_id])
                            tok_piece_cache[tok_id] = tok_piece
                        candidates_by_prompt[prompt_idx].append(
                            {
                                "context": beam["context"] + tok_piece,
                                "sum_logprob": beam["sum_logprob"] + tok_logprob,
                                "pos_sum": beam["pos_sum"] + p,
                                "neg_sum": beam["neg_sum"] + n,
                            }
                        )
            for prompt_idx, candidates in enumerate(candidates_by_prompt):
                candidates.sort(key=lambda x: x["sum_logprob"], reverse=True)
                beams_by_prompt[prompt_idx] = candidates[:beam_width]

        for prompt_idx, beams in enumerate(beams_by_prompt):
            if not beams or n_tokens <= 0:
                continue
            weights = _normalize_logweights([float(x["sum_logprob"]) for x in beams])
            pos_mat[prompt_idx, j] = float(sum(weights[b] * (beams[b]["pos_sum"] / n_tokens) for b in range(len(beams))))
            neg_mat[prompt_idx, j] = float(sum(weights[b] * (beams[b]["neg_sum"] / n_tokens) for b in range(len(beams))))
    return pos_mat, neg_mat


def main():
    args = parse_args()
    torch.set_grad_enabled(False)
    random.seed(args.template_seed)
    np.random.seed(args.template_seed)

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

    coeffs = coeffs_from_args(args)
    compare_coeff = _resolve_beam_compare_coeff(artifact_dir, args.beam_compare_coeff)
    select_coeffs = sorted(set([0.0, float(compare_coeff)]))

    rows = json.loads((artifact_dir / "datasplits/val.json").read_text())
    captions = [x["text"] for x in rows]
    if args.captions_limit > 0:
        captions = captions[: args.captions_limit]

    multi_template_rows = MULTI_TOKEN_TEMPLATE_ROWS[: max(1, int(args.multi_template_limit))]
    multi_candidates = build_multi_token_candidates(captions, multi_template_rows)
    multi_prompts = [x["prompt"] for x in multi_candidates]

    # Prescreen multi-token prompts quickly, then rerank that shortlist with beam.
    print(
        f"[1/6] Prescreening {len(multi_prompts)} multi-token prompts with next-token scoring "
        f"to top {args.multi_screen_top_k}...",
        flush=True,
    )
    multi_screen_pos, multi_screen_neg = next_token_class_prob_curve(
        model=model,
        prompts=multi_prompts,
        layer=args.layer,
        steering_vec=steering_vec,
        coeffs=select_coeffs,
        pos_ids=pos_ids,
        neg_ids=neg_ids,
        constrained=args.constrained,
    )
    multi_screen_reduction = per_prompt_gap_reduction(
        multi_screen_pos,
        multi_screen_neg,
        select_coeffs,
        compare_coeff,
    )
    multi_screen_k = min(len(multi_prompts), max(args.multi_screen_top_k, args.beam_shortlist_k, args.num_cases))
    multi_screen_idx = np.argsort(-multi_screen_reduction)[:multi_screen_k].tolist()
    multi_screen_rows = [multi_candidates[i] for i in multi_screen_idx]
    multi_screen_prompts = [x["prompt"] for x in multi_screen_rows]
    multi_screen_captions = [x["caption"] for x in multi_screen_rows]

    print(
        f"[2/6] Beam-ranking {len(multi_screen_prompts)} prescreened multi-token prompts...",
        flush=True,
    )
    multi_short_pos, multi_short_neg = teacher_forced_multi_token_curve_beam_batched(
        model=model,
        prompts=multi_screen_prompts,
        layer=args.layer,
        steering_vec=steering_vec,
        coeffs=select_coeffs,
        pos_ids=pos_ids,
        neg_ids=neg_ids,
        n_tokens=args.multi_tokens,
        constrained=args.constrained,
        beam_width=args.beam_width,
        beam_top_k=args.beam_top_k,
    )
    multi_reduction = per_prompt_gap_reduction(multi_short_pos, multi_short_neg, select_coeffs, compare_coeff)
    multi_shortlist_idx = np.argsort(-multi_reduction)[: max(args.beam_shortlist_k, args.num_cases)].tolist()
    multi_short_rows = [multi_screen_rows[i] for i in multi_shortlist_idx]
    multi_short_prompts = [x["prompt"] for x in multi_short_rows]
    multi_short_captions = [x["caption"] for x in multi_short_rows]

    print(f"[3/6] Rescoring {len(multi_short_prompts)} multi-token shortlist prompts across full coeff grid...", flush=True)
    multi_beam_pos, multi_beam_neg = teacher_forced_multi_token_curve_beam_batched(
        model=model,
        prompts=multi_short_prompts,
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
    multi_greedy_short_pos, multi_greedy_short_neg = teacher_forced_multi_token_curve(
        model=model,
        prompts=multi_short_prompts,
        layer=args.layer,
        steering_vec=steering_vec,
        coeffs=coeffs,
        pos_ids=pos_ids,
        neg_ids=neg_ids,
        n_tokens=args.multi_tokens,
        constrained=args.constrained,
    )
    multi_selected_rows = select_centered_indices_with_greedy_guard(
        beam_pos_mat=multi_beam_pos,
        beam_neg_mat=multi_beam_neg,
        greedy_pos_mat=multi_greedy_short_pos,
        greedy_neg_mat=multi_greedy_short_neg,
        coeffs=coeffs,
        compare_coeff=compare_coeff,
        num_cases=args.num_cases,
        max_abs_cross_coeff=args.multi_max_abs_cross_coeff,
        max_near_zero_gap=args.multi_max_near_zero_gap,
        min_greedy_full_swing=args.multi_min_greedy_full_swing,
        min_greedy_directional_consistency=args.min_greedy_directional_consistency,
        unique_caption_keys=multi_short_captions,
    )

    selected_multi_rows_meta = [multi_short_rows[x["idx"]] for x in multi_selected_rows]
    selected_multi_prompts = [x["prompt"] for x in selected_multi_rows_meta]
    selected_multi_captions = [x["caption"] for x in selected_multi_rows_meta]

    # Build and screen fill-in candidates, then rerank by beam.
    print(f"[4/6] Building fill-in candidates from {len(captions)} captions...", flush=True)
    template_rows = [{"template_id": "simple_object_fillin", "template": "Caption: {caption}\nFinish the phrase about the {object}: The {object} is", "source": "seed"}]
    custom_candidates_raw = build_custom_fillin_candidates(captions, max_objects=args.custom_max_objects, template_rows=template_rows)
    custom_candidates = filter_fillin_candidates(custom_candidates_raw, min_object_quality=args.min_fill_object_quality)
    if len(custom_candidates) < max(args.custom_screen_top_k, args.num_cases):
        custom_candidates = custom_candidates_raw
    print(
        f"[5/6] Screening {len(custom_candidates)} fill-in candidates down to top {args.custom_screen_top_k}...",
        flush=True,
    )
    screened_candidates, _ = screen_custom_fillin_candidates(
        model=model,
        candidates=custom_candidates,
        layer=args.layer,
        steering_vec=steering_vec,
        coeffs=coeffs,
        use_chat_template=False,
        top_k=args.custom_screen_top_k,
        strict_center_ratio=args.strict_center_ratio,
        strict_near_zero_gap=args.strict_near_zero_gap,
        strict_orientation_margin=args.strict_orientation_margin,
        strict_wrong_side_max=args.strict_wrong_side_max,
        strict_directional_consistency=args.strict_directional_consistency,
        min_object_quality=None,
    )
    screened_prompts = [x["prompt"] for x in screened_candidates]
    fill_short_pos, fill_short_neg = continuation_multi_token_curve_beam_batched(
        model=model,
        prompts=screened_prompts,
        layer=args.layer,
        steering_vec=steering_vec,
        coeffs=select_coeffs,
        pos_ids=pos_ids,
        neg_ids=neg_ids,
        n_tokens=args.multi_tokens,
        constrained=args.constrained,
        beam_width=args.beam_width,
        beam_top_k=args.beam_top_k,
    )
    fill_reduction = per_prompt_gap_reduction(fill_short_pos, fill_short_neg, select_coeffs, compare_coeff)
    fill_short_idx = np.argsort(-fill_reduction)[: max(args.beam_shortlist_k, args.num_cases)].tolist()
    fill_short_candidates = [screened_candidates[i] for i in fill_short_idx]
    fill_short_prompts = [x["prompt"] for x in fill_short_candidates]

    print(f"[6/6] Rescoring {len(fill_short_prompts)} fill-in shortlist prompts across full coeff grid...", flush=True)
    fill_beam_pos, fill_beam_neg = continuation_multi_token_curve_beam_batched(
        model=model,
        prompts=fill_short_prompts,
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
    fill_greedy_short_pos, fill_greedy_short_neg = continuation_multi_token_curve_greedy(
        model=model,
        prompts=fill_short_prompts,
        layer=args.layer,
        steering_vec=steering_vec,
        coeffs=coeffs,
        pos_ids=pos_ids,
        neg_ids=neg_ids,
        n_tokens=args.multi_tokens,
        constrained=args.constrained,
    )
    fill_selected_rows = select_centered_indices_with_greedy_guard(
        beam_pos_mat=fill_beam_pos,
        beam_neg_mat=fill_beam_neg,
        greedy_pos_mat=fill_greedy_short_pos,
        greedy_neg_mat=fill_greedy_short_neg,
        coeffs=coeffs,
        compare_coeff=compare_coeff,
        num_cases=args.num_cases,
        max_abs_cross_coeff=args.fill_max_abs_cross_coeff,
        max_near_zero_gap=args.fill_max_near_zero_gap,
        min_greedy_full_swing=args.fill_min_greedy_full_swing,
        min_greedy_directional_consistency=args.min_greedy_directional_consistency,
        unique_caption_keys=[x["caption"] for x in fill_short_candidates],
    )

    selected_fill_rows = fill_selected_rows
    selected_fill_prompts = [fill_short_candidates[x["idx"]]["prompt"] for x in selected_fill_rows]
    actual_cases = min(args.num_cases, len(selected_multi_prompts), len(selected_fill_prompts))
    if actual_cases <= 0:
        raise RuntimeError("No valid prompt pairs were selected for the final report.")
    selected_multi_rows = multi_selected_rows[:actual_cases]
    selected_multi_prompts = selected_multi_prompts[:actual_cases]
    selected_multi_captions = selected_multi_captions[:actual_cases]
    selected_fill_rows = selected_fill_rows[:actual_cases]
    selected_fill_prompts = selected_fill_prompts[:actual_cases]

    # Evaluate the selected prompts under both decoders using the full coeff grid.
    print(
        f"[7/7] Evaluating final {actual_cases} prompt pairs under greedy and beam, then writing report...",
        flush=True,
    )
    multi_g_pos, multi_g_neg = teacher_forced_multi_token_curve(
        model=model,
        prompts=selected_multi_prompts,
        layer=args.layer,
        steering_vec=steering_vec,
        coeffs=coeffs,
        pos_ids=pos_ids,
        neg_ids=neg_ids,
        n_tokens=args.multi_tokens,
        constrained=args.constrained,
    )
    multi_b_pos_final, multi_b_neg_final = teacher_forced_multi_token_curve_beam_batched(
        model=model,
        prompts=selected_multi_prompts,
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
    fill_g_pos, fill_g_neg = continuation_multi_token_curve_greedy(
        model=model,
        prompts=selected_fill_prompts,
        layer=args.layer,
        steering_vec=steering_vec,
        coeffs=coeffs,
        pos_ids=pos_ids,
        neg_ids=neg_ids,
        n_tokens=args.multi_tokens,
        constrained=args.constrained,
    )
    fill_b_pos_final, fill_b_neg_final = continuation_multi_token_curve_beam_batched(
        model=model,
        prompts=selected_fill_prompts,
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

    fig = make_subplots(
        rows=actual_cases,
        cols=4,
        shared_xaxes=False,
        vertical_spacing=0.045,
        horizontal_spacing=0.035,
        subplot_titles=[
            label
            for _ in range(actual_cases)
            for label in (
                f"8-token greedy",
                f"8-token beam",
                f"fill-in greedy",
                f"fill-in beam",
            )
        ],
    )

    cases = []
    for i in range(actual_cases):
        multi_row = selected_multi_rows[i]
        multi_meta = selected_multi_rows_meta[i]
        fill_row = selected_fill_rows[i]
        fill_candidate = fill_short_candidates[fill_row["idx"]]
        fixed_prob_block(fig, i + 1, 1, coeffs, multi_g_pos[i], multi_g_neg[i], selected_multi_captions[i], showlegend=(i == 0))
        fixed_prob_block(fig, i + 1, 2, coeffs, multi_b_pos_final[i], multi_b_neg_final[i], selected_multi_captions[i], showlegend=False)
        fill_title = f"{fill_candidate['caption']} | object={fill_candidate['object']}"
        if len(fill_title) > 150:
            fill_title = fill_title[:147] + "..."
        fixed_prob_block(fig, i + 1, 3, coeffs, fill_g_pos[i], fill_g_neg[i], fill_title, showlegend=False)
        fixed_prob_block(fig, i + 1, 4, coeffs, fill_b_pos_final[i], fill_b_neg_final[i], fill_title, showlegend=False)

        cases.append(
            {
                "case": i + 1,
                "multi_caption": selected_multi_captions[i],
                "multi_prompt": selected_multi_prompts[i],
                "multi_template_id": multi_meta["template_id"],
                "multi_selection": {
                    "hard_ok": bool(multi_row["hard_ok"]),
                    "beam_cross_coeff": float(multi_row["beam_cross_coeff"]),
                    "beam_near_zero_gap": float(multi_row["beam_near_zero_gap"]),
                    "greedy_full_swing": float(multi_row["greedy_full_swing"]),
                    "greedy_directional_consistency": float(multi_row["greedy_directional_consistency"]),
                },
                "fill_caption": fill_candidate["caption"],
                "fill_object": fill_candidate["object"],
                "fill_prompt": fill_candidate["prompt"],
                "fill_selection": {
                    "hard_ok": bool(fill_row["hard_ok"]),
                    "beam_cross_coeff": float(fill_row["beam_cross_coeff"]),
                    "beam_near_zero_gap": float(fill_row["beam_near_zero_gap"]),
                    "greedy_full_swing": float(fill_row["greedy_full_swing"]),
                    "greedy_directional_consistency": float(fill_row["greedy_directional_consistency"]),
                },
                "multi_greedy_metrics": _scenario_delta_metrics(
                    np.asarray([multi_g_pos[i]]), np.asarray([multi_g_neg[i]]), coeffs, compare_coeff
                ),
                "multi_beam_metrics": _scenario_delta_metrics(
                    np.asarray([multi_b_pos_final[i]]), np.asarray([multi_b_neg_final[i]]), coeffs, compare_coeff
                ),
                "fill_greedy_metrics": _scenario_delta_metrics(
                    np.asarray([fill_g_pos[i]]), np.asarray([fill_g_neg[i]]), coeffs, compare_coeff
                ),
                "fill_beam_metrics": _scenario_delta_metrics(
                    np.asarray([fill_b_pos_final[i]]), np.asarray([fill_b_neg_final[i]]), coeffs, compare_coeff
                ),
            }
        )

    fig.update_layout(
        title=f"Beam-selected prompt report: greedy vs beam on the same {actual_cases} prompts",
        height=max(900, 340 * actual_cases),
        width=2200,
        template="plotly_white",
    )
    fig.write_html(str(output_html))

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
                "custom_candidate_count_total": int(len(custom_candidates)),
                "custom_candidate_count_raw": int(len(custom_candidates_raw)),
                "custom_screen_top_k": int(args.custom_screen_top_k),
                "multi_screen_top_k": int(args.multi_screen_top_k),
                "beam_shortlist_k": int(args.beam_shortlist_k),
                "multi_template_limit": int(args.multi_template_limit),
                "multi_template_ids": [x["template_id"] for x in multi_template_rows],
                "min_fill_object_quality": float(args.min_fill_object_quality),
                "multi_max_abs_cross_coeff": float(args.multi_max_abs_cross_coeff),
                "fill_max_abs_cross_coeff": float(args.fill_max_abs_cross_coeff),
                "multi_max_near_zero_gap": float(args.multi_max_near_zero_gap),
                "fill_max_near_zero_gap": float(args.fill_max_near_zero_gap),
                "multi_min_greedy_full_swing": float(args.multi_min_greedy_full_swing),
                "fill_min_greedy_full_swing": float(args.fill_min_greedy_full_swing),
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


if __name__ == "__main__":
    main()

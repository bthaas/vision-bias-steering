#!/usr/bin/env python3
"""Master prompt experiment runner for vision steering.

Runs three scenario families with separate prompt sets:
1) "The image shows" next-token class probabilities
2) "The image shows" multi-token class probabilities (teacher-forced path)
3) Custom object-focused fill-in reprompts with forced-choice completion scoring

Outputs one combined HTML plot and one JSON summary.
"""

from __future__ import annotations

import argparse
import json
import random
import re
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
from bias_steering.utils import loop_coeffs

DATASET_DIR = Path(__file__).resolve().parents[1] / "bias_steering" / "data" / "datasets"


IMAGE_SHOWS_ONE_TOKEN = [
    "A red couch sits to the left of a small wooden table in a bright room.",
    "A white car is parked in front of a brick building near a dark alley.",
    "A blue chair is beside a desk under a large window with sunlight.",
    "A yellow bus is behind a silver truck on a narrow city street.",
    "A green plant is above a black shelf next to a beige wall.",
]

IMAGE_SHOWS_MULTI_TOKEN = [
    "A brown dog stands near a blue door to the right of a white fence.",
    "A round lamp hangs above a gray sofa in a clean living room.",
    "A black bike is parked beside a red car in front of a store.",
    "A small table sits between two chairs under a bright painting.",
    "A silver laptop is on a desk near a window behind a lamp.",
]

CUSTOM_FILLIN_SPATIAL = [
    " left of another object.",
    " right of another object.",
    " above another object.",
    " below another object.",
    " near another object.",
    " behind another object.",
]

CUSTOM_FILLIN_DESCRIPTIVE = [
    " red.",
    " blue.",
    " green.",
    " large.",
    " small.",
    " bright.",
]

CUSTOM_FILLIN_TEMPLATES = [
    {
        "template_id": "simple_object_fillin",
        "template": (
            "Caption: {caption}\n"
            "Finish the phrase about the {object}: The {object} is"
        ),
    },
]


CURATED_MANUAL_V2_CASES = [
    {
        "caption": "A red mug sits to the left of a silver laptop on a wooden desk near a window.",
        "object": "mug",
    },
    {
        "caption": "A yellow bicycle is parked near a brick wall to the right of a bench.",
        "object": "bicycle",
    },
    {
        "caption": "A small dog lies behind a green sofa in front of a white wall.",
        "object": "dog",
    },
    {
        "caption": "A white plate is on the counter behind a red kettle and near a sink.",
        "object": "plate",
    },
    {
        "caption": "A black backpack rests below a coat rack next to a glass door.",
        "object": "backpack",
    },
]


OBJECT_STOPWORDS = {
    "a",
    "an",
    "the",
    "this",
    "that",
    "these",
    "those",
    "is",
    "are",
    "was",
    "were",
    "be",
    "being",
    "been",
    "am",
    "to",
    "of",
    "in",
    "on",
    "at",
    "by",
    "with",
    "from",
    "for",
    "and",
    "or",
    "but",
    "as",
    "while",
    "through",
    "across",
    "over",
    "under",
    "above",
    "below",
    "left",
    "right",
    "front",
    "behind",
    "near",
    "next",
    "beside",
    "between",
    "inside",
    "outside",
    "up",
    "down",
    "top",
    "bottom",
    "middle",
    "small",
    "large",
    "big",
    "little",
    "bright",
    "dark",
    "clean",
    "dirty",
    "old",
    "new",
    "young",
    "red",
    "blue",
    "green",
    "yellow",
    "black",
    "white",
    "brown",
    "gray",
    "grey",
    "orange",
    "pink",
    "purple",
    "silver",
    "gold",
    "wooden",
    "metal",
    "image",
    "photo",
    "picture",
    "scene",
}

OBJECT_HINT_PREFIX = {"a", "an", "the", "this", "that", "these", "those", "two", "three", "four", "five"}


def parse_args():
    parser = argparse.ArgumentParser(description="Run master prompt steering experiments.")
    parser.add_argument("--model_name", default="gpt2")
    parser.add_argument("--artifact_dir", default="runs_vision/gpt2")
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--min_coeff", type=float, default=-240.0)
    parser.add_argument("--max_coeff", type=float, default=240.0)
    parser.add_argument("--increment", type=float, default=40.0)
    parser.add_argument("--multi_tokens", type=int, default=8)
    parser.add_argument("--beam_compare", action="store_true", help="Compute greedy-vs-beam comparison metrics for the 3 scenario types.")
    parser.add_argument("--beam_width", type=int, default=4, help="Beam width for multi-token beam comparison.")
    parser.add_argument("--beam_top_k", type=int, default=8, help="Top-k token expansions per beam step.")
    parser.add_argument(
        "--beam_compare_coeff",
        type=float,
        default=None,
        help="Coefficient used to compute reduction metrics (defaults to debiased_results.json coeff if present).",
    )
    parser.add_argument("--num_cases", type=int, default=5)
    parser.add_argument("--search_pool", type=int, default=50, help="How many val captions to scan per scenario when searching best prompts.")
    parser.add_argument(
        "--captions_file",
        default="",
        help="Optional path to a JSON list or newline-delimited text file of captions. Overrides val split captions.",
    )
    parser.add_argument("--custom_max_objects", type=int, default=3, help="Max object candidates extracted per caption for custom reprompt search.")
    parser.add_argument("--custom_allow_repeat_captions", action="store_true", help="Allow multiple selected custom prompts from the same source caption.")
    parser.add_argument("--template_auto_generate", action="store_true", help="Auto-generate additional custom templates and rank them before prompt selection.")
    parser.add_argument("--template_programmatic_pool", type=int, default=24, help="How many combinatorial template variants to generate.")
    parser.add_argument("--template_llm_samples", type=int, default=16, help="How many LLM-generated template attempts to sample.")
    parser.add_argument("--template_keep_top", type=int, default=8, help="How many templates to keep after template-level ranking.")
    parser.add_argument("--template_screen_cases", type=int, default=4, help="How many captions to use when ranking templates.")
    parser.add_argument("--template_seed", type=int, default=4238, help="Random seed used for template generation/sampling.")
    parser.add_argument("--strict_center_ratio", type=float, default=0.25, help="Max normalized |cross coeff| for strict centered selections.")
    parser.add_argument("--strict_near_zero_gap", type=float, default=0.22, help="Max |spatial-descriptive| gap at coeff near zero for strict selections.")
    parser.add_argument("--strict_orientation_margin", type=float, default=0.08, help="Minimum directional edge margin for strict selections.")
    parser.add_argument("--strict_wrong_side_max", type=float, default=0.03, help="Max wrong-side edge mass for strict selections.")
    parser.add_argument("--strict_directional_consistency", type=float, default=0.65, help="Minimum directional sign-consistency for strict selections.")
    parser.add_argument("--custom_screen_top_k", type=int, default=120, help="Shortlist this many custom candidates before full coeff sweep (0 disables).")
    parser.add_argument("--constrained", action="store_true", help="Constrain class probs to spatial+descriptive token set.")
    parser.add_argument(
        "--use_curated_manual_v2",
        action="store_true",
        help="Use a fixed set of curated captions and custom prompts (manual v2).",
    )
    parser.add_argument("--output_html", default="runs_vision/gpt2/validation/master_prompt_experiments.html")
    parser.add_argument("--output_json", default="runs_vision/gpt2/validation/master_prompt_experiments.json")
    parser.add_argument("--output_png", default="", help="Optional static PNG export path.")
    parser.add_argument("--open_browser", action="store_true")
    parser.add_argument("--use_chat_template", action="store_true", help="Wrap prompts with chat template (off by default for GPT-2).")
    return parser.parse_args()


def load_target_words(target_concept="vision"):
    return json.loads((DATASET_DIR / "target_words.json").read_text())[target_concept]


def _load_captions_from_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"captions file not found: {path}")

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        if not isinstance(data, list):
            raise ValueError("captions JSON must be a list of strings.")
        captions = [str(x).strip() for x in data if str(x).strip()]
    else:
        captions = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]

    if len(captions) == 0:
        raise ValueError(f"captions file is empty: {path}")
    return captions


def class_probs_from_logits(logits_last, pos_ids, neg_ids, constrained: bool):
    if constrained:
        tracked = pos_ids + neg_ids
        n_pos = len(pos_ids)
        target_logits = logits_last[tracked]
        probs = F.softmax(target_logits, dim=-1)
        pos_prob = float(probs[:n_pos].sum().item())
        neg_prob = float(probs[n_pos:].sum().item())
    else:
        probs = F.softmax(logits_last, dim=-1)
        pos_prob = float(probs[pos_ids].sum().item())
        neg_prob = float(probs[neg_ids].sum().item())
    return pos_prob, neg_prob


def next_token_class_prob_curve(
    model,
    prompts,
    layer,
    steering_vec,
    coeffs,
    pos_ids,
    neg_ids,
    constrained: bool,
    batch_size: int = 64,
):
    batch_size = max(1, int(batch_size))
    pos_mat = np.zeros((len(prompts), len(coeffs)), dtype=np.float64)
    neg_mat = np.zeros((len(prompts), len(coeffs)), dtype=np.float64)
    for j, coeff in enumerate(coeffs):
        intervene = get_intervention_func(steering_vec, method="constant", coeff=coeff)
        for s in range(0, len(prompts), batch_size):
            e = min(len(prompts), s + batch_size)
            logits = model.get_logits(prompts[s:e], layer=layer, intervene_func=intervene)
            for i in range(e - s):
                p, n = class_probs_from_logits(logits[i, -1, :], pos_ids, neg_ids, constrained=constrained)
                pos_mat[s + i, j] = p
                neg_mat[s + i, j] = n
    return pos_mat, neg_mat


def greedy_reference_ids(model, prompt, layer, n_tokens):
    ids = []
    context = prompt
    for _ in range(n_tokens):
        logits = model.get_logits([context], layer=layer, intervene_func=None)
        probs = F.softmax(logits[0, -1, :], dim=-1)
        tok_id = int(torch.argmax(probs).item())
        ids.append(tok_id)
        context += model.tokenizer.decode([tok_id])
    return ids


def _token_lengths(model, prompts: list[str]) -> list[int]:
    enc = model.tokenizer(prompts, padding=False, truncation=False)
    return [len(x) for x in enc["input_ids"]]


def _length_buckets(lengths: list[int]):
    buckets = {}
    for i, ln in enumerate(lengths):
        buckets.setdefault(int(ln), []).append(i)
    return buckets


def greedy_reference_ids_batch(model, prompts, layer, n_tokens, batch_size: int = 64):
    batch_size = max(1, int(batch_size))
    n_tokens = max(0, int(n_tokens))
    contexts = list(prompts)
    lengths = _token_lengths(model, contexts)
    ref_ids = [[] for _ in prompts]
    for _ in range(n_tokens):
        for _, idxs in sorted(_length_buckets(lengths).items()):
            for s in range(0, len(idxs), batch_size):
                batch_idxs = idxs[s : s + batch_size]
                batch_prompts = [contexts[i] for i in batch_idxs]
                logits = model.get_logits(batch_prompts, layer=layer, intervene_func=None)
                tok_ids = torch.argmax(logits[:, -1, :], dim=-1).tolist()
                for bi, tok_id in enumerate(tok_ids):
                    idx = batch_idxs[bi]
                    tid = int(tok_id)
                    ref_ids[idx].append(tid)
                    contexts[idx] += model.tokenizer.decode([tid])
                    lengths[idx] += 1
    return ref_ids


def teacher_forced_multi_token_curve(
    model,
    prompts,
    layer,
    steering_vec,
    coeffs,
    pos_ids,
    neg_ids,
    n_tokens,
    constrained: bool,
    batch_size: int = 64,
):
    batch_size = max(1, int(batch_size))
    n_tokens = max(0, int(n_tokens))
    ref_ids = greedy_reference_ids_batch(model, prompts, layer, n_tokens, batch_size=batch_size)
    pos_mat = np.zeros((len(prompts), len(coeffs)), dtype=np.float64)
    neg_mat = np.zeros((len(prompts), len(coeffs)), dtype=np.float64)
    base_lengths = _token_lengths(model, prompts)

    for j, coeff in enumerate(coeffs):
        intervene = get_intervention_func(steering_vec, method="constant", coeff=coeff)
        contexts = list(prompts)
        lengths = list(base_lengths)
        pos_sum = np.zeros((len(prompts),), dtype=np.float64)
        neg_sum = np.zeros((len(prompts),), dtype=np.float64)
        for t in range(n_tokens):
            for _, idxs in sorted(_length_buckets(lengths).items()):
                for s in range(0, len(idxs), batch_size):
                    batch_idxs = idxs[s : s + batch_size]
                    batch_prompts = [contexts[i] for i in batch_idxs]
                    logits = model.get_logits(batch_prompts, layer=layer, intervene_func=intervene)
                    logits_last = logits[:, -1, :]
                    for bi, idx in enumerate(batch_idxs):
                        p, n = class_probs_from_logits(logits_last[bi], pos_ids, neg_ids, constrained=constrained)
                        pos_sum[idx] += p
                        neg_sum[idx] += n
                        tok_id = ref_ids[idx][t]
                        contexts[idx] += model.tokenizer.decode([tok_id])
                        lengths[idx] += 1
        if n_tokens > 0:
            pos_mat[:, j] = pos_sum / float(n_tokens)
            neg_mat[:, j] = neg_sum / float(n_tokens)
        else:
            pos_mat[:, j] = 0.0
            neg_mat[:, j] = 0.0
    return pos_mat, neg_mat


def _normalize_logweights(log_weights: list[float]) -> list[float]:
    if len(log_weights) == 0:
        return []
    m = float(max(log_weights))
    shifted = np.exp(np.asarray(log_weights, dtype=np.float64) - m)
    z = float(shifted.sum())
    if (not np.isfinite(z)) or z <= 0.0:
        return [1.0 / len(log_weights)] * len(log_weights)
    return (shifted / z).tolist()


def beam_reference_paths(model, prompt: str, layer: int, n_tokens: int, beam_width: int, beam_top_k: int):
    beam_width = max(1, int(beam_width))
    beam_top_k = max(1, int(beam_top_k))
    n_tokens = max(0, int(n_tokens))

    beams = [{"ids": [], "sum_logprob": 0.0, "context": prompt}]
    for _ in range(n_tokens):
        contexts = [b["context"] for b in beams]
        logits_batch = model.get_logits(contexts, layer=layer, intervene_func=None)
        candidates = []
        for beam_idx, beam in enumerate(beams):
            logits = logits_batch[beam_idx, -1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            k = min(beam_top_k, int(log_probs.shape[-1]))
            top_log_probs, top_ids = torch.topk(log_probs, k=k)
            for j in range(k):
                tok_id = int(top_ids[j].item())
                tok_logprob = float(top_log_probs[j].item())
                tok_piece = model.tokenizer.decode([tok_id])
                candidates.append(
                    {
                        "ids": beam["ids"] + [tok_id],
                        "sum_logprob": beam["sum_logprob"] + tok_logprob,
                        "context": beam["context"] + tok_piece,
                    }
                )
        candidates = sorted(candidates, key=lambda x: x["sum_logprob"], reverse=True)
        beams = candidates[:beam_width]

    weights = _normalize_logweights([float(x["sum_logprob"]) for x in beams])
    out = []
    for i, beam in enumerate(beams):
        out.append(
            {
                "ids": beam["ids"],
                "sum_logprob": float(beam["sum_logprob"]),
                "weight": float(weights[i]) if i < len(weights) else 0.0,
            }
        )
    return out


def teacher_forced_multi_token_curve_beam(
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
):
    ref_beams = [
        beam_reference_paths(
            model=model,
            prompt=p,
            layer=layer,
            n_tokens=n_tokens,
            beam_width=beam_width,
            beam_top_k=beam_top_k,
        )
        for p in prompts
    ]

    pos_mat = np.zeros((len(prompts), len(coeffs)), dtype=np.float64)
    neg_mat = np.zeros((len(prompts), len(coeffs)), dtype=np.float64)
    for j, coeff in enumerate(coeffs):
        intervene = get_intervention_func(steering_vec, method="constant", coeff=coeff)
        for i, prompt in enumerate(prompts):
            seq_pos = []
            seq_neg = []
            for beam in ref_beams[i]:
                tok_ids = beam["ids"]
                weight = float(beam["weight"])
                context = prompt
                pos_vals = []
                neg_vals = []
                for tok_id in tok_ids:
                    logits = model.get_logits([context], layer=layer, intervene_func=intervene)
                    p, n = class_probs_from_logits(logits[0, -1, :], pos_ids, neg_ids, constrained=constrained)
                    pos_vals.append(p)
                    neg_vals.append(n)
                    context += model.tokenizer.decode([tok_id])
                seq_pos.append(weight * (float(np.mean(pos_vals)) if pos_vals else 0.0))
                seq_neg.append(weight * (float(np.mean(neg_vals)) if neg_vals else 0.0))
            pos_mat[i, j] = float(np.sum(seq_pos)) if seq_pos else 0.0
            neg_mat[i, j] = float(np.sum(seq_neg)) if seq_neg else 0.0
    return pos_mat, neg_mat


def continuation_multi_token_curve_greedy(
    model,
    prompts,
    layer,
    steering_vec,
    coeffs,
    pos_ids,
    neg_ids,
    n_tokens,
    constrained: bool,
    batch_size: int = 64,
):
    n_tokens = max(0, int(n_tokens))
    batch_size = max(1, int(batch_size))
    pos_mat = np.zeros((len(prompts), len(coeffs)), dtype=np.float64)
    neg_mat = np.zeros((len(prompts), len(coeffs)), dtype=np.float64)
    for j, coeff in enumerate(coeffs):
        intervene = get_intervention_func(steering_vec, method="constant", coeff=coeff)
        contexts = list(prompts)
        pos_sum = np.zeros((len(prompts),), dtype=np.float64)
        neg_sum = np.zeros((len(prompts),), dtype=np.float64)

        for _ in range(n_tokens):
            new_contexts = [None] * len(prompts)
            for s in range(0, len(prompts), batch_size):
                e = min(len(prompts), s + batch_size)
                batch_contexts = contexts[s:e]
                logits = model.get_logits(batch_contexts, layer=layer, intervene_func=intervene)
                logits_last = logits[:, -1, :]
                for bi in range(e - s):
                    row = logits_last[bi]
                    p, n = class_probs_from_logits(row, pos_ids, neg_ids, constrained=constrained)
                    pos_sum[s + bi] += p
                    neg_sum[s + bi] += n
                    tok_id = int(torch.argmax(row).item())
                    tok_piece = model.tokenizer.decode([tok_id])
                    new_contexts[s + bi] = batch_contexts[bi] + tok_piece
            contexts = new_contexts

        if n_tokens > 0:
            pos_mat[:, j] = pos_sum / float(n_tokens)
            neg_mat[:, j] = neg_sum / float(n_tokens)
        else:
            pos_mat[:, j] = 0.0
            neg_mat[:, j] = 0.0
    return pos_mat, neg_mat


def continuation_multi_token_curve_beam(
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
):
    beam_width = max(1, int(beam_width))
    beam_top_k = max(1, int(beam_top_k))
    n_tokens = max(0, int(n_tokens))

    pos_mat = np.zeros((len(prompts), len(coeffs)), dtype=np.float64)
    neg_mat = np.zeros((len(prompts), len(coeffs)), dtype=np.float64)
    for j, coeff in enumerate(coeffs):
        intervene = get_intervention_func(steering_vec, method="constant", coeff=coeff)
        for i, prompt in enumerate(prompts):
            beams = [{"context": prompt, "sum_logprob": 0.0, "pos_sum": 0.0, "neg_sum": 0.0}]
            for _ in range(n_tokens):
                contexts = [b["context"] for b in beams]
                logits_batch = model.get_logits(contexts, layer=layer, intervene_func=intervene)
                candidates = []
                for beam_idx, beam in enumerate(beams):
                    logits_last = logits_batch[beam_idx, -1, :]
                    p, n = class_probs_from_logits(logits_last, pos_ids, neg_ids, constrained=constrained)
                    log_probs = F.log_softmax(logits_last, dim=-1)
                    k = min(beam_top_k, int(log_probs.shape[-1]))
                    top_log_probs, top_ids = torch.topk(log_probs, k=k)
                    for t in range(k):
                        tok_id = int(top_ids[t].item())
                        tok_logprob = float(top_log_probs[t].item())
                        tok_piece = model.tokenizer.decode([tok_id])
                        candidates.append(
                            {
                                "context": beam["context"] + tok_piece,
                                "sum_logprob": beam["sum_logprob"] + tok_logprob,
                                "pos_sum": beam["pos_sum"] + p,
                                "neg_sum": beam["neg_sum"] + n,
                            }
                        )
                candidates = sorted(candidates, key=lambda x: x["sum_logprob"], reverse=True)
                beams = candidates[:beam_width]

            if len(beams) == 0:
                pos_mat[i, j] = 0.0
                neg_mat[i, j] = 0.0
                continue

            weights = _normalize_logweights([float(x["sum_logprob"]) for x in beams])
            if n_tokens <= 0:
                pos_mat[i, j] = 0.0
                neg_mat[i, j] = 0.0
            else:
                exp_pos = float(sum(weights[b] * (beams[b]["pos_sum"] / n_tokens) for b in range(len(beams))))
                exp_neg = float(sum(weights[b] * (beams[b]["neg_sum"] / n_tokens) for b in range(len(beams))))
                pos_mat[i, j] = exp_pos
                neg_mat[i, j] = exp_neg
    return pos_mat, neg_mat


def _resolve_beam_compare_coeff(artifact_dir: Path, explicit_coeff: float | None) -> float:
    if explicit_coeff is not None:
        return float(explicit_coeff)
    debias_path = artifact_dir / "validation" / "debiased_results.json"
    if debias_path.exists():
        try:
            data = json.loads(debias_path.read_text())
            if len(data) > 0 and "coeff" in data[0]:
                return float(data[0]["coeff"])
        except Exception:
            pass
    raise ValueError(
        "Could not resolve beam comparison coefficient. "
        "Pass --beam_compare_coeff explicitly or provide validation/debiased_results.json."
    )


def _scenario_delta_metrics(pos_mat, neg_mat, coeffs, compare_coeff: float):
    coeff_arr = np.asarray(coeffs, dtype=np.float64)
    zero_idx = int(np.argmin(np.abs(coeff_arr)))
    steer_idx = int(np.argmin(np.abs(coeff_arr - float(compare_coeff))))
    diff = np.asarray(pos_mat, dtype=np.float64) - np.asarray(neg_mat, dtype=np.float64)
    baseline = diff[:, zero_idx]
    steered = diff[:, steer_idx]
    delta = steered - baseline

    baseline_gap = np.abs(baseline)
    steered_gap = np.abs(steered)
    baseline_gap_mean = float(np.mean(baseline_gap)) if baseline_gap.size > 0 else 0.0
    steered_gap_mean = float(np.mean(steered_gap)) if steered_gap.size > 0 else 0.0
    if baseline_gap_mean > 0:
        gap_reduction_pct = float((baseline_gap_mean - steered_gap_mean) / baseline_gap_mean * 100.0)
    else:
        gap_reduction_pct = 0.0

    return {
        "baseline_coeff": float(coeff_arr[zero_idx]),
        "steering_coeff_requested": float(compare_coeff),
        "steering_coeff_used": float(coeff_arr[steer_idx]),
        "mean_diff_baseline": float(np.mean(baseline)) if baseline.size > 0 else 0.0,
        "mean_diff_steered": float(np.mean(steered)) if steered.size > 0 else 0.0,
        "mean_diff_delta": float(np.mean(delta)) if delta.size > 0 else 0.0,
        "mean_abs_gap_baseline": baseline_gap_mean,
        "mean_abs_gap_steered": steered_gap_mean,
        "gap_reduction_pct": gap_reduction_pct,
        "num_cases": int(diff.shape[0]) if diff.ndim >= 2 else 0,
    }


def build_image_shows_prompts(captions):
    return [f"Describe this image:\n{x}\nThe image shows" for x in captions]


def _prompt_grammar_forms(object_token: str) -> dict[str, str]:
    t = object_token.lower()
    is_plural = t.endswith("s")
    return {
        "be_verb": "are" if is_plural else "is",
        "demo": "those" if is_plural else "that",
    }


def extract_object_candidates(caption: str, max_objects: int):
    max_objects = max(1, int(max_objects))
    toks = [t for t in re.findall(r"[A-Za-z][A-Za-z'-]*", caption.lower()) if t]
    if len(toks) == 0:
        return [("object", 0.0)]

    candidates = []
    for i, tok in enumerate(toks):
        if len(tok) < 3:
            continue
        if tok in OBJECT_STOPWORDS:
            continue
        if tok.endswith("ing") and len(tok) > 5:
            continue
        if tok.endswith("ly"):
            continue
        norm = tok[:-1] if tok.endswith("s") and len(tok) > 3 and not tok.endswith("ss") else tok
        score = 1.0
        if i > 0 and toks[i - 1] in OBJECT_HINT_PREFIX:
            score += 0.9
        if i > 1 and toks[i - 2] in OBJECT_HINT_PREFIX:
            score += 0.35
        score += 0.18 * (i / max(1, len(toks) - 1))
        candidates.append((norm, score))

    if len(candidates) == 0:
        return [("object", 0.0)]

    best_by_obj = {}
    for obj, score in candidates:
        if obj not in best_by_obj or score > best_by_obj[obj]:
            best_by_obj[obj] = score
    ranked = sorted(best_by_obj.items(), key=lambda x: (-x[1], x[0]))
    return ranked[:max_objects]


def _render_custom_template(template: str, caption: str, obj: str) -> str:
    grammar = _prompt_grammar_forms(obj)
    return template.format(
        caption=caption,
        object=obj,
        be_verb=grammar["be_verb"],
        demo=grammar["demo"],
    )


def _normalize_template_text(text: str) -> str | None:
    t = text.strip().replace("\r", "\n").replace("’", "'").replace("“", "\"").replace("”", "\"")
    if not t:
        return None
    t = re.sub(r"```[a-zA-Z]*", "", t).replace("```", "").strip()
    lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
    if not lines:
        return None

    # Prefer a line that already carries placeholders.
    picked = None
    for ln in lines:
        cleaned = re.sub(r"^\d+[\)\.\-:]\s*", "", ln).strip(" \"'")
        if "{caption}" in cleaned and "{object}" in cleaned:
            picked = cleaned
            break
    if picked is None:
        picked = " ".join(lines)
        picked = re.sub(r"^\d+[\)\.\-:]\s*", "", picked).strip(" \"'")

    allowed_placeholders = {"caption", "object", "be_verb", "demo"}
    found_placeholders = set(re.findall(r"\{([a-zA-Z_]+)\}", picked))
    if not found_placeholders.issubset(allowed_placeholders):
        return None

    if "{caption}" not in picked or "{object}" not in picked:
        return None

    picked = re.sub(
        r"\bthe\s+\{object\}\s+(is|are)\b",
        "The {object} {be_verb}",
        picked,
        flags=re.IGNORECASE,
    )
    picked = re.sub(
        r"\b\{object\}\s+(is|are)\b",
        "{object} {be_verb}",
        picked,
        flags=re.IGNORECASE,
    )
    if "{be_verb}" not in picked:
        picked = picked.rstrip(". ")
        picked += "\nFill in the blank: The {object} {be_verb}"

    if "\n" not in picked and "Fill in the blank:" in picked:
        picked = picked.replace(" Fill in the blank:", "\nFill in the blank:")

    return picked.strip()


def _dedupe_template_rows(rows):
    seen = set()
    out = []
    for row in rows:
        key = row["template"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def generate_programmatic_templates(count: int, seed: int):
    prefixes = [
        "Caption: {caption}",
        "Scene: {caption}",
        "Original description: {caption}",
        "Given this caption: {caption}",
        "Image caption: {caption}",
        "From this caption: {caption}",
    ]
    focus_clauses = [
        "Focus on one visual attribute of the {object}.",
        "Describe only the appearance of the {object}.",
        "Write one color/size detail about the {object}.",
        "Describe a visual trait of {demo} {object} from this scene.",
        "Continue with an appearance detail for the {object}.",
        "Name the {object} and describe its appearance.",
        "Ignore position and describe the look of the {object}.",
        "Stay specific to how the {object} looks.",
    ]
    tails = [
        "Fill in the blank: The {object} {be_verb}",
        "Complete the phrase: The {object} {be_verb}",
        "One short clause: The {object} {be_verb}",
    ]
    combos = []
    for p in prefixes:
        for f in focus_clauses:
            for t in tails:
                combos.append(f"{p}\n{f} {t}")
    rng = random.Random(seed)
    rng.shuffle(combos)
    rows = []
    for i, text in enumerate(combos[: max(0, count)]):
        normalized = _normalize_template_text(text)
        if normalized is None:
            continue
        rows.append(
            {
                "template_id": f"auto_prog_{i+1:03d}",
                "template": normalized,
                "source": "programmatic",
            }
        )
    return rows


def generate_llm_templates(model, n_samples: int, use_chat_template: bool):
    if n_samples <= 0:
        return []

    style_targets = [
        "short and direct",
        "question then reprompt",
        "scene-to-object transition",
        "follow-up tone",
        "evidence-style phrasing",
        "instructional style",
        "neutral descriptive style",
        "compact one-sentence style",
    ]
    tasks = []
    for i in range(n_samples):
        style = style_targets[i % len(style_targets)]
        tasks.append(
            "\n".join(
                [
                    "Write exactly one prompt template.",
                    "Rules:",
                    "1) Include placeholders {caption} and {object}.",
                    "2) Use {be_verb} instead of is/are.",
                    "3) You may use {demo}.",
                    "4) End with: The {object} {be_verb}",
                    "5) Keep under 28 words.",
                    f"Style target: {style}.",
                    "Return only the template text.",
                ]
            )
        )

    prompts = model.apply_chat_template(tasks) if use_chat_template else tasks
    generations = model.generate(
        prompts,
        max_new_tokens=56,
        do_sample=True,
        temperature=0.9,
        top_p=0.92,
    )

    rows = []
    for i, gen in enumerate(generations):
        normalized = _normalize_template_text(gen)
        if normalized is None:
            continue
        rows.append(
            {
                "template_id": f"auto_llm_{i+1:03d}",
                "template": normalized,
                "source": "llm",
            }
        )
    return rows


def rank_template_pool(
    model,
    template_rows,
    captions,
    layer,
    steering_vec,
    coeffs,
    screen_cases: int,
    keep_top: int,
    use_chat_template: bool,
    strict_center_ratio: float,
    strict_near_zero_gap: float,
    strict_orientation_margin: float,
    strict_wrong_side_max: float,
    strict_directional_consistency: float,
):
    if len(template_rows) <= keep_top:
        return template_rows, []

    n_screen = min(max(1, screen_cases), len(captions))
    screen_captions = captions[:n_screen]
    screen_coeffs = [coeffs[0], coeffs[int(np.argmin(np.abs(np.array(coeffs))))], coeffs[-1]]
    # Preserve order while removing duplicates.
    screen_coeffs = list(dict.fromkeys(screen_coeffs))

    ranking = []
    for row in template_rows:
        prompts_raw = []
        obj_quality_vals = []
        for cap in screen_captions:
            obj, obj_q = extract_object_candidates(cap, max_objects=1)[0]
            obj_quality_vals.append(float(obj_q))
            prompts_raw.append(_render_custom_template(row["template"], cap, obj))
        prompts_eval = model.apply_chat_template(prompts_raw) if use_chat_template else prompts_raw
        pos_mat, neg_mat = forced_choice_fillin_curve(
            model,
            prompts_eval,
            layer,
            steering_vec,
            screen_coeffs,
            spatial_completions=CUSTOM_FILLIN_SPATIAL,
            descriptive_completions=CUSTOM_FILLIN_DESCRIPTIVE,
        )

        q_rows = [summarize_curve_quality(pos_mat[i], neg_mat[i], screen_coeffs) for i in range(len(prompts_eval))]
        mean_score = float(np.mean([x["score"] for x in q_rows]))
        mean_center = float(np.mean([x["center_swing"] for x in q_rows]))
        mean_near0 = float(np.mean([x["near_zero_gap"] for x in q_rows]))
        mean_orientation = float(np.mean([x["orientation_margin"] for x in q_rows]))
        mean_wrong_side = float(np.mean([x["wrong_side_penalty"] for x in q_rows]))
        mean_cross_ratio = float(np.mean([x["cross_coeff_abs_ratio"] for x in q_rows]))
        mean_consistency = float(np.mean([x["directional_consistency"] for x in q_rows]))
        mean_left = float(np.mean([x["left_diff"] for x in q_rows]))
        mean_right = float(np.mean([x["right_diff"] for x in q_rows]))
        mean_obj_q = float(np.mean(obj_quality_vals)) if obj_quality_vals else 0.0
        agg = (
            mean_score
            + (0.9 * mean_orientation)
            + (0.5 * mean_consistency)
            + (0.2 * mean_center)
            + (0.15 * mean_obj_q)
            - (1.4 * mean_wrong_side)
            - (0.9 * mean_cross_ratio)
            - (0.4 * mean_near0)
        )
        probe = {
            "left_diff": mean_left,
            "right_diff": mean_right,
            "orientation_margin": mean_orientation,
            "wrong_side_penalty": mean_wrong_side,
            "cross_coeff_abs_ratio": mean_cross_ratio,
            "near_zero_gap": mean_near0,
            "directional_consistency": mean_consistency,
            "center_swing": mean_center,
            "full_swing": float(np.mean([x["full_swing"] for x in q_rows])),
        }
        tier = _quality_tier(
            probe,
            strict_center_ratio=strict_center_ratio,
            strict_near_zero_gap=strict_near_zero_gap,
            strict_orientation_margin=strict_orientation_margin,
            strict_wrong_side_max=strict_wrong_side_max,
            strict_directional_consistency=strict_directional_consistency,
            min_object_quality=None,
        )

        ranking.append(
            {
                **row,
                "template_screen_score": float(agg),
                "template_screen_tier": int(tier),
                "template_screen_mean_score": mean_score,
                "template_screen_center_swing": mean_center,
                "template_screen_near_zero_gap": mean_near0,
                "template_screen_orientation_margin": mean_orientation,
                "template_screen_wrong_side": mean_wrong_side,
                "template_screen_cross_ratio": mean_cross_ratio,
                "template_screen_directional_consistency": mean_consistency,
                "template_screen_object_quality": mean_obj_q,
            }
        )

    ranking = sorted(
        ranking,
        key=lambda x: (
            x["template_screen_tier"],
            -x["template_screen_score"],
            x["template_screen_wrong_side"],
            x["template_screen_cross_ratio"],
            x["template_screen_near_zero_gap"],
            -x["template_screen_orientation_margin"],
        ),
    )
    top = ranking[: max(1, keep_top)]
    return top, ranking


def build_custom_fillin_candidates(captions: list[str], max_objects: int, template_rows):
    candidates = []
    for caption in captions:
        obj_rows = extract_object_candidates(caption, max_objects=max_objects)
        for obj, obj_quality in obj_rows:
            grammar = _prompt_grammar_forms(obj)
            for template_row in template_rows:
                prompt = _render_custom_template(template_row["template"], caption, obj)
                candidates.append(
                    {
                        "caption": caption,
                        "object": obj,
                        "object_quality": float(obj_quality),
                        "be_verb": grammar["be_verb"],
                        "template_id": template_row["template_id"],
                        "template_source": template_row.get("source", "seed"),
                        "prompt": prompt,
                    }
                )
    return candidates


def screen_custom_fillin_candidates(
    model,
    candidates,
    layer,
    steering_vec,
    coeffs,
    use_chat_template: bool,
    top_k: int,
    strict_center_ratio: float,
    strict_near_zero_gap: float,
    strict_orientation_margin: float,
    strict_wrong_side_max: float,
    strict_directional_consistency: float,
    min_object_quality: float,
):
    if top_k <= 0 or len(candidates) <= top_k:
        return candidates, []

    # Runtime guard: only model-score a compact high-quality presample.
    pre_screen_pool = min(len(candidates), max((2 * top_k), (top_k + 8)))
    screened_pool = sorted(
        candidates,
        key=lambda x: (-x["object_quality"], x["caption"], x["template_id"], len(x["prompt"])),
    )[:pre_screen_pool]

    mid_idx = int(np.argmin(np.abs(np.asarray(coeffs))))
    screen_coeffs = [coeffs[0], coeffs[mid_idx], coeffs[-1]]
    screen_coeffs = list(dict.fromkeys(screen_coeffs))
    prompts = [x["prompt"] for x in screened_pool]
    if use_chat_template:
        prompts = model.apply_chat_template(prompts)

    pos_mat, neg_mat = forced_choice_fillin_curve(
        model,
        prompts,
        layer,
        steering_vec,
        screen_coeffs,
        spatial_completions=CUSTOM_FILLIN_SPATIAL,
        descriptive_completions=CUSTOM_FILLIN_DESCRIPTIVE,
    )

    ranked = []
    for i in range(pos_mat.shape[0]):
        quality = summarize_curve_quality(pos_mat[i], neg_mat[i], screen_coeffs)
        row = {"idx": i, **quality, **screened_pool[i]}
        row["quality_tier"] = _quality_tier(
            row,
            strict_center_ratio=strict_center_ratio,
            strict_near_zero_gap=strict_near_zero_gap,
            strict_orientation_margin=strict_orientation_margin,
            strict_wrong_side_max=strict_wrong_side_max,
            strict_directional_consistency=strict_directional_consistency,
            min_object_quality=min_object_quality,
        )
        ranked.append(row)

    ranked = sorted(ranked, key=lambda x: _quality_sort_key(x, include_object_quality=True))
    kept = ranked[: max(1, top_k)]
    screened_candidates = [screened_pool[x["idx"]] for x in kept]
    return screened_candidates, kept


def _zero_crossing_coeff(diff_curve, coeffs):
    if len(coeffs) == 0:
        return 0.0
    if len(coeffs) == 1:
        return float(coeffs[0])

    c_arr = np.asarray(coeffs, dtype=np.float64)
    d_arr = np.asarray(diff_curve, dtype=np.float64)
    best = None
    for i in range(len(d_arr) - 1):
        d0, d1 = float(d_arr[i]), float(d_arr[i + 1])
        c0, c1 = float(c_arr[i]), float(c_arr[i + 1])
        if d0 == 0.0:
            cand = c0
        elif d1 == 0.0:
            cand = c1
        elif d0 * d1 < 0.0:
            denom = (d1 - d0)
            if denom == 0.0:
                continue
            t = (-d0) / denom
            cand = c0 + t * (c1 - c0)
        else:
            continue
        dist = abs(cand)
        if best is None or dist < best[0]:
            best = (dist, cand)

    if best is not None:
        return float(best[1])
    balance_idx = int(np.argmin(np.abs(d_arr)))
    return float(c_arr[balance_idx])


def summarize_curve_quality(pos_curve, neg_curve, coeffs):
    mid_idx = int(np.argmin(np.abs(np.array(coeffs))))
    max_abs = max(abs(coeffs[0]), abs(coeffs[-1])) if coeffs else 1.0

    diff_curve = np.asarray(pos_curve, dtype=np.float64) - np.asarray(neg_curve, dtype=np.float64)
    left_diff = float(diff_curve[0])
    right_diff = float(diff_curve[-1])
    left_descriptive_margin = max(0.0, -left_diff)
    right_spatial_margin = max(0.0, right_diff)
    orientation_margin = min(left_descriptive_margin, right_spatial_margin)
    wrong_side_penalty = max(0.0, left_diff) + max(0.0, -right_diff)
    side_margin_gap = abs(left_descriptive_margin - right_spatial_margin)

    neg_mask = np.asarray(coeffs) < 0
    pos_mask = np.asarray(coeffs) > 0
    neg_consistency = float(np.mean(diff_curve[neg_mask] < 0.0)) if np.any(neg_mask) else 1.0
    pos_consistency = float(np.mean(diff_curve[pos_mask] > 0.0)) if np.any(pos_mask) else 1.0
    directional_consistency = 0.5 * (neg_consistency + pos_consistency)

    cross_coeff = _zero_crossing_coeff(diff_curve, coeffs)
    cross_ratio = abs(cross_coeff) / max_abs if max_abs > 0 else 0.0

    neg_edge = float(-diff_curve[0])
    pos_edge = float(diff_curve[-1])
    gap = np.abs(diff_curve)
    balance_idx = int(np.argmin(gap))
    balance_coeff = float(coeffs[balance_idx])
    balance_gap = float(gap[balance_idx])
    near_zero_gap = float(gap[mid_idx])
    pos_gain = float(pos_curve[-1] - pos_curve[0])
    neg_drop = float(neg_curve[0] - neg_curve[-1])
    transition_strength = min(pos_gain, neg_drop)
    low_idx = max(0, mid_idx - 1)
    high_idx = min(len(diff_curve) - 1, mid_idx + 1)
    center_swing = float(diff_curve[high_idx] - diff_curve[low_idx]) if high_idx > low_idx else 0.0
    full_swing = float(diff_curve[-1] - diff_curve[0])
    center_penalty = abs(balance_coeff) / max_abs if max_abs > 0 else 0.0

    score = (
        (2.0 * orientation_margin)
        + (1.8 * directional_consistency)
        + (1.4 * center_swing)
        + (1.1 * full_swing)
        + (0.6 * transition_strength)
        - (2.5 * near_zero_gap)
        - (1.0 * balance_gap)
        - (2.4 * cross_ratio)
        - (3.0 * wrong_side_penalty)
        - (0.5 * side_margin_gap)
    )
    return {
        "score": score,
        "neg_edge": neg_edge,
        "pos_edge": pos_edge,
        "left_diff": left_diff,
        "right_diff": right_diff,
        "orientation_margin": orientation_margin,
        "wrong_side_penalty": wrong_side_penalty,
        "directional_consistency": directional_consistency,
        "cross_coeff": cross_coeff,
        "cross_coeff_abs_ratio": cross_ratio,
        "balance_coeff": balance_coeff,
        "balance_coeff_abs_ratio": center_penalty,
        "balance_gap": balance_gap,
        "near_zero_gap": near_zero_gap,
        "transition_strength": transition_strength,
        "center_swing": center_swing,
        "full_swing": full_swing,
    }


def _quality_tier(
    row,
    strict_center_ratio: float,
    strict_near_zero_gap: float,
    strict_orientation_margin: float,
    strict_wrong_side_max: float,
    strict_directional_consistency: float,
    min_object_quality: float | None = None,
):
    if min_object_quality is not None and row.get("object_quality", 0.0) < min_object_quality:
        return 5

    directional = row["left_diff"] < 0.0 and row["right_diff"] > 0.0
    strict = (
        directional
        and row["orientation_margin"] >= strict_orientation_margin
        and row["wrong_side_penalty"] <= strict_wrong_side_max
        and row["cross_coeff_abs_ratio"] <= strict_center_ratio
        and row["near_zero_gap"] <= strict_near_zero_gap
        and row["directional_consistency"] >= strict_directional_consistency
        and row["center_swing"] > 0.0
        and row["full_swing"] > 0.0
    )
    if strict:
        return 0

    centered = (
        directional
        and row["orientation_margin"] >= (0.7 * strict_orientation_margin)
        and row["wrong_side_penalty"] <= max(0.08, 2.5 * strict_wrong_side_max)
        and row["cross_coeff_abs_ratio"] <= max(0.45, 1.8 * strict_center_ratio)
        and row["near_zero_gap"] <= max(0.34, 1.8 * strict_near_zero_gap)
        and row["directional_consistency"] >= max(0.45, 0.8 * strict_directional_consistency)
    )
    if centered:
        return 1

    directional_ok = directional and row["orientation_margin"] >= max(0.02, 0.45 * strict_orientation_margin)
    if directional_ok:
        return 2
    if row["orientation_margin"] > 0.0:
        return 3
    return 4


def _quality_sort_key(row, include_object_quality: bool):
    key = (
        row.get("quality_tier", 9),
        -row["score"],
        -row.get("orientation_margin", 0.0),
        -row.get("directional_consistency", 0.0),
        row.get("wrong_side_penalty", 0.0),
        row.get("cross_coeff_abs_ratio", 1.0),
        row.get("near_zero_gap", 1.0),
        abs(row.get("balance_coeff", 0.0)),
        row.get("balance_gap", 1.0),
    )
    if include_object_quality:
        key = key + (-row.get("object_quality", 0.0),)
    return key


def select_best_custom_prompt_indices(
    pos_mat,
    neg_mat,
    coeffs,
    candidates,
    num_cases: int,
    unique_caption: bool,
    strict_center_ratio: float,
    strict_near_zero_gap: float,
    strict_orientation_margin: float,
    strict_wrong_side_max: float,
    strict_directional_consistency: float,
    min_object_quality: float,
):
    ranked = []
    for i in range(pos_mat.shape[0]):
        quality = summarize_curve_quality(pos_mat[i], neg_mat[i], coeffs)
        row = {"idx": i, **quality, **candidates[i]}
        row["quality_tier"] = _quality_tier(
            row,
            strict_center_ratio=strict_center_ratio,
            strict_near_zero_gap=strict_near_zero_gap,
            strict_orientation_margin=strict_orientation_margin,
            strict_wrong_side_max=strict_wrong_side_max,
            strict_directional_consistency=strict_directional_consistency,
            min_object_quality=min_object_quality,
        )
        ranked.append(row)

    ranked = sorted(ranked, key=lambda x: _quality_sort_key(x, include_object_quality=True))
    pools = [ranked]

    selected = []
    selected_idx = set()
    selected_captions = set()
    for pool in pools:
        for row in pool:
            if row["idx"] in selected_idx:
                continue
            if unique_caption and row["caption"] in selected_captions:
                continue
            selected.append(row)
            selected_idx.add(row["idx"])
            selected_captions.add(row["caption"])
            if len(selected) >= num_cases:
                return selected
    return selected


def completion_scores(model, prompt: str, completions: list[str], layer: int, intervene_func):
    scores = []
    for completion in completions:
        token_ids = model.tokenizer.encode(completion, add_special_tokens=False)
        if len(token_ids) == 0:
            scores.append(float("-inf"))
            continue
        context = prompt
        logps = []
        for tok_id in token_ids:
            logits = model.get_logits([context], layer=layer, intervene_func=intervene_func)
            log_probs = F.log_softmax(logits[0, -1, :], dim=-1)
            logps.append(float(log_probs[tok_id].item()))
            context += model.tokenizer.decode([tok_id])
        scores.append(float(np.mean(logps)))
    return scores


def forced_choice_fillin_curve(model, prompts, layer, steering_vec, coeffs, spatial_completions, descriptive_completions):
    """Compute average completion probabilities (not renormalized to a 2-way forced choice).

    For each prompt and steering coeff, we:
    - score a small set of spatial completions and descriptive completions
    - convert their mean log-probs back to probabilities
    - average within each set separately

    The resulting curves are 'how probable are spatial completions' vs
    'how probable are descriptive completions', without forcing them to sum to 1.
    """
    pos_mat = np.zeros((len(prompts), len(coeffs)), dtype=np.float64)
    neg_mat = np.zeros((len(prompts), len(coeffs)), dtype=np.float64)
    for j, coeff in enumerate(coeffs):
        intervene = get_intervention_func(steering_vec, method="constant", coeff=coeff)
        for i, prompt in enumerate(prompts):
            pos_scores = completion_scores(model, prompt, spatial_completions, layer, intervene)
            neg_scores = completion_scores(model, prompt, descriptive_completions, layer, intervene)

            pos_probs = [float(np.exp(s)) for s in pos_scores]
            neg_probs = [float(np.exp(s)) for s in neg_scores]

            pos_mat[i, j] = float(np.mean(pos_probs)) if pos_probs else 0.0
            neg_mat[i, j] = float(np.mean(neg_probs)) if neg_probs else 0.0

    return pos_mat, neg_mat


def select_best_indices(
    pos_mat,
    neg_mat,
    coeffs,
    num_cases: int,
    strict_center_ratio: float,
    strict_near_zero_gap: float,
    strict_orientation_margin: float,
    strict_wrong_side_max: float,
    strict_directional_consistency: float,
):
    ranked = []
    for i in range(pos_mat.shape[0]):
        quality = summarize_curve_quality(pos_mat[i], neg_mat[i], coeffs)
        row = {"idx": i, **quality}
        row["quality_tier"] = _quality_tier(
            row,
            strict_center_ratio=strict_center_ratio,
            strict_near_zero_gap=strict_near_zero_gap,
            strict_orientation_margin=strict_orientation_margin,
            strict_wrong_side_max=strict_wrong_side_max,
            strict_directional_consistency=strict_directional_consistency,
            min_object_quality=None,
        )
        ranked.append(row)

    ranked = sorted(ranked, key=lambda x: _quality_sort_key(x, include_object_quality=False))
    return [x["idx"] for x in ranked[:num_cases]]


def add_subplot_block(fig, row, col, coeffs, pos_curve, neg_curve, title_text, showlegend):
    fig.add_trace(
        go.Scatter(x=coeffs, y=neg_curve, mode="lines+markers", name="descriptive", line=dict(color="#d95f02"), showlegend=showlegend),
        row=row, col=col,
    )
    fig.add_trace(
        go.Scatter(x=coeffs, y=pos_curve, mode="lines+markers", name="spatial", line=dict(color="#1b9e77"), showlegend=showlegend),
        row=row, col=col,
    )
    fig.add_vline(x=0, line_dash="solid", line_color="black", row=row, col=col)
    gap = np.abs(pos_curve - neg_curve)
    balance = float(coeffs[int(np.argmin(gap))])
    fig.add_vline(x=balance, line_dash="dash", line_color="gray", row=row, col=col)
    local_max = float(max(np.max(pos_curve), np.max(neg_curve)))
    y_max = float(min(1.0, max(0.08, 1.18 * local_max)))
    fig.update_yaxes(title_text="Class prob", range=[0.0, y_max], row=row, col=col)
    fig.update_xaxes(title_text="Steering coeff (lambda)", row=row, col=col)
    fig.add_annotation(
        xref=f"x{((row - 1) * 3 + col)}" if (row, col) != (1, 1) else "x",
        yref=f"y{((row - 1) * 3 + col)}" if (row, col) != (1, 1) else "y",
        x=coeffs[0],
        y=1.08,
        text=title_text,
        showarrow=False,
        xanchor="left",
        font=dict(size=10),
    )
    return balance


def main():
    args = parse_args()
    torch.set_grad_enabled(False)
    random.seed(args.template_seed)
    np.random.seed(args.template_seed)

    model = load_model(args.model_name)
    artifact_dir = Path(args.artifact_dir)
    candidate_vectors = torch.load(artifact_dir / "activations/candidate_vectors.pt")
    steering_vec = model.set_dtype(candidate_vectors[args.layer])

    words = load_target_words(target_concept="vision")
    pos_ids = get_target_token_ids(model.tokenizer, words["spatial"])
    neg_ids = get_target_token_ids(model.tokenizer, words["descriptive"])
    overlap = set(pos_ids).intersection(set(neg_ids))
    if overlap:
        pos_ids = [x for x in pos_ids if x not in overlap]
        neg_ids = [x for x in neg_ids if x not in overlap]

    coeffs = [float(x) for x in loop_coeffs(args.min_coeff, args.max_coeff, args.increment)]
    if 0.0 not in coeffs:
        coeffs.append(0.0)
        coeffs = sorted(coeffs)

    if args.use_curated_manual_v2:
        captions = [case["caption"] for case in CURATED_MANUAL_V2_CASES]
    else:
        if args.captions_file:
            captions = _load_captions_from_file(Path(args.captions_file))
        else:
            val_path = artifact_dir / "datasplits/val.json"
            if val_path.exists():
                rows = json.loads(val_path.read_text())
                captions = [x["text"] for x in rows]
            else:
                captions = IMAGE_SHOWS_ONE_TOKEN + IMAGE_SHOWS_MULTI_TOKEN

        if args.search_pool > 0:
            captions = captions[: args.search_pool]

    prompts_1 = build_image_shows_prompts(captions)
    prompts_2 = build_image_shows_prompts(captions)
    prompts_1_raw = list(prompts_1)
    prompts_2_raw = list(prompts_2)

    if args.use_curated_manual_v2:
        template_rows = []
        template_ranking = []
        custom_screen_ranking = []
        custom_candidates = []
        custom_prompts_raw = []
        for case in CURATED_MANUAL_V2_CASES:
            obj = case["object"]
            cap = case["caption"]
            grammar = _prompt_grammar_forms(obj)
            prompt = (
                f"Caption: {cap}\n"
                f"Finish the phrase about the {obj}: "
                f"The {obj} {grammar['be_verb']}"
            )
            custom_candidates.append(
                {
                    "caption": cap,
                    "object": obj,
                    "object_quality": 0.0,
                    "be_verb": grammar["be_verb"],
                    "template_id": "curated_manual_v2",
                    "template_source": "manual",
                    "prompt": prompt,
                }
            )
            custom_prompts_raw.append(prompt)
        custom_candidate_count_total = len(custom_candidates)
    else:
        seed_template_rows = [
            {"template_id": x["template_id"], "template": x["template"], "source": "seed"}
            for x in CUSTOM_FILLIN_TEMPLATES
        ]
        template_rows = seed_template_rows
        template_ranking = []
        if args.template_auto_generate:
            prog_templates = generate_programmatic_templates(
                count=args.template_programmatic_pool,
                seed=args.template_seed,
            )
            llm_templates = generate_llm_templates(
                model,
                n_samples=args.template_llm_samples,
                use_chat_template=args.use_chat_template,
            )
            template_pool = _dedupe_template_rows(seed_template_rows + prog_templates + llm_templates)
            template_rows, template_ranking = rank_template_pool(
                model=model,
                template_rows=template_pool,
                captions=captions,
                layer=args.layer,
                steering_vec=steering_vec,
                coeffs=coeffs,
                screen_cases=args.template_screen_cases,
                keep_top=args.template_keep_top,
                use_chat_template=args.use_chat_template,
                strict_center_ratio=args.strict_center_ratio,
                strict_near_zero_gap=args.strict_near_zero_gap,
                strict_orientation_margin=args.strict_orientation_margin,
                strict_wrong_side_max=args.strict_wrong_side_max,
                strict_directional_consistency=args.strict_directional_consistency,
            )

        custom_candidates = build_custom_fillin_candidates(
            captions,
            max_objects=args.custom_max_objects,
            template_rows=template_rows,
        )
        custom_candidate_count_total = len(custom_candidates)
        custom_screen_ranking = []
    custom_candidates, custom_screen_ranking = screen_custom_fillin_candidates(
        model=model,
        candidates=custom_candidates,
        layer=args.layer,
        steering_vec=steering_vec,
        coeffs=coeffs,
        use_chat_template=args.use_chat_template,
        top_k=args.custom_screen_top_k,
        strict_center_ratio=args.strict_center_ratio,
        strict_near_zero_gap=args.strict_near_zero_gap,
        strict_orientation_margin=args.strict_orientation_margin,
        strict_wrong_side_max=args.strict_wrong_side_max,
        strict_directional_consistency=args.strict_directional_consistency,
        min_object_quality=None,
    )
    custom_prompts_raw = [x["prompt"] for x in custom_candidates]

    custom_prompts_eval = custom_prompts_raw
    if args.use_chat_template:
        prompts_1 = model.apply_chat_template(prompts_1)
        prompts_2 = model.apply_chat_template(prompts_2)
        custom_prompts_eval = model.apply_chat_template(custom_prompts_raw)

    one_pos, one_neg = next_token_class_prob_curve(
        model, prompts_1, args.layer, steering_vec, coeffs, pos_ids, neg_ids, constrained=args.constrained
    )
    multi_image_pos, multi_image_neg = teacher_forced_multi_token_curve(
        model, prompts_2, args.layer, steering_vec, coeffs, pos_ids, neg_ids, args.multi_tokens, constrained=args.constrained
    )
    multi_custom_pos, multi_custom_neg = next_token_class_prob_curve(
        model,
        custom_prompts_eval,
        args.layer,
        steering_vec,
        coeffs,
        pos_ids,
        neg_ids,
        constrained=args.constrained,
    )

    if args.use_curated_manual_v2:
        n_manual = min(args.num_cases, len(captions), multi_custom_pos.shape[0])
        idx_1 = list(range(n_manual))
        idx_2 = list(range(n_manual))
        selected_custom = []
        for i in range(n_manual):
            quality = summarize_curve_quality(multi_custom_pos[i], multi_custom_neg[i], coeffs)
            row = {"idx": i, **quality, **custom_candidates[i]}
            row["quality_tier"] = _quality_tier(
                row,
                strict_center_ratio=args.strict_center_ratio,
                strict_near_zero_gap=args.strict_near_zero_gap,
                strict_orientation_margin=args.strict_orientation_margin,
                strict_wrong_side_max=args.strict_wrong_side_max,
                strict_directional_consistency=args.strict_directional_consistency,
                min_object_quality=None,
            )
            selected_custom.append(row)
    else:
        idx_1 = select_best_indices(
            one_pos,
            one_neg,
            coeffs,
            args.num_cases,
            strict_center_ratio=args.strict_center_ratio,
            strict_near_zero_gap=args.strict_near_zero_gap,
            strict_orientation_margin=args.strict_orientation_margin,
            strict_wrong_side_max=args.strict_wrong_side_max,
            strict_directional_consistency=args.strict_directional_consistency,
        )
        idx_2 = select_best_indices(
            multi_image_pos,
            multi_image_neg,
            coeffs,
            args.num_cases,
            strict_center_ratio=args.strict_center_ratio,
            strict_near_zero_gap=args.strict_near_zero_gap,
            strict_orientation_margin=args.strict_orientation_margin,
            strict_wrong_side_max=args.strict_wrong_side_max,
            strict_directional_consistency=args.strict_directional_consistency,
        )
    selected_custom = select_best_custom_prompt_indices(
        multi_custom_pos,
        multi_custom_neg,
        coeffs,
        custom_candidates,
        args.num_cases,
        unique_caption=not args.custom_allow_repeat_captions,
        strict_center_ratio=args.strict_center_ratio,
        strict_near_zero_gap=args.strict_near_zero_gap,
        strict_orientation_margin=args.strict_orientation_margin,
        strict_wrong_side_max=args.strict_wrong_side_max,
        strict_directional_consistency=args.strict_directional_consistency,
        min_object_quality=None,
    )

    n_cases = min(args.num_cases, len(idx_1), len(idx_2), len(selected_custom))
    if n_cases <= 0:
        raise RuntimeError("No valid cases were selected. Increase --search_pool or relax custom prompt constraints.")

    beam_compare_summary = None
    if args.beam_compare:
        compare_coeff = _resolve_beam_compare_coeff(artifact_dir, args.beam_compare_coeff)

        sel_one_pos = np.asarray([one_pos[idx_1[i]] for i in range(n_cases)], dtype=np.float64)
        sel_one_neg = np.asarray([one_neg[idx_1[i]] for i in range(n_cases)], dtype=np.float64)

        sel_multi_greedy_pos = np.asarray([multi_image_pos[idx_2[i]] for i in range(n_cases)], dtype=np.float64)
        sel_multi_greedy_neg = np.asarray([multi_image_neg[idx_2[i]] for i in range(n_cases)], dtype=np.float64)
        sel_multi_prompts = [prompts_2[idx_2[i]] for i in range(n_cases)]
        sel_multi_beam_pos, sel_multi_beam_neg = teacher_forced_multi_token_curve_beam(
            model=model,
            prompts=sel_multi_prompts,
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

        sel_custom_prompts = [custom_prompts_eval[selected_custom[i]["idx"]] for i in range(n_cases)]
        sel_custom_pos, sel_custom_neg = continuation_multi_token_curve_greedy(
            model=model,
            prompts=sel_custom_prompts,
            layer=args.layer,
            steering_vec=steering_vec,
            coeffs=coeffs,
            pos_ids=pos_ids,
            neg_ids=neg_ids,
            n_tokens=args.multi_tokens,
            constrained=args.constrained,
        )
        sel_custom_beam_pos, sel_custom_beam_neg = continuation_multi_token_curve_beam(
            model=model,
            prompts=sel_custom_prompts,
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

        one_greedy = _scenario_delta_metrics(sel_one_pos, sel_one_neg, coeffs, compare_coeff)
        one_beam = _scenario_delta_metrics(sel_one_pos, sel_one_neg, coeffs, compare_coeff)
        multi_greedy = _scenario_delta_metrics(sel_multi_greedy_pos, sel_multi_greedy_neg, coeffs, compare_coeff)
        multi_beam = _scenario_delta_metrics(sel_multi_beam_pos, sel_multi_beam_neg, coeffs, compare_coeff)
        fill_greedy = _scenario_delta_metrics(sel_custom_pos, sel_custom_neg, coeffs, compare_coeff)
        fill_beam = _scenario_delta_metrics(sel_custom_beam_pos, sel_custom_beam_neg, coeffs, compare_coeff)

        def _bundle(greedy_metrics, beam_metrics):
            return {
                "greedy": greedy_metrics,
                "beam": beam_metrics,
                "beam_minus_greedy_mean_diff_delta": float(
                    beam_metrics["mean_diff_delta"] - greedy_metrics["mean_diff_delta"]
                ),
                "beam_minus_greedy_gap_reduction_pct": float(
                    beam_metrics["gap_reduction_pct"] - greedy_metrics["gap_reduction_pct"]
                ),
            }

        beam_compare_summary = {
            "enabled": True,
            "beam_width": int(args.beam_width),
            "beam_top_k": int(args.beam_top_k),
            "compare_coeff_requested": float(compare_coeff),
            "scenarios": {
                "image_shows_one_token": _bundle(one_greedy, one_beam),
                "image_shows_multi_token_mean": _bundle(multi_greedy, multi_beam),
                "custom_fill_in_blank": _bundle(fill_greedy, fill_beam),
            },
            "notes": [
                "image_shows_one_token remains next-token; beam equals greedy by construction.",
                "beam search is applied to the multi-token mean scenario using fixed baseline beam reference paths (teacher-forced scoring).",
                "custom_fill_in_blank beam compare uses open-ended multi-token continuation from the fill-in prompt.",
            ],
        }

    fig = make_subplots(
        rows=n_cases,
        cols=3,
        shared_xaxes=False,
        vertical_spacing=0.045,
        horizontal_spacing=0.04,
        subplot_titles=[
            x
            for i in range(n_cases)
            for x in (
                f"Case {i+1}: image shows (1 token)",
                f"Case {i+1}: image shows ({args.multi_tokens} token mean)",
                f"Case {i+1}: custom object fill-in",
            )
        ],
    )

    cases = []
    for i in range(n_cases):
        i1, i2 = idx_1[i], idx_2[i]
        custom_pick = selected_custom[i]
        i3 = custom_pick["idx"]

        # Plotting still uses the full curves, but JSON only keeps the text prompts.
        custom_title = (
            f"{custom_pick['caption']} | object={custom_pick['object']} | "
            f"template={custom_pick['template_id']}"
        )
        if len(custom_title) > 150:
            custom_title = custom_title[:147] + "..."
        add_subplot_block(fig, i + 1, 1, coeffs, one_pos[i1], one_neg[i1], captions[i1], showlegend=(i == 0))
        add_subplot_block(fig, i + 1, 2, coeffs, multi_image_pos[i2], multi_image_neg[i2], captions[i2], showlegend=False)
        add_subplot_block(
            fig,
            i + 1,
            3,
            coeffs,
            multi_custom_pos[i3],
            multi_custom_neg[i3],
            custom_title,
            showlegend=False,
        )

        cases.append(
            {
                "case": i + 1,
                "image_shows_one_token_caption": captions[i1],
                "image_shows_one_token_prompt": prompts_1_raw[i1],
                "image_shows_multi_token_caption": captions[i2],
                "image_shows_multi_token_prompt": prompts_2_raw[i2],
                "custom_caption": custom_pick["caption"],
                "custom_object": custom_pick["object"],
                "custom_template_id": custom_pick.get("template_id", ""),
                "custom_prompt": custom_prompts_raw[i3],
            }
        )

    fig.update_layout(
        title=f"Master prompt experiments ({args.model_name}, layer {args.layer})",
        height=max(1700, 330 * n_cases),
        width=1700,
        template="plotly_white",
    )

    out_html = Path(args.output_html)
    out_json = Path(args.output_json)
    out_png = Path(args.output_png) if args.output_png else None
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    if out_png is not None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html))
    if out_png is not None:
        try:
            fig.write_image(str(out_png), scale=2)
        except Exception as exc:
            print(f"Warning: failed to write PNG: {out_png} ({exc})")
    if args.use_curated_manual_v2:
        # For the curated setup, keep the JSON lightweight: just metadata and captions.
        out_json.write_text(
            json.dumps(
                {
                    "model_name": args.model_name,
                    "layer": int(args.layer),
                    "coeffs": coeffs,
                    "beam_compare": beam_compare_summary,
                    "cases": cases,
                },
                indent=2,
            )
        )
    else:
        out_json.write_text(
            json.dumps(
                {
                    "model_name": args.model_name,
                    "artifact_dir": str(artifact_dir),
                    "layer": int(args.layer),
                    "coeffs": coeffs,
                    "multi_tokens": int(args.multi_tokens),
                    "requested_num_cases": int(args.num_cases),
                    "num_cases": int(n_cases),
                    "search_pool": int(args.search_pool),
                    "custom_max_objects": int(args.custom_max_objects),
                    "custom_allow_repeat_captions": bool(args.custom_allow_repeat_captions),
                    "custom_candidate_count_total": int(custom_candidate_count_total),
                    "custom_candidate_count": int(len(custom_candidates)),
                    "custom_screen_top_k": int(args.custom_screen_top_k),
                    "strict_center_ratio": float(args.strict_center_ratio),
                    "strict_near_zero_gap": float(args.strict_near_zero_gap),
                    "strict_orientation_margin": float(args.strict_orientation_margin),
                    "strict_wrong_side_max": float(args.strict_wrong_side_max),
                    "strict_directional_consistency": float(args.strict_directional_consistency),
                    "template_auto_generate": bool(args.template_auto_generate),
                    "template_programmatic_pool": int(args.template_programmatic_pool),
                    "template_llm_samples": int(args.template_llm_samples),
                    "template_keep_top": int(args.template_keep_top),
                    "template_screen_cases": int(args.template_screen_cases),
                    "template_seed": int(args.template_seed),
                    "custom_template_count_used": int(len(template_rows)),
                    "custom_template_rows_used": [
                        {
                            "template_id": x["template_id"],
                            "source": x.get("source", "seed"),
                            "template": x["template"],
                            "template_screen_score": x.get("template_screen_score"),
                            "template_screen_tier": x.get("template_screen_tier"),
                            "template_screen_mean_score": x.get("template_screen_mean_score"),
                            "template_screen_center_swing": x.get("template_screen_center_swing"),
                            "template_screen_near_zero_gap": x.get("template_screen_near_zero_gap"),
                            "template_screen_orientation_margin": x.get("template_screen_orientation_margin"),
                            "template_screen_wrong_side": x.get("template_screen_wrong_side"),
                            "template_screen_cross_ratio": x.get("template_screen_cross_ratio"),
                            "template_screen_directional_consistency": x.get(
                                "template_screen_directional_consistency"
                            ),
                        }
                        for x in template_rows
                    ],
                    "custom_template_ranking_top20": [
                        {
                            "template_id": x["template_id"],
                            "source": x.get("source", "seed"),
                            "template": x["template"],
                            "template_screen_score": x.get("template_screen_score"),
                            "template_screen_tier": x.get("template_screen_tier"),
                            "template_screen_mean_score": x.get("template_screen_mean_score"),
                            "template_screen_center_swing": x.get("template_screen_center_swing"),
                            "template_screen_near_zero_gap": x.get("template_screen_near_zero_gap"),
                            "template_screen_orientation_margin": x.get("template_screen_orientation_margin"),
                            "template_screen_wrong_side": x.get("template_screen_wrong_side"),
                            "template_screen_cross_ratio": x.get("template_screen_cross_ratio"),
                            "template_screen_directional_consistency": x.get(
                                "template_screen_directional_consistency"
                            ),
                        }
                        for x in template_ranking[:20]
                    ],
                    "custom_screen_ranking_top20": [
                        {
                            "caption": x["caption"],
                            "object": x["object"],
                            "template_id": x["template_id"],
                            "quality_tier": x.get("quality_tier"),
                            "score": x.get("score"),
                            "orientation_margin": x.get("orientation_margin"),
                            "wrong_side_penalty": x.get("wrong_side_penalty"),
                            "cross_coeff_abs_ratio": x.get("cross_coeff_abs_ratio"),
                            "near_zero_gap": x.get("near_zero_gap"),
                            "directional_consistency": x.get("directional_consistency"),
                        }
                        for x in custom_screen_ranking[:20]
                    ],
                    "constrained": bool(args.constrained),
                    "use_chat_template": bool(args.use_chat_template),
                    "output_png": str(out_png) if out_png is not None else "",
                    "beam_compare": beam_compare_summary,
                    "cases": cases,
                },
                indent=2,
            )
        )
    print(f"Saved HTML: {out_html}")
    print(f"Saved JSON: {out_json}")
    if beam_compare_summary is not None:
        m = beam_compare_summary["scenarios"]["image_shows_multi_token_mean"]
        print(
            "Beam compare (multi-token mean): "
            f"greedy gap_reduction_pct={m['greedy']['gap_reduction_pct']:+.2f}%, "
            f"beam gap_reduction_pct={m['beam']['gap_reduction_pct']:+.2f}%, "
            f"beam-greedy={m['beam_minus_greedy_gap_reduction_pct']:+.2f}%"
        )
    if out_png is not None:
        print(f"Saved PNG: {out_png}")

    if args.open_browser:
        import webbrowser

        webbrowser.open(out_html.resolve().as_uri())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Professor-friendly demo of next-token prediction and steering.

Shows three things in plain terms:
1) The model predicts a full probability distribution for the next token.
2) We can inspect top next-token candidates for any prompt.
3) Steering shifts that distribution (baseline vs steered).

Includes both normal prompts and project-style prompts that end with an
output prefix ("The scene is").
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

from bias_steering.steering import load_model, get_intervention_func
from bias_steering.data.load_dataset import load_target_words
from bias_steering.steering.steering_utils import get_target_token_ids


def format_token(token: str) -> str:
    return token.replace("\n", "\\n")


def top_next_tokens(model, prompt: str, top_k: int, layer: int | None = None, intervene_func=None):
    logits = model.get_logits([prompt], layer=layer, intervene_func=intervene_func)
    probs = F.softmax(logits[0, -1, :], dim=-1)
    top_probs, top_ids = torch.topk(probs, k=top_k)

    rows = []
    for rank, (tok_id, p) in enumerate(zip(top_ids.tolist(), top_probs.tolist()), start=1):
        tok = model.tokenizer.decode([tok_id])
        rows.append((rank, tok, p))
    return rows, probs


def class_sums(probs: torch.Tensor, pos_ids: list[int], neg_ids: list[int]):
    pos_prob = probs[pos_ids].sum().item()
    neg_prob = probs[neg_ids].sum().item()
    return pos_prob, neg_prob, pos_prob - neg_prob


def print_case(
    model,
    title: str,
    prompt: str,
    top_k: int,
    pos_ids: list[int],
    neg_ids: list[int],
    layer: int,
    base_func,
    steer_func,
):
    print("=" * 80)
    print(title)
    print("- Prompt:")
    print(prompt)
    print()

    base_rows, base_probs = top_next_tokens(model, prompt, top_k=top_k, layer=layer, intervene_func=base_func)
    steer_rows, steer_probs = top_next_tokens(model, prompt, top_k=top_k, layer=layer, intervene_func=steer_func)

    b_pos, b_neg, b_bias = class_sums(base_probs, pos_ids, neg_ids)
    s_pos, s_neg, s_bias = class_sums(steer_probs, pos_ids, neg_ids)

    print("Baseline top next tokens:")
    for rank, tok, p in base_rows:
        print(f"  {rank:>2}. token={format_token(tok)!r:>14}  prob={p:.4f}")

    print("Steered top next tokens:")
    for rank, tok, p in steer_rows:
        print(f"  {rank:>2}. token={format_token(tok)!r:>14}  prob={p:.4f}")

    print("Class probability sums (next-token distribution):")
    print(f"  Baseline: spatial={b_pos:.6f}  descriptive={b_neg:.6f}  diff={b_bias:+.6f}")
    print(f"  Steered : spatial={s_pos:.6f}  descriptive={s_neg:.6f}  diff={s_bias:+.6f}")
    print(f"  Delta diff (steered - baseline): {s_bias - b_bias:+.6f}")
    print()


def build_project_prompt(model, caption: str) -> str:
    user_text = f"Continue describing this scene:\n{caption}"
    return model.apply_chat_template([user_text])[0] + "The scene is"


def main():
    parser = argparse.ArgumentParser(description="Demo next-token prediction for class discussion.")
    parser.add_argument("--model_name", default="gpt2")
    parser.add_argument("--artifact_dir", default="runs_vision/gpt2")
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--coeff", type=float, default=-200.0)
    parser.add_argument("--top_k", type=int, default=8)
    args = parser.parse_args()

    print("Loading model and steering vector...")
    model = load_model(args.model_name)

    artifact_dir = Path(args.artifact_dir)
    candidate_vectors = torch.load(artifact_dir / "activations/candidate_vectors.pt")
    steering_vec = model.set_dtype(candidate_vectors[args.layer])

    base_func = get_intervention_func(steering_vec, method="constant", coeff=0.0)
    steer_func = get_intervention_func(steering_vec, method="constant", coeff=args.coeff)

    target_words = load_target_words(target_concept="vision")
    pos_ids = get_target_token_ids(model.tokenizer, target_words["spatial"])
    neg_ids = get_target_token_ids(model.tokenizer, target_words["descriptive"])

    print(f"Model: {args.model_name}")
    print(f"Layer: {args.layer}")
    print(f"Steering coeff: {args.coeff}")
    print(f"Tracking class token ids: spatial={len(pos_ids)}, descriptive={len(neg_ids)}")
    print()

    normal_prompts = [
        "Finish the sentence: The cat sat on the",
        "Finish the sentence: The bus is parked next to the",
        "Complete this: The bright yellow kite is flying",
    ]

    scene_captions = [
        "A red bus parked beside a curb near a crosswalk.",
        "A black and white cat sleeping on a blue sofa.",
        "A train passing under a bridge next to a road.",
    ]

    for i, prompt in enumerate(normal_prompts, start=1):
        print_case(
            model,
            title=f"Normal Prompt {i}",
            prompt=prompt,
            top_k=args.top_k,
            pos_ids=pos_ids,
            neg_ids=neg_ids,
            layer=args.layer,
            base_func=base_func,
            steer_func=steer_func,
        )

    for i, caption in enumerate(scene_captions, start=1):
        project_prompt = build_project_prompt(model, caption)
        print_case(
            model,
            title=f"Project-Style Prompt {i}",
            prompt=project_prompt,
            top_k=args.top_k,
            pos_ids=pos_ids,
            neg_ids=neg_ids,
            layer=args.layer,
            base_func=base_func,
            steer_func=steer_func,
        )

    print("Demo complete.")


if __name__ == "__main__":
    main()

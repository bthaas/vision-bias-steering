#!/usr/bin/env python3
"""Generate vision steering case studies with token-level summaries.

WHAT THIS DOES:
1. Takes 5 example image captions (hardcoded below)
2. For each caption, tests different "steering coefficients" (λ) that push the model
   toward either spatial words (left, right, above) or descriptive words (red, blue, large)
3. Measures how the probability of spatial vs descriptive tokens changes as we vary λ
4. Creates plots showing these probability curves and identifies the "balance point"
   where spatial and descriptive probabilities are equal
5. Records the top predicted token at key coefficients (negative edge, zero, balance, positive edge)

The steering vector is learned from training data and allows us to control model behavior
by adding it to hidden activations at a specific layer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bias_steering.data.load_dataset import load_target_words
from bias_steering.steering import get_intervention_func, load_model
from bias_steering.steering.steering_utils import get_target_token_ids
from bias_steering.utils import loop_coeffs



EXAMPLE_CAPTIONS = [
    "A jet airplane is flying through the sky just above the highway as a group of cars stay to the left side of the road.",
    "A large beige living room with ceiling fan and beige chair and couches.",
    "A SMALL LIVING ROOM WITH A FLAT SCREEN TV AND A BLUE RUG ",
    "A young boy with a wide smile showing off his white suit, lavender shirt and purple striped tie.",
    "A man and two dogs in a horse-drawn carriage in the middle of the road beside a brick building",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate vision steering case studies.")
    parser.add_argument("--model_name", default="gpt2")
    parser.add_argument("--artifact_dir", default="runs_vision/gpt2")
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--min_coeff", type=float, default=-240.0)
    parser.add_argument("--max_coeff", type=float, default=240.0)
    parser.add_argument("--increment", type=float, default=20.0)
    parser.add_argument("--candidate_pool", type=int, default=180)
    parser.add_argument("--num_cases", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--val_json", default="runs_vision/gpt2/datasplits/val.json")
    parser.add_argument("--output_html", default="plots/vision_steering_cases.html")
    parser.add_argument("--output_json", default="plots/vision_steering_cases.json")
    return parser.parse_args()


def compute_class_probs(model, prompts, layer, intervene_func, pos_ids, neg_ids, batch_size):
    """Compute spatial and descriptive class probabilities for all prompts.
    
    This function asks the model to predict the next word for each prompt.
    Then it adds up all the probabilities for spatial words (like 'left', 'right', 'above')
    and all the probabilities for descriptive words (like 'red', 'blue', 'large').
    We process prompts in batches to save memory.
    """
    pos_all, neg_all = [], []
    # Process prompts in small groups (batches) to avoid running out of memory
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        # Get the model's predictions for what word comes next
        logits = model.get_logits(batch, layer=layer, intervene_func=intervene_func)
        # Convert raw scores into probabilities (they add up to 1.0)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        # Add up probabilities for all spatial words
        pos_all.append(probs[:, pos_ids].sum(dim=-1).detach().cpu().numpy())
        # Add up probabilities for all descriptive words
        neg_all.append(probs[:, neg_ids].sum(dim=-1).detach().cpu().numpy())
    # Combine all batches into one big list
    return np.concatenate(pos_all), np.concatenate(neg_all)


def get_top_token(model, prompt, layer, intervene_func):
    """Get the most likely next word and how confident the model is.
    
    This function asks the model what word it thinks comes next in the prompt.
    It returns the word with the highest probability and that probability value.
    """
    # Get model's predictions for what word comes next
    logits = model.get_logits([prompt], layer=layer, intervene_func=intervene_func)
    # Convert scores into probabilities
    probs = F.softmax(logits[0, -1, :], dim=-1)
    # Find which word has the highest probability
    tok_id = int(torch.argmax(probs).item())
    # Convert the word ID back into actual text
    token = model.tokenizer.decode([tok_id]).replace("\n", "\\n")
    # Return the word and its probability
    return token, float(probs[tok_id].item())


def select_cases(rows, coeffs, pos_mat, neg_mat, num_cases):
    """Pick the best examples to show in our plots.
    
    We want examples where steering actually works - meaning:
    - When we push toward descriptive words (negative coefficient), descriptive words win
    - When we push toward spatial words (positive coefficient), spatial words win
    - There's a nice balance point where both are equally likely
    """
    # Find which coefficient is closest to zero (no steering)
    mid_idx = int(np.argmin(np.abs(np.array(coeffs))))
    candidates = []
    
    # Look at each example and see how well steering works
    for i, row in enumerate(rows):
        # At the most negative coefficient, how much better are descriptive words?
        neg_edge = float(neg_mat[i, 0] - pos_mat[i, 0])
        # At the most positive coefficient, how much better are spatial words?
        pos_edge = float(pos_mat[i, -1] - neg_mat[i, -1])
        # Find where spatial and descriptive probabilities are closest (balance point)
        diffs = np.abs(pos_mat[i] - neg_mat[i])
        balance_idx = int(np.argmin(diffs))
        balance_gap = float(diffs[balance_idx])
        # How close are they at zero steering?
        near_zero_gap = float(diffs[mid_idx])
        
        # Score: higher is better (strong steering effects, good balance)
        score = (neg_edge + pos_edge) - (2.0 * balance_gap) - near_zero_gap
        candidates.append({
            "idx": i,
            "text": row["text"],
            "neg_edge": neg_edge,
            "pos_edge": pos_edge,
            "balance_idx": balance_idx,
            "balance_gap": balance_gap,
            "near_zero_gap": near_zero_gap,
            "score": score,
        })
    
    # Only keep examples where steering works in both directions
    strict = [c for c in candidates if c["neg_edge"] > 0 and c["pos_edge"] > 0]
    # Sort by score (best first), then by how close to balance
    strict.sort(key=lambda x: (-x["score"], x["balance_gap"]))
    
    if len(strict) >= num_cases:
        return strict[:num_cases]
    
    # If we don't have enough good examples, use all of them sorted by score
    all_sorted = sorted(candidates, key=lambda x: (-x["score"], x["balance_gap"]))
    merged = strict + [c for c in all_sorted if c["idx"] not in {s["idx"] for s in strict}]
    return merged[:num_cases]


def build_plot(selected, coeffs, pos_mat, neg_mat, prompts, candidate_rows, model, steering_vec, layer):
    """Create the plots showing how steering affects word probabilities.
    
    For each example caption, we make a plot with:
    - Orange line: probability of descriptive words at different steering strengths
    - Green line: probability of spatial words at different steering strengths
    - Vertical lines: zero steering (black) and balance point (gray dashed)
    - Caption text at the top
    - Top predicted word at 4 key points
    """
    n_cases = len(selected)
    # Create a figure with one subplot per example (stacked vertically)
    fig = make_subplots(
        rows=n_cases,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.06,
        subplot_titles=[f"Case {i + 1}" for i in range(n_cases)],
    )
    
    summary = []
    # Find which coefficient is zero (no steering)
    mid_idx = int(np.argmin(np.abs(np.array(coeffs))))
    
    # Build plot for each example
    for row_num, case in enumerate(selected, start=1):
        idx = case["idx"]
        # Get the probability curves for this example
        pos_curve = pos_mat[idx]  # Spatial word probabilities
        neg_curve = neg_mat[idx]  # Descriptive word probabilities
        balance_coeff = coeffs[case["balance_idx"]]  # Where they cross
        prompt = prompts[idx]
        
        # Draw the orange line showing descriptive word probabilities
        fig.add_trace(
            go.Scatter(
                x=coeffs, y=neg_curve,
                mode="lines+markers",
                name="descriptive",
                line=dict(color="#d95f02"),  # Orange color
                showlegend=(row_num == 1),  # Only show legend on first plot
            ),
            row=row_num, col=1,
        )
        # Draw the green line showing spatial word probabilities
        fig.add_trace(
            go.Scatter(
                x=coeffs, y=pos_curve,
                mode="lines+markers",
                name="spatial",
                line=dict(color="#1b9e77"),  # Green color
                showlegend=(row_num == 1),
            ),
            row=row_num, col=1,
        )
        
        # Draw a black vertical line at zero (no steering)
        fig.add_vline(x=0, line_dash="solid", line_color="black", row=row_num, col=1)
        # Draw a gray dashed line at the balance point (where lines cross)
        fig.add_vline(x=balance_coeff, line_dash="dash", line_color="gray", row=row_num, col=1)
        
        # Check what word the model predicts at 4 important points:
        # 1. Most negative steering (strong descriptive push)
        # 2. Zero steering (baseline)
        # 3. Balance point (where probabilities are equal)
        # 4. Most positive steering (strong spatial push)
        probe_coeffs = [coeffs[0], coeffs[mid_idx], balance_coeff, coeffs[-1]]
        probe_labels = ["neg_edge", "zero", "balance", "pos_edge"]
        token_summary = {}
        
        for label, coeff in zip(probe_labels, probe_coeffs):
            # Apply steering at this coefficient
            intervene = get_intervention_func(steering_vec, method="constant", coeff=coeff)
            # Ask model what word comes next
            token, prob = get_top_token(model, prompt, layer=layer, intervene_func=intervene)
            token_summary[label] = {
                "coeff": float(coeff),
                "top_token": token.strip(),
                "top_prob": prob,
            }
        
        # Save all the information about this example
        summary.append({
            "case": row_num,
            "text": candidate_rows[idx]["text"],
            "balance_coeff": float(balance_coeff),
            "neg_edge_advantage": case["neg_edge"],
            "pos_edge_advantage": case["pos_edge"],
            "near_zero_gap": case["near_zero_gap"],
            "balance_gap": case["balance_gap"],
            "token_summary": token_summary,
        })
        
        # Add the caption text above the plot
        caption = candidate_rows[idx]["text"]
        if len(caption) > 120:
            caption = caption[:117] + "..."  # Truncate if too long
        fig.add_annotation(
            xref=f"x{row_num}" if row_num > 1 else "x",
            yref=f"y{row_num}" if row_num > 1 else "y",
            x=coeffs[0],
            y=1.08,
            text=caption,
            showarrow=False,
            xanchor="left",
            font=dict(size=12),
        )
        
        # Set axis labels and ranges
        fig.update_yaxes(title_text="Class prob", range=[0, 1], row=row_num, col=1)
        fig.update_xaxes(title_text="Steering coeff (λ)", row=row_num, col=1)
    
    # Set overall figure title and size
    fig.update_layout(
        title=f"Vision Steering Cases ({model.model_name if hasattr(model, 'model_name') else 'model'}, layer {layer})",
        height=max(340 * n_cases, 900),
        width=980,
        template="plotly_white",
    )
    
    return fig, summary


def main():
    """Main function that runs everything step by step."""
    args = parse_args()
    torch.set_grad_enabled(False)
    
    coeffs = list(loop_coeffs(min_coeff=args.min_coeff, max_coeff=args.max_coeff, increment=args.increment))
    coeffs = [float(c) for c in coeffs]
    if 0.0 not in coeffs:
        coeffs.append(0.0)
    coeffs.sort() 
    
    model = load_model(args.model_name)
    artifact_dir = Path(args.artifact_dir)
    candidate_vectors = torch.load(artifact_dir / "activations/candidate_vectors.pt")
    steering_vec = model.set_dtype(candidate_vectors[args.layer])
    
    # Get lists of which words count as "spatial" and "descriptive"
    # Convert these words into token IDs that the model understands
    target_words = load_target_words(target_concept="vision")
    pos_ids = get_target_token_ids(model.tokenizer, target_words["spatial"])
    neg_ids = get_target_token_ids(model.tokenizer, target_words["descriptive"])
    
    # "Describe this image:" as the instruction and "The image shows" as the start of the response
    candidate_rows = [{"text": caption} for caption in EXAMPLE_CAPTIONS]
    prompts = [
        model.apply_chat_template(
            [f"Describe this image:\n{caption}"],
            output_prefix="The image shows"
        )[0]
        for caption in EXAMPLE_CAPTIONS
    ]
    

    pos_mat = np.zeros((len(prompts), len(coeffs)), dtype=np.float64)
    neg_mat = np.zeros((len(prompts), len(coeffs)), dtype=np.float64)
    
    # Test each steering coefficient
    print(f"Scoring {len(prompts)} prompts across {len(coeffs)} coefficients...")
    for j, coeff in enumerate(coeffs):
        # Create a function that applies steering at this coefficient
        intervene = get_intervention_func(steering_vec, method="constant", coeff=coeff)
        # Ask the model what it thinks comes next, with steering applied
        # Get probabilities for spatial words and descriptive words
        pos, neg = compute_class_probs(
            model, prompts, args.layer, intervene, pos_ids, neg_ids, args.batch_size
        )
        # Save the results
        pos_mat[:, j] = pos  # Spatial word probabilities
        neg_mat[:, j] = neg  # Descriptive word probabilities
    
    # Since we're using hardcoded examples, we don't need to select the best ones
    # But we still need to calculate some statistics for each example
    mid_idx = int(np.argmin(np.abs(np.array(coeffs))))  # Index of zero coefficient
    selected = []
    for i in range(len(EXAMPLE_CAPTIONS)):
        # Find where spatial and descriptive probabilities are closest (balance point)
        diffs = np.abs(pos_mat[i] - neg_mat[i])
        balance_idx = int(np.argmin(diffs))
        # Calculate how much steering helps at the edges
        neg_edge = float(neg_mat[i, 0] - pos_mat[i, 0])  # At most negative coefficient
        pos_edge = float(pos_mat[i, -1] - neg_mat[i, -1])  # At most positive coefficient
        selected.append({
            "idx": i,
            "text": EXAMPLE_CAPTIONS[i],
            "neg_edge": neg_edge,
            "pos_edge": pos_edge,
            "balance_idx": balance_idx,
            "balance_gap": float(diffs[balance_idx]),
            "near_zero_gap": float(diffs[mid_idx]),
            "score": 0.0,  # Not used when hardcoded
        })
    
    # Create the plots and collect summary information
    fig, summary = build_plot(
        selected, coeffs, pos_mat, neg_mat, prompts, candidate_rows,
        model, steering_vec, args.layer
    )
    
    # Save the HTML plot file and JSON summary file
    out_html = Path(args.output_html)
    out_json = Path(args.output_json)
    out_html.parent.mkdir(parents=True, exist_ok=True)  # Create directories if needed
    out_json.parent.mkdir(parents=True, exist_ok=True)
    
    fig.write_html(str(out_html))
    with open(out_json, "w") as f:
        json.dump({"coeffs": coeffs, "cases": summary}, f, indent=2)
    
    print(f"Saved plot: {out_html.resolve()}")
    print(f"Saved case summary: {out_json.resolve()}")
    
    # Print a quick summary showing the top predicted word at key points for each example
    for case in summary:
        tokens = case["token_summary"]
        print(
            f"Case {case['case']} | balance={case['balance_coeff']:+.1f} | "
            f"neg='{tokens['neg_edge']['top_token']}' "
            f"zero='{tokens['zero']['top_token']}' "
            f"balance='{tokens['balance']['top_token']}' "
            f"pos='{tokens['pos_edge']['top_token']}'"
        )


if __name__ == "__main__":
    main()

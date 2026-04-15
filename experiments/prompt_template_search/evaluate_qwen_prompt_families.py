#!/usr/bin/env python3
"""Evaluate prompt-family tradeoffs for local Qwen one-token steering.

This script compares three prompt families already discussed in the draft:
- B_positioned
- A2_main_subject
- A_subject_looks

For each family and steering coefficient, it computes:
- baseline / intervened next-token RMS bias on the handcrafted eval captions
- greedy generations with steering limited to the first generated token
- coherence counts using the same heuristic family as the coherence-frontier runs

Outputs:
- experiments/prompt_template_search/qwen_prompt_family_eval/results.json
- experiments/prompt_template_search/qwen_prompt_family_eval/SUMMARY.md
- experiments/prompt_template_search/qwen_prompt_family_eval/qwen_prompt_family_tradeoff.png
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from bias_steering.config import Config
from bias_steering.data.load_dataset import load_handcrafted_eval
from bias_steering.data.prompt_iterator import PromptIterator
from bias_steering.steering import get_target_token_ids, load_model
from bias_steering.steering.intervention import get_intervention_func


ARTIFACT_DIR = REPO_ROOT / "runs_vision" / "Qwen-1_8B-chat"
CFG_PATH = ARTIFACT_DIR / "config.yaml"
DATASET_DIR = REPO_ROOT / "bias_steering" / "data" / "datasets"
OUT_DIR = Path(__file__).parent / "qwen_prompt_family_eval"

BEST_LAYER = 11
COEFFS = [-150, -100, -50, 0, 50, 100, 150]
MAX_NEW_TOKENS = 20
BATCH_SIZE = 8

TEMPLATES = {
    "B_positioned": {
        "instruction_fn": lambda caption: f"Describe this image:\n{caption}",
        "prefix_fn": lambda caption: "Positioned",
        "label": "Spatial-first prompt family",
    },
    "A2_main_subject": {
        "instruction_fn": lambda caption: "Continue describing this image:",
        "prefix_fn": lambda caption: f"{caption}. The main subject appears",
        "label": "Appearance-description prompt family",
    },
    "A_subject_looks": {
        "instruction_fn": lambda caption: f"Describe this image:\n{caption}",
        "prefix_fn": lambda caption: "The subject looks",
        "label": "Appearance-oriented prompt family",
    },
}

COLORS = {
    "B_positioned": "#1b9e77",
    "A2_main_subject": "#7570b3",
    "A_subject_looks": "#d95f02",
}


def rms(values) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(arr ** 2)))


def coherence_score(text: str) -> tuple[str, str]:
    tokens = text.lower().split()
    if len(tokens) < 5:
        return "degenerate", "too short"

    counts = Counter(tokens)
    max_tok, max_count = counts.most_common(1)[0]
    max_freq = max_count / len(tokens)
    ttr = len(counts) / len(tokens)
    bigrams = list(zip(tokens, tokens[1:]))
    bigram_counts = Counter(bigrams)
    max_bigram = bigram_counts.most_common(1)[0][1] if bigram_counts else 0

    if max_freq >= 0.40:
        return "degenerate", f"max_freq={max_freq:.2f}('{max_tok}')"
    if ttr < 0.30:
        return "degenerate", f"ttr={ttr:.2f}"
    if max_bigram >= 4:
        return "degenerate", f"bigram_rep={max_bigram}"

    if max_freq >= 0.25 or ttr < 0.45 or max_bigram >= 3:
        return "partial", f"max_freq={max_freq:.2f}; ttr={ttr:.2f}; max_bigram={max_bigram}"

    return "coherent", ""


def build_prompts(model, instruction_fn, prefix_fn, captions: list[str]) -> list[str]:
    instructions = [instruction_fn(caption) for caption in captions]
    prefixes = [prefix_fn(caption) for caption in captions]
    return model.apply_chat_template(instructions, output_prefix=prefixes)


def compute_bias_scores(model, prompts, all_ids: list[int], n_pos: int, *, layer: int | None = None, intervene_func=None) -> np.ndarray:
    bias_all: list[float] = []
    for batch in PromptIterator(prompts, batch_size=BATCH_SIZE):
        if layer is None or intervene_func is None:
            logits_last = model.get_last_position_logits(batch)
        else:
            logits = model.get_logits(batch, layer=layer, intervene_func=intervene_func)
            logits_last = logits[:, -1, :]
        tgt = logits_last[:, all_ids]
        probs = F.softmax(tgt, dim=-1)
        pos_probs = probs[:, :n_pos].sum(-1)
        neg_probs = probs[:, n_pos:].sum(-1)
        bias_all.extend((pos_probs - neg_probs).tolist())
    return np.asarray(bias_all, dtype=float)


def generate_one_token_steered(model, prompts, layer: int, intervene_func, max_new_tokens: int) -> list[str]:
    layer_block = model.block_modules[layer]
    results: list[str] = []
    for prompt in prompts:
        inputs = model.tokenize([prompt])
        input_len = inputs.input_ids.shape[1]

        with model.model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False) as tracer:
            acts = layer_block.output[0].clone()
            new_acts = intervene_func(acts)
            layer_block.output = (new_acts,) + layer_block.output[1:]
            outputs = model.model.generator.output.detach().to("cpu").save()

        completion = outputs.value[0, input_len:]
        text = model.tokenizer.decode(completion, skip_special_tokens=True).strip()
        results.append(text)
    return results


def summarize_generations(texts: list[str]) -> dict:
    labels = []
    reasons = []
    for text in texts:
        label, reason = coherence_score(text)
        labels.append(label)
        reasons.append(reason)
    counts = Counter(labels)
    total = len(texts)
    coherent_rate = counts.get("coherent", 0) / total if total else 0.0
    usable_rate = (counts.get("coherent", 0) + counts.get("partial", 0)) / total if total else 0.0
    return {
        "counts": dict(counts),
        "coherent_rate": coherent_rate,
        "usable_rate": usable_rate,
        "labels": labels,
        "reasons": reasons,
    }


def pick_best_usable(rows: list[dict]) -> dict | None:
    usable = [row for row in rows if row["generation"]["coherent_rate"] >= 0.50]
    if not usable:
        usable = [row for row in rows if row["generation"]["usable_rate"] >= 0.50]
    if not usable:
        return None
    return max(usable, key=lambda row: (row["reduction_pct"], row["generation"]["coherent_rate"]))


def make_plot(results: dict) -> Path:
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for name, rows in results["template_results"].items():
        x = [100.0 * row["generation"]["coherent_rate"] for row in rows]
        y = [row["reduction_pct"] for row in rows]
        labels = [row["coeff"] for row in rows]
        ax.plot(x, y, color=COLORS[name], linewidth=1.8, alpha=0.8, label=TEMPLATES[name]["label"])
        ax.scatter(x, y, color=COLORS[name], s=44, zorder=3)
        for xi, yi, coeff in zip(x, y, labels):
            ax.annotate(
                f"{coeff:+d}",
                (xi, yi),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
                color=COLORS[name],
            )

    ax.set_title("Prompt-Family Tradeoff Under 1-Token Steering", fontsize=13)
    ax.set_xlabel("Coherent generations on handcrafted set (%)", fontsize=10)
    ax.set_ylabel("First-token RMS reduction (%)", fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_xlim(-2, 102)

    outpath = OUT_DIR / "qwen_prompt_family_tradeoff.png"
    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return outpath


def write_summary(results: dict) -> None:
    lines = [
        "# Qwen Prompt-Family Evaluation",
        "",
        "One-token steering on handcrafted captions.",
        "",
        f"Best layer: {BEST_LAYER}",
        f"Coefficients: {COEFFS}",
        "",
        "## Best Usable Configs",
        "",
        "| Template | Label | Coeff | Reduction | Coherent | Usable |",
        "|---|---|---|---|---|---|",
    ]

    for name in TEMPLATES:
        best = results["best_usable"].get(name)
        if best is None:
            lines.append(f"| `{name}` | {TEMPLATES[name]['label']} | — | — | — | — |")
            continue
        lines.append(
            f"| `{name}` | {TEMPLATES[name]['label']} | {best['coeff']:+d} | "
            f"{best['reduction_pct']:.1f}% | {best['generation']['coherent_rate']*100:.0f}% | "
            f"{best['generation']['usable_rate']*100:.0f}% |"
        )

    lines += [
        "",
        "## Full Sweep",
        "",
    ]

    for name in TEMPLATES:
        lines += [
            f"### `{name}` — {TEMPLATES[name]['label']}",
            "",
            "| Coeff | RMS | Reduction | Coherent | Partial | Degenerate |",
            "|---|---|---|---|---|---|",
        ]
        for row in results["template_results"][name]:
            counts = row["generation"]["counts"]
            lines.append(
                f"| {row['coeff']:+d} | {row['rms']:.4f} | {row['reduction_pct']:.1f}% | "
                f"{counts.get('coherent', 0)} | {counts.get('partial', 0)} | {counts.get('degenerate', 0)} |"
            )
        best = results["best_usable"].get(name)
        if best is not None:
            lines += [
                "",
                f"Best usable setting: λ={best['coeff']:+d}, reduction={best['reduction_pct']:.1f}%, "
                f"coherent={best['generation']['coherent_rate']*100:.0f}%, usable={best['generation']['usable_rate']*100:.0f}%",
                "",
                "Example outputs:",
            ]
            for example in best["examples"][:3]:
                lines.append(
                    f"- _{example['caption'][:80]}_  →  `{example['generated']}` [{example['label']}]"
                )
        lines.append("")

    (OUT_DIR / "SUMMARY.md").write_text("\n".join(lines))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    cfg = Config.load(CFG_PATH)
    model = load_model(cfg.model_name)

    target_words = json.loads((DATASET_DIR / "target_words.json").read_text())["vision"]
    pos_ids_raw = get_target_token_ids(model.tokenizer, target_words["spatial"])
    neg_ids_raw = get_target_token_ids(model.tokenizer, target_words["descriptive"])
    overlap = set(pos_ids_raw) & set(neg_ids_raw)
    pos_ids = [token_id for token_id in pos_ids_raw if token_id not in overlap]
    neg_ids = [token_id for token_id in neg_ids_raw if token_id not in overlap]
    all_ids = pos_ids + neg_ids
    n_pos = len(pos_ids)

    candidate_vectors = torch.load(ARTIFACT_DIR / "activations" / "candidate_vectors.pt")
    steering_vec = model.set_dtype(candidate_vectors[BEST_LAYER])
    neutral_acts = torch.load(ARTIFACT_DIR / "activations" / "neutral.pt")
    offset = model.set_dtype(neutral_acts.mean(dim=1)[BEST_LAYER])

    hc_df = load_handcrafted_eval()
    captions = hc_df["text"].tolist()

    results = {
        "model_name": cfg.model_name,
        "best_layer": BEST_LAYER,
        "max_new_tokens": MAX_NEW_TOKENS,
        "coeffs": COEFFS,
        "captions": len(captions),
        "template_results": {},
        "best_usable": {},
    }

    for name, template in TEMPLATES.items():
        prompts = build_prompts(model, template["instruction_fn"], template["prefix_fn"], captions)
        baseline_bias = compute_bias_scores(model, prompts, all_ids, n_pos)
        baseline_rms = rms(baseline_bias)

        template_rows = []
        for coeff in COEFFS:
            intervene_func = get_intervention_func(
                steering_vec,
                method="default",
                coeff=float(coeff),
                offset=offset,
            )

            steered_bias = compute_bias_scores(
                model,
                prompts,
                all_ids,
                n_pos,
                layer=BEST_LAYER,
                intervene_func=intervene_func,
            )
            current_rms = rms(steered_bias)
            reduction_pct = 100.0 * (baseline_rms - current_rms) / baseline_rms if baseline_rms else 0.0

            generations = generate_one_token_steered(
                model,
                prompts,
                BEST_LAYER,
                intervene_func,
                MAX_NEW_TOKENS,
            )
            generation_summary = summarize_generations(generations)
            examples = [
                {
                    "caption": caption,
                    "generated": generated,
                    "label": label,
                    "reason": reason,
                }
                for caption, generated, label, reason in zip(
                    captions,
                    generations,
                    generation_summary["labels"],
                    generation_summary["reasons"],
                )
            ]

            template_rows.append(
                {
                    "coeff": coeff,
                    "baseline_rms": baseline_rms,
                    "rms": current_rms,
                    "reduction_pct": reduction_pct,
                    "generation": generation_summary,
                    "examples": examples,
                }
            )

        results["template_results"][name] = template_rows
        best = pick_best_usable(template_rows)
        results["best_usable"][name] = best

    results["plot_path"] = str(make_plot(results))
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    write_summary(results)
    print(json.dumps(results["best_usable"], indent=2))


if __name__ == "__main__":
    main()

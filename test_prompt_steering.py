"""
Quick manual harness: steer a model on a small, hardcoded prompt set.

Edit `TEST_CAPTIONS` / `RAW_PROMPTS` and `LAMBDAS` to iterate quickly.
Prints per-step spatial vs descriptive probability mass and the generated text.

Run:
  python test_prompt_steering.py
  python test_prompt_steering.py --config runs_vision/gpt2/config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import torch

from bias_steering.config import Config
from bias_steering.steering import load_model
from bias_steering.steering.steering_utils import get_target_token_ids
from bias_steering.eval.generation_logger import (
    _greedy_generate_with_probs,
    _beam_generate_with_probs,
)


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# ── edit these ────────────────────────────────────────────────────────────────

# Option 1: Provide captions and a prompt wrapper will be applied.
TEST_CAPTIONS: List[str] = [
    "A lone hiker stands on top of a snow-dusted ridge looking out across a wide valley far below.",
    "A bright orange and yellow maple tree stands beside a small dark pond.",
]

# Option 2: Provide raw prompts directly (overrides TEST_CAPTIONS when non-empty).
RAW_PROMPTS: List[str] = []

# Steering coefficients to sweep.
LAMBDAS: List[float] = [-150, -50, 0, 50, 150]

# Generation settings.
DECODING: str = "greedy"  # "greedy" or "beam"
MAX_NEW_TOKENS: int = 20
BEAM_WIDTH: int = 4
BEAM_TOP_K: int = 8

# Prompt style. If output_prefix is enabled in your config, this is appended.
OUTPUT_PREFIX: str = "The image shows"
PROMPT_WRAPPER: str = "Describe this image:\n{caption}"

# Print per-step tracked probs for each run.
PRINT_STEP_PROBS: bool = True


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_target_words() -> dict:
    dataset_dir = Path("bias_steering/data/datasets")
    return json.loads((dataset_dir / "target_words.json").read_text())


def _remove_overlap(pos_ids: List[int], neg_ids: List[int]) -> Tuple[List[int], List[int]]:
    overlap = set(pos_ids) & set(neg_ids)
    if overlap:
        pos_ids = [t for t in pos_ids if t not in overlap]
        neg_ids = [t for t in neg_ids if t not in overlap]
    return pos_ids, neg_ids


def _build_prompts(model, cfg: Config) -> List[str]:
    if RAW_PROMPTS:
        raw = RAW_PROMPTS
    else:
        raw = [PROMPT_WRAPPER.format(caption=c) for c in TEST_CAPTIONS]

    if cfg.data_cfg.output_prefix:
        return model.apply_chat_template(raw, output_prefix=[OUTPUT_PREFIX] * len(raw))
    return model.apply_chat_template(raw)

def _load_captions_from_generation_log(path: str) -> List[str]:
    """
    Read a JSONL generation log and return captions in example_id order.
    Deduplicates by example_id (since logs usually contain multiple lambdas per example).
    """
    p = Path(path)
    rows = p.read_text(encoding="utf-8").splitlines()
    by_id: Dict[int, str] = {}
    for line in rows:
        if not line.strip():
            continue
        rec: Dict[str, Any] = json.loads(line)
        eid = int(rec["example_id"])
        if eid not in by_id:
            by_id[eid] = str(rec["caption"])
    return [by_id[eid] for eid in sorted(by_id.keys())]


def _seq_summary(sp_seq: List[float], dp_seq: List[float]) -> str:
    if not sp_seq:
        return "empty"
    return (
        f"first(sp={sp_seq[0]:.3f}, dp={dp_seq[0]:.3f})  "
        f"mean(sp={float(np.mean(sp_seq)):.3f}, dp={float(np.mean(dp_seq)):.3f})"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="runs_vision/Qwen-1_8B-chat/config.yaml")
    ap.add_argument(
        "--from_generation_log",
        default=None,
        help='Path to results/generation_logs/*.jsonl; uses its captions as the test set.',
    )
    ap.add_argument(
        "--device_map",
        default="auto",
        help='Passed through to load_model(...). Try "cpu" to avoid MPS/CUDA.',
    )
    ap.add_argument("--layer", type=int, default=None, help="Override selected layer (default: top_layers.json[0])")
    ap.add_argument("--intervention_method", default="default", choices=["default", "constant"])
    ap.add_argument("--constrained_softmax", type=int, default=None, help="0/1 override (default: cfg.constrained_softmax)")
    args = ap.parse_args()

    global TEST_CAPTIONS
    if args.from_generation_log:
        TEST_CAPTIONS = _load_captions_from_generation_log(args.from_generation_log)

    cfg = Config.load(args.config)
    model = load_model(cfg.model_name, device_map=args.device_map)

    artifact_dir = cfg.artifact_path()

    # Pick layer from saved validation unless overridden.
    if args.layer is None:
        top_layers = json.loads((artifact_dir / "validation/top_layers.json").read_text())
        layer = int(top_layers[0]["layer"])
    else:
        layer = int(args.layer)

    # Steering vector for that layer.
    candidate_vectors = torch.load(artifact_dir / "activations/candidate_vectors.pt")
    steering_vec = model.set_dtype(candidate_vectors[layer])

    # Optional neutral offset.
    offset = 0
    if cfg.use_offset:
        neutral_acts = torch.load(artifact_dir / "activations/neutral.pt")
        offset = model.set_dtype(neutral_acts.mean(dim=1)[layer])

    # Target token IDs (vision spatial vs descriptive).
    words = _load_target_words()["vision"]
    pos_ids_raw = get_target_token_ids(model.tokenizer, words[cfg.data_cfg.pos_label])
    neg_ids_raw = get_target_token_ids(model.tokenizer, words[cfg.data_cfg.neg_label])
    pos_ids, neg_ids = _remove_overlap(pos_ids_raw, neg_ids_raw)

    constrained = cfg.constrained_softmax if args.constrained_softmax is None else bool(int(args.constrained_softmax))

    prompts = _build_prompts(model, cfg)

    logging.info(
        "Loaded model=%s layer=%d  prompts=%d  lambdas=%d  decoding=%s  method=%s  constrained_softmax=%s",
        cfg.model_name,
        layer,
        len(prompts),
        len(LAMBDAS),
        DECODING,
        args.intervention_method,
        constrained,
    )

    # Import here so edits don’t slow module import.
    from bias_steering.steering.intervention import get_intervention_func

    for pi, prompt in enumerate(prompts):
        raw_display = RAW_PROMPTS[pi] if RAW_PROMPTS else TEST_CAPTIONS[pi]
        print("\n" + "=" * 88, flush=True)
        print(f"PROMPT {pi}", flush=True)
        print(raw_display, flush=True)
        print("=" * 88, flush=True)

        for lam in LAMBDAS:
            intervene_func = get_intervention_func(
                steering_vec, method=args.intervention_method, coeff=float(lam), offset=offset
            )

            if DECODING == "beam":
                gen_text, sp_seq, dp_seq = _beam_generate_with_probs(
                    model=model,
                    prompt=prompt,
                    layer=layer,
                    intervene_func=intervene_func,
                    pos_ids=pos_ids,
                    neg_ids=neg_ids,
                    max_new_tokens=MAX_NEW_TOKENS,
                    beam_width=BEAM_WIDTH,
                    beam_top_k=BEAM_TOP_K,
                    constrained_softmax=constrained,
                )
            else:
                gen_text, sp_seq, dp_seq = _greedy_generate_with_probs(
                    model=model,
                    prompt=prompt,
                    layer=layer,
                    intervene_func=intervene_func,
                    pos_ids=pos_ids,
                    neg_ids=neg_ids,
                    max_new_tokens=MAX_NEW_TOKENS,
                    constrained_softmax=constrained,
                )

            print(f"\nλ={lam:>7}  {_seq_summary(sp_seq, dp_seq)}", flush=True)
            if PRINT_STEP_PROBS and sp_seq:
                # compact: show up to first 12 steps
                k = min(12, len(sp_seq))
                pairs = " ".join([f"{i}:{sp_seq[i]:.2f}/{dp_seq[i]:.2f}" for i in range(k)])
                print(f"steps(sp/dp) {pairs}{' ...' if k < len(sp_seq) else ''}", flush=True)
            print(f"gen: {gen_text}", flush=True)


if __name__ == "__main__":
    main()


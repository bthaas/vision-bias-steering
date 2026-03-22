"""
Stream-write JSONL logs of generated text and tracked token probabilities
for a sweep of steering coefficients.

Output format — one JSON object per line:
  {
    "example_id":          int,
    "caption":             str,
    "prompt_template":     str,
    "lambda":              float,
    "decoding":            str,   # "greedy" | "beam"
    "intervention_method": str,   # "default" | "constant"
    "constrained_softmax": bool,
    "generated_text":      str,
    "tracked_token_probs": {
      "spatial":              float,   # prob at first generated position
      "descriptive":          float,
      "spatial_seq_mean":     float,   # mean over the full continuation
      "descriptive_seq_mean": float,
    }
  }

Records are flushed after each write so partial runs survive crashes.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
import torch.nn.functional as F


def build_output_path(output_dir: Path, prompt_type: str, decoding: str) -> Path:
    """Return a timestamped path: <output_dir>/<YYYYMMDDTHHMMSS>_<prompt_type>_<decoding>.jsonl"""
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    return Path(output_dir) / f"{ts}_{prompt_type}_{decoding}.jsonl"


def _write_record(fh, record: dict) -> None:
    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    fh.flush()


def _compute_tracked_probs(
    logits_last: torch.Tensor,
    pos_ids: List[int],
    neg_ids: List[int],
    constrained_softmax: bool,
) -> tuple:
    """
    Compute spatial and descriptive probability mass from a single-position logit vector.

    When constrained_softmax=True, softmax is computed over only the tracked token IDs
    (pos_ids + neg_ids), matching the validation pipeline's constrained mode.
    When False, softmax is over the full vocabulary.

    Returns (spatial_prob, descriptive_prob) as Python floats.
    """
    if constrained_softmax:
        all_ids = pos_ids + neg_ids
        n_pos = len(pos_ids)
        target_logits = logits_last[all_ids]
        probs = F.softmax(target_logits, dim=-1)
        sp = float(probs[:n_pos].sum().item())
        dp = float(probs[n_pos:].sum().item())
    else:
        probs = F.softmax(logits_last, dim=-1)
        sp = float(probs[pos_ids].sum().item())
        dp = float(probs[neg_ids].sum().item())
    return sp, dp


# ── greedy generation ──────────────────────────────────────────────────────────

def _greedy_generate_with_probs(
    model,
    prompt: str,
    layer: int,
    intervene_func,
    pos_ids: List[int],
    neg_ids: List[int],
    max_new_tokens: int,
    constrained_softmax: bool = False,
):
    """
    Greedy single-token-at-a-time generation with steered activations.

    Returns (generated_text, spatial_probs_per_step, descriptive_probs_per_step).
    Probs are recorded at the position *before* appending each generated token,
    matching what run_debias_test measures.

    Token selection always uses full-vocabulary argmax; constrained_softmax only
    affects the probability tracking values.
    """
    context = prompt
    spatial_probs: List[float] = []
    descriptive_probs: List[float] = []
    token_pieces: List[str] = []

    for _ in range(max_new_tokens):
        logits = model.get_logits([context], layer=layer, intervene_func=intervene_func)
        logits_last = logits[0, -1, :]
        tok_id = int(torch.argmax(logits_last).item())
        sp, dp = _compute_tracked_probs(logits_last, pos_ids, neg_ids, constrained_softmax)
        spatial_probs.append(sp)
        descriptive_probs.append(dp)
        tok_text = model.tokenizer.decode([tok_id])
        token_pieces.append(tok_text)
        context += tok_text

    return "".join(token_pieces), spatial_probs, descriptive_probs


# ── beam generation ────────────────────────────────────────────────────────────

def _beam_generate_with_probs(
    model,
    prompt: str,
    layer: int,
    intervene_func,
    pos_ids: List[int],
    neg_ids: List[int],
    max_new_tokens: int,
    beam_width: int,
    beam_top_k: int,
    constrained_softmax: bool = False,
):
    """
    Beam-search generation with steered activations.

    Returns (best-beam generated text, spatial_probs along best beam,
    descriptive_probs along best beam).  Probs are the per-step values from
    the beam that ranked first by cumulative log-probability.

    Beam expansion always uses full-vocabulary log-softmax; constrained_softmax
    only affects the probability tracking values recorded per step.
    """
    beams = [{"context": prompt, "generated": "", "sum_logprob": 0.0, "sp": [], "dp": []}]

    for _ in range(max_new_tokens):
        contexts = [b["context"] for b in beams]
        logits_batch = model.get_logits(contexts, layer=layer, intervene_func=intervene_func)
        candidates = []
        for bi, beam in enumerate(beams):
            logits_last = logits_batch[bi, -1, :]
            # Full-vocab log-softmax for beam expansion
            lp = F.log_softmax(logits_last, dim=-1)
            # Tracked probs (respects constrained_softmax flag)
            sp, dp = _compute_tracked_probs(logits_last, pos_ids, neg_ids, constrained_softmax)
            k = min(beam_top_k, lp.shape[-1])
            top_lp, top_ids = torch.topk(lp, k=k)
            for j in range(k):
                tid = int(top_ids[j].item())
                piece = model.tokenizer.decode([tid])
                candidates.append({
                    "context": beam["context"] + piece,
                    "generated": beam["generated"] + piece,
                    "sum_logprob": beam["sum_logprob"] + float(top_lp[j].item()),
                    "sp": beam["sp"] + [sp],
                    "dp": beam["dp"] + [dp],
                })
        candidates.sort(key=lambda x: x["sum_logprob"], reverse=True)
        beams = candidates[:beam_width]

    if not beams:
        return "", [], []
    best = beams[0]
    return best["generated"], best["sp"], best["dp"]


# ── public API ─────────────────────────────────────────────────────────────────

def log_generations_sweep(
    model,
    examples: List[Dict],
    layer: int,
    steering_vec,
    coeffs: List[float],
    pos_ids: List[int],
    neg_ids: List[int],
    output_path: Path,
    decoding: str = "greedy",
    max_new_tokens: int = 20,
    beam_width: int = 4,
    beam_top_k: int = 8,
    intervention_method: str = "default",
    constrained_softmax: bool = False,
    offset=0,
) -> None:
    """
    Sweep over (example, lambda) pairs, generating text and recording tracked token
    probabilities for each pair.  One JSONL line is written and flushed per record
    so partial runs survive crashes.

    Parameters
    ----------
    model                 ModelBase instance (already loaded)
    examples              List of dicts with keys:
                            example_id (int), caption (str),
                            prompt_template (str), prompt (str)
    layer                 Transformer layer at which steering is applied
    steering_vec          Steering vector, already cast to model dtype
    coeffs                List of lambda values to sweep
    pos_ids               Token IDs for the positive class (spatial)
    neg_ids               Token IDs for the negative class (descriptive)
    output_path           Destination .jsonl file; parent directory is created if absent
    decoding              "greedy" or "beam"
    max_new_tokens        Tokens to generate per (example, lambda) pair
    beam_width            Beam width (beam decoding only)
    beam_top_k            Top-k expansions per beam step (beam decoding only)
    intervention_method   "default" (orthogonal projection + coeff*unit_vec, stable for
                          multi-token generation) or "constant" (direct addition, matches
                          debiasing pipeline but degrades at large coefficients).
                          Defaults to "default".
    constrained_softmax   When True, tracked token probs are computed as softmax over
                          pos_ids + neg_ids only, matching the validation pipeline's
                          constrained mode.  Token selection is always full-vocabulary.
    offset                Neutral-mean offset tensor (or 0) applied in the "default"
                          intervention: acts_new = acts - proj(acts - offset, unit_vec)
                          + unit_vec * coeff.  Required when cfg.use_offset=True.
    """
    from ..steering.intervention import get_intervention_func

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = len(examples) * len(coeffs)
    logging.info(
        "Generation logging: %d examples × %d coeffs = %d total → %s  "
        "[method=%s  constrained_softmax=%s]",
        len(examples), len(coeffs), n_total, output_path,
        intervention_method, constrained_softmax,
    )

    completed = 0
    with open(output_path, "w", encoding="utf-8") as fh:
        for ex in examples:
            eid = ex["example_id"]
            caption = ex["caption"]
            tmpl = ex["prompt_template"]
            prompt = ex["prompt"]

            for lam in coeffs:
                intervene_func = get_intervention_func(
                    steering_vec, method=intervention_method, coeff=lam, offset=offset
                )

                if decoding == "beam":
                    gen_text, sp_seq, dp_seq = _beam_generate_with_probs(
                        model, prompt, layer, intervene_func,
                        pos_ids, neg_ids, max_new_tokens, beam_width, beam_top_k,
                        constrained_softmax=constrained_softmax,
                    )
                else:  # greedy (default)
                    gen_text, sp_seq, dp_seq = _greedy_generate_with_probs(
                        model, prompt, layer, intervene_func,
                        pos_ids, neg_ids, max_new_tokens,
                        constrained_softmax=constrained_softmax,
                    )

                record = {
                    "example_id": eid,
                    "caption": caption,
                    "prompt_template": tmpl,
                    "lambda": float(lam),
                    "decoding": decoding,
                    "intervention_method": intervention_method,
                    "constrained_softmax": constrained_softmax,
                    "generated_text": gen_text,
                    "tracked_token_probs": {
                        "spatial":              float(sp_seq[0]) if sp_seq else 0.0,
                        "descriptive":          float(dp_seq[0]) if dp_seq else 0.0,
                        "spatial_seq_mean":     float(np.mean(sp_seq)) if sp_seq else 0.0,
                        "descriptive_seq_mean": float(np.mean(dp_seq)) if dp_seq else 0.0,
                    },
                }
                _write_record(fh, record)
                completed += 1
                if completed % 50 == 0:
                    logging.info(
                        "Generation logging progress: %d / %d", completed, n_total
                    )

    logging.info("Generation log complete (%d records): %s", n_total, output_path)

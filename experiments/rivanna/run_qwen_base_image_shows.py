#!/usr/bin/env python3
"""Standardized Qwen base-model comparison for the "The image shows" prefix.

This runner keeps the local Qwen-1.8B-chat prompt-prefix curve as the anchor:

    Describe this image:
    {COCO caption}

    The image shows

The `verify-local` command reruns the saved local Qwen/Qwen-1_8B-chat setup
against the embedded reference values from `qwen_prompt_family_tradeoff_main.png`.
The larger-model `run-model` command refuses to start when the verification
marker is absent, unless explicitly told otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from operator import attrgetter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_experiment as rivanna
from bias_steering.data.prompt_iterator import PromptIterator
from bias_steering.steering import get_target_token_ids, load_model
from bias_steering.steering.intervention import get_intervention_func, intervene_generation
from bias_steering.text_metrics import SpatialRatioScorer, simple_tokens

torch.set_grad_enabled(False)

PROMPT_TEMPLATE_NAME = "image_shows"
PROMPT_INSTRUCTION_TEMPLATE = "Describe this image:\n{caption}"
PROMPT_PREFIX = "The image shows"
DEFAULT_LAMBDAS = [-60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60]
DEFAULT_VERIFY_LAMBDAS = [0, 20, 30]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "experiments" / "rivanna" / "results" / "qwen_base_image_shows"
DEFAULT_DATA_DIR = REPO_ROOT / "experiments" / "rivanna" / "data"
DEFAULT_LOCAL_ARTIFACT_DIR = REPO_ROOT / "runs_vision" / "Qwen-1_8B-chat"
LOCAL_REFERENCE_LAYER = 11
DEGENERATION_CAP_RATE = 0.20


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    hf_model: str
    batch_size: int
    slurm_time_hint: str


MODEL_SPECS = {
    "qwen18b_chat": ModelSpec(
        key="qwen18b_chat",
        display_name="Qwen 1.8B chat",
        hf_model="Qwen/Qwen-1_8B-chat",
        batch_size=16,
        slurm_time_hint="12:00:00",
    ),
    "qwen25_3b_base": ModelSpec(
        key="qwen25_3b_base",
        display_name="Qwen2.5 3B base",
        hf_model="Qwen/Qwen2.5-3B",
        batch_size=16,
        slurm_time_hint="24:00:00",
    ),
    "qwen25_7b_base": ModelSpec(
        key="qwen25_7b_base",
        display_name="Qwen2.5 7B base",
        hf_model="Qwen/Qwen2.5-7B",
        batch_size=8,
        slurm_time_hint="36:00:00",
    ),
}

MODEL_ORDER = ["qwen18b_chat", "qwen25_3b_base", "qwen25_7b_base"]

# Reference rows for the local Qwen/Qwen-1_8B-chat, A_image_shows,
# full-greedy, 20-token, full-sequence-steering curve used by
# qwen_prompt_family_tradeoff_main.png.
REFERENCE_IMAGE_SHOWS_ROWS = [
    {"lambda": -60, "mean_normalized_ratio": 0.0, "std_normalized_ratio": 0.0, "degenerate_or_repetitive_count": 999, "n_outputs": 1000},
    {"lambda": -50, "mean_normalized_ratio": 0.0, "std_normalized_ratio": 0.0, "degenerate_or_repetitive_count": 989, "n_outputs": 1000},
    {"lambda": -40, "mean_normalized_ratio": -0.002, "std_normalized_ratio": 0.044676615807377355, "degenerate_or_repetitive_count": 919, "n_outputs": 1000},
    {"lambda": -30, "mean_normalized_ratio": -0.1075, "std_normalized_ratio": 0.32510575202539865, "degenerate_or_repetitive_count": 235, "n_outputs": 1000},
    {"lambda": -20, "mean_normalized_ratio": -0.2752, "std_normalized_ratio": 0.8124793768322847, "degenerate_or_repetitive_count": 10, "n_outputs": 1000},
    {"lambda": -10, "mean_normalized_ratio": -0.19043333333333334, "std_normalized_ratio": 0.8819502574786555, "degenerate_or_repetitive_count": 0, "n_outputs": 1000},
    {"lambda": 0, "mean_normalized_ratio": -0.0989, "std_normalized_ratio": 0.8824422618820766, "degenerate_or_repetitive_count": 1, "n_outputs": 1000},
    {"lambda": 10, "mean_normalized_ratio": -0.03183333333333334, "std_normalized_ratio": 0.836749647279413, "degenerate_or_repetitive_count": 2, "n_outputs": 1000},
    {"lambda": 20, "mean_normalized_ratio": 0.23063333333333336, "std_normalized_ratio": 0.6368223535649483, "degenerate_or_repetitive_count": 1, "n_outputs": 1000},
    {"lambda": 30, "mean_normalized_ratio": 0.0, "std_normalized_ratio": 0.0, "degenerate_or_repetitive_count": 448, "n_outputs": 1000},
    {"lambda": 40, "mean_normalized_ratio": 0.0, "std_normalized_ratio": 0.0, "degenerate_or_repetitive_count": 996, "n_outputs": 1000},
    {"lambda": 50, "mean_normalized_ratio": 0.0, "std_normalized_ratio": 0.0, "degenerate_or_repetitive_count": 1000, "n_outputs": 1000},
    {"lambda": 60, "mean_normalized_ratio": 0.0, "std_normalized_ratio": 0.0, "degenerate_or_repetitive_count": 1000, "n_outputs": 1000},
]
REFERENCE_BY_LAMBDA = {int(row["lambda"]): row for row in REFERENCE_IMAGE_SHOWS_ROWS}


def parse_int_list(raw: str | Sequence[int]) -> list[int]:
    if isinstance(raw, str):
        return [int(item.strip()) for item in raw.split(",") if item.strip()]
    return [int(item) for item in raw]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, default=serialize_for_json, indent=2), encoding="utf-8")


def serialize_for_json(obj):
    if isinstance(obj, (np.float32, np.float64, float)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    raise TypeError(type(obj))


def prompt_instruction(caption: str) -> str:
    return PROMPT_INSTRUCTION_TEMPLATE.format(caption=caption)


def build_image_shows_prompts(model, captions: Sequence[str]) -> list[str]:
    instructions = [prompt_instruction(caption) for caption in captions]
    prefixes = [PROMPT_PREFIX] * len(captions)
    return model.apply_chat_template(instructions, output_prefix=prefixes)


def caption_digest(captions: Sequence[str]) -> str:
    payload = "\n".join(captions).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_caption_texts(path: Path, n_val: int, selection: str = "first", seed: int = 42) -> tuple[list[str], list[int]]:
    df = rivanna.load_caption_df(path)
    n = min(n_val, len(df))
    if selection == "first":
        indices = list(range(n))
    elif selection == "random":
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(df), size=n, replace=False)).tolist()
    else:
        raise ValueError(f"Unknown validation selection: {selection}")
    return df.iloc[indices].reset_index(drop=True)["text"].tolist(), indices


def coherence_score(text: str) -> tuple[str, str]:
    tokens = simple_tokens(text)
    if len(tokens) < 5:
        return "degenerate", "too short"

    counts = Counter(tokens)
    max_tok, max_count = counts.most_common(1)[0]
    max_freq = max_count / len(tokens)
    ttr = len(counts) / len(tokens)
    bigrams = Counter(zip(tokens, tokens[1:]))
    max_bigram = max(bigrams.values()) if bigrams else 0

    if max_freq >= 0.40:
        return "degenerate", f"max_freq={max_freq:.2f}('{max_tok}')"
    if ttr < 0.30:
        return "degenerate", f"ttr={ttr:.2f}"
    if max_bigram >= 4:
        return "degenerate", f"bigram_rep={max_bigram}"
    if max_freq >= 0.25 or ttr < 0.45 or max_bigram >= 3:
        return "partial", f"max_freq={max_freq:.2f}; ttr={ttr:.2f}; max_bigram={max_bigram}"
    return "coherent", ""


def summarize_texts(texts: Sequence[str], scorer: SpatialRatioScorer) -> dict:
    ratios = []
    zero_hit_count = 0
    label_counts: Counter[str] = Counter()
    for text in texts:
        counts = scorer.counts(text)
        ratios.append(counts.normalized_spatial_ratio)
        if counts.spatial_hits + counts.descriptive_hits == 0:
            zero_hit_count += 1
        label, _ = coherence_score(text)
        label_counts[label] += 1

    n_outputs = len(texts)
    degenerate_count = int(label_counts.get("degenerate", 0))
    return {
        "mean_normalized_ratio": float(np.mean(ratios)) if ratios else 0.0,
        "std_normalized_ratio": float(np.std(ratios)) if ratios else 0.0,
        "degenerate_or_repetitive_count": degenerate_count,
        "degenerate_or_repetitive_rate": float(degenerate_count / n_outputs) if n_outputs else 0.0,
        "partial_count": int(label_counts.get("partial", 0)),
        "coherent_count": int(label_counts.get("coherent", 0)),
        "zero_hit_count": int(zero_hit_count),
        "zero_hit_pct": float(100.0 * zero_hit_count / n_outputs) if n_outputs else 0.0,
        "n_outputs": n_outputs,
    }


def nested_attr(obj, dotted: str):
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def get_hook_layer_module(model, layer: int):
    candidates = [
        getattr(model.model, "_model", None),
        model.model,
    ]
    paths = [
        "transformer.h",
        "model.layers",
        "transformers.h",
    ]
    for base in candidates:
        if base is None:
            continue
        for path in paths:
            try:
                return nested_attr(base, path)[layer]
            except Exception:
                continue
    return None


def move_batch_to_model_device(model, inputs):
    device = getattr(model, "device", None)
    if device is None:
        return inputs
    try:
        device = torch.device(device)
    except Exception:
        return inputs
    if device.type == "meta":
        return inputs
    try:
        return inputs.to(device)
    except Exception:
        return inputs


def decode_completions(model, outputs: torch.Tensor, input_len: int) -> list[str]:
    completions = outputs[:, input_len:]
    return [
        text.strip()
        for text in model.tokenizer.batch_decode(completions, skip_special_tokens=True)
    ]


def generate_batch_with_hook(model, batch: Sequence[str], layer: int, intervene_func, max_new_tokens: int) -> list[str]:
    raw_model = getattr(model.model, "_model", None)
    layer_module = get_hook_layer_module(model, layer)
    if raw_model is None or layer_module is None:
        raise RuntimeError("Could not locate an underlying Hugging Face layer module for hook generation.")

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            return (intervene_func(hidden),) + output[1:]
        return intervene_func(output)

    inputs = move_batch_to_model_device(model, model.tokenize(list(batch)))
    input_len = inputs.input_ids.shape[1]
    handle = layer_module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            outputs = raw_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=getattr(model.tokenizer, "pad_token_id", None),
                eos_token_id=getattr(model.tokenizer, "eos_token_id", None),
            )
    finally:
        handle.remove()
    return decode_completions(model, outputs.detach().to("cpu"), input_len)


def generate_batch_with_nnsight(model, batch: Sequence[str], layer: int, intervene_func, max_new_tokens: int) -> list[str]:
    inputs = model.tokenize(list(batch))
    input_len = inputs.input_ids.shape[1]
    outputs = intervene_generation(
        model.model,
        inputs,
        model.block_modules[layer],
        intervene_func,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    return decode_completions(model, outputs.detach().to("cpu"), input_len)


def generate_batched_full_sequence(
    model,
    prompts: Sequence[str],
    layer: int,
    intervene_func,
    max_new_tokens: int,
    batch_size: int,
    *,
    prefer_hook: bool = True,
) -> list[str]:
    texts: list[str] = []
    use_hook = prefer_hook
    warned_fallback = False
    for batch in PromptIterator(list(prompts), batch_size=batch_size):
        if use_hook:
            try:
                texts.extend(generate_batch_with_hook(model, batch, layer, intervene_func, max_new_tokens))
                continue
            except Exception as exc:
                if not warned_fallback:
                    logging.warning("Hook generation failed; falling back to nnsight generation: %s", exc)
                    warned_fallback = True
                use_hook = False
        texts.extend(generate_batch_with_nnsight(model, batch, layer, intervene_func, max_new_tokens))
    return texts


def prepare_model_tensor(model, tensor: torch.Tensor) -> torch.Tensor:
    out = model.set_dtype(tensor)
    device = getattr(model, "device", None)
    try:
        if device is not None and torch.device(device).type != "meta":
            out = out.to(device)
    except Exception:
        pass
    return out


def pick_best_row(rows: Sequence[dict], degeneration_cap_rate: float = DEGENERATION_CAP_RATE) -> dict | None:
    eligible = [
        row for row in rows
        if row["n_outputs"] and row["degenerate_or_repetitive_count"] / row["n_outputs"] <= degeneration_cap_rate
    ]
    pool = eligible if eligible else list(rows)
    if not pool:
        return None
    return max(
        pool,
        key=lambda row: (
            row["mean_normalized_ratio"],
            -row["degenerate_or_repetitive_count"],
            -abs(int(row["lambda"])),
        ),
    )


def run_curve(
    *,
    model,
    prompts: Sequence[str],
    scorer: SpatialRatioScorer,
    layer: int,
    steering_vec: torch.Tensor,
    offset: torch.Tensor,
    lambdas: Sequence[int],
    batch_size: int,
    max_new_tokens: int,
    output_path: Path,
    base_payload: dict,
) -> list[dict]:
    rows: list[dict] = []
    for coeff in tqdm(lambdas, desc=f"lambda sweep layer {layer}"):
        intervene_func = get_intervention_func(
            steering_vec,
            method="default",
            coeff=int(coeff),
            offset=offset,
        )
        texts = generate_batched_full_sequence(
            model,
            prompts,
            layer,
            intervene_func,
            max_new_tokens,
            batch_size,
            prefer_hook=True,
        )
        summary = summarize_texts(texts, scorer)
        row = {"lambda": int(coeff), **summary}
        rows.append(row)
        payload = dict(base_payload)
        payload["rows"] = rows
        payload["best_under_20pct_degeneration"] = pick_best_row(rows)
        save_json(output_path, payload)
        logging.info(
            "lambda=%+d mean_ratio=%.4f std=%.4f deg=%d/%d",
            coeff,
            row["mean_normalized_ratio"],
            row["std_normalized_ratio"],
            row["degenerate_or_repetitive_count"],
            row["n_outputs"],
        )
    return rows


def load_ratio_scorer(data_dir: Path) -> SpatialRatioScorer:
    target_words = load_json(data_dir / "target_words.json")["vision"]
    return SpatialRatioScorer(target_words["spatial"], target_words["descriptive"])


def load_target_token_info(model, data_dir: Path) -> dict:
    target_words = load_json(data_dir / "target_words.json")["vision"]
    pos_ids_raw = get_target_token_ids(model.tokenizer, target_words["spatial"])
    neg_ids_raw = get_target_token_ids(model.tokenizer, target_words["descriptive"])
    overlap = set(pos_ids_raw) & set(neg_ids_raw)
    pos_ids = [token_id for token_id in pos_ids_raw if token_id not in overlap]
    neg_ids = [token_id for token_id in neg_ids_raw if token_id not in overlap]
    return {
        "target_words": target_words,
        "pos_ids": pos_ids,
        "neg_ids": neg_ids,
        "all_ids": pos_ids + neg_ids,
        "n_pos": len(pos_ids),
        "overlap_removed": len(overlap),
    }


def compare_with_reference(
    rows: Sequence[dict],
    verify_lambdas: Sequence[int],
    ratio_tolerance: float,
    degeneration_tolerance: int,
) -> tuple[bool, list[dict]]:
    observed = {int(row["lambda"]): row for row in rows}
    mismatches = []
    for coeff in verify_lambdas:
        ref = REFERENCE_BY_LAMBDA[int(coeff)]
        obs = observed.get(int(coeff))
        if obs is None:
            mismatches.append({"lambda": coeff, "problem": "missing observed row", "reference": ref})
            continue
        ratio_delta = abs(float(obs["mean_normalized_ratio"]) - float(ref["mean_normalized_ratio"]))
        deg_delta = abs(int(obs["degenerate_or_repetitive_count"]) - int(ref["degenerate_or_repetitive_count"]))
        ok = ratio_delta <= ratio_tolerance and deg_delta <= degeneration_tolerance
        if not ok:
            mismatches.append(
                {
                    "lambda": int(coeff),
                    "observed_mean_normalized_ratio": obs["mean_normalized_ratio"],
                    "reference_mean_normalized_ratio": ref["mean_normalized_ratio"],
                    "ratio_abs_delta": ratio_delta,
                    "observed_degenerate_or_repetitive_count": obs["degenerate_or_repetitive_count"],
                    "reference_degenerate_or_repetitive_count": ref["degenerate_or_repetitive_count"],
                    "degeneration_abs_delta": deg_delta,
                    "ratio_tolerance": ratio_tolerance,
                    "degeneration_tolerance": degeneration_tolerance,
                }
            )
    return not mismatches, mismatches


def verification_diagnostics(args: argparse.Namespace, base_payload: dict, mismatches: Sequence[dict]) -> dict:
    return {
        "status": "failed",
        "mismatches": list(mismatches),
        "checklist": {
            "validation_file_path": base_payload["validation"]["path"],
            "n_val": base_payload["validation"]["n_val"],
            "validation_selection": base_payload["validation"]["selection"],
            "random_seed": base_payload["settings"]["seed"],
            "caption_sample_order": base_payload["validation"]["caption_sha256"],
            "prompt_template": PROMPT_INSTRUCTION_TEMPLATE,
            "continuation_prefix_text": PROMPT_PREFIX,
            "chat_template_serialization": base_payload["prompt"]["serialization"],
            "steering_vector_path": base_payload["steering"]["vector_path"],
            "selected_layer": base_payload["selected_steering_layer"],
            "lambda_grid": base_payload["settings"]["lambdas"],
            "max_new_tokens": base_payload["settings"]["max_new_tokens"],
            "greedy_sampling_settings": "do_sample=False, num_beams=1",
            "full_sequence_vs_token_limited": "full-sequence steering across all generated tokens",
            "degeneration_detection_method": "simple_tokens repetition/TTR heuristic copied from coherence_frontier/run_template_frontier.py",
        },
        "args": vars(args),
    }


def run_verify_local(args: argparse.Namespace) -> None:
    spec = MODEL_SPECS["qwen18b_chat"]
    results_dir = args.results_root / spec.key
    results_path = results_dir / "results.json"
    pass_path = results_dir / "verification_passed.json"
    fail_path = results_dir / "verification_failed.json"

    if results_path.exists() and not args.force:
        logging.info("Using existing verification results at %s", results_path)
        rows = load_json(results_path)["rows"]
    else:
        val_path = args.local_artifact_dir / "datasplits" / "val.json"
        vector_path = args.local_artifact_dir / "activations" / "candidate_vectors.pt"
        neutral_path = args.local_artifact_dir / "activations" / "neutral.pt"
        missing = [str(path) for path in (val_path, vector_path, neutral_path) if not path.exists()]
        if missing:
            diagnostics = {
                "status": "failed",
                "reason": "missing_local_qwen18b_chat_artifacts",
                "missing_paths": missing,
                "required_for": "verification against the local qwen_prompt_family_tradeoff_main.png Image Shows curve",
                "next_step": "Copy or restore the local runs_vision/Qwen-1_8B-chat datasplits and activations before launching the 3B/7B jobs.",
            }
            save_json(fail_path, diagnostics)
            logging.error("Missing local verification artifacts: %s", ", ".join(missing))
            raise SystemExit(2)

        logging.info("Loading local verification model %s", spec.hf_model)
        model = load_model(spec.hf_model)
        scorer = load_ratio_scorer(args.data_dir)

        captions, indices = load_caption_texts(val_path, args.n_val, selection="first", seed=args.seed)
        prompts = build_image_shows_prompts(model, captions)

        candidate_vectors = torch.load(vector_path)
        steering_vec = prepare_model_tensor(model, candidate_vectors[LOCAL_REFERENCE_LAYER])
        neutral_acts = torch.load(neutral_path)
        offset = prepare_model_tensor(model, neutral_acts.mean(dim=1)[LOCAL_REFERENCE_LAYER])

        base_payload = {
            "schema_version": 1,
            "status": "running",
            "run_kind": "local_reference_verification",
            "model_key": spec.key,
            "model_name": spec.display_name,
            "hf_model_name": spec.hf_model,
            "selected_steering_layer": LOCAL_REFERENCE_LAYER,
            "layer_selection_method": "fixed layer from local qwen_prompt_family_tradeoff_main.png source curve",
            "prompt": {
                "template_name": PROMPT_TEMPLATE_NAME,
                "instruction_template": PROMPT_INSTRUCTION_TEMPLATE,
                "output_prefix": PROMPT_PREFIX,
                "serialization": "model.apply_chat_template(instruction, output_prefix='The image shows')",
            },
            "validation": {
                "path": str(val_path),
                "n_val": len(captions),
                "selection": "first",
                "indices": indices[:10],
                "caption_sha256": caption_digest(captions),
            },
            "settings": {
                "seed": args.seed,
                "lambdas": args.lambdas,
                "max_new_tokens": args.max_new_tokens,
                "batch_size": args.batch_size,
                "generation": "greedy, do_sample=False, full-sequence steering",
                "metric": "normalized_spatial_ratio",
                "degeneration_cap_rate_for_best": DEGENERATION_CAP_RATE,
            },
            "steering": {
                "vector_path": str(vector_path),
                "neutral_offset_path": str(neutral_path),
                "intervention_method": "default",
                "full_sequence": True,
            },
            "reference": {
                "source": "qwen_prompt_family_tradeoff_main.png / A_image_shows full-validation curve",
                "rows": REFERENCE_IMAGE_SHOWS_ROWS,
                "verify_lambdas": args.verify_lambdas,
                "ratio_tolerance": args.ratio_tolerance,
                "degeneration_tolerance": args.degeneration_tolerance,
            },
        }
        rows = run_curve(
            model=model,
            prompts=prompts,
            scorer=scorer,
            layer=LOCAL_REFERENCE_LAYER,
            steering_vec=steering_vec,
            offset=offset,
            lambdas=args.lambdas,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            output_path=results_path,
            base_payload=base_payload,
        )

    passed, mismatches = compare_with_reference(
        rows,
        args.verify_lambdas,
        args.ratio_tolerance,
        args.degeneration_tolerance,
    )

    payload = load_json(results_path)
    payload["status"] = "passed" if passed else "failed"
    payload["verification"] = {
        "passed": passed,
        "mismatches": mismatches,
        "verify_lambdas": args.verify_lambdas,
        "ratio_tolerance": args.ratio_tolerance,
        "degeneration_tolerance": args.degeneration_tolerance,
    }
    save_json(results_path, payload)

    if passed:
        save_json(pass_path, {"status": "passed", "results_json": str(results_path)})
        if fail_path.exists():
            fail_path.unlink()
        logging.info("Local Qwen-1.8B-chat verification passed.")
        return

    diagnostics = verification_diagnostics(args, payload, mismatches)
    save_json(fail_path, diagnostics)
    if pass_path.exists():
        pass_path.unlink()
    logging.error("Local Qwen-1.8B-chat verification failed. Diagnostics: %s", fail_path)
    raise SystemExit(2)


def select_validation_dataframe(data_dir: Path, n_val: int, selection: str, seed: int):
    val_df = rivanna.load_caption_df(data_dir / "val.json")
    n = min(n_val, len(val_df))
    if selection == "first":
        indices = list(range(n))
    elif selection == "random":
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(val_df), size=n, replace=False)).tolist()
    else:
        raise ValueError(f"Unknown validation selection: {selection}")
    return val_df.iloc[indices].reset_index(drop=True), indices


def extract_or_load_vectors(args: argparse.Namespace, model, out_dir: Path, token_info: dict):
    artifact_dir = out_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    vector_path = artifact_dir / "candidate_vectors.pt"
    neutral_path = artifact_dir / "neutral.pt"
    extraction_meta_path = artifact_dir / "extraction_meta.json"

    if args.skip_extract and vector_path.exists():
        logging.info("Loading cached model-specific vectors from %s", vector_path)
        candidate_vectors_pt = torch.load(vector_path)
        candidate_vectors = [candidate_vectors_pt[i] for i in range(len(candidate_vectors_pt))]
        neutral_acts_mean = torch.load(neutral_path) if neutral_path.exists() else None
        return candidate_vectors, neutral_acts_mean, {
            "mode": "loaded_cached",
            "vector_path": str(vector_path),
            "neutral_offset_path": str(neutral_path) if neutral_path.exists() else None,
        }

    train_df = rivanna.load_caption_df(args.data_dir / "train.json")
    if "prompt" in train_df.columns and "output_prefix" in train_df.columns:
        logging.info("Building extraction prompts from stored diverse prompt/output_prefix columns.")
        extraction_prompts = model.apply_chat_template(
            train_df["prompt"].tolist(),
            output_prefix=train_df["output_prefix"].tolist(),
        )
    else:
        logging.warning("No stored diverse prompt columns; falling back to standardized Image Shows prompts.")
        extraction_prompts = build_image_shows_prompts(model, train_df["text"].tolist())
    train_df["prompt_formatted"] = extraction_prompts

    logging.info("Computing model-specific next-token spatial/descriptive scores.")
    train_bias = rivanna.compute_bias_scores_all(
        model,
        extraction_prompts,
        token_info["all_ids"],
        token_info["n_pos"],
        args.batch_size,
    )
    train_df["bias"] = train_bias

    label_col = next((col for col in ("vision_label", "label") if col in train_df.columns), None)
    if label_col:
        pos_df = train_df[train_df[label_col] == "spatial"].copy()
        neg_df = train_df[train_df[label_col] == "descriptive"].copy()
        logging.info("Label split: spatial=%d descriptive=%d", len(pos_df), len(neg_df))
    else:
        pos_df = train_df[train_df["bias"] >= args.bias_threshold].copy()
        neg_df = train_df[train_df["bias"] <= -args.bias_threshold].copy()
        logging.info("Bias-threshold split: spatial=%d descriptive=%d", len(pos_df), len(neg_df))

    if len(pos_df) == 0 or len(neg_df) == 0:
        raise RuntimeError("Could not construct non-empty spatial/descriptive extraction groups.")

    n_extract = min(len(pos_df), len(neg_df), args.n_train)
    pos_df = pos_df.nlargest(n_extract, "bias")
    neg_df = neg_df.nsmallest(n_extract, "bias")

    if "is_neutral" in train_df.columns and train_df["is_neutral"].any():
        neutral_candidates = train_df[train_df["is_neutral"]].copy()
    else:
        neutral_candidates = train_df[
            ~train_df.index.isin(pos_df.index) & ~train_df.index.isin(neg_df.index)
        ].copy()
    n_neutral = min(args.n_neutral, len(neutral_candidates))
    if n_neutral > 0:
        neutral_df = neutral_candidates.sample(n=n_neutral, random_state=args.seed).copy()
        neutral_acts = rivanna.extract_neutral_activations(model, neutral_df, args.batch_size)
        neutral_acts_mean = neutral_acts.mean(dim=1)
    else:
        neutral_acts_mean = None

    logging.info("Extracting model-specific WMD steering vectors.")
    candidate_vectors, _, _ = rivanna.extract_wmd_vectors(
        model,
        pos_df,
        neg_df,
        neutral_acts_mean,
        args.batch_size,
        PROMPT_PREFIX,
    )
    torch.save(torch.stack(candidate_vectors), vector_path)
    if neutral_acts_mean is not None:
        torch.save(neutral_acts_mean, neutral_path)

    meta = {
        "mode": "computed_model_specific",
        "vector_path": str(vector_path),
        "neutral_offset_path": str(neutral_path) if neutral_acts_mean is not None else None,
        "n_train_per_class": int(n_extract),
        "n_neutral": int(n_neutral),
        "train_bias_mean": float(np.mean(train_bias)),
        "train_bias_std": float(np.std(train_bias)),
        "prompt_source": "stored diverse prompt/output_prefix columns" if "prompt" in train_df.columns else "image_shows fallback",
    }
    save_json(extraction_meta_path, meta)
    return candidate_vectors, neutral_acts_mean, meta


def run_model(args: argparse.Namespace) -> None:
    spec = MODEL_SPECS[args.model_key]
    if args.require_verification:
        marker = args.results_root / "qwen18b_chat" / "verification_passed.json"
        if not marker.exists():
            raise SystemExit(
                f"Missing verification marker: {marker}\n"
                "Run verify-local first. The larger base-model jobs intentionally stop until "
                "the local Qwen-1.8B-chat Image Shows curve matches the reference."
            )

    out_dir = args.results_root / spec.key
    results_path = out_dir / "results.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading model %s", spec.hf_model)
    model = load_model(spec.hf_model, torch_dtype=torch.float16)
    scorer = load_ratio_scorer(args.data_dir)
    token_info = load_target_token_info(model, args.data_dir)
    candidate_vectors, neutral_acts_mean, extraction_meta = extract_or_load_vectors(
        args,
        model,
        out_dir,
        token_info,
    )

    val_sub, val_indices = select_validation_dataframe(args.data_dir, args.n_val, args.validation_selection, args.seed)
    captions = val_sub["text"].tolist()
    eval_prompts = build_image_shows_prompts(model, captions)

    if "prompt" in val_sub.columns and "output_prefix" in val_sub.columns and "bias" in val_sub.columns:
        layer_prompts = model.apply_chat_template(
            val_sub["prompt"].tolist(),
            output_prefix=val_sub["output_prefix"].tolist(),
        )
        layer_bias = val_sub["bias"].astype(float).to_numpy()
        offset_by_layer = []
        for layer_idx in range(model.n_layer):
            if neutral_acts_mean is not None:
                offset_by_layer.append(neutral_acts_mean[layer_idx])
            else:
                offset_by_layer.append(torch.zeros(model.hidden_size, dtype=torch.float64))
        best_layer, middle_layer, corr_top3, layer_scores = rivanna.select_best_layer_by_projection(
            model,
            layer_prompts,
            candidate_vectors,
            layer_bias,
            offset_by_layer,
            args.batch_size,
            filter_last_pct=args.filter_last_pct,
        )
        layer_selection_method = "projection_mismatch_rmse_on_diverse_validation_prompts"
    else:
        raise RuntimeError(
            "Validation data must include prompt/output_prefix/bias for standardized layer selection."
        )

    steering_vec = prepare_model_tensor(model, candidate_vectors[best_layer])
    if neutral_acts_mean is not None:
        offset = prepare_model_tensor(model, neutral_acts_mean[best_layer])
        offset_path = extraction_meta.get("neutral_offset_path")
    else:
        offset = prepare_model_tensor(model, torch.zeros(model.hidden_size))
        offset_path = None

    layer_scores_path = out_dir / "layer_scores.json"
    save_json(
        layer_scores_path,
        {
            "best_layer": best_layer,
            "middle_layer": middle_layer,
            "corr_top3": corr_top3,
            "layer_scores": layer_scores,
        },
    )

    base_payload = {
        "schema_version": 1,
        "status": "running",
        "run_kind": "model_specific_base_model_sweep",
        "model_key": spec.key,
        "model_name": spec.display_name,
        "hf_model_name": spec.hf_model,
        "selected_steering_layer": int(best_layer),
        "layer_selection_method": layer_selection_method,
        "middle_layer": int(middle_layer),
        "corr_top3": [int(layer) for layer in corr_top3],
        "layer_scores_path": str(layer_scores_path),
        "prompt": {
            "template_name": PROMPT_TEMPLATE_NAME,
            "instruction_template": PROMPT_INSTRUCTION_TEMPLATE,
            "output_prefix": PROMPT_PREFIX,
            "serialization": "model.apply_chat_template(instruction, output_prefix='The image shows')",
        },
        "validation": {
            "path": str(args.data_dir / "val.json"),
            "n_val": len(captions),
            "selection": args.validation_selection,
            "seed": args.seed,
            "indices": val_indices[:10],
            "caption_sha256": caption_digest(captions),
        },
        "settings": {
            "seed": args.seed,
            "lambdas": args.lambdas,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "n_train": args.n_train,
            "n_neutral": args.n_neutral,
            "generation": "greedy, do_sample=False, full-sequence steering",
            "metric": "normalized_spatial_ratio",
            "degeneration_cap_rate_for_best": DEGENERATION_CAP_RATE,
        },
        "steering": {
            "vector_path": extraction_meta.get("vector_path"),
            "neutral_offset_path": offset_path,
            "intervention_method": "default",
            "full_sequence": True,
            "extraction": extraction_meta,
        },
    }

    rows = run_curve(
        model=model,
        prompts=eval_prompts,
        scorer=scorer,
        layer=int(best_layer),
        steering_vec=steering_vec,
        offset=offset,
        lambdas=args.lambdas,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        output_path=results_path,
        base_payload=base_payload,
    )
    payload = load_json(results_path)
    payload["status"] = "completed"
    payload["best_under_20pct_degeneration"] = pick_best_row(rows)
    save_json(results_path, payload)
    logging.info("Saved standardized model sweep to %s", results_path)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--n-val", type=int, default=1000)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--lambdas", type=parse_int_list, default=list(DEFAULT_LAMBDAS))
    parser.add_argument("--seed", type=int, default=42)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    verify = sub.add_parser("verify-local", help="Rerun and verify the local Qwen-1.8B-chat Image Shows curve.")
    add_common_args(verify)
    verify.add_argument("--local-artifact-dir", type=Path, default=DEFAULT_LOCAL_ARTIFACT_DIR)
    verify.add_argument("--batch-size", type=int, default=MODEL_SPECS["qwen18b_chat"].batch_size)
    verify.add_argument("--verify-lambdas", type=parse_int_list, default=list(DEFAULT_VERIFY_LAMBDAS))
    verify.add_argument("--ratio-tolerance", type=float, default=0.01)
    verify.add_argument("--degeneration-tolerance", type=int, default=25)
    verify.add_argument("--force", action="store_true")
    verify.set_defaults(func=run_verify_local)

    run = sub.add_parser("run-model", help="Run a model-specific standardized base-model sweep.")
    add_common_args(run)
    run.add_argument("--model-key", required=True, choices=["qwen25_3b_base", "qwen25_7b_base"])
    run.add_argument("--n-train", type=int, default=800)
    run.add_argument("--n-neutral", type=int, default=200)
    run.add_argument("--batch-size", type=int, default=None)
    run.add_argument("--bias-threshold", type=float, default=0.05)
    run.add_argument("--filter-last-pct", type=float, default=0.05)
    run.add_argument("--validation-selection", choices=["first", "random"], default="first")
    run.add_argument("--skip-extract", action="store_true")
    run.add_argument("--require-verification", action=argparse.BooleanOptionalAction, default=True)
    run.set_defaults(func=run_model)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.results_root = args.results_root.resolve()
    args.data_dir = args.data_dir.resolve()
    args.lambdas = parse_int_list(args.lambdas)
    if hasattr(args, "verify_lambdas"):
        args.verify_lambdas = parse_int_list(args.verify_lambdas)
    if getattr(args, "batch_size", None) is None and getattr(args, "model_key", None):
        args.batch_size = MODEL_SPECS[args.model_key].batch_size
    args.func(args)


if __name__ == "__main__":
    main()

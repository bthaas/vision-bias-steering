import os
import logging
from pathlib import Path
from typing import List, Callable, Dict
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import torch
import torch.nn.functional as F
from torchtyping import TensorType

from . import ModelBase
from .intervention import get_intervention_func
from .steering_utils import get_all_layer_activations, scalar_projection
from ..config import Config
from ..utils import RMS, RMSE, save_to_json_file
from ..data.prompt_iterator import PromptIterator


def compute_target_probs(
    model: ModelBase,
    prompts: List[str],
    target_token_ids: Dict,
    batch_size: int = 32,
    constrained_softmax: bool = False,
):
    """Compute pos/neg target token probabilities for each prompt."""
    prompt_iterator = PromptIterator(prompts, batch_size=batch_size)
    pos_probs_all, neg_probs_all = torch.tensor([]), torch.tensor([])

    pos_ids = target_token_ids["pos"]
    neg_ids = target_token_ids["neg"]
    all_target_ids = pos_ids + neg_ids
    n_pos = len(pos_ids)

    for prompt_batch in prompt_iterator:
        logits = model.get_last_position_logits(prompt_batch)

        if constrained_softmax:
            target_logits = logits[:, all_target_ids]
            probs = F.softmax(target_logits, dim=-1)
            pos_probs = probs[:, :n_pos].sum(dim=-1)
            neg_probs = probs[:, n_pos:].sum(dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)
            pos_probs = probs[:, pos_ids].sum(dim=-1)
            neg_probs = probs[:, neg_ids].sum(dim=-1)

        pos_probs_all = torch.concat((pos_probs_all, pos_probs))
        neg_probs_all = torch.concat((neg_probs_all, neg_probs))

    return pos_probs_all.numpy(), neg_probs_all.numpy()


def resolve_label_column(val_data: pd.DataFrame, target_concept: str) -> str | None:
    for col in (f"{target_concept}_label", "label"):
        if col in val_data.columns:
            return col
    return None


def compute_label_metrics(pos_probs: np.ndarray, neg_probs: np.ndarray, labels: np.ndarray, pos_label: str, neg_label: str) -> Dict:
    pred_labels = np.where(pos_probs >= neg_probs, pos_label, neg_label)
    accuracy = float((pred_labels == labels).mean())
    pos_rate = float((pred_labels == pos_label).mean())
    mean_pos_prob = float(pos_probs.mean())
    mean_neg_prob = float(neg_probs.mean())
    return {
        "accuracy": accuracy,
        "pos_rate": pos_rate,
        "mean_pos_prob": mean_pos_prob,
        "mean_neg_prob": mean_neg_prob,
    }

def evaluate_candidate_vectors(
    model: ModelBase, prompts: List[str],
    candidate_vectors: TensorType["n_layer", "hidden_size"], 
    bias_scores: np.ndarray, save_dir: Path, 
    filter_layer_pct: float = 0.05, batch_size: int = 32,
    offsets: TensorType["n_layer", "hidden_size"] | None = None,
) -> List[Dict]:
    os.makedirs(save_dir, exist_ok=True)

    results, projections = [], []
    prompt_acts = get_all_layer_activations(model, prompts, batch_size)

    for layer in range(model.n_layer):
        vec = candidate_vectors[layer]
        acts = prompt_acts[layer]
        if offsets is not None:
            acts = acts - offsets[layer]
        projs = scalar_projection(acts, vec).numpy()

        r = pearsonr(projs, bias_scores)
        rmse = RMSE(projs, bias_scores)

        projections.append(projs.tolist())
        results.append({
            "layer": layer, 
            "corr": r.statistic, 
            "p_val": r.pvalue,
            "RMSE": rmse
        })

    np.save(save_dir / "projections.npy", np.array(projections))
    save_to_json_file(results, save_dir / "proj_correlation.json")
    
    max_layer = round(model.n_layer * (1 - filter_layer_pct)) - 1
    filtered_results = [x for x in results if x["layer"] < max_layer] # Filter layers close to the last layer
    top_layer_results = sorted(filtered_results, key=lambda x: x["RMSE"]) # Sort layers by RMSE

    logging.info(f"Top layers: {[x['layer'] for x in top_layer_results]}")
    save_to_json_file(top_layer_results, save_dir / "top_layers.json")

    return top_layer_results


def run_debias_test(
    model: ModelBase,
    prompts: List[str],
    target_token_ids: Dict,
    layer: int,
    intervene_func: Callable,
    batch_size: int = 32,
    constrained_softmax: bool = False,
):
    """Run debiasing test with optional constrained softmax."""
    prompt_iterator = PromptIterator(prompts, batch_size=batch_size)
    pos_probs_all, neg_probs_all = torch.tensor([]), torch.tensor([])
    
    # Combine all target token ids for constrained softmax
    pos_ids = target_token_ids["pos"]
    neg_ids = target_token_ids["neg"]
    all_target_ids = pos_ids + neg_ids
    n_pos = len(pos_ids)

    for prompt_batch in prompt_iterator:
        # Note: prompts are already templated, don't apply again
        logits = model.get_logits(
            prompt_batch, layer=layer, intervene_func=intervene_func
        )

        if constrained_softmax:
            # CONSTRAINED softmax: only over target tokens
            target_logits = logits[:, -1, all_target_ids]
            probs = F.softmax(target_logits, dim=-1)
            # Split back into pos and neg (probs now sum to 1 over target tokens)
            pos_probs = probs[:, :n_pos].sum(dim=-1)
            neg_probs = probs[:, n_pos:].sum(dim=-1)
        else:
            # Unconstrained softmax over full vocab (legacy behavior)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            pos_probs = probs[:, pos_ids].sum(dim=-1)
            neg_probs = probs[:, neg_ids].sum(dim=-1)
        
        pos_probs_all = torch.concat((pos_probs_all, pos_probs))
        neg_probs_all = torch.concat((neg_probs_all, neg_probs))

    bias = (pos_probs_all - neg_probs_all).numpy()
    normalized_bias = (pos_probs_all - neg_probs_all) / (pos_probs_all + neg_probs_all + 1e-10)
    return bias, normalized_bias.numpy(), pos_probs_all.numpy(), neg_probs_all.numpy()


def validate(cfg: Config, model: ModelBase, val_data: pd.DataFrame, target_token_ids):
    save_dir = cfg.artifact_path() / "validation"
    activation_dir = cfg.artifact_path() / "activations"
    candidate_vectors = torch.load(activation_dir / "candidate_vectors.pt")
    offsets = None
    if cfg.use_offset:
        # Used both during candidate vector extraction (extract.py) and during intervention (intervention.py).
        # If we don't apply this offset consistently, validation/debias results can be misleading.
        neutral_acts = torch.load(activation_dir / "neutral.pt")
        offsets = neutral_acts.mean(dim=1)

    if cfg.data_cfg.output_prefix:
        prompts = model.apply_chat_template(val_data.prompt.tolist(), output_prefix=val_data.output_prefix.tolist())
    else:
        prompts = model.apply_chat_template(val_data.prompt.tolist())

    if "pos_prob" in val_data.columns and "neg_prob" in val_data.columns:
        pos_probs_baseline = val_data["pos_prob"].to_numpy()
        neg_probs_baseline = val_data["neg_prob"].to_numpy()
    else:
        pos_probs_baseline, neg_probs_baseline = compute_target_probs(
            model,
            prompts,
            target_token_ids,
            batch_size=cfg.batch_size,
            constrained_softmax=cfg.constrained_softmax,
        )
    bias_baseline = pos_probs_baseline - neg_probs_baseline
    normalized_bias_baseline = (pos_probs_baseline - neg_probs_baseline) / (pos_probs_baseline + neg_probs_baseline + 1e-10)
    baseline_rms = RMS(bias_baseline)
    baseline_normalized_rms = RMS(normalized_bias_baseline)

    top_layer_results = evaluate_candidate_vectors(
        model, prompts, candidate_vectors, bias_baseline, 
        save_dir, cfg.filter_layer_pct, cfg.batch_size,
        offsets=offsets,
    )
   
    debiased_results = []
    score_outputs = []
    signal_report = {
        "target_concept": cfg.data_cfg.target_concept,
        "pos_label": cfg.data_cfg.pos_label,
        "neg_label": cfg.data_cfg.neg_label,
        "constrained_softmax": cfg.constrained_softmax,
        "baseline_rms": float(baseline_rms),
        "baseline_normalized_rms": float(baseline_normalized_rms),
    }
    label_col = resolve_label_column(val_data, cfg.data_cfg.target_concept)
    if label_col is not None:
        labels = val_data[label_col].to_numpy()
        signal_report["label_column"] = label_col
        signal_report["baseline_label_metrics"] = compute_label_metrics(
            pos_probs_baseline,
            neg_probs_baseline,
            labels,
            cfg.data_cfg.pos_label,
            cfg.data_cfg.neg_label,
        )
    else:
        signal_report["label_column"] = None


    coeff_candidates = [0.0]
    if cfg.debias_coeff is not None:
        coeff_candidates = [cfg.debias_coeff]
    elif cfg.optimize_coeff:
        if cfg.coeff_search_increment > 0:
            coeff_candidates = np.arange(
                cfg.coeff_search_min,
                cfg.coeff_search_max + (cfg.coeff_search_increment / 2),
                cfg.coeff_search_increment,
            ).tolist()
        else:
            coeff_candidates = [0.0]

    for layer_results in top_layer_results[:cfg.evaluate_top_n_layer]:
        layer = layer_results["layer"]
        steering_vec = model.set_dtype(candidate_vectors[layer])
        # Use "constant" method with negative coeff to shift toward neutral
        # Baseline mean bias is ~+0.18, so we need to subtract some spatial
        # The coeff_test showed coeff≈-5 to 0 gives near-neutral mean
        # Use default projection method (projects out bias direction)
        offset = 0
        if offsets is not None:
            offset = model.set_dtype(offsets[layer])
        best = None
        for coeff in coeff_candidates:
            intervene_func = get_intervention_func(
                steering_vec,
                method=cfg.intervention_method,
                coeff=coeff,
                offset=offset,
            )
            bias, normalized_bias, pos_probs, neg_probs = run_debias_test(
                model,
                prompts,
                target_token_ids,
                layer,
                intervene_func,
                batch_size=cfg.batch_size,
                constrained_softmax=cfg.constrained_softmax,
            )
            rms = RMS(bias)
            if best is None or rms < best["rms"]:
                best = {
                    "coeff": coeff,
                    "bias": bias,
                    "normalized_bias": normalized_bias,
                    "pos_probs": pos_probs,
                    "neg_probs": neg_probs,
                    "rms": rms,
                }

        bias = best["bias"]
        normalized_bias = best["normalized_bias"]
        pos_probs = best["pos_probs"]
        neg_probs = best["neg_probs"]
        rms = best["rms"]
        reduction_pct = None
        if baseline_rms > 0:
            reduction_pct = float((baseline_rms - rms) / baseline_rms * 100)
        is_undershoot = np.where(np.sign(bias) == np.sign(bias_baseline), 1, 0)
        undershoot = RMS(bias * is_undershoot)
        overshoot = RMS(bias * (1 - is_undershoot))

        debiased_entry = {
            "layer": layer,
            "rms": rms,
            "normalized_rms": RMS(normalized_bias),
            "overshoot": overshoot, 
            "undershoot": undershoot, 
            "reduction_pct": reduction_pct,
            "coeff": best["coeff"],
        }
        if label_col is not None:
            debiased_entry["label_metrics"] = compute_label_metrics(
                pos_probs,
                neg_probs,
                labels,
                cfg.data_cfg.pos_label,
                cfg.data_cfg.neg_label,
            )
        debiased_results.append(debiased_entry)
        score_outputs.append({
            "layer": layer,
            "bias_scores": bias.tolist(),
            "normalized_bias_scores": normalized_bias.tolist() 
        })

        print(f"Layer {layer}")
        print(f"RMS bias: {baseline_rms:.4f} (before), {rms: .4f} (after)")
        print(f"Best coeff: {best['coeff']}")
        print(f"Undershoot: {undershoot:.4f}, Overshoot: {overshoot:.4f}")

    save_to_json_file(score_outputs, save_dir / "debiased_scores.json")
    debiased_results = sorted(debiased_results, key=lambda x: x["rms"])
    save_to_json_file(debiased_results, save_dir / "debiased_results.json")
    save_to_json_file(signal_report, save_dir / "signal_report.json")

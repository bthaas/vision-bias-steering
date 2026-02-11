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
from ..utils import RMS, RMSE, save_to_json_file, loop_coeffs
from ..data.prompt_iterator import PromptIterator


def compute_target_probs(model, prompts, target_token_ids, batch_size=32, constrained_softmax=False):
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


def resolve_label_column(val_data, target_concept):
    for col in (f"{target_concept}_label", "label"):
        if col in val_data.columns:
            return col
    return None


def compute_label_metrics(pos_probs, neg_probs, labels, pos_label, neg_label):
    pred_labels = np.where(pos_probs >= neg_probs, pos_label, neg_label)
    accuracy = float((pred_labels == labels).mean())
    pos_rate = float((pred_labels == pos_label).mean())
    return {"accuracy": accuracy, "pos_rate": pos_rate, "mean_pos_prob": float(pos_probs.mean()), "mean_neg_prob": float(neg_probs.mean())}


def evaluate_candidate_vectors(model, prompts, candidate_vectors, bias_scores, save_dir, filter_layer_pct=0.05, batch_size=32, offsets=None):
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
        results.append({"layer": layer, "corr": r.statistic, "p_val": r.pvalue, "RMSE": rmse})
    np.save(save_dir / "projections.npy", np.array(projections))
    save_to_json_file(results, save_dir / "proj_correlation.json")
    max_layer = round(model.n_layer * (1 - filter_layer_pct)) - 1
    filtered_results = [x for x in results if x["layer"] < max_layer]
    top_layer_results = sorted(filtered_results, key=lambda x: x["RMSE"])
    logging.info(f"Top layers: {[x['layer'] for x in top_layer_results]}")
    save_to_json_file(top_layer_results, save_dir / "top_layers.json")
    return top_layer_results


def run_debias_test(model, prompts, target_token_ids, layer, intervene_func, batch_size=32, constrained_softmax=False):
    prompt_iterator = PromptIterator(prompts, batch_size=batch_size)
    pos_probs_all, neg_probs_all = torch.tensor([]), torch.tensor([])
    pos_ids = target_token_ids["pos"]
    neg_ids = target_token_ids["neg"]
    all_target_ids = pos_ids + neg_ids
    n_pos = len(pos_ids)
    for prompt_batch in prompt_iterator:
        logits = model.get_logits(prompt_batch, layer=layer, intervene_func=intervene_func)
        if constrained_softmax:
            target_logits = logits[:, -1, all_target_ids]
            probs = F.softmax(target_logits, dim=-1)
            pos_probs = probs[:, :n_pos].sum(dim=-1)
            neg_probs = probs[:, n_pos:].sum(dim=-1)
        else:
            probs = F.softmax(logits[:, -1, :], dim=-1)
            pos_probs = probs[:, pos_ids].sum(dim=-1)
            neg_probs = probs[:, neg_ids].sum(dim=-1)
        pos_probs_all = torch.concat((pos_probs_all, pos_probs))
        neg_probs_all = torch.concat((neg_probs_all, neg_probs))
    bias = (pos_probs_all - neg_probs_all).numpy()
    normalized_bias = (pos_probs_all - neg_probs_all) / (pos_probs_all + neg_probs_all + 1e-10)
    return bias, normalized_bias.numpy(), pos_probs_all.numpy(), neg_probs_all.numpy()


def validate(cfg, model, val_data, target_token_ids):
    save_dir = cfg.artifact_path() / "validation"
    activation_dir = cfg.artifact_path() / "activations"
    candidate_vectors = torch.load(activation_dir / "candidate_vectors.pt")
    offsets = None
    if cfg.use_offset:
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
        pos_probs_baseline, neg_probs_baseline = compute_target_probs(model, prompts, target_token_ids, batch_size=cfg.batch_size, constrained_softmax=cfg.constrained_softmax)

    bias_baseline = pos_probs_baseline - neg_probs_baseline
    normalized_bias_baseline = bias_baseline / (pos_probs_baseline + neg_probs_baseline + 1e-10)
    baseline_rms = RMS(bias_baseline)
    baseline_normalized_rms = RMS(normalized_bias_baseline)

    top_layer_results = evaluate_candidate_vectors(model, prompts, candidate_vectors, bias_baseline, save_dir, cfg.filter_layer_pct, cfg.batch_size, offsets=offsets)

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
    labels = None
    if label_col is not None:
        labels = val_data[label_col].to_numpy()
        signal_report["label_column"] = label_col
        signal_report["baseline_label_metrics"] = compute_label_metrics(pos_probs_baseline, neg_probs_baseline, labels, cfg.data_cfg.pos_label, cfg.data_cfg.neg_label)
    else:
        signal_report["label_column"] = None

    # Build coefficient list
    if cfg.optimize_coeff:
        coeffs = list(loop_coeffs(min_coeff=cfg.coeff_search_min, max_coeff=cfg.coeff_search_max, increment=cfg.coeff_search_increment))
        if 0.0 not in coeffs:
            coeffs.append(0.0)
        logging.info(f"Optimizing {len(coeffs)} coefficients from {min(coeffs)} to {max(coeffs)}")
    elif cfg.debias_coeff is not None:
        coeffs = [cfg.debias_coeff]
    else:
        coeffs = [0.0]

    method = cfg.intervention_method
    logging.info(f"Intervention method: {method}")

    for layer_results in top_layer_results[:cfg.evaluate_top_n_layer]:
        layer = layer_results["layer"]
        steering_vec = model.set_dtype(candidate_vectors[layer])
        offset = 0
        if offsets is not None:
            offset = model.set_dtype(offsets[layer])

        best_rms = float("inf")
        best_entry = None
        best_scores = None

        for coeff in coeffs:
            intervene_func = get_intervention_func(steering_vec, method=method, coeff=coeff, offset=offset)
            bias, normalized_bias, pos_probs, neg_probs = run_debias_test(model, prompts, target_token_ids, layer, intervene_func, batch_size=cfg.batch_size, constrained_softmax=cfg.constrained_softmax)
            rms = RMS(bias)
            reduction_pct = float((baseline_rms - rms) / baseline_rms * 100) if baseline_rms > 0 else None
            is_undershoot = np.where(np.sign(bias) == np.sign(bias_baseline), 1, 0)
            undershoot = RMS(bias * is_undershoot)
            overshoot = RMS(bias * (1 - is_undershoot))

            entry = {"layer": layer, "rms": rms, "normalized_rms": RMS(normalized_bias), "overshoot": overshoot, "undershoot": undershoot, "reduction_pct": reduction_pct, "coeff": coeff, "method": method}
            if label_col is not None:
                entry["label_metrics"] = compute_label_metrics(pos_probs, neg_probs, labels, cfg.data_cfg.pos_label, cfg.data_cfg.neg_label)
            scores = {"layer": layer, "coeff": coeff, "bias_scores": bias.tolist(), "normalized_bias_scores": normalized_bias.tolist()}

            logging.info(f"Layer {layer}, coeff {coeff:.1f}: RMS {rms:.4f}")
            if rms < best_rms:
                best_rms = rms
                best_entry = entry
                best_scores = scores

        debiased_results.append(best_entry)
        score_outputs.append(best_scores)
        print(f"Layer {layer} (best coeff: {best_entry['coeff']}, method: {method})")
        print(f"RMS bias: {baseline_rms:.4f} (before), {best_rms:.4f} (after), reduction: {best_entry['reduction_pct']:.2f}%")

    save_to_json_file(score_outputs, save_dir / "debiased_scores.json")
    debiased_results = sorted(debiased_results, key=lambda x: x["rms"])
    save_to_json_file(debiased_results, save_dir / "debiased_results.json")
    save_to_json_file(signal_report, save_dir / "signal_report.json")

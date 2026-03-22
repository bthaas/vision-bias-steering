import os, json
import random
import argparse
import warnings
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any

import torch
import torch.nn.functional as F
import numpy as np
from .config import Config, DataConfig
from .utils import save_to_json_file, loop_coeffs
from .data.prompt_iterator import PromptIterator
from .steering import load_model, ModelBase, extract_candidate_vectors, \
    validate, get_intervention_func, get_target_token_ids, compute_projections
from .eval import load_evaluation_task

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.set_grad_enabled(False);
logging.basicConfig(level=logging.INFO)
DATASET_DIR = Path(__file__).resolve().parent / "data" / "datasets"


def _load_datasplits(cfg: DataConfig, save_dir: Path, use_cache: bool = False):
    from .data.load_dataset import load_datasplits
    return load_datasplits(cfg, save_dir, use_cache=use_cache)


def _load_target_words(target_concept: str = "vision"):
    return json.load(open(DATASET_DIR / "target_words.json", "r"))[target_concept]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default=None, help='Load configuration from file.')
    parser.add_argument('--model_name', type=str, help='Model name')
    parser.add_argument('--method', type=str, default="WMD", choices=["WMD", "MD"], help='Method for computing candidate vectors.')
    parser.add_argument('--use_offset', action='store_true', help="Offset by neutral examples.")
    parser.add_argument('--n_train_per_label', type=int, default=800, help="Number of training examples per label.")
    parser.add_argument('--n_val', type=int, default=1600, help="Number of validation examples.")
    parser.add_argument('--bias_threshold', type=float, default=0.1)
    parser.add_argument('--target_concept', type=str, default="vision", help='Target concept (vision)')
    parser.add_argument('--pos_label', type=str, default="spatial", help='Positive label (e.g., spatial)')
    parser.add_argument('--neg_label', type=str, default="descriptive", help='Negative label (e.g., descriptive)')
    parser.add_argument('--filter_layer_pct', type=float, default=0.05, help='Filter last N percentage layers.')
    parser.add_argument('--evaluate_top_n_layer', type=int, default=5, help='Evaluate top n layers.')
    parser.add_argument('--force_layer', type=int, default=None, help='Force intervention on a specific layer during validation.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--seed', type=int, default=4238, help='Random seed.')
    parser.add_argument('--save_dir', type=str, default=None, help='Save results to specified directory.')
    parser.add_argument('--use_cache', action='store_true', help='Reuse stored cached results.')
    parser.add_argument('--run_eval', action='store_true', help='Run transferability evaluation.')
    parser.add_argument('--layer', type=int, help="Intervention layer.")
    parser.add_argument('--intervention_method', type=str, default=None, choices=["default", "constant"], help="Intervention method")
    parser.add_argument('--coeff', type=float, default=0, help="Steering coefficient.")
    parser.add_argument('--debias_coeff', type=float, default=None, help="Debias coefficient override (skip coeff search).")
    parser.add_argument('--optimize_coeff', action='store_true', help="Search coefficients to maximize RMS reduction.")
    parser.add_argument('--coeff_search_min', type=float, default=None, help="Min coefficient for search.")
    parser.add_argument('--coeff_search_max', type=float, default=None, help="Max coefficient for search.")
    parser.add_argument('--coeff_search_increment', type=float, default=None, help="Coefficient search increment.")
    parser.add_argument('--constrained_softmax', action='store_true', help="Constrain softmax to target tokens only.")
    parser.add_argument('--score_mode', type=str, default=None, choices=["prob_diff", "logit_margin", "adaptive"], help="Bias scoring mode used for WMD extraction and validation.")
    parser.add_argument('--prompt_template_sweep', action='store_true', help="Evaluate best steering setting across multiple prompt templates.")
    parser.add_argument('--prompt_template_max_examples', type=int, default=None, help="Max val examples for prompt template diagnostics.")
    parser.add_argument('--generate_vision_cases', action='store_true', help="Generate 5 steering-coefficient vision case plots.")
    parser.add_argument('--case_num', type=int, default=5, help="Number of case-study prompts to include.")
    parser.add_argument('--case_candidate_pool', type=int, default=180, help="How many val prompts to scan for case selection.")
    parser.add_argument('--case_min_coeff', type=float, default=-240.0, help="Minimum coefficient for case sweep.")
    parser.add_argument('--case_max_coeff', type=float, default=240.0, help="Maximum coefficient for case sweep.")
    parser.add_argument('--case_increment', type=float, default=20.0, help="Coefficient increment for case sweep.")
    parser.add_argument('--case_zoom_floor', type=float, default=0.01, help="Minimum y-axis max for zoomed case plots.")
    parser.add_argument('--case_zoom_pad', type=float, default=1.25, help="Padding multiplier for per-case y-axis zoom.")
    parser.add_argument('--case_min_edge_advantage', type=float, default=0.008, help="Minimum class advantage required at both coefficient extremes.")
    parser.add_argument('--case_max_balance_gap', type=float, default=0.0035, help="Maximum |spatial-descriptive| at the chosen balance coefficient.")
    parser.add_argument('--case_max_near_zero_gap', type=float, default=0.008, help="Maximum |spatial-descriptive| at lambda ~= 0.")
    parser.add_argument('--case_balance_coeff_abs_max', type=float, default=90.0, help="Balance coefficient must be within +/- this value.")
    parser.add_argument('--case_constrained_softmax', action='store_true', help="Use constrained softmax over spatial+descriptive tokens for case plotting.")
    parser.add_argument('--case_auto_orient', action='store_true', help="Auto-flip coefficient direction so +lambda trends spatial on average.")
    parser.add_argument('--case_search_top_layers', type=int, default=0, help="If >0, search this many top validation layers and pick the best case-quality layer.")
    parser.add_argument('--case_strict_only', action='store_true', help="Only keep strict high-quality cases; do not backfill weak examples.")
    parser.add_argument('--case_output_html', type=str, default=None, help="Output HTML path for case plots.")
    parser.add_argument('--case_output_json', type=str, default=None, help="Output JSON path for case summaries.")
    parser.add_argument('--run_two_experiments', action='store_true', help="Run fill-in-the-blank and multi-token continuation experiments side-by-side.")
    parser.add_argument('--exp_layer', type=int, default=None, help="Layer override for the two-experiments run.")
    parser.add_argument('--exp_coeff', type=float, default=None, help="Steering coefficient override for the two-experiments run.")
    parser.add_argument('--exp_tokens', type=int, default=10, help="Number of continuation tokens for multi-token experiment.")
    parser.add_argument(
        '--exp_continuation_mode',
        type=str,
        default="greedy",
        choices=["greedy", "beam", "compare"],
        help="How to score continuation: greedy path, beam-search expectation, or both for comparison.",
    )
    parser.add_argument('--exp_beam_width', type=int, default=4, help="Beam width for continuation beam search.")
    parser.add_argument('--exp_beam_top_k', type=int, default=8, help="Top-k token expansions per beam step.")
    parser.add_argument('--exp_use_case_examples', action='store_true', help="Use examples from vision_steering_cases.json for both experiments.")
    parser.add_argument('--exp_cases_file', type=str, default=None, help="Path to vision_steering_cases.json file.")
    parser.add_argument('--exp_case_count', type=int, default=5, help="How many case examples to use when --exp_use_case_examples is set.")
    parser.add_argument('--exp_case_candidate_pool', type=int, default=160, help="How many val examples to scan when selecting scenario-specific experiment cases.")
    parser.add_argument('--exp_output_json', type=str, default=None, help="Output JSON path for two-experiments results.")
    parser.add_argument('--plot_two_experiments_cases', action='store_true', help="Plot coefficient sweeps for fill-in and multi-token continuation on the same case examples.")
    parser.add_argument('--exp_plot_output_html', type=str, default=None, help="Output HTML path for two-experiment sweep plots.")
    parser.add_argument('--exp_plot_output_json', type=str, default=None, help="Output JSON path for two-experiment sweep data.")
    # Handcrafted eval set
    parser.add_argument('--eval_set', type=str, default="benchmark", choices=["benchmark", "handcrafted"],
                        help="Evaluation dataset to use. 'handcrafted' uses the curated eval file instead of the benchmark val split.")
    parser.add_argument('--handcrafted_eval_file', type=str, default=None,
                        help="Path to handcrafted eval JSON (default: data/handcrafted_eval.json at project root).")
    # Generation logging
    parser.add_argument('--log_generations', action='store_true', help="Log generated text at each steering coefficient alongside tracked token probs.")
    parser.add_argument('--log_gen_output_dir', type=str, default=None, help="Output directory for generation logs (default: results/generation_logs).")
    parser.add_argument('--log_gen_prompt_type', type=str, default="image_shows", help="Prompt template name embedded in the log filename.")
    parser.add_argument('--log_gen_decoding', type=str, default="greedy", choices=["greedy", "beam"], help="Decoding strategy for generation logging.")
    parser.add_argument('--log_gen_max_tokens', type=int, default=20, help="Max new tokens to generate per (example, lambda) pair.")
    parser.add_argument('--log_gen_coeff_min', type=float, default=None, help="Min coefficient for generation log sweep (defaults to coeff_search_min).")
    parser.add_argument('--log_gen_coeff_max', type=float, default=None, help="Max coefficient for generation log sweep (defaults to coeff_search_max).")
    parser.add_argument('--log_gen_coeff_increment', type=float, default=None, help="Coefficient increment for generation log sweep (defaults to coeff_search_increment).")
    parser.add_argument('--log_gen_n_examples', type=int, default=None, help="Number of val examples to log (default: all).")
    parser.add_argument('--log_gen_beam_width', type=int, default=4, help="Beam width for beam decoding in generation log.")
    parser.add_argument('--log_gen_beam_top_k', type=int, default=8, help="Top-k expansions per beam step in generation log.")
    parser.add_argument('--log_gen_use_cases', action='store_true', help="Use case examples from vision_steering_cases.json instead of val examples.")
    parser.add_argument('--log_gen_cases_file', type=str, default=None, help="Path to vision_steering_cases.json (defaults to artifact_path/validation/vision_steering_cases.json).")
    parser.add_argument('--log_gen_layer', type=int, default=None, help="Layer override for generation logging (defaults to best validated layer).")
    parser.add_argument('--log_gen_intervention_method', type=str, default="default",
                        choices=["default", "constant"],
                        help="Intervention method for generation logging. 'default' (orthogonal projection) "
                             "is stable across multi-token generation steps; 'constant' matches the "
                             "debiasing pipeline but degrades at large coefficients. (default: 'default')")
    return parser.parse_args()


def get_baseline_results(
    model: ModelBase, prompts: List[str],
    target_token_ids: Dict[str, List], 
    batch_size: int = 32,
    constrained_softmax: bool = False,
    score_mode: str = "logit_margin",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get baseline probabilities and bias scores."""
    pos_probs_all, neg_probs_all = torch.tensor([]), torch.tensor([])
    bias_scores_all = torch.tensor([])
    logit_margin_all = torch.tensor([])
    prompt_iterator = PromptIterator(prompts, batch_size=batch_size)
    
    # Combine all target token ids for constrained softmax
    pos_ids = target_token_ids["pos"]
    neg_ids = target_token_ids["neg"]
    all_target_ids = pos_ids + neg_ids
    n_pos = len(pos_ids)

    for prompt_batch in prompt_iterator:
        logits = model.get_last_position_logits(prompt_batch)
        
        if constrained_softmax:
            # CONSTRAINED softmax: only over target tokens
            target_logits = logits[:, all_target_ids]
            probs = F.softmax(target_logits, dim=-1)
            # Split back into pos and neg (probs now sum to 1 over target tokens)
            pos_probs = probs[:, :n_pos].sum(dim=-1)
            neg_probs = probs[:, n_pos:].sum(dim=-1)
            pos_logits = target_logits[:, :n_pos]
            neg_logits = target_logits[:, n_pos:]
        else:
            # Unconstrained softmax over full vocab (legacy behavior)
            probs = F.softmax(logits, dim=-1)
            pos_probs = probs[:, pos_ids].sum(dim=-1)
            neg_probs = probs[:, neg_ids].sum(dim=-1)
            pos_logits = logits[:, pos_ids]
            neg_logits = logits[:, neg_ids]

        logit_margin = torch.logsumexp(pos_logits, dim=-1) - torch.logsumexp(neg_logits, dim=-1)
        prob_diff = pos_probs - neg_probs
        if score_mode == "logit_margin":
            bias_scores = logit_margin
        else:
            bias_scores = prob_diff

        pos_probs_all = torch.concat((pos_probs_all, pos_probs))
        neg_probs_all = torch.concat((neg_probs_all, neg_probs))
        bias_scores_all = torch.concat((bias_scores_all, bias_scores))
        logit_margin_all = torch.concat((logit_margin_all, logit_margin))

    return pos_probs_all.numpy(), neg_probs_all.numpy(), bias_scores_all.numpy(), logit_margin_all.numpy()


def remove_overlapping_target_ids(target_token_ids: Dict[str, List[int]]) -> Dict[str, List[int]]:
    overlap = set(target_token_ids["pos"]).intersection(set(target_token_ids["neg"]))
    if overlap:
        logging.warning(f"Removing {len(overlap)} overlapping target token ids shared by pos/neg classes.")
        target_token_ids["pos"] = [x for x in target_token_ids["pos"] if x not in overlap]
        target_token_ids["neg"] = [x for x in target_token_ids["neg"] if x not in overlap]
    if len(target_token_ids["pos"]) == 0 or len(target_token_ids["neg"]) == 0:
        raise ValueError("Target token ids became empty after overlap filtering. Please revise target_words.json.")
    return target_token_ids


def pick_adaptive_score_mode(train_df, pos_label: str, neg_label: str, target_concept: str) -> str:
    label_col = f"{target_concept}_label" if f"{target_concept}_label" in train_df.columns else ("label" if "label" in train_df.columns else None)
    candidates = ["bias_prob_diff", "bias_logit_margin"]
    if label_col is None:
        # Fallback: choose score with broader dynamic range.
        scored = [(col, float(train_df[col].abs().mean())) for col in candidates]
        best_col = sorted(scored, key=lambda x: x[1], reverse=True)[0][0]
    else:
        scored = []
        labels = train_df[label_col]
        for col in candidates:
            preds = np.where(train_df[col].to_numpy() >= 0, pos_label, neg_label)
            acc = float((preds == labels).mean())
            scored.append((col, acc))
        best_col = sorted(scored, key=lambda x: x[1], reverse=True)[0][0]
    return "prob_diff" if best_col == "bias_prob_diff" else "logit_margin"


def weighted_sample(df, sample_size, n_bins=40):
    import pandas as pd
    df2 = df.copy()
    df2["bin"] = pd.cut(df2["bias"].abs(), n_bins)
    bin_freq = df2.groupby("bin", observed=True).size().to_dict()
    df2["sample_weight"] = df2["bin"].apply(lambda x: 1 / bin_freq[x]**2)
    temp = df2.sample(sample_size, weights="sample_weight")
    return temp


def train_and_validate(cfg: Config, model: ModelBase):
    datasplits_dir = cfg.artifact_path() / "datasplits"
    data_cfg = cfg.data_cfg
    datasets = _load_datasplits(data_cfg, datasplits_dir, use_cache=cfg.use_cache)
    os.makedirs(datasplits_dir, exist_ok=True)

    logging.info("Preprocessing train/val data")
    target_words_by_label = _load_target_words(target_concept=data_cfg.target_concept)
    target_token_ids = {}
    for label, k in zip([data_cfg.pos_label, data_cfg.neg_label], ["pos", "neg"]):
        target_token_ids[k] = get_target_token_ids(model.tokenizer, target_words_by_label[label])
    target_token_ids = remove_overlapping_target_ids(target_token_ids)

    for split in ["train", "val"]:
        df = datasets[split].copy()

        has_cached_probs = "pos_prob" in df.columns and "neg_prob" in df.columns
        has_cached_bias_cols = "bias_prob_diff" in df.columns and "bias_logit_margin" in df.columns
        if cfg.use_cache is True and has_cached_probs and (has_cached_bias_cols or cfg.score_mode != "adaptive"):
            continue

        logging.info(f"Getting baseline results for {split} split")
        if data_cfg.output_prefix:
            prompts = model.apply_chat_template(df["prompt"].tolist(), output_prefix=df["output_prefix"].tolist())
        else:
            prompts = model.apply_chat_template(df["prompt"].tolist())
            
        effective_score_mode = "prob_diff" if cfg.score_mode == "adaptive" else cfg.score_mode
        pos_probs, neg_probs, bias_scores, logit_margin = get_baseline_results(
            model,
            prompts,
            target_token_ids,
            batch_size=cfg.batch_size,
            constrained_softmax=cfg.constrained_softmax,
            score_mode=effective_score_mode,
        )
        df["pos_prob"] = pos_probs
        df["neg_prob"] = neg_probs
        df["bias_prob_diff"] = pos_probs - neg_probs
        df["bias_logit_margin"] = logit_margin
        df["bias"] = bias_scores
            
        datasets[split] = df
        save_to_json_file(df.to_dict("records"), datasplits_dir / f"{split}.json")

    if cfg.score_mode == "adaptive":
        selected_mode = pick_adaptive_score_mode(
            datasets["train"],
            pos_label=data_cfg.pos_label,
            neg_label=data_cfg.neg_label,
            target_concept=data_cfg.target_concept,
        )
        logging.info(f"Adaptive score mode selected: {selected_mode}")
        cfg.score_mode = selected_mode
        for split in ["train", "val"]:
            if selected_mode == "logit_margin" and "bias_logit_margin" in datasets[split].columns:
                datasets[split]["bias"] = datasets[split]["bias_logit_margin"]
            elif "bias_prob_diff" in datasets[split].columns:
                datasets[split]["bias"] = datasets[split]["bias_prob_diff"]
            save_to_json_file(datasets[split].to_dict("records"), datasplits_dir / f"{split}.json")

    if not cfg.use_cache or not Path(cfg.artifact_path() / "activations/candidate_vectors.pt").is_file():
        train_data = datasets["train"]
        pos_examples = train_data[(train_data.bias > data_cfg.bias_threshold)]
        neg_examples = train_data[(train_data.bias < -data_cfg.bias_threshold)]
        
        if len(pos_examples) == 0 or len(neg_examples) == 0:
            logging.warning(f"No examples found with bias threshold {data_cfg.bias_threshold}. Found {len(pos_examples)} positive and {len(neg_examples)} negative examples.")
            logging.warning(f"Adjusting threshold or using all examples. Current bias range: [{train_data.bias.min():.4f}, {train_data.bias.max():.4f}]")
            # Use a lower threshold or all examples
            if len(pos_examples) == 0:
                pos_examples = train_data.nlargest(max(1, data_cfg.n_train or 10), 'bias')
            if len(neg_examples) == 0:
                neg_examples = train_data.nsmallest(max(1, data_cfg.n_train or 10), 'bias')

        if data_cfg.n_train is not None:
            if data_cfg.weighted_sample:
                pos_examples = weighted_sample(pos_examples, sample_size=min([data_cfg.n_train, pos_examples.shape[0]]))
                neg_examples = weighted_sample(neg_examples, sample_size=min([data_cfg.n_train, neg_examples.shape[0]]))
            else:
                pos_examples = pos_examples.sample(n=min([data_cfg.n_train, pos_examples.shape[0]]))
                neg_examples = neg_examples.sample(n=min([data_cfg.n_train, neg_examples.shape[0]]))

        if cfg.use_offset:
            neutral_examples = train_data[(train_data.bias.abs() <= data_cfg.bias_threshold)]
        else:
            neutral_examples = None
        extract_candidate_vectors(cfg, model, pos_examples, neg_examples, neutral_examples)

    validate(cfg, model, datasets["val"], target_token_ids)


def eval(cfg: Config, model: ModelBase, layer=None, coeff=0, eval_tasks=["winogenerated"], batch_size=32):
    if layer is None:
        layer = json.load(open(cfg.artifact_path() / "validation/top_layers.json", "r"))[0]["layer"]

    print(f"Intervene layer: {layer}")
    save_dir = cfg.artifact_path() / "evaluation"
    os.makedirs(save_dir, exist_ok=True)

    candidate_vectors = torch.load(cfg.artifact_path() / f"activations/candidate_vectors.pt")
    steering_vec = candidate_vectors[layer]
    steering_vec = model.set_dtype(steering_vec)

    for task_name in eval_tasks:
        logging.info(f"Running evaluation task: {task_name}")
        task = load_evaluation_task(task_name)
        for subtask in task.get_subtask_list():
            task.run_eval(model, steering_vec, layer, save_dir, coeff=0, batch_size=batch_size)
            inputs = task.prepare_inputs(model.apply_chat_template, subtask=subtask)
            prompts = [x["prompt"] for x in inputs]
            projections = compute_projections(model, steering_vec, layer, prompts, batch_size=cfg.batch_size)
            np.save(save_dir / f"{task.task_name}_{subtask}_projections.npy", projections.numpy())

    # task = load_evaluation_task(eval_tasks[0])
    # coeffs = loop_coeffs(min_coeff=-80, max_coeff=-20, increment=10) + loop_coeffs(min_coeff=-15, max_coeff=15, increment=5) + loop_coeffs(min_coeff=20, max_coeff=80, increment=10)
    # task.run_steering_loop(model, steering_vec, layer, save_dir, coeffs=coeffs, test_size=1200, batch_size=batch_size)


def _class_probs_for_prompts(model, prompts, layer, intervene_func, pos_ids, neg_ids, batch_size, constrained_softmax=False):
    pos_all, neg_all = [], []
    all_ids = pos_ids + neg_ids
    n_pos = len(pos_ids)
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        logits = model.get_logits(batch, layer=layer, intervene_func=intervene_func)
        if constrained_softmax:
            target_logits = logits[:, -1, all_ids]
            probs = F.softmax(target_logits, dim=-1)
            pos = probs[:, :n_pos].sum(dim=-1)
            neg = probs[:, n_pos:].sum(dim=-1)
        else:
            probs = F.softmax(logits[:, -1, :], dim=-1)
            pos = probs[:, pos_ids].sum(dim=-1)
            neg = probs[:, neg_ids].sum(dim=-1)
        pos_all.append(pos.detach().cpu().numpy())
        neg_all.append(neg.detach().cpu().numpy())
    return np.concatenate(pos_all), np.concatenate(neg_all)


def _top_token_for_prompt(model, prompt: str, layer: int, intervene_func, pos_ids, neg_ids, constrained_softmax=False):
    logits = model.get_logits([prompt], layer=layer, intervene_func=intervene_func)
    probs_full = F.softmax(logits[0, -1, :], dim=-1)
    top_vocab_id = int(torch.argmax(probs_full).item())
    top_vocab_token = model.tokenizer.decode([top_vocab_id]).replace("\n", "\\n").strip()
    top_vocab_prob = float(probs_full[top_vocab_id].item())

    tracked_ids = pos_ids + neg_ids
    if constrained_softmax:
        target_logits = logits[0, -1, tracked_ids]
        probs_tracked = F.softmax(target_logits, dim=-1)
        tracked_probs = probs_tracked
        n_pos = len(pos_ids)
        pos_prob = float(probs_tracked[:n_pos].sum().item())
        neg_prob = float(probs_tracked[n_pos:].sum().item())
    else:
        tracked_probs = probs_full[tracked_ids]
        pos_prob = float(probs_full[pos_ids].sum().item())
        neg_prob = float(probs_full[neg_ids].sum().item())
    tracked_idx = int(torch.argmax(tracked_probs).item())
    top_tracked_id = tracked_ids[tracked_idx]
    top_tracked_token = model.tokenizer.decode([top_tracked_id]).replace("\n", "\\n").strip()
    top_tracked_prob = float(tracked_probs[tracked_idx].item())
    top_tracked_class = "spatial" if top_tracked_id in set(pos_ids) else "descriptive"

    return {
        "top_vocab_token": top_vocab_token,
        "top_vocab_prob": top_vocab_prob,
        "top_tracked_token": top_tracked_token,
        "top_tracked_prob": top_tracked_prob,
        "top_tracked_class": top_tracked_class,
        "spatial_prob": pos_prob,
        "descriptive_prob": neg_prob,
    }


def _pick_case_examples(rows, coeffs, pos_mat, neg_mat, n_cases: int, args):
    mid_idx = int(np.argmin(np.abs(np.array(coeffs))))
    max_abs_coeff = max(abs(coeffs[0]), abs(coeffs[-1]))
    ranked = []
    for i, row in enumerate(rows):
        neg_edge = float(neg_mat[i, 0] - pos_mat[i, 0])  # descriptive should dominate at min coeff
        pos_edge = float(pos_mat[i, -1] - neg_mat[i, -1])  # spatial should dominate at max coeff
        abs_gap = np.abs(pos_mat[i] - neg_mat[i])
        balance_idx = int(np.argmin(abs_gap))
        balance_coeff = float(coeffs[balance_idx])
        balance_gap = float(abs_gap[balance_idx])
        near_zero_gap = float(abs_gap[mid_idx])
        pos_gain = float(pos_mat[i, -1] - pos_mat[i, 0])
        neg_drop = float(neg_mat[i, 0] - neg_mat[i, -1])
        transition_strength = min(pos_gain, neg_drop)
        center_penalty = abs(balance_coeff) / max_abs_coeff if max_abs_coeff > 0 else 0.0
        score = (neg_edge + pos_edge) - (2.0 * balance_gap) - near_zero_gap
        ranked.append({
            "idx": i,
            "text": row["text"],
            "neg_edge": neg_edge,
            "pos_edge": pos_edge,
            "balance_idx": balance_idx,
            "balance_coeff": balance_coeff,
            "balance_gap": balance_gap,
            "near_zero_gap": near_zero_gap,
            "transition_strength": transition_strength,
            "score": score + 2.0 * transition_strength - center_penalty,
        })

    strict = [
        x for x in ranked
        if x["neg_edge"] >= args.case_min_edge_advantage
        and x["pos_edge"] >= args.case_min_edge_advantage
        and x["balance_gap"] <= args.case_max_balance_gap
        and x["near_zero_gap"] <= args.case_max_near_zero_gap
        and abs(x["balance_coeff"]) <= args.case_balance_coeff_abs_max
        and x["transition_strength"] > 0
    ]
    strict = sorted(strict, key=lambda x: (-x["score"], x["balance_gap"]))
    if args.case_strict_only:
        return strict[:n_cases]
    if len(strict) >= n_cases:
        return strict[:n_cases]

    fallback = sorted(ranked, key=lambda x: (-x["score"], x["balance_gap"]))
    used = {x["idx"] for x in strict}
    strict.extend([x for x in fallback if x["idx"] not in used])
    return strict[:n_cases]


def generate_vision_cases(cfg: Config, model: ModelBase, args):
    if cfg.data_cfg.target_concept != "vision":
        raise ValueError("Case plotting is currently implemented for target_concept=vision only.")

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    artifact_dir = cfg.artifact_path()
    val_path = artifact_dir / "datasplits/val.json"
    if not val_path.exists():
        raise FileNotFoundError(f"Missing val split at {val_path}. Run training/validation first.")

    with open(val_path, "r") as f:
        val_rows = json.load(f)
    rows = val_rows[:min(args.case_candidate_pool, len(val_rows))]
    # Preferred template after diagnostics:
    # - image_shows: "Describe this image:" + "The image shows"
    # Also tested in-template diagnostics:
    # - scene_is: "Continue describing this scene:" + "The scene is"
    # - in_scene_the: "Describe this image:" + "In this scene, the"
    image_show_prompts = [f"Describe this image:\n{r['text']}" for r in rows]
    if cfg.data_cfg.output_prefix:
        prompts = model.apply_chat_template(image_show_prompts, output_prefix=["The image shows"] * len(image_show_prompts))
    else:
        prompts = model.apply_chat_template(image_show_prompts)

    target_words_by_label = _load_target_words(target_concept="vision")
    target_token_ids = {
        "pos": get_target_token_ids(model.tokenizer, target_words_by_label[cfg.data_cfg.pos_label]),
        "neg": get_target_token_ids(model.tokenizer, target_words_by_label[cfg.data_cfg.neg_label]),
    }
    target_token_ids = remove_overlapping_target_ids(target_token_ids)

    candidate_vectors = torch.load(artifact_dir / "activations/candidate_vectors.pt")
    default_layer = cfg.force_layer if cfg.force_layer is not None else 5

    coeffs = [float(x) for x in loop_coeffs(args.case_min_coeff, args.case_max_coeff, args.case_increment)]
    if 0.0 not in coeffs:
        coeffs.append(0.0)
        coeffs = sorted(coeffs)

    pos_mat = np.zeros((len(prompts), len(coeffs)), dtype=np.float64)
    neg_mat = np.zeros((len(prompts), len(coeffs)), dtype=np.float64)
    layer_candidates = [default_layer]
    if args.case_search_top_layers > 0:
        top_layers_path = artifact_dir / "validation/top_layers.json"
        if top_layers_path.exists():
            try:
                with open(top_layers_path, "r") as f:
                    top_data = json.load(f)
                top_from_file = [int(x["layer"]) for x in top_data[: args.case_search_top_layers]]
                if top_from_file:
                    layer_candidates = top_from_file
            except Exception:
                pass

    best = None
    for layer in layer_candidates:
        steering_vec = model.set_dtype(candidate_vectors[layer])
        coeff_sign = 1.0
        if args.case_auto_orient:
            probe = max(abs(coeffs[0]), abs(coeffs[-1]))
            low_intervene = get_intervention_func(steering_vec, method="constant", coeff=-probe)
            high_intervene = get_intervention_func(steering_vec, method="constant", coeff=probe)
            pos_low, neg_low = _class_probs_for_prompts(
                model, prompts, layer, low_intervene,
                target_token_ids["pos"], target_token_ids["neg"], cfg.batch_size,
                constrained_softmax=args.case_constrained_softmax,
            )
            pos_high, neg_high = _class_probs_for_prompts(
                model, prompts, layer, high_intervene,
                target_token_ids["pos"], target_token_ids["neg"], cfg.batch_size,
                constrained_softmax=args.case_constrained_softmax,
            )
            low_gap = float((pos_low - neg_low).mean())
            high_gap = float((pos_high - neg_high).mean())
            if high_gap < low_gap:
                coeff_sign = -1.0

        pos_mat_layer = np.zeros((len(prompts), len(coeffs)), dtype=np.float64)
        neg_mat_layer = np.zeros((len(prompts), len(coeffs)), dtype=np.float64)
        for j, coeff in enumerate(coeffs):
            applied_coeff = coeff_sign * coeff
            intervene = get_intervention_func(steering_vec, method="constant", coeff=applied_coeff)
            pos, neg = _class_probs_for_prompts(
                model, prompts, layer, intervene,
                target_token_ids["pos"], target_token_ids["neg"], cfg.batch_size,
                constrained_softmax=args.case_constrained_softmax,
            )
            pos_mat_layer[:, j] = pos
            neg_mat_layer[:, j] = neg

        selected_layer = _pick_case_examples(rows, coeffs, pos_mat_layer, neg_mat_layer, args.case_num, args)
        n_good = sum(1 for x in selected_layer if x["neg_edge"] > 0 and x["pos_edge"] > 0)
        mean_score = float(np.mean([x["score"] for x in selected_layer])) if selected_layer else -1e9
        layer_quality = n_good * 1000 + mean_score
        if best is None or layer_quality > best["quality"]:
            best = {
                "layer": layer,
                "coeff_sign": coeff_sign,
                "selected": selected_layer,
                "pos_mat": pos_mat_layer,
                "neg_mat": neg_mat_layer,
                "quality": layer_quality,
            }

    layer = best["layer"]
    coeff_sign = best["coeff_sign"]
    selected = best["selected"]
    pos_mat = best["pos_mat"]
    neg_mat = best["neg_mat"]
    steering_vec = model.set_dtype(candidate_vectors[layer])
    if len(selected) == 0:
        raise RuntimeError("Could not select any case prompts.")

    fig = make_subplots(
        rows=len(selected), cols=1, shared_xaxes=False, vertical_spacing=0.06,
        subplot_titles=[f"Case {i+1}" for i in range(len(selected))],
    )
    mid_idx = int(np.argmin(np.abs(np.array(coeffs))))
    summary = []

    for row_i, picked in enumerate(selected, start=1):
        idx = picked["idx"]
        pos_curve = pos_mat[idx]
        neg_curve = neg_mat[idx]
        balance_idx = picked["balance_idx"]
        balance_coeff = coeffs[balance_idx]
        prompt = prompts[idx]

        fig.add_trace(
            go.Scatter(x=coeffs, y=neg_curve, mode="lines+markers", name="descriptive", line=dict(color="#d95f02"), showlegend=(row_i == 1)),
            row=row_i, col=1,
        )
        fig.add_trace(
            go.Scatter(x=coeffs, y=pos_curve, mode="lines+markers", name="spatial", line=dict(color="#1b9e77"), showlegend=(row_i == 1)),
            row=row_i, col=1,
        )
        fig.add_vline(x=0, line_dash="solid", line_color="black", row=row_i, col=1)
        fig.add_vline(x=balance_coeff, line_dash="dash", line_color="gray", row=row_i, col=1)
        case_max = float(max(pos_curve.max(), neg_curve.max()))
        y_max = min(1.0, max(args.case_zoom_floor, case_max * args.case_zoom_pad))
        fig.update_yaxes(title_text="Class prob", range=[0, y_max], row=row_i, col=1)
        fig.update_xaxes(title_text="Steering coeff (lambda)", row=row_i, col=1)

        short_text = rows[idx]["text"]
        if len(short_text) > 120:
            short_text = short_text[:117] + "..."
        fig.add_annotation(
            xref=f"x{row_i}" if row_i > 1 else "x",
            yref=f"y{row_i}" if row_i > 1 else "y",
            x=coeffs[0], y=1.08, text=short_text,
            showarrow=False, xanchor="left", font=dict(size=12),
        )
        fig.add_annotation(
            xref=f"x{row_i}" if row_i > 1 else "x",
            yref=f"y{row_i}" if row_i > 1 else "y",
            x=coeffs[-1], y=y_max * 0.93,
            text=f"zoomed y-max={y_max:.3f}",
            showarrow=False, xanchor="right", font=dict(size=10, color="#555"),
        )

        token_summary = {}
        probe_coeffs = [coeffs[0], coeffs[mid_idx], balance_coeff, coeffs[-1]]
        probe_names = ["neg_edge", "zero", "balance", "pos_edge"]
        for name, coeff in zip(probe_names, probe_coeffs):
            applied_coeff = coeff_sign * coeff
            intervene = get_intervention_func(steering_vec, method="constant", coeff=applied_coeff)
            token_summary[name] = {"coeff": float(coeff), **_top_token_for_prompt(
                model,
                prompt,
                layer,
                intervene,
                target_token_ids["pos"],
                target_token_ids["neg"],
                constrained_softmax=args.case_constrained_softmax,
            )}

        summary.append({
            "case": row_i,
            "text": rows[idx]["text"],
            "balance_coeff": float(balance_coeff),
            "neg_edge_advantage": picked["neg_edge"],
            "pos_edge_advantage": picked["pos_edge"],
            "near_zero_gap": picked["near_zero_gap"],
            "balance_gap": picked["balance_gap"],
            "token_summary": token_summary,
        })

    fig.update_layout(
        title=f"Vision steering cases ({cfg.model_name}, layer {layer})",
        height=max(340 * len(selected), 900),
        width=980,
        template="plotly_white",
    )

    out_html = Path(args.case_output_html) if args.case_output_html else artifact_dir / "validation/vision_steering_cases.html"
    out_json = Path(args.case_output_json) if args.case_output_json else artifact_dir / "validation/vision_steering_cases.json"
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html))
    save_to_json_file({
        "coeffs": coeffs,
        "selected_layer": layer,
        "searched_layers": layer_candidates,
        "applied_coeff_sign": coeff_sign,
        "case_constrained_softmax": args.case_constrained_softmax,
        "case_auto_orient": args.case_auto_orient,
        "case_strict_only": args.case_strict_only,
        "cases": summary,
    }, out_json)
    logging.info(f"Saved case plot: {out_html}")
    logging.info(f"Saved case summary: {out_json}")


def _resolve_experiment_layer(cfg: Config, args, artifact_dir: Path) -> int:
    if args.exp_layer is not None:
        return args.exp_layer
    if cfg.force_layer is not None:
        return cfg.force_layer
    top_layers_path = artifact_dir / "validation/top_layers.json"
    if top_layers_path.exists():
        try:
            with open(top_layers_path, "r") as f:
                data = json.load(f)
            if len(data) > 0 and "layer" in data[0]:
                return int(data[0]["layer"])
        except Exception:
            pass
    return 5


def _resolve_experiment_coeff(cfg: Config, args, artifact_dir: Path) -> float:
    if args.exp_coeff is not None:
        return float(args.exp_coeff)
    if cfg.debias_coeff is not None:
        return float(cfg.debias_coeff)
    debias_path = artifact_dir / "validation/debiased_results.json"
    if debias_path.exists():
        try:
            with open(debias_path, "r") as f:
                data = json.load(f)
            if len(data) > 0 and "coeff" in data[0]:
                return float(data[0]["coeff"])
        except Exception:
            pass
    return -215.0


def _generate_multi_token_trace(model, prompt: str, layer: int, intervene_func, pos_ids: List[int], neg_ids: List[int], n_tokens: int):
    trace = []
    context = prompt
    for step in range(1, n_tokens + 1):
        logits = model.get_logits([context], layer=layer, intervene_func=intervene_func)
        probs = F.softmax(logits[0, -1, :], dim=-1)
        pos_prob = float(probs[pos_ids].sum().item())
        neg_prob = float(probs[neg_ids].sum().item())
        tok_id = int(torch.argmax(probs).item())
        token = model.tokenizer.decode([tok_id]).replace("\n", "\\n")
        trace.append({
            "step": step,
            "token": token,
            "spatial_prob": pos_prob,
            "descriptive_prob": neg_prob,
            "diff": pos_prob - neg_prob,
        })
        context += model.tokenizer.decode([tok_id])
    return trace, context


def _normalize_logweights(log_weights: List[float]) -> List[float]:
    if len(log_weights) == 0:
        return []
    m = float(max(log_weights))
    shifted = np.exp(np.array(log_weights, dtype=np.float64) - m)
    z = float(shifted.sum())
    if (not np.isfinite(z)) or z <= 0.0:
        return [1.0 / len(log_weights)] * len(log_weights)
    return (shifted / z).tolist()


def _generate_beam_multi_token_trace(
    model,
    prompt: str,
    layer: int,
    intervene_func,
    pos_ids: List[int],
    neg_ids: List[int],
    n_tokens: int,
    beam_width: int,
    beam_top_k: int,
) -> Dict[str, Any]:
    beam_width = max(1, int(beam_width))
    beam_top_k = max(1, int(beam_top_k))
    n_tokens = max(0, int(n_tokens))

    beams = [
        {
            "context": prompt,
            "generated_text": "",
            "sum_logprob": 0.0,
            "cumulative_diff": 0.0,
            "pos_sum": 0.0,
            "neg_sum": 0.0,
            "trace": [],
        }
    ]

    for step in range(1, n_tokens + 1):
        contexts = [b["context"] for b in beams]
        logits_batch = model.get_logits(contexts, layer=layer, intervene_func=intervene_func)
        candidates = []

        for beam_idx, beam in enumerate(beams):
            logits = logits_batch[beam_idx, -1, :]
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            pos_prob = float(probs[pos_ids].sum().item())
            neg_prob = float(probs[neg_ids].sum().item())
            diff = pos_prob - neg_prob

            k = min(beam_top_k, int(log_probs.shape[-1]))
            top_log_probs, top_ids = torch.topk(log_probs, k=k)
            for j in range(k):
                tok_id = int(top_ids[j].item())
                tok_logprob = float(top_log_probs[j].item())
                tok_piece_raw = model.tokenizer.decode([tok_id])
                tok_piece = tok_piece_raw.replace("\n", "\\n")
                candidates.append(
                    {
                        "context": beam["context"] + tok_piece_raw,
                        "generated_text": beam["generated_text"] + tok_piece_raw,
                        "sum_logprob": beam["sum_logprob"] + tok_logprob,
                        "cumulative_diff": beam["cumulative_diff"] + diff,
                        "pos_sum": beam["pos_sum"] + pos_prob,
                        "neg_sum": beam["neg_sum"] + neg_prob,
                        "trace": beam["trace"] + [
                            {
                                "step": step,
                                "token": tok_piece,
                                "spatial_prob": pos_prob,
                                "descriptive_prob": neg_prob,
                                "diff": diff,
                            }
                        ],
                    }
                )

        candidates = sorted(candidates, key=lambda x: x["sum_logprob"], reverse=True)
        beams = candidates[:beam_width]

    if len(beams) == 0:
        return {
            "trace": [],
            "final_text": prompt,
            "cumulative_diff": 0.0,
            "spatial_prob": 0.0,
            "descriptive_prob": 0.0,
            "top_beams": [],
        }

    log_weights = [float(b["sum_logprob"]) for b in beams]
    weights = _normalize_logweights(log_weights)
    best_beam = beams[0]

    expected_trace = []
    if n_tokens > 0:
        for step_idx in range(n_tokens):
            exp_pos = float(sum(weights[i] * beams[i]["trace"][step_idx]["spatial_prob"] for i in range(len(beams))))
            exp_neg = float(sum(weights[i] * beams[i]["trace"][step_idx]["descriptive_prob"] for i in range(len(beams))))
            exp_diff = exp_pos - exp_neg
            token = best_beam["trace"][step_idx]["token"]
            expected_trace.append(
                {
                    "step": step_idx + 1,
                    "token": token,
                    "spatial_prob": exp_pos,
                    "descriptive_prob": exp_neg,
                    "diff": exp_diff,
                }
            )

    expected_pos_sum = float(sum(weights[i] * beams[i]["pos_sum"] for i in range(len(beams))))
    expected_neg_sum = float(sum(weights[i] * beams[i]["neg_sum"] for i in range(len(beams))))
    denom = expected_pos_sum + expected_neg_sum
    if denom <= 0:
        spatial_prob, descriptive_prob = 0.0, 0.0
    else:
        spatial_prob = expected_pos_sum / denom
        descriptive_prob = expected_neg_sum / denom

    top_beams = []
    for rank, (beam, weight) in enumerate(zip(beams, weights), start=1):
        top_beams.append(
            {
                "rank": rank,
                "weight": float(weight),
                "sum_logprob": float(beam["sum_logprob"]),
                "cumulative_diff": float(beam["cumulative_diff"]),
                "generated_text": beam["generated_text"],
                "final_text": beam["context"],
            }
        )

    expected_cumulative_diff = float(sum(weights[i] * beams[i]["cumulative_diff"] for i in range(len(beams))))
    return {
        "trace": expected_trace,
        "final_text": best_beam["context"],
        "cumulative_diff": expected_cumulative_diff,
        "spatial_prob": float(spatial_prob),
        "descriptive_prob": float(descriptive_prob),
        "top_beams": top_beams,
    }


def _completion_scores(model, prompt: str, completions: List[str], layer: int, intervene_func):
    out = []
    for completion in completions:
        token_ids = model.tokenizer.encode(completion, add_special_tokens=False)
        if len(token_ids) == 0:
            out.append({"completion": completion, "avg_logprob": float("-inf"), "sum_logprob": float("-inf"), "n_tokens": 0})
            continue

        context = prompt
        logps = []
        for tok_id in token_ids:
            logits = model.get_logits([context], layer=layer, intervene_func=intervene_func)
            log_probs = F.log_softmax(logits[0, -1, :], dim=-1)
            logps.append(float(log_probs[tok_id].item()))
            context += model.tokenizer.decode([tok_id])

        out.append({
            "completion": completion,
            "avg_logprob": float(np.mean(logps)),
            "sum_logprob": float(np.sum(logps)),
            "n_tokens": int(len(token_ids)),
        })
    out = sorted(out, key=lambda x: x["avg_logprob"], reverse=True)
    return out


def _completion_class_probs(model, prompt: str, spatial_completions: List[str], descriptive_completions: List[str], layer: int, intervene_func):
    spatial_scores = _completion_scores(model, prompt, spatial_completions, layer, intervene_func)
    descriptive_scores = _completion_scores(model, prompt, descriptive_completions, layer, intervene_func)
    spatial_log = float(np.logaddexp.reduce([x["avg_logprob"] for x in spatial_scores]))
    descriptive_log = float(np.logaddexp.reduce([x["avg_logprob"] for x in descriptive_scores]))
    m = max(spatial_log, descriptive_log)
    spatial_exp = float(np.exp(spatial_log - m))
    descriptive_exp = float(np.exp(descriptive_log - m))
    z = spatial_exp + descriptive_exp
    if z <= 0:
        return 0.0, 0.0
    return spatial_exp / z, descriptive_exp / z


def _continuation_class_probs(model, prompt: str, layer: int, intervene_func, pos_ids: List[int], neg_ids: List[int], n_tokens: int):
    context = prompt
    pos_sum = 0.0
    neg_sum = 0.0
    for _ in range(n_tokens):
        logits = model.get_logits([context], layer=layer, intervene_func=intervene_func)
        probs = F.softmax(logits[0, -1, :], dim=-1)
        pos_sum += float(probs[pos_ids].sum().item())
        neg_sum += float(probs[neg_ids].sum().item())
        tok_id = int(torch.argmax(probs).item())
        context += model.tokenizer.decode([tok_id])

    denom = pos_sum + neg_sum
    if denom <= 0:
        return 0.0, 0.0
    return pos_sum / denom, neg_sum / denom


def _greedy_reference_tokens(model, prompt: str, layer: int, n_tokens: int):
    """Generate a fixed continuation path once so coeff sweeps are comparable."""
    ids = []
    context = prompt
    for _ in range(n_tokens):
        logits = model.get_logits([context], layer=layer, intervene_func=None)
        probs = F.softmax(logits[0, -1, :], dim=-1)
        tok_id = int(torch.argmax(probs).item())
        ids.append(tok_id)
        context += model.tokenizer.decode([tok_id])
    return ids


def _continuation_class_probs_teacher_forced(
    model,
    prompt: str,
    layer: int,
    intervene_func,
    pos_ids: List[int],
    neg_ids: List[int],
    reference_token_ids: List[int],
):
    """Score class probs over a fixed continuation path with constrained class softmax."""
    context = prompt
    tracked_ids = pos_ids + neg_ids
    n_pos = len(pos_ids)
    pos_vals = []
    neg_vals = []

    for tok_id in reference_token_ids:
        logits = model.get_logits([context], layer=layer, intervene_func=intervene_func)
        target_logits = logits[0, -1, tracked_ids]
        probs = F.softmax(target_logits, dim=-1)
        pos_vals.append(float(probs[:n_pos].sum().item()))
        neg_vals.append(float(probs[n_pos:].sum().item()))
        context += model.tokenizer.decode([tok_id])

    if len(pos_vals) == 0:
        return 0.0, 0.0
    return float(np.mean(pos_vals)), float(np.mean(neg_vals))


def plot_two_experiment_case_sweeps(cfg: Config, model: ModelBase, args):
    if cfg.data_cfg.target_concept != "vision":
        raise ValueError("Two-experiment case plotting is currently implemented for target_concept=vision only.")

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    artifact_dir = cfg.artifact_path()
    layer = _resolve_experiment_layer(cfg, args, artifact_dir)
    candidate_vectors = torch.load(artifact_dir / "activations/candidate_vectors.pt")
    steering_vec = model.set_dtype(candidate_vectors[layer])

    cases_path = Path(args.exp_cases_file) if args.exp_cases_file else artifact_dir / "validation/vision_steering_cases.json"
    if not cases_path.exists():
        raise FileNotFoundError(f"Case file not found: {cases_path}")
    with open(cases_path, "r") as f:
        case_data = json.load(f)
    continuation_case_texts = [x["text"] for x in case_data.get("cases", [])][: args.exp_case_count]
    if len(continuation_case_texts) == 0:
        raise ValueError(f"No case examples found in {cases_path}")

    coeffs = [float(x) for x in loop_coeffs(args.case_min_coeff, args.case_max_coeff, args.case_increment)]
    if 0.0 not in coeffs:
        coeffs.append(0.0)
        coeffs = sorted(coeffs)

    target_words = _load_target_words(target_concept="vision")
    pos_ids = get_target_token_ids(model.tokenizer, target_words[cfg.data_cfg.pos_label])
    neg_ids = get_target_token_ids(model.tokenizer, target_words[cfg.data_cfg.neg_label])
    target_token_ids = remove_overlapping_target_ids({"pos": pos_ids, "neg": neg_ids})
    pos_ids, neg_ids = target_token_ids["pos"], target_token_ids["neg"]

    fill_spatial = [" to the left of the object.", " to the right of the object.", " near the object."]
    fill_descriptive = [" visually bright in color.", " stylistically clean in look.", " highly textured in detail."]

    # Scenario-specific set for fill-in: pick examples that maximize clean class transitions.
    val_path = artifact_dir / "datasplits/val.json"
    if not val_path.exists():
        raise FileNotFoundError(f"Missing val split at {val_path}")
    with open(val_path, "r") as f:
        val_rows = json.load(f)
    fill_rows = val_rows[: min(args.exp_case_candidate_pool, len(val_rows))]
    fill_prompts_all = [f"Describe this image:\n{r['text']}\nFill in the blank: The main detail is" for r in fill_rows]
    fill_pos_mat = np.zeros((len(fill_rows), len(coeffs)), dtype=np.float64)
    fill_neg_mat = np.zeros((len(fill_rows), len(coeffs)), dtype=np.float64)

    for j, coeff in enumerate(coeffs):
        intervene = get_intervention_func(steering_vec, method="constant", coeff=coeff)
        for i in range(len(fill_rows)):
            p_fill, n_fill = _completion_class_probs(
                model, fill_prompts_all[i], fill_spatial, fill_descriptive, layer, intervene
            )
            fill_pos_mat[i, j] = p_fill
            fill_neg_mat[i, j] = n_fill

    selected_fill = _pick_case_examples(fill_rows, coeffs, fill_pos_mat, fill_neg_mat, args.exp_case_count, args)
    fill_case_texts = [fill_rows[x["idx"]]["text"] for x in selected_fill]
    if len(fill_case_texts) < args.exp_case_count:
        seen = set(fill_case_texts)
        for r in fill_rows:
            t = r["text"]
            if t not in seen:
                fill_case_texts.append(t)
                seen.add(t)
            if len(fill_case_texts) >= args.exp_case_count:
                break
    fill_case_texts = fill_case_texts[: args.exp_case_count]

    fill_prompts = [f"Describe this image:\n{text}\nFill in the blank: The main detail is" for text in fill_case_texts]
    continuation_prompts = model.apply_chat_template(
        [f"Describe this image:\n{text}" for text in continuation_case_texts],
        output_prefix=["The image shows"] * len(continuation_case_texts),
    )
    continuation_reference_ids = [
        _greedy_reference_tokens(model, p, layer, args.exp_tokens)
        for p in continuation_prompts
    ]

    n_cases = min(len(fill_case_texts), len(continuation_case_texts))
    fill_pos = np.zeros((n_cases, len(coeffs)), dtype=np.float64)
    fill_neg = np.zeros((n_cases, len(coeffs)), dtype=np.float64)
    cont_pos = np.zeros((n_cases, len(coeffs)), dtype=np.float64)
    cont_neg = np.zeros((n_cases, len(coeffs)), dtype=np.float64)

    for j, coeff in enumerate(coeffs):
        intervene = get_intervention_func(steering_vec, method="constant", coeff=coeff)
        for i in range(n_cases):
            p_fill, n_fill = _completion_class_probs(
                model, fill_prompts[i], fill_spatial, fill_descriptive, layer, intervene
            )
            fill_pos[i, j] = p_fill
            fill_neg[i, j] = n_fill
            p_cont, n_cont = _continuation_class_probs_teacher_forced(
                model,
                continuation_prompts[i],
                layer,
                intervene,
                pos_ids,
                neg_ids,
                continuation_reference_ids[i],
            )
            cont_pos[i, j] = p_cont
            cont_neg[i, j] = n_cont

    fig = make_subplots(
        rows=n_cases, cols=2, shared_xaxes=False, vertical_spacing=0.06,
        subplot_titles=[x for i in range(n_cases) for x in (f"Case {i+1}: Fill-in", f"Case {i+1}: Next-{args.exp_tokens} tokens")],
    )

    summary_cases = []
    for row_i in range(1, n_cases + 1):
        idx = row_i - 1
        fill_text = fill_case_texts[idx]
        continuation_text = continuation_case_texts[idx]
        fill_gap = np.abs(fill_pos[idx] - fill_neg[idx])
        cont_gap = np.abs(cont_pos[idx] - cont_neg[idx])
        fill_balance_coeff = float(coeffs[int(np.argmin(fill_gap))])
        cont_balance_coeff = float(coeffs[int(np.argmin(cont_gap))])

        fig.add_trace(go.Scatter(x=coeffs, y=fill_neg[idx], mode="lines+markers", name="descriptive", line=dict(color="#d95f02"), showlegend=(row_i == 1)), row=row_i, col=1)
        fig.add_trace(go.Scatter(x=coeffs, y=fill_pos[idx], mode="lines+markers", name="spatial", line=dict(color="#1b9e77"), showlegend=(row_i == 1)), row=row_i, col=1)
        fig.add_trace(go.Scatter(x=coeffs, y=cont_neg[idx], mode="lines+markers", name="descriptive", line=dict(color="#d95f02"), showlegend=False), row=row_i, col=2)
        fig.add_trace(go.Scatter(x=coeffs, y=cont_pos[idx], mode="lines+markers", name="spatial", line=dict(color="#1b9e77"), showlegend=False), row=row_i, col=2)

        fig.add_vline(x=0, line_dash="solid", line_color="black", row=row_i, col=1)
        fig.add_vline(x=0, line_dash="solid", line_color="black", row=row_i, col=2)
        fig.add_vline(x=fill_balance_coeff, line_dash="dash", line_color="gray", row=row_i, col=1)
        fig.add_vline(x=cont_balance_coeff, line_dash="dash", line_color="gray", row=row_i, col=2)

        fig.update_yaxes(title_text="Class prob", range=[0, 1], row=row_i, col=1)
        fig.update_yaxes(title_text="Class prob", range=[0, 1], row=row_i, col=2)
        fig.update_xaxes(title_text="Steering coeff (lambda)", row=row_i, col=1)
        fig.update_xaxes(title_text="Steering coeff (lambda)", row=row_i, col=2)

        short_fill = fill_text if len(fill_text) <= 90 else fill_text[:87] + "..."
        fig.add_annotation(
            xref=f"x{2*row_i-1}" if row_i > 1 else "x",
            yref=f"y{2*row_i-1}" if row_i > 1 else "y",
            x=coeffs[0],
            y=1.08,
            text=short_fill,
            showarrow=False,
            xanchor="left",
            font=dict(size=11),
        )
        short_cont = continuation_text if len(continuation_text) <= 90 else continuation_text[:87] + "..."
        fig.add_annotation(
            xref=f"x{2*row_i}" if row_i > 1 else "x2",
            yref=f"y{2*row_i}" if row_i > 1 else "y2",
            x=coeffs[0],
            y=1.08,
            text=short_cont,
            showarrow=False,
            xanchor="left",
            font=dict(size=11),
        )

        summary_cases.append({
            "case": row_i,
            "fill_text": fill_text,
            "continuation_text": continuation_text,
            "fill_balance_coeff": fill_balance_coeff,
            "continuation_balance_coeff": cont_balance_coeff,
            "fill_spatial_probs": [float(x) for x in fill_pos[idx]],
            "fill_descriptive_probs": [float(x) for x in fill_neg[idx]],
            "continuation_spatial_probs": [float(x) for x in cont_pos[idx]],
            "continuation_descriptive_probs": [float(x) for x in cont_neg[idx]],
        })

    fig.update_layout(
        title=f"Two-experiment coefficient sweeps ({cfg.model_name}, layer {layer})",
        height=max(340 * n_cases, 1000),
        width=1300,
        template="plotly_white",
    )

    out_html = Path(args.exp_plot_output_html) if args.exp_plot_output_html else artifact_dir / "validation/two_experiments_case_sweeps.html"
    out_json = Path(args.exp_plot_output_json) if args.exp_plot_output_json else artifact_dir / "validation/two_experiments_case_sweeps.json"
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html))
    save_to_json_file(
        {
            "model_name": cfg.model_name,
            "layer": int(layer),
            "exp_tokens": int(args.exp_tokens),
            "fill_case_selection": {
                "strategy": "scenario_specific_auto_selected",
                "candidate_pool": int(min(args.exp_case_candidate_pool, len(val_rows))),
            },
            "continuation_case_selection": {
                "strategy": "vision_steering_cases_json",
                "source": str(cases_path),
            },
            "coeffs": coeffs,
            "cases": summary_cases,
        },
        out_json,
    )
    logging.info(f"Saved two-experiment sweep plot: {out_html}")
    logging.info(f"Saved two-experiment sweep summary: {out_json}")


def run_two_experiments(cfg: Config, model: ModelBase, args):
    artifact_dir = cfg.artifact_path()
    if cfg.data_cfg.target_concept != "vision":
        raise ValueError("two-experiments mode is currently implemented for target_concept=vision.")

    layer = _resolve_experiment_layer(cfg, args, artifact_dir)
    coeff = _resolve_experiment_coeff(cfg, args, artifact_dir)

    candidate_vectors = torch.load(artifact_dir / "activations/candidate_vectors.pt")
    steering_vec = model.set_dtype(candidate_vectors[layer])
    baseline_func = get_intervention_func(steering_vec, method="constant", coeff=0.0)
    steered_func = get_intervention_func(steering_vec, method="constant", coeff=coeff)

    target_words = _load_target_words(target_concept="vision")
    pos_ids = get_target_token_ids(model.tokenizer, target_words[cfg.data_cfg.pos_label])
    neg_ids = get_target_token_ids(model.tokenizer, target_words[cfg.data_cfg.neg_label])
    target_token_ids = remove_overlapping_target_ids({"pos": pos_ids, "neg": neg_ids})
    pos_ids, neg_ids = target_token_ids["pos"], target_token_ids["neg"]

    case_texts = None
    if args.exp_use_case_examples:
        cases_path = Path(args.exp_cases_file) if args.exp_cases_file else artifact_dir / "validation/vision_steering_cases.json"
        if not cases_path.exists():
            raise FileNotFoundError(f"Case file not found: {cases_path}")
        with open(cases_path, "r") as f:
            case_data = json.load(f)
        case_texts = [x["text"] for x in case_data.get("cases", [])][: args.exp_case_count]
        if len(case_texts) == 0:
            raise ValueError(f"No examples found in case file: {cases_path}")

    # Experiment 1: fill-in-the-blank (multi-token candidates)
    if case_texts is not None:
        fill_cases = []
        for text in case_texts:
            fill_cases.append(
                {
                    "prompt": f"Describe this image:\n{text}\nFill in the blank: The scene is",
                    "spatial": [" near the building.", " beside the road.", " in front of the object."],
                    "descriptive": [" bright and colorful.", " large and clean.", " dark and textured."],
                }
            )
    else:
        fill_cases = [
            {
                "prompt": "The couch is red and next to the bed. Continue: The bed is",
                "spatial": [" near the window.", " behind the couch.", " next to the table."],
                "descriptive": [" very soft and clean.", " bright blue and large.", " small and comfortable."],
            },
            {
                "prompt": "A bus is parked by the curb. Continue: The curb is",
                "spatial": [" beside the crosswalk.", " in front of the station.", " near the bus stop."],
                "descriptive": [" painted bright yellow.", " wide and smooth.", " old and cracked."],
            },
            {
                "prompt": "A cat is sleeping on the sofa. Continue: The sofa is",
                "spatial": [" next to the window.", " across from the table.", " near the wall."],
                "descriptive": [" dark blue and soft.", " large and comfortable.", " clean and modern."],
            },
        ]

    fill_results = []
    for case in fill_cases:
        row = {"prompt": case["prompt"], "baseline": {}, "steered": {}}
        for name, func in [("baseline", baseline_func), ("steered", steered_func)]:
            spatial_scores = _completion_scores(model, case["prompt"], case["spatial"], layer, func)
            descriptive_scores = _completion_scores(model, case["prompt"], case["descriptive"], layer, func)
            best_spatial = spatial_scores[0]
            best_descriptive = descriptive_scores[0]
            row[name] = {
                "best_spatial": best_spatial,
                "best_descriptive": best_descriptive,
                "margin_avg_logprob": float(best_spatial["avg_logprob"] - best_descriptive["avg_logprob"]),
                "all_spatial": spatial_scores,
                "all_descriptive": descriptive_scores,
            }
        row["delta_margin"] = float(row["steered"]["margin_avg_logprob"] - row["baseline"]["margin_avg_logprob"])
        fill_results.append(row)

    # Experiment 2: current prompt style, but over next N generated tokens
    continuation_texts = case_texts if case_texts is not None else [
        "The couch is red and next to the bed.",
        "A bus is parked beside a curb near a crosswalk.",
        "A black and white cat sleeping on a blue sofa.",
        "A train passing under a bridge next to a road.",
    ]
    continuation_prompts = model.apply_chat_template(
        [f"Describe this image:\n{x}" for x in continuation_texts],
        output_prefix=["The image shows"] * len(continuation_texts),
    )
    continuation_mode = args.exp_continuation_mode
    beam_width = max(1, int(args.exp_beam_width))
    beam_top_k = max(1, int(args.exp_beam_top_k))

    continuation_results = []
    for text, prompt in zip(continuation_texts, continuation_prompts):
        if continuation_mode == "greedy":
            base_trace, base_final = _generate_multi_token_trace(model, prompt, layer, baseline_func, pos_ids, neg_ids, args.exp_tokens)
            steer_trace, steer_final = _generate_multi_token_trace(model, prompt, layer, steered_func, pos_ids, neg_ids, args.exp_tokens)
            base_cum = float(sum(x["diff"] for x in base_trace))
            steer_cum = float(sum(x["diff"] for x in steer_trace))
            continuation_results.append({
                "text": text,
                "prompt": prompt,
                "continuation_mode": "greedy",
                "baseline_trace": base_trace,
                "steered_trace": steer_trace,
                "baseline_cumulative_diff": base_cum,
                "steered_cumulative_diff": steer_cum,
                "delta_cumulative_diff": float(steer_cum - base_cum),
                "baseline_final_text": base_final,
                "steered_final_text": steer_final,
            })
        elif continuation_mode == "beam":
            base_beam = _generate_beam_multi_token_trace(
                model, prompt, layer, baseline_func, pos_ids, neg_ids, args.exp_tokens, beam_width, beam_top_k
            )
            steer_beam = _generate_beam_multi_token_trace(
                model, prompt, layer, steered_func, pos_ids, neg_ids, args.exp_tokens, beam_width, beam_top_k
            )
            continuation_results.append({
                "text": text,
                "prompt": prompt,
                "continuation_mode": "beam",
                "beam_width": int(beam_width),
                "beam_top_k": int(beam_top_k),
                "baseline_trace": base_beam["trace"],
                "steered_trace": steer_beam["trace"],
                "baseline_cumulative_diff": float(base_beam["cumulative_diff"]),
                "steered_cumulative_diff": float(steer_beam["cumulative_diff"]),
                "delta_cumulative_diff": float(steer_beam["cumulative_diff"] - base_beam["cumulative_diff"]),
                "baseline_final_text": base_beam["final_text"],
                "steered_final_text": steer_beam["final_text"],
                "baseline_top_beams": base_beam["top_beams"],
                "steered_top_beams": steer_beam["top_beams"],
            })
        else:
            base_trace, base_final = _generate_multi_token_trace(model, prompt, layer, baseline_func, pos_ids, neg_ids, args.exp_tokens)
            steer_trace, steer_final = _generate_multi_token_trace(model, prompt, layer, steered_func, pos_ids, neg_ids, args.exp_tokens)
            base_cum = float(sum(x["diff"] for x in base_trace))
            steer_cum = float(sum(x["diff"] for x in steer_trace))
            greedy_delta = float(steer_cum - base_cum)

            base_beam = _generate_beam_multi_token_trace(
                model, prompt, layer, baseline_func, pos_ids, neg_ids, args.exp_tokens, beam_width, beam_top_k
            )
            steer_beam = _generate_beam_multi_token_trace(
                model, prompt, layer, steered_func, pos_ids, neg_ids, args.exp_tokens, beam_width, beam_top_k
            )
            beam_delta = float(steer_beam["cumulative_diff"] - base_beam["cumulative_diff"])

            continuation_results.append(
                {
                    "text": text,
                    "prompt": prompt,
                    "continuation_mode": "compare",
                    "greedy": {
                        "baseline_trace": base_trace,
                        "steered_trace": steer_trace,
                        "baseline_cumulative_diff": base_cum,
                        "steered_cumulative_diff": steer_cum,
                        "delta_cumulative_diff": greedy_delta,
                        "baseline_final_text": base_final,
                        "steered_final_text": steer_final,
                    },
                    "beam": {
                        "beam_width": int(beam_width),
                        "beam_top_k": int(beam_top_k),
                        "baseline_trace": base_beam["trace"],
                        "steered_trace": steer_beam["trace"],
                        "baseline_cumulative_diff": float(base_beam["cumulative_diff"]),
                        "steered_cumulative_diff": float(steer_beam["cumulative_diff"]),
                        "delta_cumulative_diff": beam_delta,
                        "baseline_final_text": base_beam["final_text"],
                        "steered_final_text": steer_beam["final_text"],
                        "baseline_top_beams": base_beam["top_beams"],
                        "steered_top_beams": steer_beam["top_beams"],
                    },
                    "delta_improvement_beam_over_greedy": float(beam_delta - greedy_delta),
                }
            )

    summary = {
        "model_name": cfg.model_name,
        "target_concept": cfg.data_cfg.target_concept,
        "layer": int(layer),
        "steering_coeff": float(coeff),
        "exp_tokens": int(args.exp_tokens),
        "exp_continuation_mode": continuation_mode,
        "exp_beam_width": int(beam_width),
        "exp_beam_top_k": int(beam_top_k),
        "exp_use_case_examples": bool(args.exp_use_case_examples),
        "fill_in_blank": fill_results,
        "multi_token_continuation": continuation_results,
    }

    out_json = Path(args.exp_output_json) if args.exp_output_json else artifact_dir / "validation/two_experiments.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    save_to_json_file(summary, out_json)
    logging.info(f"Saved two-experiments results: {out_json}")

    fill_delta_mean = float(np.mean([x["delta_margin"] for x in fill_results]))
    print(f"Two-experiments complete | layer={layer} coeff={coeff:+.1f}")
    print(f"Avg fill-in margin delta (steered-baseline): {fill_delta_mean:+.4f}")
    if continuation_mode == "compare":
        greedy_mean = float(np.mean([x["greedy"]["delta_cumulative_diff"] for x in continuation_results]))
        beam_mean = float(np.mean([x["beam"]["delta_cumulative_diff"] for x in continuation_results]))
        improvement_mean = float(np.mean([x["delta_improvement_beam_over_greedy"] for x in continuation_results]))
        print(f"Avg {args.exp_tokens}-token cumulative diff delta (greedy): {greedy_mean:+.4f}")
        print(f"Avg {args.exp_tokens}-token cumulative diff delta (beam): {beam_mean:+.4f}")
        print(f"Beam improvement vs greedy delta: {improvement_mean:+.4f} (beam_width={beam_width}, top_k={beam_top_k})")
    else:
        cont_delta_mean = float(np.mean([x["delta_cumulative_diff"] for x in continuation_results]))
        mode_label = "beam" if continuation_mode == "beam" else "greedy"
        print(f"Avg {args.exp_tokens}-token cumulative diff delta ({mode_label}): {cont_delta_mean:+.4f}")


def run_handcrafted_eval(cfg: Config, model: ModelBase, args):
    """
    Run validation on the hand-crafted eval set using pre-computed steering vectors.
    Results are saved to <artifact_path>/handcrafted_eval/ to avoid overwriting
    any existing benchmark validation results.

    Requires that training has already been run (candidate_vectors.pt must exist).
    """
    from .data.load_dataset import load_handcrafted_eval

    artifact_dir = cfg.artifact_path()
    vectors_path = artifact_dir / "activations" / "candidate_vectors.pt"
    if not vectors_path.exists():
        raise FileNotFoundError(
            f"Steering vectors not found at {vectors_path}. "
            "Run training first (without --eval_set handcrafted) to compute them."
        )

    logging.info("Loading handcrafted eval set")
    hc_data = load_handcrafted_eval(filepath=args.handcrafted_eval_file)
    logging.info(f"Handcrafted eval: {len(hc_data)} examples")

    target_words_by_label = _load_target_words(target_concept=cfg.data_cfg.target_concept)
    target_token_ids = {}
    for label, k in zip([cfg.data_cfg.pos_label, cfg.data_cfg.neg_label], ["pos", "neg"]):
        target_token_ids[k] = get_target_token_ids(model.tokenizer, target_words_by_label[label])
    target_token_ids = remove_overlapping_target_ids(target_token_ids)

    validate(cfg, model, hc_data, target_token_ids, save_subdir="handcrafted_eval")

    out_dir = artifact_dir / "handcrafted_eval"
    print(f"Handcrafted eval complete. Results saved to: {out_dir}")
    if (out_dir / "debiased_results.json").exists():
        import json as _json
        results = _json.load(open(out_dir / "debiased_results.json"))
        if results:
            best = results[0]
            print(f"Best layer: {best['layer']}  RMS bias: {best['rms']:.4f}  coeff: {best.get('coeff', 0)}")


def run_generation_log(cfg: Config, model: ModelBase, args):
    from .eval.generation_logger import log_generations_sweep, build_output_path

    artifact_dir = cfg.artifact_path()

    if args.log_gen_layer is not None:
        layer = args.log_gen_layer
    else:
        layer = _resolve_experiment_layer(cfg, args, artifact_dir)

    candidate_vectors = torch.load(artifact_dir / "activations/candidate_vectors.pt")
    steering_vec = model.set_dtype(candidate_vectors[layer])

    offset = 0
    if cfg.use_offset:
        neutral_acts = torch.load(artifact_dir / "activations/neutral.pt")
        offset = model.set_dtype(neutral_acts.mean(dim=1)[layer])

    target_words = _load_target_words(target_concept=cfg.data_cfg.target_concept)
    pos_ids = get_target_token_ids(model.tokenizer, target_words[cfg.data_cfg.pos_label])
    neg_ids = get_target_token_ids(model.tokenizer, target_words[cfg.data_cfg.neg_label])
    target_token_ids = remove_overlapping_target_ids({"pos": pos_ids, "neg": neg_ids})
    pos_ids, neg_ids = target_token_ids["pos"], target_token_ids["neg"]

    coeff_min = args.log_gen_coeff_min if args.log_gen_coeff_min is not None else cfg.coeff_search_min
    coeff_max = args.log_gen_coeff_max if args.log_gen_coeff_max is not None else cfg.coeff_search_max
    coeff_increment = args.log_gen_coeff_increment if args.log_gen_coeff_increment is not None else cfg.coeff_search_increment
    coeffs = [float(x) for x in loop_coeffs(coeff_min, coeff_max, coeff_increment)]

    prompt_type = args.log_gen_prompt_type

    examples = []
    if args.log_gen_use_cases:
        cases_path = (
            Path(args.log_gen_cases_file) if args.log_gen_cases_file
            else artifact_dir / "validation/vision_steering_cases.json"
        )
        if not cases_path.exists():
            raise FileNotFoundError(f"Case file not found: {cases_path}")
        with open(cases_path, "r") as f:
            case_data = json.load(f)
        for i, c in enumerate(case_data.get("cases", [])):
            caption = c["text"]
            raw_prompt = f"Describe this image:\n{caption}"
            if cfg.data_cfg.output_prefix:
                prompt = model.apply_chat_template([raw_prompt], output_prefix=["The image shows"])[0]
            else:
                prompt = model.apply_chat_template([raw_prompt])[0]
            examples.append({
                "example_id": i,
                "caption": caption,
                "prompt_template": prompt_type,
                "prompt": prompt,
            })
    else:
        val_path = artifact_dir / "datasplits/val.json"
        if not val_path.exists():
            raise FileNotFoundError(f"Missing val split at {val_path}. Run training/validation first.")
        with open(val_path, "r") as f:
            val_rows = json.load(f)
        n = args.log_gen_n_examples if args.log_gen_n_examples is not None else len(val_rows)
        val_rows = val_rows[:n]
        for i, row in enumerate(val_rows):
            caption = row.get("text", row.get("prompt", ""))
            raw_prompt = f"Describe this image:\n{caption}"
            if cfg.data_cfg.output_prefix:
                prompt = model.apply_chat_template([raw_prompt], output_prefix=["The image shows"])[0]
            else:
                prompt = model.apply_chat_template([raw_prompt])[0]
            examples.append({
                "example_id": i,
                "caption": caption,
                "prompt_template": prompt_type,
                "prompt": prompt,
            })

    output_dir = Path(args.log_gen_output_dir) if args.log_gen_output_dir else Path("results/generation_logs")
    output_path = build_output_path(output_dir, prompt_type, args.log_gen_decoding)

    log_generations_sweep(
        model=model,
        examples=examples,
        layer=layer,
        steering_vec=steering_vec,
        coeffs=coeffs,
        pos_ids=pos_ids,
        neg_ids=neg_ids,
        output_path=output_path,
        decoding=args.log_gen_decoding,
        max_new_tokens=args.log_gen_max_tokens,
        beam_width=args.log_gen_beam_width,
        beam_top_k=args.log_gen_beam_top_k,
        intervention_method=args.log_gen_intervention_method,
        constrained_softmax=cfg.constrained_softmax,
        offset=offset,
    )
    print(f"Generation log saved: layer={layer}, {len(examples)} examples × {len(coeffs)} coeffs → {output_path}")
    print(f"  method={args.log_gen_intervention_method}  constrained_softmax={cfg.constrained_softmax}")


def main():
    args = parse_arguments()
    
    if args.config_file is not None:
        cfg = Config.load(args.config_file)
        logging.info(f"Loaded config file: {args.config_file}")
        # Allow CLI override for constrained softmax even when loading config.
        if args.constrained_softmax:
            cfg.constrained_softmax = True
        if args.use_cache:
            cfg.use_cache = True
        if args.intervention_method is not None:
            cfg.intervention_method = args.intervention_method
        if args.optimize_coeff:
            cfg.optimize_coeff = True
        if args.score_mode is not None:
            cfg.score_mode = args.score_mode
        if args.prompt_template_sweep:
            cfg.prompt_template_sweep = True
        if args.prompt_template_max_examples is not None:
            cfg.prompt_template_max_examples = args.prompt_template_max_examples
        if args.force_layer is not None:
            cfg.force_layer = args.force_layer
        if args.debias_coeff is not None:
            cfg.debias_coeff = args.debias_coeff
        if args.coeff_search_min is not None:
            cfg.coeff_search_min = args.coeff_search_min
        if args.coeff_search_max is not None:
            cfg.coeff_search_max = args.coeff_search_max
        if args.coeff_search_increment is not None:
            cfg.coeff_search_increment = args.coeff_search_increment
    else:
        data_cfg = DataConfig(
            target_concept=args.target_concept,
            pos_label=args.pos_label,
            neg_label=args.neg_label,
            n_train=args.n_train_per_label, n_val=args.n_val,
            bias_threshold=args.bias_threshold, 
        )
        cfg = Config(
            model_name=args.model_name, data_cfg=data_cfg, 
            method=args.method, use_offset=args.use_offset, seed=args.seed,
            evaluate_top_n_layer=args.evaluate_top_n_layer, 
            force_layer=args.force_layer,
            filter_layer_pct=args.filter_layer_pct, save_dir=args.save_dir,
            batch_size=args.batch_size, use_cache=args.use_cache,
            constrained_softmax=args.constrained_softmax,
            score_mode=args.score_mode or "prob_diff",  # REALIGNED: default to prob_diff to match reference; was hardcoded "adaptive"
            prompt_template_sweep=args.prompt_template_sweep,
            prompt_template_max_examples=args.prompt_template_max_examples if args.prompt_template_max_examples is not None else 400,
            intervention_method=args.intervention_method or "default",
            optimize_coeff=args.optimize_coeff,
            debias_coeff=args.debias_coeff,
            coeff_search_min=args.coeff_search_min if args.coeff_search_min is not None else -30.0,
            coeff_search_max=args.coeff_search_max if args.coeff_search_max is not None else 30.0,
            coeff_search_increment=args.coeff_search_increment if args.coeff_search_increment is not None else 5.0,
        )
        cfg.save()

    print("Model:", cfg.model_name)
    print("Configuration:\n", repr(cfg))

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    model = load_model(cfg.model_name)

    if args.eval_set == "handcrafted":
        run_handcrafted_eval(cfg, model, args)
    elif args.log_generations:
        run_generation_log(cfg, model, args)
    elif args.run_two_experiments:
        run_two_experiments(cfg, model, args)
    elif args.plot_two_experiments_cases:
        plot_two_experiment_case_sweeps(cfg, model, args)
    elif args.generate_vision_cases:
        generate_vision_cases(cfg, model, args)
    elif args.run_eval:
        eval(cfg, model, layer=args.layer, coeff=args.coeff, batch_size=args.batch_size)
    else:
        train_and_validate(cfg, model)


if __name__ == "__main__":
    main()

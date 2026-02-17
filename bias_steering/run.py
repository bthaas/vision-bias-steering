import os, json
import random
import argparse
import warnings
import logging
from pathlib import Path
from typing import Tuple, List, Dict

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from .config import Config, DataConfig
from .utils import save_to_json_file, loop_coeffs
from .data.load_dataset import load_datasplits, load_target_words
from .data.prompt_iterator import PromptIterator
from .steering import load_model, ModelBase, extract_candidate_vectors, \
    validate, get_intervention_func, get_target_token_ids, compute_projections
from .eval import load_evaluation_task

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.set_grad_enabled(False);
logging.basicConfig(level=logging.INFO)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default=None, help='Load configuration from file.')
    parser.add_argument('--model_name', type=str, help='Model name')
    parser.add_argument('--method', type=str, default="WMD", choices=["WMD", "MD"], help='Method for computing candidate vectors.')
    parser.add_argument('--use_offset', action='store_true', help="Offset by neutral examples.")
    parser.add_argument('--n_train_per_label', type=int, default=800, help="Number of training examples per label.")
    parser.add_argument('--n_val', type=int, default=1600, help="Number of validation examples.")
    parser.add_argument('--bias_threshold', type=float, default=0.1)
    parser.add_argument('--target_concept', type=str, default="gender", help='Target concept (gender, race, vision, etc.)')
    parser.add_argument('--pos_label', type=str, default="F", help='Positive label (e.g., F, spatial)')
    parser.add_argument('--neg_label', type=str, default="M", help='Negative label (e.g., M, descriptive)')
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


def pick_adaptive_score_mode(train_df: pd.DataFrame, pos_label: str, neg_label: str, target_concept: str) -> str:
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
    df2 = df.copy()
    df2["bin"] = pd.cut(df2["bias"].abs(), n_bins)
    bin_freq = df2.groupby("bin", observed=True).size().to_dict()
    df2["sample_weight"] = df2["bin"].apply(lambda x: 1 / bin_freq[x]**2)
    temp = df2.sample(sample_size, weights="sample_weight")
    return temp


def train_and_validate(cfg: Config, model: ModelBase):
    datasplits_dir = cfg.artifact_path() / "datasplits"
    data_cfg = cfg.data_cfg
    datasets = load_datasplits(data_cfg, datasplits_dir, use_cache=cfg.use_cache)
    os.makedirs(datasplits_dir, exist_ok=True)

    logging.info("Preprocessing train/val data")
    target_words_by_label = load_target_words(target_concept=data_cfg.target_concept)
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
            score_mode=args.score_mode or "adaptive",
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

    if args.run_eval:
        eval(cfg, model, layer=args.layer, coeff=args.coeff, batch_size=args.batch_size)    
    else:
        train_and_validate(cfg, model)


if __name__ == "__main__":
    main()

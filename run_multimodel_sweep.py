#!/usr/bin/env python3
"""
Caption-level token-probability sweeps for multiple models.

This mirrors `run_local_sweep.py` but makes the model and artifact directory
configurable so the same prompt setup can be rendered for Rivanna base-model runs.

Supported artifact layouts:
- Legacy local runs: `<artifact_dir>/activations/candidate_vectors.pt` and
  `<artifact_dir>/validation/top_layers.json`
- Rivanna runs: `<artifact_dir>/artifacts/candidate_vectors.pt` and
  `<artifact_dir>/results.json`
"""

from __future__ import annotations

import argparse
import gc
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = ROOT / "experiments" / "rivanna" / "results"
DEFAULT_OUTPUT_ROOT = ROOT / "plots" / "multimodel_sweeps"

KNOWN_TARGETS: dict[str, tuple[str, Path | None]] = {
    "qwen18b_chat": ("Qwen/Qwen-1_8B-chat", ROOT / "runs_vision" / "Qwen-1_8B-chat"),
    "qwen18b_base": ("Qwen/Qwen-1_8B", None),
    "qwen25_3b_base": ("Qwen/Qwen2.5-3B", None),
    "qwen25_7b_base": ("Qwen/Qwen2.5-7B", None),
}
DEFAULT_MODEL_SLUGS: tuple[str, ...] = (
    "qwen25_3b_base",
    "qwen25_7b_base",
)

MULTI_TEMPLATE = "Describe this image:\n{text}"
MULTI_PREFIX = "The image shows"
FILL_IN_PREFIX = "Positioned"

COLOR_SPATIAL = "#1b9e77"
COLOR_DESCRIPTIVE = "#d95f02"


@dataclass(frozen=True)
class SweepTarget:
    slug: str
    model_name: str
    artifact_dir: Path
    layer: int | None = None


def build_lambda_values(lambda_min: int, lambda_max: int, lambda_step: int) -> list[int]:
    if lambda_step <= 0:
        raise ValueError("lambda_step must be positive")
    if lambda_min > lambda_max:
        raise ValueError("lambda_min must be <= lambda_max")
    span = lambda_max - lambda_min
    if span % lambda_step != 0:
        raise ValueError("lambda range must be divisible by lambda_step")
    return list(range(lambda_min, lambda_max + lambda_step, lambda_step))


def parse_target_spec(spec: str) -> SweepTarget:
    parts = [part.strip() for part in spec.split("::")]
    if len(parts) not in (3, 4) or any(not part for part in parts[:3]):
        raise ValueError(
            "target must be 'slug::model_name::artifact_dir' "
            "or 'slug::model_name::artifact_dir::layer', "
            f"got {spec!r}"
        )
    slug, model_name, artifact_dir = parts[:3]
    layer = int(parts[3]) if len(parts) == 4 else None
    return SweepTarget(
        slug=slug,
        model_name=model_name,
        artifact_dir=Path(artifact_dir).expanduser(),
        layer=layer,
    )


def default_base_targets(results_root: Path) -> list[SweepTarget]:
    return build_targets_from_model_slugs(results_root, DEFAULT_MODEL_SLUGS)


def build_targets_from_model_slugs(results_root: Path, model_slugs: Sequence[str]) -> list[SweepTarget]:
    unknown = [slug for slug in model_slugs if slug not in KNOWN_TARGETS]
    if unknown:
        choices = ", ".join(sorted(KNOWN_TARGETS))
        missing = ", ".join(unknown)
        raise ValueError(f"Unknown model slug(s): {missing}. Known choices: {choices}")

    targets: list[SweepTarget] = []
    for slug in model_slugs:
        model_name, explicit_artifact_dir = KNOWN_TARGETS[slug]
        artifact_dir = explicit_artifact_dir if explicit_artifact_dir is not None else results_root / slug
        targets.append(SweepTarget(slug=slug, model_name=model_name, artifact_dir=artifact_dir))
    return targets


def select_targets(
    target_specs: Sequence[str],
    model_slugs: Sequence[str] | None,
    results_root: Path,
) -> list[SweepTarget]:
    if target_specs:
        return [parse_target_spec(spec) for spec in target_specs]
    if model_slugs:
        return build_targets_from_model_slugs(results_root, model_slugs)
    return default_base_targets(results_root)


def resolve_candidate_vector_path(artifact_dir: Path) -> Path:
    candidates = (
        artifact_dir / "activations" / "candidate_vectors.pt",
        artifact_dir / "artifacts" / "candidate_vectors.pt",
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find candidate_vectors.pt under {artifact_dir}. "
        "Expected either activations/ or artifacts/."
    )


def infer_best_layer(artifact_dir: Path, explicit_layer: int | None = None) -> int:
    if explicit_layer is not None:
        return int(explicit_layer)

    results_path = artifact_dir / "results.json"
    if results_path.exists():
        data = json.loads(results_path.read_text(encoding="utf-8"))
        if data.get("best_layer") is not None:
            return int(data["best_layer"])

    top_layers_path = artifact_dir / "validation" / "top_layers.json"
    if top_layers_path.exists():
        data = json.loads(top_layers_path.read_text(encoding="utf-8"))
        if data:
            return int(data[0]["layer"])

    raise FileNotFoundError(
        f"Could not infer a best layer for {artifact_dir}. "
        "Expected results.json or validation/top_layers.json."
    )


def _make_output_dir(output_root: Path, slug: str) -> Path:
    out_dir = output_root / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _build_prompts(model, output_prefix: str, captions: Sequence[dict[str, str]]) -> list[str]:
    instructions = [MULTI_TEMPLATE.format(text=caption["text"]) for caption in captions]
    return model.apply_chat_template(instructions, output_prefix=output_prefix)


def _remove_token_overlap(pos_ids: list[int], neg_ids: list[int]) -> tuple[list[int], list[int]]:
    overlap = set(pos_ids) & set(neg_ids)
    return (
        [token_id for token_id in pos_ids if token_id not in overlap],
        [token_id for token_id in neg_ids if token_id not in overlap],
    )


def _caption_title(caption: dict[str, str], max_chars: int = 90) -> str:
    label = caption["label"].upper()
    short = textwrap.shorten(caption["text"], width=max_chars, placeholder="…")
    return f"[{label}]  {short}"


def _draw_subplot(
    ax,
    lambdas: Sequence[int],
    pos_curve,
    neg_curve,
    *,
    xlabel: bool = False,
    ylabel: bool = False,
    title: str | None = None,
    legend: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    ax.plot(
        lambdas,
        pos_curve,
        color=COLOR_SPATIAL,
        linewidth=2,
        marker="o",
        markersize=4,
        label="spatial",
    )
    ax.plot(
        lambdas,
        neg_curve,
        color=COLOR_DESCRIPTIVE,
        linewidth=2,
        marker="o",
        markersize=4,
        label="descriptive",
    )
    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_ylim(0, 1)
    ax.set_xlim(min(lambdas) * 1.05, max(lambdas) * 1.05)
    ax.set_xticks(lambdas)
    ax.tick_params(axis="x", labelsize=6, rotation=45)
    ax.tick_params(axis="y", labelsize=7)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    if xlabel:
        ax.set_xlabel("lambda (+ => spatial)", fontsize=8)
    if ylabel:
        ax.set_ylabel("Tracked prob", fontsize=8)
    if title:
        ax.set_title(title, fontsize=9, pad=4)
    if legend:
        ax.legend(fontsize=8, loc="upper right", framealpha=0.85)


def _plot_caption_row(axes_row, caption_idx: int, results: dict, lambdas: Sequence[int], *, is_first_row: bool, is_last_row: bool) -> None:
    column_configs = [
        ("multi_greedy", "8-token greedy"),
        ("multi_beam", "8-token beam"),
        ("fill_greedy", "fill-in greedy"),
        ("fill_beam", "fill-in beam"),
    ]
    for column_index, (key, column_label) in enumerate(column_configs):
        ax = axes_row[column_index]
        pos_curve = results[key]["pos"][caption_idx]
        neg_curve = results[key]["neg"][caption_idx]
        _draw_subplot(
            ax,
            lambdas,
            pos_curve,
            neg_curve,
            xlabel=is_last_row,
            ylabel=(column_index == 0),
            title=column_label if is_first_row else None,
            legend=(is_first_row and column_index == 0),
        )


def plot_all_captions(
    *,
    captions: Sequence[dict[str, str]],
    results: dict,
    lambdas: Sequence[int],
    layer: int,
    model_name: str,
    n_tokens: int,
    constrained: bool,
    outpath: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows = len(captions)
    row_height = 3.2
    fig, axes = plt.subplots(
        n_rows,
        4,
        figsize=(20, row_height * n_rows),
        squeeze=False,
        gridspec_kw={"hspace": 0.55, "wspace": 0.22},
    )

    fig.suptitle(
        f"{model_name} — Spatial vs Descriptive Token Probability\n"
        f"Layer {layer}  |  method=constant  |  {n_tokens}-token curves  |  constrained={constrained}  |  +lambda => spatial",
        fontsize=12,
        y=1.002,
    )

    for index, caption in enumerate(captions):
        _plot_caption_row(
            axes[index],
            index,
            results,
            lambdas,
            is_first_row=(index == 0),
            is_last_row=(index == len(captions) - 1),
        )
        axes[index, 0].annotate(
            _caption_title(caption, max_chars=95),
            xy=(0, 1.06),
            xycoords="axes fraction",
            fontsize=6.5,
            color="#222222",
            ha="left",
            va="bottom",
            clip_on=False,
        )

    fig.savefig(str(outpath), dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_single_caption(
    *,
    caption_idx: int,
    captions: Sequence[dict[str, str]],
    results: dict,
    lambdas: Sequence[int],
    outpath: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    caption = captions[caption_idx]
    fig, axes = plt.subplots(1, 4, figsize=(20, 3.5), gridspec_kw={"wspace": 0.22})
    fig.suptitle(_caption_title(caption, max_chars=110), fontsize=10, y=1.04)

    for column_index, (key, column_label) in enumerate(
        [
            ("multi_greedy", "8-token greedy"),
            ("multi_beam", "8-token beam"),
            ("fill_greedy", "fill-in greedy"),
            ("fill_beam", "fill-in beam"),
        ]
    ):
        pos_curve = results[key]["pos"][caption_idx]
        neg_curve = results[key]["neg"][caption_idx]
        _draw_subplot(
            axes[column_index],
            lambdas,
            pos_curve,
            neg_curve,
            xlabel=True,
            ylabel=(column_index == 0),
            title=column_label,
            legend=(column_index == 0),
        )

    fig.savefig(str(outpath), dpi=120, bbox_inches="tight")
    plt.close(fig)


def _log_generations(
    *,
    model,
    fill_prompts: Sequence[str],
    steering_vec,
    captions: Sequence[dict[str, str]],
    lambdas: Sequence[int],
    layer: int,
    max_gen_tokens: int,
    outpath: Path,
) -> None:
    from bias_steering.steering import get_intervention_func

    lines = [
        f"{model.model.config._name_or_path} — Steered generations (layer {layer}, fill-in / B_positioned template)",
        f"Method: constant  |  {max_gen_tokens} new tokens  |  greedy",
        "",
    ]
    for coeff in lambdas:
        intervene = get_intervention_func(steering_vec, method="constant", coeff=coeff)
        texts = model.generate(
            fill_prompts,
            layer=layer,
            intervene_func=intervene,
            max_new_tokens=max_gen_tokens,
        )
        lines += [f"\n{'-' * 68}", f"lambda = {coeff}", "-" * 68]
        for caption, generation in zip(captions, texts):
            label = caption["label"].upper()
            lines.append(f"\n  [{label}] {caption['text'][:70]}...")
            lines.append(f"  -> Positioned {generation.strip()}")
    outpath.write_text("\n".join(lines), encoding="utf-8")


def _write_manifest(
    *,
    out_dir: Path,
    target: SweepTarget,
    layer: int,
    lambdas: Sequence[int],
    n_tokens: int,
    constrained: bool,
    captions: Sequence[dict[str, str]],
    include_beam: bool,
    vector_flipped: bool,
    calibration_probe_lambda: int | None,
) -> None:
    manifest = {
        "slug": target.slug,
        "model_name": target.model_name,
        "artifact_dir": str(target.artifact_dir),
        "layer": int(layer),
        "lambdas": list(lambdas),
        "n_tokens": int(n_tokens),
        "constrained": bool(constrained),
        "include_beam": bool(include_beam),
        "vector_flipped_for_spatial_positive": bool(vector_flipped),
        "calibration_probe_lambda": calibration_probe_lambda,
        "lambda_direction": "positive => spatial",
        "captions": [
            {"index": index, "label": caption["label"], "text": caption["text"]}
            for index, caption in enumerate(captions)
        ],
    }
    (out_dir / "sweep_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def run_target_sweep(
    *,
    target: SweepTarget,
    captions: Sequence[dict[str, str]],
    lambdas: Sequence[int],
    n_tokens: int,
    batch_size: int,
    beam_width: int,
    beam_top_k: int,
    constrained: bool,
    max_gen_tokens: int,
    output_root: Path,
    include_beam: bool,
) -> None:
    import torch

    from bias_steering.steering import load_model
    from bias_steering.steering.steering_utils import get_target_token_ids
    from plotting.beam_selected_prompt_report import (
        continuation_multi_token_curve_beam_batched,
        teacher_forced_multi_token_curve_beam_batched,
    )
    from plotting.master_prompt_experiments import (
        continuation_multi_token_curve_greedy,
        load_target_words,
        teacher_forced_multi_token_curve,
    )

    out_dir = _make_output_dir(output_root, target.slug)
    vector_path = resolve_candidate_vector_path(target.artifact_dir)
    layer = infer_best_layer(target.artifact_dir, explicit_layer=target.layer)

    print(f"Loading {target.model_name} from {target.artifact_dir} (layer {layer})...")
    model = load_model(target.model_name)

    candidate_vectors = torch.load(vector_path, map_location="cpu")
    steering_vec = model.set_dtype(candidate_vectors[layer])

    words = load_target_words("vision")
    pos_ids, neg_ids = _remove_token_overlap(
        get_target_token_ids(model.tokenizer, words["spatial"]),
        get_target_token_ids(model.tokenizer, words["descriptive"]),
    )

    multi_prompts = _build_prompts(model, MULTI_PREFIX, captions)
    fill_prompts = _build_prompts(model, FILL_IN_PREFIX, captions)

    probe_lambda = next((abs(coeff) for coeff in lambdas if coeff != 0), None)
    vector_flipped = False
    if probe_lambda is not None:
        probe_pos, probe_neg = continuation_multi_token_curve_greedy(
            model=model,
            prompts=fill_prompts,
            layer=layer,
            steering_vec=steering_vec,
            coeffs=[-probe_lambda, probe_lambda],
            pos_ids=pos_ids,
            neg_ids=neg_ids,
            n_tokens=1,
            constrained=constrained,
            batch_size=batch_size,
        )
        neg_probe_bias = float(np.mean(probe_pos[:, 0] - probe_neg[:, 0]))
        pos_probe_bias = float(np.mean(probe_pos[:, 1] - probe_neg[:, 1]))
        if pos_probe_bias < neg_probe_bias:
            steering_vec = -steering_vec
            vector_flipped = True
        print(
            f"Calibrated sign for {target.slug}: probe=±{probe_lambda}  "
            f"bias(-probe)={neg_probe_bias:.4f}  bias(+probe)={pos_probe_bias:.4f}  "
            f"flipped={vector_flipped}"
        )

    common = dict(
        model=model,
        layer=layer,
        steering_vec=steering_vec,
        coeffs=lambdas,
        pos_ids=pos_ids,
        neg_ids=neg_ids,
        n_tokens=n_tokens,
        constrained=constrained,
    )

    multi_greedy_pos, multi_greedy_neg = teacher_forced_multi_token_curve(
        prompts=multi_prompts,
        batch_size=batch_size,
        **common,
    )
    fill_greedy_pos, fill_greedy_neg = continuation_multi_token_curve_greedy(
        prompts=fill_prompts,
        batch_size=batch_size,
        **common,
    )

    if include_beam:
        multi_beam_pos, multi_beam_neg = teacher_forced_multi_token_curve_beam_batched(
            prompts=multi_prompts,
            beam_width=beam_width,
            beam_top_k=beam_top_k,
            batch_size=batch_size,
            **common,
        )
        fill_beam_pos, fill_beam_neg = continuation_multi_token_curve_beam_batched(
            prompts=fill_prompts,
            beam_width=beam_width,
            beam_top_k=beam_top_k,
            batch_size=batch_size,
            **common,
        )
    else:
        multi_beam_pos, multi_beam_neg = multi_greedy_pos.copy(), multi_greedy_neg.copy()
        fill_beam_pos, fill_beam_neg = fill_greedy_pos.copy(), fill_greedy_neg.copy()

    results = {
        "multi_greedy": {"pos": multi_greedy_pos, "neg": multi_greedy_neg},
        "multi_beam": {"pos": multi_beam_pos, "neg": multi_beam_neg},
        "fill_greedy": {"pos": fill_greedy_pos, "neg": fill_greedy_neg},
        "fill_beam": {"pos": fill_beam_pos, "neg": fill_beam_neg},
    }

    _log_generations(
        model=model,
        fill_prompts=fill_prompts,
        steering_vec=steering_vec,
        captions=captions,
        lambdas=lambdas,
        layer=layer,
        max_gen_tokens=max_gen_tokens,
        outpath=out_dir / "generated_text.txt",
    )

    plot_all_captions(
        captions=captions,
        results=results,
        lambdas=lambdas,
        layer=layer,
        model_name=target.model_name,
        n_tokens=n_tokens,
        constrained=constrained,
        outpath=out_dir / "sweep_all.png",
    )

    for index, caption in enumerate(captions):
        slug = f"sweep_caption_{index:02d}_{caption['label']}.png"
        plot_single_caption(
            caption_idx=index,
            captions=captions,
            results=results,
            lambdas=lambdas,
            outpath=out_dir / slug,
        )

    _write_manifest(
        out_dir=out_dir,
        target=target,
        layer=layer,
        lambdas=lambdas,
        n_tokens=n_tokens,
        constrained=constrained,
        captions=captions,
        include_beam=include_beam,
        vector_flipped=vector_flipped,
        calibration_probe_lambda=probe_lambda,
    )

    print(f"Saved plots to {out_dir}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render caption-level token-probability sweep plots for multiple models.",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help=(
            "Explicit target in the form slug::model_name::artifact_dir "
            "or slug::model_name::artifact_dir::layer. May be repeated."
        ),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Root directory containing default base-model Rivanna result folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where per-model plots will be written.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "Known model slugs to run from --results-root, e.g. "
            "qwen25_3b_base qwen25_7b_base. "
            f"Default: {' '.join(DEFAULT_MODEL_SLUGS)}"
        ),
    )
    parser.add_argument(
        "--captions-limit",
        type=int,
        default=5,
        help="Number of prompts from bias_steering.captions.CAPTIONS to include.",
    )
    parser.add_argument("--lambda-min", type=int, default=-150)
    parser.add_argument("--lambda-max", type=int, default=150)
    parser.add_argument("--lambda-step", type=int, default=5)
    parser.add_argument(
        "--n-tokens",
        type=int,
        default=8,
        help="How many continuation tokens to average for the token-probability curves.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--beam-top-k", type=int, default=8)
    parser.add_argument("--max-gen-tokens", type=int, default=20)
    parser.add_argument(
        "--greedy-only",
        action="store_true",
        help="Skip beam decoding and reuse greedy curves for the beam columns.",
    )
    parser.add_argument(
        "--unconstrained",
        action="store_true",
        help="Do not constrain the tracked probability mass to the spatial/descriptive token sets.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved targets, vector paths, and layers without loading any model.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    from bias_steering.captions import CAPTIONS

    try:
        targets = select_targets(args.target, args.models, args.results_root)
    except ValueError as exc:
        parser.error(str(exc))
    captions = CAPTIONS[: args.captions_limit]
    lambdas = build_lambda_values(args.lambda_min, args.lambda_max, args.lambda_step)
    constrained = not args.unconstrained

    print(
        f"Targets: {len(targets)}  captions: {len(captions)}  lambdas: {len(lambdas)}  "
        f"beam: {not args.greedy_only}"
    )

    if args.dry_run:
        for target in targets:
            try:
                vector_path = resolve_candidate_vector_path(target.artifact_dir)
                layer = infer_best_layer(target.artifact_dir, explicit_layer=target.layer)
                print(
                    f"[ok] {target.slug}  model={target.model_name}  layer={layer}  "
                    f"vectors={vector_path}"
                )
            except FileNotFoundError as exc:
                print(f"[missing] {target.slug}  model={target.model_name}  error={exc}")
        return

    ran_any = False
    for target in targets:
        try:
            run_target_sweep(
                target=target,
                captions=captions,
                lambdas=lambdas,
                n_tokens=args.n_tokens,
                batch_size=args.batch_size,
                beam_width=args.beam_width,
                beam_top_k=args.beam_top_k,
                constrained=constrained,
                max_gen_tokens=args.max_gen_tokens,
                output_root=args.output_root,
                include_beam=not args.greedy_only,
            )
            ran_any = True
        except FileNotFoundError as exc:
            print(f"[warning] skipping {target.slug}: {exc}")

    if not ran_any:
        print("No runnable targets found. Nothing was plotted.")


if __name__ == "__main__":
    main()

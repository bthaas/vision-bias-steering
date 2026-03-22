"""
Cross-model comparison for Rivanna bias-steering experiments.

Reads results.json from each model's output directory and generates
a MODEL_COMPARISON.md with summary tables and key findings.

Usage:
  python experiments/rivanna/analyze_all.py \\
      --results-dir /scratch/jea7vy/vision-bias-steering/results \\
      --output experiments/rivanna/MODEL_COMPARISON.md

  # Or from Rivanna scratch with default paths:
  python experiments/rivanna/analyze_all.py
"""
import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=None,
                   help="Directory containing per-model result subdirectories. "
                        "Default: experiments/rivanna/results/ (local) or "
                        "/scratch/jea7vy/vision-bias-steering/results (Rivanna)")
    p.add_argument("--output", default=None,
                   help="Output path for MODEL_COMPARISON.md")
    p.add_argument("--include-qwen18b", action="store_true",
                   help="Include Qwen-1.8B results from runs_vision/Qwen-1_8B-chat")
    return p.parse_args()


EXPECTED_MODELS = [
    ("qwen25_3b",  "Qwen/Qwen2.5-3B-Instruct",  "Qwen2.5-3B"),
    ("qwen25_7b",  "Qwen/Qwen2.5-7B-Instruct",  "Qwen2.5-7B"),
    ("qwen25_14b", "Qwen/Qwen2.5-14B-Instruct", "Qwen2.5-14B"),
]

# Qwen-1.8B coherence frontier from our local experiments (hardcoded reference)
QWEN18B_REFERENCE = {
    "model_alias": "Qwen-1.8B",
    "best_layer": 11,
    "baseline_rms": 0.9670,
    "coherence_frontier": {"coeff": -20, "rms": 0.6450, "reduction_pct": 33.3, "mode": "full_steering"},
    "best_1tok":          {"coeff": -50, "rms": 0.3167, "reduction_pct": 67.2, "mode": "1_token_steering"},
    "n_layers": 24,
}


def load_result(path: Path) -> dict:
    return json.loads(path.read_text())


def fmt_pct(v):
    return f"{v:.1f}%" if v is not None and not isinstance(v, str) else "—"


def fmt_rms(v):
    return f"{v:.4f}" if v is not None and not isinstance(v, str) else "—"


def fmt_coeff(v):
    return f"{v:+d}" if v is not None and not isinstance(v, str) else "—"


def find_coherence_frontier(sweep_results):
    """Find max-reduction coherent result from sweep_results list."""
    coherent = [r for r in sweep_results if r.get("coherence") == "coherent"]
    if not coherent:
        return None
    return max(coherent, key=lambda r: r["reduction_pct"])


def find_best_1tok(tok_results):
    """Find max-reduction coherent/partial result from 1-token sweep."""
    rows = tok_results.get("1", [])
    best = None
    for r in rows:
        if r.get("coherence") in ("coherent", "partial"):
            if best is None or r["reduction_pct"] > best["reduction_pct"]:
                best = r
    return best


def summarize_result(data: dict) -> dict:
    """Extract key summary from a results.json dict."""
    cf = data.get("coherence_frontier") or find_coherence_frontier(data.get("sweep_results", []))
    b1 = data.get("best_1tok") or find_best_1tok(data.get("tok_results", {}))
    return {
        "model_alias": data.get("model_alias", data.get("model", "?")),
        "best_layer": data.get("best_layer"),
        "baseline_rms": data.get("baseline_rms"),
        "coherence_frontier": cf,
        "best_1tok": b1,
    }


def build_comparison_md(summaries: list[dict], template: str) -> str:
    lines = [
        "# Cross-Model Bias Steering Comparison",
        "",
        f"Template: `{template}` (output prefix: \"Positioned\")",
        "",
        "---",
        "",
        "## Summary Table",
        "",
        "| Model | Layers | Best Layer | Baseline RMS | Full-steer λ | Full-steer Δ% | 1-token λ | 1-token Δ% |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for s in summaries:
        cf = s.get("coherence_frontier") or {}
        b1 = s.get("best_1tok") or {}
        n_layers = s.get("n_layers", "?")
        lines.append(
            f"| {s['model_alias']} "
            f"| {n_layers} "
            f"| {s.get('best_layer', '—')} "
            f"| {fmt_rms(s.get('baseline_rms'))} "
            f"| {fmt_coeff(cf.get('coeff'))} "
            f"| {fmt_pct(cf.get('reduction_pct'))} "
            f"| {fmt_coeff(b1.get('coeff'))} "
            f"| {fmt_pct(b1.get('reduction_pct'))} "
            f"|"
        )

    lines += [
        "",
        "---",
        "",
        "## Key Findings",
        "",
    ]

    # RMS reduction progression
    full_reduc = [(s["model_alias"], s["coherence_frontier"]["reduction_pct"])
                  for s in summaries if s.get("coherence_frontier")]
    tok_reduc  = [(s["model_alias"], s["best_1tok"]["reduction_pct"])
                  for s in summaries if s.get("best_1tok")]

    if full_reduc:
        best_full = max(full_reduc, key=lambda x: x[1])
        worst_full = min(full_reduc, key=lambda x: x[1])
        lines += [
            "### Full Steering (Coherent Frontier)",
            "",
            f"- Best:  {best_full[0]} at {best_full[1]:.1f}% reduction",
            f"- Worst: {worst_full[0]} at {worst_full[1]:.1f}% reduction",
            "",
        ]

    if tok_reduc:
        best_tok = max(tok_reduc, key=lambda x: x[1])
        worst_tok = min(tok_reduc, key=lambda x: x[1])
        lines += [
            "### 1-Token Steering (Coherent Frontier)",
            "",
            f"- Best:  {best_tok[0]} at {best_tok[1]:.1f}% reduction",
            f"- Worst: {worst_tok[0]} at {worst_tok[1]:.1f}% reduction",
            "",
        ]

    # Best layer summary
    lines += [
        "### Best Layer by Model",
        "",
        "| Model | Best Layer | Total Layers | Layer % |",
        "|---|---|---|---|",
    ]
    for s in summaries:
        bl = s.get("best_layer")
        nl = s.get("n_layers")
        pct = f"{bl/nl*100:.0f}%" if bl is not None and nl and nl != "?" else "?"
        lines.append(f"| {s['model_alias']} | {bl if bl is not None else '—'} | {nl} | {pct} |")

    lines += [
        "",
        "---",
        "",
        "## Per-Model Details",
        "",
    ]

    for s in summaries:
        cf = s.get("coherence_frontier") or {}
        b1 = s.get("best_1tok") or {}
        lines += [
            f"### {s['model_alias']}",
            "",
            f"- **Baseline RMS**: {fmt_rms(s.get('baseline_rms'))}",
            f"- **Best layer**: {s.get('best_layer', '—')} of {s.get('n_layers', '?')}",
            f"- **Full steering frontier**: λ={fmt_coeff(cf.get('coeff'))}  →  {fmt_pct(cf.get('reduction_pct'))} reduction  (RMS {fmt_rms(cf.get('rms'))})",
            f"- **1-token frontier**: λ={fmt_coeff(b1.get('coeff'))}  →  {fmt_pct(b1.get('reduction_pct'))} reduction  (RMS {fmt_rms(b1.get('rms'))})",
            "",
        ]

    return "\n".join(lines)


def main():
    args = parse_args()

    script_dir = Path(__file__).parent
    results_dir = Path(args.results_dir) if args.results_dir else script_dir / "results"
    out_path = Path(args.output) if args.output else script_dir / "MODEL_COMPARISON.md"

    summaries = []

    # Qwen-1.8B reference (local coherence frontier results)
    if args.include_qwen18b:
        ref = dict(QWEN18B_REFERENCE)
        ref["n_layers"] = ref.pop("n_layers", 24)
        summaries.append(ref)

    # Load results for each model
    for subdir, model_id, display_name in EXPECTED_MODELS:
        json_path = results_dir / subdir / "results.json"
        if not json_path.exists():
            # Try finding by model_alias pattern
            candidates = list(results_dir.glob(f"*{subdir}*/results.json"))
            if candidates:
                json_path = candidates[0]

        if json_path.exists():
            print(f"Loading {display_name} from {json_path}")
            data = load_result(json_path)
            s = summarize_result(data)
            s["n_layers"] = s.get("n_layers") or _infer_n_layers(model_id)
            summaries.append(s)
        else:
            print(f"WARNING: No results found for {display_name} (expected {json_path})")
            summaries.append({
                "model_alias": display_name,
                "best_layer": None,
                "baseline_rms": None,
                "coherence_frontier": None,
                "best_1tok": None,
                "n_layers": _infer_n_layers(model_id),
            })

    if not any(s.get("baseline_rms") for s in summaries):
        print("No results found. Run experiments first.")
        print(f"Expected results in: {results_dir}")
        return

    md = build_comparison_md(summaries, template="B_positioned")
    out_path.write_text(md)
    print(f"\nSaved: {out_path}")
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for s in summaries:
        cf = s.get("coherence_frontier") or {}
        b1 = s.get("best_1tok") or {}
        print(f"\n{s['model_alias']}:")
        print(f"  Baseline RMS:    {fmt_rms(s.get('baseline_rms'))}")
        print(f"  Best layer:      {s.get('best_layer', '—')}")
        print(f"  Full steering:   λ={fmt_coeff(cf.get('coeff'))}  {fmt_pct(cf.get('reduction_pct'))}")
        print(f"  1-token:         λ={fmt_coeff(b1.get('coeff'))}  {fmt_pct(b1.get('reduction_pct'))}")


def _infer_n_layers(model_id: str) -> int:
    id_lower = model_id.lower()
    if "3b" in id_lower:  return 36
    if "7b" in id_lower:  return 28
    if "14b" in id_lower: return 48
    if "8b" in id_lower:  return 32
    return -1


if __name__ == "__main__":
    main()

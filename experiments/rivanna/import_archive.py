#!/usr/bin/env python3
"""Import a Rivanna results archive into the local paper workflow.

This script is meant for archives such as `rivanna_results.tar.gz` that bundle
`results/` and optionally `plots/` directories from Rivanna. It extracts the
archive into a temporary directory, syncs recognized model result folders into
`experiments/rivanna/results/`, optionally syncs multimodel sweep plots into
`plots/`, and then rebuilds the paper summary artifacts.

Example:
  python experiments/rivanna/import_archive.py \
      --archive ~/Downloads/rivanna_results.tar.gz \
      --include-local-qwen
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = ROOT / "experiments" / "rivanna" / "results"
DEFAULT_PLOTS_DIR = ROOT / "plots"
BUILD_SCRIPT = ROOT / "experiments" / "rivanna" / "build_paper_assets.py"
EXPECTED_MODELS = {
    "qwen25_3b",
    "qwen25_7b",
    "qwen25_14b",
    "qwen25_3b_base",
    "qwen25_7b_base",
    "qwen18b_base",
}
SWEEP_DIR_NAMES = {
    "multimodel_sweeps",
    "multimodel_sweeps_beam_overnight",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import a Rivanna results tarball.")
    parser.add_argument("--archive", required=True, help="Path to a .tar.gz archive from Rivanna.")
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Destination for per-model results folders.",
    )
    parser.add_argument(
        "--plots-dir",
        default=str(DEFAULT_PLOTS_DIR),
        help="Destination root for imported plot folders.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Do not import sweep plot directories from the archive.",
    )
    parser.add_argument(
        "--include-local-qwen",
        action="store_true",
        help="Pass through to build_paper_assets.py so the local Qwen reference is included.",
    )
    return parser.parse_args()


def find_named_dirs(root: Path, name: str) -> list[Path]:
    return sorted(path for path in root.rglob(name) if path.is_dir())


def choose_results_dir(extract_root: Path) -> Path | None:
    for path in find_named_dirs(extract_root, "results"):
        children = [child for child in path.iterdir() if child.is_dir()]
        if any(child.name in EXPECTED_MODELS for child in children):
            return path
        if any((child / "results.json").exists() for child in children):
            return path
    return None


def choose_plots_dirs(extract_root: Path) -> list[Path]:
    matches: list[Path] = []
    for name in SWEEP_DIR_NAMES:
        matches.extend(find_named_dirs(extract_root, name))
    return sorted(matches)


def sync_tree(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def import_results(results_src: Path, results_dst: Path) -> list[Path]:
    results_dst.mkdir(parents=True, exist_ok=True)
    imported: list[Path] = []
    for child in sorted(results_src.iterdir()):
        if not child.is_dir():
            continue
        if not ((child / "results.json").exists() or (child / "validation").exists()):
            continue
        sync_tree(child, results_dst / child.name)
        imported.append(results_dst / child.name)
    return imported


def import_plots(plot_dirs: list[Path], plots_dst_root: Path) -> list[Path]:
    imported: list[Path] = []
    plots_dst_root.mkdir(parents=True, exist_ok=True)
    for plot_dir in plot_dirs:
        destination = plots_dst_root / plot_dir.name
        sync_tree(plot_dir, destination)
        imported.append(destination)
    return imported


def rebuild_paper_assets(results_dir: Path, include_local_qwen: bool) -> None:
    cmd = ["python", str(BUILD_SCRIPT), "--results-dir", str(results_dir)]
    if include_local_qwen:
        cmd.append("--include-local-qwen")
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    archive = Path(args.archive).expanduser().resolve()
    results_dir = Path(args.results_dir).expanduser().resolve()
    plots_dir = Path(args.plots_dir).expanduser().resolve()

    if not archive.exists():
        raise FileNotFoundError(f"Archive not found: {archive}")

    with tempfile.TemporaryDirectory(prefix="rivanna_import_") as tmp:
        extract_root = Path(tmp)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(extract_root)

        results_src = choose_results_dir(extract_root)
        if results_src is None:
            raise FileNotFoundError(
                "Could not find a results/ directory containing model outputs in the archive."
            )

        imported_results = import_results(results_src, results_dir)
        imported_plots: list[Path] = []
        if not args.skip_plots:
            imported_plots = import_plots(choose_plots_dirs(extract_root), plots_dir)

    rebuild_paper_assets(results_dir, args.include_local_qwen)

    print(f"Imported {len(imported_results)} result folders into {results_dir}")
    for path in imported_results:
        print(f"  - {path}")

    if args.skip_plots:
        print("Skipped plot import.")
    else:
        if imported_plots:
            print(f"Imported {len(imported_plots)} plot directories into {plots_dir}")
            for path in imported_plots:
                print(f"  - {path}")
        else:
            print("No sweep plot directories were found in the archive.")

    print("Rebuilt paper summary assets.")


if __name__ == "__main__":
    main()

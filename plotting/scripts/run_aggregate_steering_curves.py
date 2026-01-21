#!/usr/bin/env python3
"""
Script to run aggregate_steering_curves.ipynb
This notebook shows aggregate steering curves across all examples.
"""
import sys
from pathlib import Path

# Import from same directory
from run_notebook import run_notebook

if __name__ == "__main__":
    # Scripts are in plotting/scripts/, notebooks are in plotting/
    notebook_path = Path(__file__).parent.parent / "aggregate_steering_curves.ipynb"
    print("=" * 60)
    print("Running: aggregate_steering_curves.ipynb")
    print("=" * 60)
    success = run_notebook(notebook_path, timeout=300)
    if success:
        print("\n✓ Graph generation complete!")
    else:
        print("\n✗ Failed to generate graphs")
        sys.exit(1)

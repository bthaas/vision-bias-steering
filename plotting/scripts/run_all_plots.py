#!/usr/bin/env python3
"""
Script to run all plotting notebooks in sequence.
"""
import sys
from pathlib import Path

# Import from same directory
from run_notebook import run_notebook

NOTEBOOKS = [
    ("aggregate_steering_curves.ipynb", 300, "Aggregate steering curves"),
    ("model_bias_statistics.ipynb", 300, "Model bias statistics"),
    ("bias-mitigation.ipynb", 300, "Bias mitigation plots"),
]

if __name__ == "__main__":
    # Scripts are in plotting/scripts/, notebooks are in plotting/
    plotting_dir = Path(__file__).parent.parent
    results = []
    
    print("=" * 60)
    print("Running all plotting notebooks")
    print("=" * 60)
    print()
    
    for notebook_name, timeout, description in NOTEBOOKS:
        notebook_path = plotting_dir / notebook_name
        print(f"\n[{len(results) + 1}/{len(NOTEBOOKS)}] {description}")
        print(f"Notebook: {notebook_name}")
        print("-" * 60)
        
        success = run_notebook(notebook_path, timeout=timeout)
        results.append((notebook_name, success))
        
        if success:
            print(f"✓ {notebook_name} completed successfully")
        else:
            print(f"✗ {notebook_name} failed")
        print()
    
    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    successful = [name for name, success in results if success]
    failed = [name for name, success in results if not success]
    
    if successful:
        print(f"\n✓ Successfully completed ({len(successful)}/{len(results)}):")
        for name in successful:
            print(f"  - {name}")
    
    if failed:
        print(f"\n✗ Failed ({len(failed)}/{len(results)}):")
        for name in failed:
            print(f"  - {name}")
        sys.exit(1)
    else:
        print("\n✓ All notebooks completed successfully!")

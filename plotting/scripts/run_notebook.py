#!/usr/bin/env python3
"""
Script to execute a Jupyter notebook programmatically.
Usage: python run_notebook.py <notebook_path> [--output <output_path>]
"""
import sys
import subprocess
from pathlib import Path

def run_notebook(notebook_path, output_path=None, timeout=None):
    """
    Execute a Jupyter notebook using nbconvert.
    
    Args:
        notebook_path: Path to the notebook file
        output_path: Optional path for output notebook (default: overwrites input)
        timeout: Optional timeout in seconds
    """
    notebook_path = Path(notebook_path)
    if not notebook_path.exists():
        print(f"Error: Notebook not found: {notebook_path}")
        return False
    
    # Use nbconvert to execute the notebook
    cmd = [
        "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
    ]
    
    if output_path is None:
        cmd.append("--inplace")
    else:
        cmd.extend(["--output", str(output_path)])
    
    if timeout:
        cmd.extend(["--ExecutePreprocessor.timeout", str(timeout)])
    
    cmd.append(str(notebook_path))
    
    print(f"Executing notebook: {notebook_path}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run with real-time output
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        
        if process.returncode == 0:
            print(f"\n✓ Successfully executed: {notebook_path}")
            return True
        else:
            print(f"\n✗ Error executing notebook: {notebook_path}")
            print(f"Return code: {process.returncode}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"✗ Error executing notebook: {notebook_path}")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False
    except FileNotFoundError:
        print("Error: 'jupyter' command not found. Make sure Jupyter is installed.")
        print("Install with: pip install jupyter nbconvert")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_notebook.py <notebook_path> [--output <output_path>] [--timeout <seconds>]")
        sys.exit(1)
    
    notebook_path = sys.argv[1]
    output_path = None
    timeout = None
    
    # Parse arguments
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--output" and i + 1 < len(sys.argv):
            output_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--timeout" and i + 1 < len(sys.argv):
            timeout = int(sys.argv[i + 1])
            i += 2
        else:
            i += 1
    
    success = run_notebook(notebook_path, output_path, timeout)
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""Test script to verify vision steering plotting works"""

import json
from pathlib import Path
import numpy as np
import plotly.graph_objects as go

# Colors for language types in the vision setting
COLORS = {
    "spatial": "#FF7F0E",     # Orange
    "descriptive": "#1F77B4", # Blue
}
FILL_COLORS = {
    "spatial": 'rgba(255, 127, 14, 0.2)',
    "descriptive": 'rgba(31, 119, 180, 0.2)',
}

def load_results(artifact_path, normalized=True):
    """Load coefficient test results"""
    coeffs = json.load(open(artifact_path / "coeff_test/coeffs.json", "r"))["coeff"]
    outputs = json.load(open(artifact_path / "coeff_test/outputs.json", "r"))
    
    # For vision: pos_probs (spatial) and neg_probs (descriptive)
    if normalized:
        for x in outputs:
            pos_probs = np.array(x["pos_probs"])
            neg_probs = np.array(x["neg_probs"])
            total = pos_probs + neg_probs
            x["pos_probs"] = (pos_probs / total).tolist()
            x["neg_probs"] = (neg_probs / total).tolist()
    
    return coeffs, outputs

def get_avg_std(x):
    """Calculate mean and std across first axis"""
    avg = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    return avg, std

def plot_steering(coeffs, outputs, title_text=None, width=425, height=300, legend_title="Language Type", error_band=False, x_range=None):
    """Create steering plot"""
    fig = go.Figure()

    # For vision: plot spatial (pos) and descriptive (neg)
    for group, label in [("pos", "spatial"), ("neg", "descriptive")]:
        avg, std = get_avg_std([x[f'{group}_probs'] for x in outputs])
        fig.add_trace(go.Scatter(
            x=coeffs, y=avg, mode='lines+markers', name=label, 
            marker_color=COLORS[label], showlegend=True
        ))
        if error_band:
            fig.add_trace(go.Scatter(
                x=coeffs, y=avg+std, mode='lines', marker=dict(color="#444"), 
                line=dict(width=0), showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=coeffs, y=avg-std, mode='lines', marker=dict(color="#444"), 
                line=dict(width=0), 
                fillcolor=FILL_COLORS[label], fill='tonexty', showlegend=False
            ))
        
    fig.update_layout(
        width=width, height=height, plot_bgcolor='white',
        margin=dict(l=10, r=10, t=20, b=25),
        font=dict(size=14), title_text=title_text, 
        title_font=dict(size=16), title_x=0.48, title_y=0.98,
        legend_title_text=legend_title, legend_title_font=dict(size=15),
    )
    fig.update_xaxes(
        mirror=True, showgrid=True, gridcolor='darkgrey',
        zeroline = True, zerolinecolor='black',
        title_text="Steering Coefficient (λ)",
        title_font=dict(size=15), tickfont=dict(size=13),
        showline=True, linewidth=1, linecolor='darkgrey',
        title_standoff=1, nticks=10, range=x_range, 
    )
    fig.update_yaxes(
        mirror=True, showgrid=True, gridcolor='darkgrey',
        zeroline = True, zerolinecolor='darkgrey',
        title_text="Probability (%)",
        title_font=dict(size=15), tickfont=dict(size=13),
        showline=True, linewidth=1, linecolor='darkgrey',
        title_standoff=2, range=[0, 1],
    )
    return fig

if __name__ == "__main__":
    # Load data - adjust path based on where you run from
    # If running from plotting/ directory: use "../runs_vision/gpt2"
    # If running from project root: use "runs_vision/gpt2"
    artifact_path = Path("../runs_vision/gpt2")
    
    print(f"Loading from: {artifact_path.absolute()}")
    
    try:
        coeffs, outputs = load_results(artifact_path, normalized=True)
        print(f"✓ Loaded {len(coeffs)} coefficients and {len(outputs)} samples")
        
        fig = plot_steering(
            coeffs, outputs, 
            width=470, height=300, 
            error_band=True, 
            title_text="Vision Steering (Spatial vs Descriptive)", 
            x_range=[-41, 41]
        )
        
        print("✓ Plot created successfully!")
        print("  Opening in browser...")
        fig.show()
        
    except FileNotFoundError as e:
        print(f"✗ File not found: {e}")
        print(f"  Make sure you're running from the plotting/ directory")
        print(f"  Or change the path in the script")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

"""
Plotting utilities for Jupyter notebooks using Plotly.

This module provides standardized plotting functions for visualizing
steering coefficient tests, projections, and comparisons.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Dict, Optional, Union
from pathlib import Path


# ============================================================================
# Color Schemes
# ============================================================================

# For categorical data (e.g., gender groups)
COLORS_CATEGORICAL = {
    "F": "#FF7F0E",  # Orange
    "M": "#1F77B4",  # Blue
    "N": "#2CA02C",   # Green
    "pos": "#FF7F0E",  # Orange (alias for positive)
    "neg": "#1F77B4",  # Blue (alias for negative)
    "spatial": "#FF7F0E",  # Orange (for vision/spatial)
    "descriptive": "#1F77B4",  # Blue (for non-vision/descriptive)
}

# For filled areas/error bands (same colors with transparency)
FILL_COLORS = {
    "F": 'rgba(255, 127, 14, 0.2)',  # 20% opacity
    "M": 'rgba(31, 119, 180, 0.2)',
    "N": "rgba(44,160,44, 0.2)",
    "pos": 'rgba(255, 127, 14, 0.2)',
    "neg": 'rgba(31, 119, 180, 0.2)',
    "spatial": 'rgba(255, 127, 14, 0.2)',
    "descriptive": 'rgba(31, 119, 180, 0.2)',
}

# For before/after comparisons
COLORS_COMPARISON = {
    "before": "#4C78A8",  # Blue
    "after": "#E45756"    # Red
}

# For token/line plots (use Plotly's D3 palette)
COLORS_D3 = px.colors.qualitative.D3


# ============================================================================
# Helper Functions
# ============================================================================

def get_avg_std(x: Union[List, np.ndarray], axis: int = 0) -> tuple:
    """
    Calculate mean and std across specified axis.
    
    Args:
        x: Array or list of arrays
        axis: Axis to compute statistics over
        
    Returns:
        Tuple of (mean, std) arrays
    """
    x = np.array(x)
    avg = np.mean(x, axis=axis)
    std = np.std(x, axis=axis)
    return avg, std


# ============================================================================
# Standard Layout Function
# ============================================================================

def apply_standard_layout(
    fig: go.Figure,
    width: int = 425,
    height: int = 300,
    title_text: Optional[str] = None,
    legend_title: Optional[str] = None,
    x_range: Optional[List[float]] = None,
    y_range: Optional[List[float]] = None,
    x_title: str = "X Axis",
    y_title: str = "Y Axis",
    legend_x: float = 0.02,
    legend_y: float = 0.98
) -> None:
    """
    Apply consistent styling to all plots.
    
    Args:
        fig: Plotly figure to style
        width: Figure width in pixels
        height: Figure height in pixels
        title_text: Plot title
        legend_title: Legend title text
        x_range: Optional [min, max] for x-axis
        y_range: Optional [min, max] for y-axis
        x_title: X-axis title
        y_title: Y-axis title
        legend_x: Legend x position (0-1, paper coordinates)
        legend_y: Legend y position (0-1, paper coordinates)
    """
    # Layout
    layout_dict = {
        "width": width,
        "height": height,
        "plot_bgcolor": 'white',  # White background
        "margin": dict(l=10, r=10, t=20, b=25),  # Tight margins
        "font": dict(size=14),
        "title_text": title_text,
        "title_font": dict(size=16),
        "title_x": 0.48,  # Center title
        "title_y": 0.98,  # Near top
    }
    
    # Add legend if legend_title is provided
    if legend_title:
        layout_dict["legend_title_text"] = legend_title
        layout_dict["legend_title_font"] = dict(size=15)
        layout_dict["legend"] = dict(
            yanchor="top", y=legend_y,
            xanchor="left", x=legend_x,
            bordercolor="darkgrey", borderwidth=1,
            font=dict(size=15)
        )
    
    fig.update_layout(**layout_dict)
    
    # X-axis styling
    fig.update_xaxes(
        mirror=True,  # Mirror ticks on opposite side
        showgrid=True,
        gridcolor='darkgrey',
        zeroline=True,
        zerolinecolor='black',  # Black zero line
        title_text=x_title,
        title_font=dict(size=15),
        tickfont=dict(size=13),
        showline=True,
        linewidth=1,
        linecolor='darkgrey',
        title_standoff=1,
        nticks=10,
        range=x_range,
    )
    
    # Y-axis styling
    fig.update_yaxes(
        mirror=True,
        showgrid=True,
        gridcolor='darkgrey',
        zeroline=True,
        zerolinecolor='darkgrey',  # Grey zero line
        title_text=y_title,
        title_font=dict(size=15),
        tickfont=dict(size=13),
        showline=True,
        linewidth=1,
        linecolor='darkgrey',
        title_standoff=2,
        range=y_range if y_range else [0, 1],  # Default 0-1 for probabilities
    )


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_steering(
    coeffs: Union[List, np.ndarray],
    outputs: List[Dict],
    title_text: Optional[str] = None,
    width: int = 425,
    height: int = 300,
    legend_title: str = "Category",
    error_band: bool = False,
    x_range: Optional[List[float]] = None,
    y_range: Optional[List[float]] = None,
    categories: Optional[List[str]] = None,
    colors: Optional[Dict[str, str]] = None,
    fill_colors: Optional[Dict[str, str]] = None,
    x_title: str = "Coefficient",
    y_title: str = "Probability"
) -> go.Figure:
    """
    Create line plot with optional error bands for steering coefficient tests.
    
    Args:
        coeffs: x-axis values (list/array of coefficients)
        outputs: list of dicts with probability arrays per category
                 Each dict should have keys like "pos_probs", "neg_probs", 
                 "F_probs", "M_probs", "N_probs", etc.
        title_text: Plot title
        width, height: Figure dimensions
        legend_title: Legend title text
        error_band: If True, add shaded error bands
        x_range: Optional [min, max] for x-axis
        y_range: Optional [min, max] for y-axis
        categories: List of category keys to plot (e.g., ["pos", "neg"] or ["F", "M", "N"])
                    If None, will try to infer from outputs
        colors: Optional dict mapping category to color (overrides defaults)
        fill_colors: Optional dict mapping category to fill color (overrides defaults)
        x_title: X-axis title
        y_title: Y-axis title
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Determine categories if not provided
    if categories is None:
        # Try to infer from first output dict
        if outputs and len(outputs) > 0:
            # Look for common patterns
            keys = outputs[0].keys()
            if "pos_probs" in keys and "neg_probs" in keys:
                categories = ["pos", "neg"]
            elif "F_probs" in keys and "M_probs" in keys:
                categories = ["F", "M"]
                if "N_probs" in keys:
                    categories.append("N")
            elif "spatial_probs" in keys and "descriptive_probs" in keys:
                categories = ["spatial", "descriptive"]
            else:
                # Fallback: use all keys ending in "_probs"
                categories = [k.replace("_probs", "") for k in keys if k.endswith("_probs")]
        else:
            categories = ["pos", "neg"]  # Default fallback
    
    # Use provided colors or defaults
    if colors is None:
        colors = COLORS_CATEGORICAL
    if fill_colors is None:
        fill_colors = FILL_COLORS
    
    # Process each category
    for group in categories:
        prob_key = f"{group}_probs"
        
        # Extract probabilities for this category
        prob_arrays = []
        for output in outputs:
            if prob_key in output:
                prob_arrays.append(output[prob_key])
        
        if not prob_arrays:
            continue
        
        # Calculate statistics
        avg, std = get_avg_std(prob_arrays)
        
        # Get color for this group
        color = colors.get(group, COLORS_D3[0])
        fill_color = fill_colors.get(group, f'rgba(128, 128, 128, 0.2)')
        
        # Main line trace
        fig.add_trace(go.Scatter(
            x=coeffs,
            y=avg,
            mode='lines+markers',  # Both lines and markers
            name=group,
            marker_color=color,
            showlegend=True
        ))
        
        # Error band (if requested)
        if error_band:
            # Upper bound (invisible)
            fig.add_trace(go.Scatter(
                x=coeffs, y=avg + std,
                mode='lines',
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            # Lower bound with fill
            fig.add_trace(go.Scatter(
                x=coeffs, y=avg - std,
                mode='lines',
                marker=dict(color="#444"),
                line=dict(width=0),
                fillcolor=fill_color,
                fill='tonexty',  # Fill to previous trace
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Apply standard styling
    apply_standard_layout(
        fig, width, height, title_text, legend_title, 
        x_range, y_range, x_title, y_title
    )
    
    return fig


def plot_projection(
    projections: Union[List, np.ndarray],
    scores: Union[List, np.ndarray],
    width: int = 350,
    height: int = 300,
    x_range: Optional[List[float]] = None,
    y_range: Optional[List[float]] = None,
    title_text: Optional[str] = None,
    x_title: str = "Projection",
    y_title: str = "Score",
    color: str = None
) -> go.Figure:
    """
    Scatter plot with diagonal reference line (y=x).
    
    Args:
        projections: x-axis values
        scores: y-axis values
        width, height: Figure dimensions
        x_range: Optional [min, max] for x-axis
        y_range: Optional [min, max] for y-axis
        title_text: Plot title
        x_title: X-axis title
        y_title: Y-axis title
        color: Marker color (defaults to "before" color)
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    if color is None:
        color = COLORS_COMPARISON["before"]
    
    # Scatter trace
    fig.add_trace(go.Scatter(
        x=projections,
        y=scores,
        mode="markers",  # Only markers
        marker_color=color,
        showlegend=False
    ))
    
    # Diagonal reference line (y=x)
    # Use paper coordinates for line that spans full plot
    if x_range is None:
        x_min, x_max = min(projections), max(projections)
    else:
        x_min, x_max = x_range
    
    if y_range is None:
        y_min, y_max = min(scores), max(scores)
    else:
        y_min, y_max = y_range
    
    # Add diagonal line using data coordinates
    fig.add_shape(
        type="line",
        x0=x_min, y0=x_min,
        x1=x_max, y1=x_max,
        line=dict(color="#66AA00", width=3, dash="dash"),
        layer="below"  # Behind data
    )
    
    # Apply styling
    apply_standard_layout(
        fig, width, height, title_text,
        x_title=x_title, y_title=y_title,
        x_range=x_range, y_range=y_range
    )
    
    return fig


def plot_comparison(
    before_data: Union[List, np.ndarray],
    after_data: Union[List, np.ndarray],
    x_values: Union[List, np.ndarray],
    width: int = 360,
    height: int = 300,
    x_range: Optional[List[float]] = None,
    y_range: Optional[List[float]] = None,
    showlegend: bool = True,
    title_text: Optional[str] = None,
    legend_x: float = 0.02,
    legend_y: float = 0.98,
    opacity: float = 0.8,
    x_title: str = "X Axis",
    y_title: str = "Y Axis"
) -> go.Figure:
    """
    Before/after comparison scatter plot.
    
    Args:
        before_data: Y values for "before" condition
        after_data: Y values for "after" condition
        x_values: X values (shared for both conditions)
        width, height: Figure dimensions
        x_range: Optional [min, max] for x-axis
        y_range: Optional [min, max] for y-axis
        showlegend: Whether to show legend
        title_text: Plot title
        legend_x: Legend x position (0-1, paper coordinates)
        legend_y: Legend y position (0-1, paper coordinates)
        opacity: Opacity for "after" markers
        x_title: X-axis title
        y_title: Y-axis title
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Before trace
    fig.add_trace(go.Scatter(
        x=x_values, y=before_data,
        mode="markers",
        marker_color=COLORS_COMPARISON["before"],
        marker_size=5,
        name="before",
        showlegend=showlegend
    ))
    
    # After trace
    fig.add_trace(go.Scatter(
        x=x_values, y=after_data,
        mode="markers",
        marker_color=COLORS_COMPARISON["after"],
        marker_size=5,
        name="after",
        showlegend=showlegend,
        opacity=opacity
    ))
    
    # Apply styling with custom legend
    apply_standard_layout(
        fig, width, height, title_text,
        x_title=x_title, y_title=y_title,
        x_range=x_range, y_range=y_range,
        legend_x=legend_x, legend_y=legend_y
    )
    
    return fig


def plot_single_trace(
    x: Union[List, np.ndarray],
    y: Union[List, np.ndarray],
    mode: str = "lines+markers",
    title_text: Optional[str] = None,
    width: int = 425,
    height: int = 300,
    x_title: str = "X Axis",
    y_title: str = "Y Axis",
    color: Optional[str] = None,
    name: Optional[str] = None,
    x_range: Optional[List[float]] = None,
    y_range: Optional[List[float]] = None
) -> go.Figure:
    """
    Simple single-trace plot (line or scatter).
    
    Args:
        x: X-axis values
        y: Y-axis values
        mode: Plot mode ("lines", "markers", or "lines+markers")
        title_text: Plot title
        width, height: Figure dimensions
        x_title: X-axis title
        y_title: Y-axis title
        color: Trace color
        name: Trace name (for legend)
        x_range: Optional [min, max] for x-axis
        y_range: Optional [min, max] for y-axis
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode=mode,
        marker_color=color,
        name=name,
        showlegend=name is not None
    ))
    
    apply_standard_layout(
        fig, width, height, title_text,
        x_title=x_title, y_title=y_title,
        x_range=x_range, y_range=y_range
    )
    
    return fig

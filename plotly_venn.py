"""Tiny helper to build a 2-set Venn diagram with Plotly.

Usage::

    from plotly_venn import venn
    fig = venn(120, 200, 20, labels=("Model A", "Model B"), title="Results")
    fig.show()

Features: proportional areas, containment handling, collision avoidance,
color coded count chips, optional legend (labels > auto counts > none).
Kept deliberately small & dependency-minimal; for multi-set diagrams use a
dedicated library.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, Optional, Sequence, Tuple

import plotly.graph_objects as go

__all__ = ["venn"]

_RGBA_RE = re.compile(r"rgba?\(([^)]+)\)")

def _circle_intersection_area(r1: float, r2: float, d: float) -> float:
    """Calculate the intersection area of two circles.
    
    Args:
        r1: Radius of first circle
        r2: Radius of second circle  
        d: Distance between circle centers
        
    Returns:
        Intersection area
    """
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return math.pi * min(r1, r2) ** 2

    def clamp(x: float) -> float:
        return max(-1.0, min(1.0, x))

    a1 = math.acos(clamp((d * d + r1 * r1 - r2 * r2) / (2 * d * r1)))
    a2 = math.acos(clamp((d * d + r2 * r2 - r1 * r1) / (2 * d * r2)))
    return (
        r1 * r1 * a1
        + r2 * r2 * a2
        - 0.5 * math.sqrt(
            max(
                0.0,
                (-d + r1 + r2)
                * (d + r1 - r2)
                * (d - r1 + r2)
                * (d + r1 + r2),
            )
        )
    )


def _solve_distance(
    r1: float, 
    r2: float, 
    target_area: float, 
    eps: float = 1e-3,
    tol: float = 1e-6, 
    iters: int = 60
) -> float:
    """Solve for the distance between circle centers to achieve target intersection area.
    
    Args:
        r1: Radius of first circle
        r2: Radius of second circle
        target_area: Desired intersection area
        eps: Small epsilon for boundary cases
        tol: Tolerance for binary search convergence
        iters: Maximum iterations for binary search
        
    Returns:
        Distance between circle centers
    """
    lo, hi = abs(r1 - r2), r1 + r2
    full = math.pi * min(r1, r2) ** 2
    
    if target_area <= 0:
        return hi + eps
    if target_area >= full:
        return max(1e-6, lo - eps)
        
    for _ in range(iters):
        mid = (lo + hi) * 0.5
        if _circle_intersection_area(r1, r2, mid) > target_area:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) * 0.5


def _parse_rgba(color: str) -> Tuple[int, int, int, float]:
    """Parse RGBA color from CSS format, fallback to mid gray.
    
    Args:
        color: CSS color string (rgba(), rgb(), or hex)
        
    Returns:
        RGBA tuple (r, g, b, a) where RGB are 0-255 and alpha is 0-1
    """
    if not isinstance(color, str):
        return 120, 120, 120, 1.0
        
    color = color.strip()
    match = _RGBA_RE.match(color)
    
    if match:
        vals = [p.strip() for p in match.group(1).split(',')]
        r, g, b = [int(float(x)) for x in vals[:3]]
        a = float(vals[3]) if len(vals) > 3 else 1.0
        return r, g, b, a
        
    if color.startswith('#'):
        hex_color = color[1:]
        if len(hex_color) == 3:
            hex_color = ''.join(ch * 2 for ch in hex_color)
        if len(hex_color) == 6:
            return (
                int(hex_color[0:2], 16), 
                int(hex_color[2:4], 16), 
                int(hex_color[4:6], 16), 
                1.0
            )
            
    return 120, 120, 120, 1.0


def _rgba_str(rgba: Iterable[float]) -> str:
    """Convert RGBA tuple to CSS rgba string."""
    r, g, b, a = rgba
    return f"rgba({int(r)},{int(g)},{int(b)},{a:.3f})"


def _blend(color1: str, color2: str) -> str:
    """Blend two colors by averaging RGB and taking max alpha."""
    r1, g1, b1, a1 = _parse_rgba(color1)
    r2, g2, b2, a2 = _parse_rgba(color2)
    return _rgba_str(((r1 + r2) / 2, (g1 + g2) / 2, (b1 + b2) / 2, max(a1, a2)))


def _derive_colors(base_color: str) -> Tuple[str, str, str]:
    """Derive background, border, and font colors for chip styling.
    
    Args:
        base_color: Base color to derive from
        
    Returns:
        Tuple of (background_color, border_color, font_color)
    """
    r, g, b, a = _parse_rgba(base_color)
    bg = _rgba_str((r, g, b, min(0.85, max(0.5, a + 0.25))))
    border = _rgba_str((r, g, b, 0.95))
    
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    font = "#111" if luminance > 150 else "#f5f5f5"
    
    return bg, border, font


def _ensure_alpha(color: str, default_alpha: float = 0.5) -> str:
    """Ensure color has transparency for proper Venn diagram rendering.
    
    Args:
        color: Input color string
        default_alpha: Default alpha to use if color is opaque
        
    Returns:
        Color string with appropriate alpha
    """
    r, g, b, a = _parse_rgba(color)
    
    # If the color is fully opaque (alpha = 1.0), apply default transparency
    if a >= 0.99:  
        a = default_alpha
        
    return _rgba_str((r, g, b, a))


def venn(
    a_count: int,
    b_count: int,
    ab_count: int,
    *,
    labels: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    colors: Sequence[str] = ("rgba(31,119,180,0.5)", "rgba(255,127,14,0.5)"),
    outline: str = "#333",
    max_radius: float = 1.8,
    font_size: int = 16,
    avoid_collision: bool = True,
    show_zero: bool = False,
    auto_legend_counts: bool = True,
    show_legend: Optional[bool] = None,
    inside: bool = True,
) -> go.Figure:
    """Create a two-set Venn diagram with proportional areas.

    Args:
        a_count: Count for set A
        b_count: Count for set B
        ab_count: Count for intersection of A and B
        labels: Optional labels for the sets
        title: Optional title for the diagram
        colors: Colors for sets A and B
        outline: Color for circle outlines
        max_radius: Maximum radius for scaling
        font_size: Font size for count labels
        avoid_collision: Whether to avoid label collision
        show_zero: Whether to show zero counts
        auto_legend_counts: Whether to auto-generate legend with counts
        show_legend: Whether to show legend (auto-determined if None)
        inside: Whether to place labels inside circles or below

    Returns:
        Plotly Figure object
    """
    # Normalize counts to non-negative integers
    a_count = max(0, int(a_count))
    b_count = max(0, int(b_count))
    ab_count = max(0, int(ab_count))
    
    # Ensure colors have enough transparency
    colors = (_ensure_alpha(colors[0]), _ensure_alpha(colors[1]))
    
    # Calculate effective intersection (cannot exceed either set)
    ab_effective = max(0, min(ab_count, a_count, b_count))
    only_a = a_count - ab_effective
    only_b = b_count - ab_effective
    both = ab_effective

    # Calculate circle radii and positions
    def radius_from_area(area: float) -> float:
        return math.sqrt(max(area, 1e-9) / math.pi)

    r1_raw = radius_from_area(a_count)
    r2_raw = radius_from_area(b_count)
    scale = max_radius / max(r1_raw, r2_raw) if max(r1_raw, r2_raw) > 0 else 1.0
    r1, r2 = r1_raw * scale, r2_raw * scale
    
    # Solve for distance to achieve target intersection area
    distance = _solve_distance(r1, r2, ab_effective * scale * scale)
    
    # Center circles around x=0 for balanced layout
    offset = distance * 0.5
    x1, y1 = -offset, 0.0
    x2, y2 = distance - offset, 0.0
    
    # Calculate label positions
    x_both_local = ((r1 * r1 - r2 * r2 + distance * distance) / (2 * distance) 
                    if distance > 1e-9 else 0.0)
    x_both = x1 + x_both_local
    x_left = x1 - 0.45 * r1
    x_right = x2 + 0.45 * r2

    # Create label configuration
    labels_pos = [
        {
            "name": "only_a", 
            "x": x_left, 
            "y": 0.0, 
            "text": str(only_a), 
            "color": colors[0]
        },
        {
            "name": "both", 
            "x": x_both, 
            "y": 0.0, 
            "text": str(both), 
            "color": _blend(colors[0], colors[1])
        },
        {
            "name": "only_b", 
            "x": x_right, 
            "y": 0.0, 
            "text": str(only_b), 
            "color": colors[1]
        },
    ]

    # Handle label positioning based on inside/outside preference
    _position_labels(labels_pos, r1, r2, distance, x1, x2, y1, y2, 
                    inside, avoid_collision, show_zero, labels)

    # Calculate plot bounds
    pad = 0.35
    xmin, xmax, ymin, ymax = _calculate_bounds(
        labels_pos, x1, y1, r1, x2, y2, r2, pad, inside
    )

    # Create the figure
    fig = _create_figure(
        labels_pos, x1, y1, r1, x2, y2, r2, colors, outline, font_size,
        xmin, xmax, ymin, ymax, title, labels, show_legend, 
        auto_legend_counts, a_count, b_count
    )

    return fig

def _position_labels(
    labels_pos: list,
    r1: float, 
    r2: float, 
    distance: float,
    x1: float, 
    x2: float, 
    y1: float, 
    y2: float,
    inside: bool,
    avoid_collision: bool,
    show_zero: bool,
    labels: Optional[Sequence[str]]
) -> None:
    """Position labels either inside circles or outside below them."""
    containment = distance <= abs(r1 - r2) + 1e-6
    
    if inside:
        if containment:  # Full overlap of smaller circle inside bigger
            by_name = {lp["name"]: lp for lp in labels_pos}
            if r1 >= r2:
                by_name["both"]["x"] = x2
                small_exclusive = by_name["only_b"]
            else:
                by_name["both"]["x"] = x1
                small_exclusive = by_name["only_a"]
                
            if not show_zero and int(small_exclusive["text"]) == 0:
                small_exclusive["hidden"] = True
                
        if avoid_collision and not containment:
            desired_min_dx = 0.25 * max(r1, r2)
            left, over, right = labels_pos
            
            if abs(over["x"] - left["x"]) < desired_min_dx:
                left["x"] -= desired_min_dx - (over["x"] - left["x"])
            if abs(right["x"] - over["x"]) < desired_min_dx:
                right["x"] += desired_min_dx - (right["x"] - over["x"])
                
            def too_close(a, b):
                return abs(a["x"] - b["x"]) < 0.18 * max(r1, r2)
                
            if too_close(left, over):
                left["y"] = 0.22 * max(r1, r2)
            if too_close(right, over):
                right["y"] = -0.22 * max(r1, r2)
    else:
        # Outside placement: labels below circles
        span = max(r1, r2)
        by_name = {lp["name"]: lp for lp in labels_pos}
        
        label_a = labels[0] if labels else "A"
        label_b = labels[1] if labels else "B"
        
        if labels:
            name_a = f"Only {label_a}"
            name_b = f"Only {label_b}"
        else:
            name_a = "A"
            name_b = "B"
        
        # Position labels below circles
        y_offset = -(span + 0.5 * span)
        spacing = span * 1.2
        
        by_name["only_a"]["x"] = -spacing
        by_name["only_a"]["y"] = y_offset
        by_name["only_a"]["text"] = f"{name_a}: {by_name['only_a']['text']}"
        
        by_name["both"]["x"] = 0.0
        by_name["both"]["y"] = y_offset
        by_name["both"]["text"] = f"Intersection: {by_name['both']['text']}"
        
        by_name["only_b"]["x"] = spacing
        by_name["only_b"]["y"] = y_offset
        by_name["only_b"]["text"] = f"{name_b}: {by_name['only_b']['text']}"
        
        # Hide zero counts if requested
        if not show_zero:
            if int(by_name["only_a"]["text"].split(": ")[1]) == 0:
                by_name["only_a"]["hidden"] = True
            if int(by_name["only_b"]["text"].split(": ")[1]) == 0:
                by_name["only_b"]["hidden"] = True


def _calculate_bounds(
    labels_pos: list,
    x1: float, 
    y1: float, 
    r1: float,
    x2: float, 
    y2: float, 
    r2: float,
    pad: float,
    inside: bool
) -> Tuple[float, float, float, float]:
    """Calculate plot bounds including circles and labels."""
    if inside:
        visible_x = [lp["x"] for lp in labels_pos if not lp.get("hidden")]
        visible_y = [lp["y"] for lp in labels_pos if not lp.get("hidden")]
        
        xmin = min(x1 - r1, x2 - r2, *visible_x) - pad
        xmax = max(x1 + r1, x2 + r2, *visible_x) + pad
        ymin = min(y1 - r1, y2 - r2, *visible_y) - pad
        ymax = max(y1 + r1, y2 + r2, *visible_y) + pad
    else:
        xmin = min(x1 - r1, x2 - r2) - pad
        xmax = max(x1 + r1, x2 + r2) + pad
        
        circle_ymin = min(y1 - r1, y2 - r2)
        circle_ymax = max(y1 + r1, y2 + r2)
        
        visible_y = [lp["y"] for lp in labels_pos if not lp.get("hidden")]
        min_label_y = min(visible_y) if visible_y else circle_ymin
        max_label_y = max(visible_y) if visible_y else circle_ymax
        
        ymin = min(circle_ymin, min_label_y) - pad
        ymax = max(circle_ymax, max_label_y) + pad
    
    return xmin, xmax, ymin, ymax


def _create_figure(
    labels_pos: list,
    x1: float, 
    y1: float, 
    r1: float,
    x2: float, 
    y2: float, 
    r2: float,
    colors: Sequence[str],
    outline: str,
    font_size: int,
    xmin: float, 
    xmax: float, 
    ymin: float, 
    ymax: float,
    title: Optional[str],
    labels: Optional[Sequence[str]],
    show_legend: Optional[bool],
    auto_legend_counts: bool,
    a_count: int,
    b_count: int
) -> go.Figure:
    """Create the Plotly figure with circles, labels, and legend."""
    fig = go.Figure()

    # Determine legend settings
    if show_legend is None:
        show_legend = bool(labels or (auto_legend_counts and (a_count or b_count)))
        
    if show_legend:
        if labels:
            legend_names = (str(labels[0]), str(labels[1]))
        elif auto_legend_counts:
            legend_names = (str(a_count), str(b_count))
        else:
            legend_names = ("", "")
            
        # Add legend traces
        if legend_names[0]:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=12, color=colors[0]),
                name=legend_names[0], showlegend=True
            ))
        if legend_names[1]:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=12, color=colors[1]),
                name=legend_names[1], showlegend=True
            ))

    # Add count annotations
    for lp in labels_pos:
        if lp.get("hidden"):
            continue
        bg, border, font_color = _derive_colors(lp["color"])
        fig.add_annotation(
            x=lp["x"], y=lp["y"], text=lp["text"],
            showarrow=False,
            font=dict(size=font_size, color=font_color),
            xanchor="center", yanchor="middle", align="center",
            bordercolor=border, borderwidth=1, borderpad=4,
            bgcolor=bg, name=lp["name"]
        )

    # Configure layout with circles
    fig.update_layout(
        shapes=[
            dict(
                type="circle", xref="x", yref="y",
                x0=x1 - r1, y0=y1 - r1, x1=x1 + r1, y1=y1 + r1,
                line=dict(color=outline, width=1), fillcolor=colors[0]
            ),
            dict(
                type="circle", xref="x", yref="y",
                x0=x2 - r2, y0=y2 - r2, x1=x2 + r2, y1=y2 + r2,
                line=dict(color=outline, width=1), fillcolor=colors[1]
            ),
        ],
        xaxis=dict(visible=False, range=[xmin, xmax]),
        yaxis=dict(visible=False, range=[ymin, ymax], scaleanchor="x", scaleratio=1),
        margin=dict(l=20, r=20, t=30, b=20),
        title=title if title else None,
        showlegend=show_legend,
    )

    return fig

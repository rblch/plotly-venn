# plotly-venn

A tiny helper to build 2-set Venn diagrams with Plotly.

## Installation

```bash
pip install git+https://github.com/rblch/plotly-venn.git
```

Or clone and install locally:
```bash
git clone https://github.com/rblch/plotly-venn.git
cd plotly-venn
pip install .
```

## Usage

```python
from plotly_venn import venn

# Basic usage
fig = venn(120, 200, 20, labels=("Model A", "Model B"), title="Results")
fig.show()

# With custom colors
fig = venn(
    a_count=150, 
    b_count=100, 
    ab_count=30,
    labels=("Set A", "Set B"),
    colors=("#FF6B6B", "#4ECDC4"),  # Hex colors automatically get transparency
    title="Custom Venn Diagram"
)
fig.show()

# Labels positioned below (table style)
fig = venn(120, 200, 20, labels=("Model A", "Model B"), inside=False)
fig.show()
```

## Parameters

- `a_count`: Count for set A
- `b_count`: Count for set B  
- `ab_count`: Count for intersection of A and B
- `labels`: Optional labels for the sets
- `title`: Optional title for the diagram
- `colors`: Colors for sets A and B (automatically adds transparency if needed)
- `outline`: Color for circle outlines (default: "#333")
- `max_radius`: Maximum radius for scaling (default: 1.8)
- `font_size`: Font size for count labels (default: 16)
- `avoid_collision`: Whether to avoid label collision (default: True)
- `show_zero`: Whether to show zero counts (default: False)
- `auto_legend_counts`: Whether to auto-generate legend with counts (default: True)
- `show_legend`: Whether to show legend (auto-determined if None)
- `inside`: Whether to place labels inside circles or below (default: True)

## Requirements

- Python 3.8+
- plotly

## License

MIT License

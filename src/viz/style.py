"""IJCAI-friendly matplotlib defaults and color palette.

Imported once via `apply_style()` at the top of any figure-producing script.
Goal: clean publication-grade figures, 300 DPI, sans-serif, no over-styling.
"""
from __future__ import annotations

import matplotlib.pyplot as plt


# Stable agent / quadrant colors used throughout the paper figures.
# Picked for color-blind friendliness (close to Tol/Wong palettes).
COLORS = {
    'vader':        '#888888',  # neutral gray (cheap edge)
    'finbert':      '#1f77b4',  # blue (specialist)
    'llm':          '#d62728',  # red (foundation)
    'consensus':    '#2ca02c',  # green
    'domain_shift': '#ff7f0e',  # orange
    'ambiguous':    '#9467bd',  # purple
    'mixed':        '#7f7f7f',  # gray
    'sdi_le':       '#1f77b4',
    'sdi_lr':       '#9467bd',
    'sdi_er':       '#d62728',
}

# Order used when iterating quadrants in plots / tables for consistency.
QUADRANT_ORDER = ['consensus', 'mixed', 'domain_shift', 'ambiguous']


def apply_style() -> None:
    """Set matplotlib rcParams for IJCAI-friendly figures."""
    plt.rcParams.update({
        'figure.dpi':         110,
        'savefig.dpi':        300,
        'savefig.bbox':       'tight',
        'font.family':        'sans-serif',
        'font.sans-serif':    ['DejaVu Sans', 'Helvetica', 'Arial'],
        'font.size':          10,
        'axes.titlesize':     11,
        'axes.titleweight':   'bold',
        'axes.labelsize':     10,
        'xtick.labelsize':    9,
        'ytick.labelsize':    9,
        'legend.fontsize':    9,
        'legend.frameon':     False,
        'axes.spines.top':    False,
        'axes.spines.right':  False,
        'axes.grid':          True,
        'grid.linestyle':     '--',
        'grid.alpha':         0.3,
    })

"""Shared publication style for paper figures.

Defaults vs the diagnostic style in src/viz/style.py:
  * Font sizes ~2x larger (16-22pt range; readable at column-width and
    when projected on a 4K screen)
  * Tight axes / minimal whitespace; pad_inches = 0.02
  * Wong color-vision-deficiency-safe palette + distinct markers per
    series so figures remain readable in greyscale and to color-blind
    readers
  * PDF as primary output (vector); PNG as a 300dpi rasterised twin
    for paper-internal previewing
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


# Wong palette (Bang Wong, Nature Methods 2011) — safe across the
# common types of colour vision deficiency. Plus black.
WONG = {
    'black':       '#000000',
    'orange':      '#E69F00',
    'sky_blue':    '#56B4E9',
    'green':       '#009E73',
    'yellow':      '#F0E442',
    'blue':        '#0072B2',
    'vermillion':  '#D55E00',
    'purple':      '#CC79A7',
    'grey':        '#999999',
}

# Stable assignments used across paper figures. We keep model-tier
# colours consistent (V/F/L always the same hue), and reuse Wong
# colours for protocol curves.
PAPER_COLORS = {
    'vader':        WONG['grey'],        # cheap edge agent
    'finbert':      WONG['blue'],        # specialist
    'llm':          WONG['vermillion'],  # reasoner
    'critic':       WONG['sky_blue'],    # interaction protocol — flat/blue
    'debate':       WONG['orange'],      # interaction protocol — ramping/orange
    'vote':         WONG['purple'],
    'consensus':    WONG['green'],
    'domain_shift': WONG['orange'],
    'ambiguous':    WONG['purple'],
    'mixed':        WONG['grey'],
    'gold':         WONG['black'],
    'background':   '#ffffff',
}

PAPER_MARKERS = {
    'single':  'o',
    'critic':  's',
    'debate':  'D',
    'vote':    '^',
    'finbert': 'P',
    'llm':     'X',
    'gold':    '*',
}


def apply_paper_style(font_scale: float = 1.0) -> None:
    """Set matplotlib rcParams for publication figures.

    The font_scale=1.0 default already lands at ~2x the diagnostic
    style (16-22pt vs the previous 9-11pt range).
    """
    base = 16.0 * font_scale
    mpl.rcParams.update({
        # Backend / output ---------------------------------------------------
        'figure.dpi':          120,
        'savefig.dpi':         300,
        'savefig.bbox':        'tight',
        'savefig.pad_inches':  0.02,        # tighter than mpl default 0.1
        'pdf.fonttype':        42,          # editable text in vector PDFs
        'ps.fonttype':         42,
        # Fonts (~2x diagnostic) --------------------------------------------
        'font.family':         'sans-serif',
        'font.sans-serif':     ['DejaVu Sans', 'Helvetica', 'Arial'],
        'font.size':           base,                     # 16
        'axes.titlesize':      base * 1.2,               # 19.2
        'axes.titleweight':    'bold',
        'axes.labelsize':      base * 1.05,              # 16.8
        'axes.labelweight':    'bold',
        'xtick.labelsize':     base,                     # 16
        'ytick.labelsize':     base,                     # 16
        'legend.fontsize':     base * 0.85,              # 13.6
        'legend.title_fontsize': base * 0.9,
        'figure.titlesize':    base * 1.3,
        # Lines --------------------------------------------------------------
        'lines.linewidth':     2.4,
        'lines.markersize':    9,
        'patch.linewidth':     1.3,
        # Axes / spines ------------------------------------------------------
        'axes.spines.top':     False,
        'axes.spines.right':   False,
        'axes.linewidth':      1.4,
        'axes.titlepad':       8.0,
        'axes.labelpad':       4.0,
        # Grid ---------------------------------------------------------------
        'axes.grid':           True,
        'grid.linestyle':      '--',
        'grid.alpha':          0.32,
        'grid.linewidth':      0.7,
        # Legend ------------------------------------------------------------
        'legend.frameon':      False,
        'legend.handlelength': 2.0,
        'legend.borderaxespad':0.4,
        # Color cycle (Wong palette by default for any unstyled plot) -------
        'axes.prop_cycle':     mpl.cycler(color=[
            WONG['blue'], WONG['vermillion'], WONG['green'], WONG['orange'],
            WONG['sky_blue'], WONG['purple'], WONG['yellow'], WONG['grey'],
        ]),
    })


def save_paper_figure(fig, basename: str, out_dir: Path | None = None) -> None:
    """Save figure as both PDF (vector, paper-grade) and PNG (preview).

    Files land at <out_dir>/<basename>.{pdf,png}.
    """
    if out_dir is None:
        # Default: paper/figures/, i.e. one level up from this script's dir
        out_dir = Path(__file__).resolve().parent.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f'{basename}.pdf')
    fig.savefig(out_dir / f'{basename}.png', dpi=300)
    plt.close(fig)
    print(f"  ✓ saved {basename}.pdf + .png in {out_dir}")

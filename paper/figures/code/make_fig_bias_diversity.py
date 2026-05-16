"""§5.1 Bias diversity (v3) — Times serif, Wong palette, custom
heat-map (no seaborn YlGnBu) so the colours stay on-brand.

Left:  pairwise Cohen's κ between V/F/L (lower = more diverse).
Right: per-sample disagreement entropy (bits) — bimodal: most
       sentences are either fully aligned (0 bits) or 2-vs-1
       (~0.92 bits).
"""
from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter
from math import log2

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _style import save_paper_figure, WONG

ROOT = HERE.parent.parent.parent
RESULTS = ROOT / 'results' / 'data'


def main():
    mpl.rcParams.update({
        'font.family':         'serif',
        'font.serif':          ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset':    'stix',
        'font.size':           17,
        'axes.titlesize':      21,
        'axes.titleweight':    'bold',
        'axes.labelsize':      19,
        'axes.labelweight':    'bold',
        'xtick.labelsize':     16,
        'ytick.labelsize':     16,
        'legend.fontsize':     15,
        'pdf.fonttype':        42,
        'ps.fonttype':         42,
        'figure.dpi':          120,
        'savefig.dpi':         300,
        'savefig.bbox':        'tight',
        'savefig.pad_inches':  0.04,
        'axes.spines.top':     False,
        'axes.spines.right':   False,
        'axes.linewidth':      1.3,
        'axes.grid':           True,
        'grid.linestyle':      '--',
        'grid.alpha':          0.25,
        'grid.linewidth':      0.6,
    })

    df = pd.read_csv(RESULTS / 'sdi_data_fpb_backup2.csv')
    cols  = ['vader_label', 'finbert_label', 'llm_label']
    short = ['VADER', 'FinBERT', 'Qwen-7B']

    # ── Pairwise Cohen's kappa matrix ───────────────────────────────
    K = np.zeros((3, 3))
    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            K[i, j] = 1.0 if i == j else cohen_kappa_score(df[ci], df[cj])

    # ── Per-sample disagreement entropy ─────────────────────────────
    ents = []
    for _, r in df[cols].iterrows():
        cnt = Counter(r.tolist()); n = sum(cnt.values())
        ents.append(-sum((c / n) * log2(c / n) for c in cnt.values() if c > 0))
    ents = np.array(ents)

    # Wong-aligned 2-tone colormap for the heatmap: light grey → green
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'wong_kappa', ['#F5F5F5', WONG['green']], N=256)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2),
                             gridspec_kw={'wspace': 0.45,
                                          'width_ratios': [1.0, 1.1]})

    # ── Panel 1: Cohen's kappa heatmap ──────────────────────────────
    ax = axes[0]
    ax.grid(False)
    im = ax.imshow(K, cmap=cmap, vmin=0, vmax=1, aspect='equal')
    ax.set_xticks(range(3)); ax.set_xticklabels(short)
    ax.set_yticks(range(3)); ax.set_yticklabels(short)
    ax.tick_params(top=False, bottom=True, left=True, right=False)
    # Annotate cells
    for i in range(3):
        for j in range(3):
            color = 'white' if K[i, j] > 0.55 else 'black'
            ax.text(j, i, f'{K[i, j]:.2f}', ha='center', va='center',
                    color=color, fontsize=24, fontweight='bold')
    ax.set_title('(a)  Pairwise Cohen\'s $\\kappa$\n(lower = more diverse)',
                 loc='left')
    # Colorbar slimmer + with its own gap
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.05, shrink=0.92)
    cbar.set_label('Cohen\'s $\\kappa$', fontsize=17, fontweight='bold')
    cbar.ax.tick_params(labelsize=14)
    cbar.outline.set_linewidth(0.6)

    # ── Panel 2: disagreement entropy histogram ─────────────────────
    ax = axes[1]
    bins = np.linspace(0, log2(3) + 0.05, 25)
    ax.hist(ents, bins=bins, color=WONG['green'], alpha=0.85,
            edgecolor='black', linewidth=0.7)
    ax.set_xlabel('Per-sentence disagreement entropy  (bits)')
    ax.set_ylabel('Sentence count')
    ax.set_title('(b)  Committee disagreement entropy', loc='left')

    # Annotate the two modes
    ax.axvline(0.0,    color='black', lw=0.8, ls=':')
    ax.axvline(0.918,  color='black', lw=0.8, ls=':')
    ax.text(0.04, 2500,
            'all-agree\n(0 bits)',
            fontsize=15, color='black', va='top', ha='left')
    ax.text(0.96, 2500,
            '2-vs-1 split\n($\\approx$0.92 bits)',
            fontsize=15, color='black', va='top', ha='left')

    save_paper_figure(fig, 'fig_bias_diversity')


if __name__ == '__main__':
    main()

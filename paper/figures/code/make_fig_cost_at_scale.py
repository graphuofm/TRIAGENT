"""§1 Cost-at-scale: 1K → 10M users economic case (v4).

One SOLID hero line (TriAgent, green); all four baselines dashed and
desaturated so the eye lands on the proposed method. Headline
comparison is vs.\ GPT-4 (the premium reasoner) where the saving is
the largest and most dramatic.

Per-1000-query cost (USD): FinBERT 0.0005, Qwen-7B 0.0288,
GPT-4o-mini 0.30, GPT-4 10.00. TriAgent ≈ 15% LLM + 30% FinBERT.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _style import save_paper_figure, WONG


# All four baselines — DASHED & light
BASELINES = {
    'FinBERT (self-hosted)':  (0.0005, WONG['blue'],       'P'),
    'Qwen-7B (self-hosted)':  (0.0288, WONG['purple'],     'X'),
    'GPT-4o-mini API':        (0.30,   WONG['orange'],     'D'),
    'GPT-4 API':              (10.00,  WONG['vermillion'], 'o'),
}
GPT4_PPK = 10.00
TRI_PPK  = 0.15 * 0.30 + 0.30 * 0.0005   # ≈ $0.045/1k


def main():
    # ── Times serif throughout (matches LaTeX body text) ─────────────
    mpl.rcParams.update({
        'font.family':         'serif',
        'font.serif':          ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset':    'stix',
        'font.size':           17,
        'axes.titlesize':      20,
        'axes.titleweight':    'bold',
        'axes.labelsize':      18,
        'axes.labelweight':    'bold',
        'xtick.labelsize':     16,
        'ytick.labelsize':     16,
        'legend.fontsize':     14,
        'pdf.fonttype':        42,
        'ps.fonttype':         42,
        'figure.dpi':          120,
        'savefig.dpi':         300,
        'savefig.bbox':        'tight',
        'savefig.pad_inches':  0.04,
        'axes.spines.top':     False,
        'axes.spines.right':   False,
        'axes.linewidth':      1.4,
        'axes.grid':           True,
        'grid.linestyle':      '--',
        'grid.alpha':          0.25,
        'grid.linewidth':      0.6,
        'legend.frameon':      True,
    })

    n_users = np.array([1e3, 1e4, 1e5, 1e6, 1e7])
    qpy = 10 * 365

    fig, ax = plt.subplots(figsize=(11.0, 6.6))

    # ── All baselines: dashed, light, thin ───────────────────────────
    for name, (ppk, color, marker) in BASELINES.items():
        cost = n_users * qpy * ppk / 1000
        ax.plot(n_users, cost, marker=marker, markersize=8,
                lw=1.6, ls='--', color=color, alpha=0.55,
                markerfacecolor=color, markeredgecolor=color,
                label=name, zorder=2)

    # ── TriAgent: solid, thick, green (the only emphasized line) ─────
    tri_cost = n_users * qpy * TRI_PPK / 1000
    ax.plot(n_users, tri_cost, marker='*', markersize=20,
            lw=4.0, color=WONG['green'],
            label='TriAgent  (ours)', zorder=4)

    # ── Annotation: explicit "vs.\ GPT-4" with arrow to TriAgent ─────
    gpt4_at_10m = n_users[-1] * qpy * GPT4_PPK / 1000
    tri_at_10m  = n_users[-1] * qpy * TRI_PPK / 1000
    saved = gpt4_at_10m - tri_at_10m
    ax.annotate(
        f'\\${saved/1e6:.0f}M / yr saved\nvs. GPT-4\nat 10M users',
        xy=(n_users[-1], tri_at_10m), xytext=(1.5e3, 3e7),
        fontsize=15, fontweight='bold', color='#CC0000',
        ha='left', va='center',
        arrowprops=dict(arrowstyle='->', color='black', lw=1.6,
                        connectionstyle='arc3,rad=-0.22',
                        shrinkA=2, shrinkB=8),
        zorder=5,
    )

    # ── Axes: log + scientific tick labels ───────────────────────────
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Users  (10 queries / user / day)')
    ax.set_ylabel('Annual inference cost  (USD)')
    ax.set_title('Inference cost vs. deployment scale')

    ax.set_xticks([1e3, 1e4, 1e5, 1e6, 1e7])
    ax.set_xticklabels(['$10^3$', '$10^4$', '$10^5$', '$10^6$', '$10^7$'])
    yticks = [1, 1e2, 1e4, 1e6, 1e8]
    ax.set_yticks(yticks)
    ax.set_yticklabels(['$10^0$', '$10^2$', '$10^4$', '$10^6$', '$10^8$'])
    ax.set_ylim(0.5, 5e8)
    ax.set_xlim(8e2, 1.4e7)
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.set_minor_locator(plt.NullLocator())

    # ── Legend (lower-right, the only empty quadrant) ────────────────
    leg = ax.legend(loc='lower right', fontsize=13,
                    frameon=True, framealpha=0.94, edgecolor='0.55',
                    handletextpad=0.7, borderpad=0.55)
    leg.get_frame().set_linewidth(0.7)

    save_paper_figure(fig, 'fig_cost_at_scale')


if __name__ == '__main__':
    main()

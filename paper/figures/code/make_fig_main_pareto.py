"""§5.3 Token-Economic Pareto Frontier (line chart).

Story: SDI-routed strategies (S5/S6 = ours) trace the lower-cost edge
of the F1-vs-cost frontier. Baselines as named single points; Random
& Confidence escalation as dashed grey context curves.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _style import save_paper_figure, WONG

ROOT = HERE.parent.parent.parent
RESULTS = ROOT / 'results' / 'data'


def _pareto_front(df, x, y):
    pts = df[[x, y, 'name']].sort_values(x).reset_index(drop=True)
    out, best = [], -np.inf
    for _, p in pts.iterrows():
        if p[y] > best:
            out.append(p); best = p[y]
    return pd.DataFrame(out)


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
        'legend.fontsize':     13,
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

    sweep = pd.read_csv(RESULTS / 'pareto_points_7b.csv')

    fig, ax = plt.subplots(figsize=(10.4, 6.4))

    # ── Naive escalation baselines (dashed light grey — context) ────
    for tag, ls, lbl in [
        ('S3', '--', 'Random escalation'),
        ('S4', '-.', 'Confidence escalation'),
    ]:
        sub = sweep[sweep['strategy'] == tag].sort_values('cost_per_1k')
        env = _pareto_front(sub, 'cost_per_1k', 'f1_macro')
        ax.plot(env['cost_per_1k'], env['f1_macro'], lw=1.8, marker='o',
                markersize=6, color=WONG['grey'], ls=ls, alpha=0.65,
                label=lbl, zorder=2)

    # ── S5 — sky_blue (critic colour family) ────────────────────────
    sub5 = sweep[sweep['strategy'] == 'S5'].sort_values('cost_per_1k')
    env5 = _pareto_front(sub5, 'cost_per_1k', 'f1_macro')
    ax.plot(env5['cost_per_1k'], env5['f1_macro'], lw=2.8, marker='s',
            markersize=10, color=WONG['sky_blue'],
            label='S5  SDI single-stage  (ours)', zorder=4)

    # ── S6 — green (TriAgent hero) ──────────────────────────────────
    sub6 = sweep[sweep['strategy'] == 'S6'].copy()
    env6 = _pareto_front(sub6, 'cost_per_1k', 'f1_macro')
    ax.plot(env6['cost_per_1k'], env6['f1_macro'], lw=3.6, marker='D',
            markersize=12, color=WONG['green'],
            label='S6  SDI two-stage  (ours)', zorder=5)

    # ── Single-point baselines (large markers, on top) ──────────────
    bp = sweep[sweep['strategy'].isin(['S0', 'S1', 'S2'])]
    for _, r in bp.iterrows():
        marker, color, lbl = {
            'S0': ('P', WONG['grey'],       'Always-L1  (VADER)'),
            'S1': ('o', WONG['blue'],       'Always-L2  (FinBERT)'),
            'S2': ('X', WONG['vermillion'], 'Always-L3  (Qwen-7B)'),
        }[r['strategy']]
        # Snap zero-cost VADER to a finite floor on the log axis
        x_plot = max(float(r['cost_per_1k']), 1.5e-5)
        ax.scatter(x_plot, r['f1_macro'], s=340, marker=marker,
                   color=color, edgecolor='black', lw=1.4, zorder=6,
                   label=lbl)

    ax.set_xscale('log')
    ax.set_xlabel('Inference cost  (USD per 1000 sentences)')
    ax.set_ylabel('F1-Macro on FPB')
    ax.set_title('Token-economic Pareto frontier')
    ax.set_ylim(0.45, 0.95)
    ax.xaxis.set_minor_locator(plt.NullLocator())

    leg = ax.legend(loc='lower right', frameon=True, framealpha=0.94,
                    edgecolor='0.55', handletextpad=0.6, borderpad=0.5,
                    fontsize=12)
    leg.get_frame().set_linewidth(0.6)

    save_paper_figure(fig, 'fig_main_pareto')


if __name__ == '__main__':
    main()

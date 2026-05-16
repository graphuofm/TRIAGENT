"""§5.3 Token-Economic Pareto Frontier.

X axis: cost per 1000 sentences (USD, log scale)
Y axis: F1-Macro on FPB

Curves:
  * Always-L1 / L2 / L3 — single-point baselines
  * S3 random escalation — light grey curve
  * S4 confidence escalation — medium grey
  * S5 SDI single-stage  — colour
  * S6 SDI two-stage     — colour, thicker

Plus three named "operating points" along the S6 curve as diamond markers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _style import apply_paper_style, save_paper_figure, PAPER_COLORS

ROOT = HERE.parent.parent.parent
RESULTS = ROOT / 'results' / 'data'


def _pareto_front(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    pts = df[[x, y, 'name']].sort_values(x).reset_index(drop=True)
    out = []
    best = -np.inf
    for _, p in pts.iterrows():
        if p[y] > best:
            out.append(p); best = p[y]
    return pd.DataFrame(out)


def main():
    apply_paper_style()
    sweep = pd.read_csv(RESULTS / 'pareto_points_7b.csv')

    fig, ax = plt.subplots(figsize=(9.0, 5.6))

    # Single-point baselines
    bp = sweep[sweep['strategy'].isin(['S0', 'S1', 'S2'])]
    for _, r in bp.iterrows():
        marker, color, lbl = {
            'S0': ('s', PAPER_COLORS['vader'],   'Always-L1  (VADER)'),
            'S1': ('P', PAPER_COLORS['finbert'], 'Always-L2  (FinBERT)'),
            'S2': ('X', PAPER_COLORS['llm'],     'Always-L3  (Qwen-7B)'),
        }[r['strategy']]
        ax.scatter(r['cost_per_1k'], r['f1_macro'], s=320, marker=marker,
                   color=color, edgecolor='black', lw=1.4, zorder=6, label=lbl)

    # Curves
    for tag, color, lbl in [
        ('S3', '#cccccc', 'S3 Random escalation'),
        ('S4', '#777777', 'S4 Confidence escalation'),
        ('S5', PAPER_COLORS['critic'], 'S5 SDI$_{\\mathrm{LE}}$ single-stage  (ours)'),
    ]:
        sub = sweep[sweep['strategy'] == tag].sort_values('cost_per_1k')
        env = _pareto_front(sub, 'cost_per_1k', 'f1_macro')
        ax.plot(env['cost_per_1k'], env['f1_macro'], lw=2.6, marker='o',
                markersize=8, color=color, label=lbl, alpha=0.95)

    # S6 highlighted as the headline
    sub6 = sweep[sweep['strategy'] == 'S6'].copy()
    env6 = _pareto_front(sub6, 'cost_per_1k', 'f1_macro')
    ax.plot(env6['cost_per_1k'], env6['f1_macro'], lw=3.4, marker='D',
            markersize=12, color=PAPER_COLORS['debate'],
            label='S6 SDI two-stage  (ours, full)', zorder=5)

    ax.set_xscale('log')
    ax.set_xlabel('Inference cost  (USD per 1000 sentences, log scale)')
    ax.set_ylabel('F1-Macro on FPB')
    ax.set_title('Token-Economic Pareto Frontier on FPB')
    ax.set_ylim(0.45, 0.95)
    ax.legend(loc='lower right', fontsize=12, ncol=1, handletextpad=0.6)

    save_paper_figure(fig, 'fig_main_pareto')


if __name__ == '__main__':
    main()

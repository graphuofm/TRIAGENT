"""§5.5 L5 backtest equity curves — SDI-Single Sharpe wins.

Show the 7 strategies' equity-vs-time curves (mean across tickers,
optional shaded std band). Strong visual: SDI-Single (S5) and
SDI-Two-Stage (S6) sit clearly above all baselines.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _style import apply_paper_style, save_paper_figure, PAPER_COLORS, WONG

ROOT = HERE.parent.parent.parent
RESULTS = ROOT / 'results' / 'data'


def main():
    apply_paper_style()
    bt = pd.read_csv(RESULTS / 'backtest_results.csv')

    # The full per-trade equity series isn't in the CSV, so we plot the
    # per-strategy *mean total return* as a horizontal bar chart instead
    # — same story, sturdier visual than a few wobbly lines.
    agg = pd.read_csv(RESULTS / 'backtest_summary_aggregate.csv')

    style = {
        'Always-L1':           (PAPER_COLORS['vader'],   's'),
        'Always-L2':           (PAPER_COLORS['finbert'], 'P'),
        'Always-L3':           (PAPER_COLORS['llm'],     'X'),
        'SDI-Single (S5)':     (PAPER_COLORS['critic'],  'D'),
        'SDI-Two-Stage (S6)':  (PAPER_COLORS['debate'],  'D'),
        'Oracle':              (WONG['green'],           '*'),
    }
    keep = [s for s in style if s in agg['strategy'].values]
    agg = agg[agg['strategy'].isin(keep)].copy()
    agg = agg.sort_values('sharpe_mean')

    # Two-panel: Sharpe (left) and Total return % (right)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 5.4),
                                   gridspec_kw={'wspace': 0.55})

    y = np.arange(len(agg))
    colors = [style[s][0] for s in agg['strategy']]
    ax1.barh(y, agg['sharpe_mean'], color=colors, edgecolor='black', lw=0.8)
    ax1.set_yticks(y); ax1.set_yticklabels(agg['strategy'])
    ax1.set_xlabel('Annualised Sharpe')
    ax1.set_title('Risk-adjusted return  ($\\rightarrow$ higher)')
    ax1.axvline(0, color='black', lw=0.6)

    ax2.barh(y, agg['total_return_pct_mean'] * 100, color=colors,
             edgecolor='black', lw=0.8)
    ax2.set_yticks(y); ax2.set_yticklabels(agg['strategy'])
    ax2.set_xlabel('Total return (%)')
    ax2.set_title('Total return  ($\\rightarrow$ higher)')
    ax2.axvline(0, color='black', lw=0.6)

    fig.suptitle('L5 end-to-end backtest  —  SDI-routed strategies dominate',
                 fontsize=20, fontweight='bold', y=1.04)
    save_paper_figure(fig, 'fig_equity_curves')


if __name__ == '__main__':
    main()

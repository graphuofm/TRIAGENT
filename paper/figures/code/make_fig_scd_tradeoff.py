"""§5 SCD accuracy / hit-rate trade-off.

Single dual-axis plot:
  * Left axis (blue, primary):  SCD-hybrid F1-Macro vs τ
  * Right axis (orange, secondary): cache hit rate vs τ
  * Dashed reference: always-committee F1 (no SCD)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _style import apply_paper_style, save_paper_figure, PAPER_COLORS

ROOT = HERE.parent.parent.parent
RESULTS = ROOT / 'results' / 'data'


def main():
    apply_paper_style()
    sweep = pd.read_csv(RESULTS / 'scd_threshold_sweep.csv')

    fig, ax1 = plt.subplots(figsize=(8.2, 5.4))
    ax1.plot(sweep['tau'], sweep['f1_overall'],
             marker='o', lw=3.0, color=PAPER_COLORS['critic'],
             label='SCD hybrid F1  (cache + miss$\\to$committee)')
    committee_f1 = float(sweep['f1_committee_only'].iloc[0])
    ax1.axhline(committee_f1, ls='--', lw=2.0, color='#666',
                 label=f'Always-committee F1 $=$ {committee_f1:.3f}')
    ax1.set_xlabel('Similarity threshold  $\\tau$')
    ax1.set_ylabel('F1-Macro on test set', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_ylim(0.6, 0.95)

    ax2 = ax1.twinx()
    ax2.plot(sweep['tau'], sweep['hit_rate'],
             marker='s', lw=3.0, color=PAPER_COLORS['debate'],
             label='Cache hit rate')
    ax2.set_ylabel('Cache hit rate', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_ylim(0, 1)
    ax2.grid(False)            # avoid double grid

    # combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='center right', fontsize=12)
    ax1.set_title('Shared Consensus Dictionary  —  cost / accuracy trade-off')
    save_paper_figure(fig, 'fig_scd_tradeoff')


if __name__ == '__main__':
    main()

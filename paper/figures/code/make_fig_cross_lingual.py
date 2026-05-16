"""§7 Cross-lingual committee + SCD canonicalizer.

Two panels side-by-side:
  Left  — EN/ZH committee per Qwen size: agreement rate + per-side F1
  Right — SCD as cross-lingual bridge: hit rate vs hit-F1 across τ
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _style import apply_paper_style, save_paper_figure, PAPER_COLORS, WONG

ROOT = HERE.parent.parent.parent
RESULTS = ROOT / 'results' / 'data'


def main():
    apply_paper_style(font_scale=0.92)

    cl  = pd.read_csv(RESULTS / 'cross_lingual_consistency.csv')
    scd = pd.read_csv(RESULTS / 'scd_cross_lingual.csv')

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.6),
                             gridspec_kw={'wspace': 0.32})

    # --- Left panel: naive cross-lingual ---
    ax = axes[0]
    sizes_x = [1.5, 3.0, 7.0]
    ax.plot(sizes_x, cl['cross_lingual_agreement_rate'],
            marker='o', lw=3.0, color=PAPER_COLORS['critic'],
            label='EN $\\leftrightarrow$ ZH committee agreement')
    ax.plot(sizes_x, cl['en_committee_f1'],
            marker='s', lw=2.0, ls='--', color=WONG['blue'],
            label='EN committee F1 vs gold')
    ax.plot(sizes_x, cl['zh_committee_f1'],
            marker='D', lw=2.0, ls='--', color=PAPER_COLORS['debate'],
            label='ZH committee F1 vs gold')
    ax.set_xscale('log')
    ax.set_xticks(sizes_x); ax.set_xticklabels(['1.5B', '3B', '7B'])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel('Qwen-N reasoning tier')
    ax.set_ylabel('rate / F1')
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc='lower right', fontsize=11)
    ax.set_title('Naive cross-lingual committee')

    # --- Right panel: SCD canonicalizer ---
    ax = axes[1]
    ax.plot(scd['tau'], scd['cross_lingual_hit_rate'],
            marker='s', lw=3.0, color=PAPER_COLORS['debate'],
            label='ZH$\\to$EN cache hit rate')
    ax.plot(scd['tau'], scd['hit_f1_vs_gold'],
            marker='o', lw=3.0, color=PAPER_COLORS['critic'],
            label='Cached-label F1 vs gold')
    # annotate the τ=0.85 sweet-spot
    row85 = scd[scd['tau'] == 0.85].iloc[0]
    ax.scatter([0.85], [row85['hit_f1_vs_gold']], s=320,
               color=WONG['vermillion'], marker='*', zorder=6,
               label=f'$\\tau$=0.85: hit$=${row85["cross_lingual_hit_rate"]:.0%}, F1$=${row85["hit_f1_vs_gold"]:.2f}')
    ax.set_xlabel('Similarity threshold  $\\tau$')
    ax.set_ylabel('rate / F1')
    ax.set_ylim(0.5, 1.05)
    ax.legend(loc='lower left', fontsize=11)
    ax.set_title('SCD as cross-lingual canonicalizer')

    fig.suptitle('Cross-lingual deployment: naive committee vs SCD bridge',
                 fontsize=20, fontweight='bold', y=1.02)
    save_paper_figure(fig, 'fig_cross_lingual')


if __name__ == '__main__':
    main()

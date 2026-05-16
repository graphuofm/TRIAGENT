"""§5.5 Per-class F1 — debate@7B beats FinBERT on the negative class.

Horizontal bar chart sorted by F1 on the (hard) negative class. Each
strategy has 3 bars (negative / neutral / positive). The negative bar
is highlighted; the dashed grey line marks FinBERT-alone's negative-class
F1 to make the comparison visual.
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
    apply_paper_style(font_scale=0.95)   # slightly smaller — many y-tick rows
    df = pd.read_csv(RESULTS / 'per_class_f1.csv')

    # Friendly relabel + filter (drop noisy TFNS entries that are out of scope)
    rename = {
        'VADER': 'VADER',
        'FinBERT': 'FinBERT',
        'Qwen-0.5B alone': 'Qwen-0.5B',
        'Qwen-1.5B alone': 'Qwen-1.5B',
        'Qwen-3B alone':   'Qwen-3B',
        'Qwen-7B alone':   'Qwen-7B',
        'critic@Qwen-1.5B': 'critic @ 1.5B',
        'critic@Qwen-3B':   'critic @ 3B',
        'critic@Qwen-7B':   'critic @ 7B',
        'debate@Qwen-1.5B': 'debate @ 1.5B',
        'debate@Qwen-3B':   'debate @ 3B',
        'debate@Qwen-7B':   'debate @ 7B',
        'critic@Qwen-MISTRAL-7B': 'critic @ Mistral-7B',
        'debate@Qwen-MISTRAL-7B': 'debate @ Mistral-7B',
    }
    df = df[df['strategy'].isin(rename)].copy()
    df['agent'] = df['strategy'].map(rename)
    df = df.sort_values('f1_negative')

    # Plot
    fig, ax = plt.subplots(figsize=(10.0, 0.55 * len(df) + 1.4))
    y = np.arange(len(df))
    bar_h = 0.27
    ax.barh(y - bar_h, df['f1_negative'], height=bar_h,
            color=PAPER_COLORS['llm'],   label='F1  (negative — hard class)', alpha=0.95)
    ax.barh(y,         df['f1_neutral'],  height=bar_h,
            color=PAPER_COLORS['vader'], label='F1  (neutral)', alpha=0.85)
    ax.barh(y + bar_h, df['f1_positive'], height=bar_h,
            color=PAPER_COLORS['critic'],label='F1  (positive)', alpha=0.95)

    # FinBERT-negative reference
    fb_neg = df.loc[df['agent'] == 'FinBERT', 'f1_negative'].iloc[0]
    ax.axvline(fb_neg, ls='--', lw=1.6, color=WONG['black'], alpha=0.55,
               label=f'FinBERT alone, negative F1 $=$ {fb_neg:.2f}')

    ax.set_yticks(y); ax.set_yticklabels(df['agent'])
    ax.set_xlabel('F1 per class')
    ax.set_xlim(0, 1.0)
    ax.set_title('Per-class F1 — sorted by F1 on the (hard) negative class')
    ax.legend(loc='lower right', fontsize=12)
    save_paper_figure(fig, 'fig_per_class_f1')


if __name__ == '__main__':
    main()

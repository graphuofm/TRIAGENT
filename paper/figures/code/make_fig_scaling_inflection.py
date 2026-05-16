"""§5.4 Single-agent F1 vs LLM size."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _style import apply_paper_style, save_paper_figure, PAPER_COLORS, WONG

ROOT = HERE.parent.parent.parent
RESULTS = ROOT / 'results' / 'data'


def _f1(df, col):
    return f1_score(df['label_text'], df[col], average='macro')


def main():
    apply_paper_style()
    df = pd.read_csv(RESULTS / 'sdi_data_fpb_backup2.csv')

    sizes_b = []
    f1s = []
    labels = []
    for size, suffix in [(0.5, '0p5b'), (1.5, '1p5b'),
                         (3.0, '3b'), (7.0, None), (14.0, '14b')]:
        col = 'llm_label' if suffix is None else f'llm_{suffix}_label'
        if col not in df.columns:
            continue
        sizes_b.append(size); f1s.append(_f1(df, col))
        labels.append(f'{size}B')

    finbert_f1 = _f1(df, 'finbert_label')
    vader_f1   = _f1(df, 'vader_label')

    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    ax.plot(sizes_b, f1s, marker='o', markersize=12, lw=2.6,
            color=PAPER_COLORS['llm'],
            label='Qwen-$N$ single-agent')
    ax.axhline(finbert_f1, ls='--', lw=2.4, color=PAPER_COLORS['finbert'],
               label=f'FinBERT alone  (F1$=${finbert_f1:.3f})')
    ax.axhline(vader_f1, ls=':', lw=2.0, color=PAPER_COLORS['vader'],
               label=f'VADER alone  (F1$=${vader_f1:.3f})')
    ax.set_xscale('log')
    ax.set_xticks(sizes_b); ax.set_xticklabels(labels)
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel('LLM parameter count  (B, log scale)')
    ax.set_ylabel('F1-Macro on FPB')
    ax.set_ylim(0.40, 0.95)
    ax.set_title('Single-agent F1 across Qwen sizes  (the 3B dip)')
    ax.legend(loc='lower right', fontsize=12)
    save_paper_figure(fig, 'fig_scaling_inflection')


if __name__ == '__main__':
    main()

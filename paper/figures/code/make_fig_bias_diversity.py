"""§5.2 Bias diversity: pairwise Cohen's kappa heatmap + entropy hist."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import cohen_kappa_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _style import apply_paper_style, save_paper_figure, PAPER_COLORS, WONG

ROOT = HERE.parent.parent.parent
RESULTS = ROOT / 'results' / 'data'


def main():
    apply_paper_style(font_scale=0.95)
    df = pd.read_csv(RESULTS / 'sdi_data_fpb_backup2.csv')
    cols = ['vader_label', 'finbert_label', 'llm_label']
    short = ['VADER', 'FinBERT', 'Qwen-7B']

    K = np.zeros((3, 3))
    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            K[i, j] = 1.0 if i == j else cohen_kappa_score(df[ci], df[cj])
    Kdf = pd.DataFrame(K, index=short, columns=short)

    # Per-sample disagreement entropy
    from collections import Counter
    from math import log2
    ents = []
    for _, r in df[cols].iterrows():
        cnt = Counter(r.tolist()); n = sum(cnt.values())
        ents.append(-sum((c / n) * log2(c / n) for c in cnt.values() if c > 0))
    ents = np.array(ents)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0),
                             gridspec_kw={'wspace': 0.45})
    sns.heatmap(Kdf, annot=True, fmt='.2f', vmin=0, vmax=1, cmap='YlGnBu',
                cbar_kws={'label': 'Cohen $\\kappa$', 'pad': 0.04},
                annot_kws={'size': 18, 'weight': 'bold'},
                ax=axes[0], square=True)
    axes[0].set_title('Pairwise Cohen $\\kappa$\n(low = high diversity)')

    bins = np.linspace(0, log2(3) + 0.05, 25)
    axes[1].hist(ents, bins=bins, color=PAPER_COLORS['critic'], alpha=0.85)
    axes[1].set_xlabel('Per-sample disagreement entropy (bits)')
    axes[1].set_ylabel('count')
    axes[1].set_title('Committee disagreement entropy')

    save_paper_figure(fig, 'fig_bias_diversity')


if __name__ == '__main__':
    main()

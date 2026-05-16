"""§5.6 Edge predictor ROC: 7 variants spanning word/phrase/sentence."""
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
    df = pd.read_csv(RESULTS / 'predictor_results.csv')

    # Plot AUC bars (we don't keep per-variant ROC raw curves; AUC bar is fine)
    order = ['random', 'LR-unigram', 'LR-uni+bigram', 'LR-uni+bi+trigram',
             'LR-sentence-only', 'LR-uni+bi+sentence', 'xgboost',
             'XGBoost+reasoning (NOT DEPLOYABLE)']
    df = df.set_index('model').reindex([m for m in order if m in df['model'].values
                                        or m in df.index])
    df = df.dropna()

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    y = np.arange(len(df))
    colors = [WONG['grey'], WONG['vermillion'], WONG['vermillion'],
              WONG['vermillion'], WONG['orange'], WONG['orange'],
              PAPER_COLORS['critic'], WONG['green']][:len(df)]
    bars = ax.barh(y, df['auc_roc'], color=colors, edgecolor='black', lw=1.0)
    for i, (idx, v) in enumerate(zip(df.index, df['auc_roc'])):
        ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=12, fontweight='bold')

    # Highlight deployable ceiling
    ax.axvline(0.85, ls=':', color='black', lw=1.5, alpha=0.6)
    ax.text(0.85, -0.6, 'deployable\nceiling', fontsize=11, ha='center',
            color='black', alpha=0.7)

    ax.set_yticks(y); ax.set_yticklabels(df.index)
    ax.set_xlabel('AUC-ROC for the high-SDI prediction task')
    ax.set_xlim(0.45, 1.0)
    ax.set_title('Three-granularity edge predictor: word / phrase / sentence ablation')
    save_paper_figure(fig, 'fig_predictor_auc')


if __name__ == '__main__':
    main()

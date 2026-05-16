"""§5.7 SDI as a post-hoc LLM-hallucination detector.

ROC for "predict whether the LLM is wrong, given some SDI" using
sdi_le / sdi_lr / sdi_er / sdi_max as the discriminator. SDI_ER is the
star (AUC ≈ 0.90 on FPB).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _style import apply_paper_style, save_paper_figure, PAPER_COLORS, WONG

ROOT = HERE.parent.parent.parent
RESULTS = ROOT / 'results' / 'data'


def main():
    apply_paper_style()
    sdi = pd.read_csv(RESULTS / 'sdi_data_fpb_backup2.csv')
    y = (sdi['llm_label'] != sdi['label_text']).astype(int).to_numpy()

    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    series = [
        ('sdi_le',  WONG['grey'],       'SDI$_{\\mathrm{LE}}$  (Lex–Exp)'),
        ('sdi_lr',  WONG['purple'],     'SDI$_{\\mathrm{LR}}$  (Lex–Reas)'),
        ('sdi_er',  PAPER_COLORS['llm'],'SDI$_{\\mathrm{ER}}$  (Exp–Reas, headline)'),
        ('sdi_max', PAPER_COLORS['critic'],'SDI$_{\\max}$'),
    ]
    for col, color, lbl in series:
        x = sdi[col].to_numpy()
        fpr, tpr, _ = roc_curve(y, x)
        auc = roc_auc_score(y, x)
        lw = 3.4 if col == 'sdi_er' else 2.0
        ax.plot(fpr, tpr, lw=lw, color=color,
                label=f'{lbl}  (AUC$=${auc:.3f})')
    ax.plot([0, 1], [0, 1], '--', lw=1.4, color='#999')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('False positive rate  (LLM is right but flagged)')
    ax.set_ylabel('True positive rate  (LLM is wrong, caught)')
    ax.set_title('SDI as a post-hoc LLM hallucination detector')
    ax.legend(loc='lower right', fontsize=12)
    save_paper_figure(fig, 'fig_e2_halluc_roc')


if __name__ == '__main__':
    main()
